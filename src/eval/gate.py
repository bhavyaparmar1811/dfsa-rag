"""The release gate: an e2e eval run against eval/thresholds.yaml.

    uv run python -m src.eval.gate --run latest      # re-check a stored run, exit 1 if it fails
    uv run python -m src.eval.gate --run 12

A run passes when:
    - it is an e2e run over reviewed items only, with at least `min_reviewed_items`
    - every floor (metric >= limit) and ceiling (metric <= limit) holds; a limit
      that is still null fails as "not set"
    - no floor metric dropped more than `max_regression` (relative) below the last
      passing run of the production config on the same golden set

The runner applies the gate at the end of every e2e run and stores the report in
`eval_runs.gate_report`. Re-checking a stored run (e.g. after setting the floors
from the baseline) costs nothing: it reads the stored metrics.

`promotion_evidence()` is what `scripts/promote_prompt.py` asks: is there a
passing e2e run that used exactly this prompt content on the current golden set?
"""

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine

from src.config.settings import PathSettings
from src.storage.db import get_engine, session_scope
from src.storage.models import EvalRun
from src.storage.repositories import EvalRunRepository


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    min_reviewed_items: int
    max_false_refusal: float
    floors: dict[str, float | None]
    ceilings: dict[str, float | None]
    max_regression: float


def load_thresholds(path: Path | None = None) -> Thresholds:
    with open(path or PathSettings().eval_thresholds, encoding="utf-8") as f:
        return Thresholds.model_validate(yaml.safe_load(f))


@dataclass
class Check:
    name: str
    op: str  # ">=", "<=", "=="
    value: float | str | None
    limit: float | str | None
    passed: bool
    note: str = ""


@dataclass
class GateReport:
    run_id: int
    checks: list[Check] = field(default_factory=list)
    baseline_run: int | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "baseline_run": self.baseline_run,
            "checks": [asdict(c) for c in self.checks],
        }

    def render(self) -> str:
        lines = [f"Gate for eval run #{self.run_id}: {self.status.upper()}"]
        if self.baseline_run:
            lines.append(f"  regression baseline: run #{self.baseline_run}")
        for c in self.checks:
            value = f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value)
            limit = f"{c.limit:.4f}" if isinstance(c.limit, float) else str(c.limit)
            mark = "ok  " if c.passed else "FAIL"
            note = f"  ({c.note})" if c.note else ""
            lines.append(f"  {mark} {c.name:<28} {value:>10} {c.op} {limit:<10}{note}")
        return "\n".join(lines)


def evaluate(run: EvalRun, thresholds: Thresholds, baseline: EvalRun | None) -> GateReport:
    report = GateReport(run_id=run.id, baseline_run=baseline.id if baseline else None)
    add = report.checks.append
    m = run.metrics or {}

    add(Check("kind", "==", run.kind, "e2e", run.kind == "e2e"))
    add(Check("status", "==", run.status, "ok", run.status == "ok"))
    add(Check("reviewed_only", "==", str(run.reviewed_only), "True", bool(run.reviewed_only)))
    add(
        Check(
            "items",
            ">=",
            run.n_items,
            thresholds.min_reviewed_items,
            run.n_items >= thresholds.min_reviewed_items,
        )
    )

    for name, limit in thresholds.floors.items():
        add(_limit(name, m.get(name), ">=", limit))
    for name, limit in thresholds.ceilings.items():
        add(_limit(name, m.get(name), "<=", limit))

    if baseline is not None:
        for name in thresholds.floors:
            value, base = m.get(name), (baseline.metrics or {}).get(name)
            if value is None or base is None:
                continue
            floor = base * (1 - thresholds.max_regression)
            add(
                Check(
                    f"{name} vs #{baseline.id}",
                    ">=",
                    value,
                    round(floor, 4),
                    value >= floor,
                    note=f"baseline {base:.4f}",
                )
            )
    return report


def _limit(name: str, value, op: str, limit: float | None) -> Check:
    if limit is None:
        return Check(name, op, value, None, False, note="limit not set in thresholds.yaml")
    if value is None:
        return Check(name, op, None, limit, False, note="metric missing from the run")
    passed = value >= limit if op == ">=" else value <= limit
    return Check(name, op, value, limit, passed)


def baseline_for(repo: EvalRunRepository, run: EvalRun, thresholds: Thresholds) -> EvalRun | None:
    """The last passing e2e run of the production config on the same golden set."""
    return repo.latest(
        kind="e2e",
        config_name=thresholds.config,
        golden_hash=run.golden_hash,
        gate_status="passed",
        before=run.id,
    )


def apply(engine: Engine, run_id: int, thresholds: Thresholds | None = None) -> GateReport:
    """Evaluate a stored run and store the result on it."""
    thresholds = thresholds or load_thresholds()
    with session_scope(engine) as s:
        repo = EvalRunRepository(s)
        run = repo.get(run_id)
        if run is None:
            raise KeyError(f"no eval run #{run_id}")
        report = evaluate(run, thresholds, baseline_for(repo, run, thresholds))
        repo.set_gate(run_id, report.status, report.to_dict())
    return report


@dataclass
class Evidence:
    ok: bool
    reason: str
    run_id: int | None = None


def promotion_evidence(engine: Engine, name: str, content_hash: str, golden_hash: str) -> Evidence:
    """May this prompt content be promoted? Only if the newest e2e run that used it
    (on the current golden set) passed the gate."""
    with session_scope(engine) as s:
        runs = EvalRunRepository(s).runs_with_prompt(name, content_hash, golden_hash)
        if not runs:
            return Evidence(
                False,
                f"no e2e eval run used {name} with content {content_hash[:12]} on the current "
                "golden set. Run: uv run python -m src.eval.run --label staging",
            )
        run = runs[0]
        if run.gate_status != "passed":
            return Evidence(
                False,
                f"eval run #{run.id} with this prompt {run.gate_status} "
                f"the gate (uv run python -m src.eval.gate --run {run.id})",
                run.id,
            )
        return Evidence(True, f"eval run #{run.id} passed the gate", run.id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check an eval run against eval/thresholds.yaml.")
    ap.add_argument("--run", default="latest", help="run id, or 'latest' (latest e2e run)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    engine = get_engine()
    if args.run == "latest":
        with session_scope(engine) as s:
            run = EvalRunRepository(s).latest(kind="e2e")
        if run is None:
            print("No finished e2e eval run yet.")
            return 1
        run_id = run.id
    else:
        run_id = int(args.run)
    report = apply(engine, run_id)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
