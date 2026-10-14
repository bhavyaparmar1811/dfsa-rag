"""Calibrate `RETRIEVAL_REFUSAL_THRESHOLD` on the golden set.

    uv run python -m src.eval.calibrate                 # latest retrieval run of hybrid_rerank
    uv run python -m src.eval.calibrate --run 7

The confidence of a turn is its best rerank score (stored per eval result), and
the retrieval gate refuses when confidence < threshold, before any LLM call. So
the sweep is offline and free.

Rule (decided in Phase H): the highest threshold that refuses at most
`max_false_refusal` (eval/thresholds.yaml) of the answerable turns. Within the
gap below it, where every threshold gives the same outcomes on the golden set,
the midpoint is chosen, for the largest margin on both sides. Unanswerable turns
the threshold doesn't catch are left to the model's own `confident` flag and the
citation check.

With ~50 questions this is a coarse estimate, calibrated on the same set it is
evaluated on: a known limitation, re-run whenever the golden set grows.
"""

import argparse
import sys
from dataclasses import dataclass, field

from src.eval.configs import PRODUCTION_CONFIG
from src.eval.gate import load_thresholds
from src.storage.db import get_engine, session_scope
from src.storage.repositories import EvalRunRepository

MIN_UNANSWERABLE = 8


@dataclass
class Point:
    item: str
    expect: str  # answer | refuse
    confidence: float


@dataclass
class Calibration:
    threshold: float | None
    false_refusal_rate: float
    catch_rate: float | None  # share of unanswerable turns refused by the threshold
    refused_answerable: list[Point] = field(default_factory=list)
    caught: list[Point] = field(default_factory=list)
    missed: list[Point] = field(default_factory=list)  # unanswerable, passes the threshold
    margin_below: float | None = None  # threshold - the highest confidence it refuses
    margin_above: float | None = None  # the lowest confidence it lets through - threshold
    warnings: list[str] = field(default_factory=list)


def calibrate(points: list[Point], max_false_refusal: float) -> Calibration:
    answerable = sorted(p.confidence for p in points if p.expect == "answer")
    refuse = [p for p in points if p.expect == "refuse"]
    if not answerable:
        raise ValueError("no answerable turns with a confidence: was the run reranked?")
    warnings = []
    if len(refuse) < MIN_UNANSWERABLE:
        warnings.append(f"only {len(refuse)} unanswerable turns (want >= {MIN_UNANSWERABLE})")

    # Refusing the k lowest answerable turns is allowed; the (k+1)-th must pass, so
    # the highest allowed threshold is that turn's confidence (conf >= t answers).
    k = int(max_false_refusal * len(answerable) + 1e-9)
    upper = answerable[k] if k < len(answerable) else None
    if upper is None:  # every answerable turn may be refused: no meaningful cap
        raise ValueError("max_false_refusal allows refusing every answerable turn")
    below = [p.confidence for p in points if p.confidence < upper]
    threshold = (max(below) + upper) / 2 if below else upper

    refused_answerable = [p for p in points if p.expect == "answer" and p.confidence < threshold]
    caught = [p for p in refuse if p.confidence < threshold]
    return Calibration(
        threshold=round(threshold, 6),
        false_refusal_rate=len(refused_answerable) / len(answerable),
        catch_rate=len(caught) / len(refuse) if refuse else None,
        refused_answerable=sorted(refused_answerable, key=lambda p: p.confidence),
        caught=sorted(caught, key=lambda p: p.confidence),
        missed=sorted((p for p in refuse if p.confidence >= threshold), key=lambda p: p.confidence),
        margin_below=round(threshold - max(below), 6) if below else None,
        margin_above=round(
            min(p.confidence for p in points if p.confidence >= threshold) - threshold, 6
        ),
        warnings=warnings,
    )


def points_of(results) -> list[Point]:
    return [
        Point(f"{r.item_id}[{r.turn}]", r.expect, r.confidence)
        for r in results
        if r.confidence is not None and r.status != "error"
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the refusal threshold.")
    ap.add_argument("--run", type=int, help=f"default: latest retrieval run of {PRODUCTION_CONFIG}")
    ap.add_argument("--max-false-refusal", type=float, help="default: eval/thresholds.yaml")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    max_frr = args.max_false_refusal
    if max_frr is None:
        max_frr = load_thresholds().max_false_refusal

    with session_scope(get_engine()) as s:
        repo = EvalRunRepository(s)
        run = (
            repo.get(args.run)
            if args.run
            else repo.latest(kind="retrieval", config_name=PRODUCTION_CONFIG)
        )
        if run is None:
            print(
                f"No retrieval eval run of {PRODUCTION_CONFIG} yet: run "
                "`uv run python -m src.eval.run --retrieval-only` first."
            )
            return 1
        points = points_of(repo.results(run.id))
        run_id, reviewed, config = run.id, run.reviewed_only, run.retrieval_config

    cal = calibrate(points, max_frr)
    print(
        f"Eval run #{run_id} ({config.get('reranker_model')}), {len(points)} turns, "
        f"max false refusals {max_frr:.0%}"
    )
    if not reviewed:
        cal.warnings.append("the run includes unreviewed items: preview only, don't commit it")
    print()
    print(f"  threshold            {cal.threshold:.4f}")
    print(f"  false refusals       {cal.false_refusal_rate:.1%} of answerable turns")
    if cal.catch_rate is not None:
        print(f"  refused before LLM   {cal.catch_rate:.1%} of unanswerable turns")
    print(f"  margin               {cal.margin_below} below, {cal.margin_above} above")
    for title, group in (
        ("Answerable turns refused", cal.refused_answerable),
        ("Unanswerable turns caught", cal.caught),
        ("Unanswerable turns let through (left to the model)", cal.missed),
    ):
        if group:
            print(f"\n{title}:")
            for p in group:
                print(f"  {p.confidence:.4f}  {p.item}")
    for w in cal.warnings:
        print(f"\nWARNING: {w}")
    print(
        f"\nSet it in src/config/settings.py (RetrievalSettings.refusal_threshold = "
        f"{cal.threshold:.4f}, eval run #{run_id}) or RETRIEVAL_REFUSAL_THRESHOLD in .env."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
