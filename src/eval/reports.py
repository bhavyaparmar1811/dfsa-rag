"""Read-only views of the eval tables for the web UI (GET /eval/...).

Nothing here runs an eval or changes a gate: it serialises what `src.eval.run`,
`src.eval.gate` and `src.eval.cache_eval` stored, inside one session each, so the
API never touches a detached ORM object.
"""

from sqlalchemy import Engine

from src.eval.gate import load_thresholds
from src.eval.golden import Item, load_golden
from src.storage.db import session_scope
from src.storage.models import EvalResult, EvalRun
from src.storage.repositories import EvalRunRepository

# The metrics a runs list shows per row; the full set is on the run itself.
HEADLINE = ("recall@6", "mrr", "ndcg@6", "faithfulness", "relevance", "false_refusal_rate")


class RunNotFound(KeyError):
    pass


class EvalReports:
    def __init__(self, engine: Engine):
        self.engine = engine

    def runs(self, kind: str | None = None) -> list[dict]:
        with session_scope(self.engine) as s:
            return [_summary(r) for r in EvalRunRepository(s).runs(kind)]

    def run(self, run_id: int) -> dict:
        with session_scope(self.engine) as s:
            run = EvalRunRepository(s).get(run_id)
            if run is None:
                raise RunNotFound(run_id)
            return _summary(run) | {
                "metrics": run.metrics,
                "gate_report": run.gate_report,
                "retrieval_config": run.retrieval_config,
                "prompts": run.prompts,
                "cost_usd": run.cost_usd,
                "golden_hash": run.golden_hash,
                "langfuse_run": run.langfuse_run,
                "notes": run.notes,
                "error": run.error,
            }

    def results(self, run_id: int) -> list[dict]:
        """Each turn's stored result, with its golden item's kind, module and gold
        provisions (from the current golden set: an item since removed has none)."""
        items = {i.id: i for i in load_golden().items}
        with session_scope(self.engine) as s:
            repo = EvalRunRepository(s)
            if repo.get(run_id) is None:
                raise RunNotFound(run_id)
            return [_result(r) | _golden(items, r) for r in repo.results(run_id)]

    def compare(self, kind: str = "retrieval") -> dict:
        """The newest run of each config on the current golden set (reviewed items
        only), so the configs are compared on the same questions. If the golden set
        has changed since the last run, the newest set that has runs is used."""
        with session_scope(self.engine) as s:
            repo = EvalRunRepository(s)
            golden_hash = load_golden().hash
            runs = repo.latest_per_config(kind=kind, golden_hash=golden_hash)
            current = bool(runs)
            if not runs:
                finished = (r for r in repo.runs(kind) if r.status == "ok" and r.reviewed_only)
                newest = next(finished, None)
                if newest is not None:
                    golden_hash = newest.golden_hash
                    runs = repo.latest_per_config(kind=kind, golden_hash=golden_hash)
            return {
                "kind": kind,
                "golden_hash": golden_hash,
                "golden_is_current": current,
                "runs": [_summary(r) | {"metrics": r.metrics} for r in runs],
            }

    @staticmethod
    def thresholds() -> dict:
        return load_thresholds().model_dump()


def _summary(run: EvalRun) -> dict:
    metrics = run.metrics or {}
    return {
        "id": run.id,
        "kind": run.kind,
        "config_name": run.config_name,
        "status": run.status,
        "gate_status": run.gate_status,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "n_items": run.n_items,
        "n_turns": run.n_turns,
        "reviewed_only": run.reviewed_only,
        "prompt_label": run.prompt_label,
        "git_commit": run.git_commit,
        "cost_total_usd": (run.cost_usd or {}).get("total", metrics.get("cost_usd")),
        "headline": {k: metrics[k] for k in HEADLINE if metrics.get(k) is not None},
    }


def _golden(items: dict[str, Item], r: EvalResult) -> dict:
    item = items.get(r.item_id)
    turn = item.turns[r.turn] if item and r.turn < len(item.turns) else None
    return {
        "item_kind": item.kind if item else None,
        "module": item.module if item else None,
        "gold": turn.gold if turn else [],
        "supporting": turn.supporting if turn else [],
    }


def _result(r: EvalResult) -> dict:
    return {
        "id": r.id,
        "item_id": r.item_id,
        "turn": r.turn,
        "question": r.question,
        "rewritten_query": r.rewritten_query,
        "expect": r.expect,
        "ranked": r.ranked,
        "confidence": r.confidence,
        "status": r.status,
        "refusal_code": r.refusal_code,
        "answer": r.answer,
        "cited": r.cited,
        "metrics": r.metrics,
        "judges": r.judges,
        "cost_usd": r.cost_usd,
        "latency_ms": r.latency_ms,
        "trace_id": r.trace_id,
    }
