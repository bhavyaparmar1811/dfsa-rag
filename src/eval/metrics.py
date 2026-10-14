"""Eval metrics: pure functions over provision ids, no I/O.

Retrieval is scored on **provision ids** (`golden.provision`: chunk key without
"#pN"), so the numbers survive re-ingestion and a split rule counts once.

The ranked list a turn is scored on is what the answer step actually sees
first: the hits (exact citation hits pinned first, then the reranked top N),
followed by the remaining candidates in rank order, de-duplicated.

    recall@k        share of gold provisions in the top k (k=40 = candidate recall)
    mrr             1 / rank of the first gold provision (0 if none)
    ndcg@k          graded: gold 2, supporting 1
    context_recall  share of gold provisions among the context blocks sent to the LLM
                    (parent expansion can bring in a rule that wasn't a hit)

Answer-level metrics compare the outcome with the turn's `expect`, and the
cited provisions with gold + supporting.
"""

import math
from collections.abc import Iterable, Sequence

from src.eval.golden import Turn, provision

K_VALUES = (1, 3, 6, 20, 40)
NDCG_K = 6
VALIDATOR_FAILURES = {"invalid_output", "citation_check_failed"}

# expect x refused -> outcome
OUTCOMES = {
    ("answer", False): "correct_answer",
    ("answer", True): "false_refusal",
    ("refuse", True): "correct_refusal",
    ("refuse", False): "wrong_answer",
}


def dedupe(keys: Iterable[str]) -> list[str]:
    """Provision ids in first-seen order."""
    seen: dict[str, None] = {}
    for key in keys:
        seen.setdefault(provision(key), None)
    return list(seen)


def ranked_provisions(result) -> list[str]:
    """Hits first (as sent to the LLM), then the other candidates, by provision."""
    return dedupe([c.key for c in result.hits] + [c.key for c in result.candidates])


def context_provisions(result) -> set[str]:
    return {provision(b.key) for b in result.context}


def recall_at(ranked: Sequence[str], gold: Iterable[str], k: int) -> float | None:
    gold = set(gold)
    if not gold:
        return None
    return len(gold & set(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float | None:
    gold = set(gold)
    if not gold:
        return None
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[str], relevance: dict[str, int], k: int) -> float | None:
    """nDCG with gain 2^rel - 1 and a log2 discount; ideal = relevance sorted."""
    if not relevance:
        return None

    def dcg(gains: Iterable[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return dcg(relevance.get(p, 0) for p in ranked[:k]) / ideal


def retrieval_scores(ranked: Sequence[str], context: set[str], turn: Turn) -> dict:
    """The retrieval metrics of one answerable turn (None for a refusal turn)."""
    if turn.expect == "refuse":
        return {}
    scores = {f"recall@{k}": recall_at(ranked, turn.gold, k) for k in K_VALUES}
    scores["mrr"] = reciprocal_rank(ranked, turn.gold)
    scores[f"ndcg@{NDCG_K}"] = ndcg_at(ranked, turn.relevance, NDCG_K)
    scores["context_recall"] = len(set(turn.gold) & context) / len(turn.gold)
    return scores


def refusal_outcome(expect: str, refused: bool) -> str:
    return OUTCOMES[(expect, refused)]


def citation_scores(cited: Iterable[str], turn: Turn) -> dict:
    """Precision: share of cited provisions that are gold or supporting.
    Recall: share of gold provisions cited. Only for answered, answerable turns."""
    cited = set(dedupe(cited))
    if turn.expect == "refuse" or not cited:
        return {}
    relevant = set(turn.gold) | set(turn.supporting)
    return {
        "citation_precision": len(cited & relevant) / len(cited),
        "citation_recall": len(cited & set(turn.gold)) / len(turn.gold),
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


MEAN_KEYS = (
    *(f"recall@{k}" for k in K_VALUES),
    "mrr",
    f"ndcg@{NDCG_K}",
    "context_recall",
    "citation_precision",
    "citation_recall",
    "faithfulness",
    "relevance",
)


def aggregate(rows: Sequence[dict]) -> dict:
    """Run-level numbers from per-turn metric dicts (as stored in eval_results).

    Each row may hold any of MEAN_KEYS, plus `outcome`, `refusal_code`,
    `contradicted`, `judge_error`, `error`, `cost_usd` and `latency_ms` (stage -> ms).
    A metric missing from a row (e.g. relevance on a refusal) is skipped, not 0.
    """
    out: dict = {"turns": len(rows)}
    for key in MEAN_KEYS:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            out[key] = round(_mean(values), 4)
            out[f"n_{key}"] = len(values)

    outcomes = [r["outcome"] for r in rows if r.get("outcome")]
    if outcomes:
        n = {o: outcomes.count(o) for o in set(OUTCOMES.values())}
        answerable = n["correct_answer"] + n["false_refusal"]
        unanswerable = n["correct_refusal"] + n["wrong_answer"]
        refused = n["correct_refusal"] + n["false_refusal"]
        out["outcomes"] = n
        out["false_refusal_rate"] = _ratio(n["false_refusal"], answerable)
        out["refusal_recall"] = _ratio(n["correct_refusal"], unanswerable)
        out["refusal_precision"] = _ratio(n["correct_refusal"], refused)
        out["validator_failure_rate"] = _ratio(
            sum(r.get("refusal_code") in VALIDATOR_FAILURES for r in rows if r.get("outcome")),
            len(outcomes),
        )

    judged = [r for r in rows if r.get("contradicted") is not None]
    if judged:
        out["contradicted_claims"] = sum(r["contradicted"] for r in judged)
        out["answers_with_contradiction"] = sum(r["contradicted"] > 0 for r in judged)
    out["judge_errors"] = sum(bool(r.get("judge_error")) for r in rows)
    out["errors"] = sum(bool(r.get("error")) for r in rows)  # turns that raised
    out["truncated"] = sum(r.get("truncated") or 0 for r in rows)  # rerank inputs cut at 512

    costs = [r.get("cost_usd") or 0.0 for r in rows]
    out["cost_usd"] = round(sum(costs), 6)
    out["cost_per_turn_usd"] = round(sum(costs) / len(rows), 6) if rows else None

    stages = sorted({s for r in rows for s in (r.get("latency_ms") or {})})
    out["latency_ms"] = {
        s: {
            "p50": round(percentile(v, 50), 1),
            "p95": round(percentile(v, 95), 1),
        }
        for s in stages
        if (v := [r["latency_ms"][s] for r in rows if s in (r.get("latency_ms") or {})])
    }
    return out
