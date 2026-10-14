import math
from types import SimpleNamespace

import pytest

from src.eval.golden import Turn
from src.eval.metrics import (
    aggregate,
    citation_scores,
    ndcg_at,
    percentile,
    ranked_provisions,
    recall_at,
    reciprocal_rank,
    refusal_outcome,
    retrieval_scores,
)


def turn(gold=("A",), supporting=(), expect="answer") -> Turn:
    return Turn(question="A question?", expect=expect, gold=list(gold), supporting=list(supporting))


def test_ranked_provisions_hits_first_and_parts_merged():
    c = lambda key: SimpleNamespace(key=key)  # noqa: E731
    result = SimpleNamespace(
        hits=[c("COB 3.3.1"), c("GEN 2.3.2#p2")],
        candidates=[c("GEN 2.3.2#p1"), c("COB 3.3.1"), c("COB 3.3.2"), c("GEN 2.3.2#p2")],
    )
    assert ranked_provisions(result) == ["COB 3.3.1", "GEN 2.3.2", "COB 3.3.2"]


def test_recall_and_mrr():
    ranked = ["X", "A", "Y", "B"]
    assert recall_at(ranked, {"A", "B"}, 1) == 0
    assert recall_at(ranked, {"A", "B"}, 2) == 0.5
    assert recall_at(ranked, {"A", "B"}, 40) == 1
    assert reciprocal_rank(ranked, {"A", "B"}) == 0.5
    assert reciprocal_rank(ranked, {"Z"}) == 0
    assert recall_at(ranked, set(), 3) is None and reciprocal_rank(ranked, set()) is None


def test_ndcg_graded():
    rel = {"A": 2, "B": 1}
    assert ndcg_at(["A", "B"], rel, 6) == pytest.approx(1.0)
    # B first, A second: (1/log2 2 + 3/log2 3) / (3/log2 2 + 1/log2 3)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at(["B", "A"], rel, 6) == pytest.approx(expected)
    assert ndcg_at(["X", "Y"], rel, 6) == 0


def test_retrieval_scores_with_context():
    t = turn(gold=["A", "B"], supporting=["C"])
    scores = retrieval_scores(["A", "C", "X"], {"A", "B"}, t)
    assert scores["recall@1"] == 0.5 and scores["recall@40"] == 0.5
    assert scores["mrr"] == 1.0
    assert scores["context_recall"] == 1.0
    assert retrieval_scores(["A"], set(), turn(gold=[], expect="refuse")) == {}


def test_refusal_outcomes():
    assert refusal_outcome("answer", False) == "correct_answer"
    assert refusal_outcome("answer", True) == "false_refusal"
    assert refusal_outcome("refuse", True) == "correct_refusal"
    assert refusal_outcome("refuse", False) == "wrong_answer"


def test_citation_scores():
    t = turn(gold=["A", "B"], supporting=["C"])
    assert citation_scores(["A", "C", "X#p1"], t) == {
        "citation_precision": pytest.approx(2 / 3),
        "citation_recall": 0.5,
    }
    assert citation_scores([], t) == {}


def test_percentile_matches_numpy_linear():
    xs = [10, 20, 30, 40]
    assert percentile(xs, 50) == 25
    assert percentile(xs, 95) == pytest.approx(38.5)
    assert percentile([], 50) is None


def test_aggregate():
    rows = [
        {"recall@6": 1.0, "mrr": 1.0, "outcome": "correct_answer", "faithfulness": 1.0,
         "contradicted": 0, "relevance": 5, "cost_usd": 0.02, "latency_ms": {"total": 100}},
        {"recall@6": 0.0, "mrr": 0.0, "outcome": "false_refusal",
         "refusal_code": "citation_check_failed", "cost_usd": 0.01,
         "latency_ms": {"total": 300}},
        {"outcome": "correct_refusal", "refusal_code": "low_confidence", "cost_usd": 0.0,
         "latency_ms": {"total": 20}},
        {"outcome": "wrong_answer", "faithfulness": 0.5, "contradicted": 1, "relevance": 3,
         "judge_error": None, "cost_usd": 0.03, "latency_ms": {"total": 200}},
    ]  # fmt: skip
    agg = aggregate(rows)
    assert agg["recall@6"] == 0.5 and agg["n_recall@6"] == 2
    assert agg["faithfulness"] == 0.75 and agg["relevance"] == 4
    assert agg["false_refusal_rate"] == 0.5
    assert agg["refusal_recall"] == 0.5
    assert agg["refusal_precision"] == 0.5
    assert agg["validator_failure_rate"] == 0.25
    assert agg["contradicted_claims"] == 1 and agg["answers_with_contradiction"] == 1
    assert agg["cost_usd"] == 0.06 and agg["cost_per_turn_usd"] == 0.015
    assert agg["latency_ms"]["total"]["p50"] == 150
