"""The cache calibration: scoring pairs, the threshold rule, and the cache gate."""

import dataclasses
import math

import pytest

from src.chat.cache import normalise, question_features
from src.eval import cache_eval, gate
from src.eval.cache_eval import Pair, PairSet, evaluate_pairs, outcome, recommend, score
from src.storage.models import EvalRun
from tests.factories import FakeEmbedder

TERMS = ("Market Counterparty", "Retail Client")


def unit(sim: float) -> list[float]:
    return [sim, math.sqrt(1 - sim**2)] + [0.0] * 6


BASE = [1.0] + [0.0] * 7
SIMILARITY = {
    "paraphrase": 0.93,
    "same words, another client type: Retail Client": 0.97,  # stopped by the term guard
    "same words, another adjective": 0.95,  # nothing for a guard to see
}


class PairEmbedder(FakeEmbedder):
    def embed_documents(self, texts):
        result = super().embed_documents(texts)
        vectors = [BASE if normalise(t) == "cached" else unit(SIMILARITY[t]) for t in texts]
        return dataclasses.replace(result, vectors=vectors)


def features(q):
    return question_features(q, lambda _: [], lambda q: [t for t in TERMS if t in q])


PAIRS = [
    Pair(id="p1", kind="positive", cached="cached", asked="paraphrase"),
    Pair(id="p2", kind="positive", cached="cached", asked="  CACHED? "),  # an exact repeat
    Pair(id="n1", kind="negative", cached="cached",
         asked="same words, another client type: Retail Client", why="term"),
    Pair(id="n2", kind="negative", cached="cached", asked="same words, another adjective",
         why="adjective"),
]  # fmt: skip


@pytest.fixture
def scored():
    return score(PAIRS, PairEmbedder().embed_documents, features)[0]


def test_scoring_and_guards(scored):
    by_id = {s.pair.id: s for s in scored}
    assert by_id["p1"].similarity == pytest.approx(0.93)
    assert by_id["n1"].guards == ["terms"] and by_id["n2"].guards == []


def test_naive_vs_guarded_outcomes(scored):
    assert outcome(scored, 0.90, guarded=False)["false_hit_ids"] == ["n1", "n2"]
    assert outcome(scored, 0.90)["false_hit_ids"] == ["n2"]  # the guard stops n1 only
    assert outcome(scored, 0.96)["false_hits"] == 0
    assert outcome(scored, None)["positive_hit_rate"] == 0.5  # exact repeats always hit


def test_the_threshold_clears_the_worst_unguarded_negative_by_the_margin(scored):
    assert recommend(scored, margin=0.02, floor=0.90) == 0.97  # 0.95 + 0.02
    assert recommend(scored, margin=0.0, floor=0.99) == 0.99
    metrics = evaluate_pairs(scored, 0.97, margin=0.02, floor=0.90)
    assert (metrics["false_hits"], metrics["positive_hit_rate"]) == (0, 0.5)
    assert metrics["max_guard_passing_negative"]["id"] == "n2"
    assert metrics["negatives_stopped_by_guards"] == 1


def test_a_negative_must_say_why():
    with pytest.raises(ValueError, match="why"):
        Pair(id="n", kind="negative", cached="a", asked="b")
    with pytest.raises(ValueError, match="duplicate"):
        PairSet(reviewed=True, pairs=[PAIRS[0], PAIRS[0]])


def test_the_committed_pairs_load():
    pairs, digest = cache_eval.load_pairs()
    assert len(pairs.pairs) >= 50 and len(digest) == 64
    assert {p.kind for p in pairs.pairs} == {"positive", "negative"}


def cache_run(**metrics) -> EvalRun:
    return EvalRun(id=7, kind="cache", status="ok", reviewed_only=True, n_items=64,
                   metrics={"threshold": None, "false_hits": 0, "positive_hit_rate": 0.0}
                   | metrics)  # fmt: skip


def test_the_cache_gate():
    limits = gate.load_thresholds()
    assert gate.evaluate(cache_run(), limits, None).passed
    failed = gate.evaluate(cache_run(false_hits=1), limits, None)
    assert not failed.passed and [c.name for c in failed.checks if not c.passed] == ["false_hits"]
    other = gate.evaluate(cache_run(threshold=0.9), limits, None)  # not the configured value
    assert [c.name for c in other.checks if not c.passed] == ["threshold"]
