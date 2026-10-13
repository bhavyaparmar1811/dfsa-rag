import logging

import pytest

from src.retrieval.hybrid import HybridSearcher
from src.retrieval.models import Candidate, Filters
from src.retrieval.rerank import Reranker
from tests.factories import FakeEmbedder, FakeSparse
from tests.retrieval.conftest import EMB, QD, FakeScorer, settings


@pytest.fixture
def searcher(loaded):
    _, store = loaded

    def make(**kw):
        return HybridSearcher(store, FakeEmbedder(EMB), FakeSparse(QD), settings(**kw))

    return make


def test_hybrid_fuses_dense_and_bm25_ranks_with_rrf(searcher):
    result = searcher(mode="hybrid", candidate_k=10, rrf_k=60).search("Market Counterparty")
    assert result.embed_tokens == 7 and set(result.timings_ms) >= {"embed", "search"}
    for c in result.candidates:
        expected = sum(1 / (60 + r) for r in (c.dense_rank, c.sparse_rank) if r is not None)
        assert c.fused_score == pytest.approx(expected)
    scores = [c.fused_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True) and len(result.candidates) <= 10
    assert len({c.id for c in result.candidates}) == len(result.candidates)  # de-duplicated
    # Found by both retrievers -> ranks from both.
    assert any(c.dense_rank and c.sparse_rank for c in result.candidates)


def test_sparse_mode_ranks_by_bm25_and_embeds_nothing(searcher):
    result = searcher(mode="sparse").search("Market Counterparty")
    assert result.embed_tokens == 0 and "embed" not in result.timings_ms
    top = result.candidates[0]
    assert top.key in ("GLO Market Counterparty", "COB 3.3.1")
    assert top.dense_rank is None and top.fused_score == top.sparse_score


def test_dense_mode_keeps_cosine_scores(searcher):
    result = searcher(mode="dense", candidate_k=5).search("anything")
    assert len(result.candidates) == 5
    assert all(c.sparse_rank is None and c.fused_score == c.dense_score for c in result.candidates)


def test_filters_run_inside_qdrant(searcher):
    s = searcher(mode="hybrid", candidate_k=50)
    glo = s.search("Client", Filters(modules=["GLO"])).candidates
    assert glo and {c.payload["module"] for c in glo} == {"GLO"}
    binding = s.search("Client Agreement", Filters(binding_only=True)).candidates
    assert binding and all(c.payload["binding"] for c in binding)
    rules = s.search("Client Agreement", Filters(chunk_types=["guidance"])).candidates
    assert [c.key for c in rules] == ["COB 3.3.2 Guidance"]


def test_a_stopword_only_question_still_searches_dense(searcher):
    result = searcher(mode="hybrid").search("the of and")
    assert result.candidates and all(c.sparse_rank is None for c in result.candidates)


def _candidate(key, text):
    payload = {"key": key, "citation": key, "header_text": f"Header {key}", "text": text}
    return Candidate(key, payload, source="search")


def test_rerank_orders_by_score_and_flags_truncation(caplog):
    cands = [
        _candidate("A", "nothing relevant here"),
        _candidate("B", "client agreement with market counterparty " + "word " * 600),
        _candidate("C", "client agreement"),
    ]
    reranker = Reranker(FakeScorer(), max_length=512)
    with caplog.at_level(logging.WARNING):
        ranked = reranker.rerank("client agreement market counterparty", cands)
    assert [c.key for c in ranked] == ["B", "C", "A"]
    assert ranked[0].rerank_score == 1.0 and ranked[-1].rerank_score == 0.0
    assert [c.truncated for c in ranked] == [True, False, False]
    assert "B is" in caplog.text and "over 512" in caplog.text


def test_rerank_scores_header_and_text_together():
    seen = []

    class Recorder(FakeScorer):
        def score(self, question, texts):
            seen.extend(texts)
            return [0.5] * len(texts)

    Reranker(Recorder(), 512).rerank("q", [_candidate("A", "body")])
    assert seen == ["Header A\n\nbody"]


def test_rerank_of_nothing_is_nothing():
    assert Reranker(FakeScorer(), 512).rerank("q", []) == []
