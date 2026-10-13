import json

import pytest

from src.retrieval.models import Filters
from src.retrieval.pipeline import Retriever
from tests.factories import FakeEmbedder, FakeSparse
from tests.retrieval.conftest import EMB, QD, FakeScorer, settings


@pytest.fixture
def build(loaded):
    engine, store = loaded

    def make(**cfg):
        return Retriever.build(
            settings(**cfg), engine=engine, store=store, embedder=FakeEmbedder(EMB),
            sparse=FakeSparse(QD), scorer=FakeScorer(),
        )  # fmt: skip

    return make


def test_retrieve_end_to_end(build):
    r = build(top_n=3).retrieve("Does the Client Agreement rule apply to a Market Counterparty?")
    assert len(r.hits) == 3 and r.confidence == max(h.rerank_score for h in r.hits)
    assert r.confident is None  # no calibrated threshold yet
    assert r.embed_tokens == 7
    assert set(r.timings_ms) >= {"citations", "embed", "search", "rerank", "context", "total"}
    keys = {b.key for b in r.context}
    assert "COB 3.3.1" in keys  # the scope rule that answers the question
    assert "GLO Market Counterparty" in keys  # and the definition of the question's term
    json.dumps(r.trace())  # the trace is plain JSON


def test_an_exact_citation_is_always_kept_first(build):
    r = build(top_n=2).retrieve("Explain COB 3.4.1 in the context of Market Counterparty")
    assert r.hits[0].key == "COB 3.4.1" and r.hits[0].source == "citation"
    assert r.citations.resolved == {"COB 3.4.1": ["COB 3.4.1"]}
    assert r.hits[0].rerank_score is not None  # still scored, for the trace


def test_citation_hits_can_exceed_top_n(build):
    r = build(top_n=1).retrieve("What do COB 3.3.2 and COB 3.3.3 say?")
    assert [h.key for h in r.hits] == ["COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.3"]


def test_an_unknown_citation_is_reported(build):
    r = build().retrieve("Is COB 16.6.4 still in force?")
    assert r.citations.unresolved == ["COB 16.6.4"]


def test_the_threshold_decides_confidence(build):
    assert build(refusal_threshold=0.0).retrieve("Market Counterparty").confident is True
    assert build(refusal_threshold=1.01).retrieve("Market Counterparty").confident is False


def test_without_rerank_there_is_no_confidence_signal(build):
    r = build(rerank=False, top_n=2).retrieve("Market Counterparty")
    assert r.confidence is None and r.confident is None and "rerank" not in r.timings_ms
    assert len(r.hits) == 2


def test_filters_reach_search_and_citations(build):
    r = build().retrieve("COB 3.3.2 and the Market Counterparty", Filters(modules=["GLO"]))
    assert all(h.payload["module"] == "GLO" for h in r.hits)
    assert r.citations.filtered_out == ["COB 3.3.2", "COB 3.3.2 Guidance"]


def test_citation_lookup_can_be_switched_off(build):
    r = build(citation_lookup=False).retrieve("COB 3.4.1")
    assert r.citations.found == [] and all(h.source == "search" for h in r.hits)


def test_a_mismatched_sparse_encoder_is_refused(loaded):
    engine, store = loaded
    other = FakeSparse(QD)
    other.fingerprint = "Qdrant/bm25-k1.2-b0.75-avg256"
    with pytest.raises(ValueError, match="Sparse encoder"):
        Retriever.build(settings(), engine=engine, store=store, embedder=FakeEmbedder(EMB),
                        sparse=other, scorer=FakeScorer())  # fmt: skip


def test_only_rerank_k_candidates_reach_the_reranker(build):
    r = build(rerank_k=3, candidate_k=10, top_n=6).retrieve("What does COB 3.4.1 say?")
    scored = [c for c in r.candidates if c.rerank_score is not None]
    assert len(scored) == 1 + 3  # the exact hit + the best 3 search candidates
    assert len(r.candidates) > len(scored)  # the rest stay in the trace, unscored
    assert all(h.rerank_score is not None for h in r.hits) and len(r.hits) == 4
