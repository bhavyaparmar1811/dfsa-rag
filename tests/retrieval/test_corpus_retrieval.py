"""Retrieval over the real ingested corpus (live Qdrant + data/app.db).

Fast and free: BM25 only and no reranker, so no OpenAI call and no model load.
The reranker's quality is measured in Phase H, on the golden set.
"""

import pytest

from src.config.settings import EmbeddingSettings, QdrantSettings, RetrievalSettings
from src.retrieval.models import Filters
from src.retrieval.pipeline import Retriever
from src.storage.db import get_engine
from src.storage.qdrant_store import QdrantStore

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def retriever():
    store = QdrantStore.from_settings(QdrantSettings(), EmbeddingSettings())
    try:
        if not store.exists():
            pytest.skip("collection not built: run ingest")
    except Exception as e:
        pytest.skip(f"Qdrant not reachable: {e}")
    cfg = RetrievalSettings(mode="sparse", rerank=False)
    return Retriever.build(cfg, engine=get_engine(), store=store)


def test_an_exact_citation_comes_first_with_its_section_and_scope_rule(retriever):
    r = retriever.retrieve("What does COB 3.3.2 require?")
    assert [h.key for h in r.hits[:2]] == ["COB 3.3.2", "COB 3.3.2 Guidance"]
    keys = [b.key for b in r.context]
    assert keys[:5] == ["COB 3.3.1", "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.3", "COB 3.3.4"]
    assert next(b for b in r.context if b.key == "COB 3.3.1").reason == "scope"


def test_a_law_article_and_an_unknown_rule(retriever):
    r = retriever.retrieve("Compare Article 76 of the Regulatory Law with COB 16.6.4")
    assert r.citations.resolved == {"Regulatory Law Art. 76": ["Regulatory Law Art. 76"]}
    assert r.citations.unresolved == ["COB 16.6.4"]  # the typo in the source PDF


def test_bm25_finds_the_market_counterparty_exemption(retriever):
    r = retriever.retrieve("Do we need a Client Agreement with a Market Counterparty?")
    keys = [c.key for c in r.candidates]
    assert "COB 3.3.1" in keys[:10]
    defs = [b.key for b in r.context if b.group == "definitions"]
    assert {"GLO Market Counterparty", "GLO Client Agreement"} & (
        {h.key for h in r.hits} | set(defs)
    )


def test_module_filter(retriever):
    r = retriever.retrieve("customer due diligence third party reliance", Filters(modules=["AML"]))
    assert r.hits and {h.payload["module"] for h in r.hits} == {"AML"}
    assert r.hits[0].key.startswith("AML 8.1.1")


def test_a_split_rule_is_shown_whole(retriever):
    r = retriever.retrieve("GEN 2.3.2")
    rule = next(b for b in r.context if b.key == "GEN 2.3.2")
    assert rule.reason == "part" and "(2) A Person who is a Body Corporate" in rule.text
