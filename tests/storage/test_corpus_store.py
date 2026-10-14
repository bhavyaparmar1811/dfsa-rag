"""The ingested corpus v1, as it really is in data/app.db and the live Qdrant.

Read-only: nothing here writes to SQLite or Qdrant. Skipped unless the Qdrant
container is up, the collection exists and SQLite has been loaded:

    docker compose -f infra/docker-compose.yml --env-file .env up -d
    uv run python -m src.ingestion.ingest
    uv run pytest -m corpus tests/storage/test_corpus_store.py

The numbers are the corpus v1 figures in docs/ROADMAP.MD; a new manifest
version changes them on purpose.
"""

import pytest
from qdrant_client import models
from sqlalchemy import func, select

from src.config.settings import (
    CacheSettings,
    ChunkingSettings,
    EmbeddingSettings,
    PathSettings,
    QdrantSettings,
)
from src.embeddings.sparse import SparseEncoder
from src.ingestion.chunk_corpus import chunk_document, rulebook_codes, supported
from src.ingestion.ingest import Index, ingest
from src.ingestion.manifest import load_manifest
from src.ingestion.reconcile import reconcile
from src.storage.cache_index import CacheIndex
from src.storage.db import get_engine, session_scope
from src.storage.models import ChunkRow, Document
from src.storage.qdrant_store import BOOL_FIELDS, KEYWORD_FIELDS, SPARSE, QdrantStore
from src.storage.repositories import ChunkRepository

pytestmark = pytest.mark.corpus

ROWS, EMBEDDABLE = 4060, 3397
POINTS_PER_DOC = {
    "dfsa-rulebook-glo": 663, "dfsa-rulebook-gen": 644, "dfsa-rulebook-cob": 731,
    "dfsa-rulebook-aml": 186, "dfsa-rulebook-mkt": 461, "dfsa-rulebook-fpr": 97,
    "dfsa-laws-regulatory_law": 292, "dfsa-laws-markets_law": 109,
    "dfsa-laws-collective_investment_law": 124, "dfsa-laws-investment_trust_law": 58,
    "dfsa-laws-law_regulating_islamic_financial_business": 32,
}  # fmt: skip


@pytest.fixture(scope="module")
def emb():
    return EmbeddingSettings()


@pytest.fixture(scope="module")
def store(emb):
    store = QdrantStore.from_settings(QdrantSettings(), emb)
    try:
        exists = store.exists()
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {QdrantSettings().url}: {e}")
    if not exists:
        pytest.skip(f"Collection {store.collection} not built yet: run ingest")
    return store


@pytest.fixture(scope="module")
def engine():
    engine = get_engine()
    try:
        with session_scope(engine) as s:
            rows = s.scalar(select(func.count()).select_from(ChunkRow))
    except Exception as e:
        pytest.skip(f"data/app.db not migrated or unreadable: {e}")
    if not rows:
        pytest.skip("data/app.db has no chunks: run ingest")
    return engine


def test_collection_matches_the_configuration(store):
    store.check()  # raises on any embedding / BM25 / schema mismatch
    meta = store.client.get_collection(store.collection).config.metadata
    assert meta["fingerprint"] == "text-embedding-3-small-1536"
    assert meta["sparse_fingerprint"] == QdrantSettings().sparse_fingerprint


def test_all_payload_indexes_exist_on_the_server(store):
    wanted = {f: "keyword" for f in KEYWORD_FIELDS} | {f: "bool" for f in BOOL_FIELDS}
    assert store.payload_indexes() == wanted


def test_counts(store, engine):
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(ChunkRow)) == ROWS
        embedded = dict(s.execute(select(Document.id, Document.embedded_count)).all())
        chunk_counts = s.scalar(select(func.sum(Document.chunk_count)))
    assert chunk_counts == ROWS
    assert embedded == POINTS_PER_DOC
    assert store.count() == EMBEDDABLE == sum(POINTS_PER_DOC.values())
    assert {doc: store.count(doc) for doc in POINTS_PER_DOC} == POINTS_PER_DOC


def test_reconcile_is_clean(store, engine, emb):
    report = reconcile(engine, store, emb)
    assert report.clean, {n: getattr(report, n)[:5] for n in report.PROBLEMS}
    assert (report.rows, report.embeddable, report.points) == (ROWS, EMBEDDABLE, EMBEDDABLE)


def test_citation_lookup_of_cob_3_3_2(store, engine):
    records = store.get_by_citation("COB 3.3.2")
    found = [(r.payload["key"], r.payload["chunk_type"], r.payload["binding"]) for r in records]
    assert found == [("COB 3.3.2", "rule", True), ("COB 3.3.2 Guidance", "guidance", False)]
    rule = records[0]
    with session_scope(engine) as s:
        row = ChunkRepository(s).by_key("COB 3.3.2")
        assert str(rule.id) == row.id
        assert rule.payload["content_hash"] == row.content_hash
        assert rule.payload["text"] == row.text and rule.payload["parent_key"] == "COB 3.3"
    assert rule.payload["text"].startswith("3.3.2 (1) Subject to (2)")


def test_a_split_rule_is_indexed_by_its_parts_only(store):
    keys = sorted(r.payload["key"] for r in store.get_by_citation("GEN 2.3.2"))
    assert keys == ["GEN 2.3.2#p1", "GEN 2.3.2#p2"]


def test_payload_filters_run_in_qdrant(store):
    binding_aml = models.Filter(must=[
        models.FieldCondition(key="module", match=models.MatchValue(value="AML")),
        models.FieldCondition(key="binding", match=models.MatchValue(value=True)),
    ])  # fmt: skip
    count = store.client.count(store.collection, count_filter=binding_aml, exact=True).count
    assert 0 < count < POINTS_PER_DOC["dfsa-rulebook-aml"]  # AML has non-binding Guidance too


def test_bm25_finds_an_exact_term(store):
    """Sparse search alone (no OpenAI call): the Glossary entry ranks first."""
    query = SparseEncoder(QdrantSettings()).encode_query("Market Counterparty")
    points = store.client.query_points(
        store.collection,
        query=models.SparseVector(indices=query.indices, values=query.values),
        using=SPARSE, limit=5, with_payload=["key"],
    ).points  # fmt: skip
    assert points[0].payload["key"] == "GLO Market Counterparty"


def test_a_dry_run_reingest_would_change_nothing(store, engine, emb):
    """Re-chunks every document and plans against SQLite and Qdrant: all current."""
    qdrant, cfg, paths = QdrantSettings(), ChunkingSettings(), PathSettings()
    manifest = load_manifest()
    entries = [e for e in manifest if supported(e)]
    if not all(e.path(paths.root).is_file() for e in entries):
        pytest.skip("source PDFs not present in data/raw")
    codes = rulebook_codes(manifest)
    index = Index(store, None, SparseEncoder(qdrant))  # a dry run never embeds
    result = ingest(
        entries, lambda e: chunk_document(e, cfg, paths, codes).chunks, engine,
        fingerprint=emb.fingerprint, sparse_model=qdrant.sparse_fingerprint, index=index,
        dry_run=True, prune_orphans=True,
    )  # fmt: skip
    assert result.run_id is None and result.orphans == {}
    assert result.stats.chunks_unchanged == ROWS
    assert (
        sum(len(d.diff.new) + len(d.diff.changed) + len(d.diff.removed) for d in result.docs) == 0
    )
    for d in result.docs:
        assert (d.plan.to_embed, d.plan.payload_only, d.plan.stale_ids) == ([], [], []), d.doc_id


def test_the_cache_collection_matches_the_embedding_model(store, emb):
    """Read-only: checked only once the API (or scripts/cache.py) has created it."""
    index = CacheIndex.from_settings(CacheSettings(), emb, client=store.client)
    if not store.client.collection_exists(index.collection):
        pytest.skip(f"Cache collection {index.collection} not created yet")
    index.check()  # raises EmbeddingModelMismatch on another model
    info = store.client.get_collection(index.collection)
    assert info.config.params.vectors["dense"].size == emb.dimensions
    assert set(info.payload_schema) == {"namespace", "expires_at"}
