"""reconcile() on a temporary SQLite and in-memory Qdrant, after real ingest() runs."""

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from src.ingestion.ingest import Index, ingest
from src.ingestion.reconcile import fix, reconcile
from src.storage.db import session_scope
from src.storage.models import ChunkRow
from src.storage.qdrant_store import CollectionMismatch, Point, QdrantStore
from tests.factories import FakeEmbedder, FakeSparse, embedding_settings, qdrant_settings
from tests.ingestion.test_ingest import COB, GLO, corpus

EMB = embedding_settings()
QD = qdrant_settings()


@pytest.fixture
def index():
    store = QdrantStore.from_settings(QD, EMB, client=QdrantClient(":memory:"))
    return Index(store, FakeEmbedder(EMB), FakeSparse(QD))


def run(engine, index=None):
    return ingest([COB, GLO], lambda e: corpus()[e.id], engine, fingerprint=EMB.fingerprint,
                  sparse_model=QD.sparse_fingerprint, index=index)  # fmt: skip


def point_id(engine, key):
    with session_scope(engine) as s:
        return s.scalar(select(ChunkRow.id).where(ChunkRow.key == key))


def test_clean_after_ingest(engine, index):
    run(engine, index)
    report = reconcile(engine, index.store, EMB)
    assert report.clean
    assert (report.rows, report.embeddable, report.points) == (4, 3, 3)


def test_rows_loaded_without_embedding_are_reported(engine, index):
    index.store.ensure_collection()
    run(engine)  # --no-embed
    report = reconcile(engine, index.store, EMB)
    assert not report.clean
    assert sorted(report.unembedded) == ["COB 3.3.1", "COB 3.3.2", "GLO Retail Client"]


def test_a_missing_point_is_found_fixed_and_reembedded(engine, index):
    run(engine, index)
    index.store.delete([point_id(engine, "COB 3.3.1")])
    report = reconcile(engine, index.store, EMB)
    assert report.missing_points == ["COB 3.3.1"] and not report.clean

    fix(engine, index.store, report)
    assert reconcile(engine, index.store, EMB).unembedded == ["COB 3.3.1"]
    index.embedder.calls.clear()
    run(engine, index)
    assert index.embedder.embedded_texts == 1
    assert reconcile(engine, index.store, EMB).clean


def test_a_point_with_a_stale_hash_is_reported(engine, index):
    run(engine, index)
    pid = point_id(engine, "COB 3.3.2")
    index.store.client.set_payload(index.store.collection, {"content_hash": "stale"}, [pid])
    assert reconcile(engine, index.store, EMB).hash_mismatch == ["COB 3.3.2"]


def test_a_row_embedded_with_another_model_is_reported(engine, index):
    run(engine, index)
    with session_scope(engine) as s:
        row = s.scalar(select(ChunkRow).where(ChunkRow.key == "GLO Retail Client"))
        row.embedding_dimensions = 3072
    assert reconcile(engine, index.store, EMB).wrong_model == ["GLO Retail Client"]


def test_orphan_points_are_found_and_deleted_by_fix(engine, index):
    run(engine, index)
    orphan = "00000000-0000-0000-0000-000000000001"
    sparse = FakeSparse().encode_query("orphan")
    index.store.upsert([Point(orphan, [0.1] * EMB.dimensions, sparse, {"doc_id": "gone"})])
    report = reconcile(engine, index.store, EMB)
    assert report.orphan_points == [orphan] and report.points == 4

    fix(engine, index.store, report)
    assert reconcile(engine, index.store, EMB).clean


def test_a_collection_built_differently_is_refused(engine, index):
    run(engine, index)
    other = QdrantStore(index.store.client, index.store.collection, EMB, "Qdrant/bm25-other")
    with pytest.raises(CollectionMismatch):
        reconcile(engine, other, EMB)
