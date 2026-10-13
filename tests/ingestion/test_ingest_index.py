"""`ingest()` with an index: in-memory Qdrant, a fake embedder, a fake BM25 encoder."""

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from src.embeddings.client import EmbeddingModelMismatch
from src.ingestion.ingest import Index, ingest
from src.storage.db import session_scope
from src.storage.models import ChunkRow, IngestionRun
from src.storage.qdrant_store import QdrantStore
from tests.factories import FakeEmbedder, FakeSparse, embedding_settings, qdrant_settings
from tests.ingestion.test_ingest import COB, GLO, corpus

EMB = embedding_settings()
QD = qdrant_settings()


@pytest.fixture
def index():
    store = QdrantStore.from_settings(QD, EMB, client=QdrantClient(":memory:"))
    return Index(store, FakeEmbedder(EMB), FakeSparse(QD))


def run(engine, index, chunks, entries=(COB, GLO), **kw):
    index.embedder.calls.clear()
    return ingest(list(entries), lambda e: chunks[e.id], engine, fingerprint=EMB.fingerprint,
                  sparse_model=QD.sparse_fingerprint, index=index, **kw)  # fmt: skip


def sqlite_state(engine) -> dict[str, tuple[bool, str | None]]:
    with session_scope(engine) as s:
        rows = s.scalars(select(ChunkRow)).all()
        return {r.key: (r.is_embedded, r.embedding_model) for r in rows}


def test_first_run_embeds_and_indexes_every_embeddable_chunk(engine, index):
    result = run(engine, index, corpus())
    assert result.collection_created
    assert index.store.count() == 3  # the section is stored in SQLite only
    assert index.embedder.embedded_texts == 3
    assert (result.stats.embed_tokens, result.stats.embed_cost_usd) == (21, 21 * 0.02 / 1e6)

    state = sqlite_state(engine)
    assert state["COB 3.3"] == (False, None)
    assert state["COB 3.3.1"] == (True, "text-embedding-3-small")
    with session_scope(engine) as s:
        run_row = s.get(IngestionRun, result.run_id)
        assert run_row.embed_tokens == 21 and run_row.sparse_model == QD.sparse_fingerprint
        row = s.scalar(select(ChunkRow).where(ChunkRow.key == "COB 3.3.1"))
        assert row.embedding_dimensions == EMB.dimensions and row.embedded_at is not None

    point = index.store.get_by_citation("COB 3.3.1")[0]
    assert point.payload["content_hash"] == row.content_hash


def test_rerun_embeds_nothing_and_costs_nothing(engine, index):
    run(engine, index, corpus())
    again = run(engine, index, corpus())
    assert index.embedder.calls == []
    assert (again.stats.embed_tokens, again.stats.embed_cost_usd) == (0, 0.0)
    assert again.stats.chunks_unchanged == 4 and index.store.count() == 3


def test_editing_one_chunk_reembeds_only_that_chunk(engine, index):
    run(engine, index, corpus())
    result = run(engine, index, corpus(cob_text="amended"))
    assert index.embedder.embedded_texts == 1
    assert index.embedder.calls[0][0].endswith("amended")
    assert result.stats.embed_tokens == 7 and index.store.count() == 3


def test_metadata_only_change_rewrites_the_payload_without_embedding(engine, index):
    run(engine, index, corpus())
    chunks = corpus()
    chunks[COB.id] = [c.model_copy(update={"page_start": 77}) if c.key == "COB 3.3.1" else c
                      for c in chunks[COB.id]]  # fmt: skip
    result = run(engine, index, chunks)
    assert index.embedder.calls == []
    cob = next(d for d in result.docs if d.doc_id == COB.id)
    assert len(cob.plan.payload_only) == 1
    assert index.store.get_by_citation("COB 3.3.1")[0].payload["page_start"] == 77


def test_removed_chunk_loses_its_point(engine, index):
    run(engine, index, corpus())
    result = run(engine, index, corpus(drop={"COB 3.3.1"}))
    cob = next(d for d in result.docs if d.doc_id == COB.id)
    assert len(cob.plan.stale_ids) == 1
    assert index.store.count() == 2 and index.store.get_by_citation("COB 3.3.1") == []


def test_a_lost_collection_is_rebuilt_from_scratch(engine, index):
    """SQLite still says 'embedded', but the points are gone: the next run re-embeds."""
    run(engine, index, corpus())
    index.store.client.delete_collection(index.store.collection)
    result = run(engine, index, corpus())
    assert result.collection_created and index.embedder.embedded_texts == 3
    assert index.store.count() == 3


def test_a_no_embed_run_then_an_index_run(engine, index):
    ingest([COB, GLO], lambda e: corpus()[e.id], engine, fingerprint=EMB.fingerprint,
           sparse_model=QD.sparse_fingerprint)  # fmt: skip
    assert not any(embedded for embedded, _ in sqlite_state(engine).values())
    run(engine, index, corpus())
    assert index.embedder.embedded_texts == 3
    assert sqlite_state(engine)["GLO Retail Client"][0] is True


def test_points_that_already_exist_are_marked_without_reembedding(engine, index):
    """E.g. SQLite was rebuilt with --no-embed while the collection survived."""
    run(engine, index, corpus())
    with session_scope(engine) as s:
        for row in s.scalars(select(ChunkRow)):
            row.is_embedded, row.embedding_model = False, None
    run(engine, index, corpus())
    assert index.embedder.calls == []
    assert sqlite_state(engine)["COB 3.3.2"] == (True, "text-embedding-3-small")


def test_dry_run_creates_nothing_and_embeds_nothing(engine, index):
    result = run(engine, index, corpus(), dry_run=True)
    assert not index.store.exists() and index.embedder.calls == []
    assert sum(len(d.plan.to_embed) for d in result.docs) == 3
    assert result.estimated_tokens == 30


def test_pruning_a_document_deletes_its_points(engine, index):
    run(engine, index, corpus())
    result = run(engine, index, corpus(), entries=[GLO], prune_orphans=True)
    assert list(result.orphans) == [COB.id]
    assert index.store.count() == 1 and index.store.count(COB.id) == 0


def test_a_mismatched_fingerprint_is_refused(engine, index):
    with pytest.raises(EmbeddingModelMismatch):
        ingest([COB], lambda e: corpus()[e.id], engine, fingerprint="other-model-1536",
               sparse_model=QD.sparse_fingerprint, index=index)  # fmt: skip
