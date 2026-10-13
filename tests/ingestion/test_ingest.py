"""`ingest()` end to end on a temporary database, with a fake chunker."""

import pytest
from sqlalchemy import func, select

from src.ingestion.ingest import ingest
from src.ingestion.models import ChunkType
from src.storage.db import session_scope
from src.storage.models import ChunkRow, Document, IngestionRun
from src.storage.repositories import ChunkRepository
from tests.factories import FINGERPRINT, chunk, entry

GLO = entry("dfsa-rulebook-glo", "GLO", "66")
COB = entry()


def corpus(cob_text="COB 3.3.2 text", drop=()):
    chunks = {
        GLO.id: [chunk("GLO Retail Client", doc_id=GLO.id, version="66", parent_key=None,
                       kind=ChunkType.DEFINITION, term="Retail Client")],
        COB.id: [
            chunk("COB 3.3", kind=ChunkType.SECTION),
            chunk("COB 3.3.1", order=1, defined_terms=["Retail Client"]),
            chunk("COB 3.3.2", order=2, text=cob_text, cross_refs=["GEN 2.26.1"]),
        ],
    }  # fmt: skip
    chunks[COB.id] = [c for c in chunks[COB.id] if c.key not in drop]
    return chunks


def run(engine, chunks, entries=(COB, GLO), **kw):
    order = []

    def chunker(e):
        order.append(e.id)
        return chunks[e.id]

    result = ingest(list(entries), chunker, engine, fingerprint=FINGERPRINT,
                    sparse_model="Qdrant/bm25", **kw)  # fmt: skip
    return result, order


def count(engine, model=ChunkRow):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_first_run_loads_everything_glossary_first(engine):
    result, order = run(engine, corpus())
    assert order == [GLO.id, COB.id]
    assert (result.stats.chunks_new, result.stats.docs_processed) == (4, 2)
    assert result.embed_tokens == 30  # 3 embeddable chunks x 10 tokens; the section isn't
    assert count(engine) == 4
    with session_scope(engine) as s:
        run_row = s.get(IngestionRun, result.run_id)
        assert run_row.status == "ok" and run_row.chunks_new == 4 and run_row.embed_cost_usd == 0
        assert s.get(Document, COB.id).chunk_count == 3


def test_rerun_changes_nothing_and_an_edit_changes_one_chunk(engine):
    run(engine, corpus())
    again, _ = run(engine, corpus())
    assert (again.stats.chunks_new, again.stats.chunks_changed) == (0, 0)
    assert again.stats.chunks_unchanged == 4

    edited, _ = run(engine, corpus(cob_text="amended"))
    assert edited.stats.chunks_changed == 1
    cob = next(d for d in edited.docs if d.doc_id == COB.id)
    assert [c.key for c in cob.diff.changed] == ["COB 3.3.2"]
    with session_scope(engine) as s:
        assert ChunkRepository(s).by_key("COB 3.3.2").text == "amended"
        assert s.scalar(select(func.count()).select_from(IngestionRun)) == 3


def test_removed_chunk_is_deleted(engine):
    run(engine, corpus())
    result, _ = run(engine, corpus(drop={"COB 3.3.1"}))
    assert result.stats.chunks_removed == 1
    with session_scope(engine) as s:
        assert ChunkRepository(s).by_key("COB 3.3.1") is None
        assert s.get(Document, COB.id).chunk_count == 2


def test_dry_run_writes_nothing(engine):
    result, _ = run(engine, corpus(), dry_run=True)
    assert result.stats.chunks_new == 4 and result.run_id is None
    assert count(engine) == 0 and count(engine, Document) == 0 and count(engine, IngestionRun) == 0


def test_documents_dropped_from_the_manifest_are_pruned_on_a_full_run(engine):
    run(engine, corpus())
    kept_only_glo, _ = run(engine, corpus(), entries=[GLO], prune_orphans=False)
    assert count(engine, Document) == 2  # a --doc run never prunes

    pruned, _ = run(engine, corpus(), entries=[GLO], prune_orphans=True)
    assert list(pruned.orphans) == [COB.id] and pruned.stats.chunks_removed == 3
    assert count(engine, Document) == 1 and count(engine) == 1


def test_a_failure_is_recorded_and_earlier_documents_stay_committed(engine):
    def chunker(e):
        if e.id == COB.id:
            raise RuntimeError("parse failed")
        return corpus()[e.id]

    with pytest.raises(RuntimeError, match="parse failed"):
        ingest([COB, GLO], chunker, engine, fingerprint=FINGERPRINT, sparse_model="Qdrant/bm25")
    with session_scope(engine) as s:
        failed = s.scalar(select(IngestionRun))
        assert failed.status == "failed" and "parse failed" in failed.error
        assert failed.docs_processed == 1 and failed.finished_at is not None
    assert count(engine) == 1  # GLO went first and its transaction committed
