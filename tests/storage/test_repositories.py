from sqlalchemy import func, select

from src.ingestion.diff import diff_chunks
from src.ingestion.ids import chunk_id
from src.ingestion.models import ChunkType
from src.storage.db import session_scope
from src.storage.models import ChunkRef, ChunkRow, ChunkTerm, Document, IngestionRun
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionRunRepository,
    RunStats,
)
from tests.factories import FINGERPRINT, chunk, entry


def section_3_3():
    return [
        chunk("COB 3.3", kind=ChunkType.SECTION, order=0),
        chunk("COB 3.3.1", order=1, scope_rule=True, defined_terms=["Retail Client"]),
        chunk("COB 3.3.2", order=2, parts=["COB 3.3.2#p1", "COB 3.3.2#p2"],
              cross_refs=["GEN 2.26.1", "COB 3.3.4"], defined_terms=["Client Agreement"]),
        chunk("COB 3.3.2#p1", order=3, part_of="COB 3.3.2", part=1, cross_refs=["GEN 2.26.1"]),
        chunk("COB 3.3.2#p2", order=4, part_of="COB 3.3.2", part=2, cross_refs=["COB 3.3.4"]),
        chunk("COB 3.3.2 Guidance", order=5, kind=ChunkType.GUIDANCE, guidance_for="COB 3.3.2",
              defined_terms=["Retail Client", "Retail Client"]),  # duplicates are stored once
        chunk("COB 3.3.4", order=6, cross_refs=["GEN 2.26.1"]),
    ]  # fmt: skip


def load(engine, chunks, e=None):
    e = e or entry()
    with session_scope(engine) as s:
        repo = ChunkRepository(s)
        diff = diff_chunks(chunks, repo.existing(e.id), FINGERPRINT)
        DocumentRepository(s).upsert(e)
        repo.delete(r.key for r in diff.removed)
        repo.upsert(chunks, diff.hashes)
        DocumentRepository(s).update_counts(e.id)
    return diff


def test_upsert_writes_rows_refs_and_terms(engine):
    load(engine, section_3_3())
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(ChunkRow)) == 7
        row = ChunkRepository(s).by_key("COB 3.3.2")
        assert row.citation == "COB 3.3.2" and row.parts == ["COB 3.3.2#p1", "COB 3.3.2#p2"]
        assert row.content_hash and not row.is_embedded and row.embedding_model is None
        assert ChunkRepository(s).refs_of(row.id) == ["COB 3.3.4", "GEN 2.26.1"]
        guidance = ChunkRepository(s).by_key("COB 3.3.2 Guidance")
        assert ChunkRepository(s).terms_of(guidance.id) == ["Retail Client"]
        doc = s.get(Document, "dfsa-rulebook-cob")
        assert (doc.chunk_count, doc.embedded_count, doc.version) == (7, 0, "51")


def test_lookups(engine):
    load(engine, section_3_3())
    with session_scope(engine) as s:
        repo = ChunkRepository(s)
        assert [r.key for r in repo.children_of("COB 3.3")] == [
            "COB 3.3.1", "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.4",
        ]  # fmt: skip
        assert [r.key for r in repo.parts_of("COB 3.3.2")] == ["COB 3.3.2#p1", "COB 3.3.2#p2"]
        assert [r.key for r in repo.referenced_by("GEN 2.26.1")] == [
            "COB 3.3.2", "COB 3.3.2#p1", "COB 3.3.4",
        ]  # fmt: skip
        assert [r.key for r in repo.with_term("Retail Client")] == [
            "COB 3.3.1", "COB 3.3.2 Guidance",
        ]  # fmt: skip
        assert repo.by_key("COB 9.9.9") is None


def test_embedding_state_survives_an_unchanged_reload_but_not_an_edit(engine):
    chunks = section_3_3()
    load(engine, chunks)
    with session_scope(engine) as s:
        ids = [c.chunk_id for c in chunks if c.embedded]
        ChunkRepository(s).mark_embedded(ids, "text-embedding-3-small", 1536)
        DocumentRepository(s).update_counts("dfsa-rulebook-cob")
        assert s.get(Document, "dfsa-rulebook-cob").embedded_count == 5

    edited = [c.model_copy(update={"text": "amended"}) if c.key == "COB 3.3.4" else c
              for c in section_3_3()]  # fmt: skip
    diff = load(engine, edited)
    assert [c.key for c in diff.to_embed] == ["COB 3.3.4"]
    with session_scope(engine) as s:
        repo = ChunkRepository(s)
        kept = repo.by_key("COB 3.3.1")
        assert kept.is_embedded and kept.embedding_model == "text-embedding-3-small"
        assert kept.embedding_dimensions == 1536 and kept.embedded_at is not None
        reset = repo.by_key("COB 3.3.4")
        assert not reset.is_embedded and reset.embedding_model is None and reset.text == "amended"
        assert s.get(Document, "dfsa-rulebook-cob").embedded_count == 4


def test_references_and_terms_are_replaced_not_accumulated(engine):
    load(engine, section_3_3())
    changed = [c.model_copy(update={"cross_refs": ["MKT 1.1.1"], "defined_terms": []})
               if c.key == "COB 3.3.4" else c for c in section_3_3()]  # fmt: skip
    load(engine, changed)
    with session_scope(engine) as s:
        repo = ChunkRepository(s)
        assert repo.refs_of(repo.by_key("COB 3.3.4").id) == ["MKT 1.1.1"]
        assert [r.key for r in repo.referenced_by("GEN 2.26.1")] == ["COB 3.3.2", "COB 3.3.2#p1"]


def test_removed_chunks_are_deleted_with_their_refs_and_terms(engine):
    load(engine, section_3_3())
    diff = load(engine, [c for c in section_3_3() if c.key != "COB 3.3.1"])
    assert [r.key for r in diff.removed] == ["COB 3.3.1"]
    with session_scope(engine) as s:
        assert ChunkRepository(s).by_key("COB 3.3.1") is None
        assert ChunkRepository(s).with_term("Retail Client")[0].key == "COB 3.3.2 Guidance"
        assert s.get(Document, "dfsa-rulebook-cob").chunk_count == 6
        orphans = s.scalar(
            select(func.count()).select_from(ChunkTerm).where(ChunkTerm.chunk_id.not_in(
                select(ChunkRow.id)))
        )  # fmt: skip
        assert orphans == 0


def test_a_new_document_version_replaces_rows_under_the_same_keys(engine):
    load(engine, section_3_3())
    v52 = [c.model_copy(update={"version": "52", "chunk_id": chunk_id(c.doc_id, "52", c.key)})
           for c in section_3_3()]  # fmt: skip
    diff = load(engine, v52, entry(version="52"))
    assert len(diff.changed) == 7
    with session_scope(engine) as s:
        row = ChunkRepository(s).by_key("COB 3.3.2")
        assert row.id == v52[2].chunk_id
        assert s.scalar(select(func.count()).select_from(ChunkRow)) == 7
        assert s.scalar(select(func.count()).select_from(ChunkRef)) == 5
        assert s.get(Document, "dfsa-rulebook-cob").version == "52"


def test_deleting_a_document_returns_what_it_removed(engine):
    load(engine, section_3_3())
    with session_scope(engine) as s:
        removed = DocumentRepository(s).delete("dfsa-rulebook-cob")
    assert len(removed) == 7
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(ChunkRow)) == 0


def test_ingestion_run_lifecycle(engine):
    with session_scope(engine) as s:
        run_id = IngestionRunRepository(s).start(FINGERPRINT, "Qdrant/bm25")
    with session_scope(engine) as s:
        assert s.get(IngestionRun, run_id).status == "running"
        IngestionRunRepository(s).finish(run_id, RunStats(docs_processed=2, chunks_new=7))
    with session_scope(engine) as s:
        run = IngestionRunRepository(s).latest()
        assert (run.id, run.status, run.chunks_new, run.error) == (run_id, "ok", 7, None)
        assert run.finished_at is not None and run.sparse_model == "Qdrant/bm25"
