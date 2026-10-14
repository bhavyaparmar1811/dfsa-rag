from datetime import date, datetime

from sqlalchemy import func, select

from src.ingestion.diff import diff_chunks
from src.ingestion.ids import chunk_id
from src.ingestion.models import ChunkType
from src.storage.db import session_scope
from src.storage.models import (
    ChunkRef,
    ChunkRow,
    ChunkTerm,
    Document,
    EvalRun,
    IngestionRun,
    Trace,
)
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EvalRunRepository,
    IngestionRunRepository,
    RunStats,
    TraceRepository,
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


# --- cost summaries -----------------------------------------------------------


def trace(i: int, day: str, *, refused=None, code=None, status="ok", answer=0.01, total_ms=1000):
    return Trace(
        id=f"t{i}", created_at=datetime.fromisoformat(f"{day}T12:00:00"), status=status,
        original_query="q", refused=refused, refusal_code=code,
        embed_tokens=10, embed_cost_usd=0.000001, rewrite_input_tokens=100,
        rewrite_output_tokens=10, rewrite_cost_usd=0.001, answer_input_tokens=5000 if answer else 0,
        answer_output_tokens=150 if answer else 0, answer_cost_usd=answer,
        total_cost_usd=0.000001 + 0.001 + answer, latency_ms={"total": total_ms},
    )  # fmt: skip


def seed_traces(engine):
    with session_scope(engine) as s:
        repo = TraceRepository(s)
        repo.add(trace(1, "2026-10-01", refused=False, total_ms=1000))
        repo.add(trace(2, "2026-10-01", refused=True, code="low_confidence", answer=0.0,
                       total_ms=2000))  # fmt: skip
        repo.add(trace(3, "2026-10-02", refused=True, code="citation_check_failed",
                       total_ms=3000))  # fmt: skip
        repo.add(trace(4, "2026-10-03", status="error", answer=0.0, total_ms=9000))


def test_cost_stats_breaks_down_refusals_providers_and_latency(engine):
    seed_traces(engine)
    with session_scope(engine) as s:
        stats = TraceRepository(s).cost_stats()
    assert (stats["requests"], stats["refused"], stats["errors"]) == (4, 2, 1)
    assert stats["refusal_rate"] == round(2 / 3, 4)  # errors are not refusals
    assert stats["refusals_by_code"] == {"citation_check_failed": 1, "low_confidence": 1}
    assert stats["gate_refusals"] == 1  # low_confidence never reached the LLM
    assert stats["cost_by_provider_usd"]["openai_embeddings"] == round(4 * 0.000001, 6)
    assert stats["cost_by_provider_usd"]["azure_generation"] == round(4 * 0.001 + 0.02, 6)
    assert stats["tokens"]["answer_input"] == 10000 and stats["tokens"]["embedding"] == 40
    assert stats["latency_ms"] == {"p50": 2000.0, "p95": 2900.0}  # error turns excluded
    assert [d["day"] for d in stats["by_day"]] == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert stats["by_day"][0]["requests"] == 2 and stats["by_day"][0]["gate_refusals"] == 1


def test_cost_stats_window_is_inclusive_utc_days(engine):
    seed_traces(engine)
    with session_scope(engine) as s:
        repo = TraceRepository(s)
        window = repo.cost_stats(date(2026, 10, 2), date(2026, 10, 3))
        empty = repo.cost_stats(date(2026, 11, 1))
    assert window["requests"] == 2 and (window["since"], window["until"]) == (
        "2026-10-02",
        "2026-10-03",
    )
    assert empty["requests"] == 0 and empty["cost_per_request_usd"] is None
    assert empty["latency_ms"] == {"p50": None, "p95": None} and empty["by_day"] == []


def test_eval_and_ingestion_cost_summaries(engine):
    with session_scope(engine) as s:
        for kind, cost, day in (
            ("e2e", {"embed": 0.0001, "answer": 0.8, "judge": 0.7, "total": 1.5001}, "2026-10-01"),
            ("e2e", {"embed": 0.0001, "answer": 0.9, "judge": 0.8, "total": 1.7001}, "2026-10-02"),
            ("retrieval", {"embed": 0.00002, "total": 0.00002}, "2026-10-02"),
        ):
            at = datetime.fromisoformat(f"{day}T09:00:00")
            s.add(EvalRun(kind=kind, config_name="hybrid_rerank", golden_hash="g", n_turns=64,
                          cost_usd=cost, created_at=at))  # fmt: skip
        runs = IngestionRunRepository(s)
        run_id = runs.start("fp", "Qdrant/bm25")
        runs.finish(run_id, RunStats(embed_tokens=1000, embed_cost_usd=0.00002))
        failed = runs.start("fp", "Qdrant/bm25")
        runs.finish(failed, RunStats(), error="boom")

    with session_scope(engine) as s:
        evals = EvalRunRepository(s).cost_summary()
        later = EvalRunRepository(s).cost_summary(since=date(2026, 10, 2))
        ingestion = IngestionRunRepository(s).cost_summary()
    e2e = evals["by_kind"]["e2e"]
    assert (e2e["runs"], e2e["turns"]) == (2, 128)
    assert e2e["cost_usd"] == {"answer": 1.7, "embed": 0.0002, "judge": 1.5, "total": 3.2002}
    assert evals["cost_usd"] == round(3.2002 + 0.00002, 6)
    assert later["by_kind"]["e2e"]["runs"] == 1 and "retrieval" in later["by_kind"]
    assert ingestion == {"runs": 2, "failed": 1, "embed_tokens": 1000, "cost_usd": 0.00002}
