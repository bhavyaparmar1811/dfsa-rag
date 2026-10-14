"""Reads and writes for the ingestion, chat and eval tables. The rest of the app never touches
the ORM models directly, so the storage backend can change behind this file.

Each repository wraps a `Session` and never commits: the caller decides the
transaction (`session_scope()`), e.g. one transaction per ingested document.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta

from sqlalchemy import Integer, delete, func, insert, select, update
from sqlalchemy.orm import Session

from src.ingestion.diff import StoredChunk
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk
from src.observability.stats import percentile
from src.storage.models import (
    CacheEntry,
    ChunkRef,
    ChunkRow,
    ChunkTerm,
    Conversation,
    Document,
    EvalResult,
    EvalRun,
    IngestionRun,
    Message,
    Trace,
    utcnow,
)


class DocumentRepository:
    def __init__(self, session: Session):
        self.s = session

    def upsert(self, entry: ManifestEntry) -> Document:
        doc = self.s.get(Document, entry.id) or Document(id=entry.id)
        doc.regulator = entry.regulator
        doc.doc_type = entry.doc_type
        doc.module_code = entry.module_code
        doc.module_name = entry.module_name
        doc.version = entry.version
        doc.effective_date = entry.effective_date
        doc.source_url = entry.source_url
        doc.file_path = entry.file_path
        doc.file_hash = entry.file_hash
        doc.ingested_at = utcnow()
        self.s.add(doc)
        self.s.flush()
        return doc

    def update_counts(self, doc_id: str) -> None:
        total, embedded = self.s.execute(
            select(func.count(), func.coalesce(func.sum(ChunkRow.is_embedded.cast(Integer)), 0))
            .where(ChunkRow.document_id == doc_id)
        ).one()  # fmt: skip
        doc = self.s.get(Document, doc_id)
        doc.chunk_count, doc.embedded_count = total, embedded

    def ids(self) -> set[str]:
        return set(self.s.scalars(select(Document.id)))

    def delete(self, doc_id: str) -> list[StoredChunk]:
        """Delete a document and (by cascade) its chunks; returns what was removed."""
        removed = list(ChunkRepository(self.s).existing(doc_id).values())
        self.s.execute(delete(Document).where(Document.id == doc_id))
        return removed


class ChunkRepository:
    def __init__(self, session: Session):
        self.s = session

    def existing(self, doc_id: str) -> dict[str, StoredChunk]:
        """{key: StoredChunk} for a document, for change detection."""
        rows = self.s.execute(
            select(ChunkRow.id, ChunkRow.key, ChunkRow.content_hash, ChunkRow.is_embedded)
            .where(ChunkRow.document_id == doc_id)
        )  # fmt: skip
        return {r.key: StoredChunk(r.id, r.key, r.content_hash, r.is_embedded) for r in rows}

    def index_states(self) -> list:
        """Every row's index-related columns, for reconcile."""
        return list(
            self.s.execute(
                select(
                    ChunkRow.id, ChunkRow.key, ChunkRow.document_id, ChunkRow.chunk_type,
                    ChunkRow.deleted, ChunkRow.parts, ChunkRow.content_hash, ChunkRow.is_embedded,
                    ChunkRow.embedding_model, ChunkRow.embedding_dimensions,
                ).order_by(ChunkRow.document_id, ChunkRow.order_idx)
            )
        )  # fmt: skip

    def content_hashes(self, ids: Iterable[str]) -> dict[str, str]:
        """{chunk id: content_hash} for the ids that exist (the cache checks a cited
        chunk is still the text it was answered from)."""
        rows = self.s.execute(
            select(ChunkRow.id, ChunkRow.content_hash).where(ChunkRow.id.in_(list(ids)))
        )
        return dict(rows.all())

    def corpus_hash(self) -> str:
        """sha256 over every embedded chunk's (key, content_hash): changes whenever an
        ingest changes searchable text."""
        digest = hashlib.sha256()
        rows = self.s.execute(
            select(ChunkRow.key, ChunkRow.content_hash)
            .where(ChunkRow.is_embedded.is_(True))
            .order_by(ChunkRow.key)
        )
        for key, content_hash in rows:
            digest.update(f"{key}|{content_hash}".encode())
        return digest.hexdigest()

    def clear_embedded(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if ids:
            self.s.execute(
                update(ChunkRow)
                .where(ChunkRow.id.in_(ids))
                .values(is_embedded=False, embedding_model=None, embedding_dimensions=None,
                        embedded_at=None),
                execution_options={"synchronize_session": False},
            )  # fmt: skip

    def upsert(self, chunks: list[Chunk], hashes: dict[str, str]) -> None:
        """Write `chunks` (one document's) and replace their references and terms.

        `hashes` is {chunk_id: content_hash}. A row keeps its embedding state only
        if its id and content_hash are unchanged and it is still embeddable;
        otherwise it is marked not embedded until it is embedded again.
        """
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        keys = [c.key for c in chunks]
        # A key stored under another id (a new document version): drop the old row
        # first, so the unique key is free for the new one.
        self.s.execute(
            delete(ChunkRow).where(ChunkRow.key.in_(keys), ChunkRow.id.not_in(ids)),
            execution_options={"synchronize_session": False},
        )
        rows = {r.id: r for r in self.s.scalars(select(ChunkRow).where(ChunkRow.id.in_(ids)))}

        for chunk in chunks:
            h = hashes[chunk.chunk_id]
            row = rows.get(chunk.chunk_id)
            if row is None:
                row = ChunkRow(id=chunk.chunk_id, is_embedded=False)
                self.s.add(row)
            elif row.content_hash != h or not chunk.embedded:
                _clear_embedding(row)
            _fill(row, chunk, h)
        self.s.flush()

        self.s.execute(delete(ChunkRef).where(ChunkRef.chunk_id.in_(ids)))
        self.s.execute(delete(ChunkTerm).where(ChunkTerm.chunk_id.in_(ids)))
        refs = [
            {"chunk_id": c.chunk_id, "target_key": k} for c in chunks for k in set(c.cross_refs)
        ]
        terms = [{"chunk_id": c.chunk_id, "term": t} for c in chunks for t in set(c.defined_terms)]
        if refs:
            self.s.execute(insert(ChunkRef), refs)
        if terms:
            self.s.execute(insert(ChunkTerm), terms)

    def delete(self, keys: Iterable[str]) -> int:
        keys = list(keys)
        if not keys:
            return 0
        result = self.s.execute(
            delete(ChunkRow).where(ChunkRow.key.in_(keys)),
            execution_options={"synchronize_session": False},
        )
        return result.rowcount

    def mark_embedded(
        self,
        ids: Iterable[str],
        model: str,
        dimensions: int,
        at: datetime | None = None,
        only_unmarked: bool = False,
    ) -> None:
        """Record that these chunks have a current point. With `only_unmarked`, rows
        already marked keep their original `embedded_at`."""
        ids = list(ids)
        if ids:
            stmt = update(ChunkRow).where(ChunkRow.id.in_(ids))
            if only_unmarked:
                stmt = stmt.where(ChunkRow.is_embedded.is_(False))
            self.s.execute(
                stmt.values(is_embedded=True, embedding_model=model,
                            embedding_dimensions=dimensions, embedded_at=at or utcnow()),
                execution_options={"synchronize_session": False},
            )  # fmt: skip

    # --- lookups -------------------------------------------------------------

    def by_key(self, key: str) -> ChunkRow | None:
        return self.s.scalar(select(ChunkRow).where(ChunkRow.key == key))

    def by_citations(self, citations: Iterable[str]) -> list[ChunkRow]:
        """Every row citing one of `citations`: a rule, its parts and its Guidance."""
        rows = select(ChunkRow).where(ChunkRow.citation.in_(list(citations)))
        return list(self.s.scalars(rows.order_by(ChunkRow.document_id, ChunkRow.order_idx)))

    def by_keys(self, keys: Iterable[str]) -> list[ChunkRow]:
        return list(self.s.scalars(select(ChunkRow).where(ChunkRow.key.in_(list(keys)))))

    def definitions_for(self, terms: Iterable[str]) -> list[ChunkRow]:
        """Definition chunks (Glossary and Law Schedules) of these Defined Terms."""
        rows = select(ChunkRow).where(
            ChunkRow.chunk_type == "definition", ChunkRow.term.in_(list(terms))
        )
        return list(self.s.scalars(rows.order_by(ChunkRow.document_id, ChunkRow.order_idx)))

    def term_shares(self) -> dict[str, float]:
        """{term: share of non-section chunks using it}: how common each Defined Term is."""
        total = self.s.scalar(
            select(func.count()).select_from(ChunkRow).where(ChunkRow.chunk_type != "section")
        )
        counts = self.s.execute(
            select(ChunkTerm.term, func.count())
            .join(ChunkRow, ChunkRow.id == ChunkTerm.chunk_id)
            .where(ChunkRow.chunk_type != "section")
            .group_by(ChunkTerm.term)
        )  # fmt: skip
        return {term: n / total for term, n in counts} if total else {}

    def defined_terms(self) -> list[str]:
        return sorted(set(self.s.scalars(select(ChunkRow.term).where(ChunkRow.term.is_not(None)))))

    def by_ids(self, ids: Iterable[str]) -> list[ChunkRow]:
        return list(self.s.scalars(select(ChunkRow).where(ChunkRow.id.in_(list(ids)))))

    def children_of(self, parent_key: str) -> list[ChunkRow]:
        """A section's children in document order (parts of split rules excluded)."""
        return list(
            self.s.scalars(
                select(ChunkRow)
                .where(ChunkRow.parent_key == parent_key, ChunkRow.part_of.is_(None))
                .order_by(ChunkRow.order_idx)
            )
        )

    def parts_of(self, key: str) -> list[ChunkRow]:
        return list(
            self.s.scalars(select(ChunkRow).where(ChunkRow.part_of == key).order_by(ChunkRow.part))
        )

    def referenced_by(self, target_key: str) -> list[ChunkRow]:
        """Chunks whose text cites `target_key` ("Regulatory Law Art. 76")."""
        return list(
            self.s.scalars(
                select(ChunkRow)
                .join(ChunkRef, ChunkRef.chunk_id == ChunkRow.id)
                .where(ChunkRef.target_key == target_key)
                .order_by(ChunkRow.document_id, ChunkRow.order_idx)
            )
        )

    def with_term(self, term: str) -> list[ChunkRow]:
        """Chunks that use the Defined Term `term` ("Retail Client")."""
        return list(
            self.s.scalars(
                select(ChunkRow)
                .join(ChunkTerm, ChunkTerm.chunk_id == ChunkRow.id)
                .where(ChunkTerm.term == term)
                .order_by(ChunkRow.document_id, ChunkRow.order_idx)
            )
        )

    def refs_of(self, chunk_id: str) -> list[str]:
        return sorted(
            self.s.scalars(select(ChunkRef.target_key).where(ChunkRef.chunk_id == chunk_id))
        )

    def terms_of(self, chunk_id: str) -> list[str]:
        return sorted(self.s.scalars(select(ChunkTerm.term).where(ChunkTerm.chunk_id == chunk_id)))


def _fill(row: ChunkRow, chunk: Chunk, h: str) -> None:
    row.document_id = chunk.doc_id
    row.key = chunk.key
    row.citation = chunk.citation
    row.chunk_type = chunk.chunk_type.value
    row.binding = chunk.binding
    row.rule_number = chunk.rule_number
    row.section = chunk.section
    row.term = chunk.term
    row.parent_key = chunk.parent_key
    row.part_of = chunk.part_of
    row.guidance_for = chunk.guidance_for
    row.part = chunk.part
    row.order_idx = chunk.order
    row.heading_path = chunk.heading_path
    row.guidance_keys = chunk.guidance_keys
    row.parts = chunk.parts
    row.subheading = chunk.subheading
    row.scope_rule = chunk.scope_rule
    row.deleted = chunk.deleted
    row.header_text = chunk.header_text
    row.text = chunk.text
    row.page_start = chunk.page_start
    row.page_end = chunk.page_end
    row.token_count = chunk.token_count
    row.content_hash = h


def _clear_embedding(row: ChunkRow) -> None:
    row.is_embedded = False
    row.embedding_model = None
    row.embedding_dimensions = None
    row.embedded_at = None


@dataclass
class RunStats:
    """The counters of one `ingest` run; field names match `ingestion_runs`."""

    docs_processed: int = 0
    chunks_new: int = 0
    chunks_changed: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    embed_tokens: int = 0
    embed_cost_usd: float = 0.0


class IngestionRunRepository:
    def __init__(self, session: Session):
        self.s = session

    def start(self, embedding_fingerprint: str, sparse_model: str) -> int:
        run = IngestionRun(embedding_fingerprint=embedding_fingerprint, sparse_model=sparse_model)
        self.s.add(run)
        self.s.flush()
        return run.id

    def finish(self, run_id: int, stats: RunStats, error: str | None = None) -> None:
        run = self.s.get(IngestionRun, run_id)
        for f in fields(stats):
            setattr(run, f.name, getattr(stats, f.name))
        run.status = "failed" if error else "ok"
        run.error = error
        run.finished_at = utcnow()

    def latest(self) -> IngestionRun | None:
        return self.s.scalar(select(IngestionRun).order_by(IngestionRun.id.desc()).limit(1))

    def last_finished_id(self) -> int:
        """The newest finished run's id (0 if none): cheap to poll for "has the corpus
        changed?" (a run still in progress counts once it finishes)."""
        query = select(func.coalesce(func.max(IngestionRun.id), 0)).where(
            IngestionRun.finished_at.is_not(None)
        )
        return self.s.scalar(query)

    def cost_summary(self, since: date | None = None, until: date | None = None) -> dict:
        """Ingestion runs started in a window of UTC days: runs, embedding tokens and $."""
        cols = (
            func.count(),
            func.coalesce(func.sum((IngestionRun.status == "failed").cast(Integer)), 0),
            func.coalesce(func.sum(IngestionRun.embed_tokens), 0),
            func.coalesce(func.sum(IngestionRun.embed_cost_usd), 0.0),
        )
        query = _in_window(select(*cols), IngestionRun.started_at, since, until)
        runs, failed, tokens, cost = self.s.execute(query).one()
        return {"runs": runs, "failed": failed, "embed_tokens": tokens, "cost_usd": round(cost, 6)}


# --- chat ---------------------------------------------------------------------


class ConversationRepository:
    def __init__(self, session: Session):
        self.s = session

    def create(self, conversation_id: str, title: str = "") -> Conversation:
        conv = Conversation(id=conversation_id, title=title)
        self.s.add(conv)
        self.s.flush()
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self.s.get(Conversation, conversation_id)

    def touch(self, conv: Conversation, title: str | None = None) -> None:
        """Mark it updated, and give it a title if it has none yet."""
        if title and not conv.title:
            conv.title = title
        conv.updated_at = utcnow()

    def recent_messages(self, conversation_id: str, n: int) -> list[Message]:
        """The last `n` messages, oldest first."""
        rows = self.s.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(n)
        )
        return list(rows)[::-1]

    def messages(self, conversation_id: str) -> list[Message]:
        query = select(Message).where(Message.conversation_id == conversation_id)
        return list(self.s.scalars(query.order_by(Message.id)))


class MessageRepository:
    def __init__(self, session: Session):
        self.s = session

    def add(self, message: Message) -> Message:
        self.s.add(message)
        self.s.flush()
        return message


class TraceRepository:
    def __init__(self, session: Session):
        self.s = session

    def add(self, trace: Trace) -> Trace:
        self.s.add(trace)
        self.s.flush()
        return trace

    def get(self, trace_id: str) -> Trace | None:
        return self.s.get(Trace, trace_id)

    def set_cache_entry(self, trace_id: str, entry_id: str) -> None:
        """The answer of this turn was stored in the cache as `entry_id`."""
        trace = self.s.get(Trace, trace_id)
        trace.cache_entry_id = entry_id
        trace.cache = (trace.cache or {}) | {"stored_entry_id": entry_id}

    def cost_stats(self, since: date | None = None, until: date | None = None) -> dict:
        """Production requests in a window of UTC days (both inclusive), in total and
        per day: requests, refusals by code, tokens, $ by stage and by provider, latency."""
        query = _in_window(select(*_COST_COLUMNS), Trace.created_at, since, until)
        rows = self.s.execute(query.order_by(Trace.created_at)).all()
        by_day: dict[str, list] = {}
        for r in rows:
            by_day.setdefault(r.created_at.date().isoformat(), []).append(r)
        return (
            {"since": _iso(since), "until": _iso(until)}
            | _trace_summary(rows)
            | {"by_day": [{"day": d} | _trace_summary(day) for d, day in by_day.items()]}
        )


# Refused by the confidence gate, before any LLM call (src/generation/answer.py).
GATE_REFUSALS = frozenset({"no_context", "low_confidence"})

# Only the columns the summary needs: the JSON traces of a request are large.
_COST_COLUMNS = (
    Trace.created_at,
    Trace.status,
    Trace.refused,
    Trace.refusal_code,
    Trace.embed_tokens,
    Trace.embed_cost_usd,
    Trace.rewrite_input_tokens,
    Trace.rewrite_output_tokens,
    Trace.rewrite_cost_usd,
    Trace.answer_input_tokens,
    Trace.answer_output_tokens,
    Trace.answer_cost_usd,
    Trace.total_cost_usd,
    Trace.latency_ms,
    Trace.cache_status,
    Trace.cache,
)


def _trace_summary(rows) -> dict:
    n = len(rows)
    errors = sum(r.status == "error" for r in rows)
    codes = Counter(r.refusal_code for r in rows if r.refused)
    refused = sum(codes.values())
    embed, rewrite, answer, total = (
        sum(getattr(r, f) for r in rows)
        for f in ("embed_cost_usd", "rewrite_cost_usd", "answer_cost_usd", "total_cost_usd")
    )
    latencies = [r.latency_ms["total"] for r in rows if r.status == "ok" and r.latency_ms]
    return {
        "requests": n,
        "refused": refused,
        "errors": errors,
        "refusal_rate": round(refused / (n - errors), 4) if n - errors else None,
        "refusals_by_code": {str(code): k for code, k in sorted(codes.items(), key=str)},
        "gate_refusals": sum(codes[c] for c in GATE_REFUSALS),
        "cost_usd": {
            "embedding": round(embed, 6),
            "rewrite": round(rewrite, 6),
            "answer": round(answer, 6),
            "total": round(total, 6),
        },
        "cost_by_provider_usd": {
            "openai_embeddings": round(embed, 6),
            "azure_generation": round(rewrite + answer, 6),
        },
        "cost_per_request_usd": round(total / n, 6) if n else None,
        "tokens": {
            "embedding": sum(r.embed_tokens for r in rows),
            "rewrite_input": sum(r.rewrite_input_tokens for r in rows),
            "rewrite_output": sum(r.rewrite_output_tokens for r in rows),
            "answer_input": sum(r.answer_input_tokens for r in rows),
            "answer_output": sum(r.answer_output_tokens for r in rows),
        },
        "latency_ms": {
            "p50": _round(percentile(latencies, 50), 1),
            "p95": _round(percentile(latencies, 95), 1),
        },
        "cache": _cache_summary(rows),
    }


def _cache_summary(rows) -> dict:
    """Lookups are turns that asked the cache (hit or miss); bypassed turns (staging
    labels, the cache switched off) and turns from before the cache don't count."""
    hits = [r for r in rows if r.cache_status == "hit"]
    lookups = len(hits) + sum(r.cache_status == "miss" for r in rows)
    latencies = [r.latency_ms["total"] for r in hits if r.latency_ms]
    return {
        "lookups": lookups,
        "hits": len(hits),
        "exact_hits": sum((r.cache or {}).get("kind") == "exact" for r in hits),
        "hit_rate": round(len(hits) / lookups, 4) if lookups else None,
        # what the hits would have cost uncached (the original turns), minus what they cost
        "saved_usd": round(
            sum((r.cache or {}).get("source_cost_usd", 0.0) - r.total_cost_usd for r in hits), 6
        ),
        "hit_latency_p50_ms": _round(percentile(latencies, 50), 1),
    }


def _in_window(query, column, since: date | None, until: date | None):
    """Rows from the start of `since` to the end of `until` (UTC days, both inclusive)."""
    if since is not None:
        query = query.where(column >= datetime.combine(since, time.min))
    if until is not None:
        query = query.where(column < datetime.combine(until + timedelta(days=1), time.min))
    return query


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


# --- semantic cache -------------------------------------------------------------


class CacheRepository:
    def __init__(self, session: Session):
        self.s = session

    def add(self, entry: CacheEntry) -> CacheEntry:
        self.s.add(entry)
        self.s.flush()
        return entry

    def get(self, entry_id: str) -> CacheEntry | None:
        return self.s.get(CacheEntry, entry_id)

    def by_hash(self, namespace: str, query_hash: str, now: datetime) -> CacheEntry | None:
        """The newest unexpired entry for exactly this (normalised) question."""
        return self.s.scalar(
            select(CacheEntry)
            .where(
                CacheEntry.namespace == namespace,
                CacheEntry.query_hash == query_hash,
                CacheEntry.expires_at > now,
            )
            .order_by(CacheEntry.created_at.desc())
            .limit(1)
        )

    def record_hit(self, entry_id: str, now: datetime) -> None:
        self.s.execute(
            update(CacheEntry)
            .where(CacheEntry.id == entry_id)
            .values(hit_count=CacheEntry.hit_count + 1, last_hit_at=now)
        )

    def meta(self) -> list[tuple[str, dict, datetime]]:
        """(id, components, expires_at) of every entry, for purging stale ones."""
        query = select(CacheEntry.id, CacheEntry.components, CacheEntry.expires_at)
        return [tuple(r) for r in self.s.execute(query)]

    def ids(self) -> set[str]:
        return set(self.s.scalars(select(CacheEntry.id)))

    def delete(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if ids:
            self.s.execute(delete(CacheEntry).where(CacheEntry.id.in_(ids)))
        return len(ids)

    def stats(self, now: datetime) -> dict:
        n, live, hits, namespaces = self.s.execute(
            select(
                func.count(),
                func.coalesce(func.sum((CacheEntry.expires_at > now).cast(Integer)), 0),
                func.coalesce(func.sum(CacheEntry.hit_count), 0),
                func.count(func.distinct(CacheEntry.namespace)),
            )
        ).one()
        return {"entries": n, "unexpired": live, "hits_served": hits, "namespaces": namespaces}


# --- evals --------------------------------------------------------------------


class EvalRunRepository:
    def __init__(self, session: Session):
        self.s = session

    def start(self, run: EvalRun) -> EvalRun:
        self.s.add(run)
        self.s.flush()
        return run

    def add_result(self, result: EvalResult) -> EvalResult:
        self.s.add(result)
        self.s.flush()
        return result

    def finish(
        self,
        run_id: int,
        *,
        metrics: dict,
        cost_usd: dict,
        gate_status: str = "not_gated",
        gate_report: dict | None = None,
        error: str | None = None,
    ) -> EvalRun:
        run = self.s.get(EvalRun, run_id)
        run.metrics, run.cost_usd = metrics, cost_usd
        run.gate_status, run.gate_report = gate_status, gate_report
        run.status = "error" if error else "ok"
        run.error = error
        run.finished_at = utcnow()
        return run

    def set_gate(self, run_id: int, gate_status: str, gate_report: dict) -> None:
        run = self.s.get(EvalRun, run_id)
        run.gate_status, run.gate_report = gate_status, gate_report

    def get(self, run_id: int) -> EvalRun | None:
        return self.s.get(EvalRun, run_id)

    def latest(
        self,
        *,
        kind: str | None = None,
        config_name: str | None = None,
        golden_hash: str | None = None,
        gate_status: str | None = None,
        before: int | None = None,
    ) -> EvalRun | None:
        """The newest finished (status ok) run matching every filter given."""
        query = select(EvalRun).where(EvalRun.status == "ok")
        for column, value in (
            (EvalRun.kind, kind),
            (EvalRun.config_name, config_name),
            (EvalRun.golden_hash, golden_hash),
            (EvalRun.gate_status, gate_status),
        ):
            if value is not None:
                query = query.where(column == value)
        if before is not None:
            query = query.where(EvalRun.id < before)
        return self.s.scalar(query.order_by(EvalRun.id.desc()).limit(1))

    def results(self, run_id: int) -> list[EvalResult]:
        query = select(EvalResult).where(EvalResult.run_id == run_id)
        return list(self.s.scalars(query.order_by(EvalResult.id)))

    def runs_with_prompt(self, name: str, content_hash: str, golden_hash: str) -> list[EvalRun]:
        """Finished e2e runs on this golden set that used exactly this prompt
        content, newest first. Few rows, so the JSON is matched in Python."""
        query = select(EvalRun).where(
            EvalRun.kind == "e2e", EvalRun.status == "ok", EvalRun.golden_hash == golden_hash
        )
        return [
            run
            for run in self.s.scalars(query.order_by(EvalRun.id.desc()))
            if any(p.get("name") == name and p.get("hash") == content_hash for p in run.prompts)
        ]

    def cost_summary(self, since: date | None = None, until: date | None = None) -> dict:
        """Eval runs started in a window of UTC days, by kind: runs, turns, $ per stage."""
        query = select(EvalRun.kind, EvalRun.n_turns, EvalRun.cost_usd)
        kinds: dict[str, dict] = {}
        for kind, turns, cost in self.s.execute(
            _in_window(query, EvalRun.created_at, since, until).order_by(EvalRun.id)
        ):
            k = kinds.setdefault(kind, {"runs": 0, "turns": 0, "stages": Counter()})
            k["runs"] += 1
            k["turns"] += turns or 0
            k["stages"].update({st: v for st, v in (cost or {}).items() if st != "total"})
        by_kind = {
            kind: {
                "runs": k["runs"],
                "turns": k["turns"],
                "cost_usd": {st: round(v, 6) for st, v in sorted(k["stages"].items())}
                | {"total": round(sum(k["stages"].values()), 6)},
            }
            for kind, k in sorted(kinds.items())
        }
        total = sum(k["cost_usd"]["total"] for k in by_kind.values())
        return {"by_kind": by_kind, "cost_usd": round(total, 6)}
