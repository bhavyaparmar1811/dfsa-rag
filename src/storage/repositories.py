"""Reads and writes for the ingestion tables. The rest of the app never touches
the ORM models directly, so the storage backend can change behind this file.

Each repository wraps a `Session` and never commits: the caller decides the
transaction (`session_scope()`), e.g. one transaction per ingested document.
"""

from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import datetime

from sqlalchemy import Integer, delete, func, insert, select, update
from sqlalchemy.orm import Session

from src.ingestion.diff import StoredChunk
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk
from src.storage.models import ChunkRef, ChunkRow, ChunkTerm, Document, IngestionRun, utcnow


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
        self, ids: Iterable[str], model: str, dimensions: int, at: datetime | None = None
    ) -> None:
        ids = list(ids)
        if ids:
            self.s.execute(
                update(ChunkRow)
                .where(ChunkRow.id.in_(ids))
                .values(is_embedded=True, embedding_model=model, embedding_dimensions=dimensions,
                        embedded_at=at or utcnow()),
                execution_options={"synchronize_session": False},
            )  # fmt: skip

    # --- lookups -------------------------------------------------------------

    def by_key(self, key: str) -> ChunkRow | None:
        return self.s.scalar(select(ChunkRow).where(ChunkRow.key == key))

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
