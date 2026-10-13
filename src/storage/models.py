"""SQLite tables for the ingested corpus (Alembic migration 0001).

    documents 1 ──< chunks >── chunk_refs     what each chunk references
                       └────── chunk_terms    which Defined Terms it uses
    ingestion_runs                            one row per ingest: counts, tokens, $

SQLite is the system of record: it holds every chunk, including the ones that
are never embedded (section parents, rules that were split into parts, deleted
provisions). Qdrant holds only the embedded chunks; `chunks.id` is the Qdrant
point id, so a search result maps straight back to its row here.

Only portable SQLAlchemy types are used, so moving to Postgres is a config change.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One row per manifest entry."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "dfsa-rulebook-cob"
    regulator: Mapped[str] = mapped_column(String)
    doc_type: Mapped[str] = mapped_column(String)  # "rulebook" | "law"
    module_code: Mapped[str] = mapped_column(String)
    module_name: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    effective_date: Mapped[str] = mapped_column(String)
    source_url: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    file_hash: Mapped[str] = mapped_column(String(64))  # sha256 of the PDF
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedded_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkRow(Base):
    """Every chunk, embedded or not. Mirrors `src.ingestion.models.Chunk`."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid5 = Qdrant point id
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String, unique=True)  # "COB 3.3.2", "COB 3.3.2#p1"
    citation: Mapped[str] = mapped_column(String, index=True)
    chunk_type: Mapped[str] = mapped_column(String, index=True)
    binding: Mapped[bool] = mapped_column(Boolean)

    rule_number: Mapped[str | None] = mapped_column(String)
    section: Mapped[str] = mapped_column(String)
    term: Mapped[str | None] = mapped_column(String, index=True)  # set on definitions
    parent_key: Mapped[str | None] = mapped_column(String, index=True)
    part_of: Mapped[str | None] = mapped_column(String, index=True)
    guidance_for: Mapped[str | None] = mapped_column(String)
    part: Mapped[int | None] = mapped_column(Integer)
    order_idx: Mapped[int] = mapped_column(Integer)

    heading_path: Mapped[list[str]] = mapped_column(JSON)
    guidance_keys: Mapped[list[str]] = mapped_column(JSON, default=list)
    parts: Mapped[list[str]] = mapped_column(JSON, default=list)
    subheading: Mapped[str | None] = mapped_column(String)
    scope_rule: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    header_text: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)

    # sha256 of the embedding fingerprint + embedding_text(): a chunk is only
    # re-embedded when this changes.
    content_hash: Mapped[str] = mapped_column(String(64))
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False)
    # CLAUDE.md: the embedding model and dimension are stored with every chunk.
    embedding_model: Mapped[str | None] = mapped_column(String)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="chunks")
    refs: Mapped[list["ChunkRef"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    terms: Mapped[list["ChunkTerm"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkRef(Base):
    """A cross-reference: chunk -> the key it cites ("GEN 2.26.1").

    The index on target_key answers the reverse question cheaply: which
    chunks reference "Regulatory Law Art. 76"? The target may be outside the
    corpus (e.g. "Insolvency Law Art. 98"), so it isn't a foreign key.
    """

    __tablename__ = "chunk_refs"

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    target_key: Mapped[str] = mapped_column(String, primary_key=True, index=True)


class ChunkTerm(Base):
    """A Defined Term used in a chunk ("Retail Client")."""

    __tablename__ = "chunk_terms"

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    term: Mapped[str] = mapped_column(String, primary_key=True, index=True)


class IngestionRun(Base):
    """One row per `ingest` run: what changed and what it cost."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="running")  # running | ok | failed
    error: Mapped[str | None] = mapped_column(Text)
    embedding_fingerprint: Mapped[str] = mapped_column(String)
    sparse_model: Mapped[str] = mapped_column(String)
    docs_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_new: Mapped[int] = mapped_column(Integer, default=0)
    chunks_changed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    chunks_removed: Mapped[int] = mapped_column(Integer, default=0)
    embed_tokens: Mapped[int] = mapped_column(Integer, default=0)
    embed_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
