"""The Alembic migrations against a temporary SQLite file."""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, select, text

from src.config.settings import PROJECT_ROOT
from src.storage.db import make_engine, session_scope
from src.storage.models import Base, ChunkRef, ChunkRow, ChunkTerm, Document


@pytest.fixture
def engine(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={url}"]})()
    command.upgrade(cfg, "head")
    return make_engine(url)


def test_migration_creates_the_tables(engine):
    tables = set(inspect(engine).get_table_names())
    assert {"documents", "chunks", "chunk_refs", "chunk_terms", "ingestion_runs"} <= tables


def test_models_and_migrations_agree(engine):
    """Fails if a model changes without a new migration."""
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_pragmas(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def _document():
    return Document(
        id="dfsa-rulebook-cob", regulator="DFSA", doc_type="rulebook", module_code="COB",
        module_name="Conduct of Business Module", version="51", effective_date="2026-07",
        source_url="https://example.com", file_path="data/raw/cob.pdf", file_hash="0" * 64,
    )  # fmt: skip


def _chunk(key="COB 3.3.2", id=None):
    return ChunkRow(
        id=id or f"id-{key}", document_id="dfsa-rulebook-cob", key=key, citation=key,
        chunk_type="rule", binding=True, section="3.3", order_idx=0, heading_path=["COB"],
        header_text="Rule COB 3.3.2", text="3.3.2 (1) Subject to (2) ...", page_start=32,
        page_end=32, token_count=80, content_hash="h" * 64,
    )  # fmt: skip


def test_deleting_a_document_cascades_to_chunks_refs_and_terms(engine):
    with session_scope(engine) as s:
        doc = _document()
        chunk = _chunk()
        chunk.refs = [ChunkRef(target_key="COB 3.3.4")]
        chunk.terms = [ChunkTerm(term="Client Agreement")]
        doc.chunks = [chunk]
        s.add(doc)

    with session_scope(engine) as s:
        assert s.scalar(select(ChunkRef.chunk_id).where(ChunkRef.target_key == "COB 3.3.4"))
        s.delete(s.get(Document, "dfsa-rulebook-cob"))

    with session_scope(engine) as s:
        assert s.scalars(select(ChunkRow)).all() == []
        assert s.scalars(select(ChunkRef)).all() == []
        assert s.scalars(select(ChunkTerm)).all() == []


def test_chunk_keys_are_unique(engine):
    with pytest.raises(Exception, match="UNIQUE"), session_scope(engine) as s:
        s.add(_document())
        s.add_all([_chunk(), _chunk(id="another-id")])  # same key, different id
