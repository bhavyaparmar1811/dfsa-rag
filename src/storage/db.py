"""SQLite engine and sessions. All DB access goes through `src/storage`.

Every connection gets:
    journal_mode=WAL    readers (the API) don't block on the writer (ingest)
    foreign_keys=ON     SQLite ignores FOREIGN KEY constraints unless told otherwise
    synchronous=NORMAL  safe with WAL, and much faster than FULL
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import PROJECT_ROOT, PathSettings


class SchemaOutOfDate(Exception):
    """The database isn't at the latest Alembic migration."""


def make_engine(url: str | None = None) -> Engine:
    url = url or PathSettings().sqlite_url
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def _sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


@lru_cache
def get_engine() -> Engine:
    return make_engine()


def assert_migrated(engine: Engine) -> None:
    """Raise unless the database is at the Alembic head, so a stale or empty
    database fails before any work is done instead of halfway through."""
    head = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini"))).get_current_head()
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current != head:
        raise SchemaOutOfDate(
            f"Database {engine.url} is at migration {current or 'none'}, the code expects {head}. "
            "Run `uv run alembic upgrade head`."
        )


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """A transaction: committed if the block succeeds, rolled back if it raises."""
    session = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
