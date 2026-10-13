"""SQLite engine and sessions. All DB access goes through `src/storage`.

Every connection gets:
    journal_mode=WAL    readers (the API) don't block on the writer (ingest)
    foreign_keys=ON     SQLite ignores FOREIGN KEY constraints unless told otherwise
    synchronous=NORMAL  safe with WAL, and much faster than FULL
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import PathSettings


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
