"""The golden set against the live corpus in data/app.db (read-only).

Every gold and supporting provision must be a live, retrievable chunk, so a
re-ingest that renames or drops a provision breaks here, not in an eval run.
Skipped when SQLite hasn't been loaded.
"""

import pytest
from sqlalchemy import func, select

from src.eval.golden import check_against_corpus, load_golden
from src.storage.db import get_engine, session_scope
from src.storage.models import ChunkRow

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def engine():
    engine = get_engine()
    try:
        with session_scope(engine) as s:
            rows = s.scalar(select(func.count()).select_from(ChunkRow))
    except Exception as e:
        pytest.skip(f"data/app.db not available: {e}")
    if not rows:
        pytest.skip("data/app.db has no chunks: run the ingest first")
    return engine


def test_every_golden_provision_is_in_the_corpus(engine):
    assert check_against_corpus(load_golden(), engine) == []
