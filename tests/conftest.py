import pytest
from alembic import command
from alembic.config import Config

from src.config.settings import PROJECT_ROOT
from src.observability.langfuse import get_langfuse
from src.storage.db import make_engine


@pytest.fixture(autouse=True)
def langfuse_off(monkeypatch):
    """Tests never trace to, or fetch prompts from, the Langfuse in `.env`."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_langfuse.cache_clear()
    yield
    get_langfuse.cache_clear()


@pytest.fixture
def engine(tmp_path):
    """A temporary SQLite database, migrated to the Alembic head."""
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={url}"]})()
    command.upgrade(cfg, "head")
    return make_engine(url)
