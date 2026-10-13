import pytest
from alembic import command
from alembic.config import Config

from src.config.settings import PROJECT_ROOT
from src.storage.db import make_engine


@pytest.fixture
def engine(tmp_path):
    """A temporary SQLite database, migrated to the Alembic head."""
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={url}"]})()
    command.upgrade(cfg, "head")
    return make_engine(url)
