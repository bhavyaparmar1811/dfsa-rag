"""Alembic environment.

The database URL comes from the app's settings (PathSettings.sqlite_url), so
migrations and the app always point at the same file. An explicit
`sqlalchemy.url` (e.g. `alembic -x url=sqlite:///tmp.db`) overrides it, which
the tests use.

`render_as_batch=True`: SQLite can't ALTER most columns, so Alembic rebuilds
the table (copy, drop, rename) instead. Harmless on Postgres.
"""

from logging.config import fileConfig

from alembic import context

from src.config.settings import PathSettings
from src.storage.db import make_engine
from src.storage.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    return (
        context.get_x_argument(as_dictionary=True).get("url")
        or config.get_main_option("sqlalchemy.url")
        or PathSettings().sqlite_url
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (`alembic upgrade head --sql`)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(database_url())  # same pragmas as the app (foreign keys, WAL)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
