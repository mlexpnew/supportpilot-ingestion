import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from supportpilot.feedback.store import (
    get_feedback_db_path,
    metadata,
    normalize_database_url,
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging if not running under pytest
if config.config_file_name is not None and "pytest" not in sys.modules:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Model metadata for autogenerate support
target_metadata = metadata


def get_url() -> str:
    """Read DB URL from config, DATABASE_URL, or fallback to configured SQLite path."""
    cfg_url = config.get_main_option("sqlalchemy.url")
    if cfg_url and not cfg_url.startswith("driver://"):
        return cfg_url

    env_url = os.getenv("DATABASE_URL")
    if env_url is not None and env_url.strip():
        return normalize_database_url(env_url)

    allow_fallback = os.getenv("ALLOW_SQLITE_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not allow_fallback:
        raise RuntimeError(
            "DATABASE_URL is not set and SQLite fallback is disabled. "
            "Set DATABASE_URL to a valid database connection string, "
            "or set ALLOW_SQLITE_FALLBACK=1 for local development."
        )

    sqlite_path = get_feedback_db_path()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.resolve()}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = config.attributes.get("connection", None)

    if connectable is not None:
        if connectable.dialect.name == "postgresql":
            connectable.execute(text("SET lock_timeout = '30s'"))
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            render_as_batch=(connectable.dialect.name == "sqlite"),
        )

        with context.begin_transaction():
            context.run_migrations()
    else:
        configuration = config.get_section(config.config_ini_section, {}) or {}
        configuration["sqlalchemy.url"] = get_url()

        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            if connection.dialect.name == "postgresql":
                connection.execute(text("SET lock_timeout = '30s'"))
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=(connection.dialect.name == "sqlite"),
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
