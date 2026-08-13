from __future__ import annotations

import asyncio
from logging.config import fileConfig

from evolver_server.config import get_settings
from evolver_server.evolver.schema import (  # noqa: F401 — register all models
    Base,
    Memory,
    MemoryAnnotation,
    MemoryEvent,
    MemoryLink,
    MemoryWatch,
    ScopeAccess,
    StatsSnapshot,
)
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    settings = get_settings()
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    settings = get_settings()
    connectable = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    try:
        asyncio.get_running_loop()
        # Called from within a running event loop (e.g. pytest-asyncio fixture).
        # Run in a new thread so asyncio.run() gets its own loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: asyncio.run(run_migrations_online())).result()
    except RuntimeError:
        asyncio.run(run_migrations_online())
