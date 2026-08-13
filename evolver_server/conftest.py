# pyright: reportMissingImports=false
"""Root conftest — shared async-Postgres fixtures for all test suites.

Every test gets a fresh schema on every session (schema = test_<worker>),
and tables are TRUNCATED between individual tests via an autouse fixture.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from evolver_server.evolver.db import build_sessionmaker

_TEST_DB_URL = os.environ.get(
    "EVOLVER_DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5442/simplemem",
)

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
_SCHEMA = f"test_{_WORKER}"

_CONNECT_ARGS = {"server_settings": {"search_path": f"{_SCHEMA},public"}}


@pytest.fixture(scope="session")
def event_loop_policy():
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _worker_schema():
    """Create the worker schema + run Alembic migrations once per session."""
    from alembic.config import Config as AlembicConfig

    from alembic import command

    engine = create_async_engine(_TEST_DB_URL, connect_args=_CONNECT_ARGS, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE'))
        await conn.execute(text(f'CREATE SCHEMA "{_SCHEMA}"'))
    await engine.dispose()

    alembic_cfg = AlembicConfig()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", _TEST_DB_URL.replace("+asyncpg", ""))
    os.environ["EVOLVER_DATABASE_URL"] = _TEST_DB_URL
    command.upgrade(alembic_cfg, "head")

    yield

    engine2 = create_async_engine(_TEST_DB_URL, connect_args=_CONNECT_ARGS, pool_pre_ping=True)
    async with engine2.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE'))
    await engine2.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(_TEST_DB_URL, connect_args=_CONNECT_ARGS, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_sm(test_engine):
    return build_sessionmaker(test_engine)


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(test_engine):
    """Truncate all tables between tests for isolation."""
    async with test_engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE memories, memory_events, memory_links, memory_watches, "
            "memory_annotations, scope_access, stats_snapshots RESTART IDENTITY CASCADE"
        ))
