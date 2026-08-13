# pyright: reportMissingImports=false
"""API test fixtures.

DB scaffolding (test_engine, test_sm, _worker_schema, _clean_tables) lives in
the root evolver_server/conftest.py and is inherited here automatically.
"""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from evolver_server.app import app
from evolver_server.evolver.manager import MemoryManager
from evolver_server.evolver.store import MemoryStore
from evolver_server.evolver.tests.conftest import create_test_units

USER_ID = "user-alice"
SCOPE = "alice"
OTHER_SCOPE = "bob"
CORPUS_SIZE = 6  # len(create_test_units())


@pytest_asyncio.fixture()
async def store(test_sm):
    return MemoryStore(test_sm)


@pytest_asyncio.fixture()
async def mgr(store):
    return MemoryManager(store=store, retrieval_mode="keyword", auto_consolidate=False)


@pytest_asyncio.fixture()
async def seeded_store(store):
    """Store pre-seeded with the standard 6-unit corpus."""
    await store.add_memories(create_test_units(user_id=USER_ID, scope_id=SCOPE))
    return store


@pytest_asyncio.fixture()
async def client(test_sm):
    """AsyncClient against the FastAPI app with the test sessionmaker."""
    app.state.store = MemoryStore(test_sm)
    app.state.mgr = MemoryManager(
        store=app.state.store,
        retrieval_mode="keyword",
        auto_consolidate=True,
    )
    await app.state.store.add_memories(create_test_units(user_id=USER_ID, scope_id=SCOPE))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture()
async def client_and_store(test_sm):
    """AsyncClient + direct MemoryStore access for asserting DB state after writes."""
    store = MemoryStore(test_sm)
    app.state.store = store
    app.state.mgr = MemoryManager(
        store=store,
        retrieval_mode="keyword",
        auto_consolidate=True,
    )
    await store.add_memories(create_test_units(user_id=USER_ID, scope_id=SCOPE))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, store
