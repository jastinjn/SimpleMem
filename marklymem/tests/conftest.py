# pyright: reportMissingImports=false
"""API test fixtures.

DB scaffolding (test_engine, test_sm, _worker_schema, _clean_tables) lives in
the root marklymem/conftest.py and is inherited here automatically.
"""

from __future__ import annotations

import secrets

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from marklymem.app import app
from marklymem.config import Settings
from marklymem.evolver.manager import MemoryManager
from marklymem.evolver.store import MemoryStore
from marklymem.evolver.tests.conftest import create_test_units

TEST_API_KEY = secrets.token_hex(32)

USER_ID = "user-alice"
SCOPE = "alice"
OTHER_SCOPE = "bob"
CORPUS_SIZE = 6  # len(create_test_units())


def _non_local_settings() -> Settings:
    return Settings(APP_ENV="test", INTERNAL_API_KEY=TEST_API_KEY)


# --- HTTP fixtures ---

@pytest_asyncio.fixture(autouse=True)
async def app_state(test_sm, monkeypatch):
    """Wires app store/manager, patches auth settings to non-local, seeds corpus.

    Yields the store directly so tests that need DB assertions can use it.
    """
    _settings = _non_local_settings()
    monkeypatch.setattr("marklymem.utils.auth.get_settings", lambda: _settings)
    store = MemoryStore(test_sm)
    app.state.store = store
    app.state.mgr = MemoryManager(store=store, retrieval_mode="keyword", auto_consolidate=True)
    await store.add_memories(create_test_units(user_id=USER_ID, scope_id=SCOPE))
    yield store


@pytest_asyncio.fixture()
async def authed_client():
    """Authenticated HTTP client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"API-Key": TEST_API_KEY},
    ) as ac:
        yield ac


@pytest_asyncio.fixture()
async def unauthed_client():
    """Unauthenticated HTTP client — for 403 tests only."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
