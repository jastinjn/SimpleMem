# pyright: reportMissingImports=false
"""Shared fixtures for the Evolver API test suite.

Every test gets a fresh isolated SQLite DB pre-seeded with the standard 6-unit
corpus injected directly into scope ``"alice"``.

Corpus summary:
  unit-001  SEMANTIC          "The project uses PostgreSQL as the primary database"
  unit-002  EPISODIC          "Alice and Bob discussed the authentication strategy for the API"
  unit-003  PREFERENCE        "User prefers TypeScript over JavaScript for frontend development"
  unit-004  PROJECT_STATE     "The deployment pipeline uses Kubernetes for container orchestration"
  unit-005  PROCEDURAL_OBS    "Running tests requires the pytest framework with coverage enabled"
  unit-006  SEMANTIC          "Redis is used for caching session tokens and rate limiting"
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from evolver_server.evolver.tests.conftest import create_test_units  # noqa: E402

SCOPE = "alice"
OTHER_SCOPE = "bob"
CORPUS_SIZE = 6  # len(create_test_units())


def _make_app(tmp_path, monkeypatch):
    """Return a freshly reloaded app module pointed at an isolated DB."""
    monkeypatch.setenv("EVOLVER_DB_PATH", str(tmp_path / "test.db"))
    from evolver_server import config as config_mod
    config_mod.get_settings.cache_clear()
    return importlib.reload(importlib.import_module("evolver_server.app"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh DB pre-seeded with the standard corpus."""
    app_mod = _make_app(tmp_path, monkeypatch)
    with TestClient(app_mod.app) as c:
        app_mod.app.state.store.add_memories(create_test_units(scope_id=SCOPE))
        yield c


@pytest.fixture()
def client_and_store(tmp_path, monkeypatch):
    """TestClient + direct MemoryStore access for asserting DB state after writes."""
    app_mod = _make_app(tmp_path, monkeypatch)
    with TestClient(app_mod.app) as c:
        store = app_mod.app.state.store
        store.add_memories(create_test_units(scope_id=SCOPE))
        yield c, store
