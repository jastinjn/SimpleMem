# pyright: reportMissingImports=false
"""E2E tests for the /memory/stats endpoint."""

from __future__ import annotations

from .conftest import CORPUS_SIZE, OTHER_SCOPE, SCOPE


class TestMemoryStats:
    def test_missing_scope_id_is_422(self, client):
        assert client.post("/memory/stats", json={}).status_code == 422

    def test_empty_scope_id_is_422(self, client):
        assert client.post("/memory/stats", json={"scope_id": ""}).status_code == 422

    def test_entry_count_matches_corpus(self, client):
        body = client.post("/memory/stats", json={"scope_id": SCOPE}).json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0

    def test_active_by_type(self, client):
        body = client.post("/memory/stats", json={"scope_id": SCOPE}).json()
        assert body["active_by_type"].get("semantic", 0) == 2
        assert body["active_by_type"].get("episodic", 0) == 1
        assert body["active_by_type"].get("preference", 0) == 1
        assert body["active_by_type"].get("project_state", 0) == 1
        assert body["active_by_type"].get("procedural_observation", 0) == 1

    def test_type_count(self, client):
        body = client.post("/memory/stats", json={"scope_id": SCOPE}).json()
        assert body["type_count"] == 5

    def test_empty_scope_returns_zeros(self, client):
        body = client.post("/memory/stats", json={"scope_id": OTHER_SCOPE}).json()
        assert body["entry_count"] == 0
        assert body["total"] == 0
        assert body["superseded"] == 0
