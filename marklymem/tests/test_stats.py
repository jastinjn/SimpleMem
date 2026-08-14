# pyright: reportMissingImports=false
"""E2E tests for the /memory/stats endpoint."""

from __future__ import annotations

from .conftest import CORPUS_SIZE, OTHER_SCOPE, SCOPE, USER_ID


class TestMemoryStats:
    async def test_missing_user_id_is_422(self, client):
        assert (await client.post("/api/memory/stats", json={})).status_code == 422

    async def test_empty_user_id_is_422(self, client):
        assert (await client.post("/api/memory/stats", json={"user_id": ""})).status_code == 422

    async def test_entry_count_matches_corpus(self, client):
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0

    async def test_active_by_type(self, client):
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["active_by_type"].get("semantic", 0) == 2
        assert body["active_by_type"].get("episodic", 0) == 1
        assert body["active_by_type"].get("preference", 0) == 1
        assert body["active_by_type"].get("project_state", 0) == 1
        assert body["active_by_type"].get("procedural_observation", 0) == 1

    async def test_type_count(self, client):
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["type_count"] == 5

    async def test_empty_scope_returns_zeros(self, client):
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})).json()
        assert body["entry_count"] == 0
        assert body["total"] == 0
        assert body["superseded"] == 0

    async def test_no_scope_returns_all_user_memories(self, client):
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE

    async def test_other_user_same_scope_not_counted(self, client):
        other_user = "user-bob"
        await client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE

    async def test_no_scope_excludes_other_user_memories(self, client):
        other_user = "user-bob"
        await client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        body = (await client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE
