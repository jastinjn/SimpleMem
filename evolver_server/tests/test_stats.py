# pyright: reportMissingImports=false
"""E2E tests for the /memory/stats endpoint."""

from __future__ import annotations

from .conftest import CORPUS_SIZE, OTHER_SCOPE, SCOPE, USER_ID


class TestMemoryStats:
    def test_missing_user_id_is_422(self, client):
        assert client.post("/memory/stats", json={}).status_code == 422

    def test_empty_user_id_is_422(self, client):
        assert client.post("/memory/stats", json={"user_id": ""}).status_code == 422

    def test_entry_count_matches_corpus(self, client):
        body = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0

    def test_active_by_type(self, client):
        body = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert body["active_by_type"].get("semantic", 0) == 2
        assert body["active_by_type"].get("episodic", 0) == 1
        assert body["active_by_type"].get("preference", 0) == 1
        assert body["active_by_type"].get("project_state", 0) == 1
        assert body["active_by_type"].get("procedural_observation", 0) == 1

    def test_type_count(self, client):
        body = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert body["type_count"] == 5

    def test_empty_scope_returns_zeros(self, client):
        body = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE}).json()
        assert body["entry_count"] == 0
        assert body["total"] == 0
        assert body["superseded"] == 0

    def test_no_scope_returns_all_user_memories(self, client):
        body = client.post("/memory/stats", json={"user_id": USER_ID}).json()
        assert body["entry_count"] == CORPUS_SIZE

    def test_other_user_same_scope_not_counted(self, client):
        other_user = "user-bob"
        # Give other_user memories in the same scope name.
        client.post("/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        # USER_ID's scoped count must be unchanged.
        body = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert body["entry_count"] == CORPUS_SIZE

    def test_no_scope_excludes_other_user_memories(self, client):
        other_user = "user-bob"
        client.post("/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        # USER_ID's total across all scopes must not include other_user's memories.
        body = client.post("/memory/stats", json={"user_id": USER_ID}).json()
        assert body["entry_count"] == CORPUS_SIZE
