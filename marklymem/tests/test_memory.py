# pyright: reportMissingImports=false
"""E2E tests for the memory system, grouped by endpoint."""

from __future__ import annotations

from marklymem.tests.utils.db import (
    CORPUS_SIZE,
    OTHER_NAMESPACE,
    OTHER_NAMESPACE_SIZE,
    SCOPE,
    USER_ID,
)


class TestMemoryAddDialogue:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/add_dialogue", json={"user_id": USER_ID, "namespace": SCOPE, "turns": [{"prompt_text": "hi", "response_text": "hello"}]})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add_dialogue", json={"turns": [{"prompt_text": "hi"}]})).status_code == 422

    async def test_empty_turns_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add_dialogue", json={"user_id": USER_ID, "namespace": SCOPE, "turns": []})).status_code == 422

    async def test_over_50_turns_is_422(self, authed_client):
        turns = [{"prompt_text": f"Turn {i}", "response_text": "Ok."} for i in range(51)]
        assert (await authed_client.post("/api/memory/add_dialogue", json={"user_id": USER_ID, "namespace": SCOPE, "turns": turns})).status_code == 422

    async def test_returns_units_added(self, authed_client):
        r = await authed_client.post("/api/memory/add_dialogue", json={
            "user_id": USER_ID,
            "namespace": SCOPE,
            "turns": [
                {"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."},
                {"prompt_text": "Secrets are stored in HashiCorp Vault", "response_text": "Understood."},
            ],
        })
        assert r.status_code == 200
        assert r.json()["units_added"] > 0

    async def test_new_content_is_persisted(self, authed_client, app_state):
        await authed_client.post("/api/memory/add_dialogue", json={
            "user_id": USER_ID,
            "namespace": SCOPE,
            "turns": [{"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."}],
        })
        active = await app_state.list_active(USER_ID, SCOPE)
        assert any("helm" in u.content.lower() for u in active)

    async def test_does_not_write_to_other_scope(self, authed_client, app_state):
        await authed_client.post("/api/memory/add_dialogue", json={
            "user_id": USER_ID,
            "namespace": OTHER_NAMESPACE,
            "turns": [{"prompt_text": "The project uses PostgreSQL as the primary database", "response_text": "Understood."}],
        })
        assert len(await app_state.list_active(USER_ID, SCOPE)) == CORPUS_SIZE


class TestMemoryRetrieve:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "hi"})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/retrieve", json={"query": "x"})).status_code == 422

    async def test_missing_query_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE})).status_code == 422

    async def test_returns_units_for_matching_query(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "database", "top_k": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] > 0
        hit = body["results"][0]
        for field in ("memory_id", "content", "memory_type", "importance",
                      "score", "matched_terms", "entities", "topics", "updated_at"):
            assert field in hit, f"missing field: {field}"
        assert isinstance(hit["score"], float)
        assert isinstance(hit["matched_terms"], list)

    async def test_user_cannot_access_other_user_memories_in_same_scope(self, authed_client):
        # Representative tenant-isolation contract at the route boundary; the exhaustive
        # scope/cross-user matrix lives in test_store.py and test_retriever.py.
        # OTHER_USER + SCOPE is pre-seeded with Ansible; USER_ID must not see it.
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0


class TestMemoryClear:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/clear", json={"user_id": USER_ID, "namespace": SCOPE})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/clear", json={})).status_code == 422

    async def test_clears_existing_memories(self, authed_client, app_state):
        r = await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "namespace": SCOPE})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == USER_ID
        assert body["namespace"] == SCOPE
        assert body["archived"] == CORPUS_SIZE
        assert body["pinned_kept"] == 0
        assert body["total_before"] == CORPUS_SIZE
        assert len(await app_state.list_active(USER_ID, SCOPE)) == 0


class TestMemoryStats:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={})).status_code == 422

    async def test_empty_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={"user_id": ""})).status_code == 422

    async def test_returns_full_stats_shape(self, authed_client):
        # Route → StatsResponse mapping, including fields derived in summarize_memory_store
        # (type_count, dominant_type). The aggregation itself is covered by
        # test_store.py::TestGetStats; here we assert the wiring and serialized shape.
        r = await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})
        assert r.status_code == 200
        body = r.json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0
        assert body["type_count"] == 5
        assert isinstance(body["active_by_type"], dict)
        assert isinstance(body["dominant_type"], str)


class TestCloneScope:
    CLONE_TARGET = "clone-target"

    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": OTHER_NAMESPACE, "target_namespace": self.CLONE_TARGET,
        })
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        r = await authed_client.post("/api/memory/clone_namespace", json={
            "source_namespace": OTHER_NAMESPACE, "target_namespace": self.CLONE_TARGET,
        })
        assert r.status_code == 422

    async def test_missing_source_namespace_is_422(self, authed_client):
        r = await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "target_namespace": self.CLONE_TARGET,
        })
        assert r.status_code == 422

    async def test_missing_target_namespace_is_422(self, authed_client):
        r = await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": OTHER_NAMESPACE,
        })
        assert r.status_code == 422

    async def test_clones_memories_to_target_namespace(self, authed_client, app_state):
        r = await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": OTHER_NAMESPACE, "target_namespace": self.CLONE_TARGET,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["cloned"] == OTHER_NAMESPACE_SIZE
        assert body["user_id"] == USER_ID
        assert body["source_namespace"] == OTHER_NAMESPACE
        assert body["target_namespace"] == self.CLONE_TARGET
        cloned = await app_state.list_active(USER_ID, self.CLONE_TARGET)
        assert len(cloned) == OTHER_NAMESPACE_SIZE
