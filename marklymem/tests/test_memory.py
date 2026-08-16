# pyright: reportMissingImports=false
"""E2E tests for the memory system, grouped by endpoint."""

from __future__ import annotations

from marklymem.tests.utils.db import (
    CORPUS_SIZE,
    OTHER_NAMESPACE,
    OTHER_NAMESPACE_SIZE,
    OTHER_USER,
    OTHER_USER_SIZE,
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

    async def test_top_k_limits_result_count(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "database", "top_k": 1})
        assert len(r.json()["results"]) <= 1

    async def test_returns_empty_for_unrelated_query(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "xyzzy_zap_unrelated_42", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_returns_empty_for_unknown_scope(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": "no-such-scope", "query": "database", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_scope_filters_out_other_scopes(self, authed_client):
        # OTHER_NAMESPACE is pre-seeded with Terraform content; it must not bleed into SCOPE.
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_no_scope_returns_all_user_scopes(self, authed_client):
        # OTHER_NAMESPACE is pre-seeded with Terraform; querying with no namespace must find it.
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] > 0

    async def test_user_cannot_access_other_user_memories_in_same_scope(self, authed_client):
        # OTHER_USER + SCOPE is pre-seeded with Ansible; USER_ID must not see it.
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "namespace": SCOPE, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_user_cannot_access_other_user_memories_across_all_scopes(self, authed_client):
        # OTHER_USER + SCOPE is pre-seeded with Ansible; USER_ID must not see it with no-namespace query.
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "query": "Ansible", "top_k": 10})
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

    async def test_empty_scope_returns_zero_archived(self, authed_client):
        r = await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "namespace": "no-such-scope"})
        assert r.status_code == 200
        assert r.json()["archived"] == 0

    async def test_does_not_affect_other_scope(self, authed_client, app_state):
        await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "namespace": SCOPE})
        assert len(await app_state.list_active(USER_ID, OTHER_NAMESPACE)) == OTHER_NAMESPACE_SIZE

    async def test_does_not_affect_other_user_same_scope(self, authed_client, app_state):
        await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "namespace": SCOPE})
        assert len(await app_state.list_active(OTHER_USER, SCOPE)) == OTHER_USER_SIZE


class TestMemoryStats:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={})).status_code == 422

    async def test_empty_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={"user_id": ""})).status_code == 422

    async def test_entry_count_matches_corpus(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0

    async def test_active_by_type(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})).json()
        assert body["active_by_type"].get("semantic", 0) == 2
        assert body["active_by_type"].get("episodic", 0) == 1
        assert body["active_by_type"].get("preference", 0) == 1
        assert body["active_by_type"].get("project_state", 0) == 1
        assert body["active_by_type"].get("procedural_observation", 0) == 1

    async def test_type_count(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})).json()
        assert body["type_count"] == 5

    async def test_empty_scope_returns_zeros(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": "no-such-scope"})).json()
        assert body["entry_count"] == 0
        assert body["total"] == 0
        assert body["superseded"] == 0

    async def test_no_scope_returns_all_user_memories(self, authed_client):
        # USER_ID spans SCOPE (6) and OTHER_NAMESPACE (2); no-namespace must sum both.
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE + OTHER_NAMESPACE_SIZE

    async def test_other_user_same_scope_not_counted(self, authed_client):
        # OTHER_USER + SCOPE is pre-seeded; it must not appear in USER_ID + SCOPE stats.
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "namespace": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE

    async def test_no_scope_excludes_other_user_memories(self, authed_client):
        # OTHER_USER's units must not appear in USER_ID's no-namespace total.
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE + OTHER_NAMESPACE_SIZE


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

    async def test_source_namespace_unchanged_after_clone(self, authed_client, app_state):
        await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": OTHER_NAMESPACE, "target_namespace": self.CLONE_TARGET,
        })
        assert len(await app_state.list_active(USER_ID, OTHER_NAMESPACE)) == OTHER_NAMESPACE_SIZE

    async def test_clone_creates_new_ids(self, authed_client, app_state):
        await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": OTHER_NAMESPACE, "target_namespace": self.CLONE_TARGET,
        })
        source_ids = {u.memory_id for u in await app_state.list_active(USER_ID, OTHER_NAMESPACE)}
        cloned_ids = {u.memory_id for u in await app_state.list_active(USER_ID, self.CLONE_TARGET)}
        assert source_ids.isdisjoint(cloned_ids)

    async def test_empty_source_namespace_returns_zero_cloned(self, authed_client):
        r = await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": "no-such-scope", "target_namespace": self.CLONE_TARGET,
        })
        assert r.status_code == 200
        assert r.json()["cloned"] == 0

    async def test_does_not_clone_other_user_memories(self, authed_client, app_state):
        # Cloning USER_ID's namespace must not touch OTHER_USER's units.
        await authed_client.post("/api/memory/clone_namespace", json={
            "user_id": USER_ID, "source_namespace": SCOPE, "target_namespace": self.CLONE_TARGET,
        })
        assert len(await app_state.list_active(OTHER_USER, SCOPE)) == OTHER_USER_SIZE
