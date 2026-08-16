# pyright: reportMissingImports=false
"""E2E tests for the memory system, grouped by endpoint."""

from __future__ import annotations

from .conftest import CORPUS_SIZE, OTHER_SCOPE, SCOPE, USER_ID


class TestMemoryAdd:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/add", json={"user_id": USER_ID, "scope_id": SCOPE, "prompt_text": "hi", "response_text": "hello"})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add", json={"prompt_text": "hi"})).status_code == 422

    async def test_empty_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add", json={"user_id": "", "prompt_text": "hi"})).status_code == 422

    async def test_both_fields_empty_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add", json={"user_id": USER_ID})).status_code == 422

    async def test_returns_units_added(self, authed_client, app_state):
        r = await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We now use Terraform for infrastructure as code",
            "response_text": "Understood.",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == USER_ID
        assert body["scope_id"] == SCOPE
        assert body["units_added"] > 0
        active = await app_state.list_active(USER_ID, SCOPE)
        contents = " ".join(u.content for u in active).lower()
        assert "terraform" in contents

    async def test_new_content_is_retrievable(self, authed_client, app_state):
        await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We now use Terraform for infrastructure provisioning",
            "response_text": "Understood.",
        })
        r = await authed_client.post("/api/memory/retrieve", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "query": "Terraform infrastructure",
            "top_k": 10,
        })
        assert r.status_code == 200
        assert r.json()["total"] > 0
        active = await app_state.list_active(USER_ID, SCOPE)
        assert any("terraform" in u.content.lower() for u in active)

    async def test_consolidates_after_add(self, authed_client):
        # First add — no duplicates yet, nothing to consolidate.
        r1 = await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We deploy with Helm charts for all services",
            "response_text": "Got it.",
        })
        assert r1.status_code == 200
        assert r1.json()["units_consolidated"] == 0

    async def test_does_not_write_to_other_scope(self, authed_client):
        await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "The project uses PostgreSQL as the primary database",
            "response_text": "Understood.",
        })
        assert (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()["entry_count"] == CORPUS_SIZE
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10})
        for hit in r.json()["results"]:
            assert hit["memory_id"].startswith("unit-")

    async def test_does_not_write_to_other_user(self, authed_client):
        other_user = "user-bob"
        await authed_client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        assert (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()["entry_count"] == CORPUS_SIZE
        assert (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID})).json()["entry_count"] == CORPUS_SIZE


class TestMemoryAddBatch:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/add_batch", json={"user_id": USER_ID, "scope_id": SCOPE, "turns": [{"prompt_text": "hi", "response_text": "hello"}]})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add_batch", json={"turns": [{"prompt_text": "hi"}]})).status_code == 422

    async def test_empty_turns_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/add_batch", json={"user_id": USER_ID, "scope_id": SCOPE, "turns": []})).status_code == 422

    async def test_over_50_turns_is_422(self, authed_client):
        turns = [{"prompt_text": f"Turn {i}", "response_text": "Ok."} for i in range(51)]
        assert (await authed_client.post("/api/memory/add_batch", json={"user_id": USER_ID, "scope_id": SCOPE, "turns": turns})).status_code == 422

    async def test_returns_units_added(self, authed_client):
        r = await authed_client.post("/api/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [
                {"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."},
                {"prompt_text": "Secrets are stored in HashiCorp Vault", "response_text": "Understood."},
            ],
        })
        assert r.status_code == 200
        assert r.json()["units_added"] > 0

    async def test_new_content_is_retrievable(self, authed_client):
        await authed_client.post("/api/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [
                {"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."},
            ],
        })
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Helm kubernetes", "top_k": 10})
        assert r.status_code == 200
        assert r.json()["total"] > 0

    async def test_consolidates_after_add_batch(self, authed_client):
        # Sending the same turn twice in one batch bypasses pre-store dedup,
        # so both units are written and consolidation supersedes the older one.
        r = await authed_client.post("/api/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [
                {"prompt_text": "We deploy with Helm charts for all services", "response_text": "Got it."},
                {"prompt_text": "We deploy with Helm charts for all services", "response_text": "Got it."},
            ],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["units_added"] > 0
        assert body["units_consolidated"] >= 1

    async def test_does_not_write_to_other_scope(self, authed_client):
        await authed_client.post("/api/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "turns": [{"prompt_text": "The project uses PostgreSQL as the primary database", "response_text": "Understood."}],
        })
        assert (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()["entry_count"] == CORPUS_SIZE


class TestMemoryRetrieve:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "hi"})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/retrieve", json={"query": "x"})).status_code == 422

    async def test_missing_query_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE})).status_code == 422

    async def test_returns_units_for_matching_query(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10})
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
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 1})
        assert len(r.json()["results"]) <= 1

    async def test_returns_empty_for_unrelated_query(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "xyzzy_zap_unrelated_42", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_returns_empty_for_unknown_scope(self, authed_client):
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE, "query": "database", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_scope_filters_out_other_scopes(self, authed_client):
        await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "We use Terraform for provisioning",
            "response_text": "Noted.",
        })
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] == 0

    async def test_no_scope_returns_all_user_scopes(self, authed_client):
        await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "We use Terraform for provisioning",
            "response_text": "Noted.",
        })
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] > 0

    async def test_user_cannot_access_other_user_memories_in_same_scope(self, authed_client):
        other_user = "user-bob"
        await authed_client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0
        r = await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0


class TestMemoryClear:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/clear", json={})).status_code == 422

    async def test_clears_existing_memories(self, authed_client):
        r = await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == USER_ID
        assert body["scope_id"] == SCOPE
        assert body["archived"] == CORPUS_SIZE
        assert body["pinned_kept"] == 0
        assert body["total_before"] == CORPUS_SIZE
        assert (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()["entry_count"] == 0
        assert (await authed_client.post("/api/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10})).json()["total"] == 0

    async def test_empty_scope_returns_zero_archived(self, authed_client):
        r = await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})
        assert r.status_code == 200
        assert r.json()["archived"] == 0

    async def test_does_not_affect_other_scope(self, authed_client):
        await authed_client.post("/api/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "The project uses PostgreSQL as the primary database",
            "response_text": "Understood.",
        })
        bob_before = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})).json()["entry_count"]
        await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        bob_after = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})).json()["entry_count"]
        assert bob_after == bob_before

    async def test_does_not_affect_other_user_same_scope(self, authed_client):
        other_user = "user-bob"
        await authed_client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        other_before = (await authed_client.post("/api/memory/stats", json={"user_id": other_user, "scope_id": SCOPE})).json()["entry_count"]
        await authed_client.post("/api/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        other_after = (await authed_client.post("/api/memory/stats", json={"user_id": other_user, "scope_id": SCOPE})).json()["entry_count"]
        assert other_after == other_before


class TestMemoryStats:
    async def test_missing_key_returns_403(self, unauthed_client):
        r = await unauthed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})
        assert r.status_code == 403

    async def test_missing_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={})).status_code == 422

    async def test_empty_user_id_is_422(self, authed_client):
        assert (await authed_client.post("/api/memory/stats", json={"user_id": ""})).status_code == 422

    async def test_entry_count_matches_corpus(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE
        assert body["total"] == CORPUS_SIZE
        assert body["superseded"] == 0

    async def test_active_by_type(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["active_by_type"].get("semantic", 0) == 2
        assert body["active_by_type"].get("episodic", 0) == 1
        assert body["active_by_type"].get("preference", 0) == 1
        assert body["active_by_type"].get("project_state", 0) == 1
        assert body["active_by_type"].get("procedural_observation", 0) == 1

    async def test_type_count(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["type_count"] == 5

    async def test_empty_scope_returns_zeros(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})).json()
        assert body["entry_count"] == 0
        assert body["total"] == 0
        assert body["superseded"] == 0

    async def test_no_scope_returns_all_user_memories(self, authed_client):
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE

    async def test_other_user_same_scope_not_counted(self, authed_client):
        other_user = "user-bob"
        await authed_client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE})).json()
        assert body["entry_count"] == CORPUS_SIZE

    async def test_no_scope_excludes_other_user_memories(self, authed_client):
        other_user = "user-bob"
        await authed_client.post("/api/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        body = (await authed_client.post("/api/memory/stats", json={"user_id": USER_ID})).json()
        assert body["entry_count"] == CORPUS_SIZE
