# pyright: reportMissingImports=false
"""E2E tests for the memory system, grouped by endpoint."""

from __future__ import annotations

from .conftest import CORPUS_SIZE, OTHER_SCOPE, SCOPE, USER_ID


class TestMemoryAdd:
    def test_missing_user_id_is_422(self, client):
        assert client.post("/memory/add", json={"prompt_text": "hi"}).status_code == 422

    def test_empty_user_id_is_422(self, client):
        assert client.post("/memory/add", json={"user_id": "", "prompt_text": "hi"}).status_code == 422

    def test_both_fields_empty_is_422(self, client):
        assert client.post("/memory/add", json={"user_id": USER_ID}).status_code == 422

    def test_returns_units_added(self, client_and_store):
        client, store = client_and_store
        r = client.post("/memory/add", json={
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
        # Verify the units are actually present in the DB.
        active = store.list_active(USER_ID, SCOPE)
        contents = " ".join(u.content for u in active).lower()
        assert "terraform" in contents

    def test_new_content_is_retrievable(self, client_and_store):
        client, store = client_and_store
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We now use Terraform for infrastructure provisioning",
            "response_text": "Understood.",
        })
        # Check via the API.
        r = client.post("/memory/retrieve", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "query": "Terraform infrastructure",
            "top_k": 10,
        })
        assert r.status_code == 200
        assert r.json()["total"] > 0
        # Also verify directly in the store.
        active = store.list_active(USER_ID, SCOPE)
        assert any("terraform" in u.content.lower() for u in active)

    def test_consolidates_after_add(self, client):
        """Every ingest appends a WORKING_SUMMARY. After the first add only one
        exists — nothing to supersede. After a second add with different content,
        consolidation supersedes the older WS: total > active and superseded >= 1.
        """
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We deploy with Helm charts for all services",
            "response_text": "Got it.",
        })
        s1 = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert s1["superseded"] == 0

        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "prompt_text": "We use Datadog for infrastructure monitoring",
            "response_text": "Understood.",
        })
        s2 = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert s2["superseded"] >= 1
        assert s2["entry_count"] < s2["total"]
        assert s2["active_by_type"].get("working_summary", 0) == 1

    def test_does_not_write_to_other_scope(self, client):
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "The project uses PostgreSQL as the primary database",
            "response_text": "Understood.",
        })
        assert client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()["entry_count"] == CORPUS_SIZE
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10})
        for hit in r.json()["results"]:
            assert hit["memory_id"].startswith("unit-")

    def test_does_not_write_to_other_user(self, client):
        other_user = "user-bob"
        # Write to the same scope name but for a different user.
        client.post("/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        # USER_ID's count in SCOPE must be unchanged.
        assert client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()["entry_count"] == CORPUS_SIZE
        # USER_ID's total across all scopes must also be unchanged.
        assert client.post("/memory/stats", json={"user_id": USER_ID}).json()["entry_count"] == CORPUS_SIZE


class TestMemoryAddBatch:
    def test_missing_user_id_is_422(self, client):
        assert client.post("/memory/add_batch", json={"turns": [{"prompt_text": "hi"}]}).status_code == 422

    def test_empty_turns_is_422(self, client):
        assert client.post("/memory/add_batch", json={"user_id": USER_ID, "scope_id": SCOPE, "turns": []}).status_code == 422

    def test_returns_units_added(self, client):
        r = client.post("/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [
                {"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."},
                {"prompt_text": "Secrets are stored in HashiCorp Vault", "response_text": "Understood."},
            ],
        })
        assert r.status_code == 200
        assert r.json()["units_added"] > 0

    def test_new_content_is_retrievable(self, client):
        client.post("/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [
                {"prompt_text": "We use Helm for Kubernetes package management", "response_text": "Got it."},
            ],
        })
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Helm kubernetes", "top_k": 10})
        assert r.status_code == 200
        assert r.json()["total"] > 0

    def test_consolidates_after_add_batch(self, client):
        client.post("/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [{"prompt_text": "We deploy with Helm charts for all services", "response_text": "Got it."}],
        })
        s1 = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert s1["superseded"] == 0

        client.post("/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": SCOPE,
            "turns": [{"prompt_text": "We use Datadog for infrastructure monitoring", "response_text": "Understood."}],
        })
        s2 = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()
        assert s2["superseded"] >= 1
        assert s2["active_by_type"].get("working_summary", 0) == 1

    def test_does_not_write_to_other_scope(self, client):
        client.post("/memory/add_batch", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "turns": [{"prompt_text": "The project uses PostgreSQL as the primary database", "response_text": "Understood."}],
        })
        assert client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()["entry_count"] == CORPUS_SIZE


class TestMemoryRetrieve:
    def test_missing_user_id_is_422(self, client):
        assert client.post("/memory/retrieve", json={"query": "x"}).status_code == 422

    def test_missing_query_is_422(self, client):
        assert client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE}).status_code == 422

    def test_returns_units_for_matching_query(self, client):
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] > 0
        hit = body["results"][0]
        for field in ("memory_id", "content", "summary", "memory_type", "importance",
                      "score", "matched_terms", "entities", "topics", "updated_at"):
            assert field in hit, f"missing field: {field}"
        assert isinstance(hit["score"], float)
        assert isinstance(hit["matched_terms"], list)

    def test_top_k_limits_result_count(self, client):
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 1})
        assert len(r.json()["results"]) <= 1

    def test_returns_empty_for_unrelated_query(self, client):
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "xyzzy_zap_unrelated_42", "top_k": 10})
        assert r.json()["total"] == 0

    def test_returns_empty_for_unknown_scope(self, client):
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE, "query": "database", "top_k": 10})
        assert r.json()["total"] == 0

    def test_scope_filters_out_other_scopes(self, client):
        # Seed a second scope for the same user with a distinct term.
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "We use Terraform for provisioning",
            "response_text": "Noted.",
        })
        # Querying SCOPE must not return the Terraform unit from OTHER_SCOPE.
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] == 0

    def test_no_scope_returns_all_user_scopes(self, client):
        # Seed OTHER_SCOPE for the same user.
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "We use Terraform for provisioning",
            "response_text": "Noted.",
        })
        # Querying without scope_id must find Terraform (in OTHER_SCOPE) alongside
        # corpus memories (in SCOPE).
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "query": "Terraform", "top_k": 10})
        assert r.json()["total"] > 0

    def test_user_cannot_access_other_user_memories_in_same_scope(self, client):
        other_user = "user-bob"
        # Seed the same scope name for a different user with a distinct term.
        client.post("/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        # USER_ID querying the same scope must not see Ansible (belongs to other_user).
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0
        # USER_ID querying without scope must also not see Ansible.
        r = client.post("/memory/retrieve", json={"user_id": USER_ID, "query": "Ansible", "top_k": 10})
        assert r.json()["total"] == 0


class TestMemoryClear:
    def test_missing_user_id_is_422(self, client):
        assert client.post("/memory/clear", json={}).status_code == 422

    def test_clears_existing_memories(self, client):
        r = client.post("/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == USER_ID
        assert body["scope_id"] == SCOPE
        assert body["archived"] == CORPUS_SIZE
        assert body["pinned_kept"] == 0
        assert body["total_before"] == CORPUS_SIZE
        assert client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": SCOPE}).json()["entry_count"] == 0
        assert client.post("/memory/retrieve", json={"user_id": USER_ID, "scope_id": SCOPE, "query": "database", "top_k": 10}).json()["total"] == 0

    def test_empty_scope_returns_zero_archived(self, client):
        r = client.post("/memory/clear", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE})
        assert r.status_code == 200
        assert r.json()["archived"] == 0

    def test_does_not_affect_other_scope(self, client):
        client.post("/memory/add", json={
            "user_id": USER_ID,
            "scope_id": OTHER_SCOPE,
            "prompt_text": "The project uses PostgreSQL as the primary database",
            "response_text": "Understood.",
        })
        bob_before = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE}).json()["entry_count"]
        client.post("/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        bob_after = client.post("/memory/stats", json={"user_id": USER_ID, "scope_id": OTHER_SCOPE}).json()["entry_count"]
        assert bob_after == bob_before

    def test_does_not_affect_other_user_same_scope(self, client):
        other_user = "user-bob"
        # Give other_user memories in the same scope name.
        client.post("/memory/add", json={
            "user_id": other_user,
            "scope_id": SCOPE,
            "prompt_text": "We use Ansible for configuration management",
            "response_text": "Noted.",
        })
        other_before = client.post("/memory/stats", json={"user_id": other_user, "scope_id": SCOPE}).json()["entry_count"]
        # Clearing USER_ID's scope must not touch other_user's memories.
        client.post("/memory/clear", json={"user_id": USER_ID, "scope_id": SCOPE})
        other_after = client.post("/memory/stats", json={"user_id": other_user, "scope_id": SCOPE}).json()["entry_count"]
        assert other_after == other_before
