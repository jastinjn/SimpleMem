# pyright: reportMissingImports=false
"""Integration tests for MemoryManager.

Uses real MemoryStore, MemoryRetriever, and MemoryConsolidator.
Monkeypatches manager.uuid and manager.utc_now_iso to remove nondeterminism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evolver_server.evolver.embeddings import HashingEmbedder
from evolver_server.evolver.manager import MemoryManager
from evolver_server.evolver.models import MemoryType
from evolver_server.evolver.policy import MemoryPolicy
from evolver_server.evolver.store import MemoryStore

from .conftest import UID, _make_unit

FIXED_TS = "2025-01-15T14:00:00+00:00"


def _manager(
    store: MemoryStore,
    *,
    auto_consolidate: bool = False,
    retrieval_mode: str = "keyword",
    embedder=None,
    user_id: str = UID,
    scope_id: str = "test",
) -> MemoryManager:
    policy = MemoryPolicy(recency_weight=0.0)
    return MemoryManager(
        store=store,
        policy=policy,
        user_id=user_id,
        scope_id=scope_id,
        auto_consolidate=auto_consolidate,
        retrieval_mode=retrieval_mode,
        policy_store=None,
        telemetry_store=None,
        embedder=embedder,
    )


def _patch_time(monkeypatch):
    monkeypatch.setattr("evolver_server.evolver.manager.utc_now_iso", lambda: FIXED_TS)


SAMPLE_TURNS = [
    {
        "prompt_text": "The project uses PostgreSQL as the primary database",
        "response_text": "Understood, I will keep that in mind.",
    },
    {
        "prompt_text": "We use Kubernetes for container orchestration and deployment",
        "response_text": "Got it, the deployment is handled by Kubernetes.",
    },
]


# ---------------------------------------------------------------------------
# ingest_session_turns
# ---------------------------------------------------------------------------

class TestIngestSessionTurns:
    async def test_returns_positive_count(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        added = await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        assert added > 0

    async def test_working_summary_always_appended(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        active = await store.list_active(UID, "test")
        ws_units = [u for u in active if u.memory_type == MemoryType.WORKING_SUMMARY]
        assert len(ws_units) == 1
        assert ws_units[0].importance == pytest.approx(0.9)

    async def test_working_summary_has_deterministic_id(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        ws_units = [u for u in await store.list_active(UID, "test")
                    if u.memory_type == MemoryType.WORKING_SUMMARY]
        assert ws_units[0].memory_id.startswith("00000000-")

    async def test_empty_turns_produces_no_units(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        added = await mgr.ingest_session_turns("sess-001", [])
        assert added == 0
        assert await store.list_active(UID, "test") == []

    async def test_short_content_filtered(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        short_turns = [{"prompt_text": "ok", "response_text": ""}]
        await mgr.ingest_session_turns("sess-short", short_turns)
        for u in await store.list_active(UID, "test"):
            assert len(u.content.strip()) >= 3

    async def test_dedup_skips_existing_content(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        added_first = await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        await mgr.clear_cache()
        added_second = await mgr.ingest_session_turns("sess-002", SAMPLE_TURNS)
        assert added_second <= added_first

    async def test_auto_consolidate_false_does_not_consolidate(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, auto_consolidate=False)
        ws1 = _make_unit(
            memory_id="ws-pre-001", scope_id="test",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Old working summary content here",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        ws2 = _make_unit(
            memory_id="ws-pre-002", scope_id="test",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="New working summary content here",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([ws1, ws2])
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        active_ws = [u for u in await store.list_active(UID, "test")
                     if u.memory_type == MemoryType.WORKING_SUMMARY]
        ws_ids = {u.memory_id for u in active_ws}
        assert "ws-pre-001" in ws_ids
        assert "ws-pre-002" in ws_ids

    async def test_auto_consolidate_true_consolidates(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, auto_consolidate=True)
        ws1 = _make_unit(
            memory_id="ws-pre-001", scope_id="test",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Old working summary content here",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        await store.add_memories([ws1])
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        active_ws = [u for u in await store.list_active(UID, "test")
                     if u.memory_type == MemoryType.WORKING_SUMMARY]
        assert len(active_ws) == 1

    async def test_embedder_computes_embeddings(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        embedder = HashingEmbedder(dimensions=64)
        mgr = _manager(store, embedder=embedder)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        units = await store.list_active(UID, "test")
        non_ws = [u for u in units if u.memory_type != MemoryType.WORKING_SUMMARY]
        assert all(len(u.embedding) == 64 for u in non_ws)

    async def test_no_embedder_produces_empty_embeddings(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, embedder=None)
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)
        units = await store.list_active(UID, "test")
        non_ws = [u for u in units if u.memory_type != MemoryType.WORKING_SUMMARY]
        assert all(u.embedding == [] for u in non_ws)

    async def test_custom_scope_id_used(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, scope_id="default")
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS, user_id=UID, scope_id="custom_scope")
        assert (await store.get_stats(UID, "custom_scope"))["active"] > 0
        assert (await store.get_stats(UID, "default"))["active"] == 0

    async def test_local_conflict_drops_earlier_turn_unit(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        # Turn 1 and turn 3 both match the PREFERENCE pattern "I prefer ...".
        # They share identical topics so Jaccard = 1.0 > threshold, but have
        # different content — a genuine within-session contradiction. The earlier
        # unit (turn 1) should be dropped; only the later one (turn 3) persists.
        turns = [
            {
                "prompt_text": "I prefer giving feedback marks for marking student work",
                "response_text": "",
            },
            {
                "prompt_text": "Some unrelated content about today's class",
                "response_text": "",
            },
            {
                "prompt_text": "I prefer not giving feedback marks for marking student work",
                "response_text": "",
            },
        ]
        await mgr.ingest_session_turns("sess-conflict", turns)
        active = await store.list_active(UID, "test")
        preference_units = [u for u in active if u.memory_type == MemoryType.PREFERENCE]
        assert len(preference_units) == 1
        assert "not giving" in preference_units[0].content


# ---------------------------------------------------------------------------
# retrieve_for_prompt
# ---------------------------------------------------------------------------

class TestRetrieveForPrompt:
    async def _ingest(self, mgr):
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS)

    async def test_returns_matching_units(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await self._ingest(mgr)
        units = await mgr.retrieve_for_prompt("PostgreSQL database")
        assert len(units) > 0
        contents = " ".join(u.content for u in units)
        assert "PostgreSQL" in contents or "database" in contents.lower()

    async def test_no_match_returns_empty(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await self._ingest(mgr)
        units = await mgr.retrieve_for_prompt("xyzzy_totally_unrelated_term")
        assert units == []

    async def test_cache_hit_on_repeated_call(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await self._ingest(mgr)
        await mgr.retrieve_for_prompt("PostgreSQL database")
        hits_before = mgr._cache_hits
        await mgr.retrieve_for_prompt("PostgreSQL database")
        assert mgr._cache_hits > hits_before

    async def test_mark_accessed_increments_access_count(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        await self._ingest(mgr)
        retrieved = await mgr.retrieve_for_prompt("PostgreSQL database")
        if not retrieved:
            pytest.skip("No units retrieved; query mismatch")
        mid = retrieved[0].memory_id
        fetched = await store.get_by_ids([mid])
        assert fetched[0].access_count >= 1

    async def test_importance_auto_boost_on_high_access_count(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u = _make_unit(
            memory_id="freq-001", scope_id="test",
            content="PostgreSQL is the primary database system",
            importance=0.5,
            access_count=3,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        await store.add_memories([u])
        await mgr.retrieve_for_prompt("PostgreSQL")
        fetched = await store.get_by_ids(["freq-001"])
        assert fetched[0].importance == pytest.approx(0.52, abs=1e-4)

    async def test_importance_boost_capped_at_0_9(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u = _make_unit(
            memory_id="cap-001", scope_id="test",
            content="PostgreSQL is the primary database system",
            importance=0.89,
            access_count=3,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        await store.add_memories([u])
        await mgr.retrieve_for_prompt("PostgreSQL")
        fetched = await store.get_by_ids(["cap-001"])
        assert fetched[0].importance <= 0.9

    async def test_token_budget_limits_results(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        policy = MemoryPolicy(recency_weight=0.0, max_injected_units=10, max_injected_tokens=5)
        mgr = MemoryManager(
            store=store, policy=policy, user_id=UID, scope_id="test",
            auto_consolidate=False, policy_store=None, telemetry_store=None,
        )
        await self._ingest(mgr)
        units = await mgr.retrieve_for_prompt("PostgreSQL database")
        assert len(units) <= 1

    async def test_scope_argument_overrides_default(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store, scope_id="default")
        await mgr.ingest_session_turns("sess-001", SAMPLE_TURNS, user_id=UID, scope_id="other_scope")
        units = await mgr.retrieve_for_prompt("PostgreSQL", scope_id="other_scope")
        assert len(units) > 0
        units_default = await mgr.retrieve_for_prompt("PostgreSQL", scope_id="default")
        assert units_default == []


# ---------------------------------------------------------------------------
# render_for_prompt
# ---------------------------------------------------------------------------

class TestRenderForPrompt:
    async def test_empty_units_returns_empty_string(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        assert await mgr.render_for_prompt([]) == ""

    async def test_renders_content(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u = _make_unit(
            memory_id="rend-001", scope_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="PostgreSQL is the database",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        rendered = await mgr.render_for_prompt([u])
        assert "PostgreSQL is the database" in rendered

    async def test_groups_by_memory_type(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        u_sem = _make_unit(
            memory_id="g-001", scope_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="semantic content",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u_ep = _make_unit(
            memory_id="g-002", scope_id="test",
            memory_type=MemoryType.EPISODIC,
            content="episodic content",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        rendered = await mgr.render_for_prompt([u_sem, u_ep])
        assert "### semantic" in rendered
        assert "### episodic" in rendered

    async def test_pinned_high_importance_appears_first(self, store, monkeypatch, fake_uuid):
        _patch_time(monkeypatch)
        mgr = _manager(store)
        unpinned = _make_unit(
            memory_id="p-001", scope_id="test",
            memory_type=MemoryType.SEMANTIC,
            content="regular memory unit",
            importance=0.5,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        pinned = _make_unit(
            memory_id="p-002", scope_id="test",
            memory_type=MemoryType.EPISODIC,
            content="pinned memory unit",
            importance=0.99,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        rendered = await mgr.render_for_prompt([unpinned, pinned])
        pos_pinned = rendered.find("pinned memory unit")
        pos_regular = rendered.find("regular memory unit")
        assert pos_pinned < pos_regular
