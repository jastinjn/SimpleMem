# pyright: reportMissingImports=false
"""Tests for MemoryConsolidator.

decay_factor=0 by default (no time-based decay) unless a test explicitly
exercises decay and uses frozen_consolidator_clock.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evolver_server.evolver.consolidator import MemoryConsolidator
from evolver_server.evolver.models import MemoryType, MemoryUnit

from .conftest import UID, _make_unit, create_test_units


def _consolidator(store, threshold=0.80, decay_factor=0.0, **kwargs) -> MemoryConsolidator:
    return MemoryConsolidator(
        store=store,
        similarity_threshold=threshold,
        decay_factor=decay_factor,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Exact-duplicate deduplication
# ---------------------------------------------------------------------------

class TestExactDuplicateDedup:
    async def test_exact_dup_superseded(self, store):
        u1 = _make_unit(
            memory_id="dup-001", scope_id="scope",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dup-002", scope_id="scope",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        result = await _consolidator(store).consolidate(UID, "scope")
        assert result["superseded"] >= 1
        active_ids = {u.memory_id for u in await store.list_active(UID, "scope")}
        assert len(active_ids & {"dup-001", "dup-002"}) == 1

    async def test_different_content_not_deduped(self, store):
        await store.add_memories(create_test_units())
        result = await _consolidator(store).consolidate(UID, "test")
        assert result["superseded"] == 0

    async def test_same_content_different_type_not_deduped(self, store):
        u1 = _make_unit(
            memory_id="ct-001", scope_id="scope",
            memory_type=MemoryType.SEMANTIC,
            content="common content here",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="ct-002", scope_id="scope",
            memory_type=MemoryType.EPISODIC,
            content="common content here",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        result = await _consolidator(store).consolidate(UID, "scope")
        assert result["superseded"] == 0


# ---------------------------------------------------------------------------
# Near-duplicate merge
# ---------------------------------------------------------------------------

class TestNearDuplicateMerge:
    def _near_dup_pair(self, scope="nd") -> tuple[MemoryUnit, MemoryUnit]:
        # Jaccard ≥ 0.80: 7 shared tokens, 8 total.
        u1 = _make_unit(
            memory_id="nd-001", scope_id=scope,
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            importance=0.6,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="nd-002", scope_id=scope,
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for data storage",
            importance=0.4,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        return u1, u2

    async def test_near_dup_merged(self, store):
        u1, u2 = self._near_dup_pair()
        await store.add_memories([u1, u2])
        result = await _consolidator(store, threshold=0.80).consolidate(UID, "nd")
        assert result["superseded"] >= 1
        active_ids = {u.memory_id for u in await store.list_active(UID, "nd")}
        assert len(active_ids & {"nd-001", "nd-002"}) == 1

    async def test_tie_break_higher_importance_kept(self, store):
        u1, u2 = self._near_dup_pair()
        await store.add_memories([u1, u2])
        await _consolidator(store, threshold=0.80).consolidate(UID, "nd")
        active_ids = {u.memory_id for u in await store.list_active(UID, "nd")}
        assert "nd-001" in active_ids
        assert "nd-002" not in active_ids

    async def test_tie_break_newer_updated_at_when_equal_importance(self, store):
        u1 = _make_unit(
            memory_id="ti-001", scope_id="ti",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            importance=0.5,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="ti-002", scope_id="ti",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for data storage",
            importance=0.5,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store, threshold=0.80).consolidate(UID, "ti")
        active_ids = {u.memory_id for u in await store.list_active(UID, "ti")}
        assert "ti-002" in active_ids
        assert "ti-001" not in active_ids

    async def test_different_types_not_merged(self, store):
        u1 = _make_unit(
            memory_id="dt-001", scope_id="dt",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dt-002", scope_id="dt",
            memory_type=MemoryType.EPISODIC,
            content="The project uses PostgreSQL database for data storage",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store, threshold=0.80).consolidate(UID, "dt")
        active_ids = {u.memory_id for u in await store.list_active(UID, "dt")}
        assert "dt-001" in active_ids
        assert "dt-002" in active_ids


# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------

class TestReinforceSharedEntities:
    async def test_shared_entity_boosts_reinforcement(self, store):
        u1 = _make_unit(
            memory_id="re-001", scope_id="re",
            content="PostgreSQL is the primary database",
            entities=["PostgreSQL"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="re-002", scope_id="re",
            content="PostgreSQL handles all data storage",
            entities=["PostgreSQL"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        result = await _consolidator(store).consolidate(UID, "re")
        assert result["reinforced"] >= 1
        fetched = await store.get_by_ids(["re-001", "re-002"])
        assert any(f.reinforcement_score > 0.0 for f in fetched)

    async def test_boost_value(self, store):
        u1 = _make_unit(
            memory_id="bv-001", scope_id="bv",
            content="Redis cache configuration",
            entities=["Redis"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="bv-002", scope_id="bv",
            content="Redis used for session tokens",
            entities=["Redis"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store).consolidate(UID, "bv")
        fetched = {f.memory_id: f for f in await store.get_by_ids(["bv-001", "bv-002"])}
        for f in fetched.values():
            assert f.reinforcement_score == pytest.approx(0.05)

    async def test_boost_capped(self, store):
        u1 = _make_unit(
            memory_id="cap-001", scope_id="cap",
            content="Redis cache system",
            entities=["Redis"],
            reinforcement_score=0.28,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="cap-002", scope_id="cap",
            content="Redis is used for caching",
            entities=["Redis"],
            reinforcement_score=0.28,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store).consolidate(UID, "cap")
        fetched = await store.get_by_ids(["cap-001"])
        assert fetched[0].reinforcement_score == pytest.approx(0.30, abs=1e-4)


# ---------------------------------------------------------------------------
# Importance decay
# ---------------------------------------------------------------------------

class TestImportanceDecay:
    async def test_linear_decay(self, store, frozen_consolidator_clock):
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="decay-lin-001", scope_id="decay",
            content="some old memory content",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        await store.add_memories([u])
        result = await MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.05,
            min_importance=0.1,
            decay_mode="linear",
        ).consolidate(UID, "decay")
        assert result["decayed"] == 1
        fetched = await store.get_by_ids(["decay-lin-001"])
        assert fetched[0].importance == pytest.approx(0.475, abs=1e-4)

    async def test_exponential_decay(self, store, frozen_consolidator_clock):
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="decay-exp-001", scope_id="decay_exp",
            content="some old memory content",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        await store.add_memories([u])
        result = await MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.05,
            min_importance=0.1,
            decay_mode="exponential",
        ).consolidate(UID, "decay_exp")
        assert result["decayed"] == 1
        fetched = await store.get_by_ids(["decay-exp-001"])
        expected = max(0.1, 0.5 * math.exp(-0.05 * 1.0))
        assert fetched[0].importance == pytest.approx(round(expected, 4), abs=1e-4)

    async def test_no_decay_when_factor_zero(self, store, frozen_consolidator_clock):
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="no-decay-001", scope_id="nd",
            content="old memory",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        await store.add_memories([u])
        result = await MemoryConsolidator(store=store, decay_factor=0.0).consolidate(UID, "nd")
        assert result["decayed"] == 0

    async def test_decay_clamped_to_min_importance(self, store, frozen_consolidator_clock):
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="clamp-001", scope_id="clamp",
            content="very old memory",
            importance=0.2,
            updated_at=old_ts,
            created_at=old_ts,
        )
        await store.add_memories([u])
        await MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.5,
            min_importance=0.15,
            decay_mode="linear",
        ).consolidate(UID, "clamp")
        fetched = await store.get_by_ids(["clamp-001"])
        assert fetched[0].importance >= 0.15


# ---------------------------------------------------------------------------
# Working-summary pruning
# ---------------------------------------------------------------------------

class TestWorkingSummaryPruning:
    async def test_only_newest_summary_kept(self, store):
        ws1 = _make_unit(
            memory_id="ws-001", scope_id="ws",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Old working summary",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        ws2 = _make_unit(
            memory_id="ws-002", scope_id="ws",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Newer working summary",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        ws3 = _make_unit(
            memory_id="ws-003", scope_id="ws",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Newest working summary",
            updated_at="2025-01-01T00:00:03+00:00",
        )
        await store.add_memories([ws1, ws2, ws3])
        result = await _consolidator(store).consolidate(UID, "ws")
        assert result["superseded"] == 2
        active_ids = {u.memory_id for u in await store.list_active(UID, "ws")}
        assert "ws-003" in active_ids
        assert "ws-001" not in active_ids
        assert "ws-002" not in active_ids


# ---------------------------------------------------------------------------
# Return value and dry_run
# ---------------------------------------------------------------------------

class TestConsolidateReturnValue:
    async def test_returns_dict_with_keys(self, store):
        await store.add_memories(create_test_units())
        result = await _consolidator(store).consolidate(UID, "test")
        assert "superseded" in result
        assert "decayed" in result
        assert "reinforced" in result


class TestDryRun:
    async def test_dry_run_returns_counts(self, store):
        ws1 = _make_unit(
            memory_id="dws-001", scope_id="dry",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="Old summary",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        ws2 = _make_unit(
            memory_id="dws-002", scope_id="dry",
            memory_type=MemoryType.WORKING_SUMMARY,
            content="New summary",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([ws1, ws2])
        result = await _consolidator(store).dry_run(UID, "dry")
        assert result["stale_summaries"] == 1

    async def test_dry_run_does_not_mutate(self, store):
        u1 = _make_unit(
            memory_id="dm-001", scope_id="dry2",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dm-002", scope_id="dry2",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for data storage",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store, threshold=0.80).dry_run(UID, "dry2")
        active_ids = {u.memory_id for u in await store.list_active(UID, "dry2")}
        assert "dm-001" in active_ids
        assert "dm-002" in active_ids
