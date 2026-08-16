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

from marklymem.evolver.consolidator import MemoryConsolidator
from marklymem.evolver.models import MemoryType, MemoryUnit

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
            memory_id="dup-001", namespace="scope",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dup-002", namespace="scope",
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
            memory_id="ct-001", namespace="scope",
            memory_type=MemoryType.SEMANTIC,
            content="common content here",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="ct-002", namespace="scope",
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
    def _near_dup_pair(self, namespace="nd") -> tuple[MemoryUnit, MemoryUnit]:
        # Jaccard ≥ 0.80: 7 shared tokens, 8 total.
        u1 = _make_unit(
            memory_id="nd-001", namespace=namespace,
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            importance=0.6,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="nd-002", namespace=namespace,
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
            memory_id="ti-001", namespace="ti",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            importance=0.5,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="ti-002", namespace="ti",
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
            memory_id="dt-001", namespace="dt",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dt-002", namespace="dt",
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
            memory_id="re-001", namespace="re",
            content="PostgreSQL is the primary database",
            entities=["PostgreSQL"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="re-002", namespace="re",
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
            memory_id="bv-001", namespace="bv",
            content="Redis cache configuration",
            entities=["Redis"],
            reinforcement_score=0.0,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="bv-002", namespace="bv",
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
            memory_id="cap-001", namespace="cap",
            content="Redis cache system",
            entities=["Redis"],
            reinforcement_score=0.28,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="cap-002", namespace="cap",
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
            memory_id="decay-lin-001", namespace="decay",
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
            memory_id="decay-exp-001", namespace="decay_exp",
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
            memory_id="no-decay-001", namespace="nd",
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
            memory_id="clamp-001", namespace="clamp",
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
        u1 = _make_unit(
            memory_id="ddr-001", namespace="dry",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="ddr-002", namespace="dry",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for data storage",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        result = await _consolidator(store, threshold=0.80).dry_run(UID, "dry")
        assert "exact_duplicates" in result
        assert "near_duplicates" in result
        assert "total_actions" in result

    async def test_dry_run_does_not_mutate(self, store):
        u1 = _make_unit(
            memory_id="dm-001", namespace="dry2",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dm-002", namespace="dry2",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for data storage",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2])
        await _consolidator(store, threshold=0.80).dry_run(UID, "dry2")
        active_ids = {u.memory_id for u in await store.list_active(UID, "dry2")}
        assert "dm-001" in active_ids
        assert "dm-002" in active_ids


# ---------------------------------------------------------------------------
# Hierarchical namespace isolation during consolidation
# ---------------------------------------------------------------------------

class TestHierarchicalNamespaceConsolidation:
    async def test_consolidation_does_not_cross_sibling_scopes(self, store):
        # cs-001 and cs-003 are exact duplicates inside scope1/subscope1 — consolidation
        # must fire and merge them, but must not touch the identical unit in subscope2.
        content = "identical content for dedup test"
        u1 = _make_unit(
            memory_id="cs-001", namespace="scope1/subscope1", content=content,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(memory_id="cs-002", namespace="scope1/subscope2", content=content)
        u3 = _make_unit(
            memory_id="cs-003", namespace="scope1/subscope1", content=content,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u1, u2, u3])
        result = await _consolidator(store).consolidate(UID, "scope1/subscope1")
        assert result["superseded"] >= 1  # consolidation actually fired
        active_subscope2 = {u.memory_id for u in await store.list_active(UID, "scope1/subscope2")}
        assert "cs-002" in active_subscope2

    async def test_consolidation_does_not_expand_to_child_scopes(self, store):
        # Consolidating the parent namespace must not touch memories in child scopes.
        content = "identical content for dedup test"
        u1 = _make_unit(memory_id="cp-001", namespace="scope1/subscope1", content=content)
        u2 = _make_unit(memory_id="cp-002", namespace="scope1/subscope2", content=content)
        await store.add_memories([u1, u2])
        await _consolidator(store).consolidate(UID, "scope1")
        # Both child units must survive — parent consolidation can't see them.
        active_subscope1 = {u.memory_id for u in await store.list_active(UID, "scope1/subscope1")}
        active_subscope2 = {u.memory_id for u in await store.list_active(UID, "scope1/subscope2")}
        assert "cp-001" in active_subscope1
        assert "cp-002" in active_subscope2

    async def test_consolidation_within_exact_scope_still_works(self, store):
        # Duplicates within the same exact namespace are still consolidated.
        content = "identical content for dedup test"
        u1 = _make_unit(memory_id="ce-001", namespace="scope1/subscope1", content=content)
        u2 = _make_unit(memory_id="ce-002", namespace="scope1/subscope1", content=content)
        await store.add_memories([u1, u2])
        await _consolidator(store).consolidate(UID, "scope1/subscope1")
        active = {u.memory_id for u in await store.list_active(UID, "scope1/subscope1")}
        assert len(active) == 1
