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

from simplemem.evolver.consolidator import MemoryConsolidator
from simplemem.evolver.models import MemoryStatus, MemoryType, MemoryUnit

from .conftest import _make_store, _make_unit, create_test_units, FROZEN_NOW


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
    def test_exact_dup_superseded(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([u1, u2])
        result = _consolidator(store).consolidate("scope")
        assert result["superseded"] >= 1
        active_ids = {u.memory_id for u in store.list_active("scope")}
        # Exactly one of the pair must remain active.
        assert len(active_ids & {"dup-001", "dup-002"}) == 1

    def test_different_content_not_deduped(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        result = _consolidator(store).consolidate("test")
        # All 6 units have distinct content so no exact dups.
        assert result["superseded"] == 0

    def test_same_content_different_type_not_deduped(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([u1, u2])
        result = _consolidator(store).consolidate("scope")
        # Different types → not deduped.
        assert result["superseded"] == 0


# ---------------------------------------------------------------------------
# Near-duplicate merge
# ---------------------------------------------------------------------------

class TestNearDuplicateMerge:
    def _near_dup_pair(self, scope="nd") -> tuple[MemoryUnit, MemoryUnit]:
        # Jaccard ≥ 0.80: 7 shared tokens, 8 total.
        # "The project uses PostgreSQL database for storage" (7 tokens)
        # "The project uses PostgreSQL database for data storage" (8 tokens)
        # intersection=7, union=8, jaccard=7/8=0.875
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

    def test_near_dup_merged(self, tmp_path):
        store = _make_store(tmp_path)
        u1, u2 = self._near_dup_pair()
        store.add_memories([u1, u2])
        result = _consolidator(store, threshold=0.80).consolidate("nd")
        assert result["superseded"] >= 1
        active_ids = {u.memory_id for u in store.list_active("nd")}
        assert len(active_ids & {"nd-001", "nd-002"}) == 1

    def test_tie_break_higher_importance_kept(self, tmp_path):
        store = _make_store(tmp_path)
        u1, u2 = self._near_dup_pair()
        # u1 has higher importance (0.6 > 0.4) → keep u1.
        store.add_memories([u1, u2])
        _consolidator(store, threshold=0.80).consolidate("nd")
        active_ids = {u.memory_id for u in store.list_active("nd")}
        assert "nd-001" in active_ids
        assert "nd-002" not in active_ids

    def test_tie_break_newer_updated_at_when_equal_importance(self, tmp_path):
        store = _make_store(tmp_path)
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
            importance=0.5,  # equal importance
            updated_at="2025-01-01T00:00:02+00:00",  # newer
        )
        store.add_memories([u1, u2])
        _consolidator(store, threshold=0.80).consolidate("ti")
        active_ids = {u.memory_id for u in store.list_active("ti")}
        # u2 is newer → should be kept.
        assert "ti-002" in active_ids
        assert "ti-001" not in active_ids

    def test_different_types_not_merged(self, tmp_path):
        store = _make_store(tmp_path)
        u1 = _make_unit(
            memory_id="dt-001", scope_id="dt",
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL database for storage",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u2 = _make_unit(
            memory_id="dt-002", scope_id="dt",
            memory_type=MemoryType.EPISODIC,  # different type
            content="The project uses PostgreSQL database for data storage",
            updated_at="2025-01-01T00:00:02+00:00",
        )
        store.add_memories([u1, u2])
        result = _consolidator(store, threshold=0.80).consolidate("dt")
        # Cross-type merge must not happen.
        active_ids = {u.memory_id for u in store.list_active("dt")}
        assert "dt-001" in active_ids
        assert "dt-002" in active_ids


# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------

class TestReinforceSharedEntities:
    def test_shared_entity_boosts_reinforcement(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([u1, u2])
        result = _consolidator(store).consolidate("re")
        assert result["reinforced"] >= 1
        fetched = store.get_by_ids(["re-001", "re-002"])
        assert any(f.reinforcement_score > 0.0 for f in fetched)

    def test_boost_value(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([u1, u2])
        _consolidator(store).consolidate("bv")
        fetched = {f.memory_id: f for f in store.get_by_ids(["bv-001", "bv-002"])}
        # Boost = min(0.05, 0.3 - 0.0) = 0.05, new = round(0.0 + 0.05, 4) = 0.05
        for f in fetched.values():
            assert f.reinforcement_score == pytest.approx(0.05)

    def test_boost_capped(self, tmp_path):
        store = _make_store(tmp_path)
        # Already at 0.28: boost = min(0.05, 0.3 - 0.28) = 0.02
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
        store.add_memories([u1, u2])
        _consolidator(store).consolidate("cap")
        fetched = store.get_by_ids(["cap-001"])
        assert fetched[0].reinforcement_score == pytest.approx(0.30, abs=1e-4)


# ---------------------------------------------------------------------------
# Importance decay
# ---------------------------------------------------------------------------

class TestImportanceDecay:
    def test_linear_decay(self, tmp_path, frozen_consolidator_clock):
        store = _make_store(tmp_path)
        # FROZEN_NOW = 2025-02-15; decay_after_days=1; unit updated 2025-02-13 (2 days ago)
        # periods = (2-1)/1 = 1.0; new = 0.5 * (1 - 0.05*1) = 0.475
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="decay-lin-001", scope_id="decay",
            content="some old memory content",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        store.add_memories([u])
        result = MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.05,
            min_importance=0.1,
            decay_mode="linear",
        ).consolidate("decay")
        assert result["decayed"] == 1
        fetched = store.get_by_ids(["decay-lin-001"])[0]
        assert fetched.importance == pytest.approx(0.475, abs=1e-4)

    def test_exponential_decay(self, tmp_path, frozen_consolidator_clock):
        store = _make_store(tmp_path)
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="decay-exp-001", scope_id="decay_exp",
            content="some old memory content",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        store.add_memories([u])
        result = MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.05,
            min_importance=0.1,
            decay_mode="exponential",
        ).consolidate("decay_exp")
        assert result["decayed"] == 1
        fetched = store.get_by_ids(["decay-exp-001"])[0]
        # periods=1.0, new = 0.5 * exp(-0.05 * 1.0)
        expected = max(0.1, 0.5 * math.exp(-0.05 * 1.0))
        assert fetched.importance == pytest.approx(round(expected, 4), abs=1e-4)

    def test_no_decay_when_factor_zero(self, tmp_path, frozen_consolidator_clock):
        store = _make_store(tmp_path)
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="no-decay-001", scope_id="nd",
            content="old memory",
            importance=0.5,
            updated_at=old_ts,
            created_at=old_ts,
        )
        store.add_memories([u])
        result = MemoryConsolidator(store=store, decay_factor=0.0).consolidate("nd")
        assert result["decayed"] == 0

    def test_decay_clamped_to_min_importance(self, tmp_path, frozen_consolidator_clock):
        store = _make_store(tmp_path)
        # Even with many periods, importance cannot go below min_importance.
        old_ts = "2025-02-13T00:00:00+00:00"
        u = _make_unit(
            memory_id="clamp-001", scope_id="clamp",
            content="very old memory",
            importance=0.2,
            updated_at=old_ts,
            created_at=old_ts,
        )
        store.add_memories([u])
        MemoryConsolidator(
            store=store,
            decay_after_days=1,
            decay_factor=0.5,
            min_importance=0.15,
            decay_mode="linear",
        ).consolidate("clamp")
        fetched = store.get_by_ids(["clamp-001"])[0]
        assert fetched.importance >= 0.15


# ---------------------------------------------------------------------------
# Working-summary pruning
# ---------------------------------------------------------------------------

class TestWorkingSummaryPruning:
    def test_only_newest_summary_kept(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([ws1, ws2, ws3])
        result = _consolidator(store).consolidate("ws")
        assert result["superseded"] == 2
        active_ids = {u.memory_id for u in store.list_active("ws")}
        assert "ws-003" in active_ids
        assert "ws-001" not in active_ids
        assert "ws-002" not in active_ids


# ---------------------------------------------------------------------------
# Return value and dry_run
# ---------------------------------------------------------------------------

class TestConsolidateReturnValue:
    def test_returns_dict_with_keys(self, tmp_path):
        store = _make_store(tmp_path)
        store.add_memories(create_test_units())
        result = _consolidator(store).consolidate("test")
        assert "superseded" in result
        assert "decayed" in result
        assert "reinforced" in result


class TestDryRun:
    def test_dry_run_returns_counts(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([ws1, ws2])
        result = _consolidator(store).dry_run("dry")
        assert result["stale_summaries"] == 1

    def test_dry_run_does_not_mutate(self, tmp_path):
        store = _make_store(tmp_path)
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
        store.add_memories([u1, u2])
        _consolidator(store, threshold=0.80).dry_run("dry2")
        active_ids = {u.memory_id for u in store.list_active("dry2")}
        assert "dm-001" in active_ids
        assert "dm-002" in active_ids
