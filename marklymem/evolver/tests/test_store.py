# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.models import MemoryStatus, MemoryType

from .conftest import UID, _make_unit, create_test_units

# ---------------------------------------------------------------------------
# add_memories
# ---------------------------------------------------------------------------

class TestAddMemories:
    async def test_round_trip_by_id(self, store):
        units = create_test_units()
        await store.add_memories(units)
        fetched = await store.get_by_ids(["unit-001"])
        assert len(fetched) == 1
        assert fetched[0].content == units[0].content
        assert fetched[0].memory_type == MemoryType.SEMANTIC

    async def test_returns_count(self, store):
        n = await store.add_memories(create_test_units())
        assert n == 6

    async def test_upsert_on_same_memory_id(self, store):
        u = _make_unit(memory_id="x-001", content="original")
        await store.add_memories([u])
        u2 = _make_unit(memory_id="x-001", content="updated", updated_at="2025-06-01T00:00:10+00:00")
        await store.add_memories([u2])
        fetched = await store.get_by_ids(["x-001"])
        assert fetched[0].content == "updated"

    async def test_empty_iterable(self, store):
        assert await store.add_memories([]) == 0


# ---------------------------------------------------------------------------
# list_active
# ---------------------------------------------------------------------------

class TestListActive:
    async def test_only_active_returned(self, store):
        units = create_test_units()
        await store.add_memories(units)
        await store.supersede("unit-001", "unit-002", updated_at="2025-01-15T15:00:00+00:00")
        active = await store.list_active(UID, "test")
        ids = {u.memory_id for u in active}
        assert "unit-001" not in ids
        assert "unit-002" in ids

    async def test_scope_isolation(self, store):
        u_a = _make_unit(memory_id="a-001", scope_id="scope_a", updated_at="2025-01-01T00:00:01+00:00")
        u_b = _make_unit(memory_id="b-001", scope_id="scope_b", updated_at="2025-01-01T00:00:02+00:00")
        await store.add_memories([u_a, u_b])
        assert all(u.scope_id == "scope_a" for u in await store.list_active(UID, "scope_a"))
        assert all(u.scope_id == "scope_b" for u in await store.list_active(UID, "scope_b"))

    async def test_ordered_by_updated_at_desc(self, store):
        await store.add_memories(create_test_units())
        active = await store.list_active(UID, "test")
        timestamps = [u.updated_at for u in active]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_filters_expired(self, store):
        expired = _make_unit(
            memory_id="exp-001",
            expires_at="2020-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        alive = _make_unit(
            memory_id="alive-001",
            expires_at="",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        await store.add_memories([expired, alive])
        ids = {u.memory_id for u in await store.list_active(UID, "test")}
        assert "exp-001" not in ids
        assert "alive-001" in ids

    async def test_limit_respected(self, store):
        await store.add_memories(create_test_units())
        assert len(await store.list_active(UID, "test", limit=2)) == 2


# ---------------------------------------------------------------------------
# search_keyword
# ---------------------------------------------------------------------------

class TestSearchKeyword:
    async def test_basic_match(self, store):
        await store.add_memories(create_test_units())
        hits = await store.search_keyword(UID, "test", "PostgreSQL", limit=6)
        ids = {h.unit.memory_id for h in hits}
        assert "unit-001" in ids

    async def test_no_match_returns_empty(self, store):
        await store.add_memories(create_test_units())
        hits = await store.search_keyword(UID, "test", "xyzzy_no_match", limit=6)
        assert hits == []

    async def test_scope_isolation(self, store):
        u_a = _make_unit(
            memory_id="sa-001", scope_id="scope_a",
            content="PostgreSQL database", updated_at="2025-01-01T00:00:01+00:00",
        )
        u_b = _make_unit(
            memory_id="sb-001", scope_id="scope_b",
            content="PostgreSQL database", updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u_a, u_b])
        hits = await store.search_keyword(UID, "scope_a", "PostgreSQL")
        assert all(h.unit.scope_id == "scope_a" for h in hits)

    async def test_limit(self, store):
        await store.add_memories(create_test_units())
        hits = await store.search_keyword(UID, "test", "the", limit=2)
        assert len(hits) <= 2

    async def test_higher_importance_ranks_higher(self, store):
        low = _make_unit(
            memory_id="low-001", scope_id="rank",
            content="the project database", importance=0.1,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        high = _make_unit(
            memory_id="high-001", scope_id="rank",
            content="the project database", importance=0.9,
            updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([low, high])
        hits = await store.search_keyword(UID, "rank", "project database", limit=6)
        assert hits[0].unit.memory_id == "high-001"


# ---------------------------------------------------------------------------
# Mutation methods
# ---------------------------------------------------------------------------

class TestSupersede:
    async def test_status_becomes_superseded(self, store):
        u = _make_unit(memory_id="s-001")
        await store.add_memories([u])
        await store.supersede("s-001", "s-002", updated_at="2025-02-01T00:00:00+00:00")
        fetched = await store.get_by_ids(["s-001"])
        assert fetched[0].status == MemoryStatus.SUPERSEDED
        assert fetched[0].superseded_by == "s-002"

    async def test_superseded_excluded_from_list_active(self, store):
        u = _make_unit(memory_id="s-001")
        await store.add_memories([u])
        await store.supersede("s-001", "s-999", updated_at="2025-02-01T00:00:00+00:00")
        ids = {u.memory_id for u in await store.list_active(UID, "test")}
        assert "s-001" not in ids


class TestUpdateImportance:
    async def test_importance_updated(self, store):
        u = _make_unit(memory_id="i-001", importance=0.5)
        await store.add_memories([u])
        await store.update_importance("i-001", 0.9, updated_at="2025-02-01T00:00:00+00:00")
        fetched = await store.get_by_ids(["i-001"])
        assert fetched[0].importance == pytest.approx(0.9)


class TestUpdateReinforcement:
    async def test_reinforcement_updated(self, store):
        u = _make_unit(memory_id="r-001", reinforcement_score=0.0)
        await store.add_memories([u])
        await store.update_reinforcement("r-001", 0.15, updated_at="2025-02-01T00:00:00+00:00")
        fetched = await store.get_by_ids(["r-001"])
        assert fetched[0].reinforcement_score == pytest.approx(0.15)


class TestMarkAccessed:
    async def test_access_count_incremented(self, store):
        u = _make_unit(memory_id="a-001", access_count=0)
        await store.add_memories([u])
        await store.mark_accessed(["a-001"], accessed_at="2025-02-01T00:00:00+00:00")
        fetched = await store.get_by_ids(["a-001"])
        assert fetched[0].access_count == 1

    async def test_last_accessed_at_set(self, store):
        u = _make_unit(memory_id="a-002")
        await store.add_memories([u])
        await store.mark_accessed(["a-002"], accessed_at="2025-02-01T12:00:00+00:00")
        fetched = await store.get_by_ids(["a-002"])
        assert fetched[0].last_accessed_at == "2025-02-01T12:00:00+00:00"

    async def test_empty_list_no_error(self, store):
        await store.mark_accessed([], accessed_at="2025-02-01T00:00:00+00:00")


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    async def test_counts(self, store):
        await store.add_memories(create_test_units())
        stats = await store.get_stats(UID, "test")
        assert stats["total"] == 6
        assert stats["active"] == 6

    async def test_active_by_type(self, store):
        await store.add_memories(create_test_units())
        stats = await store.get_stats(UID, "test")
        by_type = stats["active_by_type"]
        assert by_type["semantic"] == 2
        assert by_type["episodic"] == 1
        assert by_type["preference"] == 1
        assert by_type["project_state"] == 1
        assert by_type["procedural_observation"] == 1

    async def test_superseded_not_counted_in_active(self, store):
        await store.add_memories(create_test_units())
        await store.supersede("unit-001", "unit-002", updated_at="2025-02-01T00:00:00+00:00")
        stats = await store.get_stats(UID, "test")
        assert stats["active"] == 5
        assert stats["total"] == 6
