# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.models import MemoryStatus, MemoryType

from .conftest import UID, UID2, _make_unit, create_test_units

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
        assert n == 9  # 6 primary + 2 secondary-namespace + 1 secondary-user

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
        u_a = _make_unit(memory_id="a-001", namespace="namespace_a", updated_at="2025-01-01T00:00:01+00:00")
        u_b = _make_unit(memory_id="b-001", namespace="namespace_b", updated_at="2025-01-01T00:00:02+00:00")
        await store.add_memories([u_a, u_b])
        assert all(u.namespace == "namespace_a" for u in await store.list_active(UID, "namespace_a"))
        assert all(u.namespace == "namespace_b" for u in await store.list_active(UID, "namespace_b"))

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
            memory_id="sa-001", namespace="namespace_a",
            content="PostgreSQL database", updated_at="2025-01-01T00:00:01+00:00",
        )
        u_b = _make_unit(
            memory_id="sb-001", namespace="namespace_b",
            content="PostgreSQL database", updated_at="2025-01-01T00:00:02+00:00",
        )
        await store.add_memories([u_a, u_b])
        hits = await store.search_keyword(UID, "namespace_a", "PostgreSQL")
        assert all(h.unit.namespace == "namespace_a" for h in hits)

    async def test_limit(self, store):
        await store.add_memories(create_test_units())
        hits = await store.search_keyword(UID, "test", "the", limit=2)
        assert len(hits) <= 2

    async def test_higher_importance_ranks_higher(self, store):
        low = _make_unit(
            memory_id="low-001", namespace="rank",
            content="the project database", importance=0.1,
            updated_at="2025-01-01T00:00:01+00:00",
        )
        high = _make_unit(
            memory_id="high-001", namespace="rank",
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


# ---------------------------------------------------------------------------
# Hierarchical namespace
# ---------------------------------------------------------------------------

class TestHierarchicalNamespace:
    async def test_list_active_includes_child_and_grandchild(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        grandchild = _make_unit(memory_id="g-001", namespace="proj/api/auth")
        await store.add_memories([parent, child, grandchild])
        ids = {u.memory_id for u in await store.list_active(UID, "proj")}
        assert ids == {"p-001", "c-001", "g-001"}

    async def test_exact_child_scope_excludes_parent(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        await store.add_memories([parent, child])
        ids = {u.memory_id for u in await store.list_active(UID, "proj/api")}
        assert "c-001" in ids
        assert "p-001" not in ids

    async def test_sibling_prefix_not_matched(self, store):
        u_a = _make_unit(memory_id="a-001", namespace="scope")
        u_b = _make_unit(memory_id="b-001", namespace="scope-b")
        await store.add_memories([u_a, u_b])
        ids = {u.memory_id for u in await store.list_active(UID, "scope")}
        assert "a-001" in ids
        assert "b-001" not in ids

    async def test_get_stats_aggregates_subtree(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/backend")
        await store.add_memories([parent, child])
        stats = await store.get_stats(UID, "proj")
        assert stats["total"] == 2
        assert stats["active"] == 2

    async def test_expire_stale_applies_to_subtree(self, store):
        past = "2020-01-01T00:00:00+00:00"
        parent = _make_unit(memory_id="p-001", namespace="proj", expires_at=past)
        child = _make_unit(memory_id="c-001", namespace="proj/api", expires_at=past)
        await store.add_memories([parent, child])
        count = await store.expire_stale(UID, "proj")
        assert count == 2
        # Verify both units are archived in the DB, not just counted.
        from marklymem.evolver.models import MemoryStatus
        fetched = {u.memory_id: u for u in await store.get_by_ids(["p-001", "c-001"])}
        assert fetched["p-001"].status == MemoryStatus.ARCHIVED
        assert fetched["c-001"].status == MemoryStatus.ARCHIVED

    async def test_set_type_ttl_applies_to_subtree(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        await store.add_memories([parent, child])
        count = await store.set_type_ttl(UID, "proj", MemoryType.SEMANTIC, "2030-01-01T00:00:00+00:00")
        assert count == 2

    async def test_get_namespace_analytics_aggregates_subtree(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        await store.add_memories([parent, child])
        analytics = await store.get_namespace_analytics(UID, "proj")
        assert analytics["total"] == 2

    async def test_none_scope_returns_all(self, store):
        u_a = _make_unit(memory_id="a-001", namespace="proj")
        u_b = _make_unit(memory_id="b-001", namespace="other")
        await store.add_memories([u_a, u_b])
        ids = {u.memory_id for u in await store.list_active(UID, None)}
        assert {"a-001", "b-001"} == ids

    async def test_cross_user_isolation(self, store):
        # user1's subtree must not be visible when querying under user2's identical namespace path.
        u1 = _make_unit(memory_id="u1-001", namespace="proj/api")
        u2 = _make_unit(memory_id="u2-001", namespace="proj/api", user_id=UID2)
        await store.add_memories([u1, u2])
        ids_user1 = {u.memory_id for u in await store.list_active(UID, "proj")}
        ids_user2 = {u.memory_id for u in await store.list_active(UID2, "proj")}
        assert ids_user1 == {"u1-001"}
        assert ids_user2 == {"u2-001"}

    async def test_scope_with_underscore_does_not_match_wildcard(self, store):
        # "namespace_a" contains a SQL LIKE wildcard character "_". Querying "namespace_a"
        # must NOT match "scope-a" (the "_" must be escaped as "\\_" in the LIKE pattern).
        u_underscore = _make_unit(memory_id="u-001", namespace="namespace_a",
                                  content="underscore namespace unit")
        u_dash = _make_unit(memory_id="d-001", namespace="scope-a",
                            content="dash namespace unit")
        await store.add_memories([u_underscore, u_dash])
        ids_underscore = {u.memory_id for u in await store.list_active(UID, "namespace_a")}
        ids_dash = {u.memory_id for u in await store.list_active(UID, "scope-a")}
        assert "u-001" in ids_underscore
        assert "d-001" not in ids_underscore
        assert "d-001" in ids_dash
        assert "u-001" not in ids_dash


# ---------------------------------------------------------------------------
# list_active_exact — consolidation uses this to avoid cross-subtree merging
# ---------------------------------------------------------------------------

class TestListActiveExact:
    async def test_excludes_child_scopes(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        await store.add_memories([parent, child])
        ids = {u.memory_id for u in await store.list_active_exact(UID, "proj")}
        assert ids == {"p-001"}

    async def test_exact_child_scope_returns_only_that_scope(self, store):
        parent = _make_unit(memory_id="p-001", namespace="proj")
        child = _make_unit(memory_id="c-001", namespace="proj/api")
        await store.add_memories([parent, child])
        ids = {u.memory_id for u in await store.list_active_exact(UID, "proj/api")}
        assert ids == {"c-001"}

    async def test_none_scope_returns_all(self, store):
        u_a = _make_unit(memory_id="a-001", namespace="proj")
        u_b = _make_unit(memory_id="b-001", namespace="proj/api")
        await store.add_memories([u_a, u_b])
        ids = {u.memory_id for u in await store.list_active_exact(UID, None)}
        assert {"a-001", "b-001"} == ids
