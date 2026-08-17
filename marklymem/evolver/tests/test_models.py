# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.models import (
    MemoryQuery,
    MemorySearchHit,
    MemoryStatus,
    MemoryType,
    MemoryUnit,
)


class TestMemoryUnit:
    def test_defaults(self):
        u = MemoryUnit(
            memory_id="m1",
            user_id="user-test",
            namespace="s1",
            memory_type=MemoryType.SEMANTIC,
            content="some content",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        assert u.importance == 0.5
        assert u.confidence == 0.7
        assert u.access_count == 0
        assert u.reinforcement_score == 0.0
        assert u.status == MemoryStatus.ACTIVE
        assert u.superseded_by == ""
        assert u.last_accessed_at == ""
        assert u.expires_at == ""

    def test_list_fields_are_independent(self):
        u1 = MemoryUnit(
            memory_id="m1", user_id="user-test", namespace="s", memory_type=MemoryType.SEMANTIC,
            content="a", created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        u2 = MemoryUnit(
            memory_id="m2", user_id="user-test", namespace="s", memory_type=MemoryType.SEMANTIC,
            content="b", created_at="2025-01-01T00:00:01+00:00",
            updated_at="2025-01-01T00:00:01+00:00",
        )
        u1.entities.append("X")
        assert "X" not in u2.entities

    def test_explicit_timestamps_preserved(self):
        u = MemoryUnit(
            memory_id="m1", user_id="user-test", namespace="s", memory_type=MemoryType.EPISODIC,
            content="c",
            created_at="2025-06-01T10:00:00+00:00",
            updated_at="2025-06-02T10:00:00+00:00",
        )
        assert u.created_at == "2025-06-01T10:00:00+00:00"
        assert u.updated_at == "2025-06-02T10:00:00+00:00"


class TestMemoryQuery:
    def test_defaults(self):
        q = MemoryQuery(user_id="user-test", namespace="s", query_text="hello")
        assert q.top_k == 6
        assert q.max_tokens == 800
        assert q.include_types == []

    def test_custom_values(self):
        q = MemoryQuery(user_id="user-test", namespace="s", query_text="x", top_k=3, max_tokens=200)
        assert q.top_k == 3
        assert q.max_tokens == 200


class TestMemorySearchHit:
    def test_defaults(self):
        u = MemoryUnit(
            memory_id="m1", user_id="user-test", namespace="s", memory_type=MemoryType.SEMANTIC,
            content="c", created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        h = MemorySearchHit(unit=u, score=0.5)
        assert h.matched_terms == []
        assert h.reason == ""


class TestMemoryTypeEnum:
    def test_values(self):
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.PROJECT_STATE.value == "project_state"
        assert MemoryType.PROCEDURAL_OBSERVATION.value == "procedural_observation"


class TestMemoryStatusEnum:
    def test_values(self):
        assert MemoryStatus.ACTIVE.value == "active"
        assert MemoryStatus.SUPERSEDED.value == "superseded"
        assert MemoryStatus.ARCHIVED.value == "archived"
