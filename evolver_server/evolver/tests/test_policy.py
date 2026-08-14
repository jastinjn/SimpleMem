# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evolver_server.evolver.policy import MemoryPolicy
from evolver_server.evolver.policy_store import MemoryPolicyState


class TestMemoryPolicyDefaults:
    def test_default_weights(self):
        p = MemoryPolicy()
        assert p.max_injected_units == 6
        assert p.max_injected_tokens == 800
        assert p.recent_bonus_hours == 72
        assert p.keyword_weight == pytest.approx(1.0)
        assert p.metadata_weight == pytest.approx(0.45)
        assert p.importance_weight == pytest.approx(0.5)
        assert p.recency_weight == pytest.approx(0.3)

    def test_default_type_boosts(self):
        p = MemoryPolicy()
        assert p.type_boosts["project_state"] == pytest.approx(1.1)
        assert p.type_boosts["preference"] == pytest.approx(1.0)
        assert p.type_boosts["semantic"] == pytest.approx(1.0)
        assert p.type_boosts["episodic"] == pytest.approx(0.8)
        assert p.type_boosts["procedural_observation"] == pytest.approx(0.9)


class TestMemoryPolicyFromProfile:
    def test_balanced_is_defaults(self):
        p = MemoryPolicy.from_profile("balanced")
        d = MemoryPolicy()
        assert p.max_injected_units == d.max_injected_units
        assert p.keyword_weight == pytest.approx(d.keyword_weight)

    def test_recall(self):
        p = MemoryPolicy.from_profile("recall")
        assert p.max_injected_units == 10
        assert p.max_injected_tokens == 1200
        assert p.keyword_weight == pytest.approx(0.8)
        assert p.metadata_weight == pytest.approx(0.6)
        assert p.importance_weight == pytest.approx(0.3)
        assert p.recency_weight == pytest.approx(0.2)

    def test_precision(self):
        p = MemoryPolicy.from_profile("precision")
        assert p.max_injected_units == 4
        assert p.max_injected_tokens == 500
        assert p.keyword_weight == pytest.approx(1.2)
        assert p.metadata_weight == pytest.approx(0.3)
        assert p.importance_weight == pytest.approx(0.7)
        assert p.recency_weight == pytest.approx(0.1)

    def test_recent(self):
        p = MemoryPolicy.from_profile("recent")
        assert p.recency_weight == pytest.approx(0.8)
        assert p.recent_bonus_hours == 24

    def test_unknown_profile_returns_defaults(self):
        p = MemoryPolicy.from_profile("nonexistent_profile")
        d = MemoryPolicy()
        assert p.max_injected_units == d.max_injected_units
        assert p.keyword_weight == pytest.approx(d.keyword_weight)


class TestMemoryPolicyFromState:
    def test_from_state_maps_fields(self):
        state = MemoryPolicyState(
            max_injected_units=8,
            max_injected_tokens=1000,
            recent_bonus_hours=48,
            keyword_weight=1.5,
            metadata_weight=0.6,
            importance_weight=0.4,
            recency_weight=0.2,
        )
        p = MemoryPolicy.from_state(state)
        assert p.max_injected_units == 8
        assert p.max_injected_tokens == 1000
        assert p.recent_bonus_hours == 48
        assert p.keyword_weight == pytest.approx(1.5)
        assert p.metadata_weight == pytest.approx(0.6)
        assert p.importance_weight == pytest.approx(0.4)
        assert p.recency_weight == pytest.approx(0.2)

    def test_from_state_with_type_boosts(self):
        state = MemoryPolicyState(
            type_boosts={"semantic": 1.5, "episodic": 0.5}
        )
        p = MemoryPolicy.from_state(state)
        assert p.type_boosts["semantic"] == pytest.approx(1.5)
        assert p.type_boosts["episodic"] == pytest.approx(0.5)

    def test_from_state_empty_type_boosts_uses_defaults(self):
        state = MemoryPolicyState(type_boosts={})
        p = MemoryPolicy.from_state(state)
        # When type_boosts is empty, from_state does not pass it → defaults apply.
        d = MemoryPolicy()
        assert p.type_boosts == d.type_boosts
