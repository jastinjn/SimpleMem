# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver.policy import MemoryPolicy


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


