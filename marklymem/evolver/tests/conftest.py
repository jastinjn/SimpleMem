# pyright: reportMissingImports=false
"""Shared helpers and fixtures for the evolver unit-test suite.

Determinism strategy
--------------------
Three sources of nondeterminism are pinned throughout this suite:

1. Wall-clock time
   - store._utc_now_iso        – prefer passing explicit updated_at/accessed_at
   - manager.utc_now_iso       – monkeypatched in test_manager.py
   - retriever.datetime        – monkeypatched via `frozen_retriever_clock` fixture;
                                  or neutralised with recency_weight=0
   - consolidator.datetime     – monkeypatched via `frozen_consolidator_clock` fixture;
                                  or neutralised with decay_factor=0
   - models.utc_now_iso        – a dataclass default_factory bound at class-definition
                                  time; cannot be patched. Always pass explicit
                                  created_at/updated_at when constructing MemoryUnit.

2. UUIDs
   - manager.uuid.uuid4        – monkeypatched in test_manager.py with a counter fake.
   - Store-level tests never depend on uuid (units built with explicit ids).

3. Sort-tie nondeterminism
   - All retriever modes sort by (score, unit.updated_at) reverse=True.
   - list_active has no secondary tiebreaker.
   - create_test_units() gives every unit a DISTINCT updated_at so ordering is total.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: F401 — used by callers

from marklymem.evolver.models import MemoryType, MemoryUnit
from marklymem.evolver.store import MemoryStore
from marklymem.tests.utils.db import OTHER_SCOPE as SCOPE_B  # noqa: F401
from marklymem.tests.utils.db import OTHER_USER as UID2  # noqa: F401
from marklymem.tests.utils.db import USER_ID as UID  # noqa: F401
from marklymem.tests.utils.db import create_test_units  # noqa: F401

# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------

def _make_store(sm: async_sessionmaker) -> MemoryStore:
    return MemoryStore(sm)


# ---------------------------------------------------------------------------
# Single-unit factory
# ---------------------------------------------------------------------------

_unit_counter = 0


def _make_unit(**overrides) -> MemoryUnit:
    """Build a single MemoryUnit with safe explicit defaults."""
    global _unit_counter
    _unit_counter += 1
    n = _unit_counter
    ts = f"2025-03-01T00:00:{n:02d}+00:00"
    defaults = dict(
        memory_id=f"mu-{n:04d}",
        user_id=UID,
        scope_id="test",
        memory_type=MemoryType.SEMANTIC,
        content=f"Memory unit content number {n}",
        entities=[],
        topics=[],
        tags=[],
        importance=0.5,
        confidence=0.7,
        created_at=ts,
        updated_at=ts,
    )
    defaults.update(overrides)
    return MemoryUnit(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FrozenClock – replaces the `datetime` class in a module under test
# ---------------------------------------------------------------------------

class FrozenClock:
    """Fake for the `datetime` class; returns a fixed instant from now()."""

    def __init__(self, fixed_now: datetime):
        self._now = fixed_now

    def now(self, tz=None) -> datetime:
        return self._now

    @staticmethod
    def fromisoformat(s: str) -> datetime:
        return datetime.fromisoformat(s)

    @staticmethod
    def replace(*args, **kwargs):
        return datetime.replace(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FROZEN_NOW = datetime(2025, 2, 15, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def frozen_retriever_clock(monkeypatch):
    """Patch retriever.datetime so _estimate_recency_bonus uses a fixed now."""
    clock = FrozenClock(FROZEN_NOW)
    monkeypatch.setattr("marklymem.evolver.retriever.datetime", clock)
    return clock


@pytest.fixture
def frozen_consolidator_clock(monkeypatch):
    """Patch consolidator.datetime so decay uses a fixed now."""
    clock = FrozenClock(FROZEN_NOW)
    monkeypatch.setattr("marklymem.evolver.consolidator.datetime", clock)
    return clock


@pytest_asyncio.fixture()
async def store(test_sm):
    """A fresh MemoryStore backed by the test DB (tables truncated by root conftest)."""
    return MemoryStore(test_sm)


@pytest.fixture
def fake_uuid(monkeypatch):
    """Replace manager.uuid with a deterministic counter-based fake."""
    import uuid as _real_uuid

    class _FakeUUID:
        _counter = 0

        @classmethod
        def uuid4(cls):
            cls._counter += 1
            return _real_uuid.UUID(f"00000000-0000-0000-0000-{cls._counter:012d}")

    _FakeUUID._counter = 0
    monkeypatch.setattr("marklymem.evolver.manager.uuid", _FakeUUID)
    return _FakeUUID
