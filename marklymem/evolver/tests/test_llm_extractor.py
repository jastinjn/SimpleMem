# pyright: reportMissingImports=false
"""Unit tests for the LLM ingestion extractor (``evolver/llm_extractor.py``).

Pure unit tests — no network, no database. The LLM call is replaced by an
in-process async fake, so these deterministically exercise:

* window segmentation (single / multiple / empty, overlap, 1-based bounds),
* dialogue rendering,
* entry → MemoryUnit field mapping (type, scores, topics, entities, bounds,
  content/summary truncation, short-restatement filtering),
* score coercion / clamping,
* retry logic (transient error, empty entries, exhausted retries),
* per-window error isolation, and
* bounded concurrency (max_parallel).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from marklymem.evolver import llm_extractor as le
from marklymem.evolver.llm_extractor import (
    AssignableMemoryType,
    ExtractedEntries,
    ExtractedEntry,
    LLMExtractionConfig,
    LLMMemoryExtractor,
    _coerce_score,
)
from marklymem.evolver.models import MemoryType


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _entry(
    text: str = "Alice scored Level 4 on the essay task",
    mtype: AssignableMemoryType = AssignableMemoryType.EPISODIC,
    importance: float = 0.7,
    confidence: float = 0.8,
    persons: list[str] | None = None,
    entities: list[str] | None = None,
    topics: list[str] | None = None,
) -> ExtractedEntry:
    return ExtractedEntry(
        lossless_restatement=text,
        memory_type=mtype,
        importance=importance,
        confidence=confidence,
        persons=persons or [],
        entities=entities or [],
        topics=topics or [],
    )


def _canned(entries: list[ExtractedEntry]):
    """Async fake that always returns the same entries."""
    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        return ExtractedEntries(entries=list(entries))

    return fake


async def _never_called(instructions: str, dialogue_text: str) -> ExtractedEntries:
    raise AssertionError("llm_acall should not be invoked")


def _patch_sleep(monkeypatch):
    """Neutralise retry backoff so retry tests run instantly."""
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(le.asyncio, "sleep", _no_sleep)


def _turns(n: int) -> list[dict]:
    """n turns whose rendered dialogue comfortably exceeds the 30-char floor."""
    return [
        {
            "prompt_text": f"prompt number {i} with enough words here",
            "response_text": f"response number {i} with enough words here",
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Window segmentation
# --------------------------------------------------------------------------- #
def test_segment_single_window_when_fewer_than_window_size():
    ex = LLMMemoryExtractor(_never_called, LLMExtractionConfig(window_size=15, overlap=2))
    windows = ex._segment_windows(_turns(5))
    assert len(windows) == 1
    window, start, end = windows[0]
    assert (start, end, len(window)) == (1, 5, 5)


def test_segment_multiple_windows_with_overlap():
    ex = LLMMemoryExtractor(_never_called, LLMExtractionConfig(window_size=3, overlap=1))
    windows = ex._segment_windows(_turns(7))
    # step = window_size - overlap = 2 → starts at turns 1, 3, 5, 7 (1-based).
    assert [(s, e) for _, s, e in windows] == [(1, 3), (3, 5), (5, 7), (7, 7)]


def test_segment_no_overlap():
    ex = LLMMemoryExtractor(_never_called, LLMExtractionConfig(window_size=3, overlap=0))
    windows = ex._segment_windows(_turns(6))
    assert [(s, e) for _, s, e in windows] == [(1, 3), (4, 6)]


def test_segment_empty_turns():
    ex = LLMMemoryExtractor(_never_called)
    assert ex._segment_windows([]) == []


# --------------------------------------------------------------------------- #
# Dialogue rendering
# --------------------------------------------------------------------------- #
def test_render_dialogue_formats_and_skips_empty_sides():
    ex = LLMMemoryExtractor(_never_called)
    text = ex._render_dialogue(
        [
            {"prompt_text": "Hello", "response_text": "Hi there"},
            {"prompt_text": "", "response_text": "Only assistant"},
            {"prompt_text": "Only user", "response_text": ""},
            {},
        ]
    )
    assert text == (
        "User: Hello\n"
        "Assistant: Hi there\n"
        "Assistant: Only assistant\n"
        "User: Only user"
    )


# --------------------------------------------------------------------------- #
# extract_session — trivial cases
# --------------------------------------------------------------------------- #
async def test_extract_session_no_turns_returns_empty():
    ex = LLMMemoryExtractor(_never_called)
    assert await ex.extract_session([], "u", "s", "sess") == []


async def test_tiny_window_skips_llm_call():
    # "User: hi" is 8 chars, below the 30-char floor → llm_acall never invoked.
    ex = LLMMemoryExtractor(_never_called, LLMExtractionConfig(window_size=1))
    units = await ex.extract_session([{"prompt_text": "hi", "response_text": ""}], "u", "s", "sess")
    assert units == []


# --------------------------------------------------------------------------- #
# Field mapping
# --------------------------------------------------------------------------- #
async def test_field_mapping_from_entry():
    entry = _entry(
        text="Alice scored Level 4 on the persuasive essay task",
        mtype=AssignableMemoryType.SEMANTIC,
        importance=0.9,
        confidence=0.85,
        persons=["Alice"],
        entities=["Persuasive Essay"],
        topics=["essay", "writing"],
    )
    ex = LLMMemoryExtractor(_canned([entry]))
    turns = [{"prompt_text": "How did Alice do on the essay?", "response_text": "Alice scored Level 4."}]

    units = await ex.extract_session(turns, "user-1", "scope-1", "sess-9")

    assert len(units) == 1
    u = units[0]
    assert u.memory_type == MemoryType.SEMANTIC
    assert u.importance == 0.9
    assert u.confidence == 0.85
    assert u.topics == ["essay", "writing"]
    # entities = persons + entities, in that order.
    assert u.entities == ["Alice", "Persuasive Essay"]
    assert u.user_id == "user-1"
    assert u.namespace == "scope-1"
    assert u.source_session_id == "sess-9"
    assert (u.source_turn_start, u.source_turn_end) == (1, 1)
    assert u.content == "Alice scored Level 4 on the persuasive essay task"


@pytest.mark.parametrize(
    "atype,expected",
    [
        (AssignableMemoryType.PREFERENCE, MemoryType.PREFERENCE),
        (AssignableMemoryType.PROCEDURAL_OBSERVATION, MemoryType.PROCEDURAL_OBSERVATION),
        (AssignableMemoryType.SEMANTIC, MemoryType.SEMANTIC),
        (AssignableMemoryType.EPISODIC, MemoryType.EPISODIC),
    ],
)
async def test_all_assignable_types_map_to_memory_type(atype, expected):
    ex = LLMMemoryExtractor(_canned([_entry(text="Mapping restatement content goes here", mtype=atype)]))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")
    assert units[0].memory_type == expected


async def test_content_kept_full():
    long_text = ("word " * 700).strip()  # 700 words, ~3499 chars
    ex = LLMMemoryExtractor(_canned([_entry(text=long_text)]))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")
    assert units[0].content == long_text


async def test_short_restatements_are_dropped():
    entries = [
        _entry(text="Too short here"),  # 3 words < min_restatement_words (4) → dropped
        _entry(text="This restatement has enough words"),  # 5 words → kept
    ]
    ex = LLMMemoryExtractor(_canned(entries))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")
    assert len(units) == 1
    assert units[0].content == "This restatement has enough words"


async def test_scores_are_clamped_into_unit_range():
    entry = _entry(text="A perfectly valid restatement here", importance=1.8, confidence=-0.5)
    ex = LLMMemoryExtractor(_canned([entry]))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")
    assert units[0].importance == 1.0
    assert units[0].confidence == 0.0


# --------------------------------------------------------------------------- #
# _coerce_score
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,default,expected",
    [
        (0.5, 0.5, 0.5),
        (1.5, 0.5, 1.0),
        (-0.2, 0.5, 0.0),
        ("0.3", 0.5, 0.3),
        ("not-a-number", 0.4, 0.4),
        (None, 0.6, 0.6),
        (float("nan"), 0.7, 0.7),
    ],
)
def test_coerce_score(value, default, expected):
    assert _coerce_score(value, default) == expected


# --------------------------------------------------------------------------- #
# Multiple windows
# --------------------------------------------------------------------------- #
async def test_multiple_windows_combined_with_correct_bounds():
    calls: list[str] = []

    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        calls.append(dialogue_text)
        return ExtractedEntries(entries=[_entry(text=f"Window restatement number {len(calls)} here")])

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(window_size=3, overlap=1))
    units = await ex.extract_session(_turns(7), "u", "s", "sess")

    assert len(calls) == 4
    assert len(units) == 4
    assert sorted((u.source_turn_start, u.source_turn_end) for u in units) == [
        (1, 3),
        (3, 5),
        (5, 7),
        (7, 7),
    ]


# --------------------------------------------------------------------------- #
# Retry logic
# --------------------------------------------------------------------------- #
async def test_retry_recovers_from_transient_error(monkeypatch):
    _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return ExtractedEntries(entries=[_entry(text="Recovered restatement after two retries")])

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(max_retries=3))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")

    assert attempts["n"] == 3
    assert len(units) == 1


async def test_retry_on_empty_entries(monkeypatch):
    _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return ExtractedEntries(entries=[])
        return ExtractedEntries(entries=[_entry(text="Second attempt restatement content here")])

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(max_retries=3))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")

    assert attempts["n"] == 2
    assert len(units) == 1


async def test_all_none_yields_no_units(monkeypatch):
    _patch_sleep(monkeypatch)

    async def fake(instructions: str, dialogue_text: str) -> None:
        return None

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(max_retries=2))
    units = await ex.extract_session(_turns(1), "u", "s", "sess")
    assert units == []


# --------------------------------------------------------------------------- #
# Error isolation across windows
# --------------------------------------------------------------------------- #
async def test_failed_window_is_isolated(monkeypatch):
    _patch_sleep(monkeypatch)

    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        if "number 0" in dialogue_text:  # only the first window (turns 1-3)
            raise RuntimeError("boom")
        return ExtractedEntries(entries=[_entry(text="Good window restatement content here")])

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(window_size=3, overlap=0, max_retries=2))
    units = await ex.extract_session(_turns(6), "u", "s", "sess")

    # First window fails after retries and is dropped; the second still yields a unit.
    assert len(units) == 1
    assert (units[0].source_turn_start, units[0].source_turn_end) == (4, 6)


# --------------------------------------------------------------------------- #
# Bounded concurrency
# --------------------------------------------------------------------------- #
async def test_respects_max_parallel():
    state = {"current": 0, "peak": 0}

    async def fake(instructions: str, dialogue_text: str) -> ExtractedEntries:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.01)  # hold the slot so overlap is observable
        state["current"] -= 1
        return ExtractedEntries(entries=[_entry(text="Concurrency restatement content here")])

    ex = LLMMemoryExtractor(fake, LLMExtractionConfig(window_size=1, overlap=0, max_parallel=2))
    units = await ex.extract_session(_turns(6), "u", "s", "sess")

    assert len(units) == 6
    # The semaphore caps concurrency at 2; with 6 windows it should reach exactly 2.
    assert state["peak"] == 2
