"""LLM-based memory extraction for the evolver write pipeline.

This is the ``ingestion_mode="llm"`` counterpart to the regex/keyword extraction
in :mod:`evolver_server.evolver.manager`. Instead of matching hard-coded patterns
turn by turn, it sends *windows* of the session to an LLM and receives structured
memory entries back — the same idea as :mod:`simplemem.core.memory_builder`,
adapted to the evolver's async, direct-call world and richer :class:`MemoryUnit`
(importance / confidence / typed memory).

Key differences from ``simplemem/core/memory_builder.py``:
- No queued dialogue buffer — ``extract_session`` is a direct async call over a
  completed session's turns.
- Windows are processed **concurrently** (bounded by a semaphore).
- The LLM additionally infers ``memory_type``, ``importance`` and ``confidence``
  for each unit so the evolver's weighting fields are populated at write time.
- Uses the OpenAI Responses API with **Pydantic structured outputs**
  (``client.responses.parse`` + ``text_format``), so parsing is schema-guaranteed
  rather than best-effort JSON repair.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from .models import MemoryType, MemoryUnit

logger = logging.getLogger(__name__)


class AssignableMemoryType(str, Enum):
    """Memory types the LLM is allowed to assign.

    Mirrors :class:`MemoryType` minus ``working_summary`` (which the manager
    synthesises separately from the whole session) and ``project_state``. Used as
    the structured-output field type so the JSON schema constrains the model to
    exactly these values.
    """

    PREFERENCE = "preference"
    PROCEDURAL_OBSERVATION = "procedural_observation"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class ExtractedEntry(BaseModel):
    """One structured memory entry as returned by the LLM."""

    lossless_restatement: str = Field(
        description=(
            "Exactly ONE complete, self-contained sentence with all subjects, "
            "objects, time, and location. Never more than one sentence."
        )
    )
    memory_type: AssignableMemoryType = Field(
        description="The kind of memory this entry represents."
    )
    importance: float = Field(description="0.0-1.0 value of this memory for future recall.")
    confidence: float = Field(description="0.0-1.0 certainty of the statement given the dialogue.")
    persons: list[str] = Field(
        default_factory=list, description="People mentioned (e.g. student or teacher names)."
    )
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Class names, syllabus topics, skills assessed."
        ),
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Single-word topic keywords.",
    )


class ExtractedEntries(BaseModel):
    """Container the LLM returns — a list of extracted entries."""

    entries: list[ExtractedEntry] = Field(default_factory=list)


# Async callable: (instructions, dialogue_text) -> parsed entries (or None on no output).
LLMParseCall = Callable[[str, str], Awaitable["ExtractedEntries | None"]]


EXTRACTION_INSTRUCTIONS = """
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

Requirements:
1. Complete Coverage: Generate entries for ALL facts, events, opinions, plans,
   preferences, instructions, and expectations expressed in the dialogue.
2. Force Disambiguation: PROHIBIT pronouns (he/she/it/they/this/that) and relative
   time (yesterday/today/last week). Use actual names and absolute dates.
3. Lossless Restatement: Each entry must be exactly ONE complete, independent
   sentence that stands on its own without the surrounding dialogue. If a fact
   needs more than one sentence, split it into multiple separate entries.
4. For each entry, infer three weighting fields:
   - memory_type: EXACTLY one of:
       * "preference" — a stable like/dislike, style, or convention the user holds.
       * "procedural_observation" — a rule/workflow/instruction on how to do things
         ("always ...", "never ...", "make sure ...").
       * "semantic" — a durable fact or piece of knowledge worth remembering.
       * "episodic" — a specific event, action, or exchange tied to this session.
   - importance: float 0.0-1.0. How valuable is this for future recall?
     (stable preferences/instructions high ~0.8-0.9; incidental chatter low ~0.3.)
   - confidence: float 0.0-1.0. How certain is the statement given the dialogue?
"""


@dataclass
class LLMExtractionConfig:
    """Configuration for LLM-based session extraction."""

    window_size: int = 15
    overlap: int = 2
    max_parallel: int = 4
    model: str = "gpt-4.1-mini"
    max_retries: int = 3
    min_restatement_words: int = 4


def _coerce_score(value: object, default: float) -> float:
    """Parse and clamp an importance/confidence score to [0.0, 1.0]."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if score != score:  # NaN guard
        return default
    return max(0.0, min(1.0, score))


class LLMMemoryExtractor:
    """Extract memory units from a completed session using an LLM.

    Args:
        llm_acall: Async callable ``(instructions, dialogue_text) -> ExtractedEntries``
            returning the parsed structured output (or ``None`` when the model
            produced nothing). Decouples extraction from any specific provider and
            makes the class trivial to fake in tests.
        config: :class:`LLMExtractionConfig` controlling windowing and the LLM call.
    """

    def __init__(self, llm_acall: LLMParseCall, config: LLMExtractionConfig | None = None):
        self.llm_acall = llm_acall
        self.config = config or LLMExtractionConfig()

    async def extract_session(
        self,
        turns: list[dict],
        user_id: str,
        scope_id: str,
        session_id: str,
    ) -> list[MemoryUnit]:
        """Extract memory units from all turns of one session.

        Windows are processed concurrently (bounded by ``max_parallel``). A window
        whose LLM call fails or returns nothing contributes no units and is logged;
        it never aborts the rest of the session.
        """
        windows = self._segment_windows(turns)
        if not windows:
            return []

        sem = asyncio.Semaphore(max(1, self.config.max_parallel))

        async def _run(window: list[dict], start: int, end: int) -> list[MemoryUnit]:
            async with sem:
                return await self._extract_window(
                    window, start, end, user_id, scope_id, session_id
                )

        results = await asyncio.gather(
            *(_run(w, s, e) for (w, s, e) in windows),
            return_exceptions=True,
        )

        units: list[MemoryUnit] = []
        for res in results:
            if isinstance(res, BaseException):
                logger.warning("[LLM extract] window failed: %s", res)
                continue
            units.extend(res)

        logger.info(
            "[LLM extract] session=%s → %d units from %d window(s)",
            session_id, len(units), len(windows),
        )
        return units

    # ---- Internal methods ----

    def _segment_windows(self, turns: list[dict]) -> list[tuple[list[dict], int, int]]:
        """Sliding-window segmentation with overlap.

        Returns tuples of ``(window_turns, start_index, end_index)`` where the
        indices are 1-based turn positions within the session (for
        ``source_turn_start`` / ``source_turn_end`` on the resulting units).
        """
        ws = max(1, self.config.window_size)
        step = max(1, ws - self.config.overlap)
        windows: list[tuple[list[dict], int, int]] = []
        i = 0
        n = len(turns)
        while i < n:
            window = turns[i:i + ws]
            if window:
                windows.append((window, i + 1, i + len(window)))
            i += step
        return windows

    def _render_dialogue(self, turns: list[dict]) -> str:
        lines: list[str] = []
        for t in turns:
            prompt = str(t.get("prompt_text", "") or "").strip()
            response = str(t.get("response_text", "") or "").strip()
            if prompt:
                lines.append(f"User: {prompt}")
            if response:
                lines.append(f"Assistant: {response}")
        return "\n".join(lines)

    async def _extract_window(
        self,
        turns: list[dict],
        start: int,
        end: int,
        user_id: str,
        scope_id: str,
        session_id: str,
    ) -> list[MemoryUnit]:
        dialogue_text = self._render_dialogue(turns)
        if len(dialogue_text.strip()) < 30:
            return []

        for attempt in range(self.config.max_retries):
            try:
                parsed = await self.llm_acall(EXTRACTION_INSTRUCTIONS, dialogue_text)
            except Exception as exc:  # noqa: BLE001 - retry on transient LLM errors
                logger.debug(
                    "[LLM extract] window %d-%d attempt %d error: %s",
                    start, end, attempt + 1, exc,
                )
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                continue

            entries = parsed.entries if parsed is not None else []
            if entries:
                return self._entries_to_units(
                    entries, start, end, user_id, scope_id, session_id
                )
            logger.debug(
                "[LLM extract] window %d-%d attempt %d: no entries",
                start, end, attempt + 1,
            )

        return []

    def _entries_to_units(
        self,
        entries: list[ExtractedEntry],
        start: int,
        end: int,
        user_id: str,
        scope_id: str,
        session_id: str,
    ) -> list[MemoryUnit]:
        units: list[MemoryUnit] = []
        for entry in entries:
            content = (entry.lossless_restatement or "").strip()
            if len(content.split()) < self.config.min_restatement_words:
                continue

            # memory_type is a schema-constrained enum whose values are a subset of
            # MemoryType, so this conversion is always valid.
            mtype = MemoryType(entry.memory_type.value)
            importance = _coerce_score(entry.importance, 0.75)
            confidence = _coerce_score(entry.confidence, 0.75)
            topics = list(entry.topics)

            units.append(
                MemoryUnit(
                    memory_id=str(uuid.uuid4()),
                    user_id=user_id,
                    scope_id=scope_id,
                    memory_type=mtype,
                    content=content,
                    # summary left blank: the restatement is already a single
                    # self-contained sentence, so a separate summary would just
                    # duplicate the content (and double-weight it in retrieval).
                    summary="",
                    source_session_id=session_id,
                    source_turn_start=start,
                    source_turn_end=end,
                    topics=topics,
                    entities=list(entry.persons) + list(entry.entities),
                    importance=importance,
                    confidence=confidence,
                )
            )
        return units


def create_llm_extractor(settings) -> LLMMemoryExtractor | None:
    """Build an :class:`LLMMemoryExtractor` backed by ``AsyncOpenAI``.

    Returns ``None`` when no OpenAI API key is configured, so callers can decide
    whether the missing key is fatal (see ``app.py`` lifespan).
    """
    if not getattr(settings, "OPENAI_API_KEY", ""):
        return None

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "openai package not installed. Install with: pip install openai"
        ) from exc

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Windowing / model / temperature / max-tokens all default from LLMExtractionConfig
    # (and the literals in _llm_acall) — they are not server Settings.
    config = LLMExtractionConfig()

    async def _llm_acall(instructions: str, dialogue_text: str) -> ExtractedEntries | None:
        response = await client.responses.parse(
            model=config.model,
            instructions=instructions,
            input=dialogue_text,
            text_format=ExtractedEntries,
            temperature=0.1,
            max_output_tokens=2048,
        )
        return response.output_parsed

    return LLMMemoryExtractor(_llm_acall, config)
