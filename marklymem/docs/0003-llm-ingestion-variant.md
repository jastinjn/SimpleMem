# LLM-based ingestion as an alternative to pattern extraction

## Context

The evolver engine's original ingestion path extracts memory units turn-by-turn using regex and keyword pattern matching. This is fast and deterministic but brittle — it relies on surface-level signals to classify content and assign memory types, and will miss facts that are expressed in ways the patterns do not anticipate. The AI practice team's original SimpleMem experiment combined the SimpleMem core ingestion module (LLM-based extraction) with the evolver's retrieval and consolidation layer, validating that the two can work together. That combination was not carried into the evolver server directly, leaving an opportunity to bring the higher-quality extraction path forward.

## Decision

Add an LLM-based ingestion variant (`ingestion_mode=llm`) implemented in `llm_extractor.py`, adapted from SimpleMem core for the evolver's async, direct-call world. Rather than processing turns one at a time, it segments the session into overlapping windows of up to 15 turns and sends each window concurrently to OpenAI using structured outputs (`client.responses.parse` with a Pydantic schema). The LLM infers `memory_type`, `importance`, and `confidence` per unit, populating the evolver's weighting fields at write time. `ingestion_mode=llm` is the default. `ingestion_mode=pattern` remains available as an explicit choice for environments where latency or cost matter more than extraction quality.

## Considered Options

- **Pattern extraction only.** Fast and requires no API key, but misses implicitly stated facts and produces low-quality type and importance assignments. The AI practice team's experiment showed the LLM path produces meaningfully better retrieval results. Rejected as the primary mode.
- **Use the original SimpleMem core ingestion module unchanged.** SimpleMem core's extraction targets different output fields — it has no concept of `memory_type`, `importance`, or `confidence`, which are first-class columns in the evolver schema used by retrieval scoring and consolidation. Additionally, the module is synchronous and buffered through a dialogue queue, and was not designed for the evolver's async per-session call model. Both differences require a clean reimplementation rather than a thin wrapper. Rejected.
- **LLM extraction adapted for the evolver, with pattern as a peer mode.** Reimplements the sliding-window LLM extraction idea from SimpleMem core as a self-contained async module that natively returns `MemoryUnit` objects with `memory_type`, `importance`, and `confidence` populated. Preserves the evolver's dedup, embedding, and consolidation pipeline unchanged — only the extraction step differs. Both modes are first-class: `llm` is the default; `pattern` is an explicit opt-in for low-latency or keyless environments. Accepted.

## Consequences

- `ingestion_mode=llm` requires `OPENAI_API_KEY`; the server will refuse to start without it when this mode is set.
- Memory units produced by the LLM path carry calibrated `importance` and `confidence` scores and one of four assignable types (`preference`, `procedural_observation`, `semantic`, `episodic`), making retrieval ranking more accurate than fixed defaults.
- Windows are processed concurrently (bounded by `max_parallel=4`), so ingestion latency scales with the slowest window rather than the total number of turns.
- `ingestion_mode=pattern` is an explicit peer mode, not a fallback — callers choose it deliberately when extraction latency or API cost is a priority over quality.
