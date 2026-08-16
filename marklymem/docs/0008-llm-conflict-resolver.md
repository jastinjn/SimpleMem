# Two-stage LLM conflict resolver over Jaccard-only resolution

## Context

marklymem inherits `auto_resolve_conflicts` from the upstream simplemem evolver. That method detects conflicts between newly ingested memories and the active pool using Jaccard similarity over `topics + entities`, gated on identical `memory_type`, with a threshold of 0.80. It is fast and deterministic and requires no API key.

During testing against real dialogue, the Jaccard path produced systematic false positives: memories that merely shared domain vocabulary (e.g. two procedural rules both mentioning "feedback", "criteria", and "marking") were treated as conflicts and the older unit was superseded, destroying valid information. At the same time, genuine contradictions — pairs where a fact was directly reversed across two ingest calls — were missed because they shared little surface text or differed in `memory_type`, preventing them from clearing the same-type gate.

Both failure modes stem from the same root cause: token-overlap is the wrong signal for contradiction. Shared vocabulary indicates topic proximity, not semantic conflict; the absence of shared vocabulary does not rule out a factual reversal.

## Decision

Replace the ingest-time Jaccard path with a two-stage pipeline implemented in `evolver/resolver.py`:

**Stage 1 — recall.** Cosine similarity over embeddings finds candidate pairs above a threshold (default 0.65). No `memory_type` gate — cross-type contradictions are candidates. When embeddings are unavailable for a pool unit, Jaccard over `topics + entities` is used as a per-unit fallback (same-type gate preserved for that path only). Results are capped at `max_candidates_per_unit=3` per new unit and `max_verified_pairs=50` globally.

**Stage 2 — LLM verification.** Candidate pairs are chunked into batches of 10 and sent concurrently to OpenAI structured output (`gpt-4.1-mini`, `temperature=0`) under a semaphore. The model returns a `CONTRADICTION`, `DUPLICATE`, or `INDEPENDENT` verdict for each pair. Only `CONTRADICTION` and `DUPLICATE` verdicts supersede the older unit. `INDEPENDENT` verdicts are a no-op, meaning the recall stage can be liberal without risking false supersessions.

The original `auto_resolve_conflicts` method is retained for admin/manual use and as the fallback when `resolution_mode=jaccard` is set or when `OPENAI_API_KEY` is absent.

## Considered Options

- **Raise the Jaccard threshold.** Reducing false positives by requiring higher overlap would also reduce recall for genuine contradictions, which already have low overlap. The threshold cannot be tuned to solve both failure modes simultaneously.

- **Cosine similarity alone (no LLM verification).** Replace Jaccard recall with cosine recall and supersede pairs above a threshold directly, without an LLM verification step. Rejected: cosine similarity measures semantic proximity, not contradiction. Memories about the same topic that are fully complementary — for example, two procedural rules both concerning marking criteria — score highly similar but must both be kept. A threshold cannot distinguish "related but compatible" from "related and contradictory", so this approach would reproduce the Jaccard false-positive problem at a higher recall level.

- **Two-stage pipeline with LLM verification.** Loose cosine recall as a recall gate, LLM as the precision gate. The LLM call is batched (10 pairs per call) and bounded by `max_verified_pairs=50`, capping cost. `INDEPENDENT` verdicts are safe no-ops, so over-recall in stage 1 does not cause harm. Accepted.

## Consequences

- `resolution_mode=llm` requires `OPENAI_API_KEY`. If the key is absent, `create_conflict_resolver` returns `None` and conflict resolution is skipped entirely.
- LLM verification adds latency to each ingest. With `max_verified_pairs=50` and `batch_size=10`, a worst-case ingest fires five parallel LLM calls. The semaphore (`max_parallel=4`) bounds concurrent calls.
- Each `resolve.verify_batch` call is traced as a Langfuse generation span with token usage, so cost is observable.
- The cosine threshold (0.65) may need tuning per deployment. Too low: noisy pairs reach the LLM and cost tokens. Too high: genuine contradictions with moderate similarity are missed at the recall stage before the LLM ever sees them.
- Pinned memories (`importance >= 0.99`) are never superseded regardless of verdict.
