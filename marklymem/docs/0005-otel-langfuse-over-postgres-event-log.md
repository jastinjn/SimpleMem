# OpenTelemetry + Langfuse as the observability platform

## Context

The evolver write pipeline maintains a memory events table in PostgreSQL (`event_log`) that records mutations such as ingest, supersede, decay, and reinforcement. The intent was to give operators visibility into what the memory system is doing over time.

The base application that marklymem is deployed into already instruments its LLM calls with OpenTelemetry exported to a self-hosted Langfuse instance. Langfuse ingests OTLP/HTTP traces and provides a live trace explorer, session grouping, and an online evaluation framework. Adding a second, bespoke observation mechanism in Postgres would duplicate that infrastructure while offering far less.

## Decision

Use OpenTelemetry (OTLP/HTTP → Langfuse) as the sole observation platform for memory operations. The two primary operations — ingestion and retrieval — are each emitted as a single OTel trace with child spans covering every sub-step:

- **`memory.ingest`**: `extract.session` → per-window `extract.window` generation spans (LLM calls with token usage) → `embedding` batch span with per-chunk `embedding.chunk` generation spans → `consolidate` span (output includes the content of every superseded memory).
- **`memory.retrieve`**: `embedding` span (model, token count) → output is the retrieved memories with scores and content.

Traces carry `langfuse.session.id`, `langfuse.user.id`, and `scope_id` metadata so every memory operation can be correlated to the originating conversation session in the Langfuse UI.

## Considered Options

- **Keep the Postgres event log as the primary observation mechanism.** The event log captures mutation counts and memory IDs but not the content of LLM calls, token usage, or retrieval scores. Querying it requires direct database access or a bespoke API endpoint. It cannot surface latency breakdowns per sub-step. Rejected.
- **Keep the Postgres event log alongside OTel.** Dual-writing the same information adds maintenance overhead with no new signal — everything the event log captures is already a span attribute or event in Langfuse. Rejected.
- **OpenTelemetry → Langfuse only.** A single trace per operation captures the full execution tree: LLM calls with prompt/response content and token counts, embedding model calls with per-chunk granularity, consolidation outcomes including the content of removed memories, and retrieval results with scores. Langfuse provides instant remote visibility without a VPN or database client, session-scoped grouping across operations, and an online evaluation framework that can score memory quality against ground truth. Accepted.

## Consequences

- Tracing is opt-in: the server starts with tracing disabled unless `OTEL_ENABLED=true` and the three Langfuse connection settings are provided. All instrumentation is a no-op when disabled — no performance cost in unconfigured deployments.
- The Postgres event log remains in the schema but is no longer the canonical record of memory system activity. It may be removed in a future migration once the OTel-based observability is validated in production.
- A self-hosted Langfuse instance (or a Langfuse Cloud account) is required to receive traces. The OTLP endpoint and Basic-auth credentials are the only configuration needed.
- Full content capture is always on when tracing is enabled: raw dialogue windows, extracted memory text, retrieved memory content, and the content of superseded memories all appear in spans. This is intentional — the primary use case is debugging and evaluation, where content is essential. Deployments with strict data residency requirements should use a self-hosted Langfuse instance.
- Online evaluations (LLM-as-judge, human review) can be run directly against ingestion and retrieval traces in Langfuse, enabling iterative improvement of extraction prompts and retrieval scoring without a separate evaluation pipeline.
