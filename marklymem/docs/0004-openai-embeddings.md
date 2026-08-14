# OpenAI embeddings over locally-hosted models

## Context

The upstream evolver used sentence-transformers (a locally-downloaded model) for embedding; SimpleMem core defaulted to Qwen3-Embedding-0.6B, also run locally. Both approaches require the server process to load and run an embedding model. On ECS Fargate this means bundling a model into the container image, increasing image size, cold-start time, and memory footprint — all for a component that is not the core responsibility of the service. Hosting a model inference runtime in a containerised API adds operational complexity without a clear quality advantage over managed embedding APIs.

## Decision

Replace the local embedding model with the OpenAI embeddings API (`text-embedding-3-small` by default, configurable via `EMBEDDING_MODEL`). The same `OPENAI_API_KEY` already required for LLM extraction is reused — no new credentials are needed. All embedding calls go through the shared OpenAI client used by the LLM extractor, so rate-limit handling and retry logic are centralised in one place. The hybrid retrieval pipeline (FTS + vector) is retained: keyword search handles exact and near-exact matches; vector search handles semantic similarity. Both legs contribute to the final retrieval score.

## Considered Options

- **Keyword-only retrieval (no embedder).** Removes all embedding infrastructure and eliminates the added latency of an embedding API call at ingest time. However, keyword-only retrieval degrades as the number of memories per user grows — it cannot surface semantically related memories that use different vocabulary. Rejected: the hybrid pipeline is expected to become increasingly valuable in production as memory stores grow, and the marginal cost of an embedding call is low.
- **Retain locally-hosted model (sentence-transformers or Qwen).** Avoids an external API dependency for embeddings and keeps embedding latency in-process. But it requires a large model artefact in the container image, significant memory headroom on the Fargate task, and a separate inference runtime to maintain. The operational cost outweighs the benefit. Rejected.
- **OpenAI embeddings API.** No model artefact in the image; no separate inference runtime. Quality is well-established and consistent. The `OPENAI_API_KEY` is already a hard requirement for `ingestion_mode=llm`. Accepted.

## Consequences

- Container images are significantly smaller — no sentence-transformers, PyTorch, or Qwen model weights bundled.
- Fargate task memory requirements are reduced; cold starts are faster.
- Embedding calls add network round-trip latency at ingest time. This is bounded by the OpenAI API and is acceptable given that ingest is already an async operation.
- `OPENAI_API_KEY` is now required for both extraction and embedding — there is no supported configuration that runs without it.
- The hybrid retrieval pipeline (FTS + pgvector cosine) is preserved. As memories per user grow, vector search is expected to provide meaningful lift over keyword search alone.
