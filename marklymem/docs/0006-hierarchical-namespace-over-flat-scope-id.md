# Hierarchical namespaces over flat scope IDs

> **Supersedes** [ADR 0002 — User ID as the primary memory partition key](0002-user-id-memory-partitioning.md)

## Context

The original `scope_id` field was a flat, opaque string — a single key that partitioned memories into isolated buckets. Queries and mutations on `scope_id = "year10-english"` could only reach memories stored in exactly that bucket; there was no relationship between `"year10-english"` and `"year10-english/term2"` or `"year10-english/term2/week3"`. This forced callers into one of two undesirable patterns:

- **Over-broad queries**: omit `scope_id` entirely to see all memories for a user, pulling in unrelated contexts.
- **Over-narrow queries**: provide an exact scope and miss memories stored in related sub-contexts.

In practice, memories naturally form a graph — facts connect across contexts, sessions, and entities in ways that a strict hierarchy cannot fully express. However, maintaining a true graph requires a dedicated graph database and significantly more complex query infrastructure. A tree is a lightweight approximation: it captures the most common access pattern (broad-to-narrow contextual zoom) without the overhead. A tutoring agent organising memories by `subject/class/student` needs to answer questions at every level — what does this student know, what does this class know, what is known across the whole subject — without running separate queries for each leaf scope. A flat partition cannot model this without duplicating memories across every ancestor scope.

## Decision

`scope_id` is renamed to `namespace` and given hierarchical semantics:

- **Reads are subtree queries.** Querying `namespace = "subject"` matches memories stored at `"subject"`, `"subject/class"`, `"subject/class/student"`, and any other descendant. This is implemented as `namespace = X OR namespace LIKE 'X/%'` with `_` escaped to prevent SQL wildcard collision.
- **Writes target the exact namespace.** Ingesting into `"subject/class/student"` stores the unit at that exact path — it does not propagate up to `"subject"` or `"subject/class"`.
- **Consolidation is exact-match only.** Dedup and decay run against `list_active_exact`, which matches only the specific namespace, preventing cross-branch consolidation.

Namespace segments must match `[a-zA-Z0-9_-]+`, separated by `/`. Leading slashes are allowed to support the AWS AgentCore agent-per-customer convention (e.g. `/retail-agent/customer-123`). Trailing slashes are stripped silently. `"/"` and `""` both normalise to `null`, targeting the global (unscoped) namespace.

A `text_pattern_ops` index on the `namespace` column ensures the `LIKE 'prefix/%'` predicate is backed by a btree and does not seq-scan as the table grows.

## Considered Options

- **Flat `scope_id` (original design).** Each scope is a discrete, unrelated partition. The application must be designed with explicit knowledge of every scope it intends to query — there is no concept of a broader context that spans related scopes. Callers have to decide at design time which specific scopes to target for each operation, and a query on `"subject"` has no relationship to `"subject/class"`. This works when the set of contexts is small and fixed, but becomes a hard design constraint as the domain grows: adding a new level of context (e.g. per-student memories within a class) requires explicit changes to every call site that needs to span them. Rejected.
- **Hierarchical namespaces with LIKE prefix queries.** Minimal schema change — rename the column, add a `text_pattern_ops` index. The application can add context levels freely without revisiting existing query logic. A query on `"subject"` automatically spans all memories stored at any depth under it. Accepted.

## Consequences

- Retrieval, stats, archival, TTL, and analytics all return results from the full subtree when a namespace is provided — callers no longer need to enumerate child scopes.
- Writes and consolidation remain exact-match, so memories are stored at the intended level of the tree and dedup does not cross branch boundaries.
- Callers that previously relied on exact isolation between `"subject"` and `"subject/class"` on reads must now use the full child path when they want to exclude the parent, since querying `"subject"` now includes `"subject/class"`.
- The `clone_namespace` operation is flat: every unit in the source subtree is copied to exactly `target_namespace`, collapsing the branch structure. Callers that need branch-preserving clones must implement that at the application layer.
- `namespace=None` on reads returns all memories for the user with no namespace filter. On writes it targets the null namespace (global scope). This asymmetry is intentional — see ADR 0002.
