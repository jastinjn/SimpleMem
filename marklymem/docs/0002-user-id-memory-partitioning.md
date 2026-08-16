# User ID as the primary memory partition key

> **Superseded by** [ADR 0006 — Hierarchical namespaces over flat scope IDs](0006-hierarchical-namespace-over-flat-scope-id.md)

## Context

SimpleMem originally partitioned memories by `scope_id` alone — a single key grouping related conversations (e.g. a class or assignment). In a multi-tenant API where many users share a single Aurora RDS instance, scope alone is insufficient: there is no guarantee that scope names are unique across users, and a query without a user boundary could return or overwrite another user's memories. Additionally, aggregating stats or archiving memories for a single user requires a reliable top-level key that spans all of their scopes.

## Decision

`user_id` is added to the `memories` table as a required, indexed column and becomes the primary partition key for all reads and writes. Every store method — search, insert, archive, stats — accepts `user_id` as a mandatory argument and includes it in the `WHERE` clause. `scope_id` remains an optional secondary filter: omitting it queries or writes across all scopes for that user; providing it narrows to a specific sub-context. The API rejects any request that omits `user_id` with a 422.

## Considered Options

- **Embed user identity in `scope_id` by concatenating `user_id + scope_id`.** Maintains the single-key model but is fragile — callers must construct and parse the compound key consistently, and any mismatch silently queries the wrong partition. More critically, retrieving all memories for a user across all their scopes requires either a prefix scan or knowledge of every scope name, which is not reliably available at query time. Rejected.
- **Add `user_id` as a required first-class field on all reads and writes.** Enforces tenant isolation cleanly at the query layer. `scope_id` remains an optional secondary filter — omitting it queries all scopes for that user, providing it narrows to one. Stats, archival, and retrieval work naturally at both the user and scope level. Accepted.

## Consequences

- No mixing of memories between users is possible at the query layer — every read and write is scoped to a `user_id`.
- Stats, archival, and retrieval can be computed across all of a user's scopes by omitting `scope_id`, or narrowed to a single scope by providing it.
- `user_id` was added to `MemoryUnit` and the `memories` schema. All callers must supply it; the API enforces `min_length=1` on the field.
- `scope_id=None` means "no scope filter" (all scopes for this user), not a default scope. This is a deliberate API contract — callers should not rely on a default scope being applied silently.
