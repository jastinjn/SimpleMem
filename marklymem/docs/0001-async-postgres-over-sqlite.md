# Async PostgreSQL over SQLite as the persistence layer

## Context

The upstream EvolveMem library uses SQLite + FTS5 as its backing store. This works well for a single-process agent running on a machine with persistent local disk. marklymem is designed to run as a containerised REST API on ECS Fargate, which provides no persistent filesystem — a container restart or replacement silently loses all SQLite data. Even with an EFS mount, SQLite does not support the concurrent writer model needed when multiple Fargate tasks serve traffic behind a load balancer.

## Decision

Replace the SQLite store entirely with an async PostgreSQL backend (Aurora RDS). SQLAlchemy async (`asyncpg` driver) is used for all DB access rather than raw SQL — this gives parameterised queries by default, preventing SQL injection, and provides a typed ORM layer that makes schema changes auditable and refactorable. Full-text search is handled by Postgres `websearch_to_tsquery` and a `tsvector` generated column; vector similarity search is handled by pgvector. Schema changes are managed with Alembic migrations, giving a versioned, repeatable migration history with upgrade and downgrade paths.

## Considered Options

- **Keep SQLite with an EFS mount.** SQLite supports only one writer at a time; concurrent Fargate tasks would serialize on file locks or corrupt the database. EFS also adds latency compared to RDS. Rejected.
- **Keep SQLite, run a single Fargate task.** Eliminates the concurrency problem but gives up horizontal scaling. A single task is a single point of failure and cannot scale to meet load. Rejected.
- **Async PostgreSQL on Aurora RDS with SQLAlchemy + Alembic.** Aurora Serverless v2 scales compute independently of the application tier, so Fargate tasks can scale out without any changes to the persistence layer. SQLAlchemy's ORM eliminates raw SQL string construction, and Alembic provides a versioned migration history. Accepted.

## Consequences

- ECS Fargate tasks are fully stateless — any task can handle any request, and the deployment can scale horizontally without coordination.
- Aurora RDS provides automated backups, point-in-time recovery (PITR), and slow-query logging out of the box — none of which SQLite offers.
- A running Postgres instance is now required for local development and tests. The `.env.example` documents the expected `DATABASE_URL` format.
- The FTS5 virtual table and SQLite-specific SQL from EvolveMem were replaced with Postgres equivalents (`tsvector`, `websearch_to_tsquery`, `gin` index, `pgvector`). The query semantics are equivalent but the SQL is not portable back to SQLite.
- All schema changes must go through an Alembic migration. Ad-hoc `ALTER TABLE` statements against the database are discouraged — the migration history is the authoritative record of schema evolution.
- SQLAlchemy's parameterised queries mean user-supplied values (content, topics, entities) are never interpolated directly into SQL strings.
