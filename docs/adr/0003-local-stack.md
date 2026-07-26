# ADR-0003: Use a small Python control plane and React operator UI

Status: accepted for local contract proof
Date: 2026-07-26

## Decision

Use Python 3.13, FastAPI, Pydantic, psycopg, PostgreSQL 17, and MinIO for the local control
and execution planes. Use React, TypeScript, Vite, and React Flow for the dashboard. Run
every application process in Docker Compose.

The orchestrator and execution service are separate deployable containers and separate
database ownership domains. They share a PostgreSQL server but use scientific and
operational tables respectively. MinIO stores immutable artifacts addressed by SHA-256.

## Why

This stack supplies typed OpenAPI contracts, mature PostgreSQL transactions and
constraints, S3-compatible persistence, and an accessible graph library without adding a
queue, cache, workflow engine, or custom component framework. PostgreSQL outbox rows and
leases are enough for local at-least-once work.

## Rejected options

- Node for both planes: viable, but the reusable CAD ideas and deterministic fixture
  tooling are Python-first.
- Django: includes more application framework than the local API needs.
- Kafka, Redis, or a workflow engine: no measured local blocker justifies another durable
  system.
- SQLite: cannot meet the acceptance durability and concurrency boundary.

## Recovery

Migrations are forward-only for the local milestone and include recovery notes. Compose
volumes preserve PostgreSQL and MinIO. The orchestrator reconciles nonterminal work from
durable operations and outbox rows on startup. Rollback is a clean local-volume reset;
scientific migrations are never silently reversed in place.
