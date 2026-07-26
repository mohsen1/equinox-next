# Reuse inventory

The local proof adapts proven algorithms and interaction ideas from the previous Equinox
repository but replaces contracts that violate the new scientific-authority boundary.
The source worktree was dirty during inventory, so no source file is copied wholesale.

| Area | Decision | Evidence and reason |
|---|---|---|
| CAD action/state vocabulary | Adapt | `packages/domain-cad` separates typed actions, geometry, observations, and executors. Keep the bounded constructive vocabulary; replace package coupling with frozen v1 contracts. |
| Geometry checks | Adapt | Existing bounds, face-count, volume, validity, and topology ideas are deterministic and testable. Re-express them as versioned reports and metric observations. |
| Canonical renderer | Adapt | Existing deterministic fixed-view rendering is useful. The local proof uses a small pinned fixture renderer so Compose stays CPU-only and reproducible on arm64 and amd64. |
| Append-only trajectory lineage | Adapt | Preserve parent/child lineage and immutable artifacts. Replace shared SQLite writes with orchestrator-owned PostgreSQL records and constraints. |
| Branch groups | Adapt | Preserve shared-prefix and sibling concepts. Add reusable snapshots, policy-specific checkpoints, runtime-cursor isolation, strict version pins, and explicit failure admission. |
| Artifact addressing | Adapt | Preserve SHA-256 content addressing. Store bytes in MinIO and roles on PostgreSQL artifact references rather than base64 payloads or hash-only rows. |
| Trajectory UI | Adapt | Preserve shared-prefix-once, true fan-out, selected-path highlighting, and React Flow. Replace static/hidden joins with typed API projections and add an accessible outline. |
| CAD demo policy | Adapt | Preserve deterministic progressive construction for fixtures. Replace direct process-local execution with typed operations and immutable proof lineage. |
| SQLite/process-local sessions | Replace | They cannot prove durable authority, leasing, fencing, cross-service retries, or restart recovery. |
| Synchronous CAD `/score` | Replace | It collapses proof, judge evidence, metrics, and rewards. Use a versioned verification DAG and named reward pipeline. |
| Existing real VLM judge | Replace | It reads ambient credentials, calls a real endpoint, and returns a scalar-like score. Local acceptance permits only `MockJudgeProvider` behind the production request/response contract. |
| Direct RunPod adapter | Replace | Local acceptance must have no resolution or contact path. Only the provider interface and mock semantics remain. |

## Dependency decisions

- FastAPI supplies typed request validation and an OpenAPI document for client contract
  checks.
- PostgreSQL supplies durable constraints, compare-and-swap updates, row leases, and
  queue consumers using `FOR UPDATE ... SKIP LOCKED`.
- MinIO supplies a local S3-compatible content-addressed artifact store with a persistent
  volume.
- React, TypeScript, Vite, and React Flow supply the operator UI and keyboard-aware DAG
  canvas. Native table/outline views remain the accessibility authority.

Primary evidence: [PostgreSQL locking clauses](https://www.postgresql.org/docs/current/sql-select.html),
[FastAPI OpenAPI metadata](https://fastapi.tiangolo.com/tutorial/metadata/),
[MinIO container deployment](https://min.io/docs/minio/container/index.html), and
[React Flow accessibility](https://reactflow.dev/learn/advanced-use/accessibility).
