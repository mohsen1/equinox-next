# Architecture audit remediation record

Date: 2026-07-27
Source: `equinox-next-architecture-code-audit.md`

This record distinguishes implemented guarantees from intentionally contained fixtures
and production capabilities that remain blocked. “Contained” means the stronger claim is
removed, the surface is hidden from research users, or startup fails closed outside the
supported local mode. It does not mean the missing production capability exists.

## Release gates

All ten release blockers named by the audit are implemented:

- `GOV-001`: `equinox-next.md` is authoritative; its generated provenance mirror is
  byte-checked by `scripts/check-spec`.
- `GOV-002`: launch requests compile to a validated immutable manifest, and workflow
  behavior reads that manifest.
- `VERIFY-013`: prompt artifacts bind exact prompt bytes and the provider request
  manifest records those bytes.
- `VERIFY-007`: nonexistent front-view claims were removed from contracts and receipts.
- `VERIFY-004`: rejudge resolves the original proof server-side and checks stored bytes
  and digests.
- `ORCH-016`: collection membership is frozen and materialization reads only the closure.
- `ORCH-018`: the local path records `SIMULATED_POLICY_COMMIT`; only the RunPod workload
  claims optimization.
- `ORCH-003` and `EXEC-006`: claims have IDs, leases, expiries, fencing tokens, and
  compare-and-set acceptance with duplicate/stale-worker tests.
- `EXEC-008`: logical snapshot restore is probed and measured.
- `DATA-025`: migration acceptance covers all current versions; advisory locking,
  checksums, and dirty-state rejection are tested.

## Finding disposition

### Governance and provenance

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `GOV-001`, `GOV-002`, `GOV-004`; `PROV-001`–`PROV-012` | Canonical spec/provenance check, manifest compiler, strict canonical JSON, immutable full-digest artifacts, explicit trust/ordinal metadata, exact prompt bytes, content-derived tree/proof identities, and cross-checked artifact reads. |
| Contained | `GOV-003`, `GOV-005` | The local fixture accepts only its compiled capability set. Reproduction remains a declared-config reproduction and is not labelled bitwise or scientific replication. A catalog-backed cross-run dependency lock remains an exit criterion for production. |

### RL and RunPod experiment validity

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `RL-003`, `RL-008`, `RL-009`, `RL-010`, `RL-011`; `EXP-001`–`EXP-009`, `EXP-011`–`EXP-014` | Normalized policy decisions and episodes; overshoot-preserving causal curriculum observations; frozen materializer artifact; honest 1.5B model/objective; no teacher path or similarity reward; disjoint task-family splits; deterministic seed receipt; optimizer/RNG checkpoint-resume; byte-verified adapter manifest; summed sequence log-probability; AST semantic verifier; authenticated allowlisted result transport. |
| Implemented, study pending | `EXP-010` | Receipts are single-seed/exploratory. `scripts/summarize-runpod-study` rejects fewer than three distinct seeds and reports mean, standard deviation, and 95% intervals. No replicated claim is made until those runs exist. |
| Contained fixture | `RL-001`, `RL-002`, `RL-004`–`RL-007`, `RL-012` | The local CAD path is explicitly deterministic fixture collection plus a simulated policy commit. It is excluded from the research UI and from algorithm-comparison claims. Adaptive repository complexity is implemented in the real RunPod workload; local CAD is not presented as real RL. |

### Execution and orchestration

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `EXEC-001`–`EXEC-009`, `EXEC-013`–`EXEC-016`, `EXEC-018`, `EXEC-019`; `ORCH-002`–`ORCH-007`, `ORCH-009`, `ORCH-011`, `ORCH-012`, `ORCH-014`–`ORCH-018`, `ORCH-021`–`ORCH-023`, `ORCH-025` | Durable activity claims, short acceptance transactions, typed failure dispositions, lease/fence checks, measured restore probes, logical session state, real capacity counts, equality checks, deadlines/backoff, restart/cancellation acceptance, serialized tree/event sequences, original-proof validation, explicit run cursor, frozen canonical JSONL inputs, simulated trainer receipt, causal complexity, and cleanup confirmation. |
| Contained | `EXEC-010`–`EXEC-012`, `EXEC-017`, `EXEC-020`; `ORCH-001`, `ORCH-004`, `ORCH-005`, `ORCH-008`, `ORCH-010`, `ORCH-013`, `ORCH-019`, `ORCH-020`, `ORCH-024` | Static `K=4` is intentional. Local CAD is not a code sandbox and synthetic costs are fixture-only. Shared/production deployment is refused. The modular-monolith transaction script, remaining insert-or-assert conversions, external broker publication, multi-iteration trainer orchestration, and dependency-locked reproduction are required before a production control plane. |

### Verification and CAD

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `VERIFY-004`–`VERIFY-014`, `VERIFY-018`, `VERIFY-019`, `VERIFY-023`, `VERIFY-024` | Server-resolved proof bytes, content-only proof identity, schema-bound ordered refs, truthful renderer metadata, explicit trust classes, semantic judge validation with strict nested models, exact prompt request manifests, verifier-only target metrics, allowlisted evidence manifests, malformed-output activity records, recomputable rewards, and corrected rejudge usage. |
| Contained fixture | `VERIFY-001`–`VERIFY-003`, `VERIFY-015`–`VERIFY-017`, `VERIFY-020`–`VERIFY-022`, `VERIFY-025`; `CAD-001`–`CAD-017` | The deterministic judge and CAD implementation are conformance fixtures, not provider judgment or real geometry. They are absent from the research UI and cannot be launched as production environments. Real CAD, calibration, uncertainty aggregation, provider blinding, region grounding, cache semantics, and a restricted real-kernel DSL remain separate future integrations. |

### Data and code structure

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `DATA-008`–`DATA-011`, `DATA-013`–`DATA-018`, `DATA-020`, `DATA-022`–`DATA-025`; `CODE-011`, `CODE-012`, `CODE-015` | Normalized decisions/episodes/reward links/collection membership, finite canonical values, content/occurrence separation, immutable verified objects, explicit retrieval policy, serialized checksummed migrations, current migration tests, truthful comments, and a complete static-quality gate. |
| Contained/local boundary | `DATA-001`–`DATA-007`, `DATA-012`, `DATA-019`, `DATA-021`, `DATA-026`–`DATA-030`; `CODE-001`–`CODE-010`, `CODE-013`, `CODE-014` | The local modular monolith retains one PostgreSQL instance, JSONB projections, direct connections, and some large typed-dictionary modules. It is not approved for tenant data, long retention, production backup/restore, or schema-independent client evolution. Those are production exit criteria, not implied guarantees. |

### Dashboard

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `UI-001`–`UI-007`, `UI-010`–`UI-012`, `UI-015` | Dead route implementations and seed UI are deleted; Runs is one focused module; primary responses use runtime decoders; no `any` escape hatches remain; polling is single-flight and abortable; GET headers and error decoding are correct; validation/test and exploratory claim language is explicit; artifacts retain sandbox headers; keyboard/focus/system-theme behavior and primary journeys are tested and browser-checked. |
| Contained | `UI-008`, `UI-009`, `UI-013`, `UI-014` | The small router remains adequate for the current read-oriented, local operator UI. Core CAD mutation routes are intentionally absent. A generated version-negotiated client and centralized mutation state are required when operator writes return. |

### Security and operations

| Status | Findings | Evidence or boundary |
|---|---|---|
| Implemented | `SEC-002`, `SEC-008`–`SEC-010`, `SEC-015`, `SEC-016`; `OPS-001`, `OPS-004`, `OPS-006` | Nginx rejects all internal routes; bearer service auth uses constant-time comparison; fixture worker has no DB/S3 credentials; edge/control/data networks are split; application containers are read-only, capability-dropped, and no-new-privileges; result serving is token-authenticated and allowlisted; request logs are structured with correlation IDs; readiness checks DB and object storage; slow work is outside acceptance transactions. |
| Fail-closed local boundary | `SEC-001`, `SEC-003`–`SEC-007`, `SEC-011`–`SEC-014`, `SEC-017`–`SEC-020`; `OPS-002`, `OPS-003`, `OPS-005`, `OPS-007`–`OPS-015` | This build starts only with `EQUINOX_DEPLOYMENT_MODE=local-only`, binds public ports to loopback, has no user/tenant model, and must not run untrusted code. Shared deployment, tenant-scoped artifact capabilities, separate object namespaces, offline dependency images/scanning, hidden-eval isolation, licensing records, traces, service metrics/SLOs, quotas, circuit breakers, archival, and production alerting remain blocking prerequisites. |

## Supported claim after remediation

Equinox Next is a credible local platform and RunPod research harness for observing static
four-way branching, causal collection membership, dynamic repository-task complexity,
and bounded exploratory post-training. It is not a multi-tenant RL service, a general
code-execution sandbox, a real CAD environment, or evidence of replicated model
improvement until the multi-seed study gate passes.
