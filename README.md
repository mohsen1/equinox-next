# Equinox Next

Equinox Next is a branch-aware post-training platform for long-horizon, verifiable
software tasks. It captures a shared trajectory prefix, restores four isolated
continuations from the same logical checkpoint, assigns sibling-relative credit, and
preserves the evidence needed to explain every policy update.

The current research environment targets multi-step micro-repository repair with:

- static branch width (`K=4`);
- adaptive task complexity and replay of earlier levels;
- deterministic task verification;
- LoRA post-training for Qwen2.5-Coder models;
- live run, branch, action, curriculum, cost, and teardown evidence;
- bounded RunPod execution with provider-side termination and local recovery guards.

## Run locally

Docker Compose runs the dashboard, orchestrator, execution service, deterministic fixture
worker, PostgreSQL, and MinIO.

```bash
./scripts/dev
./scripts/seed
```

Open [http://127.0.0.1:3100/runs](http://127.0.0.1:3100/runs). If port `3100` is already
in use, choose another dashboard port:

```bash
EQUINOX_DASHBOARD_PORT=33100 ./scripts/dev
```

The main views are:

- `/runs` — training status, progress, GPU allocation, and estimated cost;
- `/runs/:runId` — run outcome, collection, updates, and curriculum state;
- `/runs/:runId/trajectory` — the shared prefix, checkpoint, sibling branches, actions,
  observations, verifier results, returns, and policy signals;
- `/proofs` — immutable run evidence, learning outcome, hardware, cost, and teardown.

The local profile is a deterministic contract fixture. It does not contact RunPod or an
external model provider.

## Run guarded GPU research

Real GPU work is an explicit operator workflow outside the local provider registry. The
current larger-model campaign uses one Secure Cloud H100 80 GB worker and
Qwen2.5-Coder-7B-Instruct. It requires a committed eligibility screen before the pilot,
checks the exact offer and hourly rate immediately before allocation, enforces lifetime
and campaign budgets, publishes evidence atomically, and verifies pod and volume teardown.

Read the [larger-model RunPod runbook](docs/larger-model-runbook.md) before allocating a
GPU. Start with the read-only preflight:

```bash
./scripts/screen-larger-model --preflight-only
```

The paid screen and pilot are separate explicit commands:

```bash
./scripts/screen-larger-model
./scripts/run-larger-model-pilot
```

The pilot command refuses to run without an eligible, committed screen artifact set. Do
not bypass the wrappers: they bind the source revision, image digest, workload bundle,
provider offer, budget, progress contract, publication lineage, and teardown controls.

The current environment uses a deterministic in-memory repository simulator. Generated
model actions do not execute arbitrary code or shell commands. A RunPool-backed sandbox
is the next trust-boundary integration for real repositories.

## Architecture

The orchestrator owns accepted scientific state and lineage. The execution service owns
operational sessions, leases, attempts, and fencing. PostgreSQL stores metadata in
separate schemas; MinIO stores immutable, content-addressed artifacts. Large payloads move
through object storage, while API records carry typed references and SHA-256 digests.

The dashboard observes persisted facts rather than log text. A branch trajectory links
the shared prefix, logical checkpoint, four restored continuations, verifier evidence,
advantages, optimizer update, model artifact, provider receipt, and teardown record.

Key design documents:

- [branch-aware RL improvement plan](docs/branch-aware-rl-improvement-plan.md)
- [larger-model RunPod runbook](docs/larger-model-runbook.md)
- [independent-prefix GRPO comparison](docs/independent-prefix-grpo-comparison.md)
- [execution-boundary ADR](docs/adr/0001-local-docker-execution-boundary.md)
- [judge-integrity ADR](docs/adr/0002-local-judge-integrity.md)

## Verify

```bash
./scripts/check
./scripts/acceptance
```

`scripts/check` validates the frozen research contracts, builds the Compose images, runs
Python formatting and lint checks, executes the unit suites, and verifies the dashboard
with TypeScript, lint, unit, and production-build checks.

`scripts/acceptance` exercises the complete local contract with fresh project-scoped
volumes, restart persistence, idempotent delivery, rejudging, cancellation, and resource
release. It deletes only the `equinox-next` Compose project's local acceptance volumes.

## Current scope

Equinox Next is a research platform, not a multi-tenant production service. The local
stack is intentionally local-only and has no user authentication. Snapshot fidelity is
logical state restoration, not process-memory or microVM restoration. The CAD path remains
a deterministic contract fixture; the active learning work is software repair.

The canonical product contract is [equinox-next.md](equinox-next.md), with provenance in
[docs/equinox-next.provenance.json](docs/equinox-next.provenance.json).
