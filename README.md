# Equinox Next

Equinox Next is a branchable environment and observability platform for long-horizon
reinforcement learning. The active research profile is multi-step micro-repository repair:
a policy diagnoses a task, Equinox snapshots the repository and transcript, and four
isolated continuations compete from the same state.

The local Compose profile is a single-operator contract fixture. It uses deterministic
CAD execution and judge fixtures behind an authenticated internal service boundary. Real
GPU research is an explicit, bounded RunPod operator workflow outside that registry.

## Start here

```bash
./scripts/dev
./scripts/seed
```

Open [http://127.0.0.1:3100](http://127.0.0.1:3100). The first command builds and starts
PostgreSQL, MinIO, the execution/verifier service, orchestrator, fixture worker, and
dashboard. The second creates deterministic contract runs used by acceptance tests; the
dashboard only lists actual research executions and their proofs.

Ports default to `3100` for the dashboard and `8180` for the API. Override them with
`EQUINOX_DASHBOARD_PORT` and `EQUINOX_API_PORT`. MinIO, PostgreSQL, execution, and the
worker are not exposed on the host. `scripts/preflight` creates a mode-0600 local service
token in the ignored `.env` file.

The dashboard exposes `/healthz` for nginx liveness and `/readyz` for API/database
readiness. Runs and Proofs preserve the last successful response during a transient API
failure and recover through polling. The exact contracts and failure check are in
[the Runs and Proofs workspace guide](docs/runs-and-proofs-workspace.md).

## Run bounded RunPod research

`scripts/runpod-rl-proof` is an explicit operator command outside the local provider
registry. The repository-repair profile creates one A40 worker from RunPod's pinned
PyTorch 2.8 image, rejects an hourly rate above $0.50, targets 45 minutes of training,
sets a provider-side 60-minute termination deadline, and deletes the worker after the
workload finishes. The command refuses to start when the account already has a pod or
active hourly spend.

```bash
EQUINOX_RUNPOD_EXPERIMENT=repository-repair ./scripts/runpod-rl-proof
```

The repository-repair workload fine-tunes Qwen2.5-Coder-1.5B-Instruct with LoRA and a
leave-one-out group-normalized REINFORCE objective. It
collects a shared diagnostic prefix without gradient, saves an exact logical checkpoint,
restores four continuations, and applies sibling-relative credit only after the branch.
It uses disjoint task-family train, validation, and final-test splits; deterministic
verification; adaptive levels; replay; exact optimization checkpoints; live progress;
verified adapter manifests; and confirmed teardown. The dashboard observer is a launch
prerequisite.

Repository complexity adapts file count, fault count, dependency depth, and horizon.
Branch width remains static at four. The first proof uses a deterministic in-memory
repository simulator; arbitrary model-generated code and shell commands are not executed.
A single run is labelled exploratory. Use `scripts/summarize-runpod-study` on at least
three distinct-seed receipts before making a replicated learning claim. A conformant
RunPool sandbox is required before executing real repository code or test commands.

## Inspect trajectories

From `/runs`, open a research execution and select **Trajectory**. The branching view
shows the shared diagnostic prefix once, the exact logical checkpoint, and four restored
multi-step continuations. Each action exposes its observation, state digest, verifier
result, return, sibling advantage, exclusion status, and policy signal. The curriculum
view separates validation checkpoints from the one-time final test.

Completed RunPod evidence appears under `/proofs`. Each proof links back to its originating
run and trajectory and records learning outcome, hardware, estimated total cost,
curriculum progression, receipt digest, provider handle, and teardown confirmation.

## Rejudge stored evidence

Use the **Rejudge stored proof** action in a run, or call:

```bash
curl -X POST http://127.0.0.1:8180/v1/runs/RUN_ID/rejudge \
  -H 'Content-Type: application/json' \
  -d '{"verification_run_id":"VERIFICATION_ID","fixture_scenario":"integrity"}'
```

Rejudge uses `cad.pointwise.mock@2`, reuses the original proof-bundle digest, runs only
the judge step, and emits no training reward. Supported contract fixtures are `valid`,
`low`, `tie`, `abstain`, `malformed`, `retry`, `disagreement`, and `integrity`.

## Verify the repository

```bash
./scripts/check
./scripts/acceptance
```

`scripts/check` runs preflight, Compose validation, production builds, Python formatting
and lint checks, unit tests, TypeScript checks, frontend lint/format checks, and the
dashboard unit and production builds. `scripts/acceptance` starts with fresh project-scoped named
volumes, seeds both flows, runs unit and integration tests, verifies restart persistence,
and proves queued cancellation releases the allocation. It deletes only the
`equinox-next` Compose project’s local volumes.

Useful operational commands:

```bash
docker compose ps
docker compose logs -f orchestrator fixture-worker execution
docker compose restart orchestrator
docker compose down
```

Metadata survives ordinary restarts in PostgreSQL. Artifacts survive in MinIO and are
content-addressed by SHA-256. Execution sessions are logical database/object-store
records; no shared session filesystem volume exists.

## Architecture and limits

The orchestrator owns scientific authority and lineage. The execution service owns
operational sessions, cursors, jobs, attempts, and fencing. PostgreSQL isolates those
records in separate schemas and roles; MinIO stores immutable artifact bytes. Operation
inputs and digests make restart replay and duplicate delivery inspectable. Snapshot
fidelity is honestly reported as `logical_restore`, not process or kernel restoration.

The CAD slice only proves platform contracts with deterministic fixtures and is not shown
as a research environment in the dashboard. This build refuses any deployment mode other
than `local-only`; it has no user authentication, tenancy, real CAD kernel, or production
code-execution isolation. Boundaries and audit dispositions are recorded in
[PLAN.md](PLAN.md), [the execution-boundary ADR](docs/adr/0001-local-docker-execution-boundary.md),
[the judge-integrity ADR](docs/adr/0002-local-judge-integrity.md), and
[the audit remediation record](docs/audit-remediation-2026-07-27.md).
