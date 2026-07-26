# Equinox Next

Equinox Next is a branchable environment and observability platform for long-horizon
reinforcement learning. The active research profile is multi-step micro-repository repair:
a policy diagnoses a task, Equinox snapshots the repository and transcript, and four
isolated continuations compete from the same state.

The local Compose profile remains credential-free and uses deterministic CAD fixtures,
`MockRunPodProvider`, and `MockJudgeProvider`. Real GPU research is an explicit, bounded
operator workflow outside that provider registry.

## Start here

```bash
./scripts/dev
./scripts/seed
```

Open [http://127.0.0.1:3100](http://127.0.0.1:3100). The first command builds and starts
PostgreSQL, MinIO, the execution/verifier service, orchestrator, mock run agent, and
dashboard. The second creates and completes an independent four-rollout baseline and a
branch-aware CAD run.

Ports default to `3100` for the dashboard, `8180` for the API, and `9001` for the MinIO
console. Override the first two with `EQUINOX_DASHBOARD_PORT` and `EQUINOX_API_PORT`.
No cloud credentials are accepted or required.

## Run bounded RunPod research

`scripts/runpod-rl-proof` is an explicit operator command outside the local provider
registry. By default it creates one RTX 3090 community worker from RunPod's current
official PyTorch 2.8 image, rejects an hourly rate above
$0.50, sets a provider-side 15-minute termination deadline, and deletes the worker after
the workload finishes. The command refuses to start when the account already has a pod
or active hourly spend.

```bash
./scripts/runpod-rl-proof
```

The current model-repair workload fine-tunes Qwen2.5-Coder-1.5B-Instruct with LoRA. It
uses static four-sample groups, deterministic verification, adaptive levels, replay, live
progress ingestion, retained adapters, and verified teardown. The completed proof is
training-path evidence: its answers are single actions and do not yet establish restored
mid-trajectory branching.

The next workload replaces those independent answers with a structured multi-step
repository environment. It collects a shared diagnostic prefix without gradient, saves
an exact logical checkpoint, restores four continuations, and applies sibling-relative
credit only after the branch. The dashboard observer is a launch prerequisite.

Repository complexity adapts file count, fault count, dependency depth, and horizon.
Branch width remains static at four. The first proof uses a deterministic in-memory
repository simulator; arbitrary model-generated code and shell commands are not executed.
A conformant RunPool sandbox is the next environment integration.

## Inspect trajectories

From `/runs`, open **Local branch-aware CAD proof**, then select **Trace** and open its
trajectory. The explorer shows one shared three-action prefix, one logical snapshot and
decision checkpoint, and four isolated sibling cursors. Selecting any action updates the
adjacent outcome, state, verification, and provenance inspector. The fixtures include:

- a recovered render/inference retry;
- a valid negative geometry outcome caused by an oversized bore;
- a judge abstention excluded without manufacturing a zero reward; and
- a blinded, randomized sibling-group judgment.

Every transition links immutable source and candidate SVG renders, a geometry report, a
proof bundle, a real verification DAG record, judge attempt provenance, metric
observations, and named reward signals. The committed iteration input lists the exact
trees, proofs, verification runs, judge results, rewards, eligibility decisions, and
materializer digest used for the policy update.

Real research runs appear in the same run list while progress is ingested. Their
trajectory view exposes captured model responses and branch samples. The repository
profile extends that view with one shared-prefix lane, a checkpoint, and four multi-step
continuation lanes so each action and verifier result is selectable.

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
docker compose logs -f orchestrator mock-agent execution
docker compose restart orchestrator
docker compose down
```

Metadata survives ordinary restarts in PostgreSQL. Artifacts survive in MinIO and are
content-addressed by SHA-256. Candidate session directories use a named Docker volume.

## Architecture and limits

The orchestrator owns scientific authority and lineage. The execution service owns
operational sessions, cursors, jobs, attempts, and fencing. PostgreSQL isolates those
records in separate schemas and roles; MinIO stores immutable artifact bytes. Operation
inputs and digests make restart replay and duplicate delivery inspectable. Snapshot
fidelity is honestly reported as `logical_restore`, not process or kernel restoration.

The CAD slice proves the platform contracts with deterministic fixtures. It does not claim
that the mock judge is human-aligned, run a real CAD kernel, or provide production
container isolation. The repository research profile tests learning behavior without
changing those claims. Boundaries and triggers are recorded in
[PLAN.md](PLAN.md), [the execution-boundary ADR](docs/adr/0001-local-docker-execution-boundary.md),
and [the judge-integrity ADR](docs/adr/0002-local-judge-integrity.md).
