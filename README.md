# Equinox Next

Equinox Next is a local contract proof for branchable CAD reinforcement learning. It runs
entirely in Docker Compose, uses deterministic CAD fixtures, and resolves only
`MockRunPodProvider` and `MockJudgeProvider`. The Compose profile never contacts RunPod
or a real judge.

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

## Run the bounded RunPod research proof

`scripts/runpod-rl-proof` is an explicit operator command outside the local provider
registry. By default it creates one RTX 3090 community worker from RunPod's current
official PyTorch 2.8 image, rejects an hourly rate above
$0.50, sets a provider-side 15-minute termination deadline, and deletes the worker after
the workload finishes. The command refuses to start when the account already has a pod
or active hourly spend.

```bash
./scripts/runpod-rl-proof
```

The remote workload runs a seeded PyTorch REINFORCE ladder for a synthetic, deterministically
verifiable repair plan. Every update samples static K=4 sibling trajectories. Evaluation
windows promote the horizon only after the current level reaches the configured mastery
threshold. One worker runs small, medium, and large cases with maximum horizons of 8, 24,
and 64 actions. A passing proof requires CUDA execution, increasing horizons, curriculum
promotion in every case, at least 50 million sampled action decisions, a reward gain of
at least 0.20 per case, and final step reward of at least 0.85 per case. After RunPod no
longer lists the worker, the command writes a sanitized receipt under
`var/research-proofs/` and records it on the System page.

This proves authenticated scheduling, bounded GPU execution, multi-turn policy
optimization at increasing action horizons, adaptive curriculum metrics, receipt
ingestion, and teardown. The repair plan is synthetic and its per-action verifier signal
is deliberately simple; it does not prove long-horizon credit assignment by a
large-model policy, CAD reconstruction quality, RunPool sandbox execution, or a
production RunPod provider adapter.

The dashboard also exposes typed contracts for SQLite data repair, filesystem and CLI
debugging, and micro-repository code repair. Those environments include adaptive
difficulty configuration but remain launch-gated until a RunPool sandbox adapter is
connected. Branch-aware runs keep a static width of four.

## Inspect the proof

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

This slice proves the production contracts with deterministic fixtures. It does not claim
that the mock judge is human-aligned, run a real CAD kernel, or provide production
container isolation. Those boundaries and their triggers are recorded in
[PLAN.md](PLAN.md), [the execution-boundary ADR](docs/adr/0001-local-docker-execution-boundary.md),
and [the judge-integrity ADR](docs/adr/0002-local-judge-integrity.md). The next production
integration is a calibrated strong multimodal judge in a separate CAD research profile.
