# Implementation plan

## Current phase

Phase 7 — multi-step repository repair with restored continuations.

Acceptance statement: a bounded real-model run collects a policy-generated shared prefix,
restores static `K=4` continuations from the same repository checkpoint, adapts task
complexity, replays earlier levels, trains only on post-branch actions, and exposes the
complete trajectory and update lineage before GPU launch.

The next immutable larger-model candidate is
`qwen2.5-coder-7b-runpod-h100@8`, with eligibility screen
`larger-model-eligibility-screen@8`, pilot
`runpod-repository-repair-large-model-pilot@6`, objective
`verified-repair-chain-transactional-retention-policy-gradient@18`, and transactional
retention `adapter-optimizer-policy-lineage@1`.

## Completed evidence

- The Compose product passes the local CAD contract, recovery, cancellation, and
  persistence acceptance flow without cloud credentials.
- Scientific and operational authority are separated in PostgreSQL; immutable artifacts
  are content-addressed in MinIO.
- The dashboard exposes persisted runs, evidence, verification, rewards, operations, and
  a keyboard-accessible trajectory explorer.
- Environment contracts declare SQLite, filesystem/CLI, and micro-repository repair with
  persisted adaptive-complexity state.
- A bounded Qwen2.5-Coder-1.5B RunPod proof completed LoRA training with static `K=4`,
  replay, three complexity promotions, held-out improvement from 25% to 100%, retained
  all earlier levels, persisted branch samples, and confirmed teardown.
- That proof sampled four independent single-action answers per prompt. It did not restore
  mid-trajectory state and is therefore compute and training-path evidence, not the
  branch-aware research result.

## Active increments

1. Commit the immutable `@8` source contract and stage its deterministic bundle on the
   existing RunPod network volume with the bounded CPU-only operator.
2. Run the budget-capped eligibility screen and expose its live evidence in the dashboard.
3. If and only if the screen authorizes it, run one `@6` H100 pilot, ingest progress and
   artifacts, confirm teardown, and report learning and branch-signal quality.

## Research follow-up

After the vertical proof, run a matched-budget matrix with identical tasks, seeds, action
budgets, and effective learning-rate controls:

- supervised teacher fallback only;
- `K=4` sampling without sibling-relative updates;
- `K=4` sibling-relative updates;
- sibling-relative updates plus adaptive complexity;
- sibling-relative updates plus adaptive complexity and replay.

Compare held-out success, actions to solve, informative-group rate, regression on mastered
levels, and GPU cost. Scale the model only if restored branching adds signal.

## Boundaries

- Do not use AWS.
- Do not execute candidate-controlled code or shell commands in the first proof.
- Do not claim production isolation until a pinned RunPool image and sandbox adapter pass
  conformance.
- Keep CAD as the completed contract fixture; do not force repository evidence through
  CAD-specific schema constants.
- Generalize the orchestrator lineage schema only after the repository vertical reveals
  the stable cross-environment contract.
