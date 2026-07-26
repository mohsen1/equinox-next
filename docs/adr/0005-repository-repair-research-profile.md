# ADR-0005: Prove restored branching with multi-step repository repair

Status: accepted  
Date: 2026-07-26

## Decision

The next Equinox research profile is multi-step micro-repository repair. A policy uses a
structured action protocol to inspect and edit a deterministic repository. Equinox
collects a policy-generated diagnostic prefix, captures the exact logical repository and
transcript state, restores four isolated continuations, and trains only on actions after
that checkpoint.

Branch width is fixed at `K=4`. Task complexity adapts independently across file count,
fault count, dependency depth, and action horizon. Replay samples previously mastered
levels.

The first proof runs the environment as a safe deterministic simulator alongside the
trainer on a bounded RunPod worker. It does not execute model-generated code, shell
commands, or arbitrary test processes.

## Why this vertical

The local CAD fixture proves platform contracts but current affordable models are poor
CAD policies. The existing real model-repair workload proves GPU scheduling, LoRA
optimization, adaptive levels, replay, progress ingestion, artifacts, and teardown, but
each prompt produces only one action. Its four samples are not restored continuations
from a shared mid-trajectory state.

Repository repair provides meaningful state, deterministic verification, procedural
difficulty, and action horizons at a practical model and compute budget. It directly
tests the research claim that sibling-relative outcomes from the same state provide
cleaner credit than independent trajectories.

## Action and trust contract

Policy output is one JSON object per turn with an allowlisted tool name and typed
arguments. Initial tools are `list`, `read`, `search`, `edit`, `test`, and `finish`.
Parsing, path normalization, edit bounds, observation size, and action count are
deterministically limited.

Candidate content is untrusted. The first verifier compares simulator state against
hidden deterministic task assertions. It never imports candidate files or invokes a
candidate-controlled process. RunPool execution is required before replacing the
simulator with real repositories and test commands.

## Learning contract

- Prefix actions condition every continuation but receive no sibling-relative gradient.
- A checkpoint is eligible only when the environment state, model, policy, tokenizer,
  task, verifier, and prefix tokenization are identical across siblings.
- Siblings use distinct sampling seeds.
- Infrastructure failures are excluded rather than scored as candidate failure.
- At least two trustworthy sibling returns are required.
- Equal sibling returns produce exactly zero relative advantage.
- A sibling advantage is applied to every accepted post-branch action in that sibling.
- Teacher fallback is recorded separately and cannot be presented as RL improvement.
- Evaluation uses held-out tasks and greedy full trajectories.

## Observability contract

Before launch, the dashboard must show the shared prefix once, the decision checkpoint,
four continuation lanes, every action and observation, terminal verifier results,
returns, sibling advantages, exclusions, complexity level, replay status, and the update
that consumed the trajectories.

Research execution JSON is the initial persistence boundary because the current
orchestrator contracts contain CAD-specific schema constants. The result must retain
complete step lineage. Generalizing the core scientific schema is a follow-up informed by
this vertical, not a semantic shortcut inside CAD records.

## Experiment contract

The first launch is a bounded integration proof. A later matched-budget study compares
teacher-only, sampling-only, sibling-relative, adaptive, and replay variants using the
same tasks, seeds, action budget, model revision, tokenizer revision, verifier revision,
and effective learning-rate control.

Headline metrics are held-out solve rate, actions to solve, trustworthy informative-group
rate, regression on mastered levels, and GPU cost. Scaling the model is conditional on
restored branching producing useful signal.

## Non-goals

- Dynamic branch width.
- AWS execution.
- Arbitrary repository or shell execution.
- A production RunPod provider inside the credential-free local registry.
- Recasting the completed CAD fixture as the repository environment.
