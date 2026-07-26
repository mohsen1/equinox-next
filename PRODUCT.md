# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

RL researchers launch branch-aware and independent-rollout experiments, inspect the
evidence behind policy updates, and debug failures. Research operators monitor resources,
cost, retries, cancellation, and recovery. Environment and verification authors inspect
typed contracts and conformance evidence.

## Product purpose

Equinox Next is a branchable execution and evidence-based verification platform for
expensive, stateful, long-horizon reinforcement-learning environments. The local
contract-proof milestone proves the complete CAD lineage and recovery contracts without
cloud credentials, RunPod access, a GPU, or a real model provider.

Success means a researcher can reconstruct the exact state, action, proof bundle,
deterministic checks, model assessment, named rewards, eligibility decision, and
materialized examples behind one committed policy update.

## Positioning

Equinox treats reusable environment snapshots and immutable verification evidence as
first-class research objects. It forks isolated continuations from a shared decision
checkpoint and keeps deterministic facts, stochastic judge evidence, reward transforms,
and trainer-owned optimization separate.

## Operating context

Researchers work from a run list into collection batches, iterations, rollout trees,
proof bundles, and verification DAGs. The first environment is bounded multi-turn CAD
reconstruction with canonical renders and deterministic geometry reports. PostgreSQL is
the scientific metadata authority and S3-compatible object storage holds large immutable
artifacts.

## Capabilities and constraints

- The orchestrator alone accepts scientific state and lineage.
- The execution service owns operational jobs, attempts, leases, runtime cursors, and
  immutable outputs.
- Local configuration resolves `MockRunPodProvider` and `MockJudgeProvider` only.
- An LLM judge result is stochastic evidence, never ground truth or a reward.
- Logical state, runtime cursor, environment snapshot, and decision checkpoint are
  separate types.
- Collection precedes optimization; a committed iteration pins immutable inputs and
  advances policy by compare-and-swap.
- All application processes run through Docker Compose.
- The current release target is local contract proof, not CAD research beta or
  production v1.

## Brand commitments

The product name is Equinox Next. Product language is precise, calm, and evidence-led.
The dashboard is an Operate surface with the clean character of drafting instruments:
flat layers, exact alignment, restrained color, and no decorative statistics or claims.

## Evidence on hand

The canonical specification is `docs/equinox-next.md`. The previous Equinox repository
contains CAD execution, renderer, geometry-verifier, append-only lineage, and React Flow
interaction patterns that can be adapted after contract review. No customer claims,
benchmark results, production-isolation evidence, or human-aligned judge calibration
evidence exists for this local milestone.

## Product principles

- Evidence before reward.
- Scientific facts are immutable and attributable.
- Operational retries never duplicate accepted facts.
- Failure, abstention, disagreement, and integrity outcomes remain distinct.
- The smallest complete vertical slice outranks speculative generality.

## Accessibility & inclusion

Primary workflows must meet WCAG 2.2 AA. Graphs require keyboard-operable nodes and an
equivalent outline or table view. Light and dark themes follow the operating-system
preference by default.
