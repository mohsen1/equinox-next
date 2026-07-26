# ADR-0004: Add environment contracts and adaptive complexity before RunPool execution

Status: accepted
Date: 2026-07-26

## Decision

Equinox exposes a typed environment catalog for CAD reconstruction, SQLite data repair,
filesystem and CLI debugging, and micro-repository code repair. CAD remains the only
locally executable environment. The other three entries are `CONFIGURATION_DRAFT` and
cannot launch until a RunPool execution adapter supplies their sandbox lifecycle and
passes environment conformance.

Every new run records an adaptive-complexity policy and a persisted complexity state.
The policy samples from a bounded level band and promotes the current level after a
minimum evaluation window reaches the configured mastery threshold. Complexity
observations are idempotent and reject stale levels.

Branch width remains static within a run. Branch-aware runs use `K=4`; independent
baselines use width one. The manifest records the mode and width.

## Why

The CAD fixture proved lineage, verification, and recovery contracts but is too expensive
and model-sensitive to remain the only product environment. SQLite, CLI, and repository
repair offer persistent state, deterministic verification, procedural difficulty, and meaningful
action boundaries at lower execution cost.

The catalog separates environment readiness from provider readiness. It lets the
dashboard configure the next domains without routing them through the CAD worker or
claiming that a real sandbox exists.

Adaptive complexity is part of the scientific configuration rather than a presentation
setting. Persisting its state makes promotion, restart recovery, and later RunPool worker
observations attributable.

## Contract

An environment catalog entry declares:

- the policy-facing action space;
- the trusted verifier boundary;
- snapshot strategy;
- task revision;
- launch readiness; and
- named complexity dimensions with default bounds.

The adaptive policy records minimum, current, and maximum level; sampling-band width;
mastery threshold; evaluation window; and promotion step. An observation includes a
level, success count, attempt count, operation ID, and canonical request digest.

## Non-goals

- This change does not execute SQL, shell, or repository actions.
- It does not contact RunPool or accept RunPool credentials.
- It does not implement dynamic branch width.
- It does not reinterpret existing CAD runs that predate the complexity contract.

## Next integration

The RunPool adapter must bind one catalog entry to a pinned sandbox image, task generator,
action protocol, snapshot implementation, and verifier plan. Only then should
`launch_enabled` become true for that environment.
