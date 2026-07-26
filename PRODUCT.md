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
stateful, long-horizon reinforcement-learning environments. Its active research profile
tests whether restoring several policy continuations from the same meaningful state makes
post-training more efficient and attributable.

The root `equinox-next.md` is the only editable product and architecture specification.

Success means a researcher can reconstruct the shared prefix, checkpoint, four restored
continuations, every action and verifier result, sibling-relative reward, and exact
materialized examples behind one policy update.

## Positioning

Equinox treats reusable environment snapshots and immutable verification evidence as
first-class research objects. It forks isolated continuations from a shared decision
checkpoint and keeps deterministic facts, stochastic judge evidence, reward transforms,
and trainer-owned optimization separate.

## Operating context

Researchers work from a run list into collection batches, iterations, rollout trees,
proof bundles, and verification DAGs. The active environment is multi-step
micro-repository repair with structured tools and deterministic tests. Task complexity
changes file count, fault count, dependency depth, and action horizon. CAD remains the
local contract fixture. PostgreSQL is the scientific metadata authority and
S3-compatible object storage holds large immutable artifacts.

## Capabilities and constraints

- The orchestrator alone accepts scientific state and lineage.
- The execution service owns operational jobs, attempts, leases, runtime cursors, and
  immutable outputs.
- The credential-free local profile still resolves `MockRunPodProvider` and
  `MockJudgeProvider` only.
- Bounded real RunPod experiments are explicit operator actions outside that provider
  registry and must be observable before launch.
- Research branch width is static at `K=4`; task complexity adapts independently.
- Repository actions use a structured protocol. The first proof uses a deterministic
  simulator and never executes candidate-controlled code.
- An LLM judge result is stochastic evidence, never ground truth or a reward.
- Logical state, runtime cursor, environment snapshot, and decision checkpoint are
  separate types.
- Collection precedes optimization; a committed iteration pins immutable inputs and
  advances policy by compare-and-swap.
- All application processes run through Docker Compose.
- RunPool sandbox execution is the next isolation integration, not part of the first
  repository-repair proof.

## Brand commitments

The product name is Equinox Next. Product language is precise, calm, and evidence-led.
The dashboard is an Operate surface with the clean character of drafting instruments:
flat layers, exact alignment, restrained color, and no decorative statistics or claims.

## Evidence on hand

The local CAD contract flow proves branch lineage, immutable evidence, retry semantics,
recovery, and cancellation. A bounded Qwen2.5-Coder-1.5B RunPod experiment proves real
LoRA training over a multi-step repository simulator, one shared diagnostic prefix,
logical snapshot restore into four continuations, sibling-relative policy updates,
adaptive levels, replay, artifact retention, live status ingestion, and teardown. It is
a deterministic integration proof, not a matched-budget claim that branching outperforms
teacher-only or independent collection.

No customer claims, production-isolation evidence, or human-aligned judge calibration
evidence exists.

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
