# Goal: prove branch-aware post-training on long-horizon software repair

Equinox Next is a branchable environment and observability platform for reinforcement
learning over stateful, verifiable tasks. The active research milestone is a real
multi-step micro-repository repair loop.

The milestone must:

1. Give a model structured repository tools and a deterministic verifier.
2. Collect a genuine shared action prefix, snapshot the repository and transcript, and
   restore four isolated continuations from that checkpoint.
3. Apply sibling-relative credit only to post-checkpoint actions.
4. Keep branch width static at `K=4`.
5. Adapt task complexity from measured performance while replaying earlier levels.
6. Persist enough step, state, branch, reward, and optimizer lineage to explain a policy
   update from the dashboard.
7. Run as a bounded, explicitly requested RunPod experiment with live local observation,
   artifact retention, and verified teardown.
8. Support a later matched-budget comparison against faithful independent-prefix GRPO.

The first proof uses a safe deterministic repository simulator. Candidate actions must
not execute arbitrary generated code or shell commands. A RunPool-backed sandbox is the
next trust-boundary integration, not a precondition for testing branch-aware credit.

Do not use AWS. Do not increase model size or GPU count until shared-prefix branching is
observable and produces useful learning signal.

The canonical [Equinox Next specification](equinox-next.md) and the acceptance
requirements below remain the contract for the completed local CAD fixture. The active
research profile extends that fixture; it does not reinterpret or weaken its accepted
evidence.

## Completed local contract baseline

## Mission

Build a branchable execution and evidence-based verification platform for reinforcement
learning over expensive, stateful, long-horizon environments.

The first vertical slice is CAD reconstruction. The local product must let a researcher:

1. Launch and monitor an independent-rollout baseline and a branch-aware CAD run.
2. Inspect each collection batch, training iteration, and rollout tree.
3. Open a rollout tree, follow its shared prefix, and inspect four sibling branches.
4. Capture a truthful logical environment snapshot and a policy-specific decision
   checkpoint.
5. Fork four isolated continuations from that checkpoint.
6. Execute a multi-turn CAD action sequence.
7. After each accepted CAD action, produce canonical source and candidate renders,
   deterministic geometry evidence, and an immutable proof bundle.
8. Invoke a deterministic `MockJudgeProvider` through the same typed multimodal judge
   contract intended for a future strong model.
9. Produce pointwise progress judgments and a blinded sibling-group judgment.
10. Convert accepted deterministic and judge evidence into separate named reward signals.
11. Link one committed policy update to the exact trees, proof bundles, judge results,
    reward signals, and materialization manifest that produced it.
12. Restart, retry, rejudge, or cancel work without duplicating accepted facts or leaking
    live resources.

Use BPO as the first branch-aware algorithm customer. Keep branch selection, judge-signal
usage, advantage estimation, propagation, scalarization, and optimization out of the
orchestrator.

The architecture must also be able to express the later full-stack verification graph:
build and tests, static architecture evidence, ephemeral deployment, Playwright user
journeys, screenshots and traces, a scoped UI taste judge, and a scoped architecture
integrity judge. Do not build that full environment before the CAD local acceptance flow
passes.

## Source of truth

- Treat the root `equinox-next.md` as the only editable product and architecture source
  of truth.
- Record its version and byte digest in `docs/equinox-next.provenance.json`.
- The build and tests must not depend on an external absolute path or a second editable
  mirror.
- Treat the specification's **local contract-proof acceptance** as the current release
  target. Do not accidentally implement the CAD research beta or production-v1
  acceptance as part of this task.
- Do not silently weaken a requirement. A deliberate semantic or scope change requires an
  ADR and, when it changes a named acceptance level, a specification amendment.
- Resolve ambiguity through the specification, repository evidence, recent primary
  sources, and a short ADR. Ask the user only when the choice changes product scope,
  safety, substantial cost, or an irreversible public interface.
- Keep product decisions in the specification or an ADR. Do not hide them in code
  comments, review transcripts, or chat.

## Hard constraints

Run `scripts/preflight` before scaffolding beyond the smallest hello path. It must verify
Docker, Compose, architecture, required local source paths, required skills, and the exact
Claude Code reviewer configuration.

Do not create a commit until the pending diff passes deterministic checks and the required
Claude Code `Fable 5` high-reasoning review. If that reviewer configuration cannot be
resolved, stop before the first commit and report the blocker.

### Local execution only

- Do not create, start, stop, inspect, or contact a real RunPod resource.
- Do not use RunPod credentials or make network calls to RunPod.
- Implement `MockRunPodProvider` behind the policy-compute provider contract.
- Add a test that fails if the local profile can resolve a real RunPod provider.
- Do not call a real LLM or multimodal model provider.
- Implement `MockJudgeProvider` behind the judge-provider contract.
- Add a test that fails if the local profile can resolve or contact a non-mock judge
  provider.
- A real strong-model adapter may be defined as an interface or disabled configuration,
  but it must not require credentials, make a network request, or be exercised by local
  acceptance.
- All application services and dependencies must run through Docker Compose.
- The host may run repository tooling, Git, browser inspection, and Docker commands. Do
  not run application servers on the host.
- Use PostgreSQL for metadata and an S3-compatible local service such as MinIO for
  artifacts.
- Use a Docker-backed local execution provider for candidate sessions and trusted
  verifier work.
- Record an ADR for the Docker control boundary before Phase 2. A raw host Docker socket
  is an explicitly unsafe development fallback, not a production isolation mechanism.
- Declare actual checkpoint fidelity. Do not label logical state, a filesystem copy, or a
  container image as a process-memory or microVM snapshot.
- Provide deterministic mock policy, trainer, CAD task, renderer fixtures, and judge
  outputs so the complete flow runs without a GPU or external API.

### Scientific authority and lineage

- The orchestrator owns accepted scientific run state and lineage.
- Execution and verifier workers own operational jobs, attempts, leases, provider handles,
  and immutable output artifacts. They do not write orchestrator scientific tables.
- The execution service emits an operational completion result. The orchestrator validates
  authorization, digests, expected state, budgets, and schemas before creating a
  scientific state, transition, verification result, metric observation, checkpoint, or
  reward signal.
- Separate semantic `State` from `RuntimeCursor`. A state does not contain a provider
  handle, lease, or runtime identity.
- Separate reusable `EnvironmentSnapshot` from policy-specific `DecisionCheckpoint`.
- Separate `CollectionBatch` from `TrainingIteration`. A rollout tree is collected first;
  an immutable iteration-input manifest later names the exact trees and signals consumed
  by an optimizer update.
- Human logs are diagnostics. Typed events, records, and artifact manifests drive product
  state.
- Mutating operations require operation IDs, idempotency keys, canonical input digests,
  and expected-state or expected-version checks where state can race.
- Use at-least-once delivery with idempotent consumers. Do not claim exactly-once
  transport.
- A repeated idempotency key with a different request digest is a conflict.
- A committed training iteration advances policy through compare-and-swap and names the
  exact rollout trees, verification runs, judge results, reward signals, eligibility
  decisions, and materialization version used by the update.
- Large payloads move through object storage. API payloads carry artifact references.
- Hidden tests, task references that are intentionally hidden, judge rubrics, calibration
  cases, and provider credentials stay outside candidate and policy containers.

### Verification and judge integrity

- Model verification as a versioned DAG, not a synchronous `score()` function.
- Support typed verifier nodes for deterministic checks, evidence production, browser or
  sandbox execution, LLM judging, and deterministic aggregation.
- An LLM judge result is stochastic evidence. It is never itself the reward and is never
  treated as ground truth.
- Preserve deterministic candidate failures separately from verifier failures, provider
  failures, invalid judge output, abstention, disagreement, and integrity violations.
- Infrastructure, verifier, judge-provider, abstention, disagreement, and integrity
  failures never become candidate reward zero.
- A valid failed test or violated geometry constraint may produce an accepted negative
  metric.
- Every judge invocation pins an immutable proof-bundle digest, judge specification,
  prompt template, rubric, output schema, sampling settings, sample index, and provider
  model identity.
- Store raw provider output and parsed structured output separately.
- Store a concise evidence-based explanation. Do not require or claim to store hidden
  chain-of-thought.
- Candidate-controlled source, strings, DOM, accessibility labels, images, meshes, and
  screenshots are untrusted prompt-injection surfaces.
- Judge execution receives only allowlisted evidence roles, no tools, no browsing, no
  shell, no candidate credentials, and no writable product systems.
- Pairwise or group judge inputs blind candidate, policy, provider, and branch identity
  and randomize presentation order.
- The mock judge must exercise valid score, low score, tie, abstention, malformed output,
  provider retry, disagreement, and integrity-violation paths.
- Rejudging an existing proof bundle creates a new verification run. It never overwrites
  the judge result used by a committed iteration.

### CAD evidence rules

- The first product environment is CAD, not a generic toy environment.
- A tiny deterministic state machine may exist as a unit-test fixture, but it is not a
  product milestone or a substitute for the CAD vertical slice.
- Each accepted CAD transition produces a pinned source-state render and candidate-state
  render from canonical camera, material, lighting, resolution, and view settings.
- The proof bundle includes the task reference, source render, candidate render, action
  summary, geometry report, renderer metadata, and artifact digests.
- Deterministic checks cover applicable syntax, execution, geometry validity, topology,
  dimensions, and task constraints.
- The pointwise judge schema returns decomposed criteria, including reference
  correspondence, silhouette and proportion, feature presence and placement, geometric
  coherence, progress, and regressions.
- The branch-group judge compares sibling proof bundles through a blinded, randomized
  presentation and can return preferences, ties, ranking, confidence, or abstention.
- Keep geometry, constraint, visual alignment, progress, regression, terminal, sibling
  preference, and execution-cost signals separate.

### Scope discipline

- Build the smallest vertical slice that proves the next requirement.
- Finish the complete local CAD path before adding software-repository or full-stack
  environments.
- Prefer a reversible choice over a framework when evidence is incomplete.
- Add an abstraction when the specification requires a boundary or when two real
  consumers need it. Do not create extension points for hypothetical consumers.
- Do not add Kubernetes, Kafka, Redis, a workflow engine, service mesh, custom component
  framework, or custom charting system unless a measured blocker and ADR justify it.
- Use PostgreSQL leasing and a transactional outbox before adding another queue or event
  system.
- Keep deployable boundaries to the dashboard, orchestrator, mock policy run agent,
  execution and verifier service, PostgreSQL, and object storage unless the specification
  requires another process.
- The verifier plane is a scheduling concept. Do not split render, browser, geometry, and
  judge work into separate services before a measured need.
- Do not build a model-jury system, learned reward model, human-review application, or
  full-stack environment during local acceptance.

## Definition of done

The local contract-proof build is complete when all items below pass in a clean checkout.

### Local startup

- One documented command builds and starts the stack with Docker Compose.
- One documented command seeds deterministic CAD demo tasks and runs.
- Health checks become ready without manual database or bucket setup.
- No service requires a cloud credential.
- The local configuration resolves only mock policy-compute and mock judge providers.
- Restarting Docker services preserves committed metadata, proof bundles, judge results,
  and artifacts.

### Local CAD branch run

- The dashboard launches an independent-rollout baseline and a branch-aware CAD run
  through `MockRunPodProvider`.
- The mock run agent emits typed heartbeats, policy versions, collection batches,
  iterations, metrics, snapshots, and checkpoints.
- A multi-turn CAD environment accepts several actions and creates semantic states.
- Each accepted transition produces canonical source and candidate renders, a geometry
  report, and an immutable proof bundle.
- A transition-level verification graph runs deterministic checks and invokes
  `MockJudgeProvider` through the production judge request and response schemas.
- The environment reaches a declared decision boundary, captures a logical snapshot, and
  forks four isolated child runtime cursors.
- Each child executes and produces versioned proof, metric observations, judge results,
  and reward signals.
- One child exercises a retryable infrastructure failure.
- One child exercises a valid negative candidate outcome.
- One judge path exercises abstention or disagreement without manufacturing a zero.
- A blinded group judge compares the siblings.
- The trainer commits an iteration linked to the exact admitted rollout trees,
  verification runs, judge results, reward signals, and materialization manifest.
- Duplicate command and result delivery do not duplicate transitions, verifier steps,
  accepted judge results, metric observations, reward signals, jobs, or optimizer commits.

### Rejudge, recovery, and cancellation

- A second mock judge specification rejudges a stored proof bundle without rerunning CAD
  execution, geometry checks, or rendering.
- Restart the orchestrator during an active collection batch and reconcile without losing
  an accepted transition or committed iteration.
- Restart the execution and verifier service during an action and during a multi-node
  verification run. Recover through durable operation records and resume from accepted
  nodes.
- Exercise duplicate and out-of-order operational result delivery.
- Cancel an active run and release mock policy allocations, jobs, sessions, runtime
  cursors, leases, and ephemeral snapshots.
- Preserve scientific snapshot, checkpoint, proof, judge, failure, and exclusion metadata
  according to retention policy.
- Show cleanup failures as operational warnings without rewriting the scientific result.

### Product UI

- `/runs` lists real persisted runs with policy-compute, verifier, judge, and cost
  summaries.
- `/runs/new` launches from typed configuration, not a raw JSON field.
- The launch form clearly labels the mock policy and mock judge providers.
- `/runs/:runId` shows lifecycle, attempts, resources, costs, deterministic metrics,
  judge metrics, reward signals, failures, logs, provenance, collection batches, and
  iterations.
- Each iteration page lists every rollout tree considered for the update, including
  exclusion reasons and the immutable iteration-input manifest.
- The rollout-tree explorer shows the shared prefix once, true branch fan-out, selected
  path, source and candidate artifacts, proof, verification components, judge results,
  rewards, and typed failures.
- A verification inspector shows the actual DAG, step attempts, evidence roles, cache
  state, deterministic results, judge model/specification, per-criterion output,
  confidence, abstention, disagreement, integrity flags, usage, and cost.
- Reference, source, candidate, and sibling render comparisons are synchronized.
- Selected state, branch group, verification run, verifier step, and filters survive
  refresh through the URL.
- The trajectory and verification graphs have keyboard-navigable outline or table views.
- Every route has loading, empty, error, degraded, hidden-evidence, and partial-data
  states.
- Automatic light and dark themes pass visual and accessibility checks.

### Engineering quality

- Formatting, linting, type checking, unit tests, integration tests, migration tests, and
  production builds pass.
- Crash, retry, duplicate-delivery, out-of-order delivery, ambiguous judge attempt,
  abstention, integrity, and cancellation tests cover durable boundaries.
- Tests prove that local profiles cannot resolve or contact real RunPod or judge
  providers.
- API and event schemas have one source of truth and generate or validate clients.
- Database constraints enforce the important scientific invariants.
- Required documentation and ADRs match shipped behavior.
- A new developer can run the entire acceptance flow from the README.

## Build order

Maintain `PLAN.md` with the current phase, completed evidence, next slice, and blockers.
Update it when a slice changes status. Do not use it as a diary.

### Phase 0: preflight, bootstrap, and contract decisions

1. Run the preflight before substantial scaffolding.
2. Initialize the repository and development toolchain.
3. Pin the root product specification and write its provenance sidecar.
4. Create `PRODUCT.md` from confirmed product facts.
5. Create `docs/reuse-inventory.md`.
6. Write the exact local CAD acceptance scenario.
7. Freeze the first schemas for state, runtime cursor, snapshot, decision checkpoint,
   collection batch, iteration input, verification plan, proof bundle, judge specification,
   judge result, metric observation, and reward signal.
8. Record the local Docker execution-boundary ADR.
9. Record the LLM-judge, calibration, prompt-injection, and local mock-provider ADR.
10. Choose the smallest stack that satisfies the specification.
11. Record unresolved choices and decision deadlines in `PLAN.md`.

Exit when:

- `scripts/preflight` verifies Docker and the required reviewer;
- the repository has a tested hello path in Docker Compose;
- the local profile cannot resolve real providers; and
- no unresolved decision blocks the first durable entity.

### Phase 1: durable control plane

Build:

- PostgreSQL migrations;
- content-addressed artifact storage;
- typed scientific and operational events;
- transactional outbox and cursor-based SSE;
- run, attempt, allocation, collection-batch, and iteration state machines;
- execution-operation authorization and scoped tokens;
- mock RunPod provider;
- detached mock run agent;
- desired-state cancellation;
- reconciliation; and
- compare-and-swap iteration commit.

Exit with a no-op run that survives orchestrator restart, accepts one policy successor,
and releases its mock allocation after cancellation.

### Phase 2: execution and verifier service

Build:

- workload-profile registry;
- durable jobs and attempts;
- leases, heartbeats, expiry, and fencing;
- Docker-backed sessions and runtime cursors;
- logical save and restore;
- environment snapshots and decision checkpoints;
- four-way fork operations;
- fidelity probes;
- verification-plan DAG runner;
- proof-bundle assembly;
- deterministic checks;
- `MockJudgeProvider` and structured judge parsing;
- resource-class capacity reporting;
- resource, network, and output limits;
- cancellation propagation; and
- recovery tests.

Exit when one logical snapshot produces four isolated child cursors after an
execution-service restart and a multi-node verification graph resumes without duplicating
accepted nodes.

### Phase 3: end-to-end local CAD flow

Build one bounded multi-turn CAD task before broadening the architecture.

Connect:

- mock policy;
- CAD action execution;
- semantic states and runtime cursors;
- canonical before/after rendering;
- geometry and constraint checks;
- proof-bundle creation;
- pointwise mock judge;
- branch plan and four-way fork;
- blinded group mock judge;
- metric observations;
- named progress and terminal rewards;
- dataset materialization; and
- iteration commit.

Exit when the API can reconstruct the exact state, action, proof, judge, reward, and
materialized data behind one committed update.

### Phase 4: dashboard, trajectory explorer, and proof inspector

Build the operator flow against real local APIs and seeded CAD runs. Do not design against
static mock objects that bypass contracts.

Build:

- typed run launch;
- run and resource monitoring;
- collection and iteration views;
- trajectory explorer;
- branch comparison;
- reference/source/candidate render comparison;
- verification DAG inspector;
- judge result and calibration metadata view;
- reward lineage;
- cancellation, restart, reproduce, and rejudge actions; and
- accessible outline alternatives for graph views.

Exit when a user can launch, monitor, inspect, cancel, restart, reproduce, and rejudge a
local CAD run through the UI.

### Phase 5: local hardening and future-contract proof

Add:

- full crash, retry, duplicate, out-of-order, timeout, abstention, disagreement, malformed
  judge, integrity, and cancellation suites;
- CAD environment, verification-plan, provider, and mock-judge conformance reports;
- retention and garbage collection required for local acceptance;
- debug bundle;
- documentation and clean-checkout verification; and
- one schema-level or fixture-level proof that the same verification graph can express a
  software task with build, tests, Playwright evidence, taste judge, and architecture
  judge nodes.

Do not implement the full software environment in this phase.

Exit when every local definition-of-done item passes from a clean checkout.

## Reuse-first workflow

Before implementing a substantial component:

1. Search this repository and `/Users/mohsen/code/equinox`.
2. Identify compatible libraries and existing code.
3. Record the decision in `docs/reuse-inventory.md` as `reuse`, `adapt`, or `replace`.
4. State evidence: contract fit, tests, coupling, security, calibration implications, and
   maintenance cost.
5. Reuse or adapt the smallest sound unit.

Do not copy a subsystem because it exists. Do not rewrite a component because the old
code lacks polish. Preserve strong ideas such as append-only lineage, branch groups,
artifact addressing, verifier evidence, React Flow tree interaction, explicit failure
classes, CAD execution, canonical rendering, and geometry checks.

Replace direct SQLite writes, process-local sessions, stdout protocols, base64 database
artifacts, CAD nouns in generic schemas, synchronous `/score`, direct judge-to-reward
plumbing, and any judge prompt that mixes trusted instructions with untrusted candidate
content.

Choose established libraries when they fit. Read official documentation and check
maintenance status before adoption. Build only product-specific branch, snapshot, proof,
judge, reward, and lineage behavior.

## Distributed-systems rules

Apply these rules at each service boundary:

- Model desired and observed state separately.
- Give logical work a stable ID and retries distinct attempt IDs.
- Record orchestrator intent before issuing an operation token.
- Bind operation tokens to run attempt, source state, expected version, profile, request
  digest, budget reservation, and expiry.
- Use leases with expiry, owner, heartbeat, and fencing token.
- Make retries bounded and classify retryable failures.
- Put timeouts on remote or provider calls and propagate cancellation.
- Use a transactional outbox for durable events.
- Make consumers idempotent and tolerate duplicate and out-of-order delivery.
- Reconcile provider and worker state after restart.
- Keep scientific outcome separate from cleanup outcome.
- Reserve and charge multidimensional budgets through an append-only ledger.
- Separate semantic task horizon from run-level resource budget.
- Use UTC timestamps for records and monotonic clocks for durations.
- Add backpressure before queues can grow without bound.
- Verify artifacts and proof bundles by digest before accepting a result.
- Do not reinvoke a successfully stored judge result during a normal retry.
- When a provider request has an ambiguous outcome and no provider idempotency support,
  retain every attempt and cost but accept at most one scientific result.
- Test process death between each durable transition.

Prefer explicit state machines and database constraints over conventions.

## Maintainability rules

- Keep domain logic outside transport handlers, ORM models, and React route components.
- Keep one source for statuses, schemas, units, metric descriptors, judge criteria, and
  artifact roles.
- Generate clients from OpenAPI or validate hand-written clients against it.
- Prefer composition and small modules over inheritance trees.
- Keep functions focused on one state transition, verification node, or calculation.
- Use typed domain objects at service boundaries.
- Keep provider adapters transport-only. Judge rubrics and reward semantics belong in
  versioned domain specifications.
- Store large and untrusted content as artifacts.
- Never pass arbitrary shell commands as workload profiles.
- Write migrations with forward and rollback or recovery notes.
- Document failure, abstention, and integrity behavior beside the public contract.
- Add tests at the level where a regression would occur.
- Delete dead code, superseded paths, and unused dependencies in the same slice.
- Update documentation in the commit that changes behavior.

## Frontend product and design rules

Use the
[Impeccable skill](/Users/mohsen/.agents/skills/impeccable/SKILL.md) for dashboard,
trajectory-explorer, and proof-inspector work.

The dashboard is an **Operate** surface. Researchers use it to make decisions, understand
why a reward exists, and debug expensive work.

Follow this sequence:

1. Run Impeccable context once for the target.
2. Use Impeccable `init` to write `PRODUCT.md` from the specification and confirmed
   constraints.
3. Use its new-work flow to choose the visual direction.
4. Write `DESIGN.md` before the first UI implementation.
5. Load the craft-floor playbook immediately before UI edits.
6. Build against live local API data.
7. Inspect desktop and laptop layouts in a real browser.
8. Run the Impeccable finish reviewer, address material findings, then run its detector
   once over finished targets.

The design rules:

1. Add content and controls that help the user decide or act. Do not add clocks,
   decorative statistics, filler copy, or status trinkets.
2. Use a flat visual system. Do not use drop shadows. Create hierarchy with borders,
   subtle background steps, spacing, typography, and scale.
3. A clickable-looking row, card, icon, or label must perform an action. Static surfaces
   must not imitate controls.
4. Use established building blocks. Prefer a maintained component system, uPlot or an
   equivalent chart library, xterm.js for terminal output, and React Flow for trajectory
   and verification graphs. Do not hand-roll primitives or SVG charts that a suitable
   library provides.
5. Define design tokens as CSS variables. Colors, spacing, radii, typography, borders,
   and motion use tokens rather than scattered literals.
6. Use one product accent. Keep semantic status colors separate. Use hue-biased neutrals
   instead of pure gray.
7. Follow the operating-system light or dark preference by default. Test both schemes.
8. Design each empty and degraded state. State what the surface contains, why data is
   absent, and the next useful action.
9. Meet WCAG 2.2 AA for primary workflows and keyboard navigation.
10. Keep selected state, branch, verification run, verifier step, and filters addressable
    in the URL.
11. Provide keyboard-navigable outline or table equivalents for canvas graphs.
12. Label judge output as a model assessment, show uncertainty and abstention, and never
    present it as objective truth.
13. Keep hidden rubrics, calibration cases, and hidden test artifacts out of browser
    payloads.

Use synthetic demo data when needed and label it. Do not invent customers, benchmark
results, costs, calibration performance, or capabilities.

## Skills

Create a small, repo-local skill set under `.agents/skills/`. Skills are development
tools, not application dependencies.

Install or vendor skills just in time rather than copying every possible skill during
bootstrap.

Use:

- `technical-writing` when writing the first ADR, README, API guide, runbook, or research
  note;
- `prompt-writing` when creating judge prompts, rubrics, structured output instructions,
  policy prompts, or agent instructions; and
- `impeccable` immediately before product UI work.

Source candidates:

- `impeccable` from `/Users/mohsen/.agents/skills/impeccable/`;
- `technical-writing` from
  `/Users/mohsen/thirdface/thirdface-assistant/.codex/skills/technical-writing/`; and
- `prompt-writing` from
  `/Users/mohsen/thirdface/thirdface-assistant/.codex/skills/prompt-writing/`.

Inventory existing local skills before downloading or creating another. Add a skill only
when it encodes a repeated workflow or review boundary that normal repository rules cannot
express well.

Useful missing skills may combine:

- distributed-system, failure, and idempotency review;
- LLM-judge calibration, prompt-injection, and reward-hacking review;
- simplicity, maintainability, and component-reuse review; and
- primary-source research and ADR writing.

Keep each skill small, testable, and tied to a workflow point. Record name, purpose,
source URL or local path, version or commit, license, and modifications in
`.agents/skills/README.md`. Review downloaded skill code before execution.

Documentation should lead with the decision or procedure, retain failure modes, and state
verification and rollback.

## Research rules

Research precedes decisions that affect public contracts, scientific integrity, trainer
correctness, isolation, judge reliability, external data handling, or long-term dependency
cost.

- Search recent literature and official documentation.
- Prefer primary sources: papers, specifications, maintained project docs, and source
  repositories.
- Compare at least two credible options for a consequential dependency or architecture
  choice.
- Record the decision, rejected option, evidence links, date, uncertainty, and exit
  trigger in an ADR.
- Verify claims against the version being installed.
- Keep a research spike bounded. If evidence does not distinguish options, choose the
  smaller reversible option and record the uncertainty.
- Do not research settled decisions from the specification again without new evidence or
  a measured blocker.

Before selecting a real judge provider for the later research profile, compare current
frontier multimodal models on the actual CAD calibration pack. Do not select a judge by
marketing claims or generic benchmark ranking.

Judge research must include position and order effects, repeated-sample consistency,
abstention, model and prompt drift, human agreement, textual and visual prompt injection,
and reward-hacking behavior.

Recent does not mean proven. Check release cadence, issue health, API stability,
maintenance ownership, license, data-retention terms, model pinning, and migration cost.

## Claude Code review before every commit

Each commit requires a read-only Claude Code review with the exact user-specified reviewer
configuration:

- model or profile: `Fable 5`;
- reasoning effort: `high`.

Do not silently substitute another reviewer. Before the first commit, verify that the
installed Claude Code can resolve this configuration. If it cannot, stop before committing
and report the unsupported configuration.

Create one repository script that invokes the reviewer with the verified local CLI
syntax. The script must review the pending staged diff without editing files or creating a
commit.

For each commit:

1. Keep the diff to one coherent, working slice.
2. Run format, lint, type checks, focused tests, affected integration tests, and the
   relevant build.
3. Stage intended files and inspect the staged diff.
4. Run Claude Code in read-only review mode over the staged diff, the specification, and
   relevant contracts.
5. Require findings to include priority, file and line, evidence, impact, and a concrete
   fix. Use priorities P0 through P3.
6. Address each P0, P1, and P2 finding. Address P3 findings that improve the slice without
   expanding scope; document the reason for any rejected P3.
7. Re-run checks.
8. Re-run review until no P0, P1, or P2 findings remain.
9. Stop after three review-and-fix passes if material findings remain. Report the blocker
   instead of training the reviewer to accept the code.
10. Commit with a conventional subject and trailers recording reviewer configuration,
    checks, and final finding count.

Example commit trailers:

```text
Reviewed-By: Claude Code (Fable 5, high)
Review-Findings: P0=0 P1=0 P2=0 P3=1-addressed
Checks: lint,typecheck,unit,integration,build
```

The review prompt must ask Claude to assess:

- correctness against `equinox-next.md` and the current local acceptance level;
- scientific authority and lineage ownership;
- data loss, races, fencing, retries, idempotency, cancellation, and recovery;
- state versus runtime-cursor separation;
- snapshot and decision-checkpoint correctness;
- verification DAG and reward provenance;
- judge output parsing, abstention, disagreement, calibration metadata, and ambiguous
  provider attempts;
- prompt injection, reward hacking, hidden-data boundaries, and external data exposure;
- API and schema compatibility;
- maintainability and reuse;
- over-engineering and avoidable dependencies;
- tests and missing failure cases; and
- UI behavior, accessibility, hidden evidence, and Impeccable rules when UI files change.

Claude reviews code. It does not approve a known failing check, edit the diff, replace
deterministic tests, or certify a mock judge as human-aligned.

## Commit and progress discipline

- Do not create placeholder commits.
- Do not mix refactors, dependency upgrades, and product behavior in one commit unless
  they cannot be separated.
- Keep the repository runnable after each commit.
- Commit generated files only when the chosen tool requires them for reproducible builds.
- Never commit credentials, local databases, object-store data, model weights, provider
  responses containing secrets, or review scratch files.
- Record deferred work in `PLAN.md` with an owner or trigger. Do not scatter TODO comments.
- Remove temporary debug instrumentation before review.
- Do not rewrite public history or use destructive Git operations.

If a slice grows beyond reviewable size, split it at a durable contract boundary.

## Rabbit-hole controls

- Start each slice with one acceptance statement.
- Time-box exploratory spikes. End them with code, an ADR, or a rejected option.
- Do not refactor unrelated code while implementing a feature.
- Do not perfect abstractions before the first complete CAD path uses them.
- Use provider interfaces for uncertain production integrations, but implement only the
  mock provider needed locally.
- Prefer one complete CAD environment over three partial environments.
- Keep deterministic hard checks separate from judge evidence; do not collapse everything
  into one score for convenience.
- Do not build judge ensembles before one well-specified mock invocation path works.
- Do not build full Playwright infrastructure before CAD acceptance; prove the future
  graph through schemas or fixtures only.
- Measure before optimizing.
- Stop adding diagnostics when existing evidence answers the operational question.
- Keep a short list of blockers. Do not create a speculative backlog.

## Before declaring completion

Verify:

- the full stack starts from a clean checkout through Docker;
- the local profile has no path to real RunPod or a real judge provider;
- independent and branch-aware CAD demo runs complete;
- every accepted CAD transition has source and candidate renders, geometry evidence, and
  a proof bundle;
- pointwise and group mock judge results use the production schemas and preserve model,
  prompt, rubric, input, uncertainty, usage, and attempt provenance;
- abstention, disagreement, malformed output, provider retry, and integrity paths do not
  become reward zero;
- a stored proof bundle can be rejudged without rerunning CAD execution or rendering;
- the exact rollout trees, proof, judge results, reward signals, and materialization behind
  an iteration are inspectable;
- duplicate, crash, restart, timeout, out-of-order, and cancellation tests pass;
- artifact, prompt-injection, hidden-data, and provider-boundary tests pass;
- light and dark UI states have browser evidence;
- graph views have keyboard-accessible alternatives;
- accessibility, lint, type, unit, integration, migration, and build checks pass;
- every commit received the required Claude Code review;
- documentation matches commands and behavior; and
- the README gives the user one clear next action.

Return a final handoff with the local URL, startup command, seeded CAD flow, rejudge flow,
test results, architecture decisions, known limitations, and the next production
integration: a calibrated strong multimodal judge in the CAD research profile. Do not
contact RunPod or a real judge provider.
