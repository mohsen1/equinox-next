# Branch-aware RL improvement plan

Status: working plan
Date: 2026-07-27

## Recommendation

Equinox should improve the existing branch-aware repository-repair loop before adding
microVM infrastructure or larger models. The next milestone is a replicated,
matched-budget study that demonstrates whether shared-prefix `K=4` continuations improve
held-out task completion without destabilizing the policy.

The platform should continue using deterministic logical snapshots and bounded
container execution. AgentENV is a relevant future sandbox backend, but process-memory
snapshots are not required for the current SQLite, filesystem, CLI, and
micro-repository tasks.

The immediate work is:

1. Separate action-protocol competence from strategic RL.
2. Make correctness the primary reward and tightly bound efficiency bonuses.
3. Stabilize sibling-relative optimization and stop on measured regression.
4. Expand dynamic complexity beyond a single scalar level.
5. Treat verifier design, environment diversity, and held-out evaluation as product
   features.
6. Make every branch, verifier decision, exclusion, curriculum change, and optimizer
   update observable.
7. Replicate the result before increasing model size, GPU count, or environment
   fidelity.

## Implementation status

Objective v3 implements most of phases A and B before the next paid run:

- a `99%` active-level held-out action-protocol gate;
- correctness-gated reward with a bounded `[-0.05, 0]` efficiency adjustment;
- pre-validation and best-adapter checkpoints;
- best-checkpoint final evaluation and rollback after two regression windows;
- stops for five consecutive uninformative groups or more than `5%` recent malformed
  actions;
- decomposed reward, effective batch weight, learning rate, loss, gradient norm, and
  failure-classification evidence;
- baseline-versus-current validation history, best-checkpoint state, curriculum
  decisions, active complexity dimensions, and provider reserves in the observer; and
- complete branch-inspector evidence for group eligibility, reward, and optimizer use.

The paid v3 run used the shared-prefix adaptive `K=4` condition. Protocol
initialization, harness diversity, durable pause and resume, the independent
matched-budget baseline, replicated seeds, and SQLite remain follow-up work. None of
those pending items is implied by the v3 receipt.

The seed `107` v3 run completed after its recent eight-group action window reached
`5.4264%` malformed actions. It retained the `5/8` baseline checkpoint, rolled back after
nine updates and five policy updates, and produced 24 unchanged paired test outcomes.
This proves the new stop and rollback path, but it does not prove learning. The run also
exposed two bounded follow-ups now included in objective v4: remember a no-signal
threshold crossing within a multi-group update, and clear stale evaluation counters and
heavy task outcomes from execution progress. Completed test rates also compare against
the matching per-level test baseline rather than the rotating validation baseline.

Objective v5 addresses the next measured confounds. The action contract now occupies the
model's actual system role instead of being embedded in user content. The shared
diagnostic prefix remains policy-generated but uses greedy decoding because sibling
credit cannot update a decision shared by all four continuations. Optimizer eligibility
now requires mixed hidden-correctness outcomes; small efficiency differences among four
solved siblings remain evidence but cannot consume a policy update. Each update samples
four frontier tasks to aggregate more independent correctness contrasts before applying
one optimizer step.

## What we learned

### From the current Equinox runs

The recent RunPod studies exposed several distinct failure classes:

- Strict parsing can produce a run with many completions but no valid policy actions.
  RL should not be expected to learn a basic JSON tool protocol from sparse terminal
  rewards.
- Evaluating every future curriculum level before training consumes the budget without
  improving the active policy. Initial calibration should evaluate only the active
  level.
- Sibling normalization can amplify an insignificant efficiency difference. A return
  difference such as `0.95` versus `0.94` must not create a full-scale advantage.
- Summing completion-token log probabilities makes longer serialized actions produce
  larger gradients for reasons unrelated to decision quality.
- A policy can receive valid updates and still regress on rotating validation. Successful
  optimization steps are not evidence of useful learning.
- Branch trees are necessary evidence. Aggregate reward and solve rate cannot explain
  which shared state was restored, how sibling behavior differed, or why an update was
  admitted.

The resulting objective revision uses a standard-deviation floor for sibling
normalization, mean completion-token log probability, and a lower adapter learning rate.
These controls need replicated evaluation rather than another single favorable
trajectory.

### From Kimi K3

Kimi K3 supports the following conclusions:

- Long-horizon RL depends on environment and rollout systems at least as much as on the
  optimizer.
- A supervised cold start establishes reliable tool use before RL develops higher-order
  reasoning and execution.
- The trainer samples a fixed group of `K` responses per prompt. Long-tail trajectories
  can be paused and resumed across optimization iterations instead of blocking the
  entire batch.
- Stale partial rollouts require bounded policy movement.
- Verifiable agent tasks should reward the final environment state rather than a model's
  claim that it finished.
- Public verifier feedback should be separated from hidden held-out verification.
- Reasoning and action budgets are curriculum dimensions. Exceeding a task-relative
  budget can override the reward.
- Training across tool schemas, system prompts, context strategies, skills, memories,
  and harnesses reduces scaffold overfitting.
- Specialized policies can be trained separately and consolidated later. Equinox does
  not need to solve multi-domain consolidation in the first study.

Kimi K3 does not establish that dynamic `K` is necessary. Its report also does not
disclose the value of `K`, the partial-rollout completion fraction, task-mixture weights,
or enough optimizer details for direct reproduction.

### From AgentENV

AgentENV shows a credible future implementation of exact process and filesystem
branching. It can pause, resume, snapshot, and fork Firecracker microVMs. That capability
would allow four continuations to inherit the same live processes, memory, filesystem,
and service state.

Equinox should not adopt it yet:

- The current tasks can be represented with deterministic logical state, filesystem
  copies, database files, or bounded containers.
- AgentENV requires Linux 6.8+, KVM access, and privileged host operations.
- Its multi-node control plane is explicitly a prototype.
- Adding a separate sandbox cluster now would increase operational risk without resolving
  the current scientific uncertainty.

Equinox should preserve a narrow sandbox-runtime boundary so AgentENV or an
E2B-compatible service can be added after a task requires real process-state
continuation.

## Research question

The next study should answer:

> Under a fixed action, token, task, model, and GPU budget, does sibling-relative
> optimization over four continuations restored from the same diagnostic state improve
> hidden-verifier task completion compared with independent sampling?

Secondary questions are:

- Does the policy solve tasks with fewer accepted actions after correctness is achieved?
- Does adaptive complexity preserve performance on mastered levels?
- Which branch states produce trustworthy non-zero learning signal?
- Does the policy generalize across task fixtures and harness variations that were not
  present during optimization?

The study must not claim broad software-repair generalization. The current environment
measures transfer within a declared repair grammar.

## Near-term system design

### Keep static K

Branch width remains `K=4` for the entire run. A trainable group requires:

- one policy-generated shared prefix;
- one immutable decision checkpoint;
- four continuations with distinct sampling seeds;
- identical task, model, tokenizer, policy revision, verifier revision, prefix tokens,
  and restored environment state;
- at least two trustworthy terminal outcomes; and
- an explicit eligibility decision.

Incomplete groups remain evidence but do not train unless the declared algorithm supports
the observed subset. Infrastructure failures are exclusions, not zero-reward candidate
outcomes.

### Keep environment state lightweight

Use the least expensive truthful snapshot for each environment:

| Environment | Current snapshot | Trusted verifier |
| --- | --- | --- |
| Micro-repository simulator | Canonical repository and transcript state | Hidden deterministic semantic probes |
| SQLite repair | Database file or transactionally consistent database copy | Hidden queries and invariant checks |
| Filesystem and CLI repair | Bounded directory tree, metadata, and declared process outputs | Hidden filesystem and command assertions |
| Real micro-repository repair | Filesystem copy or bounded container workspace | Tests outside candidate control |

Snapshot metadata must declare its fidelity. A filesystem copy must not be labeled a
process-memory snapshot.

### Separate protocol initialization from RL

Tool-call syntax and strategic behavior are different capabilities.

The recommended pipeline is:

1. Measure action-protocol validity before training.
2. If validity is below the launch threshold, run a small, separately identified
   protocol-initialization stage using verified action examples.
3. Freeze and digest the resulting cold-start adapter.
4. Run the branch-aware RL study from that adapter.
5. Attribute learning claims only to changes after the RL baseline.

Protocol initialization must have separate lineage and must not be described as
teacher-free RL. A study that claims policy-only learning can instead select a base model
that already passes the protocol threshold.

The launch target is at least `99%` parseable, schema-valid actions on a held-out protocol
set. Protocol evaluation must include all tools and representative argument boundaries.

### Use correctness-gated rewards

Rewards should be lexicographic:

1. Determine whether the hidden verifier accepts the final environment state.
2. Assign the correctness reward.
3. Apply a small bounded efficiency adjustment only among correct outcomes.

An initial reward contract is:

```text
invalid or unfinished state: 0
hidden-verifier success: 1
efficiency adjustment: [-0.05, 0]
```

The efficiency adjustment can account for accepted actions, verifier submissions, or
tokens. It must not make an incorrect branch outrank a correct branch. Tiny efficiency
differences must remain tiny after advantage normalization.

Record reward components separately:

- hidden correctness;
- public-verifier progress;
- accepted-action cost;
- token cost;
- verifier-submission cost;
- malformed-action penalty; and
- terminal aggregate.

### Bound policy updates

The optimizer must retain the current safeguards:

- equal sibling returns produce zero advantage;
- the sibling standard-deviation denominator has a non-zero floor;
- loss uses mean completion-token log probability;
- gradients apply only to accepted post-checkpoint policy actions;
- infrastructure failures and untrustworthy verifier results are excluded; and
- the learning rate, gradient norm, adapter revision, and effective batch weight are
  persisted for every update.

Add the following runtime protections:

- checkpoint the adapter before every validation window;
- retain the best validation checkpoint;
- stop or roll back after two consecutive validation windows show material regression;
- stop when five consecutive groups produce no trustworthy learning signal;
- stop when malformed actions exceed `5%` over the latest validation-sized window; and
- reserve enough provider time for paired final evaluation and teardown.

"Material regression" should initially mean a lower exact solve count on the same-size
held-out window, with no compensating improvement in the paired outcomes. Replace this
rule with a stronger statistical criterion after replicated data exists.

### Make complexity multidimensional

Dynamic complexity remains mandatory, but the controller should operate on declared
dimensions rather than an opaque level:

- file or table count;
- fault count;
- dependency depth;
- diagnostic ambiguity;
- hidden invariant count;
- required repair horizon;
- allowed action count;
- token budget;
- verifier submission budget; and
- amount of public verifier feedback.

Each task records its complete complexity vector. A named level is a versioned preset
over that vector.

Promotion requires two disjoint validation windows whose Wilson lower bounds exceed the
mastery threshold. Regression at a higher level should first increase replay of the most
recent mastered level. It should not silently lower the verifier standard.

The controller must expose:

- why a level was selected;
- which dimensions changed;
- evidence used for promotion or regression;
- replay probability;
- current mastery streak; and
- bounded minimum and maximum levels.

### Add cheap harness diversity

Before adding more infrastructure, vary inexpensive presentation details:

- tool names and ordering;
- equivalent JSON schemas;
- system-prompt wording;
- diagnostic file names;
- repository layouts;
- observation formatting;
- available public-verifier detail; and
- context truncation strategy.

Training and test configurations must be disjoint where practical. The result should
report performance by harness configuration so aggregate improvement cannot hide
scaffold overfitting.

### Support logical pause and resume

The current platform does not need VM pause and resume. It does need durable logical
continuation.

A resumable trajectory records:

- task and environment revisions;
- semantic state digest;
- transcript artifact;
- policy and tokenizer revisions;
- sampling state or seed;
- remaining action, token, time, and submission budgets;
- last accepted action;
- verifier state and revision;
- attempt and fencing token; and
- active, paused, resumed, excluded, or terminal status.

The scheduler can pause work between model calls or optimization iterations by persisting
this record and releasing the active execution slot. This exercises the partial-rollout
contract without requiring process-memory snapshots.

## Verifier design

Every environment should expose two verifier surfaces:

### Public verifier

The public verifier returns bounded diagnostic feedback that helps the policy recover.
Examples include:

- number of failing tests without hidden expected values;
- violated schema or database constraints;
- missing filesystem paths;
- command exit status; or
- build and type-check status.

Public feedback is part of the observation and therefore part of the policy input.

### Hidden verifier

The hidden verifier determines terminal correctness using held-out cases and invariants.
It runs outside candidate control and does not expose its task data, commands, or expected
outputs.

For every submission, record:

- verifier revision;
- candidate state digest;
- public and hidden result digests;
- accepted candidate failure versus verifier or infrastructure failure;
- submission index and remaining budget; and
- any integrity violation.

Limited submissions and hidden cases reduce reward hacking. A model's `finish` action is
only a request for verification, not evidence of completion.

## Environment sequence

### Phase 1: micro-repository simulator

Use the existing simulator to establish optimizer stability and replicated signal. Do not
expand the repair grammar during the replication study.

Exit criteria:

- three completed optimization seeds;
- paired baseline and final evaluation on identical held-out tasks;
- no protocol or infrastructure confound;
- complete K-branch lineage;
- positive median held-out exact-solve gain; and
- study-level reporting of improvement, regression, cost, and confidence.

### Phase 2: SQLite data repair

SQLite is the first real stateful environment because it provides cheap snapshots and
strong deterministic verification.

Initial tasks should cover:

- repairing values under cross-table invariants;
- restoring missing rows from redundant evidence;
- correcting schema or index defects;
- resolving transactionally inconsistent state; and
- producing a required query result without modifying protected data.

The policy receives an allowlisted SQL and inspection interface. The hidden verifier uses
held-out queries, integrity checks, and protected-table digests.

### Phase 3: filesystem and CLI repair

Add a bounded filesystem with allowlisted commands. Start with deterministic tools and no
network access.

Initial tasks should cover:

- incorrect permissions or paths;
- malformed configuration;
- broken command pipelines;
- missing generated artifacts;
- inconsistent manifests; and
- service configuration validated without launching an uncontrolled daemon.

Record stdout, stderr, exit status, filesystem diffs, and resource bounds as typed
evidence.

### Phase 4: real micro-repository repair

Replace the simulator with generated small repositories and real tests in bounded
containers. Candidate code remains isolated from the hidden verifier and platform
credentials.

This phase tests whether the learning result survives real parsing, execution, build,
test, and filesystem behavior. It does not require persistent VMs.

## Matched-budget experiment

Run at least three optimization seeds for each admitted condition:

| Condition | Prefix | Credit | Complexity | Replay |
| --- | --- | --- | --- | --- |
| Independent baseline | Independent | Group-relative or declared baseline | Fixed | No |
| Shared-prefix branching | Shared and restored | Sibling-relative | Fixed | No |
| Adaptive branching | Shared and restored | Sibling-relative | Adaptive | No |
| Adaptive branching with replay | Shared and restored | Sibling-relative | Adaptive | Yes |

Hold constant:

- base and cold-start adapter;
- model and tokenizer revisions;
- task identities and splits;
- verifier and action-protocol revisions;
- K, sampling parameters, and action budget;
- number of sampled actions or equivalent rollout budget;
- optimizer and effective learning-rate control;
- validation schedule;
- maximum GPU time; and
- final evaluation reserve.

Primary outcome:

- paired hidden-verifier exact solve rate.

Secondary outcomes:

- paired improvements and regressions;
- actions and tokens per solved task;
- malformed-action rate;
- trustworthy informative-group rate;
- curriculum promotions and mastered-level retention;
- wall time and GPU cost;
- infrastructure exclusion rate; and
- verifier-submission count.

Report each seed and the aggregate. Do not promote an exploratory single-seed gain into a
general claim.

## Observer requirements

The dashboard should answer four questions quickly:

1. Is the run healthy?
2. Is the policy learning or regressing?
3. Why did a specific group produce a policy update?
4. Can the complete result be trusted?

### Run overview

Keep the overview sparse. Show:

- status and current phase;
- current update and policy-update count;
- active complexity preset;
- latest validation solve rate and interval;
- baseline comparison;
- informative-group and malformed-action rates;
- GPU, elapsed time, estimated cost, and remaining reserve; and
- teardown status.

Do not add prose that repeats the page title or self-evident labels.

### Branch explorer

Show:

- the shared prefix once;
- the decision checkpoint and state digest;
- four continuation lanes;
- accepted and rejected actions;
- observations and verifier feedback;
- terminal state and reward components;
- sibling-relative advantage;
- policy-signal eligibility;
- exclusion reason;
- the optimizer update that consumed the branch; and
- parent and child runtime identifiers when real sandboxes are introduced.

The graph and outline views must expose the same facts. URL state should preserve the
selected update, sibling, and action.

### Curriculum view

Show complexity as a timeline of validation windows and controller decisions. A
promotion should link to the exact evidence that caused it. A regression or replay event
should show the affected dimensions rather than only a numeric level.

### Failure classification

Classify failures separately:

- malformed policy action;
- valid candidate failure;
- premature finish;
- verifier rejection;
- verifier or infrastructure failure;
- unproductive action loop;
- final-stage incompletion;
- strategy persistence despite repeated negative feedback;
- insufficient verification before finish; and
- provider interruption or teardown failure.

These categories should be computed from typed events where possible and marked as
heuristic when inferred.

## Implementation phases

### A. Stabilize the scientific loop

- Finish the objective-v2 same-seed comparison.
- Add best-checkpoint retention and validation-triggered stop or rollback.
- Add protocol-validity and recent malformed-action metrics.
- Persist decomposed rewards and effective optimizer weights.
- Verify that tiny efficiency differences remain bounded in stored advantages.

### B. Complete the observer

- Add baseline-versus-current validation history.
- Add curriculum decision history.
- Expose reward decomposition and group eligibility in branch detail.
- Add typed failure categories and loop indicators.
- Show remaining training, evaluation, and provider reserves.

### C. Run the replicated study

- Freeze revisions and task splits.
- Complete three seeds for the shared-prefix condition.
- Run the independent matched-budget baseline.
- Produce the aggregate study report.
- Decide whether adaptive complexity and replay deserve separate ablations.

### D. Add SQLite

- Define the action, snapshot, verifier, and complexity contracts.
- Build deterministic task generation and held-out invariants.
- Pass local conformance and reward-hacking tests.
- Run policy-free fixtures before spending GPU time.

### E. Add filesystem and real repository execution

- Introduce a bounded container backend.
- Keep the hidden verifier outside the candidate container.
- Disable network and credentials by default.
- Add resource, output, path, and process limits.
- Prove teardown and duplicate-delivery recovery.

### F. Revisit AgentENV

Consider AgentENV only when at least one condition holds:

- a task depends on live process or memory state that cannot be reconstructed cheaply;
- filesystem copying dominates rollout time or storage;
- container isolation blocks necessary agent behavior;
- logical resume cannot preserve task correctness; or
- measured concurrency requires fast snapshot-backed density.

Adopt it behind the sandbox-runtime contract. Do not make the trainer or scientific
schema depend on AgentENV-specific identifiers.

## Verification plan

Every implementation phase must include:

- unit tests for reward, advantage, curriculum, and stop conditions;
- property or boundary tests for task generation and verifier invariants;
- duplicate, stale, retry, and exclusion tests;
- paired baseline/final evaluation tests;
- dashboard tests for partial, live, failed, and completed runs;
- browser checks for branch and curriculum navigation;
- provider teardown checks; and
- a receipt proving model, policy, task, verifier, objective, cost, and artifact lineage.

Before a paid run:

1. Run the complete repository gate.
2. Confirm the dashboard can observe the proposed event revision.
3. Confirm no provider resources are active.
4. Freeze the git commit and workload configuration.
5. Verify the evaluation and teardown reserves.

After a paid run:

1. Confirm final or explicitly partial evidence was ingested.
2. Confirm every admitted update links to its branch groups.
3. Confirm paired evaluation counts match the declared task set.
4. Confirm the adapter and receipt digests.
5. Confirm provider teardown and zero ongoing spend.

## Risks and unresolved questions

- The current repair grammar may be too narrow for RL gains to generalize beyond fixture
  variations.
- Fixed K can waste generation when sibling outcomes are consistently identical, but
  dynamic K should wait until the fixed-K study establishes a baseline.
- Protocol initialization may improve tool reliability while changing the policy enough
  to complicate comparisons. Its adapter and evaluation must remain separate.
- Public verifier feedback can become an unintended shortcut. Harness and feedback
  variations should test reliance on superficial messages.
- Validation windows are small and noisy. Best-checkpoint and stop rules are operational
  safeguards, not substitutes for replicated evaluation.
- Logical snapshots cannot preserve arbitrary running processes. The real repository
  phase must keep tasks reconstructible until a stronger sandbox backend is justified.
- A single GPU run can prove execution and evidence contracts but cannot establish a
  broad learning claim.

## Sources

- [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf),
  especially sections 4.1, 4.2, and 5.3.
- [AgentENV repository](https://github.com/kvcache-ai/AgentENV).
- [AgentENV sandbox lifecycle](https://github.com/kvcache-ai/AgentENV/blob/main/docs/src/concepts/sandboxes.md).
- [AgentENV snapshot model](https://github.com/kvcache-ai/AgentENV/blob/main/docs/src/concepts/snapshots.md).
- [ADR-0004: Environment catalog and adaptive complexity](adr/0004-environment-catalog-and-adaptive-complexity.md).
- [ADR-0005: Restored branching with repository repair](adr/0005-repository-repair-research-profile.md).
