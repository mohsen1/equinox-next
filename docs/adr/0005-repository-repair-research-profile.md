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

This protocol pins environment revision `repository-repair-simulator@3` and verifier
revision `repository-repair-hidden-state@3`.

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
- Hidden-verifier correctness is lexicographic. Incorrect or unfinished trajectories
  receive zero terminal reward. Correct trajectories receive `1.0` minus an
  accepted-action adjustment capped at `0.05`.
- Equal sibling returns produce exactly zero relative advantage.
- The sibling standard-deviation denominator has a `0.1` floor, so small
  action-cost differences cannot be amplified into full-scale policy signals.
- A sibling advantage is applied to every accepted post-branch action in that sibling.
- Policy loss uses mean completion-token log probability, avoiding larger gradients
  merely because an action serialized to more tokens.
- The optimizer consumes policy samples only; no teacher targets or fallback updates are
  allowed.
- The active-level held-out baseline must show at least `99%` parseable, schema-valid
  actions before strategic RL begins.
- The adapter is checkpointed before each validation window. Equinox retains the best
  exact-solve checkpoint for the active level and uses it for final evaluation.
- Two consecutive validation windows below the retained exact-solve count stop training
  and restore the best checkpoint. Five consecutive groups without trustworthy relative
  signal also stop training. A complete recent window above `5%` malformed actions stops
  the run before another optimizer update.
- Rotating validation windows control curriculum decisions. A level advances only after
  two consecutive windows whose 95% Wilson lower bounds exceed the mastery threshold for
  both exact solve rate and checkpoint rate. Validation-window seeds use a `1,000,003`
  stride. Exact fixture IDs include the seed, so they are not the independence unit.
  The workload also records a seed-free semantic identity derived from level, split,
  and normalized fault families. It excludes the preceding window's semantic identities
  before inference and rejects sample sizes larger than the available semantic universe.
  The profile caps mastery at two windows because one-fault validation has exactly two
  disjoint eight-task halves.
- The disjoint test split is evaluated once, after training, on the same retained tasks
  for the disabled-adapter baseline and final adapter. The result records Wilson
  intervals and paired improvement and regression counts. The exploratory hypothesis
  gate requires a positive gain and a paired exact McNemar p-value below `0.05`.
- The inert semantic verifier uses explicit boundary cases plus task-identity-keyed
  probes for open-domain numeric, string, sequence, and mapping families. Boolean
  families use their exhaustive finite truth tables. It never executes candidate
  code. These probes reduce trivial
  fixed-value reward hacking but remain finite, so solve rate is evidence against this
  declared verifier rather than a proof of full Python semantic equivalence.
- Checkpoints retain cumulative elapsed time and attempt count. Resume consumes the
  remaining run budget; it never refreshes the target runtime. A fenced attempt-2 receipt
  remains eligible as the same logical run when checkpoint provenance and cumulative
  elapsed time are present. The remote runner makes one bounded
  retry when a failed workload has a valid latest-checkpoint pointer; failures before
  the first checkpoint remain terminal. Its attempt counter is persisted in the remote
  work directory so a container restart cannot reset the two-attempt budget or relabel
  a resumed workload. The launcher provisions a 10 GB pod volume at `/workspace` and
  places the attempt counter, checkpoints, pending result, and runner bundle under
  `/workspace/equinox-state`; this fence therefore survives container restarts within
  the bounded pod lifetime. A completed pending result is promoted on restart before any retry,
  and failure before the first checkpoint remains distinct from exhausted retry budget.
  The provider lifetime includes a separate
  2,700-second retry reserve. The measured final-evaluation reserve is capped separately
  at 2,700 seconds. Exceeding that ceiling stops training and proceeds to final
  evaluation; the receipt retains the unclamped measurement and the ceiling event.
  Resumed accounting includes the wall-clock tail after the latest checkpoint, capped
  by the declared 2,700-second retry reserve to bound cross-host clock skew. The raw
  and applied gaps remain in the checkpoint and result for audit. Sampled actions after
  the last durable checkpoint cannot be reconstructed after a crash; resumed receipts
  mark those crash-tail actions as unaccounted while retaining their cost in wall-clock
  time. Final evaluation has
  its own workload deadline. If it expires, the workload persists a
  partial, ineligible receipt with observed and expected sample counts instead of
  relying on provider termination; partial receipts do not report a cross-level reward
  gain. The evaluator checks its deadline before every model generation, and the
  provider ceiling retains a 900-second allowance for the one generation already in
  flight. Collections completed after the training deadline
  are not optimized, but their groups and sampled actions remain in cost accounting and
  the terminal checkpoint.

## Observability contract

Before launch, the dashboard must show the shared prefix once, the decision checkpoint,
four continuation lanes, every action and observation, terminal verifier results,
returns, sibling advantages, exclusions, complexity level, replay status, and the update
that consumed the trajectories.

The run overview also shows the baseline and rotating validation history, best retained
checkpoint, regression streak, action-protocol validity, recent malformed-action rate,
policy-update count, active complexity vector, and remaining training and evaluation
reserves. Branch detail records hidden correctness, public-verifier progress, bounded
action cost, malformed actions, verifier submissions, completion tokens, effective batch
weight, learning rate, loss, gradient norm, and typed or explicitly heuristic failure
classification.

Research execution JSON is the initial persistence boundary because the current
orchestrator contracts contain CAD-specific schema constants. The result must retain
complete step lineage. Generalizing the core scientific schema is a follow-up informed by
this vertical, not a semantic shortcut inside CAD records.

The retained branch tree is selected from the current non-replay frontier, preferring an
informative non-excluded group. The serialized tree names this selection rule. It does
not prefer groups by sibling success count.

## Experiment contract

The first launch is a bounded, explicitly single-seed exploration. A replicated claim
requires at least three distinct optimization seeds summarized by
`scripts/summarize-runpod-study`. The summarizer rejects mixed models, revisions,
training configurations, runtime budgets, complexity levels, or test task identities.
Later matched-budget ablations compare sampling-only, sibling-relative, adaptive, and
replay variants using the same tasks, seeds, action budget, model revision, tokenizer
revision, verifier revision, and effective learning-rate control.

Headline metrics are final-test solve rate, actions to solve, trustworthy informative-group
rate, regression on mastered levels, and GPU cost. Scaling the model is conditional on
restored branching producing useful signal.

## 2026-07-27 protocol revision

The first 1.5B run verified restored branching and produced useful sibling-relative
signal, but its learning claim was confounded by teacher fallback and four-example
evaluation sets. Its perfect final rate is retained as historical evidence, not treated
as a generalization result.

The next bounded run uses the pinned 3B model revision on one A40. It removes teacher
updates, doubles current-level task collection per update, uses eight-example rotating
validation windows, retains six test tasks per level for paired baseline and adapter
evaluation, and caps the workload at 120 updates or 7,200 cumulative seconds. Initial
reserve calibration evaluates only the current curriculum level; later levels are
evaluated when reached and remain present in the final paired test. The run
has a 240-minute provider ceiling so boot, one retry, final-evaluation overrun, and
teardown do not compete with the 7,200-second workload budget. It remains a
single-seed exploration and cannot
establish a replicated gain. Validation has 16 distinct one-fault semantic cases and
test samples six of 12; larger multi-fault levels draw unique combinations from those held-out
families.

The split is disjoint by fixture family ID, not by abstract program transformation.
Known structural mirrors remain across splits: `first`/`safe_head`,
`nonempty`/`is_empty`, `minimum`/`maximum`/`bounded_lower`/`maximum_three`,
`last`/`middle`, `coalesce`/`default_zero`, and simple arithmetic operator
repairs such as `combine`/`multiply`/`subtract`/`square`. Boolean operator repairs
also mirror across `different`/`negate`/`both`. This run therefore measures transfer to
unseen fixtures within a narrow repair grammar; it is not evidence of broad
repository-repair generalization.

## 2026-07-27 stability revision

Objective v2 removed amplified efficiency gradients but did not produce monotonic
validation improvement. With seed `107`, exact solve moved from a `62.5%` active-level
baseline to `75%`, `50%`, `62.5%`, `50%`, and `75%` at updates 5 through 25. This was
materially more stable than objective v1, which fell to `12.5%` at updates 15 and 20,
but v2 still trained past its best observed checkpoint.

Objective v3 is `leave-one-out-correctness-gated-reinforce@3` under workload revision
`runpod-repository-repair-loo-reinforce@9`. It adds correctness-gated reward, durable
pre-validation and best-adapter checkpoints, best-checkpoint final evaluation,
validation-regression rollback, no-signal and malformed-action stops, protocol validity,
reward decomposition, optimizer provenance, curriculum decision history, and provider
reserve telemetry.

The seed `107` v3 run completed on an A40 after nine updates and five admitted policy
updates. The rotating update-5 window remained at the `5/8` baseline. The latest
eight-group action window then reached a `5.4264%` malformed-action rate and stopped
training. Equinox restored update 0 and completed a paired 24-task test evaluation with
zero improvements, zero regressions, and zero aggregate reward gain. This run validates
the protocol stop, rollback, paired evaluation, artifact persistence, and provider
teardown paths. It is not evidence of policy improvement. Estimated cost was `$0.197841`.

Objective v4 is `leave-one-out-correctness-gated-reinforce@4` under workload revision
`runpod-repository-repair-loo-reinforce@10`. It remembers a no-signal threshold crossing
even if a later group in the same update is informative, clears transient evaluation
counters at phase boundaries, compares test results only with the matching test
baseline, and keeps completed-run validation summaries lightweight.

Objective v5 is `leave-one-out-correctness-contrast-reinforce@5` under workload revision
`runpod-repository-repair-loo-reinforce@11`. It places standing tool policy in the
tokenizer's system role, makes the untrained shared prefix greedy, and admits an optimizer
step only when a sibling group contains both solved and unsolved continuations. Four
frontier tasks per update replace two so each optimizer step aggregates more independent
correctness contrasts. Efficiency-only differences remain evidence but no longer move
the policy during this exact-solve milestone.

The v2 run ended at update 26 when a local API restart interrupted the ingestion
transport. The launcher fail-safe deleted the RunPod worker. Equinox retains the run as
partial evidence with `LOCAL_INGESTION_TRANSPORT_INTERRUPTION`, confirmed teardown, and
zero ongoing provider spend. It is not a completed paired evaluation.

## Non-goals

- Dynamic branch width.
- AWS execution.
- Arbitrary repository or shell execution.
- A production RunPod provider inside the local fixture registry.
- Recasting the completed CAD fixture as the repository environment.
