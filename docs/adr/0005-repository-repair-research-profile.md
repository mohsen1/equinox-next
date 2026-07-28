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

This protocol pins environment revision `repository-repair-simulator@4` and verifier
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

The seed `107` v5 run completed ten optimizer updates with 16 informative groups out of
40. The held-out protocol gate passed at `100%`, the final recent malformed-action rate
was `2.521%`, and the run stopped after five consecutive uninformative groups. A rotating
validation window scored `6/8`, but the retained adapter changed none of the 24 paired
test outcomes. The run therefore demonstrates denser trustworthy signal and a clean
no-signal stop, not model improvement. Estimated cost was `$0.145249`.

Objective v6 is `leave-one-out-paired-validation-reinforce@6` under workload revision
`runpod-repository-repair-loo-reinforce@12`. It raises the clipped LoRA learning rate from
`2e-5` to `8e-5`, extends the no-signal limit to twelve groups, and evaluates both a fixed
paired checkpoint-selection set and a separate rotating curriculum set. A random
curriculum window can no longer become the retained checkpoint merely by scoring higher
than an incomparable baseline window.

The seed `107` v6 run applied three optimizer updates from five informative groups out of
twelve. Its rolling eight-group action window then reached a `7.2848%` malformed-action
rate, exceeding the `5%` safety limit. Equinox rolled the adapter back to update 0 and
completed the 24-task paired evaluation with zero improvements and zero regressions. The
run validates the protocol safety stop and rollback, but it is not evidence of model
improvement.

Workload revision `runpod-repository-repair-loo-reinforce@13` keeps objective v6 and the
protocol limit unchanged. It repeats the prefilled JSON completion contract immediately
at each generation boundary and calibrates stochastic sibling sampling from temperature
`0.8`, top-p `0.95` to temperature `0.6`, top-p `0.9`. Static K remains four; the lower
entropy targets schema drift while retaining independent seeded continuations.

The seed `107` revision-13 run reduced the rolling malformed-action rate to `1.6807%`
through update 4, but the expanded reminder also reduced the fixed greedy validation
baseline from `5/8` to `1/8`. The operator stopped the run before checkpoint selection so
that later gains could not be measured against an artificially weakened baseline. The
worker was deleted and teardown confirmed.

Workload revision `runpod-repository-repair-loo-reinforce@14` restores the v6 prompt
verbatim and retains only the lower-entropy stochastic sibling sampling. This isolates
protocol calibration from greedy task competence.

The seed `107` revision-14 run retained update 10 after fixed paired validation improved
from `5/8` to `6/8`. A later candidate fell to `4/8`, so Equinox restored update 10
before the final paired 24-task evaluation. The retained adapter improved `square` and
regressed `is_empty`, producing one paired improvement, one regression, and zero net
reward gain. The run proves that branch-relative optimization can change held-out
behavior and that best-checkpoint rollback works, but it does not meet the
zero-regression improvement gate.

Workload revision `runpod-repository-repair-loo-reinforce@15` binds measured
final-evaluation reserve to the retained checkpoint. Revision 14 committed timing from
every evaluated candidate before checkpoint selection. A slow rejected seed-109
candidate therefore raised the reserve to its ceiling and stopped training after four
policy updates even though final evaluation would have restored the faster update-0
adapter. Revision 15 records candidate timing for observability but changes the active
reserve and ceiling state only when that candidate becomes the retained checkpoint.

The corrected seed `109` revision-15 run confirmed the reserve fix in the live GPU path.
Its update-5 candidate tied the `5/8` baseline and was rejected; the measured reserve
remained at 1,303 seconds instead of rising to the configured 1,500-second ceiling, and
training continued. Four policy updates were applied through update 8. The next
eight-group protocol window crossed the `5%` malformed-action threshold, so Equinox
rolled back to update 0. A transient observer PUT then timed out during the final test
phase; fail-safe teardown removed the pod, but the result was not eligible for proof
ingestion.

Workload revision `runpod-repository-repair-loo-reinforce@16` keeps the `5%` malformed
threshold and requires it to be exceeded in two consecutive complete rolling windows
before rollback. A single noisy update is recorded as a strike and must be followed by
recovery; sustained drift still stops training. The launcher also deduplicates unchanged
progress, retries idempotent observer updates, and continues remote monitoring if a live
progress publish exhausts its retries.

The seed `109` revision-16 run applied five policy updates from ten informative groups
out of 40. Update 5 tied the `5/8` fixed baseline. Updates 9 and 10 then produced two
consecutive malformed-action breaches; the final window contained 31 malformed actions
out of 83, and aggregate protocol validity fell to `94.5141%`. Equinox stopped, restored
update 0, persisted the adapter and receipt, completed 24 paired test cases with no
changes, and confirmed provider teardown. Estimated cost was `$0.341041`. This rules out
a one-window safety artifact and identifies sustained action-language collapse.

Objective v7 is `leave-one-out-reference-anchored-reinforce@7` under workload revision
`runpod-repository-repair-loo-reinforce@17`. It keeps the v6 correctness-contrast
objective and adds a `0.02` sampled K3 reverse-KL penalty between the active adapter and
the disabled-adapter base policy on accepted completion tokens. Every optimizer update
records the REINFORCE term, sampled KL, coefficient, and combined loss. The base model
already passes the greedy protocol gate, so this regularizer is intended to preserve its
action language while sibling-relative rewards modify repair choices.

The seed `109` revision-17 run showed that a continuation-only coefficient of `0.02` is
too weak. Five policy updates were applied. Sampled KL rose from `0.011714` at update 7
to `0.072603` at update 8; the next two updates could not establish their greedy shared
prefix. The final complete eight-group window contained 40 malformed actions out of 40,
and aggregate protocol validity fell to `91.9463%`. Equinox stopped after two breaches,
restored update 0, produced 24 unchanged paired test outcomes, persisted the proof, and
confirmed teardown. Estimated cost was `$0.395688`.

Objective v8 is `leave-one-out-full-trajectory-anchor-reinforce@8` under workload
revision `runpod-repository-repair-loo-reinforce@18`. It lowers the adapter learning rate
to `4e-5`, raises the K3 log-ratio coefficient to `0.1`, and anchors every accepted
policy action, including the greedy shared prefix and actions from otherwise
uninformative groups. A no-signal batch can therefore apply a restorative reference-only
optimizer step while policy-update counts remain limited to batches with sibling credit.
The observer distinguishes policy and anchor examples and publishes the terminal
malformed-window count instead of retaining stale pre-finalization telemetry.

The v2 run ended at update 26 when a local API restart interrupted the ingestion
transport. The launcher fail-safe deleted the RunPod worker. Equinox retains the run as
partial evidence with `LOCAL_INGESTION_TRANSPORT_INTERRUPTION`, confirmed teardown, and
zero ongoing provider spend. It is not a completed paired evaluation.

The completed seed `109` revision-18 run applied 25 optimizer steps, 21 with sibling
policy credit. The fixed checkpoint improved from `5/8` to `6/8`, aggregate action
validity remained `97.3556%`, and no malformed-action strike remained at termination.
The 24-task paired test moved in the wrong direction: zero improvements, two
regressions, and reward `0.416666` to `0.333334`. The hypothesis gate rejected the run.
The retained adapter and manifest were persisted, the pod was deleted, ongoing spend
returned to zero, and estimated GPU cost was `$1.096833`.

Objective v9 is `leave-one-out-retention-guarded-reinforce@9` under workload revision
`runpod-repository-repair-loo-reinforce@19` and environment revision
`repository-repair-simulator@4`. It keeps static `K=4`, the `4e-5` learning rate, and the
full-trajectory base-policy anchor. It changes four measured weak points:

- thirteen disjoint training fixture families cover the held-out repair grammar through
  disclosed structural analogues;
- each four-task batch keeps two active-frontier groups and probes up to two harder
  complexity levels;
- absolute sibling advantage is capped at `1.0`, and policy credit requires two
  independent informative groups in the batch; and
- a candidate checkpoint must preserve the fixed window, improve either the fixed or
  rotating window, and have zero paired rotating regressions against the
  disabled-adapter base before it can replace the retained checkpoint.

The observer retains one active-frontier branch tree and one complexity-probe tree per
update, reports the highest sampled level, and labels guard rejection explicitly.
Final hypothesis acceptance additionally requires zero paired test regressions.

The first seed `113` revision-19 execution was stopped after update 3. Each update
contained one informative group, so the minimum-two-groups gate applied only the
reference anchor and discarded the policy signal. No policy update occurred. The run
was not admitted as proof; its worker was deleted and provider spend returned to zero.

Objective v10 is `leave-one-out-accumulated-retention-reinforce@10` under workload
revision `runpod-repository-repair-loo-reinforce@20`. It preserves sparse informative
groups across update boundaries until at least two distinct task identities are
available. Pending weighted actions and task IDs are part of the durable training
checkpoint. A pending group's base-policy anchor and bounded REINFORCE weight are queued
without changing model weights until a later distinct group completes the policy batch.
This keeps the queued trajectories on-policy. Duplicate identities are rejected.
Branch evidence records the source task IDs and the optimizer update that consumes
earlier pending credit.
Static `K=4`, adaptive complexity, and all revision-19 retention and final-evaluation
gates remain unchanged.

The first revision-20 provisioning attempt was terminated before workload handoff when
RunPod's proxy returned `404` for bundle posts to its root despite serving authenticated
bootstrap health. A second attempt showed the same behavior on health, dedicated, and
root POST paths. Both workers were deleted and provider spend returned to zero. The
launcher now sends a compressed bundle capped at 256 KiB in the encrypted pod creation
request. The bootstrap decodes it, validates the 2 MiB hard size bound and exact file
allowlist, installs it durably, removes it from the child process environment, and
executes the runner. The authenticated HTTP upload remains a fallback for operators,
but paid automation no longer depends on proxy POST behavior.

The completed seed `113` revision-20 run stopped safely at update 17 after two recent
malformed-action windows exceeded `5%`. It rolled back to update 15, persisted a verified
adapter, and completed all 48 paired held-out tasks. The result was neutral rather than
successful: `17/48` solves before and after, two paired improvements, two paired
regressions, and a failed hypothesis gate. RunPod teardown was confirmed and ongoing
spend returned to zero.

Objective v11 is `verified-success-accumulated-retention-policy-gradient@11` under
workload revision `runpod-repository-repair-verified-success@21`. It replaces signed
whole-trajectory credit with positive credit on accepted actions from successful
siblings in a mixed-outcome group. Failed siblings remain recorded but receive zero
policy weight. Two distinct informative groups remain the atomic on-policy batch, the
learning rate is `2e-5`, and base-policy KL is `0.2`.

The retained-checkpoint guard now evaluates fixed paired suites for the active and next
complexity levels, as well as the disjoint rotating active-level suite. Any paired
regression rejects the candidate; a retained candidate must improve one guard without
losing on the other. Static `K=4`, adaptive complexity, durable pending groups, bounded
resumption, final paired evaluation, and verified teardown remain unchanged.

The seed `113` revision-21 run stopped after both rotating guards regressed by one task
without an improvement. The fixed level-0 and level-1 suites were unchanged, update 0
was restored, and all 48 final test pairs were unchanged. This isolated a narrower
credit-assignment defect: positive weight was divided across successful repair, test,
and finish actions even though only fault-fixing edits caused the verified state change.

Objective v12 is `verified-fix-accumulated-retention-policy-gradient@12` under workload
revision `runpod-repository-repair-causal-credit@25`. A mixed-correctness group is still
required, but positive policy credit is limited to accepted edits that increase the
verified fixed-fault count on a sibling that ultimately solves the task. All accepted
actions—including diagnostics, tests, finishes, failed-sibling actions, and no-signal
groups—remain under the disabled-adapter reference anchor.

The task sampler uses only active validation failures to resolve declared structural
analogues in the train split. It then covers those analogue families across the active
frontier and adjacent complexity probes. Final test outcomes never influence sampling,
checkpoint selection, or optimization. The result records both the observed validation
failure families and the selected train families so this adaptive curriculum decision is
auditable.

Revision 22 was stopped at update 1 when live inspection showed that serialized sibling
weights still used all accepted continuation actions even though optimization used only
fault-fixing edits. Revision 23 removes that independent calculation. Optimizer inputs,
sibling eligibility, step-level policy signal, and effective weights now share the same
causal-credit function.

Revision 23 retained update 15 with two fixed-guard improvements, two rotating-guard
improvements, and zero regressions. Final paired test completion improved from `17/48`
to `21/48` with zero regressions, but four discordant improvements yield `p=0.125` and
do not pass the hypothesis gate. Revision 24 changes only dynamic-complexity accounting:
mastery is accumulated across retained zero-regression checkpoints. A rejected
transient candidate does not reset evidence attached to the still-retained policy.

The seed `113` revision-24 run retained no trained checkpoint. Update 5 tied the
retained policy; update 10 improved active and fixed validation but introduced one
rotating regression and was rejected. Training stopped at update 13 after the
within-group correctness signal became homogeneous. Final paired testing was unchanged
at `17/48`, estimated cost was `$0.465480`, and verified teardown returned RunPod to
zero pods and zero hourly spend.

The branch record showed a learnability gap in the fixed probe allocation: late
level-0 groups were usually solved by all four siblings, level-2 probes were usually
solved by none, and level-1 probes retained mixed correctness. Revision 25 routes a
single adaptive probe using those K=4 outcomes. A four-task update samples two groups
at the current level and two at the probe. The probe begins at the nearest harder
level, stays where mixed correctness supplies contrast, moves up after all siblings
solve, and moves down after all siblings fail. This changes task difficulty selection,
not branch width, reward, credit assignment, checkpoint retention, or the held-out
hypothesis gate.

The seed `113` revision-25 run validated the probe controller: mixed level-1 outcomes
held the frontier, `4/4` solved outcomes raised it, and `0/4` solved level-2 outcomes
lowered it. Update 15 retained a zero-regression checkpoint, but the sampler continued
using only the validation failures discovered before training. It stopped at update 23
after contrast disappeared and reproduced the revision-23 result: `17/48` to `21/48`,
four improvements, zero regressions, and `p=0.125`. The run cost an estimated
`$0.894085`; proof ingestion, artifact verification, provider teardown, and zero
ongoing spend succeeded.

Revision 26 makes validation-derived task focus an adaptive curriculum dimension. Each
fixed and disjoint rotating validation window evaluates the disabled-adapter base. Its
failed families are accumulated without consulting final-test outcomes. The structural
mirror disclosures are treated as a graph, so a validation family can reach a
train-split analogue across multiple declared relationships. A newly reachable training
family changes the sampler and resets contrast exhaustion; repeated failures and
families with no declared train analogue do not. This feedback state is durable and
observable. Static `K=4`, the adaptive difficulty probe, causal edit credit, reference
anchor, retention guard, and final paired gate do not change.

## Non-goals

- Dynamic branch width.
- AWS execution.
- Arbitrary repository or shell execution.
- A production RunPod provider inside the local fixture registry.
- Recasting the completed CAD fixture as the repository environment.
