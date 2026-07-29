# Larger-model RunPod runbook

Installing this change creates no provider resource. The CPU prewarm and each paid H100
stage below require an explicit operator command.

The guarded larger-model flow targets the manifest-pinned
`Qwen/Qwen2.5-Coder-7B-Instruct` revision under profile
`qwen2.5-coder-7b-runpod-h100@8`. It has two paid stages:

1. a bounded eligibility screen that does not update the policy; and
2. a branch-aware training pilot authorized by the screen receipt.

A completed screen is not evidence of learning. It only decides whether the exact model,
revision, profile, and hardware combination may enter the pilot.

Profile `@8` binds simulator `repository-repair-simulator@9`, action protocol
`repository-repair-json-tools@8`, and terminal submission contract
`accepted-passing-test-or-finish@1`. An accepted passing test now completes a repair
without a redundant finish action; an accepted finish remains terminal. The pilot is
`runpod-repository-repair-large-model-pilot@6`, its objective is
`verified-repair-chain-transactional-retention-policy-gradient@18`, and its reward
contract remains `correctness-gated-efficiency@1`.

The pilot uses transactional retention revision
`adapter-optimizer-policy-lineage@1`. Every policy-bearing optimizer step is followed by
the existing fixed and rotating paired guard. A regressing candidate immediately
restores the retained adapter and optimizer, resets effective policy lineage, and clears
pending on-policy examples. A zero-regression tie may remain provisional but cannot
advance mastery or authorize final testing. A strict zero-regression improvement commits
adapter, optimizer, and policy lineage together. The pilot separately reports attempted,
effective, retained, and rolled-back policy updates. It uses a `1e-5` learning rate and
reference-KL coefficient `1.0`.

The versioned
[larger-model eligibility profile](../research/studies/larger-model-eligibility.json) is
the source of truth. It pins optimization seed `137` and the SHA-256 digest of every
larger-model scientific wrapper and interface source used by the screen and pilot:

| Guard                  |                   Screen |                    Pilot |
| ---------------------- | -----------------------: | -----------------------: |
| GPU                    | H100 SXM, at least 80 GB | H100 SXM, at least 80 GB |
| Maximum hourly cost    |                    $4.00 |                    $3.25 |
| Maximum total cost     |                    $3.00 |                   $13.00 |
| Model-load timeout     |               20 minutes |               20 minutes |
| No-progress watchdog   |               10 minutes |               15 minutes |
| Paid launcher lifetime |               43 minutes |              238 minutes |
| Cleanup cost reserve   |              120 seconds |              120 seconds |
| Workload attempts      |                        1 |                        1 |
| Storage-only idle cap  |               $0.01/hour |               $0.01/hour |

The paid cost envelope is therefore at most 45 minutes for the screen and 240 minutes for
the pilot. The two paid GPU stages have a combined `$16.00` ceiling. Including the
`$5.41` already spent, this leaves `$3.59` of the authorized `$25.00` for CPU prewarm,
network-volume storage, and contingency. The 120-second reserve is not training time.

This campaign permits exactly one ten-minute CPU staging allocation, one profile-`@8`
screen, and at most one screen-authorized pilot. Record a fresh `runpodctl user` balance
before each allocation; an ambiguous create or a changed spend envelope is a stop
condition, not permission to retry. At the current settled spend, the worst-case
pre-storage total is `$21.451606`. Delete the network volume within 30 hours of the
pre-stage balance snapshot; even at the manifest's `$0.01/hour` storage ceiling, that
keeps the campaign below `$21.76` and leaves more than `$3.24` of authorization buffer.

## Prepare the network volume and workload on CPU

Use the repository operator to refresh the exact model-readiness and workload-stage
receipts on an already populated network volume. It is the only supported profile-refresh
procedure. It loads the current profile, model, runtime image, source contract, volume
size, and bundle identity from
`research/studies/larger-model-eligibility.json`; do not copy profile IDs or bundle
digests into a manual provider command.

This command does not create a network volume, install dependencies, or download a model.
Before using it, the configured volume must already contain the manifest-pinned model
snapshot and dependency directory. If the volume is new or either prerequisite is
missing, stop and use a separately reviewed initial-volume provisioning procedure. A
failed refresh is not authority to download during GPU allocation.

Set exactly one volume and data center, then run the read-only preflight:

```bash
export EQUINOX_RUNPOD_NETWORK_VOLUME_ID=VOLUME_ID
export EQUINOX_RUNPOD_DATA_CENTER_IDS=DATA_CENTER_ID

./scripts/stage-larger-model-runpod-volume --preflight-only
```

Preflight creates no provider resource and writes no receipt. It must report
`"outcome":"preflight_passed"` and `"allocation_attempted":false`. Review the emitted
profile, manifest, source-contract, bundle, image, volume, `$0.25/hour` cost limit,
600-second lifetime, 300-second ambiguity window, installed `runpodctl` version, and
absolute-datetime termination contract. It builds both the deterministic allowlisted
operator source archive and workload bundle from the same immutable Git `HEAD` snapshot;
it never pairs live-worktree bundle bytes with a different archive. Any participating
file that differs from `HEAD` is rejected. It fails if the shared operator lease exists
or unless the account has no pod, exactly the configured manifest-sized volume, and no
more than the manifest's storage-only hourly spend.

After reviewing that output, run the operator once:

```bash
./scripts/stage-larger-model-runpod-volume
```

The operator takes the same persistent
`~/.local/state/equinox/runpod/operator.lock/lease.json` lease as the GPU launchers,
rechecks the complete idle provider state immediately before create, and issues at most
one create request. It uses a unique name, secure CPU compute, the manifest's exact
`tag@sha256` image, a 5 GB container disk, the exact volume and data center mounted at
`/workspace`, SSH, and a provider auto-termination deadline ten minutes after the frozen
start time. It has no GPU selection path and never falls back to H100.

Creation has three distinct outcomes:

- A successful response must contain one safe pod ID. The operator then attests the
  unique name, documented CPU flavor and vCPU fields, absence of GPU allocation evidence,
  Secure Cloud, pinned image, 5 GB container disk, exact volume and mount, and an hourly
  rate no greater than `$0.25` before SSH. Missing initialization fields are polled for at
  most 30 seconds; any contradictory field fails immediately. The installed CLI contract
  and exact create arguments bind the absolute termination deadline; if the provider
  response includes `computeType` or `terminateAfter`, either must match exactly.
- The provider's explicit CPU-capacity rejection is classified as
  `capacity_unavailable_confirmed_zero_allocation` with exit code `75` only after the
  unique name remains absent through the full five-minute visibility window, every
  reconciliation query succeeds, and three final zero-pod polls pass.
- Every other missing, malformed, timed-out, or contradictory create result is
  `ambiguous_create_*` with exit code `70`. A returned or later-visible pod ID is deleted
  immediately. The operator never issues a second create.

Exit `75` is a confirmed no-allocation capacity stop, not authority to retry or use a GPU.
Exit `70` retains the shared lease even when the final polls are clean, so a sequential
command cannot hide the ambiguity. Follow the lease-recovery procedure below before any
later provider action.

On an attested CPU pod, the operator verifies the mounted filesystem and rejects symlinked
staging parents. It copies the deterministic, allowlisted, `HEAD`-bound source archive and
canonical bundle, verifies both archive identities remotely, stages the bundle at its
content-addressed read-only path, and creates fresh model and stage receipts. It also runs
a real AdamW transactional-retention round trip with the image's Torch build and verifies
the exact `torch.__version__`, CUDA build, profile, checkout commit, archive and source
digests, checkpoint digest and size, restored and advanced weights, policy counters,
retained observation, and optimizer state.

Every paid remote step has a bounded timeout and may start only when the provider lifetime
has enough time left for that full timeout plus the 180-second teardown reserve. When the
reserve would be consumed, the step is not started and teardown begins. Provider
auto-termination remains the independent ten-minute backstop.

The receipts and Torch evidence are copied to pending local files and verified before
installation. Receipt verification uses a fresh provider-volume response. The operator
then deletes the exact pod and requires three consecutive successful polls with zero pods,
the exact sole volume, and storage-only spend. Failed provider queries do not count as
absence. Only after teardown is confirmed are the verified files installed with
an fsynced set-publication journal, per-file atomic replacements, and rollback:

```text
var/research-proofs/larger-model-volume-VOLUME_ID.json
var/research-proofs/larger-model-bundle-stage-VOLUME_ID.json
var/research-proofs/equinox-volume-stage-*.torch-retention.json
var/research-proofs/equinox-volume-stage-*.torch-retention.sha256
var/research-proofs/equinox-volume-stage-*.volume-stage-attempt.json
var/research-proofs/equinox-volume-stage-*.volume-stage-recovery.json
```

Existing receipts survive remote, verification, teardown, and recoverable publication
failures. An interrupted, incompletely rolled-back publication retains both its journal
and the shared lease, which blocks both CPU and GPU operators until explicit recovery. No
provider query runs after publication begins. When a receipt belongs to an older profile,
the operator archives it under a profile-qualified name before installing the new
verified receipt; an existing archive with different bytes is a fail-closed conflict, not
permission to overwrite either receipt. A successful command reports
`"outcome":"staged"`, both receipt digests, the Torch evidence digest, three idle polls,
and `"gpu_fallback_used":false`.

If the lease state is `publishing_artifacts`, `artifacts_published`,
`artifact_publication_rolled_back`, or `artifact_publication_recovery_complete`, do not
remove the lease or transaction directory manually. With the same exact volume and data
center environment, first confirm that the `operator_pid` recorded in the lease has
exited, then run:

```bash
./scripts/stage-larger-model-runpod-volume --recover-publication
```

This recovery mode issues no provider query and creates no resource. For an interrupted
publication it compares every destination with the journal's old and new digests, restores
the fsynced `old-*` set (or removes a destination that did not previously exist), verifies
the restored set, and reports `"outcome":"artifact_publication_rolled_back"`. If the lease
already records `artifacts_published`, it instead verifies every new digest, both receipt
digests and identities, and the Torch evidence/hash pair before reporting
`"outcome":"artifact_publication_completed"`. It fsyncs the result, removes the transaction
journal, and only then releases the shared lease. Any missing backup, unknown bytes,
changed identity, unsafe path, invalid receipt, live original operator process, or
concurrent recovery attempt leaves the lease in place. Before lease release it atomically
writes an idempotent `*.volume-stage-recovery.json` that retains the original pod, start,
cost, and idle-poll evidence while recording `"recovery_allocation_attempted":false`. If
the interrupted process already wrote a failed `*.volume-stage-attempt.json`, recovery
preserves that record; the paid attempt never disappears from the audit trail.

The model-readiness receipt expires after seven days. The stage receipt expires 24 hours
after `staged_at`; bundle bytes remaining on the volume do not extend that deadline.

## Before either paid stage

Confirm that:

- RunPod authentication works, the account has no pod, exactly the verified configured
  network volume is present, and hourly spend is at most `$0.01`;
- the dashboard and API are running, so the execution is observable before allocation;
- the model-readiness and workload-stage receipts exist at the paths above, still match
  the current provider volume and canonical bundle, and are fresh;
- no previous teardown or operator lease is unresolved.

Both launchers fail closed when a required check cannot be completed. A failed provider
query is not treated as proof that no pod exists.

## 1. Run the free preflight

```bash
./scripts/screen-larger-model --preflight-only
```

Preflight performs local and provider-read-only checks. It validates the profile and
exact model revision, confirms that the requested GPU class meets the minimum memory,
verifies the current network volume against its digest-bound readiness receipt, checks
that it is the account's only network volume, rebuilds the canonical workload bundle,
verifies its 24-hour stage receipt against the provider volume, checks data-center
availability, and verifies the hourly, total-cost, model-load, lifetime, and cleanup
limits. Both preflight and paid launch refuse allocation when any pod exists or hourly
spend exceeds the manifest-pinned `$0.01/hour` storage-only baseline.

Expected result: preflight succeeds and RunPod still reports no new allocation. Stop if
the model metadata cannot be verified, the GPU inventory does not satisfy the memory
floor, the volume or receipt does not match, or any cap is missing or inconsistent.

## 2. Run the eligibility screen

```bash
./scripts/screen-larger-model
```

The launcher rechecks every preallocation condition immediately before requesting a
worker. It creates the worker from the manifest's exact `tag@sha256` image reference and
rejects the allocation unless the provider reports that same immutable reference.

The paid worker reads the pre-staged XZ bundle from its content-addressed volume path. A
bundle may be at most 2 MiB. The pod-creation environment contains no workload bytes; it
contains only the result token and operational identity: the handoff revision, bundle
path, digest and size, stage-receipt digest, and bootstrap-source digest. That
identity-only environment is capped at 4 KiB.

Before execution, the container verifies the bootstrap source digest. The bootstrap then
opens the staged path without following symlinks, verifies its regular-file identity,
size and SHA-256, enforces the exact file allowlist, and installs the workload once. The
pre-model readiness check re-verifies the pinned source contract. The first structured
progress must attest the same handoff revision, bundle path, digest and size,
stage-receipt digest, bootstrap digest, and network-volume ID. Any mismatch stops the
worker before the launcher accepts readiness.

The launcher uses authenticated read-only readiness and progress probes after allocation.
The paid handoff performs no proxy POST mutation. Preflight, the execution resource
profile, provider receipt, and final proof retain the exact operational handoff identity.

After allocation, the workload verifies the actual GPU identity, memory, free cache
space, the exact `torch==2.8.0+cu128` build, and every cached file digest before model
initialization. It also requires the exact dependency versions from the volume. The
paid H100 sets `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `PIP_NO_INDEX=1`: it never installs packages or downloads
model weights. Missing or changed artifacts fail immediately.

The six-minute readiness limit is one wall-clock deadline beginning immediately before
the create request. Pod creation, provider inspection, observer updates, authenticated
readiness and progress probes, retries, and sleeps all consume that same budget.

The screen evaluates eight deterministic level-0 tasks, then runs eight static-`K=4`
branch probes with no policy mutation. Branching starts after one accepted repository
root listing. Localization and repair happen independently in each sibling. Higher
levels remain pilot curriculum targets; requiring them before training would defeat the
adaptive starting frontier.

The generic trainer requires a positive test-example placeholder, but the screen reserves
zero seconds for final-test evaluation and blocks every test-split request before task
generation. It exits as soon as the eighth branch group is recorded, before the generic
policy-update path. Any attempted test-split request is an isolation failure, not a reason
to continue into final evaluation.

The screen and pilot both reject a prompt that exceeds 2,048 input tokens. They never
silently discard the oldest prompt content. Before the screen, the reversible H100
capacity smoke exercises 2,240 tokens: the larger screen/pilot input envelope plus the
192-token generation cap.

Both stages run the same `accepted-passing-test-or-finish@1` terminal contract. A
schema-valid, accepted test action that passes hidden verification ends the trajectory
as solved immediately. A failing test remains nonterminal unless it consumes the repair
horizon, and the existing finish action retains its prior solved or
finished-with-failures behavior.

The pilot-runtime gate extrapolates the observed level-0 baseline across every pilot
level. It multiplies by `(8 + 10 + 14 + 18) / 8 = 6.25` for the four repair horizons,
then accounts for paired base/final evaluation and the pinned `1.5` safety factor. The
1,440-second threshold is unchanged; the screen does not treat level-0 timing as if all
four levels had the same horizon.

Baseline exact solves are not the nonzero competence gate. The baseline keeps only its
75% maximum headroom check. Pilot authorization instead requires mixed outcomes across
the `K=4` branches: at least two solved siblings, at least two failed siblings, a solved
rate from 5% through 80%, and at least two informative branch groups. If the model
cannot produce that mixed branch evidence, stop without launching the pilot.

The screen does not access the sealed final-test pack or persist a trained adapter. Its
result records:

- the exact model, revision, eligibility profile, and workload revisions;
- the simulator, action protocol, and terminal submission contract;
- requested and observed GPU identity and memory;
- model-load completion and peak memory;
- schema-valid action rate and semantic acceptance rate as separate metrics;
- root-checkpoint attainment, all-fault-source localization telemetry, branch signal
  density, and solution headroom;
- confirmation that policy parameters were not changed;
- the eligibility decision and each failed threshold; and
- runtime, cost, provider receipt, and teardown evidence.

Operational success and eligibility are separate. A screen can finish correctly and
return `eligible: false`. Do not rerun it merely to search for a passing sample.

## 3. Verify pilot authorization without allocating

```bash
./scripts/run-larger-model-pilot --preflight-only
```

The pilot preflight requires a screen completed within the previous seven days with
`eligible: true`. The authorization must bind the same profile, model ID, model revision,
snapshot digest, network volume, result digest, and provider receipt, and it must confirm
teardown. It also binds the handoff revision; bundle digest, size, XZ compression, and
content-addressed path; stage-receipt digest; and bootstrap-source digest. A result from
another model, profile, volume, bundle, stage receipt, or bootstrap cannot authorize this
pilot.

Do not edit or copy receipt fields to bypass this check. If the receipt does not match,
fix the underlying screen or profile and repeat the screen as an explicitly new
execution. If the 24-hour stage receipt expires or the bootstrap changes after the
screen, restage and run a new screen; the existing authorization must not cross that
operational change.

## 4. Run the bounded pilot

```bash
./scripts/run-larger-model-pilot
```

The pilot revalidates its authorization and all provider safety checks before allocation.
It then runs the static `K=4`, dynamically complex repository-repair workload. The pilot
is a bounded experiment, not authorization for a larger follow-on study.

Only verified successful siblings contribute positive policy weight. The credited
actions are the fault-fixing edit and its immediately preceding accepted read of the
same path. Unrelated reads, searches, tests, finishes, and failed siblings receive no
positive weight; every accepted action remains anchored to the disabled-adapter base
policy.

The pilot authorization is single-use. `--preflight-only` does not consume it. A paid
pilot writes a consumption record at
`~/.local/state/equinox/runpod/larger-model-authorizations/SCREEN_RESULT_DIGEST_HEX.json`
immediately before requesting a pod. A failed or ambiguous create still consumes the
authorization; another pilot requires a new eligibility screen.

Result retrieval is also inside the paid deadline. The launcher reserves 60 seconds for
the screen result, 180 seconds for the pilot result and adapter, and then the full
120-second teardown window.

## Hard stop conditions

The launcher must stop and tear down the worker when any of these conditions is met:

- observed GPU memory is below the profile minimum;
- the actual hourly rate exceeds the hourly cap;
- projected or accrued spend exceeds the total-cost cap;
- the staged bundle, bootstrap, first-progress attestation, or screen authorization does
  not match the exact handoff identity;
- model loading exceeds its timeout;
- structured progress stops changing for the configured watchdog period;
- the provider lifetime limit expires;
- CUDA reports an out-of-memory failure; or
- result or receipt verification fails.

An out-of-memory failure is not retried on the same hardware and configuration. Workload
attempts are bounded; an unchanged deterministic failure must not consume another paid
attempt.

The total-cost cap and provider lifetime limit are independent. The stricter limit wins.
A GPU that is cheaper per hour may still be rejected when its maximum lifetime could
exceed the total-cost cap.

## Operator lease and recovery

Before allocation, the launcher creates:

```text
~/.local/state/equinox/runpod/operator.lock/lease.json
```

The lease records the proof ID, unique pod name, mode, process ID, start time, and
provider deadline. It prevents a second operator from allocating concurrently. Normal
completion removes it only after teardown is confirmed; unresolved cleanup leaves it in
place. This state is shared by local worktrees. It is not a distributed lock: operate the
RunPod account from one designated host only.

After an interruption, do not delete the lease to force another launch. Inspect its
state first:

```bash
jq . ~/.local/state/equinox/runpod/operator.lock/lease.json
```

For `publishing_artifacts`, `artifacts_published`,
`artifact_publication_rolled_back`, or
`artifact_publication_recovery_complete`, the CPU volume-stage operator's
`--recover-publication` command is the only permitted lease-release path. Follow the
procedure in “Prepare the network volume and workload on CPU” only after the recorded
operator process has exited; do not run the provider reconciliation or the manual `rm`
commands below for those states.

For every non-publication lease state, reconcile the exact pod name and account spend:

```bash
runpodctl pod list --all
runpodctl user
```

If the named pod exists, delete that exact pod ID. Confirm three successful provider
queries report it absent and `currentSpendPerHr` has returned to the verified
storage-only baseline. Only then remove
`~/.local/state/equinox/runpod/operator.lock/lease.json` and its now-empty lock directory.
The next preflight reconciles a stale Runs record. This is cleanup recovery, not workload
resume.

When pod creation returns without an ID, the launcher treats provider state as ambiguous.
It watches the unique pod name for five minutes and allows up to 330 seconds for
reconciliation. A failed or malformed provider query is `unknown`, not proof of absence.
If reconciliation cannot prove both pod absence and the verified storage-only hourly
spend, the lease remains. The following manual release applies only after that
non-publication reconciliation:

```bash
rm -- ~/.local/state/equinox/runpod/operator.lock/lease.json
rmdir -- ~/.local/state/equinox/runpod/operator.lock
```

## Confirm teardown

Every terminal path—eligible, ineligible, failed, timed out, or interrupted—must request
pod deletion and then confirm absence through a successful provider query. Unknown
provider state is not confirmed teardown.

After either paid command:

1. confirm the execution is terminal in Runs;
2. confirm teardown is recorded;
3. confirm RunPod reports no active pod and only the verified storage-only hourly spend;
   and
4. compare recorded runtime and total cost with the configured caps.

If teardown is not confirmed, treat the run as an active-spend incident. Remove the pod
through RunPod, verify the storage-only baseline, and reconcile the execution before
launching anything else.

After the pilot is complete—or explicitly abandoned—copy and verify every result,
adapter, and receipt you intend to keep, then delete the network volume:

```bash
runpodctl network-volume delete "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID"
runpodctl network-volume list
```

The network volume remains billable independently of GPU teardown. Do not leave it
allocated after the larger-model sequence is finished.

## Inspect the run

Use `/runs` as the operational queue. Check the model, allocated GPU, live phase, elapsed
time, estimated total cost, and last update.

Open the execution to inspect:

- **Overview:** eligibility decision or training outcome, cap state, and current phase;
- **Trajectory:** the shared prefix, checkpoint, four sibling continuations, actions,
  verifier outcomes, and policy signal;
- **Evidence:** model and workload revisions, hardware attestation, receipt digest, cost,
  staged handoff identity, and teardown; and
- **Operations:** provider handle, timestamps, failure reason, and cleanup state.

For the screen, the deciding evidence is the threshold breakdown and no-policy-mutation
confirmation. For the pilot, inspect branch diversity, informative groups, optimizer
updates, dynamic-complexity changes, validation regressions, and the retained checkpoint.
Do not infer progress from GPU utilization alone.
