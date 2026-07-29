# Larger-model RunPod runbook

Installing this change creates no provider resource. The CPU prewarm and each paid L40
stage below require an explicit operator command.

The guarded larger-model flow targets the manifest-pinned
`Qwen/Qwen2.5-Coder-7B-Instruct` revision. It has two paid stages:

1. a bounded eligibility screen that does not update the policy; and
2. a branch-aware training pilot authorized by the screen receipt.

A completed screen is not evidence of learning. It only decides whether the exact model,
revision, profile, and hardware combination may enter the pilot.

The versioned
[larger-model eligibility profile](../research/studies/larger-model-eligibility.json) is
the source of truth:

| Guard | Screen | Pilot |
| --- | ---: | ---: |
| GPU | L40, at least 48 GB | L40, at least 48 GB |
| Maximum hourly cost | $1.00 | $1.00 |
| Maximum total cost | $0.75 | $4.00 |
| Model-load timeout | 20 minutes | 20 minutes |
| Paid launcher lifetime | 43 minutes | 238 minutes |
| Cleanup cost reserve | 120 seconds | 120 seconds |
| Workload attempts | 1 | 1 |

The paid cost envelope is therefore at most 45 minutes for the screen and 240 minutes for
the pilot. The 120-second reserve is not training time.

## Prepare the network volume on CPU

The launcher never creates or populates a network volume. With no existing volume and
matching readiness receipt, both `--preflight-only` and paid launch are blocked.

Create a 50 GB volume in a data center that offers an L40:

```bash
runpodctl network-volume create \
  --name equinox-qwen25-coder-7b \
  --size 50 \
  --data-center-id DATA_CENTER_ID

export EQUINOX_RUNPOD_NETWORK_VOLUME_ID=VOLUME_ID_FROM_RESPONSE
export EQUINOX_RUNPOD_DATA_CENTER_IDS=DATA_CENTER_ID
```

Attach it to a short-lived CPU pod. Set `--terminate-after` to a UTC timestamp soon enough
to bound CPU spend and long enough to complete the transfer.

```bash
runpodctl pod create \
  --name equinox-7b-prewarm \
  --compute-type cpu \
  --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
  --network-volume-id "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID" \
  --data-center-ids "$EQUINOX_RUNPOD_DATA_CENTER_IDS" \
  --terminate-after PREWARM_DEADLINE_UTC
```

Use the connection shown by `runpodctl ssh info PREWARM_POD_ID`. On that CPU pod:

```bash
python3 -m pip install \
  --target /workspace/equinox-state/python \
  accelerate==1.14.0 \
  peft==0.19.1 \
  transformers==5.14.1

PYTHONPATH=/workspace/equinox-state/python python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
    cache_dir="/workspace/equinox-state/huggingface",
)
PY
```

Copy `research/runpod/larger_model_gate.py` and
`research/studies/larger-model-eligibility.json` from the repository to
`/tmp/equinox-prewarm/` over the same SSH connection.

Set the same volume and data-center values in the CPU shell. Then hash the snapshot and
create its readiness receipt:

```bash
export EQUINOX_RUNPOD_NETWORK_VOLUME_ID=VOLUME_ID
export EQUINOX_RUNPOD_DATA_CENTER_IDS=DATA_CENTER_ID

PYTHONPATH=/workspace/equinox-state/python \
python3 /tmp/equinox-prewarm/larger_model_gate.py \
  --manifest /tmp/equinox-prewarm/larger-model-eligibility.json \
  create-volume-receipt \
  --volume-id "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID" \
  --data-center-id "$EQUINOX_RUNPOD_DATA_CENTER_IDS" \
  --volume-size-gb 50 \
  --snapshot /workspace/equinox-state/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  >"/workspace/equinox-state/larger-model-volume-$EQUINOX_RUNPOD_NETWORK_VOLUME_ID.json"
```

Copy that JSON back over the same SSH connection. Its required local path is:

```text
var/research-proofs/larger-model-volume-VOLUME_ID.json
```

Then verify it locally against the current provider volume:

```bash
mkdir -p var/research-proofs
export EQUINOX_LARGER_MODEL_VOLUME_RECEIPT="$PWD/var/research-proofs/larger-model-volume-$EQUINOX_RUNPOD_NETWORK_VOLUME_ID.json"

runpodctl network-volume get "$EQUINOX_RUNPOD_NETWORK_VOLUME_ID" |
  python3 research/runpod/larger_model_gate.py \
    verify-volume-receipt \
    "$EQUINOX_LARGER_MODEL_VOLUME_RECEIPT" \
    -
```

The receipt binds the profile, model revision, every required file digest, dependency
versions, volume ID, data center, and size. It expires after seven days. Copy it back
before deleting the CPU pod.

```bash
runpodctl pod delete PREWARM_POD_ID
runpodctl pod list --all
runpodctl user
```

Do not continue until the prewarm pod is absent and ongoing hourly spend is zero. The
network volume remains for the screen and pilot.

## Before either paid stage

Confirm that:

- RunPod authentication works and the account has no active pod or hourly spend;
- the dashboard and API are running, so the execution is observable before allocation;
- the volume receipt exists at the path above and is still fresh; and
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
data-center availability, and verifies the hourly, total-cost, model-load, lifetime, and
cleanup limits. The paid command separately refuses allocation when any pod or hourly
spend is active.

Expected result: preflight succeeds and RunPod still reports no new allocation. Stop if
the model metadata cannot be verified, the GPU inventory does not satisfy the memory
floor, the volume or receipt does not match, or any cap is missing or inconsistent.

## 2. Run the eligibility screen

```bash
./scripts/screen-larger-model
```

The launcher rechecks every preallocation condition immediately before requesting a
worker. After allocation, it verifies the actual GPU identity, memory, free cache space,
and every cached file digest before model initialization. It also requires the exact
dependency versions from the volume. The paid L40 sets `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `PIP_NO_INDEX=1`: it never installs packages or downloads
model weights. Missing or changed artifacts fail immediately.

The screen runs fixed baseline and `K=4` branch probes with no policy mutation. It does
not access the sealed final-test pack or persist a trained adapter. Its result records:

- the exact model, revision, eligibility profile, and workload revisions;
- requested and observed GPU identity and memory;
- model-load completion and peak memory;
- action-protocol validity, checkpoint attainment, branch signal density, and solution
  headroom;
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
teardown. A result from another model, profile, or volume cannot authorize this pilot.

Do not edit or copy receipt fields to bypass this check. If the receipt does not match,
fix the underlying screen or profile and repeat the screen as an explicitly new
execution.

## 4. Run the bounded pilot

```bash
./scripts/run-larger-model-pilot
```

The pilot revalidates its authorization and all provider safety checks before allocation.
It then runs the static `K=4`, dynamically complex repository-repair workload. The pilot
is a bounded experiment, not authorization for a larger follow-on study.

The pilot authorization is single-use. `--preflight-only` does not consume it. A paid
pilot writes a consumption record at
`var/research-proofs/larger-model-authorizations/SCREEN_RESULT_DIGEST_HEX.json`
immediately before requesting a pod. A failed or ambiguous create still consumes the
authorization; another pilot requires a new eligibility screen.

## Hard stop conditions

The launcher must stop and tear down the worker when any of these conditions is met:

- observed GPU memory is below the profile minimum;
- the actual hourly rate exceeds the hourly cap;
- projected or accrued spend exceeds the total-cost cap;
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
var/research-proofs/.runpod-operator.lock/lease.json
```

The lease records the proof ID, unique pod name, mode, process ID, start time, and
provider deadline. It prevents a second operator from allocating concurrently. Normal
completion removes it only after teardown is confirmed; unresolved cleanup leaves it in
place.

After an interruption, do not delete the lease to force another launch. Inspect it and
reconcile the exact pod name:

```bash
jq . var/research-proofs/.runpod-operator.lock/lease.json
runpodctl pod list --all
runpodctl user
```

If the named pod exists, delete that exact pod ID. Confirm three successful provider
queries report it absent and `currentSpendPerHr` is zero. Only then remove
`var/research-proofs/.runpod-operator.lock/lease.json` and its now-empty lock directory.
The next preflight reconciles a stale Runs record. This is cleanup recovery, not workload
resume.

```bash
rm -- var/research-proofs/.runpod-operator.lock/lease.json
rmdir -- var/research-proofs/.runpod-operator.lock
```

## Confirm teardown

Every terminal path—eligible, ineligible, failed, timed out, or interrupted—must request
pod deletion and then confirm absence through a successful provider query. Unknown
provider state is not confirmed teardown.

After either paid command:

1. confirm the execution is terminal in Runs;
2. confirm teardown is recorded;
3. confirm RunPod reports no active pod and zero ongoing hourly spend; and
4. compare recorded runtime and total cost with the configured caps.

If teardown is not confirmed, treat the run as an active-spend incident. Remove the pod
through RunPod, verify zero ongoing spend, and reconcile the execution before launching
anything else.

## Inspect the run

Use `/runs` as the operational queue. Check the model, allocated GPU, live phase, elapsed
time, estimated total cost, and last update.

Open the execution to inspect:

- **Overview:** eligibility decision or training outcome, cap state, and current phase;
- **Trajectory:** the shared prefix, checkpoint, four sibling continuations, actions,
  verifier outcomes, and policy signal;
- **Evidence:** model and workload revisions, hardware attestation, receipt digest, cost,
  and teardown; and
- **Operations:** provider handle, timestamps, failure reason, and cleanup state.

For the screen, the deciding evidence is the threshold breakdown and no-policy-mutation
confirmation. For the pilot, inspect branch diversity, informative groups, optimizer
updates, dynamic-complexity changes, validation regressions, and the retained checkpoint.
Do not infer progress from GPU utilization alone.
