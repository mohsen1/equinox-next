# Larger-model RunPod runbook

Preparing this workflow does not launch a GPU. A provider allocation occurs only when an
operator explicitly runs a command without `--preflight-only`.

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
| Provider lifetime | 45 minutes | 4 hours |
| Workload attempts | 1 | 1 |

## Before either stage

Confirm that:

- RunPod authentication works and the account has no active pod or hourly spend;
- the dashboard and API are running, so the execution is observable before allocation;
- Runs can receive live progress and terminal evidence;
- the versioned larger-model profile contains the intended model revision and safety
  limits;
- the complete pinned model snapshot is present in the configured cache; and
- no previous teardown is unresolved.

Both launchers fail closed when a required check cannot be completed. A failed provider
query is not treated as proof that no pod exists.

## 1. Run the free preflight

```bash
./scripts/screen-larger-model --preflight-only
```

Preflight performs local and provider-read-only checks. It validates the profile and
exact model revision, confirms that the requested GPU class meets the minimum memory,
checks the offline model snapshot, checks current RunPod allocations and spend, and
verifies the configured hourly, total-cost, model-load, and lifetime limits.

Expected result: preflight succeeds and RunPod still reports no new allocation. Stop if
the model metadata cannot be verified, the GPU inventory does not satisfy the memory
floor, the pinned snapshot is incomplete, existing spend is nonzero, or any cap is
missing or inconsistent.

## 2. Run the eligibility screen

```bash
./scripts/screen-larger-model
```

The launcher rechecks every preallocation condition immediately before requesting a
worker. After allocation, it verifies the actual GPU identity and memory before loading
the model. The screen loads only the verified cached snapshot; it does not download model
weights while the GPU is billing.

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
result digest, and provider receipt, and it must confirm teardown. A result from another
model or profile cannot authorize this pilot.

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

## Hard stop conditions

The launcher must stop and tear down the worker when any of these conditions is met:

- observed GPU memory is below the profile minimum;
- the actual hourly rate exceeds the hourly cap;
- projected or accrued spend exceeds the total-cost cap;
- model loading exceeds its timeout;
- the provider lifetime limit expires;
- CUDA reports an out-of-memory failure; or
- result or receipt verification fails.

An out-of-memory failure is not retried on the same hardware and configuration. Workload
attempts are bounded; an unchanged deterministic failure must not consume another paid
attempt.

The total-cost cap and provider lifetime limit are independent. The stricter limit wins.
A GPU that is cheaper per hour may still be rejected when its maximum lifetime could
exceed the total-cost cap.

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
