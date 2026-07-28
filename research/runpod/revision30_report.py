"""Build a failure-complete, per-seed report for the revision-30 causal study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - macOS operator compatibility.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from research.runpod.revision30_external_eval import (
    WORKLOAD as EXTERNAL_WORKLOAD,
)
from research.runpod.study_operator import STUDY_ID, load_manifest

REPORT_ID = "repository-repair-confirmatory-study@1/failure-complete-report@1"
TRAINING_WORKLOAD = "repository-repair-restored-continuation-post-training"
SCREEN_WORKLOAD = "repository-repair-protocol-eligibility"
REHEARSAL_EXECUTION_ID = "runpod-proof-ui-rehearsal"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an object")
    return value


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} does not contain an object")
        events.append(value)
    return events


def result_files(receipt_directory: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(receipt_directory.glob("runpod-proof-*.result.json")):
        suffix = ".result.json"
        execution_id = path.name[: -len(suffix)]
        results[execution_id] = read_json(path)
    return results


def estimated_cost(execution: dict[str, Any]) -> float | None:
    profile = execution.get("resource_profile")
    progress = execution.get("progress")
    rate = profile.get("hourly_cost_usd") if isinstance(profile, dict) else None
    elapsed = progress.get("elapsed_seconds") if isinstance(progress, dict) else None
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))  # noqa: UP038 - Python 3.9.
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))  # noqa: UP038 - Python 3.9.
        or not math.isfinite(float(rate))
        or not math.isfinite(float(elapsed))
    ):
        return None
    return round(float(rate) * float(elapsed) / 3_600, 6)


def condition_id_from_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    execution_id: str,
) -> str | None:
    for condition in manifest["conditions"]:
        if condition.get("execution_id") == execution_id:
            return str(condition["condition_id"])
    study = result.get("study")
    if not isinstance(study, dict):
        return None
    condition = study.get("condition")
    seed = study.get("optimization_seed")
    if not isinstance(condition, str) or isinstance(seed, bool) or not isinstance(seed, int):
        return None
    return f"{condition}_seed{seed}"


def condition_id_from_execution(execution: dict[str, Any]) -> str | None:
    progress = execution.get("progress")
    if not isinstance(progress, dict):
        return None
    condition = progress.get("study_condition")
    seed = progress.get("optimization_seed")
    if isinstance(condition, str) and isinstance(seed, int) and not isinstance(seed, bool):
        return f"{condition}_seed{seed}"
    return None


def seconds_between(left: Any, right: Any) -> float | None:
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    try:
        left_time = datetime.fromisoformat(left.replace("Z", "+00:00"))
        right_time = datetime.fromisoformat(right.replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs((left_time - right_time).total_seconds())


def correlated_attempt_condition(
    execution: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> str | None:
    workload = execution.get("workload_id")
    expected_category = {
        TRAINING_WORKLOAD: "study_condition",
        SCREEN_WORKLOAD: "eligibility_screen",
        EXTERNAL_WORKLOAD: "external_evaluation",
    }.get(workload)
    if expected_category is None:
        return None
    candidates = []
    for attempt in attempts:
        difference = seconds_between(execution.get("started_at"), attempt.get("started_at"))
        if (
            attempt.get("category") == expected_category
            and attempt.get("preflight_only") is False
            and difference is not None
            and difference <= 90
        ):
            candidates.append((difference, attempt["condition_id"]))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return str(candidates[0][1])


def result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    paired = result.get("paired_test_change")
    study = result.get("study")
    return {
        "seed": (study.get("optimization_seed") if isinstance(study, dict) else result.get("seed")),
        "branch_width": result.get("branch_width"),
        "policy_mutation_enabled": (
            study.get("policy_mutation_enabled") if isinstance(study, dict) else True
        ),
        "initial_successes": result.get("initial_reward"),
        "final_successes": result.get("final_reward"),
        "gain": result.get("reward_gain"),
        "paired_improved": paired.get("improved") if isinstance(paired, dict) else None,
        "paired_regressed": paired.get("regressed") if isinstance(paired, dict) else None,
        "paired_net_improved": (paired.get("net_improved") if isinstance(paired, dict) else None),
        "paired_p_value": (
            paired.get("mcnemar_exact_p_value") if isinstance(paired, dict) else None
        ),
        "policy_updates": result.get(
            "effective_policy_update_count",
            result.get("policy_update_count"),
        ),
        "optimizer_updates": result.get(
            "effective_optimizer_update_count",
            result.get("optimizer_update_count"),
        ),
        "sampled_completions": sampled_completions(result),
        "reached_complexity_level": result.get("reached_complexity_level"),
        "maximum_sampled_complexity_level": result.get("maximum_sampled_complexity_level"),
        "retention_passed": result.get("retention_passed"),
        "final_evaluation_complete": result.get("final_evaluation_complete"),
        "adapter_persisted": result.get("adapter_persisted"),
        "stop_reason": result.get("stop_reason"),
    }


def sampled_completions(result: dict[str, Any]) -> int | None:
    declared = result.get("total_sampled_completions")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
        return declared
    groups = result.get("total_task_groups")
    branch_width = result.get("branch_width")
    if (
        isinstance(groups, int)
        and not isinstance(groups, bool)
        and groups >= 0
        and isinstance(branch_width, int)
        and not isinstance(branch_width, bool)
        and branch_width >= 1
    ):
        return groups * branch_width
    return None


def evaluation_task_ids(result: dict[str, Any]) -> tuple[str, ...]:
    observed = []
    by_level = result.get("final_by_level")
    if not isinstance(by_level, dict):
        return ()
    for level in sorted(by_level):
        evaluation = by_level[level]
        outcomes = evaluation.get("task_outcomes") if isinstance(evaluation, dict) else None
        if not isinstance(outcomes, list):
            return ()
        for outcome in outcomes:
            task_id = outcome.get("semantic_task_id") if isinstance(outcome, dict) else None
            if not isinstance(task_id, str):
                return ()
            observed.append(task_id)
    return tuple(sorted(observed))


def evaluation_outcomes(result: dict[str, Any]) -> dict[str, bool]:
    observed: dict[str, bool] = {}
    by_level = result.get("final_by_level")
    if not isinstance(by_level, dict):
        return {}
    for evaluation in by_level.values():
        outcomes = evaluation.get("task_outcomes") if isinstance(evaluation, dict) else None
        if not isinstance(outcomes, list):
            return {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                return {}
            task_id = outcome.get("semantic_task_id")
            solved = outcome.get("solved")
            if not isinstance(task_id, str) or not isinstance(solved, bool):
                return {}
            if task_id in observed:
                return {}
            observed[task_id] = solved
    return observed


def paired_condition_change(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    reference_outcomes = evaluation_outcomes(reference)
    candidate_outcomes = evaluation_outcomes(candidate)
    if not reference_outcomes or reference_outcomes.keys() != candidate_outcomes.keys():
        return None
    improved = sum(
        not reference_outcomes[task_id] and candidate_outcomes[task_id]
        for task_id in reference_outcomes
    )
    regressed = sum(
        reference_outcomes[task_id] and not candidate_outcomes[task_id]
        for task_id in reference_outcomes
    )
    discordant = improved + regressed
    if discordant:
        smaller_tail = (
            sum(math.comb(discordant, count) for count in range(min(improved, regressed) + 1))
            / 2**discordant
        )
        exact_p_value = min(1.0, 2 * smaller_tail)
    else:
        exact_p_value = 1.0
    return {
        "examples": len(reference_outcomes),
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(reference_outcomes) - discordant,
        "net_improved": improved - regressed,
        "mcnemar_exact_p_value": exact_p_value,
    }


def frozen_runtime_identity(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "model_id",
            "model_revision",
            "workload_revision",
            "environment_revision",
            "verifier_revision",
            "action_protocol_revision",
            "objective_id",
        )
    }


def execution_rows(
    executions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for execution in sorted(executions, key=lambda item: str(item.get("started_at", ""))):
        execution_id = execution.get("execution_id")
        if execution_id == REHEARSAL_EXECUTION_ID or not isinstance(execution_id, str):
            continue
        progress = execution["progress"] if isinstance(execution.get("progress"), dict) else {}
        result = results.get(execution_id)
        condition_id = (
            condition_id_from_result(result, manifest, execution_id) if result is not None else None
        )
        condition_id = condition_id or condition_id_from_execution(execution)
        condition_id = condition_id or correlated_attempt_condition(execution, attempts)
        workload = execution.get("workload_id")
        if workload == TRAINING_WORKLOAD and condition_id is None:
            continue
        if workload not in {TRAINING_WORKLOAD, SCREEN_WORKLOAD, EXTERNAL_WORKLOAD}:
            continue
        validity = progress.get("action_protocol_validity_rate")
        outcome = execution.get("status")
        validity_is_numeric = isinstance(validity, (int, float))  # noqa: UP038
        if execution.get("workload_id") == SCREEN_WORKLOAD and validity_is_numeric:
            outcome = "ELIGIBLE" if float(validity) >= 0.99 else "INELIGIBLE"
        row = {
            "execution_id": execution_id,
            "condition_id": condition_id,
            "name": execution.get("name"),
            "workload_id": execution.get("workload_id"),
            "outcome": outcome,
            "provider_handle": execution.get("provider_handle"),
            "gpu_id": (
                execution.get("resource_profile", {}).get("gpu_id")
                if isinstance(execution.get("resource_profile"), dict)
                else None
            ),
            "optimization_seed": progress.get("optimization_seed"),
            "action_protocol_validity_rate": validity,
            "error": progress.get("error"),
            "estimated_cost_usd": estimated_cost(execution),
            "started_at": execution.get("started_at"),
            "completed_at": execution.get("completed_at"),
            "teardown_confirmed": execution.get("teardown_confirmed"),
            "has_result_artifact": result is not None,
        }
        if result is not None and result.get("workload") != EXTERNAL_WORKLOAD:
            row["metrics"] = result_metrics(result)
        rows.append(row)
    return rows


def operator_attempt_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for event in events:
        event_name = event.get("event")
        started_at = event.get("started_at")
        condition_id = event.get("condition_id", event.get("evaluation_id"))
        preflight_only = event.get("preflight_only") is True
        if (
            not isinstance(event_name, str)
            or not isinstance(started_at, str)
            or not isinstance(condition_id, str)
        ):
            continue
        key = (condition_id, started_at, preflight_only)
        if event_name.startswith("eligibility_screen_"):
            category = "eligibility_screen"
        elif event_name.startswith("external_evaluation_"):
            category = "external_evaluation"
        else:
            category = "study_condition"
        row = attempts.setdefault(
            key,
            {
                "condition_id": condition_id,
                "category": category,
                "preflight_only": preflight_only,
                "started_at": started_at,
                "finished_at": None,
                "outcome": "running",
                "exit_code": None,
                "result_artifacts": [],
            },
        )
        if event_name.endswith("_finished"):
            row.update(
                {
                    "finished_at": event.get("finished_at"),
                    "outcome": event.get("outcome", "unknown"),
                    "exit_code": event.get("exit_code"),
                    "result_artifacts": event.get("result_artifacts", []),
                }
            )
    return sorted(attempts.values(), key=lambda row: row["started_at"])


def result_by_condition(
    results: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    observed = {}
    for execution_id, result in results.items():
        if result.get("workload") == EXTERNAL_WORKLOAD:
            continue
        condition_id = condition_id_from_result(result, manifest, execution_id)
        if condition_id is None:
            continue
        if condition_id in observed:
            raise ValueError(f"multiple successful results exist for {condition_id}")
        observed[condition_id] = result
    return observed


def condition_summaries(
    manifest: dict[str, Any],
    observed_results: dict[str, dict[str, Any]],
    executions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = {}
    active_outcomes = {"PROVISIONING", "RUNNING", "FINALIZING"}
    for condition in manifest["conditions"]:
        condition_id = str(condition["condition_id"])
        result = observed_results.get(condition_id)
        condition_executions = [
            execution for execution in executions if execution.get("condition_id") == condition_id
        ]
        outcomes = [execution.get("outcome") for execution in condition_executions]
        if result is not None:
            status = "SUCCEEDED"
        elif any(outcome in active_outcomes for outcome in outcomes):
            status = "RUNNING"
        elif condition.get("status") == "failed_before_training" or "FAILED" in outcomes:
            status = "FAILED"
        elif "ELIGIBLE" in outcomes:
            status = "READY"
        else:
            status = "PENDING"
        metrics = (
            result_metrics(result)
            if result is not None
            else {
                key: None
                for key in (
                    "initial_successes",
                    "final_successes",
                    "gain",
                    "paired_improved",
                    "paired_regressed",
                    "paired_net_improved",
                    "paired_p_value",
                    "policy_updates",
                    "optimizer_updates",
                    "sampled_completions",
                    "reached_complexity_level",
                    "maximum_sampled_complexity_level",
                    "retention_passed",
                    "final_evaluation_complete",
                    "adapter_persisted",
                    "stop_reason",
                )
            }
        )
        summaries[condition_id] = {
            "role": condition.get("role"),
            "status": status,
            "seed": metrics.get("seed", condition.get("optimization_seed")),
            "branch_width": metrics.get(
                "branch_width",
                condition.get("branch_width"),
            ),
            "policy_mutation_enabled": metrics.get(
                "policy_mutation_enabled",
                condition.get("policy_mutation_enabled"),
            ),
            "declared_completion_budget": condition.get("completion_budget"),
            "execution_ids": [execution["execution_id"] for execution in condition_executions],
            **metrics,
        }
    return summaries


def decision(
    *,
    complete: bool,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "PASS" if complete and passed else "FAIL" if complete else "INCOMPLETE",
        "evidence": evidence,
    }


def clean_positive_gain(result: dict[str, Any] | None) -> bool:
    if result is None:
        return False
    paired = result.get("paired_test_change")
    gain = result.get("reward_gain")
    net_improved = paired.get("net_improved") if isinstance(paired, dict) else None
    return (
        result.get("final_evaluation_complete") is True
        and not isinstance(gain, bool)
        and isinstance(gain, (int, float))  # noqa: UP038 - Python 3.9.
        and math.isfinite(float(gain))
        and gain > 0
        and isinstance(paired, dict)
        and not isinstance(net_improved, bool)
        and isinstance(net_improved, (int, float))  # noqa: UP038 - Python 3.9.
        and net_improved > 0
        and paired.get("regressed") == 0
    )


def compare_matched_conditions(
    trained: dict[str, Any] | None,
    control: dict[str, Any] | None,
    *,
    expect_control_policy_mutation: bool,
    expected_control_branch_width: int,
) -> dict[str, Any]:
    if trained is None or control is None:
        return decision(complete=False, passed=False, evidence={})
    trained_study = trained.get("study")
    control_study = control.get("study")
    trained_paired = trained.get("paired_test_change")
    control_paired = control.get("paired_test_change")
    complete = all(
        isinstance(value, dict)
        for value in (trained_study, control_study, trained_paired, control_paired)
    )
    if not complete:
        return decision(complete=False, passed=False, evidence={})
    trained_completion_budget = sampled_completions(trained)
    control_completion_budget = sampled_completions(control)
    matched_splits = (
        trained_study.get("optimization_seed") == control_study.get("optimization_seed")
        and trained_study.get("validation_seed_base") == control_study.get("validation_seed_base")
        and trained_study.get("test_seed_base") == control_study.get("test_seed_base")
    )
    matched_completion_budget = (
        isinstance(trained_completion_budget, int)
        and trained_completion_budget > 0
        and trained_completion_budget == control_completion_budget
    )
    trained_tasks = evaluation_task_ids(trained)
    control_tasks = evaluation_task_ids(control)
    matched_tasks = bool(trained_tasks) and trained_tasks == control_tasks
    paired_trained_vs_control = paired_condition_change(control, trained)
    trained_identity = frozen_runtime_identity(trained)
    control_identity = frozen_runtime_identity(control)
    matched_runtime = (
        all(value is not None for value in trained_identity.values())
        and trained_identity == control_identity
    )
    matched_start = (
        trained.get("initial_reward") == control.get("initial_reward")
        and not isinstance(trained.get("initial_reward"), bool)
        and isinstance(trained.get("initial_reward"), (int, float))  # noqa: UP038
    )
    branch_contract = (
        trained_study.get("branch_width") == 4
        and control_study.get("branch_width") == expected_control_branch_width
        and trained.get("branch_width") == 4
        and control.get("branch_width") == expected_control_branch_width
    )
    trained_net = trained_paired.get("net_improved")
    control_net = control_paired.get("net_improved")
    trained_final = trained.get("final_reward")
    control_final = control.get("final_reward")
    numerical = all(
        isinstance(value, (int, float))  # noqa: UP038 - Python 3.9.
        and not isinstance(value, bool)
        for value in (trained_net, control_net, trained_final, control_final)
    )
    control_mutation = control_study.get("policy_mutation_enabled")
    if expect_control_policy_mutation:
        mutation_contract = (
            control_mutation is True
            and isinstance(control.get("effective_policy_update_count"), int)
            and not isinstance(control.get("effective_policy_update_count"), bool)
            and control["effective_policy_update_count"] > 0
            and isinstance(control.get("effective_optimizer_update_count"), int)
            and not isinstance(control.get("effective_optimizer_update_count"), bool)
            and control["effective_optimizer_update_count"] > 0
        )
    else:
        attempted_optimizer_updates = control_study.get("attempted_optimizer_update_count")
        optimizer_step_calls = control_study.get("optimizer_step_calls")
        mutation_contract = (
            control_mutation is False
            and control_study.get("parameter_restore_verified") is True
            and isinstance(optimizer_step_calls, int)
            and not isinstance(optimizer_step_calls, bool)
            and optimizer_step_calls > 0
            and optimizer_step_calls == attempted_optimizer_updates
            and control.get("effective_policy_update_count") == 0
            and control.get("effective_optimizer_update_count") == 0
            and isinstance(control_study.get("restored_parameter_tensors"), int)
            and not isinstance(control_study.get("restored_parameter_tensors"), bool)
            and control_study["restored_parameter_tensors"] > 0
        )
    evaluation_complete = (
        trained.get("final_evaluation_complete") is True
        and control.get("final_evaluation_complete") is True
    )
    passed = (
        matched_splits
        and matched_completion_budget
        and matched_tasks
        and paired_trained_vs_control is not None
        and paired_trained_vs_control["net_improved"] > 0
        and matched_runtime
        and matched_start
        and branch_contract
        and mutation_contract
        and evaluation_complete
        and numerical
        and trained_net > control_net
        and trained_final > control_final
    )
    return decision(
        complete=True,
        passed=passed,
        evidence={
            "matched_seed_and_splits": matched_splits,
            "matched_completion_budget": matched_completion_budget,
            "completion_budget": trained_completion_budget,
            "matched_evaluation_tasks": matched_tasks,
            "evaluation_task_count": len(trained_tasks),
            "paired_trained_vs_control": paired_trained_vs_control,
            "matched_frozen_runtime": matched_runtime,
            "matched_initial_score": matched_start,
            "branch_contract_verified": branch_contract,
            "mutation_contract_verified": mutation_contract,
            "final_evaluations_complete": evaluation_complete,
            "trained_final": trained_final,
            "control_final": control_final,
            "trained_net_improved": trained_net,
            "control_net_improved": control_net,
            "control_policy_mutation_enabled": control_mutation,
        },
    )


def external_decision(
    external_result: dict[str, Any] | None,
    expected_condition_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if external_result is None:
        incomplete = decision(complete=False, passed=False, evidence={})
        return incomplete, incomplete
    adapters = external_result.get("adapters")
    if not isinstance(adapters, list):
        incomplete = decision(complete=False, passed=False, evidence={})
        return incomplete, incomplete
    observed = {
        adapter.get("condition_id"): adapter
        for adapter in adapters
        if isinstance(adapter, dict) and isinstance(adapter.get("condition_id"), str)
    }
    all_evaluated = (
        external_result.get("external_evaluation_completed") is True
        and external_result.get("every_adapter_reported") is True
        and set(observed) == set(expected_condition_ids)
    )
    evaluation = decision(
        complete=True,
        passed=all_evaluated,
        evidence={
            "expected_condition_ids": expected_condition_ids,
            "observed_condition_ids": sorted(observed),
            "task_count": external_result.get("task_count"),
        },
    )
    fresh_ids = ["k4_train_seed307", "k4_train_seed701"]
    transfer_rows = {}
    for condition_id in fresh_ids:
        adapter = observed.get(condition_id)
        paired = adapter.get("paired_change_vs_base") if isinstance(adapter, dict) else None
        transfer_rows[condition_id] = paired
    transfer_complete = all(isinstance(value, dict) for value in transfer_rows.values())
    transfer_passed = transfer_complete and all(
        value.get("net_improved", 0) > 0 and value.get("regressed") == 0
        for value in transfer_rows.values()
    )
    transfer = decision(
        complete=transfer_complete,
        passed=transfer_passed,
        evidence=transfer_rows,
    )
    return evaluation, transfer


def policy_external_metrics(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": policy.get("policy_id"),
        "condition_id": policy.get("condition_id"),
        "role": policy.get("role"),
        "optimization_seed": policy.get("optimization_seed"),
        "examples": policy.get("examples"),
        "exact_successes": policy.get("exact_successes"),
        "exact_rate": policy.get("exact_rate"),
        "domain_successes": policy.get("domain_successes"),
        "action_protocol_validity_rate": policy.get("action_protocol_validity_rate"),
        "paired_change_vs_base": policy.get("paired_change_vs_base"),
    }


def external_evaluation_metrics(
    external_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if external_result is None:
        return None
    base = external_result.get("base")
    adapters = external_result.get("adapters")
    if not isinstance(base, dict) or not isinstance(adapters, list):
        return None
    base_outcomes = base.get("task_outcomes")
    base_by_task = (
        {
            outcome.get("task_id"): outcome
            for outcome in base_outcomes
            if isinstance(outcome, dict) and isinstance(outcome.get("task_id"), str)
        }
        if isinstance(base_outcomes, list)
        else {}
    )
    task_rows = []
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        outcomes = adapter.get("task_outcomes")
        if not isinstance(outcomes, list):
            continue
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            task_id = outcome.get("task_id")
            baseline = base_by_task.get(task_id)
            if not isinstance(task_id, str) or not isinstance(baseline, dict):
                continue
            base_solved = baseline.get("solved")
            adapter_solved = outcome.get("solved")
            if base_solved is False and adapter_solved is True:
                transition = "improved"
            elif base_solved is True and adapter_solved is False:
                transition = "regressed"
            elif base_solved is adapter_solved and isinstance(base_solved, bool):
                transition = "unchanged"
            else:
                transition = "invalid"
            task_rows.append(
                {
                    "condition_id": adapter.get("condition_id"),
                    "task_id": task_id,
                    "domain": outcome.get("domain"),
                    "base_solved": base_solved,
                    "adapter_solved": adapter_solved,
                    "transition": transition,
                    "failed_checks": outcome.get("failed_checks"),
                }
            )
    return {
        "result_digest": external_result.get("result_digest"),
        "pack_id": (
            external_result.get("pack", {}).get("pack_id")
            if isinstance(external_result.get("pack"), dict)
            else None
        ),
        "task_count": external_result.get("task_count"),
        "domain_task_counts": external_result.get("domain_task_counts"),
        "base": policy_external_metrics(base),
        "adapters": [
            policy_external_metrics(adapter) for adapter in adapters if isinstance(adapter, dict)
        ],
        "task_transitions": task_rows,
    }


def assemble_report(
    manifest: dict[str, Any],
    executions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    generated_at: str,
    operator_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_condition = result_by_condition(results, manifest)
    seed307 = by_condition.get("k4_train_seed307")
    seed701 = by_condition.get("k4_train_seed701")
    no_update = by_condition.get("k4_no_update_seed307")
    k1 = by_condition.get("k1_train_seed307")
    fresh_complete = seed307 is not None and seed701 is not None
    replication = decision(
        complete=fresh_complete,
        passed=clean_positive_gain(seed307) and clean_positive_gain(seed701),
        evidence={
            "k4_train_seed307": (result_metrics(seed307) if seed307 is not None else None),
            "k4_train_seed701": (result_metrics(seed701) if seed701 is not None else None),
        },
    )
    no_update_decision = compare_matched_conditions(
        seed307,
        no_update,
        expect_control_policy_mutation=False,
        expected_control_branch_width=4,
    )
    k1_decision = compare_matched_conditions(
        seed307,
        k1,
        expect_control_policy_mutation=True,
        expected_control_branch_width=1,
    )
    expected_adapter_ids = [
        condition["condition_id"]
        for condition in manifest["conditions"]
        if condition.get("role")
        in {
            "frozen_reference",
            "fresh_replication",
            "matched_frozen_policy_control",
            "branch_width_ablation",
        }
    ]
    external_results = [
        result for result in results.values() if result.get("workload") == EXTERNAL_WORKLOAD
    ]
    if len(external_results) > 1:
        raise ValueError("multiple successful external evaluations exist")
    evaluation, transfer = external_decision(
        external_results[0] if external_results else None,
        expected_adapter_ids,
    )
    external_metrics = external_evaluation_metrics(
        external_results[0] if external_results else None
    )
    regression_complete = fresh_complete
    regression_passed = regression_complete and all(
        result.get("retention_passed") is True
        and result.get("paired_test_change", {}).get("regressed") == 0
        for result in (seed307, seed701)
        if result is not None
    )
    regression = decision(
        complete=regression_complete,
        passed=regression_passed,
        evidence={
            condition_id: (
                {
                    "retention_passed": result.get("retention_passed"),
                    "paired_regressed": result.get("paired_test_change", {}).get("regressed"),
                }
                if result is not None
                else None
            )
            for condition_id, result in (
                ("k4_train_seed307", seed307),
                ("k4_train_seed701", seed701),
            )
        },
    )
    decisions = {
        "k4_training_beats_frozen_policy_k4": no_update_decision,
        "k4_training_beats_matched_k1": k1_decision,
        "gains_repeat_across_fresh_seeds": replication,
        "all_retained_adapters_evaluated_externally": evaluation,
        "fresh_k4_gains_transfer_without_regressions": transfer,
        "regression_guard_remains_clean": regression,
    }
    attempts = operator_attempt_rows(operator_events or [])
    rows = execution_rows(executions, results, manifest, attempts)
    conditions = condition_summaries(manifest, by_condition, rows)
    provider_failures = [
        row for row in rows if row["outcome"] == "FAILED" or row.get("error") is not None
    ]
    operator_failures = [
        row
        for row in attempts
        if row["outcome"] == "failed"
        or (
            isinstance(row.get("exit_code"), int)
            and not isinstance(row.get("exit_code"), bool)
            and row["exit_code"] != 0
        )
    ]
    overall_status = (
        "INCOMPLETE"
        if any(item["status"] == "INCOMPLETE" for item in decisions.values())
        else ("PASS" if all(item["status"] == "PASS" for item in decisions.values()) else "FAIL")
    )
    content = {
        "schema_version": 1,
        "report_id": REPORT_ID,
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "freeze": {
            key: manifest.get(key)
            for key in (
                "freeze_commit",
                "frozen_source_commit",
                "frozen_workload_revision",
                "frozen_objective_id",
                "model",
                "amendments",
            )
        },
        "aggregation_policy": (
            "Every execution, seed, and regression is reported before any conclusion; "
            "no mean replaces per-seed outcomes."
        ),
        "overall_status": overall_status,
        "conditions": conditions,
        "executions": rows,
        "operator_attempts": attempts,
        "external_evaluation": external_metrics,
        "failures": {
            "provider_executions": provider_failures,
            "operator_attempts": operator_failures,
        },
        "failure_count": {
            "provider_executions": len(provider_failures),
            "operator_attempts": len(operator_failures),
        },
        "decisions": decisions,
    }
    return {
        **content,
        "report_digest": "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest(),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Revision 30 confirmatory study",
        "",
        f"Status: **{report['overall_status']}**",
        "",
        "No seed or regression is averaged away. Every paid execution appears below.",
        "",
        f"Frozen workload: `{report['freeze']['frozen_workload_revision']}`",
        "",
        f"Frozen source: `{report['freeze']['frozen_source_commit']}`",
        "",
        "## Decisive tests",
        "",
        "| Test | Status |",
        "|---|---:|",
    ]
    for name, result in report["decisions"].items():
        lines.append(f"| {name.replace('_', ' ')} | {result['status']} |")
    lines.extend(
        [
            "",
            "## Optimization conditions",
            "",
            "| Condition | Status | Seed | K | Updates | Completions | Initial | Final | Gain | + | − |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_id, metrics in report["conditions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    condition_id,
                    format_value(metrics["status"]),
                    format_value(metrics["seed"]),
                    format_value(metrics["branch_width"]),
                    format_value(metrics["policy_updates"]),
                    format_value(metrics["sampled_completions"]),
                    format_value(metrics["initial_successes"]),
                    format_value(metrics["final_successes"]),
                    format_value(metrics["gain"]),
                    format_value(metrics["paired_improved"]),
                    format_value(metrics["paired_regressed"]),
                ]
            )
            + " |"
        )
    external = report["external_evaluation"]
    if external is not None:
        lines.extend(
            [
                "",
                "## External evaluation",
                "",
                f"Pack: `{format_value(external['pack_id'])}`",
                "",
                "| Policy | Seed | Exact | Micro repo | SQLite | Filesystem / CLI | + | − |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        policies = [external["base"], *external["adapters"]]
        for policy in policies:
            paired = policy.get("paired_change_vs_base")
            domains = policy.get("domain_successes")
            paired = paired if isinstance(paired, dict) else {}
            domains = domains if isinstance(domains, dict) else {}
            exact = (
                f"{format_value(policy.get('exact_successes'))} / "
                f"{format_value(policy.get('examples'))}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        format_value(
                            policy.get("condition_id") or policy.get("policy_id") or "base"
                        ),
                        format_value(policy.get("optimization_seed")),
                        exact,
                        format_value(domains.get("micro_repository")),
                        format_value(domains.get("sqlite_data_repair")),
                        format_value(domains.get("filesystem_cli")),
                        format_value(paired.get("improved")),
                        format_value(paired.get("regressed")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### External task transitions",
                "",
                "| Adapter | Task | Domain | Base | Adapter | Transition |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for transition in external["task_transitions"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        format_value(transition["condition_id"]),
                        format_value(transition["task_id"]),
                        format_value(transition["domain"]),
                        "pass" if transition["base_solved"] else "fail",
                        "pass" if transition["adapter_solved"] else "fail",
                        format_value(transition["transition"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Every execution",
            "",
            "| Execution | Condition | Outcome | GPU | Cost (estimated USD) | Teardown |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["executions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["execution_id"]),
                    format_value(row["condition_id"]),
                    format_value(row["outcome"]),
                    format_value(row["gpu_id"]),
                    format_value(row["estimated_cost_usd"]),
                    "yes" if row["teardown_confirmed"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Every operator attempt",
            "",
            "| Condition | Stage | Outcome | Started | Finished |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["operator_attempts"]:
        stage = row["category"] + ("_preflight" if row["preflight_only"] else "_run")
        lines.append(
            "| "
            + " | ".join(
                [
                    format_value(row["condition_id"]),
                    stage,
                    format_value(row["outcome"]),
                    format_value(row["started_at"]),
                    format_value(row["finished_at"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    provider_failures = report["failures"]["provider_executions"]
    operator_failures = report["failures"]["operator_attempts"]
    if provider_failures or operator_failures:
        for row in provider_failures:
            lines.append(
                f"- Provider `{row['execution_id']}` · "
                f"{format_value(row['condition_id'])} · "
                f"{format_value(row['error'])}"
            )
        for row in operator_failures:
            lines.append(
                f"- Operator `{row['condition_id']}` at `{row['started_at']}` · "
                f"exit {format_value(row['exit_code'])}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            f"Report digest: `{report['report_digest']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_durably(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        handle.write(payload)
        if not payload.endswith(b"\n"):
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def fetch_executions(api_root: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(
        f"{api_root}/v1/research-compute-executions",
        timeout=15,
    ) as response:
        payload = json.load(response)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("research execution API returned an invalid response")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-root",
        default=os.environ.get("EQUINOX_API_ROOT", "http://127.0.0.1:8180"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("research/studies/revision30-confirmatory-results.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/revision30-confirmatory-report.md"),
    )
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(
        repository_root / "research/studies/revision30-confirmatory-study.json"
    )
    report = assemble_report(
        manifest,
        fetch_executions(arguments.api_root),
        result_files(repository_root / "var/research-proofs"),
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        operator_events=read_json_lines(repository_root / "var/revision30-study/attempts.jsonl"),
    )
    write_durably(
        repository_root / arguments.json_output,
        json.dumps(report, sort_keys=True, indent=2).encode(),
    )
    write_durably(
        repository_root / arguments.markdown_output,
        render_markdown(report).encode(),
    )
    print(json.dumps({"status": report["overall_status"], "digest": report["report_digest"]}))


if __name__ == "__main__":
    main()
