"""Aggregate independent RunPod receipts without upgrading single-seed claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_WORKLOAD = "runpod-repository-repair-verified-success@21"
EXPECTED_OBJECTIVE = "verified-success-accumulated-retention-policy-gradient@11"
NUMBER_TYPES = (int, float)


def student_t_critical_95(sample_count: int) -> float:
    degrees_of_freedom = sample_count - 1
    critical_values = {
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if degrees_of_freedom < 2:
        raise ValueError("at least three samples are required for a study interval")
    if degrees_of_freedom in critical_values:
        return critical_values[degrees_of_freedom]
    smaller = sorted(
        (value for value in critical_values if value < degrees_of_freedom),
        reverse=True,
    )
    return critical_values[smaller[0]]


def study_identity(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_revision": result.get("workload_revision"),
        "objective_id": result.get("objective_id"),
        "model_id": result.get("model_id"),
        "model_revision": result.get("model_revision"),
        "environment_revision": result.get("environment_revision"),
        "verifier_revision": result.get("verifier_revision"),
        "action_protocol_revision": result.get("action_protocol_revision"),
        "target_runtime_seconds": result.get("target_runtime_seconds"),
        "training_configuration": result.get("training_configuration"),
        "complexity_levels": result.get("complexity_levels"),
        "test_task_ids": {
            level: [outcome.get("task_id") for outcome in observation.get("task_outcomes", [])]
            for level, observation in result.get("final_by_level", {}).items()
        },
        "test_semantic_task_ids": {
            level: [
                outcome.get("semantic_task_id") for outcome in observation.get("task_outcomes", [])
            ]
            for level, observation in result.get("final_by_level", {}).items()
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 3:
        raise ValueError("at least three independent optimization seeds are required")
    observed_seeds = [item.get("seed") for item in results]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in observed_seeds) or len(
        set(observed_seeds)
    ) != len(observed_seeds):
        raise ValueError("study receipts require distinct integer optimization seeds")
    ordered_results = sorted(results, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in ordered_results]
    expected_identity = study_identity(ordered_results[0])
    if any(value is None for value in expected_identity.values()):
        raise ValueError("study receipt omitted matched-budget identity")
    if not all(expected_identity["test_task_ids"].values()) or not all(
        expected_identity["test_semantic_task_ids"].values()
    ):
        raise ValueError("study receipt omitted paired test task identities")
    if any(
        not isinstance(task_id, str) or not task_id
        for identity_kind in ("test_task_ids", "test_semantic_task_ids")
        for level_task_ids in expected_identity[identity_kind].values()
        for task_id in level_task_ids
    ):
        raise ValueError("study receipt omitted paired test task identities")
    if any(
        len(level_task_ids) != len(set(level_task_ids))
        for level_task_ids in expected_identity["test_semantic_task_ids"].values()
    ):
        raise ValueError("study receipt repeats a semantic test task")
    for result in ordered_results:
        policy_update_count = result.get("policy_update_count")
        resumed = result.get("resumed_from_checkpoint")
        attempt_count = result.get("attempt_count")
        cumulative_elapsed_seconds = result.get("cumulative_elapsed_seconds")
        attempt_evidence_valid = (resumed is False and attempt_count == 1) or (
            resumed is True
            and attempt_count == 2
            and result.get("training_state_checkpointed") is True
        )
        if (
            result.get("workload_revision") != EXPECTED_WORKLOAD
            or result.get("objective_id") != EXPECTED_OBJECTIVE
            or result.get("teacher_data_used") is not False
            or result.get("adapter_persisted") is not True
            or result.get("probative_post_training") is not True
            or not attempt_evidence_valid
            or isinstance(cumulative_elapsed_seconds, bool)
            or not isinstance(cumulative_elapsed_seconds, NUMBER_TYPES)
            or not math.isfinite(float(cumulative_elapsed_seconds))
            or float(cumulative_elapsed_seconds) < 0
            or result.get("final_evaluation_complete") is not True
            or isinstance(policy_update_count, bool)
            or not isinstance(policy_update_count, int)
            or policy_update_count < 1
            or result.get("restored_branching_observed") is not True
        ):
            raise ValueError("study receipt identity or persistence evidence is invalid")
        observations = list(result.get("final_by_level", {}).values())
        if not observations or any(item.get("split") != "test" for item in observations):
            raise ValueError("study receipt does not contain the final test split")
        if study_identity(result) != expected_identity:
            raise ValueError("study receipts do not share one model, budget, and test set")
        reward_gain = result.get("reward_gain")
        if (
            isinstance(reward_gain, bool)
            or not isinstance(reward_gain, NUMBER_TYPES)
            or not math.isfinite(float(reward_gain))
        ):
            raise ValueError("study receipt reward gain must be a finite number")

    gains = [float(item["reward_gain"]) for item in ordered_results]
    mean_gain = statistics.mean(gains)
    standard_deviation = statistics.stdev(gains)
    standard_error = standard_deviation / math.sqrt(len(gains))
    critical_value = student_t_critical_95(len(gains))
    interval = [
        mean_gain - critical_value * standard_error,
        mean_gain + critical_value * standard_error,
    ]
    content = {
        "schema_version": 1,
        "workload_revision": EXPECTED_WORKLOAD,
        "objective_id": EXPECTED_OBJECTIVE,
        "seed_count": len(seeds),
        "seeds": seeds,
        "study_identity": expected_identity,
        "reward_gains": gains,
        "mean_reward_gain": mean_gain,
        "sample_standard_deviation": standard_deviation,
        "mean_reward_gain_95ci": interval,
        "all_runs_observed_branching": all(
            item.get("restored_branching_observed") is True for item in ordered_results
        ),
        "all_runs_had_policy_updates": all(
            int(item.get("policy_update_count", 0)) > 0 for item in ordered_results
        ),
        "resumed_seed_count": sum(
            item.get("resumed_from_checkpoint") is True for item in ordered_results
        ),
        "eligibility_policy": "FENCED_RESUME_COUNTS_AS_LOGICAL_RUN",
    }
    return {
        **content,
        "replicated_positive_gain": (
            interval[0] > 0
            and content["all_runs_observed_branching"]
            and content["all_runs_had_policy_updates"]
        ),
        "study_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    arguments = parser.parse_args()
    result = summarize([json.loads(path.read_text(encoding="utf-8")) for path in arguments.results])
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
