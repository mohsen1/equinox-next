from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.runpod.study_operator import (
    condition_by_id,
    launcher_environment,
    load_manifest,
    matching_success_results,
    resolve_completion_budget,
)


def manifest() -> dict:
    return {
        "study_id": "repository-repair-confirmatory-study@1",
        "model": {"id": "Qwen/Qwen2.5-Coder-3B-Instruct"},
        "shared_configuration": {
            "validation_examples": 8,
            "test_examples_per_level": 12,
            "mastery_windows": 2,
            "replay_tasks_per_level": 1,
            "maximum_updates": 120,
            "target_runtime_seconds": 10_800,
            "maximum_resume_gap_seconds": 2_700,
            "maximum_final_evaluation_reserve_seconds": 2_700,
        },
        "conditions": [
            {
                "condition_id": "k4_train_seed113",
                "role": "frozen_reference",
            },
            {
                "condition_id": "k4_train_seed211",
                "role": "fresh_replication",
                "optimization_seed": 211,
                "validation_seed_base": 20_000_000,
                "test_seed_base": 50_000_000,
                "training_tasks_per_update": 4,
                "completion_budget": None,
            },
            {
                "condition_id": "k1_train_seed211",
                "role": "branch_width_ablation",
                "optimization_seed": 211,
                "validation_seed_base": 20_000_000,
                "test_seed_base": 50_000_000,
                "training_tasks_per_update": 12,
                "completion_budget": "k4_train_seed211.total_sampled_completions",
            },
        ],
    }


def write_result(
    path: Path,
    *,
    completions: int = 516,
    seed: int = 211,
    validation_seed: int = 20_000_000,
    test_seed: int = 50_000_000,
) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_completed": True,
                "final_evaluation_complete": True,
                "adapter_persisted": True,
                "total_sampled_completions": completions,
                "study": {
                    "study_id": "repository-repair-confirmatory-study@1",
                    "condition": "k4_train",
                    "optimization_seed": seed,
                    "validation_seed_base": validation_seed,
                    "test_seed_base": test_seed,
                },
            }
        ),
        encoding="utf-8",
    )


def test_operator_selects_conditions_and_refuses_frozen_reference(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    loaded = load_manifest(path)

    assert condition_by_id(loaded, "k4_train_seed211")["optimization_seed"] == 211
    with pytest.raises(ValueError, match="not runnable"):
        condition_by_id(loaded, "k4_train_seed113")


def test_dependent_conditions_resolve_exactly_one_successful_budget(
    tmp_path: Path,
) -> None:
    condition = condition_by_id(manifest(), "k1_train_seed211")
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_completion_budget(condition, tmp_path)

    source = tmp_path / "runpod-proof-source.result.json"
    write_result(source)
    assert resolve_completion_budget(condition, tmp_path) == 516
    assert (
        matching_success_results(
            tmp_path,
            condition_by_id(manifest(), "k4_train_seed211"),
        )[0][0]
        == source
    )

    write_result(tmp_path / "runpod-proof-duplicate.result.json")
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_completion_budget(condition, tmp_path)


def test_dependent_budget_follows_the_shared_causal_seed(tmp_path: Path) -> None:
    condition = {
        "condition_id": "k4_no_update_seed307",
        "role": "matched_frozen_policy_control",
        "optimization_seed": 307,
        "validation_seed_base": 220_000_000,
        "test_seed_base": 260_000_000,
        "completion_budget": "k4_train_seed307.total_sampled_completions",
    }
    write_result(
        tmp_path / "runpod-proof-seed307.result.json",
        completions=444,
        seed=307,
        validation_seed=220_000_000,
        test_seed=260_000_000,
    )

    assert resolve_completion_budget(condition, tmp_path) == 444


def test_launcher_environment_matches_preregistered_k1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNRELATED_OPERATOR_SETTING", "preserved")
    condition = condition_by_id(manifest(), "k1_train_seed211")

    environment = launcher_environment(
        manifest(),
        condition,
        completion_budget=516,
        preflight_only=True,
    )

    assert environment["EQUINOX_STUDY_CONDITION"] == "k1_train"
    assert environment["EQUINOX_RL_TRAINING_TASKS_PER_UPDATE"] == "12"
    assert environment["EQUINOX_RL_TEST_EXAMPLES"] == "12"
    assert environment["EQUINOX_RL_TARGET_SECONDS"] == "10800"
    assert environment["EQUINOX_STUDY_COMPLETION_BUDGET"] == "516"
    assert environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] == "1"
    assert environment["UNRELATED_OPERATOR_SETTING"] == "preserved"
