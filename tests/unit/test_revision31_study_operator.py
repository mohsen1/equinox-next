import json
from pathlib import Path

import pytest

from research.runpod.revision31_study_operator import (
    STUDY_ID,
    all_condition_ids,
    condition_from_id,
    launcher_environment,
    load_manifest,
)


def manifest() -> dict:
    return load_manifest(
        Path(__file__).resolve().parents[2] / "research/studies/revision31-causal-study.json"
    )


def test_manifest_expands_to_twenty_matched_conditions() -> None:
    condition_ids = all_condition_ids(manifest())

    assert len(condition_ids) == 20
    assert len(set(condition_ids)) == 20
    assert "k1_scheduled_dynamic_seed137" in condition_ids
    assert "k4_adaptive_seed887" in condition_ids


def test_condition_resolves_preregistered_split_and_budget() -> None:
    condition = condition_from_id(manifest(), "k4_adaptive_seed269")

    assert condition == {
        "condition_id": "k4_adaptive_seed269",
        "condition": "k4_adaptive",
        "optimization_seed": 269,
        "validation_seed_base": 732000000,
        "test_seed_base": 832000000,
        "branch_width": 4,
        "curriculum_policy": "adaptive",
        "completion_budget": 320,
        "training_tasks_per_update": 3,
    }


def test_unregistered_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="not preregistered"):
        condition_from_id(manifest(), "k4_adaptive_seed999")


def test_launcher_environment_selects_revision31() -> None:
    study_manifest = manifest()
    condition = condition_from_id(study_manifest, "k1_scheduled_dynamic_seed443")

    environment = launcher_environment(
        study_manifest,
        condition,
        preflight_only=True,
    )

    assert environment["EQUINOX_STUDY_REVISION"] == "31"
    assert environment["EQUINOX_STUDY_CONDITION"] == "k1_scheduled_dynamic"
    assert environment["EQUINOX_RL_TRAINING_TASKS_PER_UPDATE"] == "12"
    assert environment["EQUINOX_STUDY_COMPLETION_BUDGET"] == "320"
    assert environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] == "1"


def test_study_identity_is_stable() -> None:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2] / "research/studies/revision31-causal-study.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["study_id"] == STUDY_ID
