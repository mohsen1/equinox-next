from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runpod import repository_repair_rl as frozen
from research.runpod import repository_repair_study_v31 as study


def configure(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
) -> study.StudyConfiguration:
    monkeypatch.setenv("EQUINOX_STUDY_CONDITION", condition)
    monkeypatch.setenv("EQUINOX_RL_SEED", "137")
    monkeypatch.setenv("EQUINOX_STUDY_VALIDATION_SEED_BASE", "731000000")
    monkeypatch.setenv("EQUINOX_STUDY_TEST_SEED_BASE", "831000000")
    monkeypatch.setenv("EQUINOX_STUDY_COMPLETION_BUDGET", "320")
    monkeypatch.setenv(
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE",
        "12" if condition.startswith("k1_") else "3",
    )
    return study.study_configuration_from_environment()


@pytest.mark.parametrize(
    ("condition", "branch_width", "curriculum_policy", "tasks_per_update"),
    [
        ("k1_scheduled_dynamic", 1, "scheduled_dynamic", 12),
        ("k4_scheduled_dynamic", 4, "scheduled_dynamic", 3),
        ("k1_adaptive", 1, "adaptive", 12),
        ("k4_adaptive", 4, "adaptive", 3),
    ],
)
def test_preregistered_condition_configuration(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    branch_width: int,
    curriculum_policy: str,
    tasks_per_update: int,
) -> None:
    configuration = configure(monkeypatch, condition)

    assert configuration.branch_width == branch_width
    assert configuration.curriculum_policy == curriculum_policy
    assert configuration.training_tasks_per_update == tasks_per_update
    assert configuration.completion_budget == 320


def test_non_preregistered_completion_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(monkeypatch, "k4_adaptive")
    monkeypatch.setenv("EQUINOX_STUDY_COMPLETION_BUDGET", "316")

    with pytest.raises(ValueError, match="exactly 320"):
        study.study_configuration_from_environment()


def test_scheduled_curriculum_ignores_outcomes() -> None:
    failed = [SimpleNamespace(solved_siblings=0)]
    solved = [SimpleNamespace(solved_siblings=4)]

    assert study.scheduled_frontier_probe_decision(0, 1, failed) == (
        1,
        "scheduled_nearest_harder_probe",
    )
    assert study.scheduled_frontier_probe_decision(0, 1, solved) == (
        1,
        "scheduled_nearest_harder_probe",
    )
    assert (
        study.scheduled_mastery_windows(
            1,
            candidate_retained=False,
            candidate_mastered=False,
        )
        == 2
    )


def test_revision31_interface_digest_is_frozen() -> None:
    root = Path(__file__).resolve().parents[2] / "research/runpod"

    study.verify_revision31_sources(root)


def test_result_records_factorial_condition() -> None:
    configuration = study.StudyConfiguration(
        condition="k4_scheduled_dynamic",
        optimization_seed=137,
        validation_seed_base=731000000,
        test_seed_base=831000000,
        branch_width=4,
        curriculum_policy="scheduled_dynamic",
        completion_budget=320,
        training_tasks_per_update=3,
    )
    evidence = SimpleNamespace(sampled_completions=320)
    result = {
        "total_task_groups": 80,
        "excluded_task_groups": 0,
        "policy_update_count": 3,
        "optimizer_update_count": 3,
        "hypothesis_passed": False,
        "training_configuration": {},
        "paired_test_change": {
            "regressed": 0,
            "mcnemar_exact_p_value": 0.01,
        },
        "retention_passed": True,
        "reward_gain": 1,
        "final_evaluation_complete": True,
    }

    observed = study.augment_result(result, configuration, evidence)

    assert observed["study"]["condition"] == "k4_scheduled_dynamic"
    assert observed["complexity_strategy"] == "scheduled_dynamic"
    assert observed["training_configuration"]["curriculum_schedule_outcome_blinded"] is True
    assert observed["hypothesis_passed"] is True


def test_install_uses_revision31_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = configure(monkeypatch, "k4_adaptive")
    evidence = SimpleNamespace(
        sampled_completions=0,
        optimizer_step_calls=0,
        restored_parameter_tensors=0,
        parameter_restore_verified=True,
    )
    monkeypatch.setattr(frozen, "emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(study.revision30_study, "install_collection_budget", lambda *args: None)
    frozen_names = (
        "WORKLOAD_REVISION",
        "OBJECTIVE_ID",
        "SYSTEM_PROMPT",
        "ACTION_PROTOCOL_REVISION",
        "ENVIRONMENT_REVISION",
        "RepositoryRepairEnvironment",
        "VALIDATION_SEED_BASE",
        "TEST_SEED_BASE",
        "BRANCH_WIDTH",
        "training_stop_decision",
        "MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS",
        "MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS",
    )
    previous = {name: getattr(frozen, name) for name in frozen_names}
    previous_environment_branch_width = study.frozen_environment.BRANCH_WIDTH
    try:
        study.install_revision31_condition(
            configuration,
            evidence,
            tmp_path / "evidence.json",
        )

        assert frozen.RepositoryRepairEnvironment is study.interface.RepositoryRepairEnvironment
        assert frozen.ACTION_PROTOCOL_REVISION == "repository-repair-json-tools@4"
        assert frozen.ENVIRONMENT_REVISION == "repository-repair-simulator@5"
    finally:
        for name, value in previous.items():
            setattr(frozen, name, value)
        study.frozen_environment.BRANCH_WIDTH = previous_environment_branch_width
