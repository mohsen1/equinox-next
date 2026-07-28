import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.runpod.repository_repair_rl as repository_repair_rl
from research.runpod.repository_repair_env import (
    BRANCH_WIDTH,
    diagnostic_actions,
    encode_action,
    make_task,
    make_tasks,
    teacher_continuation_actions,
)
from research.runpod.repository_repair_rl import (
    DEFAULT_MODEL_ID,
    DEFAULT_RUNTIME_CONFIGURATION,
    OBJECTIVE_ID,
    SUPPORTED_MODELS,
    WORKLOAD_REVISION,
    BranchCollection,
    GeneratedAction,
    accepted_reference_actions,
    accumulated_reference_anchored_examples,
    action_protocol_counts,
    bounded_final_evaluation_reserve,
    checkpoint_target_disposition,
    checkpoint_validation_seed,
    collect_branch_group,
    collect_greedy_trajectory,
    complete_json_object,
    configure_from_environment,
    correctness_contrast_advantages,
    discarded_collection_accounting,
    emit_progress,
    evaluation_reward_summary,
    fixed_retention_guard_levels,
    lightweight_validation_history,
    next_malformed_action_window_streak,
    next_uninformative_group_streak,
    observation_mastered,
    paired_change_summary,
    paired_retention_guard_decision,
    persist_checkpoint,
    persist_named_adapter,
    policy_examples,
    positive_environment_integer,
    post_training_claim_strength,
    recent_action_protocol_summary,
    reference_anchored_examples,
    remove_orphan_checkpoint_target,
    remove_stale_checkpoint_targets,
    render_action_prompt,
    resumed_crash_tail_actions_unaccounted,
    retained_final_evaluation_reserve,
    retention_guard_decision,
    sampled_reverse_kl_penalty,
    select_representative_collection,
    serialize_branch_group,
    sibling_advantages,
    training_level_allocation,
    training_loop_entry,
    training_stop_decision,
    uninformative_group_limit_reached,
    validate_resume_state,
    validation_regression_decision,
    validation_window_seed,
    wilson_interval,
    workload_attempt_from_environment,
)

MODEL_REVISION = SUPPORTED_MODELS[DEFAULT_MODEL_ID]


@pytest.fixture(autouse=True)
def clear_runtime_configuration_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "EQUINOX_RL_MODEL_ID",
        "EQUINOX_RL_SEED",
        "EQUINOX_RL_TARGET_SECONDS",
        "EQUINOX_RL_MAX_RESUME_GAP_SECONDS",
        "EQUINOX_RL_MAX_UPDATES",
        "EQUINOX_RL_VALIDATION_EXAMPLES",
        "EQUINOX_RL_TEST_EXAMPLES",
        "EQUINOX_RL_MASTERY_WINDOWS",
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE",
        "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL",
        "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS",
        "EQUINOX_WORKLOAD_ATTEMPT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sibling_advantage_is_leave_one_out_centered_and_zero_for_ties() -> None:
    advantages = sibling_advantages([1.0, 0.0, 0.0, 0.0])

    assert advantages == [1.0, -0.33333333, -0.33333333, -0.33333333]
    assert abs(sum(advantages)) < 1e-7
    assert sibling_advantages([0.25] * BRANCH_WIDTH) == [0.0] * BRANCH_WIDTH


def test_sibling_advantage_floor_does_not_amplify_tiny_efficiency_gaps() -> None:
    advantages = sibling_advantages([0.95, 0.94, 0.94, 0.94])

    assert advantages == [0.1, -0.03333333, -0.03333333, -0.03333333]


def test_correctness_contrast_is_required_for_policy_signal() -> None:
    assert correctness_contrast_advantages([0.97, 0.975, 0.975, 0.97]) == [0.0] * 4
    assert correctness_contrast_advantages([0.0, 0.0, 0.0, 0.0]) == [0.0] * 4
    assert correctness_contrast_advantages([0.97, 0.0, 0.0, 0.0])[0] > 0


def test_training_allocation_keeps_a_majority_on_frontier_and_probes_two_levels() -> None:
    assert training_level_allocation(0, 4) == [0, 0, 1, 2]
    assert training_level_allocation(1, 4) == [1, 1, 2, 3]
    assert training_level_allocation(2, 4) == [2, 2, 2, 3]
    assert training_level_allocation(3, 4) == [3, 3, 3, 3]
    assert training_level_allocation(0, 1) == [0]

    with pytest.raises(ValueError, match="outside"):
        training_level_allocation(4, 4)


def test_retention_guard_requires_zero_paired_regressions() -> None:
    assert retention_guard_decision(
        best_fixed_successes=5,
        best_guard_net_improved=0,
        observed_fixed_successes=6,
        guard_change={"improved": 1, "regressed": 0, "net_improved": 1},
        consecutive_regressions=1,
    ) == (True, 6, 1, 0)
    assert retention_guard_decision(
        best_fixed_successes=5,
        best_guard_net_improved=0,
        observed_fixed_successes=6,
        guard_change={"improved": 1, "regressed": 1, "net_improved": 0},
        consecutive_regressions=0,
    ) == (False, 5, 0, 1)
    assert retention_guard_decision(
        best_fixed_successes=5,
        best_guard_net_improved=0,
        observed_fixed_successes=5,
        guard_change={"improved": 1, "regressed": 0, "net_improved": 1},
        consecutive_regressions=1,
    ) == (True, 5, 1, 0)


def test_cross_level_retention_guard_is_paired_and_lexicographic() -> None:
    assert fixed_retention_guard_levels(0) == [0, 1]
    assert fixed_retention_guard_levels(2) == [0, 1, 2, 3]
    assert paired_retention_guard_decision(
        best_fixed_net_improved=0,
        best_rotating_net_improved=0,
        fixed_change={"improved": 1, "regressed": 0, "net_improved": 1},
        rotating_change={"improved": 0, "regressed": 0, "net_improved": 0},
        consecutive_regressions=1,
    ) == (True, 1, 0, 0)
    assert paired_retention_guard_decision(
        best_fixed_net_improved=1,
        best_rotating_net_improved=0,
        fixed_change={"improved": 2, "regressed": 1, "net_improved": 1},
        rotating_change={"improved": 1, "regressed": 0, "net_improved": 1},
        consecutive_regressions=0,
    ) == (False, 1, 0, 1)
    assert paired_retention_guard_decision(
        best_fixed_net_improved=1,
        best_rotating_net_improved=0,
        fixed_change={"improved": 1, "regressed": 0, "net_improved": 1},
        rotating_change={"improved": 1, "regressed": 0, "net_improved": 1},
        consecutive_regressions=1,
    ) == (True, 1, 1, 0)


def test_sampled_reverse_kl_penalty_is_zero_at_reference_and_nonnegative() -> None:
    assert sampled_reverse_kl_penalty(0.0) == 0.0
    assert sampled_reverse_kl_penalty(-0.2) > 0.0
    assert sampled_reverse_kl_penalty(0.2) > 0.0
    assert sampled_reverse_kl_penalty(0.01) == pytest.approx(0.00005, rel=0.02)


def test_recent_action_protocol_window_is_group_bounded() -> None:
    summary = recent_action_protocol_summary(
        [(20, 0), (12, 1), (30, 0)],
        maximum_groups=2,
    )

    assert summary == {
        "groups": 2,
        "window_complete": True,
        "total_actions": 42,
        "malformed_actions": 1,
        "validity_rate": 0.97619,
        "malformed_rate": 0.02381,
    }


def test_malformed_action_stop_requires_two_consecutive_bad_windows() -> None:
    healthy = {
        "window_complete": True,
        "malformed_rate": 0.05,
    }
    incomplete = {
        "window_complete": False,
        "malformed_rate": 0.20,
    }
    bad = {
        "window_complete": True,
        "malformed_rate": 0.050001,
    }

    assert next_malformed_action_window_streak(1, healthy) == 0
    assert next_malformed_action_window_streak(1, incomplete) == 0
    assert next_malformed_action_window_streak(0, bad) == 1
    assert next_malformed_action_window_streak(1, bad) == 2


def test_uninformative_group_limit_cannot_be_reset_after_it_is_reached() -> None:
    collections = [
        SimpleNamespace(informative=False),
        SimpleNamespace(informative=True),
    ]

    assert (
        uninformative_group_limit_reached(
            4,
            collections,  # type: ignore[arg-type]
            maximum_consecutive_groups=5,
        )
        is True
    )
    assert next_uninformative_group_streak(4, collections) == 0  # type: ignore[arg-type]


def test_validation_regression_requires_two_lower_windows() -> None:
    assert validation_regression_decision(
        best_exact_successes=6,
        observed_exact_successes=5,
        consecutive_regressions=0,
    ) == (False, 6, 1)
    assert validation_regression_decision(
        best_exact_successes=6,
        observed_exact_successes=4,
        consecutive_regressions=1,
    ) == (False, 6, 2)
    assert validation_regression_decision(
        best_exact_successes=6,
        observed_exact_successes=7,
        consecutive_regressions=2,
    ) == (True, 7, 0)
    assert validation_regression_decision(
        best_exact_successes=6,
        observed_exact_successes=6,
        consecutive_regressions=1,
    ) == (False, 6, 0)


def test_lightweight_validation_history_drops_task_payloads() -> None:
    assert lightweight_validation_history(
        [
            {
                "update": 5,
                "level": 0,
                "exact_rate": 0.75,
                "exact_successes": 6,
                "task_outcomes": [{"task_id": "hidden"}],
            }
        ]
    ) == [
        {
            "update": 5,
            "level": 0,
            "exact_successes": 6,
            "exact_rate": 0.75,
        }
    ]


def test_action_prompt_prefills_the_parser_contract() -> None:
    class Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert messages[0]["role"] == "system"
            assert "one JSON action at a time" in messages[0]["content"]
            assert messages[1] == {"role": "user", "content": "repair"}
            assert tokenize is False
            assert add_generation_prompt is True
            return "<assistant>"

    assert render_action_prompt(Tokenizer(), "repair") == '<assistant>{"tool":'


@pytest.mark.parametrize(
    ("response", "complete"),
    (
        ('{"tool":"test"}', True),
        ('{"tool":"edit","path":"a.py","old":"x","new":"return {\\"ok\\": true}"}', True),
        ('{"tool":"read"', False),
        ('["not","an","action"]', False),
        ('{"tool":"test"} trailing', False),
    ),
)
def test_complete_json_object_detects_only_a_closed_root_object(
    response: str,
    complete: bool,
) -> None:
    assert complete_json_object(response) is complete


def test_wilson_interval_preserves_uncertainty_at_zero_and_perfect_rates() -> None:
    assert wilson_interval(0, 4) == [0.0, 0.4899]
    assert wilson_interval(4, 4) == [0.5101, 1.0]
    assert observation_mastered(
        {
            "exact_rate_95ci": wilson_interval(7, 8),
            "checkpoint_rate_95ci": wilson_interval(8, 8),
        }
    )
    assert not observation_mastered(
        {
            "exact_rate_95ci": wilson_interval(6, 8),
            "checkpoint_rate_95ci": wilson_interval(8, 8),
        }
    )


def test_consecutive_mastery_windows_use_disjoint_validation_tasks() -> None:
    first_tasks = make_tasks(
        0,
        8,
        validation_window_seed(0, 5),
        split="validation",
    )
    first = frozenset(task.semantic_task_id for task in first_tasks)
    second = {
        task.semantic_task_id
        for task in make_tasks(
            0,
            8,
            validation_window_seed(0, 10),
            split="validation",
            exclude_semantic_task_ids=first,
        )
    }

    assert first.isdisjoint(second)


def test_checkpoint_validation_seed_is_fixed_per_level() -> None:
    assert checkpoint_validation_seed(0) == checkpoint_validation_seed(0)
    assert checkpoint_validation_seed(0) != checkpoint_validation_seed(1)


def test_final_evaluation_reserve_is_bounded_with_provenance() -> None:
    assert bounded_final_evaluation_reserve(200) == (300, False)
    assert bounded_final_evaluation_reserve(2_700) == (2_700, False)
    assert bounded_final_evaluation_reserve(2_701) == (2_700, True)
    reserve_seconds, exceeded_ceiling = bounded_final_evaluation_reserve(9_000)
    assert exceeded_ceiling is True
    assert training_stop_decision(
        elapsed_seconds=100,
        target_seconds=3_000,
        final_evaluation_reserve_seconds=reserve_seconds,
    ) == (False, None, 300)
    assert training_stop_decision(
        elapsed_seconds=100,
        target_seconds=3_000,
        final_evaluation_reserve_seconds=reserve_seconds,
        final_evaluation_reserve_exceeded_ceiling=True,
    ) == (True, "final_evaluation_reserve_ceiling", 300)


def test_final_evaluation_reserve_tracks_only_retained_checkpoints() -> None:
    current = {
        "current_seconds": 600,
        "current_maximum_measured_seconds": 500,
        "current_exceeded_ceiling": False,
        "candidate_measured_seconds": 1_900,
        "maximum_seconds": 1_500,
    }

    assert retained_final_evaluation_reserve(
        **current,
        candidate_retained=False,
    ) == (600, 500, False)
    assert retained_final_evaluation_reserve(
        **current,
        candidate_retained=True,
    ) == (1_500, 1_900, True)


def test_partial_final_evaluation_has_no_headline_reward() -> None:
    initial = {"0": {"exact_rate": 0.25}, "1": {"exact_rate": 1.0}}
    final = {"0": {"exact_rate": 0.75}, "1": {"exact_rate": 1.0}}

    assert evaluation_reward_summary(initial, final, complete=False) == (
        None,
        None,
        None,
    )
    assert evaluation_reward_summary(initial, final, complete=True) == (
        0.625,
        0.875,
        0.25,
    )
    assert (
        post_training_claim_strength(
            final_evaluation_complete=False,
            probative_post_training=False,
        )
        == "INCOMPLETE_FINAL_EVALUATION"
    )
    assert (
        post_training_claim_strength(
            final_evaluation_complete=True,
            probative_post_training=False,
        )
        == "NONPROBATIVE_RESERVE_STOP"
    )
    assert (
        post_training_claim_strength(
            final_evaluation_complete=True,
            probative_post_training=True,
        )
        == "EXPLORATORY_SINGLE_SEED"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("EQUINOX_RL_VALIDATION_EXAMPLES", "9"),
        ("EQUINOX_RL_TEST_EXAMPLES", "13"),
        ("EQUINOX_RL_TRAINING_TASKS_PER_UPDATE", "14"),
        ("EQUINOX_RL_MASTERY_WINDOWS", "3"),
    ),
)
def test_configuration_rejects_samples_larger_than_semantic_universes(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        configure_from_environment()


def test_configuration_attributes_early_failure_to_the_true_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQUINOX_WORKLOAD_ATTEMPT", "2")
    monkeypatch.setenv("EQUINOX_RL_VALIDATION_EXAMPLES", "9")

    with pytest.raises(ValueError, match="EQUINOX_RL_VALIDATION_EXAMPLES"):
        configure_from_environment()

    assert workload_attempt_from_environment() == 2


@pytest.mark.parametrize("target_seconds", ("not-a-number", "0", "21601"))
def test_configuration_rejects_invalid_target_runtime(
    monkeypatch: pytest.MonkeyPatch,
    target_seconds: str,
) -> None:
    monkeypatch.setenv("EQUINOX_RL_TARGET_SECONDS", target_seconds)

    with pytest.raises(ValueError, match="EQUINOX_RL_TARGET_SECONDS"):
        configure_from_environment()


@pytest.mark.parametrize("seed", ("not-a-number", "-1", "2147483648"))
def test_configuration_rejects_invalid_optimization_seed(
    monkeypatch: pytest.MonkeyPatch,
    seed: str,
) -> None:
    monkeypatch.setenv("EQUINOX_RL_SEED", seed)

    with pytest.raises(ValueError, match="EQUINOX_RL_SEED"):
        configure_from_environment()


def test_configuration_requires_training_time_before_final_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQUINOX_RL_TARGET_SECONDS", "2460")

    with pytest.raises(ValueError, match="at least 60 seconds"):
        configure_from_environment()


def test_terminal_resume_skips_training_and_preserves_stop_reason() -> None:
    assert training_loop_entry(
        {
            "training_complete": True,
            "stop_reason": "maximum_level_mastered",
        },
        updates_completed=45,
        maximum_updates=120,
    ) == (True, "maximum_level_mastered", 121)
    assert training_loop_entry(
        {"training_complete": False},
        updates_completed=45,
        maximum_updates=120,
    ) == (False, "maximum_updates", 46)


def test_training_stop_decision_centralizes_deadline_and_mastery_precedence() -> None:
    assert training_stop_decision(
        elapsed_seconds=100,
        target_seconds=1_000,
        final_evaluation_reserve_seconds=300,
    ) == (False, None, 700)
    assert training_stop_decision(
        elapsed_seconds=650,
        target_seconds=1_000,
        final_evaluation_reserve_seconds=300,
    ) == (True, "final_evaluation_reserve", 700)
    assert training_stop_decision(
        elapsed_seconds=650,
        target_seconds=1_000,
        final_evaluation_reserve_seconds=300,
        maximum_level_mastered=True,
    ) == (True, "maximum_level_mastered", 700)
    assert training_stop_decision(
        elapsed_seconds=100,
        target_seconds=1_000,
        final_evaluation_reserve_seconds=300,
        final_evaluation_reserve_exceeded_ceiling=True,
        maximum_level_mastered=True,
    ) == (True, "final_evaluation_reserve_ceiling", 700)


def test_progress_persists_workload_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(repository_repair_rl, "PROGRESS_PATH", str(progress_path))

    emit_progress(
        "dependency_setup",
        "Reloading the model.",
        runtime_configuration=replace(
            DEFAULT_RUNTIME_CONFIGURATION,
            workload_attempt=2,
            maximum_updates=60,
        ),
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["schema_version"] == 2
    assert progress["attempt"] == 2
    assert progress["maximum_updates"] == 60


def test_configuration_failure_progress_omits_default_runtime_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(repository_repair_rl, "PROGRESS_PATH", str(progress_path))

    emit_progress(
        "failed",
        "Repository repair workload failed.",
        runtime_configuration=None,
        attempt=2,
        error="ValueError: invalid configuration",
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["schema_version"] == 2
    assert progress["attempt"] == 2
    assert progress["error"] == "ValueError: invalid configuration"
    assert "maximum_updates" not in progress


def test_failure_progress_preserves_last_training_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "phase": "training",
                "update": 87,
                "current_level": 2,
                "maximum_updates": 120,
                "elapsed_seconds": 4100.5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(repository_repair_rl, "PROGRESS_PATH", str(progress_path))

    emit_progress(
        "failed",
        "Repository repair workload failed.",
        runtime_configuration=None,
        preserve_context=True,
        attempt=2,
        error="RuntimeError: CUDA out of memory",
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["update"] == 87
    assert progress["current_level"] == 2
    assert progress["maximum_updates"] == 120
    assert progress["elapsed_seconds"] == 4100.5


def test_crash_tail_disclosure_uses_checkpoint_training_state() -> None:
    assert not resumed_crash_tail_actions_unaccounted(None)
    assert not resumed_crash_tail_actions_unaccounted({"training_complete": True})
    assert resumed_crash_tail_actions_unaccounted({"training_complete": False})


def test_paired_change_summary_tracks_improvement_and_regression() -> None:
    summary = paired_change_summary(
        [
            {"task_id": "a", "solved": False},
            {"task_id": "b", "solved": False},
            {"task_id": "c", "solved": True},
        ],
        [
            {"task_id": "a", "solved": True},
            {"task_id": "b", "solved": False},
            {"task_id": "c", "solved": False},
        ],
    )

    assert summary == {
        "examples": 3,
        "improved": 1,
        "regressed": 1,
        "unchanged": 1,
        "net_improved": 0,
        "mcnemar_exact_p_value": 1.0,
    }
    with pytest.raises(ValueError, match="same non-empty task set"):
        paired_change_summary(
            [{"task_id": "a", "solved": False}],
            [{"task_id": "b", "solved": True}],
        )


def test_paired_change_summary_pins_significant_improvement() -> None:
    initial = [{"task_id": f"task-{index}", "solved": False} for index in range(6)]
    final = [{"task_id": f"task-{index}", "solved": True} for index in range(6)]

    summary = paired_change_summary(initial, final)

    assert summary["improved"] == 6
    assert summary["regressed"] == 0
    assert summary["mcnemar_exact_p_value"] == 0.03125


def test_paired_change_summary_never_rounds_nonzero_probability_to_zero() -> None:
    initial = [{"task_id": f"task-{index}", "solved": False} for index in range(48)]
    final = [{"task_id": f"task-{index}", "solved": True} for index in range(48)]

    probability = paired_change_summary(initial, final)["mcnemar_exact_p_value"]

    assert probability == 2 / 2**48
    assert probability > 0


def test_environment_integer_rejects_invalid_and_out_of_range_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQUINOX_TEST_COUNT", "3")
    with pytest.raises(ValueError, match="between 4 and 8"):
        positive_environment_integer(
            "EQUINOX_TEST_COUNT",
            4,
            minimum=4,
            maximum=8,
        )
    monkeypatch.setenv("EQUINOX_TEST_COUNT", "not-a-number")
    with pytest.raises(ValueError, match="integer between 4 and 8"):
        positive_environment_integer(
            "EQUINOX_TEST_COUNT",
            4,
            minimum=4,
            maximum=8,
        )


def test_resume_state_rejects_configuration_drift_and_preserves_budget() -> None:
    configuration = {
        "test_examples": 12,
        "maximum_resume_gap_seconds": 20,
    }
    state = {
        "seed": 73,
        "workload_revision": WORKLOAD_REVISION,
        "model_revision": MODEL_REVISION,
        "objective_id": OBJECTIVE_ID,
        "training_configuration": configuration,
        "cumulative_elapsed_seconds": 123.5,
        "checkpointed_at_unix_seconds": 1_000.0,
        "attempt_count": 1,
    }

    assert validate_resume_state(
        state,
        experiment_seed=73,
        training_configuration=configuration,
        resume_started_at_unix_seconds=1_010.0,
        workload_attempt=2,
        model_revision=MODEL_REVISION,
    ) == (133.5, 2, 10.0, 10.0)

    assert validate_resume_state(
        state,
        experiment_seed=73,
        training_configuration=configuration,
        resume_started_at_unix_seconds=2_000.0,
        workload_attempt=2,
        model_revision=MODEL_REVISION,
    ) == (143.5, 2, 1_000.0, 20.0)

    with pytest.raises(RuntimeError, match="identity"):
        validate_resume_state(
            state,
            experiment_seed=73,
            training_configuration={"test_examples": 16},
            resume_started_at_unix_seconds=1_010.0,
            workload_attempt=2,
            model_revision=MODEL_REVISION,
        )
    with pytest.raises(RuntimeError, match="attempt count"):
        validate_resume_state(
            state,
            experiment_seed=73,
            training_configuration=configuration,
            resume_started_at_unix_seconds=1_010.0,
            workload_attempt=1,
            model_revision=MODEL_REVISION,
        )
    del state["cumulative_elapsed_seconds"]
    with pytest.raises(RuntimeError, match="elapsed budget is missing"):
        validate_resume_state(
            state,
            experiment_seed=73,
            training_configuration=configuration,
            resume_started_at_unix_seconds=1_010.0,
            workload_attempt=2,
            model_revision=MODEL_REVISION,
        )


def test_checkpoint_target_disposition_distinguishes_live_and_orphan_state(
    tmp_path: Path,
) -> None:
    target = tmp_path / "update-0004"

    assert (
        checkpoint_target_disposition(
            str(target),
            checkpoint_name=target.name,
            previous_checkpoint=None,
        )
        == "create"
    )

    target.mkdir()
    assert (
        checkpoint_target_disposition(
            str(target),
            checkpoint_name=target.name,
            previous_checkpoint="update-0003",
        )
        == "replace_orphan"
    )
    assert (
        checkpoint_target_disposition(
            str(target),
            checkpoint_name=target.name,
            previous_checkpoint=target.name,
        )
        == "rewrite_live"
    )


@pytest.mark.parametrize("as_directory", (False, True))
def test_remove_orphan_checkpoint_target_handles_files_and_directories(
    tmp_path: Path,
    as_directory: bool,
) -> None:
    target = tmp_path / "update-0004"
    if as_directory:
        target.mkdir()
        (target / "partial").write_text("partial", encoding="utf-8")
    else:
        target.write_text("partial", encoding="utf-8")

    remove_orphan_checkpoint_target(str(target))

    assert not target.exists()


def test_remove_stale_checkpoint_targets_keeps_only_the_live_checkpoint(
    tmp_path: Path,
) -> None:
    for name in ("update-0001", "update-0002", "update-0003"):
        (tmp_path / name).mkdir()
    (tmp_path / "latest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes").mkdir()

    remove_stale_checkpoint_targets(str(tmp_path), "update-0003")

    assert not (tmp_path / "update-0001").exists()
    assert not (tmp_path / "update-0002").exists()
    assert (tmp_path / "update-0003").is_dir()
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "notes").is_dir()


@pytest.mark.parametrize(
    "crash_step",
    ("adapter_saved", "state_persisted", "pointer_persisted"),
)
def test_checkpoint_persistence_recovers_each_crash_window(
    tmp_path: Path,
    crash_step: str,
) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    latest = checkpoints_root / "latest.json"

    def save_adapter(target: str) -> None:
        Path(target, "adapter_config.json").write_text("{}", encoding="utf-8")
        Path(target, "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    def save_state(state: dict[str, object], target: str) -> None:
        Path(target).write_text(json.dumps(state), encoding="utf-8")

    def crash_after(step: str) -> None:
        if step == crash_step:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        persist_checkpoint(
            checkpoints_root=str(checkpoints_root),
            latest_checkpoint_path=str(latest),
            checkpoint_name="update-0004",
            state={"update": 4},
            save_adapter=save_adapter,
            save_state=save_state,
            after_step=crash_after,
        )

    persist_checkpoint(
        checkpoints_root=str(checkpoints_root),
        latest_checkpoint_path=str(latest),
        checkpoint_name="update-0004",
        state={"update": 4},
        save_adapter=save_adapter,
        save_state=save_state,
    )

    assert json.loads(latest.read_text(encoding="utf-8"))["checkpoint"] == "update-0004"
    live = checkpoints_root / "update-0004"
    assert json.loads((live / "training-state.pt").read_text(encoding="utf-8")) == {"update": 4}
    assert not list(checkpoints_root.rglob("*.pending"))


def test_named_adapter_checkpoint_replaces_atomically_with_metadata(
    tmp_path: Path,
) -> None:
    checkpoints_root = tmp_path / "checkpoints"

    def save_adapter(target: str) -> None:
        Path(target, "adapter_config.json").write_text("{}", encoding="utf-8")
        Path(target, "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    persist_named_adapter(
        checkpoints_root=str(checkpoints_root),
        name="best-validation",
        metadata={"kind": "best_validation", "update": 5},
        save_adapter=save_adapter,
    )
    persist_named_adapter(
        checkpoints_root=str(checkpoints_root),
        name="best-validation",
        metadata={"kind": "best_validation", "update": 10},
        save_adapter=save_adapter,
    )

    metadata = json.loads(
        (checkpoints_root / "best-validation" / "checkpoint-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata == {
        "schema_version": 1,
        "kind": "best_validation",
        "update": 10,
    }
    assert not (checkpoints_root / "best-validation.pending").exists()


def test_discarded_collection_accounting_includes_prefix_and_all_siblings() -> None:
    task = make_task(0, seed=31)

    class Steps:
        def __init__(self, count: int) -> None:
            self.steps = [object()] * count

    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=Steps(2),  # type: ignore[arg-type]
        siblings=[Steps(5) for _ in range(BRANCH_WIDTH)],  # type: ignore[list-item]
        generated_by_sibling=[
            [GeneratedAction(response="{}") for _ in range(3)] for _ in range(BRANCH_WIDTH)
        ],
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[1.0, -1.0, 0.0, 0.0],
        exclusion_reason=None,
        replay=False,
    )

    assert discarded_collection_accounting([collection]) == (1, 14, 12)


def test_branch_collection_stops_mid_group_at_training_deadline() -> None:
    task = make_task(0, seed=31)
    diagnostics = diagnostic_actions(task)
    calls = 0

    def sample(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        action = diagnostics[min(calls, len(diagnostics) - 1)]
        calls += 1
        return GeneratedAction(response=encode_action(action))

    collection = collect_branch_group(
        task,
        sample,
        stochastic=True,
        sampling_seed=5,
        deadline_reached=lambda: calls >= 3,
    )

    assert collection.exclusion_reason == "TRAINING_DEADLINE_REACHED"
    assert collection.returns == []
    assert collection.advantages == []
    assert discarded_collection_accounting([collection]) == (1, 3, 1)
    serialized = serialize_branch_group(collection, update=4)
    assert serialized["best_sibling_index"] is None
    assert serialized["siblings"][0]["return"] is None
    assert serialized["siblings"][0]["advantage"] is None
    assert serialized["siblings"][0]["policy_signal"] is False


def test_prefix_failure_is_excluded_without_manufacturing_returns() -> None:
    task = make_task(0, seed=12)

    def invalid_policy(_: str, __: bool, ___: int) -> GeneratedAction:
        return GeneratedAction(response='{"tool":"finish"}')

    collection = collect_branch_group(
        task,
        invalid_policy,
        stochastic=True,
        sampling_seed=5,
    )

    assert collection.exclusion_reason == "PREFIX_CHECKPOINT_NOT_REACHED"
    assert collection.snapshot is None
    assert collection.returns == []
    assert collection.advantages == []
    assert policy_examples(collection) == []


def test_greedy_evaluation_stops_before_sampling_after_its_deadline() -> None:
    sampled = False

    def sample(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal sampled
        sampled = True
        return GeneratedAction(response='{"tool":"test"}')

    outcome = collect_greedy_trajectory(
        make_task(0, seed=12),
        sample,
        sampling_seed=5,
        deadline_reached=lambda: True,
    )

    assert outcome is None
    assert sampled is False


def test_serialized_branch_has_one_prefix_checkpoint_and_four_step_lanes() -> None:
    task = make_task(0, seed=44)
    calls = 0
    diagnostic = (
        '{"tool":"list","path":""}',
        '{"tool":"read","path":"tests/failures.txt"}',
    )
    fault = task.faults[0]
    continuation = (
        encode_action({"tool": "read", "path": fault.path}),
        encode_action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        ),
        '{"tool":"test"}',
        '{"tool":"finish"}',
    )

    def scripted_policy(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        response = diagnostic[calls] if calls < 2 else continuation[(calls - 2) // BRANCH_WIDTH]
        calls += 1
        return GeneratedAction(response=response)

    collection = collect_branch_group(
        task,
        scripted_policy,
        stochastic=False,
        sampling_seed=100,
    )
    serialized = serialize_branch_group(
        collection,
        update=3,
        optimizer_update={"applied": False, "effective_batch_weight": 0.0},
    )

    assert serialized["checkpoint"]["fidelity"] == "logical_restore"
    assert serialized["checkpoint"]["static_branch_width"] == 4
    assert len(serialized["shared_prefix"]["steps"]) == 2
    assert len(serialized["siblings"]) == 4
    assert all(len(sibling["steps"]) == 4 for sibling in serialized["siblings"])
    assert len({sibling["sampling_seed"] for sibling in serialized["siblings"]}) == 4
    assert all(sibling["passed"] for sibling in serialized["siblings"])
    assert all(
        sibling["reward_components"]["terminal_aggregate"] == 0.97
        for sibling in serialized["siblings"]
    )
    assert serialized["optimizer_update"]["applied"] is False
    assert action_protocol_counts(collection) == (18, 0)
    assert next_uninformative_group_streak(4, [collection]) == 5


def test_training_prefix_is_greedy_while_siblings_remain_stochastic() -> None:
    task = make_task(0, seed=44)
    calls: list[bool] = []
    diagnostic = diagnostic_actions(task)
    continuation = teacher_continuation_actions(task)

    def scripted_policy(_: str, stochastic: bool, __: int) -> GeneratedAction:
        calls.append(stochastic)
        if len(calls) <= 2:
            action = diagnostic[len(calls) - 1]
        else:
            action = continuation[(len(calls) - 3) // BRANCH_WIDTH]
        return GeneratedAction(response=encode_action(action))

    collection = collect_branch_group(
        task,
        scripted_policy,
        stochastic=True,
        sampling_seed=100,
    )

    assert collection.solved_siblings == BRANCH_WIDTH
    assert len(collection.generated_prefix) == 2
    assert calls[:2] == [False, False]
    assert all(calls[2:])


def test_policy_examples_use_only_verified_success_actions() -> None:
    task = make_task(0, seed=8)
    generated = GeneratedAction(
        response='{"tool":"test"}',
        input_ids=(1, 2),
        attention_mask=(1, 1),
        completion_mask=(0, 1),
    )
    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=SimpleNamespace(steps=[object(), object()]),  # type: ignore[arg-type]
        siblings=[
            SimpleNamespace(
                steps=[
                    object(),
                    object(),
                    SimpleNamespace(accepted=True),
                    SimpleNamespace(accepted=True),
                ],
                terminal_reason="solved" if index == 0 else "horizon_exhausted",
            )
            for index in range(4)
        ],  # type: ignore[list-item]
        generated_by_sibling=[[generated, generated] for _ in range(4)],
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[3.0, -1.0, -1.0, -1.0],
        exclusion_reason=None,
        replay=False,
    )

    examples = policy_examples(collection)

    assert len(examples) == 2
    assert [example.weight for example in examples] == [0.5, 0.5]


def test_policy_examples_exclude_rejected_post_branch_actions() -> None:
    task = make_task(0, seed=8)
    accepted = GeneratedAction(
        response='{"tool":"test"}',
        input_ids=(1, 2),
        attention_mask=(1, 1),
        completion_mask=(0, 1),
    )
    rejected = GeneratedAction(
        response='{"tool":"finish"}',
        input_ids=(3, 4),
        attention_mask=(1, 1),
        completion_mask=(0, 1),
    )
    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=SimpleNamespace(steps=[object(), object()]),  # type: ignore[arg-type]
        siblings=[
            SimpleNamespace(
                steps=[
                    object(),
                    object(),
                    SimpleNamespace(accepted=True),
                    SimpleNamespace(accepted=False),
                ],
                terminal_reason="solved" if index == 0 else "horizon_exhausted",
            )
            for index in range(4)
        ],  # type: ignore[list-item]
        generated_by_sibling=[[accepted, rejected] for _ in range(4)],
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[3.0, -1.0, -1.0, -1.0],
        exclusion_reason=None,
        replay=False,
    )

    examples = policy_examples(collection)

    assert len(examples) == 1
    assert all(example.generated is accepted for example in examples)
    assert [example.weight for example in examples] == [1.0]


def test_reference_anchor_covers_prefix_and_all_accepted_continuations() -> None:
    task = make_task(0, seed=8)

    def generated(token: int) -> GeneratedAction:
        return GeneratedAction(
            response='{"tool":"test"}',
            input_ids=(1, token),
            attention_mask=(1, 1),
            completion_mask=(0, 1),
        )

    prefix_actions = [generated(10), generated(11)]
    sibling_actions = [[generated(20 + index), generated(30 + index)] for index in range(4)]
    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=SimpleNamespace(
            steps=[
                SimpleNamespace(accepted=True),
                SimpleNamespace(accepted=False),
            ]
        ),  # type: ignore[arg-type]
        siblings=[
            SimpleNamespace(
                steps=[
                    object(),
                    object(),
                    SimpleNamespace(accepted=True),
                    SimpleNamespace(accepted=False),
                ],
                terminal_reason="solved" if index == 0 else "horizon_exhausted",
            )
            for index in range(4)
        ],  # type: ignore[list-item]
        generated_by_sibling=sibling_actions,
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[3.0, -1.0, -1.0, -1.0],
        exclusion_reason=None,
        replay=False,
        generated_prefix=prefix_actions,
    )

    reference_actions = accepted_reference_actions(collection)
    anchored, policy_example_count = reference_anchored_examples([collection])

    assert reference_actions == [prefix_actions[0], *(actions[0] for actions in sibling_actions)]
    assert policy_example_count == 1
    assert [example.weight for example in anchored] == [0.0, 1.0, 0.0, 0.0, 0.0]
    gated_anchor_only, gated_policy_examples = reference_anchored_examples(
        [collection],
        minimum_informative_groups=2,
    )
    assert gated_policy_examples == 0
    assert len(gated_anchor_only) == 5
    assert all(example.weight == 0.0 for example in gated_anchor_only)
    anchor_only, no_policy_examples = reference_anchored_examples(
        [replace(collection, advantages=[0.0] * BRANCH_WIDTH)]
    )
    assert no_policy_examples == 0
    assert len(anchor_only) == 5
    assert all(example.weight == 0.0 for example in anchor_only)


def test_informative_groups_accumulate_across_anchor_only_updates() -> None:
    task = make_task(0, seed=8)

    def generated(token: int) -> GeneratedAction:
        return GeneratedAction(
            response='{"tool":"test"}',
            input_ids=(1, token),
            attention_mask=(1, 1),
            completion_mask=(0, 1),
        )

    prefix_action = generated(10)
    sibling_actions = [[generated(20 + index)] for index in range(4)]
    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=SimpleNamespace(
            steps=[SimpleNamespace(accepted=True)],
        ),  # type: ignore[arg-type]
        siblings=[
            SimpleNamespace(
                steps=[
                    object(),
                    SimpleNamespace(accepted=True),
                ],
                terminal_reason="solved" if index == 0 else "horizon_exhausted",
            )
            for index in range(4)
        ],  # type: ignore[list-item]
        generated_by_sibling=sibling_actions,
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[1.0, -0.33333333, -0.33333333, -0.33333333],
        exclusion_reason=None,
        replay=False,
        generated_prefix=[prefix_action],
    )

    (
        first_training,
        first_policy_count,
        pending_examples,
        pending_policy_count,
        pending_groups,
        first_signal_groups,
    ) = accumulated_reference_anchored_examples(
        [collection],
        [],
        0,
        [],
        minimum_informative_groups=2,
    )

    assert first_policy_count == 0
    assert first_signal_groups == []
    assert pending_groups == [collection.task.task_id]
    assert pending_policy_count == 1
    assert len(pending_examples) == 5
    assert first_training == []

    with pytest.raises(
        RuntimeError,
        match="informative task group was sampled more than once",
    ):
        accumulated_reference_anchored_examples(
            [collection],
            pending_examples,
            pending_policy_count,
            pending_groups,
            minimum_informative_groups=2,
        )

    second_collection = replace(collection, task=make_task(0, seed=9))
    (
        second_training,
        second_policy_count,
        remaining_examples,
        remaining_policy_count,
        remaining_groups,
        second_signal_groups,
    ) = accumulated_reference_anchored_examples(
        [second_collection],
        pending_examples,
        1,
        pending_groups,
        minimum_informative_groups=2,
    )

    assert second_policy_count == 2
    assert second_signal_groups == [
        collection.task.task_id,
        second_collection.task.task_id,
    ]
    assert remaining_examples == []
    assert remaining_policy_count == 0
    assert remaining_groups == []
    assert len(second_training) == 10


def test_representative_branch_prefers_informative_frontier_over_replay_and_exclusion() -> None:
    frontier = BranchCollection(
        task=make_task(2, seed=9),
        snapshot=None,
        prefix=None,  # type: ignore[arg-type]
        siblings=[],
        generated_by_sibling=[],
        sampling_seeds=[],
        returns=[],
        advantages=[1.0, -1.0, 0.0, 0.0],
        exclusion_reason=None,
        replay=False,
    )
    replay = BranchCollection(
        task=make_task(0, seed=10),
        snapshot=None,
        prefix=None,  # type: ignore[arg-type]
        siblings=[],
        generated_by_sibling=[],
        sampling_seeds=[],
        returns=[],
        advantages=[],
        exclusion_reason=None,
        replay=True,
    )
    excluded = BranchCollection(
        task=make_task(2, seed=11),
        snapshot=None,
        prefix=None,  # type: ignore[arg-type]
        siblings=[],
        generated_by_sibling=[],
        sampling_seeds=[],
        returns=[],
        advantages=[1.0, -1.0, 0.0, 0.0],
        exclusion_reason="PREFIX_CHECKPOINT_NOT_REACHED",
        replay=False,
    )

    assert (
        select_representative_collection(
            [replay, excluded, frontier],
            current_level=2,
        )
        is frontier
    )
