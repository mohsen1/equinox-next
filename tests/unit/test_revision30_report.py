from __future__ import annotations

from copy import deepcopy

from research.runpod.revision30_external_eval import WORKLOAD as EXTERNAL_WORKLOAD
from research.runpod.revision30_report import (
    assemble_report,
    clean_positive_gain,
    operator_attempt_rows,
    render_markdown,
)


def manifest() -> dict:
    return {
        "study_id": "repository-repair-confirmatory-study@1",
        "freeze_commit": "freeze",
        "frozen_source_commit": "source",
        "frozen_workload_revision": "runpod-repository-repair-causal-credit@30",
        "frozen_objective_id": "verified-fix-coverage-retention-policy-gradient@15",
        "model": {"id": "model", "revision": "model-revision"},
        "amendments": ["amendment-1"],
        "conditions": [
            {
                "condition_id": "k4_train_seed113",
                "role": "frozen_reference",
            },
            {
                "condition_id": "k4_train_seed307",
                "role": "fresh_replication",
            },
            {
                "condition_id": "k4_train_seed701",
                "role": "fresh_replication",
            },
            {
                "condition_id": "k4_no_update_seed307",
                "role": "matched_frozen_policy_control",
            },
            {
                "condition_id": "k1_train_seed307",
                "role": "branch_width_ablation",
            },
        ],
    }


def training_result(
    condition: str,
    seed: int,
    *,
    branch_width: int = 4,
    mutation_enabled: bool = True,
    initial: int = 10,
    final: int = 14,
    completions: int = 400,
    improved: int = 4,
    regressed: int = 0,
    outcome_pattern: tuple[bool, bool] = (True, True),
) -> dict:
    validation_seed = 220_000_000 if seed == 307 else 620_000_000
    test_seed = 260_000_000 if seed == 307 else 660_000_000
    study = {
        "study_id": "repository-repair-confirmatory-study@1",
        "condition": condition,
        "optimization_seed": seed,
        "validation_seed_base": validation_seed,
        "test_seed_base": test_seed,
        "branch_width": branch_width,
        "policy_mutation_enabled": mutation_enabled,
        "effective_policy_update_count": 3 if mutation_enabled else 0,
        "effective_optimizer_update_count": 3 if mutation_enabled else 0,
    }
    if not mutation_enabled:
        study.update(
            {
                "attempted_optimizer_update_count": 3,
                "optimizer_step_calls": 3,
                "parameter_restore_verified": True,
                "restored_parameter_tensors": 24,
            }
        )
    return {
        "workload": "repository-repair-restored-continuation-post-training",
        "model_id": "model",
        "model_revision": "model-revision",
        "workload_revision": "runpod-repository-repair-causal-credit@30",
        "environment_revision": "environment-revision",
        "verifier_revision": "verifier-revision",
        "action_protocol_revision": "action-revision",
        "objective_id": "verified-fix-coverage-retention-policy-gradient@15",
        "seed": seed,
        "branch_width": branch_width,
        "total_sampled_completions": completions,
        "initial_reward": initial,
        "final_reward": final,
        "reward_gain": final - initial,
        "paired_test_change": {
            "improved": improved,
            "regressed": regressed,
            "net_improved": improved - regressed,
            "mcnemar_exact_p_value": 0.03125,
        },
        "final_by_level": {
            "0": {
                "task_outcomes": [
                    {
                        "semantic_task_id": "shared-task-1",
                        "solved": outcome_pattern[0],
                    },
                    {
                        "semantic_task_id": "shared-task-2",
                        "solved": outcome_pattern[1],
                    },
                ]
            }
        },
        "effective_policy_update_count": 3 if mutation_enabled else 0,
        "effective_optimizer_update_count": 3 if mutation_enabled else 0,
        "retention_passed": True,
        "final_evaluation_complete": True,
        "adapter_persisted": True,
        "study": study,
    }


def complete_results() -> dict[str, dict]:
    reference = training_result("k4_train", 113, final=13, improved=3)
    seed307 = training_result("k4_train", 307, final=14, improved=4)
    seed701 = training_result("k4_train", 701, final=13, improved=3)
    no_update = training_result(
        "k4_no_update",
        307,
        mutation_enabled=False,
        final=11,
        improved=1,
        outcome_pattern=(True, False),
    )
    k1 = training_result(
        "k1_train",
        307,
        branch_width=1,
        final=12,
        improved=2,
        outcome_pattern=(True, False),
    )
    condition_ids = [
        "k4_train_seed113",
        "k4_train_seed307",
        "k4_train_seed701",
        "k4_no_update_seed307",
        "k1_train_seed307",
    ]
    base_outcomes = [
        {
            "task_id": "external-1",
            "domain": "micro_repository",
            "solved": False,
            "failed_checks": ["hidden-test"],
        },
        {
            "task_id": "external-2",
            "domain": "sqlite_data_repair",
            "solved": True,
            "failed_checks": [],
        },
    ]
    external = {
        "workload": EXTERNAL_WORKLOAD,
        "external_evaluation_completed": True,
        "every_adapter_reported": True,
        "task_count": 2,
        "domain_task_counts": {
            "micro_repository": 1,
            "sqlite_data_repair": 1,
            "filesystem_cli": 0,
        },
        "result_digest": "sha256:external",
        "pack": {"pack_id": "post-freeze-pack"},
        "base": {
            "policy_id": "disabled_adapter_base",
            "examples": 2,
            "exact_successes": 1,
            "exact_rate": 0.5,
            "domain_successes": {
                "micro_repository": 0,
                "sqlite_data_repair": 1,
                "filesystem_cli": 0,
            },
            "action_protocol_validity_rate": 1.0,
            "task_outcomes": base_outcomes,
        },
        "adapters": [
            {
                "condition_id": condition_id,
                "policy_id": condition_id,
                "optimization_seed": (
                    113
                    if condition_id == "k4_train_seed113"
                    else 701
                    if condition_id == "k4_train_seed701"
                    else 307
                ),
                "examples": 2,
                "exact_successes": (
                    2 if condition_id in {"k4_train_seed307", "k4_train_seed701"} else 1
                ),
                "exact_rate": (
                    1.0 if condition_id in {"k4_train_seed307", "k4_train_seed701"} else 0.5
                ),
                "domain_successes": {
                    "micro_repository": (
                        1 if condition_id in {"k4_train_seed307", "k4_train_seed701"} else 0
                    ),
                    "sqlite_data_repair": 1,
                    "filesystem_cli": 0,
                },
                "action_protocol_validity_rate": 1.0,
                "task_outcomes": [
                    {
                        **outcome,
                        "solved": (
                            True
                            if outcome["task_id"] == "external-1"
                            and condition_id in {"k4_train_seed307", "k4_train_seed701"}
                            else outcome["solved"]
                        ),
                    }
                    for outcome in base_outcomes
                ],
                "paired_change_vs_base": {
                    "improved": (
                        1 if condition_id in {"k4_train_seed307", "k4_train_seed701"} else 0
                    ),
                    "net_improved": (
                        1 if condition_id in {"k4_train_seed307", "k4_train_seed701"} else 0
                    ),
                    "regressed": 0,
                },
            }
            for condition_id in condition_ids
        ],
    }
    return {
        "runpod-proof-reference": reference,
        "runpod-proof-seed307": seed307,
        "runpod-proof-seed701": seed701,
        "runpod-proof-no-update": no_update,
        "runpod-proof-k1": k1,
        "runpod-proof-external": external,
    }


def test_failure_complete_report_passes_only_verified_causal_contracts() -> None:
    report = assemble_report(
        manifest(),
        [],
        complete_results(),
        generated_at="2026-07-28T00:00:00+00:00",
    )

    assert report["overall_status"] == "PASS"
    assert all(decision["status"] == "PASS" for decision in report["decisions"].values())
    no_update = report["decisions"]["k4_training_beats_frozen_policy_k4"]
    assert no_update["evidence"]["matched_completion_budget"] is True
    assert no_update["evidence"]["matched_evaluation_tasks"] is True
    assert no_update["evidence"]["paired_trained_vs_control"]["net_improved"] == 1
    assert no_update["evidence"]["mutation_contract_verified"] is True
    assert report["conditions"]["k4_train_seed307"]["gain"] == 4
    assert report["external_evaluation"]["base"]["exact_successes"] == 1
    assert len(report["external_evaluation"]["task_transitions"]) == 10
    assert report["external_evaluation"]["task_transitions"][2]["transition"] == "improved"
    assert "mean" not in report


def test_report_rejects_named_no_update_control_without_byte_restore_evidence() -> None:
    results = complete_results()
    results["runpod-proof-no-update"]["study"]["parameter_restore_verified"] = False

    report = assemble_report(
        manifest(),
        [],
        results,
        generated_at="2026-07-28T00:00:00+00:00",
    )

    decision = report["decisions"]["k4_training_beats_frozen_policy_k4"]
    assert decision["status"] == "FAIL"
    assert decision["evidence"]["mutation_contract_verified"] is False
    assert report["overall_status"] == "FAIL"


def test_incomplete_evidence_takes_precedence_over_an_observed_failure() -> None:
    results = complete_results()
    del results["runpod-proof-no-update"]
    del results["runpod-proof-external"]
    results["runpod-proof-seed701"]["reward_gain"] = -1
    results["runpod-proof-seed701"]["final_reward"] = 9
    results["runpod-proof-seed701"]["paired_test_change"] = {
        "improved": 0,
        "regressed": 1,
        "net_improved": -1,
        "mcnemar_exact_p_value": 1.0,
    }

    report = assemble_report(
        manifest(),
        [],
        results,
        generated_at="2026-07-28T00:00:00+00:00",
    )

    assert report["decisions"]["gains_repeat_across_fresh_seeds"]["status"] == "FAIL"
    assert report["decisions"]["k4_training_beats_frozen_policy_k4"]["status"] == "INCOMPLETE"
    assert report["overall_status"] == "INCOMPLETE"


def test_boolean_gain_cannot_masquerade_as_a_positive_result() -> None:
    result = training_result("k4_train", 307)
    result["reward_gain"] = True

    assert clean_positive_gain(result) is False


def test_every_provider_failure_and_operator_attempt_is_rendered() -> None:
    executions = [
        {
            "execution_id": "runpod-proof-failed",
            "name": "failed condition",
            "workload_id": "repository-repair-restored-continuation-post-training",
            "status": "FAILED",
            "provider_handle": "pod-1",
            "resource_profile": {"gpu_id": "NVIDIA A40", "hourly_cost_usd": 0.4},
            "progress": {
                "elapsed_seconds": 180,
                "error": "protocol gate failed",
            },
            "started_at": "2026-07-28T01:00:00Z",
            "completed_at": "2026-07-28T01:03:00Z",
            "teardown_confirmed": True,
        }
    ]
    events = [
        {
            "event": "attempt_started",
            "condition_id": "k4_train_seed307",
            "started_at": "2026-07-28T01:00:00Z",
            "preflight_only": False,
        },
        {
            "event": "attempt_finished",
            "condition_id": "k4_train_seed307",
            "started_at": "2026-07-28T01:00:00Z",
            "finished_at": "2026-07-28T01:03:00Z",
            "preflight_only": False,
            "outcome": "failed",
            "exit_code": 1,
        },
        {
            "event": "attempt_started",
            "condition_id": "k4_train_seed701",
            "started_at": "2026-07-28T02:00:00Z",
            "preflight_only": False,
        },
    ]

    report = assemble_report(
        manifest(),
        executions,
        complete_results(),
        generated_at="2026-07-28T00:00:00+00:00",
        operator_events=events,
    )
    markdown = render_markdown(report)

    assert report["failure_count"] == {
        "provider_executions": 1,
        "operator_attempts": 1,
    }
    assert operator_attempt_rows(events)[-1]["outcome"] == "running"
    assert "runpod-proof-failed" in markdown
    assert "k4_train_seed701" in markdown
    assert "protocol gate failed" in markdown
    assert "External task transitions" in markdown
    assert "post-freeze-pack" in markdown


def test_decision_digest_changes_when_a_per_seed_result_changes() -> None:
    original = complete_results()
    changed = deepcopy(original)
    changed["runpod-proof-seed701"]["final_reward"] += 1

    first = assemble_report(
        manifest(),
        [],
        original,
        generated_at="2026-07-28T00:00:00+00:00",
    )
    second = assemble_report(
        manifest(),
        [],
        changed,
        generated_at="2026-07-28T00:00:00+00:00",
    )

    assert first["report_digest"] != second["report_digest"]
