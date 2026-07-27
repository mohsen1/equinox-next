import pytest

from research.runpod.study_summary import student_t_critical_95, summarize


def receipt(seed: int, gain: float) -> dict:
    return {
        "seed": seed,
        "workload_revision": "runpod-repository-repair-loo-reinforce@7",
        "objective_id": "leave-one-out-group-normalized-reinforce@1",
        "teacher_data_used": False,
        "adapter_persisted": True,
        "probative_post_training": True,
        "resumed_from_checkpoint": False,
        "attempt_count": 1,
        "training_state_checkpointed": True,
        "cumulative_elapsed_seconds": 7_100.0,
        "final_evaluation_complete": True,
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "model_revision": "model-revision",
        "environment_revision": "environment-revision",
        "verifier_revision": "verifier-revision",
        "action_protocol_revision": "action-revision",
        "target_runtime_seconds": 7200,
        "training_configuration": {"test_examples": 12},
        "complexity_levels": [{"level": 0}],
        "final_by_level": {
            "0": {
                "split": "test",
                "task_outcomes": [
                    {
                        "task_id": "test-a",
                        "semantic_task_id": "semantic-test-a",
                        "solved": True,
                    }
                ],
            }
        },
        "reward_gain": gain,
        "restored_branching_observed": True,
        "policy_update_count": 2,
    }


def test_study_summary_requires_distinct_seeds_and_reports_uncertainty() -> None:
    summary = summarize([receipt(11, 0.2), receipt(13, 0.3), receipt(17, 0.4)])

    assert summary["seed_count"] == 3
    assert summary["sample_standard_deviation"] > 0
    assert len(summary["mean_reward_gain_95ci"]) == 2
    assert summary["study_digest"].startswith("sha256:")
    assert summary["study_identity"]["model_id"].endswith("3B-Instruct")
    assert student_t_critical_95(4) == 3.182
    assert student_t_critical_95(12) == 2.228
    assert student_t_critical_95(40) == 2.042

    with pytest.raises(ValueError, match="distinct"):
        summarize([receipt(11, 0.2), receipt(11, 0.3), receipt(17, 0.4)])
    boolean_seed = receipt(19, 0.3)
    boolean_seed["seed"] = True
    with pytest.raises(ValueError, match="distinct"):
        summarize([receipt(11, 0.2), receipt(13, 0.3), boolean_seed])

    mismatched = receipt(19, 0.3)
    mismatched["model_revision"] = "different"
    with pytest.raises(ValueError, match="do not share"):
        summarize([receipt(11, 0.2), receipt(13, 0.3), mismatched])

    missing_task_identities = [receipt(21, 0.2), receipt(23, 0.3), receipt(29, 0.4)]
    for item in missing_task_identities:
        item["final_by_level"]["0"]["task_outcomes"][0]["task_id"] = None
    with pytest.raises(ValueError, match="task identities"):
        summarize(missing_task_identities)

    resumed = receipt(19, 0.3)
    resumed["resumed_from_checkpoint"] = True
    resumed["attempt_count"] = 2
    resumed_summary = summarize([receipt(11, 0.2), receipt(13, 0.3), resumed])
    assert resumed_summary["resumed_seed_count"] == 1
    assert resumed_summary["eligibility_policy"] == "FENCED_RESUME_COUNTS_AS_LOGICAL_RUN"

    incomplete = receipt(19, 0.3)
    incomplete["final_evaluation_complete"] = False
    with pytest.raises(ValueError, match="identity or persistence"):
        summarize([receipt(11, 0.2), receipt(13, 0.3), incomplete])

    for malformed_gain in (None, True, float("nan")):
        malformed = receipt(19, 0.3)
        malformed["reward_gain"] = malformed_gain
        with pytest.raises(ValueError, match="reward gain"):
            summarize([receipt(11, 0.2), receipt(13, 0.3), malformed])

    untrained = receipt(19, 0.0)
    untrained["policy_update_count"] = 0
    untrained["restored_branching_observed"] = False
    with pytest.raises(ValueError, match="identity or persistence"):
        summarize([receipt(11, 0.2), receipt(13, 0.3), untrained])
