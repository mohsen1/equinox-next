import pytest

from research.runpod.study_summary import summarize


def receipt(seed: int, gain: float) -> dict:
    return {
        "seed": seed,
        "workload_revision": "runpod-repository-repair-loo-reinforce@2",
        "objective_id": "leave-one-out-group-normalized-reinforce@1",
        "teacher_data_used": False,
        "adapter_persisted": True,
        "final_by_level": {"0": {"split": "test"}},
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

    with pytest.raises(ValueError, match="distinct"):
        summarize([receipt(11, 0.2), receipt(11, 0.3), receipt(17, 0.4)])
