from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.runpod.repository_repair_study as study


def configure(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    *,
    completion_budget: int | None = None,
) -> study.StudyConfiguration:
    monkeypatch.setenv("EQUINOX_STUDY_CONDITION", condition)
    monkeypatch.setenv("EQUINOX_RL_SEED", "211")
    monkeypatch.setenv("EQUINOX_STUDY_VALIDATION_SEED_BASE", "20000000")
    monkeypatch.setenv("EQUINOX_STUDY_TEST_SEED_BASE", "50000000")
    monkeypatch.setenv(
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE",
        "12" if condition == "k1_train" else "4",
    )
    if completion_budget is None:
        monkeypatch.delenv("EQUINOX_STUDY_COMPLETION_BUDGET", raising=False)
    else:
        monkeypatch.setenv("EQUINOX_STUDY_COMPLETION_BUDGET", str(completion_budget))
    return study.study_configuration_from_environment()


def base_result() -> dict:
    return {
        "total_task_groups": 110,
        "excluded_task_groups": 0,
        "policy_update_count": 15,
        "optimizer_update_count": 17,
        "retention_passed": True,
        "reward_gain": 0.1875,
        "paired_test_change": {
            "regressed": 0,
            "mcnemar_exact_p_value": 0.00390625,
        },
        "final_evaluation_complete": True,
        "branch_snapshots": [{"siblings": [{}]}],
        "hypothesis_passed": True,
        "training_configuration": {},
    }


def test_fresh_k4_replication_keeps_frozen_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration = configure(monkeypatch, "k4_train")
    evidence = study.RuntimeEvidence(sampled_completions=440)

    result = study.augment_result(base_result(), configuration, evidence)

    assert configuration.branch_width == 4
    assert configuration.policy_mutation_enabled is True
    assert result["total_sampled_completions"] == 440
    assert result["effective_policy_update_count"] == 15
    assert result["study"]["frozen_source_commit"] == study.FROZEN_SOURCE_COMMIT


def test_no_update_control_records_attempts_but_no_effective_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = configure(monkeypatch, "k4_no_update", completion_budget=440)
    evidence = study.RuntimeEvidence(
        sampled_completions=440,
        optimizer_step_calls=17,
        restored_parameter_tensors=119,
        parameter_restore_verified=True,
    )

    result = study.augment_result(base_result(), configuration, evidence)

    assert result["study"]["attempted_policy_update_count"] == 15
    assert result["study"]["effective_policy_update_count"] == 0
    assert result["study"]["parameter_restore_verified"] is True
    assert result["probative_post_training"] is False
    assert result["hypothesis_passed"] is False


def test_k1_uses_declared_estimator_and_matched_completion_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = configure(monkeypatch, "k1_train", completion_budget=110)
    result_input = base_result()
    result_input["total_task_groups"] = 110
    evidence = study.RuntimeEvidence(sampled_completions=110)

    result = study.augment_result(result_input, configuration, evidence)

    assert study.k1_verified_success_advantages([0.95]) == [1.0]
    assert study.k1_verified_success_advantages([0.0]) == [0.0]
    assert result["branch_snapshots"][0]["siblings"] == [{}]
    assert result["restored_branching_observed"] is False
    assert result["single_trajectory_observed"] is True
    assert result["algorithm"] == "verified-fix-positive-only-policy-gradient-k1"
    assert result["study"]["advantage_estimator"] == study.K1_ADVANTAGE_ESTIMATOR


def test_controls_require_a_completion_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="requires EQUINOX_STUDY_COMPLETION_BUDGET"):
        configure(monkeypatch, "k4_no_update")
    with pytest.raises(ValueError, match="requires EQUINOX_STUDY_COMPLETION_BUDGET"):
        configure(monkeypatch, "k1_train")


def test_no_update_control_executes_optimizer_then_restores_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTensor:
        def __init__(self, value: float) -> None:
            self.value = value
            self.requires_grad = True

        def detach(self) -> FakeTensor:
            return self

        def clone(self) -> FakeTensor:
            return FakeTensor(self.value)

        def copy_(self, other: FakeTensor) -> None:
            self.value = other.value

    class FakeAdamW:
        def __init__(self, parameter: FakeTensor) -> None:
            self.param_groups = [{"params": [parameter]}]
            self.step_calls = 0

        def step(self, closure: object = None) -> None:
            del closure
            self.step_calls += 1
            self.param_groups[0]["params"][0].value += 0.25

    fake_torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=FakeAdamW),
        no_grad=contextlib.nullcontext,
        equal=lambda left, right: left.value == right.value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    parameter = FakeTensor(1.0)
    optimizer = FakeAdamW(parameter)
    evidence = study.RuntimeEvidence()

    study.install_matched_compute_no_update(
        evidence,
        tmp_path / "runtime-evidence.json",
    )
    optimizer.step()

    assert parameter.value == 1.0
    assert optimizer.step_calls == 1
    assert evidence.optimizer_step_calls == 1
    assert evidence.restored_parameter_tensors == 1
    assert evidence.parameter_restore_verified is True


def test_frozen_source_digests_and_preregistration_are_current() -> None:
    study.verify_frozen_sources()
    repository_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (repository_root / "research/studies/revision30-confirmatory-study.json").read_text(
            encoding="utf-8"
        )
    )
    amendment = json.loads(
        (repository_root / "research/studies/revision30-confirmatory-amendment-3.json").read_text(
            encoding="utf-8"
        )
    )
    hardware_amendment = json.loads(
        (repository_root / "research/studies/revision30-confirmatory-amendment-4.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["study_id"] == study.STUDY_ID
    assert [item["condition_id"] for item in manifest["conditions"]] == [
        "k4_train_seed113",
        "k4_train_seed211",
        "k4_train_seed307",
        "k4_train_seed701",
        "k4_no_update_seed307",
        "k1_train_seed307",
    ]
    assert manifest["conditions"][1]["role"] == "protocol_ineligible_failure"
    assert manifest["conditions"][1]["training_started"] is False
    assert amendment["task_outcomes_observed_before_roster_finalization"] is False
    assert amendment["final_condition_roster"]["fresh_replications"] == [
        "k4_train_seed307",
        "k4_train_seed701",
    ]
    assert amendment["final_condition_roster"]["matched_causal_controls"] == [
        "k4_no_update_seed307",
        "k1_train_seed307",
    ]
    assert hardware_amendment["algorithm_change"] is False
    assert hardware_amendment["triggering_failure"]["training_started"] is False
    assert manifest["conditions"][2]["gpu_id"] == "NVIDIA A40"
    assert manifest["conditions"][3]["gpu_id"] == "NVIDIA RTX PRO 4500 Blackwell"
