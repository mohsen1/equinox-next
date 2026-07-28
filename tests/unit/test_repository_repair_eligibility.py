from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import research.runpod.repository_repair_eligibility as eligibility
import research.runpod.repository_repair_study as study


def configuration() -> study.StudyConfiguration:
    return study.StudyConfiguration(
        condition="k4_train",
        optimization_seed=307,
        validation_seed_base=220_000_000,
        test_seed_base=260_000_000,
        branch_width=4,
        policy_mutation_enabled=True,
        advantage_estimator=study.K4_ADVANTAGE_ESTIMATOR,
        completion_budget=None,
        training_tasks_per_update=4,
    )


def test_eligibility_result_discloses_only_protocol_screening_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQUINOX_RL_MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")
    monkeypatch.setenv("EQUINOX_RL_VALIDATION_EXAMPLES", "8")

    result = eligibility.eligibility_result(
        configuration(),
        action_protocol_validity_rate=0.995,
        minimum_protocol_validity_rate=0.99,
        elapsed_seconds=123.4567,
        device="cuda",
        validation_examples=8,
    )

    assert result["protocol_eligible"] is True
    assert result["training_started"] is False
    assert result["optimizer_step_calls"] == 0
    assert result["test_split_accessed"] is False
    assert result["correctness_metrics_disclosed"] is False
    assert "exact_rate" not in result
    assert "task_outcomes" not in result


def test_screen_stops_frozen_trainer_at_protocol_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_progress: list[tuple[str, dict[str, object]]] = []

    def emit(
        phase: str,
        message: str,
        runtime_configuration: object,
        *,
        preserve_context: bool = False,
        **values: object,
    ) -> None:
        del message, runtime_configuration, preserve_context
        observed_progress.append((phase, values))

    monkeypatch.setattr(eligibility.frozen, "emit_progress", emit)
    eligibility.install_screen(configuration())

    with pytest.raises(eligibility.EligibilityScreenComplete) as completed:
        eligibility.frozen.emit_progress(
            "protocol_evaluation",
            "Protocol evaluated.",
            SimpleNamespace(validation_examples=8),
            action_protocol_validity_rate=0.987,
            minimum_protocol_validity_rate=0.99,
            elapsed_seconds=101.25,
            baseline_failure_family_ids=["hidden-from-screen"],
        )

    assert completed.value.result["protocol_eligible"] is False
    assert observed_progress == [
        (
            "protocol_evaluation",
            {
                "eligibility_screen_revision": eligibility.SCREEN_REVISION,
                "correctness_metrics_disclosed": False,
                "test_split_accessed": False,
                "action_protocol_validity_rate": 0.987,
                "minimum_protocol_validity_rate": 0.99,
                "elapsed_seconds": 101.25,
            },
        )
    ]
    assert not (tmp_path / "adapter").exists()
