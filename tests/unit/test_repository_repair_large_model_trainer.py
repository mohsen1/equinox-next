from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.runpod import repository_repair_large_model_study as study
from research.runpod import repository_repair_large_model_trainer as trainer


class FakeOptimizer:
    def __init__(self, state: dict[str, float] | None = None) -> None:
        self.state = state or {"momentum": 2.0}

    def state_dict(self) -> dict[str, float]:
        return self.state

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.state = state


def test_transactional_trainer_has_its_own_workload_and_objective_identity() -> None:
    assert trainer.WORKLOAD_REVISION == "runpod-repository-repair-transactional-retention@1"
    assert (
        trainer.OBJECTIVE_ID == "verified-repair-chain-transactional-retention-policy-gradient@18"
    )
    assert study.BASE_WORKLOAD_REVISION == trainer.WORKLOAD_REVISION
    assert study.BASE_OBJECTIVE_ID == trainer.OBJECTIVE_ID


def test_transaction_is_disabled_by_default_and_policy_updates_make_validation_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer, "ACTIVE_RETENTION_TRANSACTION_REVISION", None)
    assert trainer.retention_validation_due(1, policy_update_applied=True) is False
    assert trainer.retention_validation_due(5, policy_update_applied=False) is True

    monkeypatch.setattr(
        trainer,
        "ACTIVE_RETENTION_TRANSACTION_REVISION",
        trainer.TRANSACTIONAL_RETENTION_REVISION,
    )
    assert trainer.retention_validation_due(1, policy_update_applied=True) is True
    assert trainer.retention_validation_due(1, policy_update_applied=False) is False


def test_transaction_restores_adapter_optimizer_and_clears_pending() -> None:
    adapter = {"weight": 1.0}
    optimizer = FakeOptimizer({"momentum": 2.0})
    transaction = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: dict(adapter),
        optimizer=optimizer,
        effective_policy_update_count=2,
        update=5,
        retained_observation={"exact_rate": 0.5},
    )
    adapter["weight"] = 9.0
    optimizer.state["momentum"] = 99.0

    (
        effective_count,
        pending_examples,
        pending_policy_examples,
        pending_group_ids,
        retained_observation,
    ) = trainer.restore_retention_transaction(
        transaction,
        restore_trainable_state=lambda state: adapter.update(state),
        optimizer=optimizer,
    )

    assert adapter == {"weight": 1.0}
    assert optimizer.state == {"momentum": 2.0}
    assert effective_count == 2
    assert pending_examples == []
    assert pending_policy_examples == 0
    assert pending_group_ids == []
    assert retained_observation == {"exact_rate": 0.5}


def test_tie_regression_and_strict_safe_improvement_have_exact_dispositions() -> None:
    no_change = {"improved": 0, "regressed": 0, "net_improved": 0}
    regression = {"improved": 1, "regressed": 1, "net_improved": 0}
    improvement = {"improved": 1, "regressed": 0, "net_improved": 1}

    assert (
        trainer.retention_transaction_disposition(
            candidate_is_best=False,
            fixed_change=no_change,
            rotating_change=no_change,
        )
        == "provisional"
    )
    assert (
        trainer.retention_transaction_disposition(
            candidate_is_best=False,
            fixed_change=regression,
            rotating_change=no_change,
        )
        == "rollback"
    )
    assert (
        trainer.retention_transaction_disposition(
            candidate_is_best=True,
            fixed_change=improvement,
            rotating_change=no_change,
        )
        == "retain"
    )
    assert trainer.paired_retention_guard_decision(
        best_fixed_net_improved=1,
        best_rotating_net_improved=1,
        fixed_change={"improved": 1, "regressed": 0, "net_improved": 1},
        rotating_change={"improved": 1, "regressed": 0, "net_improved": 1},
        consecutive_regressions=1,
    ) == (False, 1, 1, 0)
    with pytest.raises(RuntimeError, match="cannot be retained"):
        trainer.retention_transaction_disposition(
            candidate_is_best=True,
            fixed_change=regression,
            rotating_change=no_change,
        )


def test_regression_can_rollback_then_commit_a_safe_improvement() -> None:
    adapter = {"weight": 1.0}
    optimizer = FakeOptimizer()
    retained = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: dict(adapter),
        optimizer=optimizer,
        effective_policy_update_count=1,
        update=2,
        retained_observation={"exact_rate": 0.25},
    )
    adapter["weight"] = -1.0
    optimizer.state["momentum"] = -2.0
    effective_count, _, _, _, retained_observation = trainer.restore_retention_transaction(
        retained,
        restore_trainable_state=lambda state: adapter.update(state),
        optimizer=optimizer,
    )
    assert (adapter, optimizer.state, effective_count) == (
        {"weight": 1.0},
        {"momentum": 2.0},
        1,
    )
    assert retained_observation == {"exact_rate": 0.25}

    adapter["weight"] = 1.5
    optimizer.state["momentum"] = 2.5
    safely_retained = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: dict(adapter),
        optimizer=optimizer,
        effective_policy_update_count=2,
        update=4,
        retained_observation={"exact_rate": 0.75},
    )
    assert safely_retained == {
        "update": 4,
        "trainable_state": {"weight": 1.5},
        "optimizer_state": {"momentum": 2.5},
        "effective_policy_update_count": 2,
        "retained_observation": {"exact_rate": 0.75},
    }


def test_policy_lineage_counters_and_retained_state_survive_resume() -> None:
    resume_state = {
        "attempted_policy_update_count": 5,
        "effective_policy_update_count": 3,
        "retained_policy_update_count": 2,
        "retention_rollback_count": 1,
    }
    assert trainer.policy_lineage_counters_from_resume(
        resume_state,
        policy_update_count=5,
    ) == (5, 3, 2, 1)

    transaction = {
        "update": 7,
        "trainable_state": {"adapter": [1.0]},
        "optimizer_state": {"step": 4},
        "effective_policy_update_count": 2,
        "retained_observation": {"exact_rate": 0.5},
    }
    restored = trainer.validate_retained_transaction_state(
        transaction,
        best_validation_update=7,
        retained_policy_update_count=2,
    )
    transaction["optimizer_state"]["step"] = 99
    assert restored["optimizer_state"] == {"step": 4}

    with pytest.raises(RuntimeError, match="counters"):
        trainer.policy_lineage_counters_from_resume(
            {**resume_state, "effective_policy_update_count": 6},
            policy_update_count=5,
        )
    with pytest.raises(RuntimeError, match="lineage"):
        trainer.validate_retained_transaction_state(
            restored,
            best_validation_update=8,
            retained_policy_update_count=2,
        )


def test_branch_lineage_resolves_all_pending_snapshots_without_rewriting_history() -> None:
    snapshots = [
        {
            "update": 3,
            "optimizer_update": {
                "applied": True,
                "policy_signal_applied": True,
                "retention_lineage_status": "pending",
                "retention_transaction_disposition": "provisional",
            },
        },
        {
            "update": 4,
            "optimizer_update": {
                "applied": True,
                "policy_signal_applied": True,
                "retention_lineage_status": "pending",
                "retention_transaction_disposition": None,
            },
        },
        {
            "update": 1,
            "optimizer_update": {
                "applied": True,
                "policy_signal_applied": True,
                "retention_lineage_status": "retained",
                "retention_resolution_update": 1,
                "retention_resolution_reason": "retention_guard_improvement",
            },
        },
    ]

    trainer.annotate_retention_window_disposition(
        snapshots,
        update=4,
        disposition="retain",
    )
    assert (
        trainer.resolve_pending_branch_snapshot_lineage(
            snapshots,
            status="retained",
            resolution_update=4,
            reason="retention_guard_improvement",
        )
        == 2
    )

    assert snapshots[0]["optimizer_update"] == {
        "applied": True,
        "policy_signal_applied": True,
        "retention_lineage_status": "retained",
        "retention_transaction_disposition": "provisional",
        "retention_resolution_update": 4,
        "retention_resolution_reason": "retention_guard_improvement",
    }
    assert snapshots[1]["optimizer_update"]["retention_transaction_disposition"] == "retain"
    assert snapshots[1]["optimizer_update"]["retention_lineage_status"] == "retained"
    assert snapshots[2]["optimizer_update"]["retention_resolution_update"] == 1


def test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    (parameter.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    retained_weight = parameter.detach().clone()
    transaction = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: {"weight": parameter.detach().clone()},
        optimizer=optimizer,
        effective_policy_update_count=1,
        update=2,
        retained_observation={"exact_rate": 0.5},
    )

    (parameter.square().sum()).backward()
    optimizer.step()
    checkpoints_root = tmp_path / "checkpoints"
    latest = checkpoints_root / "latest.json"

    def save_adapter(target: str) -> None:
        Path(target, "adapter_config.json").write_text("{}", encoding="utf-8")
        Path(target, "adapter_model.safetensors").write_bytes(b"adapter")

    trainer.persist_checkpoint(
        checkpoints_root=str(checkpoints_root),
        latest_checkpoint_path=str(latest),
        checkpoint_name="update-0002",
        state={"retained_transaction": transaction},
        save_adapter=save_adapter,
        save_state=torch.save,
    )
    resumed_state = torch.load(
        checkpoints_root / "update-0002" / "training-state.pt",
        map_location="cpu",
        weights_only=False,
    )
    restored = trainer.validate_retained_transaction_state(
        resumed_state["retained_transaction"],
        best_validation_update=2,
        retained_policy_update_count=1,
    )
    resumed_parameter = torch.nn.Parameter(torch.tensor([-9.0]))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=0.01)
    effective_count, _, _, _, observation = trainer.restore_retention_transaction(
        restored,
        restore_trainable_state=lambda state: resumed_parameter.data.copy_(state["weight"]),
        optimizer=resumed_optimizer,
    )
    assert torch.equal(resumed_parameter.detach(), retained_weight)
    assert effective_count == 1
    assert observation == {"exact_rate": 0.5}

    (resumed_parameter.square().sum()).backward()
    resumed_optimizer.step()
    safely_retained = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: {"weight": resumed_parameter.detach().clone()},
        optimizer=resumed_optimizer,
        effective_policy_update_count=2,
        update=3,
        retained_observation={"exact_rate": 0.75},
    )
    assert safely_retained["effective_policy_update_count"] == 2
    assert safely_retained["retained_observation"] == {"exact_rate": 0.75}


def test_terminal_progress_preserves_rich_training_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    previous = {
        "phase": "training",
        "message": "Training.",
        "update": 7,
        "branch_snapshots": [{"task_id": "branch-1", "siblings": [{}, {}]}],
        "validation_history": [{"update": 7, "retention_guard_passed": True}],
        "curriculum_history": [{"update": 7, "to_level": 1}],
    }
    progress_path.write_text(json.dumps(previous), encoding="utf-8")
    monkeypatch.setattr(trainer, "PROGRESS_PATH", str(progress_path))

    trainer.emit_progress(
        "failed",
        "Repository repair workload failed.",
        runtime_configuration=None,
        preserve_context=True,
        attempt=1,
        remote_error={
            "code": "REMOTE_WORKLOAD_FAILURE",
            "message": "RuntimeError: CUDA allocation failed",
        },
        error="RuntimeError: CUDA allocation failed",
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["phase"] == "failed"
    assert progress["message"] == "Repository repair workload failed."
    assert progress["branch_snapshots"] == previous["branch_snapshots"]
    assert progress["validation_history"] == previous["validation_history"]
    assert progress["curriculum_history"] == previous["curriculum_history"]
    assert progress["remote_error"] == {
        "code": "REMOTE_WORKLOAD_FAILURE",
        "message": "RuntimeError: CUDA allocation failed",
    }


def test_workload_exception_has_a_structured_exact_remote_error() -> None:
    assert trainer.workload_remote_error(RuntimeError("CUDA allocation failed")) == {
        "code": "REMOTE_WORKLOAD_FAILURE",
        "message": "RuntimeError: CUDA allocation failed",
    }


def test_large_model_study_adapter_verifies_and_captures_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study.verify_transactional_sources()
    monkeypatch.setattr(
        study.trainer,
        "main",
        lambda: print(json.dumps({"experiment_completed": True})),
    )

    assert study.run_trainer_and_capture_result() == {"experiment_completed": True}
