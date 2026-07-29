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
        pending_optimizer_input_group_ids,
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
    assert pending_optimizer_input_group_ids == []
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
    effective_count, _, _, _, _, retained_observation = trainer.restore_retention_transaction(
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


def test_completion_tokens_count_every_generated_prefix_and_sibling_action() -> None:
    task = trainer.make_task(0, seed=trainer.DEFAULT_SEED)
    collection = trainer.BranchCollection(
        task=task,
        snapshot=None,
        prefix=trainer.RepositoryRepairEnvironment(task),
        siblings=[],
        generated_by_sibling=[
            [
                trainer.GeneratedAction(
                    response='{"tool":"list","path":""}',
                    completion_mask=(0, 1, 1),
                )
            ],
            [
                trainer.GeneratedAction(
                    response='{"tool":"read","path":"src/core.py"}',
                    completion_mask=(0, 1),
                )
            ],
        ],
        sampling_seeds=[],
        returns=[],
        advantages=[],
        exclusion_reason="TRAINING_DEADLINE_REACHED",
        replay=False,
        generated_prefix=[
            trainer.GeneratedAction(
                response='{"tool":"list","path":""}',
                completion_mask=(0, 1, 1, 1),
            )
        ],
    )

    assert trainer.sampled_completion_token_count([collection]) == 6
    assert trainer.discarded_collection_accounting([collection]) == (1, 0, 2, 6)
    with pytest.raises(RuntimeError, match="completion mask"):
        trainer.generated_action_completion_tokens(
            trainer.GeneratedAction(response="{}", completion_mask=(0, 2)),
        )


def test_completion_token_counters_resume_and_reconcile_discarded_evidence() -> None:
    assert trainer.completion_token_counters_from_resume(None) == (0, 0)
    assert trainer.completion_token_counters_from_resume(
        {
            "total_sampled_completion_tokens": 17,
            "discarded_sampled_completion_tokens": 5,
        }
    ) == (17, 5)
    with pytest.raises(RuntimeError, match="completion-token counters"):
        trainer.completion_token_counters_from_resume(
            {
                "total_sampled_completion_tokens": 4,
                "discarded_sampled_completion_tokens": 5,
            }
        )

    snapshots = [
        {
            "sampled_completion_tokens": 12,
            "shared_prefix": {"completion_tokens": 4},
            "siblings": [
                {"completion_tokens": 2},
                {"completion_tokens": 2},
                {"completion_tokens": 2},
                {"completion_tokens": 2},
            ],
        }
    ]
    assert (
        trainer.validate_sampled_completion_token_accounting(
            snapshots,
            total_sampled_completion_tokens=17,
            discarded_sampled_completion_tokens=5,
        )
        == 12
    )
    with pytest.raises(RuntimeError, match="do not reconcile"):
        trainer.validate_sampled_completion_token_accounting(
            snapshots,
            total_sampled_completion_tokens=16,
            discarded_sampled_completion_tokens=5,
        )


def test_cross_update_reference_anchors_have_exact_optimizer_input_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer,
        "ACTIVE_RETENTION_TRANSACTION_REVISION",
        trainer.TRANSACTIONAL_RETENTION_REVISION,
    )

    def collection(seed: int, *, exclusion_reason: str | None = None) -> trainer.BranchCollection:
        task = trainer.make_task(0, seed=seed)
        return trainer.BranchCollection(
            task=task,
            snapshot=None,
            prefix=trainer.RepositoryRepairEnvironment(task),
            siblings=[],
            generated_by_sibling=[],
            sampling_seeds=[],
            returns=[],
            advantages=[],
            exclusion_reason=exclusion_reason,
            replay=False,
        )

    prior_signal = collection(801)
    prior_reference_anchor = collection(802, exclusion_reason="EXCLUDED_REFERENCE_ANCHOR")
    prior_without_action = collection(803, exclusion_reason="NO_ACCEPTED_ACTION")
    current_signal = collection(804)
    current_reference_anchor = collection(805)
    current_without_action = collection(806, exclusion_reason="NO_ACCEPTED_ACTION")
    prior_signal.advantages = [1.0]
    current_signal.advantages = [1.0]
    actions_by_group_id = {
        group.task.task_id: trainer.GeneratedAction(
            response='{"tool":"read","path":"src/core.py"}',
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            completion_mask=(0, 1, 1),
        )
        for group in (
            prior_signal,
            prior_reference_anchor,
            current_signal,
            current_reference_anchor,
        )
    }
    signal_group_ids = {
        prior_signal.task.task_id,
        current_signal.task.task_id,
    }
    monkeypatch.setattr(
        trainer,
        "accepted_reference_actions",
        lambda group: (
            [actions_by_group_id[group.task.task_id]]
            if group.task.task_id in actions_by_group_id
            else []
        ),
    )
    monkeypatch.setattr(
        trainer,
        "policy_examples",
        lambda group: (
            [
                trainer.WeightedAction(
                    generated=actions_by_group_id[group.task.task_id],
                    weight=1.0,
                )
            ]
            if group.task.task_id in signal_group_ids
            else []
        ),
    )

    (
        first_training_examples,
        first_policy_example_count,
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        first_policy_signal_group_ids,
        pending_optimizer_input_group_ids,
        first_optimizer_input_group_ids,
    ) = trainer.accumulated_reference_anchored_examples(
        [prior_signal, prior_reference_anchor, prior_without_action],
        [],
        0,
        [],
        [],
        minimum_informative_groups=2,
    )
    assert first_training_examples == []
    assert first_policy_example_count == 0
    assert first_policy_signal_group_ids == []
    assert first_optimizer_input_group_ids == []
    assert pending_policy_example_count == 1
    assert pending_informative_group_ids == [prior_signal.task.task_id]
    assert pending_optimizer_input_group_ids == [
        prior_signal.task.task_id,
        prior_reference_anchor.task.task_id,
    ]
    assert len(pending_training_examples) == 2
    assert trainer.pending_optimizer_batch_from_resume(
        {
            "pending_training_examples": pending_training_examples,
            "pending_policy_example_count": pending_policy_example_count,
            "pending_informative_group_ids": pending_informative_group_ids,
            "pending_optimizer_input_group_ids": pending_optimizer_input_group_ids,
        },
        minimum_informative_groups=2,
    ) == (
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        pending_optimizer_input_group_ids,
    )
    with pytest.raises(RuntimeError, match="pending optimizer batch"):
        trainer.pending_optimizer_batch_from_resume(
            {
                "pending_training_examples": pending_training_examples,
                "pending_policy_example_count": pending_policy_example_count,
                "pending_informative_group_ids": pending_informative_group_ids,
                "pending_optimizer_input_group_ids": [prior_signal.task.task_id],
            },
            minimum_informative_groups=2,
        )

    (
        consumed_training_examples,
        consumed_policy_example_count,
        next_pending_training_examples,
        next_pending_policy_example_count,
        next_pending_informative_group_ids,
        consumed_policy_signal_group_ids,
        next_pending_optimizer_input_group_ids,
        consumed_optimizer_input_group_ids,
    ) = trainer.accumulated_reference_anchored_examples(
        [current_signal, current_reference_anchor, current_without_action],
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        pending_optimizer_input_group_ids,
        minimum_informative_groups=2,
    )
    assert len(consumed_training_examples) == 4
    assert consumed_policy_example_count == 2
    assert next_pending_training_examples == []
    assert next_pending_policy_example_count == 0
    assert next_pending_informative_group_ids == []
    assert next_pending_optimizer_input_group_ids == []
    assert consumed_policy_signal_group_ids == [
        prior_signal.task.task_id,
        current_signal.task.task_id,
    ]
    assert consumed_optimizer_input_group_ids == [
        prior_signal.task.task_id,
        prior_reference_anchor.task.task_id,
        current_signal.task.task_id,
        current_reference_anchor.task.task_id,
    ]

    all_groups = [
        prior_signal,
        prior_reference_anchor,
        prior_without_action,
        current_signal,
        current_reference_anchor,
        current_without_action,
    ]
    snapshots = [
        branch_evidence_snapshot(
            index,
            update=1 if index < 3 else 2,
            task_id=group.task.task_id,
            optimizer_update=(
                {
                    "update": 1,
                    "applied": False,
                    "policy_signal_applied": False,
                    "attempted_policy_update_index": None,
                }
                if index < 3
                else None
            ),
        )
        for index, group in enumerate(all_groups)
    ]
    optimizer_evidence = transactional_optimizer_evidence(
        update=2,
        attempt=1,
        signal_group_ids=consumed_policy_signal_group_ids,
        optimizer_input_group_ids=consumed_optimizer_input_group_ids,
    )
    trainer.bind_optimizer_input_groups_to_update(
        snapshots,
        optimizer_input_group_ids=consumed_optimizer_input_group_ids,
        optimizer_update=optimizer_evidence,
    )
    consumed_ids_from_snapshots = [
        snapshot["task_id"]
        for snapshot in snapshots
        if isinstance(snapshot["optimizer_update"], dict)
        and snapshot["optimizer_update"].get("attempted_policy_update_index") == 1
    ]
    assert consumed_ids_from_snapshots == consumed_optimizer_input_group_ids
    assert snapshots[1]["optimizer_update"]["optimizer_input_consumed_by_update"] == 2
    assert snapshots[2]["optimizer_update"]["attempted_policy_update_index"] is None
    assert snapshots[5]["optimizer_update"] is None

    lineage = trainer.policy_update_lineage_from_branch_evidence(
        snapshots,
        attempted_policy_update_count=1,
        effective_policy_update_count=1,
        retained_policy_update_count=0,
    )
    assert lineage[0]["optimizer_input_group_ids"] == consumed_optimizer_input_group_ids
    assert lineage[0]["policy_signal_group_ids"] == consumed_policy_signal_group_ids

    overinclusive_snapshots = json.loads(json.dumps(snapshots))
    overinclusive_snapshots[2]["optimizer_update"] = {
        **optimizer_evidence,
        "optimizer_input_consumed_by_update": 2,
    }
    with pytest.raises(RuntimeError, match="exactly reconstructable"):
        trainer.policy_update_lineage_from_branch_evidence(
            overinclusive_snapshots,
            attempted_policy_update_count=1,
            effective_policy_update_count=1,
            retained_policy_update_count=0,
        )

    snapshots[1]["optimizer_update"] = {
        "update": 1,
        "applied": False,
        "policy_signal_applied": False,
        "attempted_policy_update_index": None,
    }
    with pytest.raises(RuntimeError, match="exactly reconstructable"):
        trainer.policy_update_lineage_from_branch_evidence(
            snapshots,
            attempted_policy_update_count=1,
            effective_policy_update_count=1,
            retained_policy_update_count=0,
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


def branch_evidence_snapshot(
    snapshot_index: int,
    *,
    update: int,
    task_id: str | None = None,
    replay: bool = False,
    optimizer_update: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "snapshot_id": f"update-{update}-snapshot-{snapshot_index}",
        "update": update,
        "task_id": task_id or f"group-{snapshot_index}",
        "replay": replay,
        "optimizer_update": optimizer_update,
        "shared_prefix": {"steps": [{"step_id": "step-0"}]},
        "siblings": [
            {
                "index": sibling_index,
                "steps": [{"step_id": f"sibling-{sibling_index}-1"}],
            }
            for sibling_index in range(4)
        ],
    }


def transactional_optimizer_evidence(
    *,
    update: int,
    attempt: int,
    signal_group_ids: list[str],
    optimizer_input_group_ids: list[str] | None = None,
) -> dict[str, object]:
    input_group_ids = (
        list(signal_group_ids) if optimizer_input_group_ids is None else optimizer_input_group_ids
    )
    return {
        "update": update,
        "applied": True,
        "policy_signal_applied": True,
        "objective_id": trainer.OBJECTIVE_ID,
        "adapter_revision": f"update-{update}",
        "retention_transaction_revision": trainer.TRANSACTIONAL_RETENTION_REVISION,
        "retention_lineage_status": "pending",
        "retention_transaction_disposition": None,
        "attempted_policy_update_index": attempt,
        "effective_policy_update_count_after_apply": attempt,
        "retained_policy_update_count_before_validation": attempt - 1,
        "policy_signal_group_count": len(signal_group_ids),
        "policy_signal_group_ids": signal_group_ids,
        "optimizer_input_group_count": len(input_group_ids),
        "optimizer_input_group_ids": input_group_ids,
        "training_examples": 2,
        "reference_examples": 8,
        "policy_loss": 0.25,
        "reinforce_loss": 0.2,
        "reference_kl": 0.05,
        "gradient_norm": 0.5,
    }


def test_complete_branch_evidence_keeps_replay_and_nonrepresentative_groups_past_40() -> None:
    snapshots: list[dict[str, object]] = []
    first_window = [
        branch_evidence_snapshot(
            index,
            update=index // 4 + 1,
            replay=index % 4 == 3,
        )
        for index in range(44)
    ]

    trainer.append_complete_branch_evidence(snapshots, first_window)

    assert len(snapshots) == 44
    assert snapshots[0]["snapshot_id"] == "update-1-snapshot-0"
    assert snapshots[-1]["snapshot_id"] == "update-11-snapshot-43"
    assert sum(snapshot["replay"] is True for snapshot in snapshots) == 11
    with pytest.raises(RuntimeError, match="BRANCH_EVIDENCE_CAPACITY_EXCEEDED"):
        trainer.append_complete_branch_evidence(
            snapshots,
            [branch_evidence_snapshot(44, update=12)],
            maximum_snapshots=44,
        )
    assert len(snapshots) == 44
    with pytest.raises(RuntimeError, match="BRANCH_EVIDENCE_PAYLOAD_CAPACITY_EXCEEDED"):
        trainer.append_complete_branch_evidence(
            snapshots,
            [
                {
                    **branch_evidence_snapshot(44, update=12),
                    "bounded_payload_probe": "x" * 1_000,
                }
            ],
            maximum_payload_bytes=trainer.branch_evidence_payload_size_bytes(snapshots) + 10,
        )
    assert len(snapshots) == 44


def test_every_consumed_policy_group_has_full_reconstructable_branch_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer,
        "ACTIVE_RETENTION_TRANSACTION_REVISION",
        trainer.TRANSACTIONAL_RETENTION_REVISION,
    )
    pending_group = branch_evidence_snapshot(
        0,
        update=1,
        task_id="pending-signal",
        optimizer_update={
            "update": 1,
            "applied": False,
            "policy_signal_applied": False,
            "attempted_policy_update_index": None,
        },
    )
    nonrepresentative_group = branch_evidence_snapshot(
        1,
        update=2,
        task_id="current-signal",
        replay=True,
        optimizer_update=None,
    )
    anchor_group = branch_evidence_snapshot(
        2,
        update=2,
        task_id="reference-anchor",
        optimizer_update=None,
    )
    snapshots = [pending_group, nonrepresentative_group, anchor_group]
    optimizer_evidence = transactional_optimizer_evidence(
        update=2,
        attempt=1,
        signal_group_ids=["pending-signal", "current-signal"],
        optimizer_input_group_ids=[
            "pending-signal",
            "current-signal",
            "reference-anchor",
        ],
    )

    trainer.bind_optimizer_input_groups_to_update(
        snapshots,
        optimizer_input_group_ids=[
            "pending-signal",
            "current-signal",
            "reference-anchor",
        ],
        optimizer_update=optimizer_evidence,
    )
    pending_lineage = trainer.policy_update_lineage_from_branch_evidence(
        snapshots,
        attempted_policy_update_count=1,
        effective_policy_update_count=1,
        retained_policy_update_count=0,
    )

    assert pending_lineage[0]["schema_version"] == 2
    assert pending_lineage[0]["policy_signal_group_ids"] == [
        "pending-signal",
        "current-signal",
    ]
    assert pending_lineage[0]["optimizer_input_group_ids"] == [
        "pending-signal",
        "current-signal",
        "reference-anchor",
    ]
    assert snapshots[0]["shared_prefix"] == {"steps": [{"step_id": "step-0"}]}
    assert len(snapshots[0]["siblings"]) == 4
    assert snapshots[0]["optimizer_update"]["policy_signal_consumed_by_update"] == 2
    assert snapshots[2]["optimizer_update"]["optimizer_input_consumed_by_update"] == 2
    assert "policy_signal_consumed_by_update" not in snapshots[2]["optimizer_update"]

    assert (
        trainer.resolve_pending_branch_snapshot_lineage(
            snapshots,
            status="retained",
            resolution_update=2,
            reason="retention_guard_improvement",
            effective_policy_update_count=1,
            retained_policy_update_count=1,
            retention_rollback_count=0,
        )
        == 3
    )
    retained_lineage = trainer.policy_update_lineage_from_branch_evidence(
        snapshots,
        attempted_policy_update_count=1,
        effective_policy_update_count=1,
        retained_policy_update_count=1,
    )
    assert retained_lineage[0]["retention_lineage_status"] == "retained"
    assert retained_lineage[0]["effective_policy_update_count_after_resolution"] == 1


def test_policy_update_lineage_fails_closed_when_signal_snapshot_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer,
        "ACTIVE_RETENTION_TRANSACTION_REVISION",
        trainer.TRANSACTIONAL_RETENTION_REVISION,
    )
    evidence = transactional_optimizer_evidence(
        update=3,
        attempt=1,
        signal_group_ids=["missing-group", "present-group"],
    )
    evidence["optimizer_input_consumed_by_update"] = 3
    evidence["policy_signal_consumed_by_update"] = 3
    snapshots = [
        branch_evidence_snapshot(
            0,
            update=3,
            task_id="present-group",
            optimizer_update=evidence,
        )
    ]

    with pytest.raises(RuntimeError, match="exactly reconstructable"):
        trainer.policy_update_lineage_from_branch_evidence(
            snapshots,
            attempted_policy_update_count=1,
            effective_policy_update_count=1,
            retained_policy_update_count=0,
        )


def test_active_pilot_configuration_fits_bounded_branch_evidence() -> None:
    assert (
        trainer.maximum_branch_groups_for_configuration(
            maximum_updates=40,
            training_tasks_per_update=4,
            replay_tasks_per_level=1,
        )
        == 280
    )
    assert trainer.MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS >= 280
    task = trainer.make_task(3, seed=trainer.DEFAULT_SEED)
    diagnostics = trainer.diagnostic_actions(task)
    continuation = trainer.teacher_continuation_actions(task)
    calls = 0

    def scripted_sample(_: str, __: bool, ___: int) -> trainer.GeneratedAction:
        nonlocal calls
        if calls < len(diagnostics):
            action = diagnostics[calls]
        else:
            continuation_index = (calls - len(diagnostics)) // trainer.BRANCH_WIDTH
            action = continuation[min(continuation_index, len(continuation) - 1)]
        calls += 1
        return trainer.GeneratedAction(
            response=trainer.encode_action(action),
            completion_mask=(0, 1),
        )

    collection = trainer.collect_branch_group(
        task,
        scripted_sample,
        stochastic=False,
        sampling_seed=trainer.DEFAULT_SEED,
    )
    representative_snapshot = trainer.serialize_branch_group(
        collection,
        update=1,
        optimizer_update=None,
    )
    assert representative_snapshot["sampled_completion_tokens"] == trainer.sampled_action_count(
        [collection]
    )
    assert representative_snapshot["shared_prefix"]["completion_tokens"] == len(
        collection.generated_prefix
    )
    manifest_maximum_evidence = [
        {
            **representative_snapshot,
            "snapshot_id": f"update-{index // 7 + 1}-snapshot-{index}",
            "task_id": f"group-{index}",
        }
        for index in range(280)
    ]
    assert (
        trainer.branch_evidence_payload_size_bytes(manifest_maximum_evidence)
        < trainer.MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES
    )


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
    effective_count, _, _, _, _, observation = trainer.restore_retention_transaction(
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
        "total_sampled_completion_tokens": 412,
        "discarded_sampled_completion_tokens": 17,
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
    assert progress["total_sampled_completion_tokens"] == 412
    assert progress["discarded_sampled_completion_tokens"] == 17
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
