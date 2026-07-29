from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import pytest

from research.runpod import repository_repair_independent_prefix_grpo as comparison
from research.runpod import repository_repair_large_model_pilot as pilot
from research.runpod import repository_repair_large_model_trainer as trainer


def generated_action(response: str, token: int) -> trainer.GeneratedAction:
    return trainer.GeneratedAction(
        response=response,
        input_ids=(1, token),
        attention_mask=(1, 1),
        completion_mask=(0, 1),
    )


def scripted_independent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> trainer.BranchCollection:
    task = trainer.make_task(0, seed=91)
    fault = task.faults[0]
    sampling_seed = 8_000
    monkeypatch.setattr(
        trainer,
        "RepositoryRepairEnvironment",
        pilot.PilotRepositoryRepairEnvironment,
    )
    actions = {
        0: [
            {"tool": "list", "path": ""},
            {"tool": "read", "path": fault.path},
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            },
            {"tool": "test"},
        ],
        1: [
            '{"tool":"bogus"}',
            {"tool": "list", "path": ""},
            {"tool": "test"},
            {"tool": "finish"},
        ],
        2: [
            {"tool": "list", "path": ""},
            {"tool": "test"},
            {"tool": "finish"},
        ],
        3: [
            {"tool": "list", "path": ""},
            {"tool": "test"},
            {"tool": "finish"},
        ],
    }
    calls: list[tuple[str, bool, int]] = []

    def sample_one(prompt: str, stochastic: bool, seed: int) -> trainer.GeneratedAction:
        calls.append((prompt, stochastic, seed))
        sibling_index = (seed - sampling_seed) // 10_000 - 1
        action_index = (seed - sampling_seed) % 10_000
        action = actions[sibling_index][action_index]
        return generated_action(
            action if isinstance(action, str) else trainer.encode_action(action),
            seed,
        )

    collection = comparison.collect_independent_prefix_group(
        task,
        sample_one,
        stochastic=True,
        sampling_seed=sampling_seed,
    )
    collection._test_calls = calls  # type: ignore[attr-defined]
    return collection


def shared_branch_snapshots() -> list[dict]:
    snapshots = []
    for group_index in range(2):
        siblings = [
            {
                "index": sibling_index,
                "sampling_seed": 730_000_000 + group_index * 1_000 + sibling_index,
                "completion_tokens": 8,
                "terminal_reason": "solved" if sibling_index == 0 else "finished_with_failures",
                "return": 1.0 if sibling_index == 0 else 0.0,
                "steps": [
                    {"step_id": f"group-{group_index}-sibling-{sibling_index}-step-0"},
                    {"step_id": f"group-{group_index}-sibling-{sibling_index}-step-1"},
                ],
            }
            for sibling_index in range(4)
        ]
        snapshots.append(
            {
                "schema_version": 2,
                "snapshot_id": f"shared-snapshot-{group_index}",
                "update": 1,
                "collection_index": group_index + 1,
                "collection_count": 2,
                "task_id": f"task-{group_index}",
                "level": 0,
                "replay": False,
                "curriculum_role": "active_frontier",
                "excluded": False,
                "shared_prefix": {
                    "completion_tokens": 3,
                    "steps": [{"step_id": f"group-{group_index}-prefix-0"}],
                },
                "sampled_completion_tokens": 35,
                "siblings": siblings,
            }
        )
    return snapshots


def independent_branch_snapshots() -> list[dict]:
    snapshots = copy.deepcopy(shared_branch_snapshots())
    for group_index, snapshot in enumerate(snapshots):
        snapshot["snapshot_id"] = f"independent-snapshot-{group_index}"
        snapshot["checkpoint"] = None
        snapshot["shared_prefix"] = None
        snapshot["rollout_topology"] = {
            "revision": comparison.ROLLOUT_TOPOLOGY,
            "static_group_width": 4,
            "independent_model_generated_prefixes": True,
            "shared_model_generated_prefix": False,
            "sibling_group_relative_credit": True,
            "initial_state_matching": "same_task_initial_state",
        }
        snapshot["comparison_condition"] = comparison.comparison_condition()
        snapshot["initial_state"] = {
            "state_id": f"initial-{group_index}",
            "payload_digest": "sha256:" + str(group_index) * 64,
            "fidelity": "logical_restore",
            "role": "matched_task_initial_state_not_decision_checkpoint",
        }
        token_counts = (9, 9, 9, 8)
        for sibling_index, sibling in enumerate(snapshot["siblings"]):
            sibling["completion_tokens"] = token_counts[sibling_index]
            if sibling_index == 0:
                sibling["steps"].append(
                    {"step_id": (f"group-{group_index}-sibling-{sibling_index}-step-2")}
                )
    return snapshots


def matched_result() -> dict:
    return {
        "branch_width": 4,
        "static_branch_width": 4,
        "restored_continuations": True,
        "prefix_gradient": False,
        "total_task_groups": 2,
        "excluded_task_groups": 0,
        "discarded_task_groups": 0,
        "discarded_sampled_actions": 0,
        "discarded_post_branch_actions": 0,
        "discarded_sampled_completion_tokens": 0,
        "total_sampled_actions": 18,
        "total_post_branch_actions": 16,
        "total_sampled_completion_tokens": 70,
        "persisted_branch_completion_tokens": 70,
        "branch_snapshots": shared_branch_snapshots(),
        "branch_evidence_complete": True,
        "branch_evidence_group_count": 2,
        "target_runtime_seconds": 7_200,
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "revision",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "optimization_seed": 73,
        "resumed_from_checkpoint": False,
        "environment_revision": "repository-repair-simulator@9",
        "action_protocol_revision": "repository-repair-json-tools@8",
        "verifier_revision": "repository-repair-hidden-tests@4",
        "reward_contract": {"revision": "correctness-gated-efficiency@1"},
        "complexity_strategy": "adaptive",
        "replay_enabled": True,
        "test_examples": 12,
        "training_configuration": {
            "learning_rate": 1e-5,
            "reference_kl_coefficient": 1.0,
            "reference_kl_estimator": "k3_log_ratio_penalty",
            "reference_policy": "disabled_adapter_base",
            "sibling_sampling_temperature": 0.6,
            "sibling_sampling_top_p": 0.9,
            "maximum_updates": 40,
            "training_tasks_per_update": 4,
            "replay_tasks_per_level": 1,
            "validation_examples": 8,
            "evaluation_interval": 5,
            "mastery_windows": 2,
            "task_sampling": "cumulative_validation_failure_structural_analogues",
            "frontier_probe_routing": "mixed_correctness_branch_contrast_feedback",
            "training_level_allocation": "current_and_adaptive_probe_even_split",
            "maximum_final_evaluation_reserve_seconds": 2_700,
            "test_seed_base": 190_000,
        },
        "optimizer_contract": {
            "learning_rate": 1e-5,
            "maximum_gradient_norm": 1.0,
            "sequence_reduction": "mean_completion_token_log_probabilities",
            "policy_credit_scope": "shared",
        },
        "final_evaluation_complete": True,
        "retained_policy_update_count": 1,
    }


def matched_independent_result(shared: dict) -> dict:
    return comparison.augment_comparison_result(
        {
            **copy.deepcopy(shared),
            "restored_continuations": False,
            "restored_branching_observed": False,
            "prefix_gradient": True,
            "total_post_branch_actions": 18,
            "branch_snapshots": independent_branch_snapshots(),
        }
    )


def test_module_import_does_not_change_active_shared_prefix_profile() -> None:
    assert trainer.collect_branch_group is not comparison.collect_independent_prefix_group
    assert trainer.POLICY_CREDIT_SCOPE != comparison.POLICY_CREDIT_SCOPE
    assert comparison.STATIC_BRANCH_WIDTH == 4
    frozen_path = Path(__file__).parents[2] / "research/runpod/repository_repair_rl.py"
    assert (
        __import__("hashlib").sha256(frozen_path.read_bytes()).hexdigest()
        == "449da958b75d41f5a641782980e0f0f8301122a68e011629faa4b7cc90bf7997"
    )


def test_independent_prompt_preserves_interface_contract_and_changes_only_phase() -> None:
    environment = pilot.PilotRepositoryRepairEnvironment(trainer.make_task(0, seed=90))

    prompt = comparison.independent_rollout_prompt(environment)

    assert prompt.startswith("PHASE INSTRUCTION\nAct in one complete repository-repair")
    assert "another rollout" not in prompt
    environment_data = json.loads(
        prompt.split("<untrusted-environment-data>\n", 1)[1].splitlines()[0]
    )
    assert environment_data["phase"] == "full_trajectory"
    assert environment_data["transcript"] == []
    assert environment_data["interface_state"]["mechanical_action_space"]["list_paths"] == [""]
    assert prompt.endswith(pilot.interface.ACTION_REMINDER)


def test_collection_samples_four_complete_stochastic_prefixes_from_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = scripted_independent_group(monkeypatch)
    calls = collection._test_calls  # type: ignore[attr-defined]

    assert collection.snapshot is not None
    assert collection.prefix.steps == []
    assert collection.generated_prefix == []
    assert len(collection.siblings) == 4
    assert collection.solved_siblings == 1
    assert collection.informative is True
    assert len(collection.returns) == 4
    assert collection.advantages[0] > 0
    assert all(value < 0 for value in collection.advantages[1:])
    assert sum(collection.advantages) == pytest.approx(0.0, abs=1e-7)
    assert all(stochastic is True for _prompt, stochastic, _seed in calls)
    first_prompts = {
        (seed - 8_000) // 10_000 - 1: prompt
        for prompt, _stochastic, seed in calls
        if (seed - 8_000) % 10_000 == 0
    }
    assert set(first_prompts) == {0, 1, 2, 3}
    assert all('"phase":"full_trajectory"' in prompt for prompt in first_prompts.values())
    assert all(
        sibling.steps[0].action == {"tool": "list", "path": ""}
        for sibling in (collection.siblings[0], *collection.siblings[2:])
    )
    assert collection.siblings[1].steps[0].action is None


def test_policy_examples_apply_signed_trajectory_normalized_group_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = scripted_independent_group(monkeypatch)

    examples = comparison.independent_prefix_policy_examples(collection)
    weights_by_sibling: dict[int, list[float]] = defaultdict(list)
    generated_owner = {
        id(generated): sibling_index
        for sibling_index, actions in enumerate(collection.generated_by_sibling)
        for generated in actions
    }
    for example in examples:
        weights_by_sibling[generated_owner[id(example.generated)]].append(example.weight)

    assert len(examples) == sum(len(actions) for actions in collection.generated_by_sibling)
    for sibling_index, advantage in enumerate(collection.advantages):
        assert sum(weights_by_sibling[sibling_index]) == pytest.approx(advantage)
        assert len(set(weights_by_sibling[sibling_index])) == 1
    assert sum(example.weight for example in examples) == pytest.approx(0.0, abs=1e-7)
    assert comparison.independent_prefix_reference_actions(collection) == [
        generated for actions in collection.generated_by_sibling for generated in actions
    ]
    malformed = collection.generated_by_sibling[1][0]
    assert any(example.generated is malformed and example.weight < 0 for example in examples)


def test_deadline_excludes_partial_group_without_manufacturing_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer,
        "RepositoryRepairEnvironment",
        pilot.PilotRepositoryRepairEnvironment,
    )
    sampled = False

    def sample_one(*_args: object) -> trainer.GeneratedAction:
        nonlocal sampled
        sampled = True
        return generated_action('{"tool":"finish"}', 1)

    collection = comparison.collect_independent_prefix_group(
        trainer.make_task(0, seed=92),
        sample_one,
        stochastic=True,
        sampling_seed=9_000,
        deadline_reached=lambda: True,
    )

    assert sampled is False
    assert collection.exclusion_reason == "TRAINING_DEADLINE_REACHED"
    assert collection.returns == []
    assert collection.advantages == []
    assert collection.snapshot is not None
    assert comparison.independent_prefix_policy_examples(collection) == []


def test_serialization_exposes_independent_lanes_without_fake_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = scripted_independent_group(monkeypatch)

    serialized = comparison.serialize_independent_prefix_group(
        collection,
        update=3,
        optimizer_update={
            "failed_sibling_policy_weight": 0.0,
            "policy_credit_scope": "shared_prefix_only",
        },
        base_serializer=trainer.serialize_branch_group,
    )

    assert serialized["checkpoint"] is None
    assert serialized["shared_prefix"] is None
    assert serialized["initial_state"]["payload_digest"] == collection.snapshot.payload_digest
    assert (
        serialized["initial_state"]["role"] == "matched_task_initial_state_not_decision_checkpoint"
    )
    assert serialized["rollout_topology"] == {
        "revision": "independent-prefix-k4@1",
        "static_group_width": 4,
        "independent_model_generated_prefixes": True,
        "shared_model_generated_prefix": False,
        "sibling_group_relative_credit": True,
        "initial_state_matching": "same_task_initial_state",
    }
    assert serialized["comparison_condition"] == {
        "schema_version": 1,
        "condition_id": "independent_prefix_grpo_k4",
        "short_label": "Independent · K=4",
        "prefix_topology": "independent",
        "branch_width": 4,
        "group_credit": "sibling_relative",
        "shared_prefix": False,
    }
    assert (
        serialized["optimizer_update"]["failed_sibling_policy_weight"]
        == "signed_trajectory_advantage"
    )
    assert serialized["optimizer_update"]["policy_credit_scope"] == comparison.POLICY_CREDIT_SCOPE
    assert len(serialized["siblings"]) == 4
    for sibling_index, sibling in enumerate(serialized["siblings"]):
        assert sibling["trajectory_policy_weight"] == pytest.approx(
            collection.advantages[sibling_index]
        )
        assert all("policy_weight" in step for step in sibling["steps"])
        assert sum(step["policy_weight"] for step in sibling["steps"]) == pytest.approx(
            collection.advantages[sibling_index]
        )


def test_static_k4_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trainer, "BRANCH_WIDTH", 1)

    with pytest.raises(ValueError, match="static K=4"):
        comparison.collect_independent_prefix_group(
            trainer.make_task(0, seed=1),
            lambda *_args: generated_action('{"tool":"finish"}', 1),
            stochastic=True,
            sampling_seed=1,
        )


def test_install_is_explicit_and_replaces_only_process_local_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "WORKLOAD_REVISION",
        "OBJECTIVE_ID",
        "POLICY_CREDIT_SCOPE",
        "REFERENCE_ANCHOR_SCOPE",
        "collect_branch_group",
        "policy_examples",
        "accepted_reference_actions",
        "serialize_branch_group",
        "branch_evidence_completion_token_count",
        "bind_optimizer_input_groups_to_update",
        "emit_progress",
    ):
        monkeypatch.setattr(trainer, name, getattr(trainer, name))
    original_bind_optimizer_input_groups = trainer.bind_optimizer_input_groups_to_update
    monkeypatch.setattr(comparison.study, "verify_transactional_sources", lambda: None)

    comparison.install_independent_prefix_comparison()

    assert trainer.WORKLOAD_REVISION == comparison.WORKLOAD_REVISION
    assert trainer.OBJECTIVE_ID == comparison.OBJECTIVE_ID
    assert trainer.collect_branch_group is comparison.collect_independent_prefix_group
    assert trainer.policy_examples is comparison.independent_prefix_policy_examples
    assert trainer.accepted_reference_actions is comparison.independent_prefix_reference_actions
    assert trainer.POLICY_CREDIT_SCOPE == comparison.POLICY_CREDIT_SCOPE
    assert trainer.bind_optimizer_input_groups_to_update is not original_bind_optimizer_input_groups
    assert (
        trainer.branch_evidence_completion_token_count
        is not comparison.independent_prefix_branch_evidence_completion_token_count
    )


def test_installed_snapshots_survive_checkpoint_resume_and_final_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        "WORKLOAD_REVISION",
        "OBJECTIVE_ID",
        "POLICY_CREDIT_SCOPE",
        "REFERENCE_ANCHOR_SCOPE",
        "collect_branch_group",
        "policy_examples",
        "accepted_reference_actions",
        "serialize_branch_group",
        "branch_evidence_completion_token_count",
        "bind_optimizer_input_groups_to_update",
        "emit_progress",
    ):
        monkeypatch.setattr(trainer, name, getattr(trainer, name))
    monkeypatch.setattr(comparison.study, "verify_transactional_sources", lambda: None)
    comparison.install_independent_prefix_comparison()
    assert trainer.branch_evidence_completion_token_count(shared_branch_snapshots()) == 70
    malformed_shared = shared_branch_snapshots()
    malformed_shared[0]["shared_prefix"].pop("completion_tokens")
    with pytest.raises(
        RuntimeError,
        match="branch evidence completion-token accounting is invalid",
    ):
        trainer.branch_evidence_completion_token_count(malformed_shared)

    collection = scripted_independent_group(monkeypatch)
    snapshot = {
        **trainer.serialize_branch_group(collection, update=1),
        "collection_index": 1,
        "collection_count": 1,
    }
    branch_snapshots: list[dict] = []
    trainer.append_complete_branch_evidence(branch_snapshots, [snapshot])
    total_tokens = trainer.sampled_completion_token_count([collection])

    checkpoint_tokens = trainer.validate_sampled_completion_token_accounting(
        branch_snapshots,
        total_sampled_completion_tokens=total_tokens,
        discarded_sampled_completion_tokens=0,
    )
    assert snapshot["shared_prefix"] is None
    assert checkpoint_tokens == snapshot["sampled_completion_tokens"] == total_tokens

    checkpoint_state = {
        "branch_snapshots": branch_snapshots,
        "total_sampled_completion_tokens": total_tokens,
        "discarded_sampled_completion_tokens": 0,
        "persisted_branch_completion_tokens": checkpoint_tokens,
    }
    checkpoints_root = tmp_path / "checkpoints"
    latest = checkpoints_root / "latest.json"

    def save_adapter(target: str) -> None:
        directory = Path(target)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")

    def save_state(state: dict, target: str) -> None:
        Path(target).write_text(json.dumps(state), encoding="utf-8")

    trainer.persist_checkpoint(
        checkpoints_root=str(checkpoints_root),
        latest_checkpoint_path=str(latest),
        checkpoint_name="update-0001",
        state=checkpoint_state,
        save_adapter=save_adapter,
        save_state=save_state,
    )
    restored = json.loads(
        (checkpoints_root / "update-0001" / "training-state.pt").read_text(encoding="utf-8")
    )
    resumed_total, resumed_discarded = trainer.completion_token_counters_from_resume(restored)
    finalized_tokens = trainer.validate_sampled_completion_token_accounting(
        restored["branch_snapshots"],
        total_sampled_completion_tokens=resumed_total,
        discarded_sampled_completion_tokens=resumed_discarded,
    )
    assert finalized_tokens == restored["persisted_branch_completion_tokens"]


def test_result_augmentation_and_exact_matched_budget_admission() -> None:
    shared = matched_result()
    contract = comparison.build_matched_budget_contract(shared)
    independent = matched_independent_result(shared)

    evidence = comparison.validate_matched_budget_result(contract, independent)

    assert independent["rollout_topology"] == "independent-prefix-k4@1"
    assert independent["comparison_condition"]["short_label"] == "Independent · K=4"
    assert independent["shared_prefix"] is False
    assert independent["comparison_probative_post_training"] is True
    assert independent["hypothesis_applicable"] is False
    assert independent["hypothesis_passed"] is None
    assert independent["probative_post_training"] is True
    assert independent["claim_strength"] == "INDEPENDENT_PREFIX_POST_TRAINING_SINGLE_CONDITION"
    assert (
        independent["training_configuration"]["trajectory_weight_normalization"]
        == "sum_of_action_weights_equals_trajectory_advantage"
    )
    assert evidence["admissible"] is True
    assert evidence["comparison_condition"]["condition_id"] == "independent_prefix_grpo_k4"
    assert evidence["exact_realized_budget"]["total_sampled_actions"] == 18
    assert evidence["exact_realized_budget"]["terminal_completion_count"] == 8
    assert evidence["exact_realized_budget"]["total_sampled_completion_tokens"] == 70
    assert evidence["exact_realized_budget"]["task_seed_schedule_digest"].startswith("sha256:")


def test_budget_or_identity_mismatch_is_not_admissible() -> None:
    shared = matched_result()
    contract = comparison.build_matched_budget_contract(shared)
    independent = matched_independent_result(shared)

    action_mismatch = copy.deepcopy(independent)
    action_mismatch["branch_snapshots"][0]["siblings"][0]["steps"].pop()
    action_mismatch["total_sampled_actions"] = 17
    action_mismatch["total_post_branch_actions"] = 17
    with pytest.raises(ValueError, match="total_sampled_actions"):
        comparison.validate_matched_budget_result(
            contract,
            action_mismatch,
        )
    mismatched_training = {
        **independent["training_configuration"],
        "learning_rate": 2e-5,
    }
    with pytest.raises(ValueError, match="learning_rate"):
        comparison.validate_matched_budget_result(
            contract,
            {**independent, "training_configuration": mismatched_training},
        )


@pytest.mark.parametrize("tamper", ["checkpoint", "rollout_topology"])
def test_topology_tampering_is_not_admissible(tamper: str) -> None:
    shared = matched_result()
    contract = comparison.build_matched_budget_contract(shared)
    independent = matched_independent_result(shared)
    snapshot = independent["branch_snapshots"][0]
    if tamper == "checkpoint":
        snapshot["checkpoint"] = {
            "checkpoint_id": "fabricated-shared-decision-state",
        }
    else:
        snapshot.pop("rollout_topology")

    with pytest.raises(
        ValueError,
        match="invalid independent-prefix evidence",
    ):
        comparison.validate_matched_budget_result(contract, independent)


def test_missing_complete_branch_or_token_evidence_fails_closed() -> None:
    missing_tokens = matched_result()
    missing_tokens.pop("total_sampled_completion_tokens")
    with pytest.raises(ValueError, match="total_sampled_completion_tokens"):
        comparison.build_matched_budget_contract(missing_tokens)

    missing_snapshot = matched_result()
    missing_snapshot["branch_snapshots"].pop()
    with pytest.raises(ValueError, match="cover every task group"):
        comparison.build_matched_budget_contract(missing_snapshot)

    discarded = matched_result()
    discarded["discarded_task_groups"] = 1
    with pytest.raises(ValueError, match="discarded groups without branch evidence"):
        comparison.build_matched_budget_contract(discarded)


def test_completion_and_token_budget_mismatches_are_not_admissible() -> None:
    shared = matched_result()
    contract = comparison.build_matched_budget_contract(shared)

    completion_mismatch = matched_independent_result(shared)
    completion_mismatch["branch_snapshots"][1]["excluded"] = True
    completion_mismatch["branch_snapshots"][1]["siblings"][3]["terminal_reason"] = None
    completion_mismatch["excluded_task_groups"] = 1
    with pytest.raises(ValueError, match="invalid independent-prefix evidence"):
        comparison.validate_matched_budget_result(contract, completion_mismatch)

    token_mismatch = matched_independent_result(shared)
    token_mismatch["branch_snapshots"][1]["siblings"][3]["completion_tokens"] = 7
    token_mismatch["total_sampled_completion_tokens"] = 69
    with pytest.raises(ValueError, match="completion-token"):
        comparison.validate_matched_budget_result(contract, token_mismatch)


def test_shared_source_prefix_tokens_must_reconcile_exactly() -> None:
    missing_prefix_tokens = matched_result()
    missing_prefix_tokens["branch_snapshots"][0]["shared_prefix"].pop("completion_tokens")
    with pytest.raises(ValueError, match="shared-prefix completion_tokens"):
        comparison.build_matched_budget_contract(missing_prefix_tokens)

    prefix_mismatch = matched_result()
    prefix_mismatch["branch_snapshots"][0]["shared_prefix"]["completion_tokens"] = 2
    with pytest.raises(ValueError, match="prefix and sibling evidence"):
        comparison.build_matched_budget_contract(prefix_mismatch)

    aggregate_mismatch = matched_result()
    aggregate_mismatch["total_sampled_completion_tokens"] = 69
    with pytest.raises(ValueError, match="completion-token counter disagrees"):
        comparison.build_matched_budget_contract(aggregate_mismatch)


def test_dedicated_entrypoint_emits_validated_augmented_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "WORKLOAD_REVISION",
        "OBJECTIVE_ID",
        "POLICY_CREDIT_SCOPE",
        "REFERENCE_ANCHOR_SCOPE",
        "collect_branch_group",
        "policy_examples",
        "accepted_reference_actions",
        "serialize_branch_group",
        "branch_evidence_completion_token_count",
        "bind_optimizer_input_groups_to_update",
        "emit_progress",
    ):
        monkeypatch.setattr(trainer, name, getattr(trainer, name))
    monkeypatch.setattr(comparison.study, "verify_transactional_sources", lambda: None)
    generic_result = {
        **matched_result(),
        "restored_continuations": True,
        "restored_branching_observed": True,
        "shared_prefix": True,
        "total_post_branch_actions": 18,
        "branch_snapshots": independent_branch_snapshots(),
    }

    def captured_result() -> dict:
        assert trainer.collect_branch_group is comparison.collect_independent_prefix_group
        assert all(
            snapshot["shared_prefix"] is None for snapshot in generic_result["branch_snapshots"]
        )
        return copy.deepcopy(generic_result)

    monkeypatch.setattr(
        comparison.study,
        "run_trainer_and_capture_result",
        captured_result,
    )
    monkeypatch.setattr(comparison.sys, "argv", ["independent-prefix"])

    comparison.main()

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["workload"] == "repository-repair-independent-prefix-grpo"
    assert emitted["profile_id"] == comparison.COMPARISON_PROFILE_ID
    assert emitted["restored_continuations"] is False
    assert emitted["shared_prefix"] is False
    assert emitted["prefix_gradient"] is True
    assert emitted["rollout_topology"] == comparison.ROLLOUT_TOPOLOGY
    comparison.validate_independent_comparison_result(emitted)


def test_result_validation_reconciles_a_discarded_deadline_tail() -> None:
    result = matched_independent_result(matched_result())
    result.update(
        discarded_task_groups=1,
        discarded_sampled_actions=2,
        discarded_post_branch_actions=2,
        discarded_sampled_completion_tokens=5,
        total_sampled_actions=20,
        total_post_branch_actions=20,
        total_sampled_completion_tokens=75,
    )

    comparison.validate_independent_comparison_result(result)


def test_result_validation_rejects_an_incomplete_admitted_trajectory() -> None:
    result = matched_independent_result(matched_result())
    result["branch_snapshots"][0]["siblings"][2]["terminal_reason"] = None

    with pytest.raises(RuntimeError, match="malformed trajectory evidence"):
        comparison.validate_independent_comparison_result(result)


@pytest.mark.parametrize("schedule_field", ["task", "seed"])
def test_task_or_seed_schedule_mismatch_is_not_admissible(schedule_field: str) -> None:
    shared = matched_result()
    contract = comparison.build_matched_budget_contract(shared)
    independent = matched_independent_result(shared)
    if schedule_field == "task":
        independent["branch_snapshots"][0]["task_id"] = "different-task"
    else:
        independent["branch_snapshots"][0]["siblings"][0]["sampling_seed"] += 100

    with pytest.raises(ValueError, match="task/seed schedule"):
        comparison.validate_matched_budget_result(contract, independent)
