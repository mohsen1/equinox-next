"""Opt-in independent-prefix K=4 comparison for repository-repair post-training.

The active larger-model profile remains shared-prefix branching. This module is a
separate comparison adapter: importing it has no effect, and no provider launcher
selects it. A later comparison profile may install the hooks only after freezing a
matched budget and the shared-prefix source result.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from typing import Any

try:
    import repository_repair_large_model_study as study
    import repository_repair_large_model_trainer as trainer
except ModuleNotFoundError:
    from . import repository_repair_large_model_study as study
    from . import repository_repair_large_model_trainer as trainer


COMPARISON_PROFILE_ID = "repository-repair-independent-prefix-grpo@1"
COMPARISON_CONDITION_ID = "independent_prefix_grpo_k4"
COMPARISON_CONDITION_LABEL = "Independent · K=4"
WORKLOAD_REVISION = "runpod-repository-repair-independent-prefix-grpo@1"
OBJECTIVE_ID = "independent-prefix-group-relative-policy-gradient@1"
ROLLOUT_TOPOLOGY = "independent-prefix-k4@1"
GROUP_ADVANTAGE_ESTIMATOR = "leave-one-out-standardized-terminal-return@1"
POLICY_CREDIT_SCOPE = "all_sampled_actions_with_trajectory_normalized_signed_advantage"
REFERENCE_ANCHOR_SCOPE = "all_sampled_actions_in_independent_trajectories"
MATCHED_BUDGET_CONTRACT = "shared-vs-independent-realized-budget@2"
STATIC_BRANCH_WIDTH = 4
INDEPENDENT_PHASE = "full_trajectory"
INDEPENDENT_PHASE_INSTRUCTION = (
    "Act in one complete repository-repair rollout. Diagnose from observed evidence, "
    "repair the repository, and run tests."
)
_PROMPT_HEADER = "PHASE INSTRUCTION\n"
_ENVIRONMENT_BOUNDARY = "\n\n<untrusted-environment-data>\n"
_ENVIRONMENT_END = "\n</untrusted-environment-data>"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_static_k4() -> None:
    if trainer.BRANCH_WIDTH != STATIC_BRANCH_WIDTH:
        raise ValueError("the independent-prefix comparison requires static K=4")


def comparison_condition() -> dict[str, Any]:
    """Return the compact proof and UI condition label contract."""

    return {
        "schema_version": 1,
        "condition_id": COMPARISON_CONDITION_ID,
        "short_label": COMPARISON_CONDITION_LABEL,
        "prefix_topology": "independent",
        "branch_width": STATIC_BRANCH_WIDTH,
        "group_credit": "sibling_relative",
        "shared_prefix": False,
    }


def _independent_rollout_topology() -> dict[str, Any]:
    return {
        "revision": ROLLOUT_TOPOLOGY,
        "static_group_width": STATIC_BRANCH_WIDTH,
        "independent_model_generated_prefixes": True,
        "shared_model_generated_prefix": False,
        "sibling_group_relative_credit": True,
        "initial_state_matching": "same_task_initial_state",
    }


def independent_rollout_prompt(environment: Any) -> str:
    """Turn the current continuation interface into a full-trajectory prompt.

    The environment still owns the action schema, observation boundary, recovery
    choices, and action reminder. Only the phase instruction and phase identity change.
    """

    prompt = environment.policy_prompt("continuation")
    if not prompt.startswith(_PROMPT_HEADER) or _ENVIRONMENT_BOUNDARY not in prompt:
        raise RuntimeError("the structured repository-repair prompt contract changed")
    _instruction, environment_and_suffix = prompt.split(_ENVIRONMENT_BOUNDARY, 1)
    if _ENVIRONMENT_END not in environment_and_suffix:
        raise RuntimeError("the untrusted environment boundary is incomplete")
    encoded_environment, suffix = environment_and_suffix.split(_ENVIRONMENT_END, 1)
    try:
        environment_data = json.loads(encoded_environment)
    except json.JSONDecodeError as error:
        raise RuntimeError("the structured repository-repair prompt payload is invalid") from error
    if not isinstance(environment_data, dict) or environment_data.get("phase") != "continuation":
        raise RuntimeError("the repository-repair continuation phase changed")
    environment_data["phase"] = INDEPENDENT_PHASE
    return (
        _PROMPT_HEADER
        + INDEPENDENT_PHASE_INSTRUCTION
        + _ENVIRONMENT_BOUNDARY
        + _canonical_json(environment_data)
        + _ENVIRONMENT_END
        + suffix
    )


def collect_independent_prefix_group(
    task: Any,
    sample_one: trainer.SampleOne,
    *,
    stochastic: bool,
    sampling_seed: int,
    replay: bool = False,
    curriculum_role: str = "active_frontier",
    deadline_reached: Callable[[], bool] | None = None,
) -> trainer.BranchCollection:
    """Sample four complete trajectories with no model-generated shared prefix."""

    _require_static_k4()
    initial = trainer.RepositoryRepairEnvironment(task)
    initial_snapshot = initial.capture_snapshot()
    siblings = [
        trainer.RepositoryRepairEnvironment.restore(task, initial_snapshot)
        for _ in range(STATIC_BRANCH_WIDTH)
    ]
    generated_by_sibling: list[list[trainer.GeneratedAction]] = [
        [] for _ in range(STATIC_BRANCH_WIDTH)
    ]
    sampling_seeds = [sampling_seed + 10_000 * (index + 1) for index in range(STATIC_BRANCH_WIDTH)]
    while not all(sibling.terminal for sibling in siblings):
        for sibling_index, sibling in enumerate(siblings):
            if sibling.terminal:
                continue
            if deadline_reached is not None and deadline_reached():
                return trainer.BranchCollection(
                    task=task,
                    snapshot=initial_snapshot,
                    prefix=initial,
                    siblings=siblings,
                    generated_by_sibling=generated_by_sibling,
                    sampling_seeds=sampling_seeds,
                    returns=[],
                    advantages=[],
                    exclusion_reason="TRAINING_DEADLINE_REACHED",
                    replay=replay,
                    generated_prefix=[],
                    curriculum_role=curriculum_role,
                )
            action_index = len(generated_by_sibling[sibling_index])
            generated = sample_one(
                independent_rollout_prompt(sibling),
                stochastic,
                sampling_seeds[sibling_index] + action_index,
            )
            generated_by_sibling[sibling_index].append(generated)
            sibling.step(generated.response)

    returns = [sibling.terminal_reward for sibling in siblings]
    return trainer.BranchCollection(
        task=task,
        snapshot=initial_snapshot,
        prefix=initial,
        siblings=siblings,
        generated_by_sibling=generated_by_sibling,
        sampling_seeds=sampling_seeds,
        returns=returns,
        advantages=trainer.sibling_advantages(returns),
        exclusion_reason=None,
        replay=replay,
        generated_prefix=[],
        curriculum_role=curriculum_role,
    )


def _trainable_actions(actions: list[trainer.GeneratedAction]) -> list[trainer.GeneratedAction]:
    return [generated for generated in actions if generated.input_ids]


def independent_prefix_policy_examples(
    collection: trainer.BranchCollection,
) -> list[trainer.WeightedAction]:
    """Credit every sampled action with its trajectory's signed group advantage.

    Per-action weights sum to the trajectory advantage, so longer trajectories do not
    receive more total policy weight merely because they contain more action turns.
    Rejected and malformed actions remain part of the sampled trajectory and therefore
    receive the same signed outcome credit as accepted actions.
    """

    if collection.exclusion_reason or not collection.informative:
        return []
    if len(collection.advantages) != STATIC_BRANCH_WIDTH:
        raise ValueError("the independent-prefix group has invalid advantages")
    examples: list[trainer.WeightedAction] = []
    for advantage, generated_actions in zip(
        collection.advantages,
        collection.generated_by_sibling,
        strict=True,
    ):
        trainable = _trainable_actions(generated_actions)
        if not trainable:
            continue
        per_action_weight = advantage / len(trainable)
        examples.extend(
            trainer.WeightedAction(generated=generated, weight=per_action_weight)
            for generated in trainable
        )
    if not math.isclose(
        sum(example.weight for example in examples),
        0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("independent-prefix group-relative policy weights are not centered")
    return examples


def independent_prefix_reference_actions(
    collection: trainer.BranchCollection,
) -> list[trainer.GeneratedAction]:
    """Anchor every sampled action to the disabled-adapter reference policy."""

    return [
        generated
        for generated_actions in collection.generated_by_sibling
        for generated in generated_actions
        if generated.input_ids
    ]


def _independent_optimizer_evidence(
    optimizer_update: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if optimizer_update is None:
        return None
    return {
        **optimizer_update,
        "failed_sibling_policy_weight": "signed_trajectory_advantage",
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
        "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
        "group_advantage_estimator": GROUP_ADVANTAGE_ESTIMATOR,
        "trajectory_weight_normalization": ("sum_of_action_weights_equals_trajectory_advantage"),
    }


def serialize_independent_prefix_group(
    collection: trainer.BranchCollection,
    *,
    update: int,
    optimizer_update: dict[str, Any] | None = None,
    base_serializer: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize four independent lanes without inventing a decision checkpoint."""

    serializer = base_serializer or trainer.serialize_branch_group
    if serializer is serialize_independent_prefix_group:
        raise RuntimeError("an independent-prefix serializer requires its base serializer")
    serialized = serializer(
        collection,
        update=update,
        optimizer_update=optimizer_update,
    )
    initial_snapshot = collection.snapshot
    serialized.update(
        {
            "checkpoint": None,
            "shared_prefix": None,
            "rollout_topology": _independent_rollout_topology(),
            "initial_state": (
                {
                    "state_id": initial_snapshot.snapshot_id,
                    "payload_digest": initial_snapshot.payload_digest,
                    "fidelity": initial_snapshot.fidelity,
                    "role": "matched_task_initial_state_not_decision_checkpoint",
                }
                if initial_snapshot is not None
                else None
            ),
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "group_advantage_estimator": GROUP_ADVANTAGE_ESTIMATOR,
            "comparison_condition": comparison_condition(),
        }
    )
    for sibling_index, sibling in enumerate(serialized["siblings"]):
        advantage = (
            collection.advantages[sibling_index]
            if sibling_index < len(collection.advantages)
            else None
        )
        generated_actions = collection.generated_by_sibling[sibling_index]
        trainable = _trainable_actions(generated_actions)
        per_action_weight = (
            advantage / len(trainable) if advantage is not None and trainable else 0.0
        )
        signal = bool(
            collection.exclusion_reason is None
            and advantage is not None
            and abs(advantage) > 1e-8
            and trainable
        )
        sibling.update(
            {
                "policy_signal": signal,
                "trajectory_policy_weight": (
                    round(advantage, 8) if advantage is not None else None
                ),
                "effective_batch_weight": (
                    round(abs(advantage), 8) if signal and advantage is not None else 0.0
                ),
            }
        )
        serialized_steps = sibling["steps"]
        if len(serialized_steps) != len(generated_actions):
            raise RuntimeError("serialized independent trajectory lost action lineage")
        for generated, step in zip(generated_actions, serialized_steps, strict=True):
            action_signal = signal and bool(generated.input_ids)
            step.update(
                {
                    "policy_signal": action_signal,
                    "policy_weight": round(per_action_weight, 8) if action_signal else 0.0,
                    "effective_batch_weight": (
                        round(abs(per_action_weight), 8) if action_signal else 0.0
                    ),
                }
            )
    serialized["optimizer_update"] = _independent_optimizer_evidence(
        serialized.get("optimizer_update")
    )
    return serialized


def independent_prefix_branch_evidence_completion_token_count(
    branch_snapshots: list[dict[str, Any]],
    *,
    shared_prefix_counter: Callable[[list[dict[str, Any]]], int],
) -> int:
    """Reconcile independent snapshots while preserving strict shared validation.

    The comparison replaces ``shared_prefix`` with ``None`` intentionally. Generic
    trainer checkpoint and finalization paths still call one completion-token helper,
    so the opt-in installation wraps that helper. Snapshots without the explicit
    independent topology marker are delegated unchanged to the shared-prefix counter.
    """

    total = 0
    for snapshot in branch_snapshots:
        rollout_topology = (
            snapshot.get("rollout_topology") if isinstance(snapshot, Mapping) else None
        )
        independent = isinstance(rollout_topology, Mapping) and (
            rollout_topology.get("revision") == ROLLOUT_TOPOLOGY
        )
        if not independent:
            total += shared_prefix_counter([snapshot])
            continue

        sampled_tokens = snapshot.get("sampled_completion_tokens")
        siblings = snapshot.get("siblings")
        initial_state = snapshot.get("initial_state")
        if (
            rollout_topology != _independent_rollout_topology()
            or snapshot.get("comparison_condition") != comparison_condition()
            or "checkpoint" not in snapshot
            or snapshot.get("checkpoint") is not None
            or "shared_prefix" not in snapshot
            or snapshot.get("shared_prefix") is not None
            or not isinstance(initial_state, Mapping)
            or initial_state.get("role") != "matched_task_initial_state_not_decision_checkpoint"
            or type(sampled_tokens) is not int
            or sampled_tokens < 0
            or not isinstance(siblings, list)
            or len(siblings) != STATIC_BRANCH_WIDTH
        ):
            raise RuntimeError("independent branch evidence completion-token accounting is invalid")
        reconstructed_tokens = 0
        for sibling in siblings:
            if (
                not isinstance(sibling, Mapping)
                or type(sibling.get("completion_tokens")) is not int
                or sibling["completion_tokens"] < 0
            ):
                raise RuntimeError(
                    "independent branch evidence sibling token accounting is invalid"
                )
            reconstructed_tokens += sibling["completion_tokens"]
        if reconstructed_tokens != sampled_tokens:
            raise RuntimeError("independent branch evidence completion-token total is inconsistent")
        total += sampled_tokens
    return total


def install_independent_prefix_comparison() -> None:
    """Install comparison hooks around the current byte-verified trainer.

    Callers must first install the shared profile whose model, environment, reward,
    retention, and hardware settings form the comparison source. This function changes
    process-local module attributes only.
    """

    study.verify_transactional_sources()
    _require_static_k4()
    if trainer.collect_branch_group is collect_independent_prefix_group:
        raise RuntimeError("the independent-prefix comparison is already installed")
    base_serializer = trainer.serialize_branch_group
    shared_prefix_completion_token_counter = trainer.branch_evidence_completion_token_count
    base_bind_optimizer_input_groups = trainer.bind_optimizer_input_groups_to_update
    original_emit_progress = trainer.emit_progress

    def comparison_serializer(
        collection: trainer.BranchCollection,
        *,
        update: int,
        optimizer_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return serialize_independent_prefix_group(
            collection,
            update=update,
            optimizer_update=optimizer_update,
            base_serializer=base_serializer,
        )

    def comparison_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        values.update(
            profile_id=COMPARISON_PROFILE_ID,
            comparison_condition=comparison_condition(),
            rollout_topology=ROLLOUT_TOPOLOGY,
            restored_continuations=False,
            shared_prefix=False,
            prefix_gradient=True,
            policy_credit_scope=POLICY_CREDIT_SCOPE,
            group_advantage_estimator=GROUP_ADVANTAGE_ESTIMATOR,
        )
        original_emit_progress(
            phase,
            message,
            runtime_configuration,
            preserve_context=preserve_context,
            **values,
        )

    def comparison_completion_token_counter(
        branch_snapshots: list[dict[str, Any]],
    ) -> int:
        return independent_prefix_branch_evidence_completion_token_count(
            branch_snapshots,
            shared_prefix_counter=shared_prefix_completion_token_counter,
        )

    def comparison_bind_optimizer_input_groups(
        branch_snapshots: list[dict[str, Any]],
        *,
        optimizer_input_group_ids: list[str],
        optimizer_update: dict[str, Any],
    ) -> None:
        base_bind_optimizer_input_groups(
            branch_snapshots,
            optimizer_input_group_ids=optimizer_input_group_ids,
            optimizer_update=_independent_optimizer_evidence(optimizer_update) or {},
        )

    trainer.WORKLOAD_REVISION = WORKLOAD_REVISION
    trainer.OBJECTIVE_ID = OBJECTIVE_ID
    trainer.POLICY_CREDIT_SCOPE = POLICY_CREDIT_SCOPE
    trainer.REFERENCE_ANCHOR_SCOPE = REFERENCE_ANCHOR_SCOPE
    trainer.collect_branch_group = collect_independent_prefix_group
    trainer.policy_examples = independent_prefix_policy_examples
    trainer.accepted_reference_actions = independent_prefix_reference_actions
    trainer.serialize_branch_group = comparison_serializer
    trainer.branch_evidence_completion_token_count = comparison_completion_token_counter
    trainer.bind_optimizer_input_groups_to_update = comparison_bind_optimizer_input_groups
    trainer.emit_progress = comparison_progress


def augment_comparison_result(result: dict[str, Any]) -> dict[str, Any]:
    """Replace shared-prefix-only labels with truthful comparison semantics."""

    if result.get("branch_width") != STATIC_BRANCH_WIDTH:
        raise ValueError("the independent-prefix result is not static K=4")
    training_configuration = result.get("training_configuration")
    if not isinstance(training_configuration, dict):
        raise ValueError("the independent-prefix result omitted training configuration")
    optimizer_contract = result.get("optimizer_contract")
    if not isinstance(optimizer_contract, dict):
        raise ValueError("the independent-prefix result omitted optimizer contract")
    retained_updates = result.get(
        "retained_policy_update_count",
        result.get("effective_policy_update_count"),
    )
    comparison_probative = bool(
        result.get("final_evaluation_complete") is True
        and isinstance(retained_updates, int)
        and not isinstance(retained_updates, bool)
        and retained_updates > 0
    )
    return {
        **result,
        "workload": "repository-repair-independent-prefix-grpo",
        "profile_id": COMPARISON_PROFILE_ID,
        "comparison_condition": comparison_condition(),
        "workload_revision": WORKLOAD_REVISION,
        "algorithm": "independent-prefix-single-pass-grpo",
        "objective_id": OBJECTIVE_ID,
        "rollout_topology": ROLLOUT_TOPOLOGY,
        "restored_continuations": False,
        "restored_branching_observed": False,
        "shared_prefix": False,
        "prefix_gradient": True,
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
        "group_advantage_estimator": GROUP_ADVANTAGE_ESTIMATOR,
        "hypothesis_applicable": False,
        "hypothesis_passed": None,
        "probative_post_training": comparison_probative,
        "claim_strength": (
            "INDEPENDENT_PREFIX_POST_TRAINING_SINGLE_CONDITION"
            if comparison_probative
            else "INDEPENDENT_PREFIX_POST_TRAINING_INCOMPLETE"
        ),
        "comparison_probative_post_training": comparison_probative,
        "shared_prefix_hypothesis_applicable": False,
        "total_independent_trajectory_actions": result.get("total_sampled_actions"),
        "training_configuration": {
            **training_configuration,
            "rollout_topology": ROLLOUT_TOPOLOGY,
            "shared_prefix_sampling": "not_applicable",
            "shared_prefix_checkpoint": "none_initial_task_state_only",
            "minimum_shared_prefix_actions": 0,
            "maximum_shared_prefix_actions": 0,
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
            "group_advantage_estimator": GROUP_ADVANTAGE_ESTIMATOR,
            "trajectory_weight_normalization": (
                "sum_of_action_weights_equals_trajectory_advantage"
            ),
            "grpo_update": "single_pass_sampling_policy_ratio_one",
        },
        "optimizer_contract": {
            **optimizer_contract,
            "failed_sibling_policy_weight": "signed_trajectory_advantage",
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
            "group_advantage_estimator": GROUP_ADVANTAGE_ESTIMATOR,
            "trajectory_weight_normalization": (
                "sum_of_action_weights_equals_trajectory_advantage"
            ),
            "grpo_update": "single_pass_sampling_policy_ratio_one",
        },
    }


def validate_independent_comparison_result(result: Mapping[str, Any]) -> None:
    """Fail closed before the dedicated entrypoint emits comparison evidence."""

    comparison_probative = result.get("comparison_probative_post_training")
    expected_claim_strength = (
        "INDEPENDENT_PREFIX_POST_TRAINING_SINGLE_CONDITION"
        if comparison_probative is True
        else "INDEPENDENT_PREFIX_POST_TRAINING_INCOMPLETE"
    )
    if (
        result.get("profile_id") != COMPARISON_PROFILE_ID
        or result.get("comparison_condition") != comparison_condition()
        or result.get("workload_revision") != WORKLOAD_REVISION
        or result.get("objective_id") != OBJECTIVE_ID
        or result.get("rollout_topology") != ROLLOUT_TOPOLOGY
        or result.get("branch_width") != STATIC_BRANCH_WIDTH
        or result.get("static_branch_width") != STATIC_BRANCH_WIDTH
        or result.get("restored_continuations") is not False
        or result.get("restored_branching_observed") is not False
        or result.get("shared_prefix") is not False
        or result.get("prefix_gradient") is not True
        or result.get("policy_credit_scope") != POLICY_CREDIT_SCOPE
        or "hypothesis_applicable" not in result
        or result.get("hypothesis_applicable") is not False
        or "hypothesis_passed" not in result
        or result.get("hypothesis_passed") is not None
        or not isinstance(comparison_probative, bool)
        or result.get("probative_post_training") is not comparison_probative
        or result.get("shared_prefix_hypothesis_applicable") is not False
        or result.get("claim_strength") != expected_claim_strength
    ):
        raise RuntimeError("the independent-prefix result identity is inconsistent")

    snapshots = result.get("branch_snapshots")
    total_task_groups = result.get("total_task_groups")
    branch_evidence_group_count = result.get("branch_evidence_group_count")
    if (
        not isinstance(snapshots, list)
        or result.get("branch_evidence_complete") is not True
        or type(total_task_groups) is not int
        or total_task_groups < 0
        or len(snapshots) != total_task_groups
        or type(branch_evidence_group_count) is not int
        or branch_evidence_group_count != len(snapshots)
    ):
        raise RuntimeError("the independent-prefix result omitted complete branch evidence")
    for snapshot in snapshots:
        initial_state = snapshot.get("initial_state") if isinstance(snapshot, Mapping) else None
        siblings = snapshot.get("siblings") if isinstance(snapshot, Mapping) else None
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("comparison_condition") != comparison_condition()
            or "shared_prefix" not in snapshot
            or snapshot.get("shared_prefix") is not None
            or "checkpoint" not in snapshot
            or snapshot.get("checkpoint") is not None
            or not isinstance(initial_state, Mapping)
            or initial_state.get("role") != "matched_task_initial_state_not_decision_checkpoint"
            or snapshot.get("rollout_topology") != _independent_rollout_topology()
            or snapshot.get("excluded") is not False
            or snapshot.get("exclusion_reason") is not None
            or not isinstance(siblings, list)
            or len(siblings) != STATIC_BRANCH_WIDTH
            or any(
                not isinstance(sibling, Mapping)
                or sibling.get("index") != index
                or not isinstance(sibling.get("steps"), list)
                or not isinstance(sibling.get("terminal_reason"), str)
                or not sibling["terminal_reason"]
                for index, sibling in enumerate(siblings)
            )
        ):
            raise RuntimeError(
                "the independent-prefix result contains malformed trajectory evidence"
            )

    total_completion_tokens = result.get("total_sampled_completion_tokens")
    discarded_completion_tokens = result.get("discarded_sampled_completion_tokens")
    if (
        type(total_completion_tokens) is not int
        or total_completion_tokens < 0
        or type(discarded_completion_tokens) is not int
        or discarded_completion_tokens < 0
    ):
        raise RuntimeError("the independent-prefix completion-token counters are invalid")
    persisted_tokens = independent_prefix_branch_evidence_completion_token_count(
        snapshots,
        shared_prefix_counter=trainer.branch_evidence_completion_token_count,
    )
    reported_persisted_tokens = result.get("persisted_branch_completion_tokens")
    if (
        persisted_tokens + discarded_completion_tokens != total_completion_tokens
        or type(reported_persisted_tokens) is not int
        or reported_persisted_tokens != persisted_tokens
    ):
        raise RuntimeError(
            "the independent-prefix persisted completion-token evidence is inconsistent"
        )

    sampled_actions = sum(
        len(sibling["steps"]) for snapshot in snapshots for sibling in snapshot["siblings"]
    )
    total_sampled_actions = result.get("total_sampled_actions")
    total_post_branch_actions = result.get("total_post_branch_actions")
    discarded_sampled_actions = result.get("discarded_sampled_actions")
    discarded_post_branch_actions = result.get("discarded_post_branch_actions")
    if (
        type(total_sampled_actions) is not int
        or total_sampled_actions < 0
        or type(total_post_branch_actions) is not int
        or total_post_branch_actions < 0
        or type(discarded_sampled_actions) is not int
        or discarded_sampled_actions < 0
        or type(discarded_post_branch_actions) is not int
        or discarded_post_branch_actions < 0
        or total_sampled_actions != sampled_actions + discarded_sampled_actions
        or total_post_branch_actions != sampled_actions + discarded_post_branch_actions
    ):
        raise RuntimeError("the independent-prefix action counters are inconsistent")


def main() -> None:
    """Run the opt-in comparison and emit only a validated augmented result."""

    install_independent_prefix_comparison()
    if "--validate-configuration" in sys.argv:
        trainer.main()
        return
    if "--self-test" in sys.argv:
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "profile_id": COMPARISON_PROFILE_ID,
                    "workload_revision": WORKLOAD_REVISION,
                    "comparison_condition": comparison_condition(),
                },
                sort_keys=True,
            )
        )
        return
    result = augment_comparison_result(study.run_trainer_and_capture_result())
    validate_independent_comparison_result(result)
    print(json.dumps(result, sort_keys=True))


def _matched_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    training = result.get("training_configuration")
    reward = result.get("reward_contract")
    optimizer = result.get("optimizer_contract")
    if (
        not isinstance(training, Mapping)
        or not isinstance(reward, Mapping)
        or not isinstance(optimizer, Mapping)
    ):
        raise ValueError(
            "matched comparison result omitted training, optimizer, or reward configuration"
        )
    seed = result.get("optimization_seed", result.get("seed"))
    identity = {
        "model_id": result.get("model_id"),
        "model_revision": result.get("model_revision"),
        "gpu_name": result.get("gpu_name"),
        "optimization_seed": seed,
        "resumed_from_checkpoint": result.get("resumed_from_checkpoint"),
        "environment_revision": result.get("environment_revision"),
        "action_protocol_revision": result.get("action_protocol_revision"),
        "verifier_revision": result.get("verifier_revision"),
        "reward_contract_digest": "sha256:"
        + hashlib.sha256(_canonical_json(dict(reward)).encode()).hexdigest(),
        "complexity_strategy": result.get("complexity_strategy"),
        "replay_enabled": result.get("replay_enabled"),
        "learning_rate": training.get("learning_rate"),
        "reference_kl_coefficient": training.get("reference_kl_coefficient"),
        "reference_kl_estimator": training.get("reference_kl_estimator"),
        "reference_policy": training.get("reference_policy"),
        "maximum_gradient_norm": optimizer.get("maximum_gradient_norm"),
        "sequence_reduction": optimizer.get("sequence_reduction"),
        "sibling_sampling_temperature": training.get("sibling_sampling_temperature"),
        "sibling_sampling_top_p": training.get("sibling_sampling_top_p"),
        "maximum_updates": training.get("maximum_updates"),
        "training_tasks_per_update": training.get("training_tasks_per_update"),
        "replay_tasks_per_level": training.get("replay_tasks_per_level"),
        "validation_examples": training.get("validation_examples"),
        "evaluation_interval": training.get("evaluation_interval"),
        "mastery_windows": training.get("mastery_windows"),
        "task_sampling": training.get("task_sampling"),
        "frontier_probe_routing": training.get("frontier_probe_routing"),
        "training_level_allocation": training.get("training_level_allocation"),
        "maximum_final_evaluation_reserve_seconds": training.get(
            "maximum_final_evaluation_reserve_seconds"
        ),
        "test_examples": result.get("test_examples"),
        "test_seed_base": training.get("test_seed_base"),
        "target_runtime_seconds": result.get("target_runtime_seconds"),
    }
    missing = sorted(key for key, value in identity.items() if value is None)
    if missing:
        raise ValueError("matched comparison identity is incomplete: " + ", ".join(missing))
    return identity


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _realized_budget(
    result: Mapping[str, Any],
    *,
    topology: str,
    role: str,
) -> dict[str, Any]:
    """Derive realized budgets from complete ordered branch evidence."""

    if topology not in {"shared_prefix", "independent_prefix"}:
        raise ValueError("unknown comparison topology")
    snapshots = result.get("branch_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError(f"the matched comparison {role} has no branch snapshots")
    total_task_groups = _nonnegative_integer(
        result.get("total_task_groups"),
        f"matched comparison {role} total_task_groups",
    )
    excluded_task_groups = _nonnegative_integer(
        result.get("excluded_task_groups"),
        f"matched comparison {role} excluded_task_groups",
    )
    discarded_task_groups = _nonnegative_integer(
        result.get("discarded_task_groups"),
        f"matched comparison {role} discarded_task_groups",
    )
    if discarded_task_groups:
        raise ValueError(
            f"the matched comparison {role} has discarded groups without branch evidence"
        )
    if len(snapshots) != total_task_groups:
        raise ValueError(
            f"the matched comparison {role} branch snapshots do not cover every task group"
        )

    schedule: list[dict[str, Any]] = []
    observed_order: list[tuple[int, int]] = []
    update_collections: dict[int, list[tuple[int, int]]] = {}
    observed_excluded = 0
    completed_task_groups = 0
    sampled_trajectory_count = 0
    terminal_completion_count = 0
    derived_sampled_actions = 0
    derived_post_prefix_actions = 0
    shared_prefix_completion_tokens = 0
    sibling_completion_tokens = 0
    persisted_completion_tokens = 0
    for ordinal, raw_snapshot in enumerate(snapshots, start=1):
        if not isinstance(raw_snapshot, Mapping):
            raise ValueError(f"the matched comparison {role} has a malformed branch snapshot")
        update = _nonnegative_integer(
            raw_snapshot.get("update"),
            f"matched comparison {role} branch update",
        )
        if update < 1:
            raise ValueError(f"the matched comparison {role} branch update must be positive")
        collection_index = _nonnegative_integer(
            raw_snapshot.get("collection_index"),
            f"matched comparison {role} collection_index",
        )
        collection_count = _nonnegative_integer(
            raw_snapshot.get("collection_count"),
            f"matched comparison {role} collection_count",
        )
        if collection_index < 1 or collection_count < collection_index:
            raise ValueError(f"the matched comparison {role} collection order is invalid")
        observed_order.append((update, collection_index))
        update_collections.setdefault(update, []).append((collection_index, collection_count))

        task_id = raw_snapshot.get("task_id")
        level = raw_snapshot.get("level")
        replay = raw_snapshot.get("replay")
        curriculum_role = raw_snapshot.get("curriculum_role")
        excluded = raw_snapshot.get("excluded")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(level, int)
            or isinstance(level, bool)
            or level < 0
            or not isinstance(replay, bool)
            or not isinstance(curriculum_role, str)
            or not curriculum_role
            or not isinstance(excluded, bool)
        ):
            raise ValueError(f"the matched comparison {role} task schedule is incomplete")
        observed_excluded += int(excluded)

        shared_prefix = raw_snapshot.get("shared_prefix")
        sampled_completion_tokens = _nonnegative_integer(
            raw_snapshot.get("sampled_completion_tokens"),
            f"matched comparison {role} snapshot sampled_completion_tokens",
        )
        if topology == "shared_prefix":
            if not isinstance(shared_prefix, Mapping) or not isinstance(
                shared_prefix.get("steps"), list
            ):
                raise ValueError(
                    f"the matched comparison {role} omitted shared-prefix action evidence"
                )
            prefix_action_count = len(shared_prefix["steps"])
            prefix_completion_tokens = _nonnegative_integer(
                shared_prefix.get("completion_tokens"),
                f"matched comparison {role} shared-prefix completion_tokens",
            )
        else:
            if shared_prefix is not None or not isinstance(
                raw_snapshot.get("initial_state"), Mapping
            ):
                raise ValueError(
                    f"the matched comparison {role} is not an independent-prefix snapshot"
                )
            prefix_action_count = 0
            prefix_completion_tokens = 0

        siblings = raw_snapshot.get("siblings")
        if not isinstance(siblings, list) or (
            len(siblings) != STATIC_BRANCH_WIDTH and not (excluded and not siblings)
        ):
            raise ValueError(f"the matched comparison {role} branch snapshot is not static K=4")
        sampling_seeds: list[int] = []
        terminal_siblings = 0
        sibling_action_count = 0
        snapshot_sibling_completion_tokens = 0
        for sibling_index, sibling in enumerate(siblings):
            if not isinstance(sibling, Mapping) or sibling.get("index") != sibling_index:
                raise ValueError(f"the matched comparison {role} sibling order is incomplete")
            sampling_seed = _nonnegative_integer(
                sibling.get("sampling_seed"),
                f"matched comparison {role} sibling sampling_seed",
            )
            completion_tokens = _nonnegative_integer(
                sibling.get("completion_tokens"),
                f"matched comparison {role} sibling completion_tokens",
            )
            steps = sibling.get("steps")
            if not isinstance(steps, list):
                raise ValueError(
                    f"the matched comparison {role} sibling action evidence is incomplete"
                )
            terminal_reason = sibling.get("terminal_reason")
            if terminal_reason is not None and (
                not isinstance(terminal_reason, str) or not terminal_reason
            ):
                raise ValueError(
                    f"the matched comparison {role} sibling terminal evidence is invalid"
                )
            sampling_seeds.append(sampling_seed)
            snapshot_sibling_completion_tokens += completion_tokens
            sibling_action_count += len(steps)
            terminal_siblings += int(terminal_reason is not None)
        reconstructed_completion_tokens = (
            prefix_completion_tokens + snapshot_sibling_completion_tokens
        )
        if reconstructed_completion_tokens != sampled_completion_tokens:
            raise ValueError(
                f"the matched comparison {role} snapshot completion-token "
                "counter disagrees with prefix and sibling evidence"
            )
        shared_prefix_completion_tokens += prefix_completion_tokens
        sibling_completion_tokens += snapshot_sibling_completion_tokens
        persisted_completion_tokens += sampled_completion_tokens
        if len(sampling_seeds) != len(set(sampling_seeds)):
            raise ValueError(f"the matched comparison {role} reused a sibling sampling seed")
        if not excluded and terminal_siblings != STATIC_BRANCH_WIDTH:
            raise ValueError(f"the matched comparison {role} has an incomplete admitted task group")
        completed_task_groups += int(not excluded and terminal_siblings == STATIC_BRANCH_WIDTH)
        sampled_trajectory_count += len(siblings)
        terminal_completion_count += terminal_siblings
        derived_sampled_actions += prefix_action_count + sibling_action_count
        derived_post_prefix_actions += sibling_action_count
        schedule.append(
            {
                "ordinal": ordinal,
                "update": update,
                "collection_index": collection_index,
                "collection_count": collection_count,
                "task_id": task_id,
                "level": level,
                "replay": replay,
                "curriculum_role": curriculum_role,
                "sibling_sampling_seeds": sampling_seeds,
            }
        )

    if observed_order != sorted(observed_order):
        raise ValueError(f"the matched comparison {role} branch schedule is out of order")
    for update, collections in update_collections.items():
        declared_counts = {count for _index, count in collections}
        indexes = sorted(index for index, _count in collections)
        if len(declared_counts) != 1 or indexes != list(range(1, next(iter(declared_counts)) + 1)):
            raise ValueError(
                f"the matched comparison {role} update {update} has incomplete collection order"
            )
    if observed_excluded != excluded_task_groups:
        raise ValueError(
            f"the matched comparison {role} excluded-group counter disagrees with evidence"
        )

    total_sampled_actions = _nonnegative_integer(
        result.get("total_sampled_actions"),
        f"matched comparison {role} total_sampled_actions",
    )
    total_post_branch_actions = _nonnegative_integer(
        result.get("total_post_branch_actions"),
        f"matched comparison {role} total_post_branch_actions",
    )
    if (
        total_sampled_actions != derived_sampled_actions
        or total_post_branch_actions != derived_post_prefix_actions
    ):
        raise ValueError(
            f"the matched comparison {role} action counters disagree with branch evidence"
        )
    total_completion_tokens = _nonnegative_integer(
        result.get("total_sampled_completion_tokens"),
        f"matched comparison {role} total_sampled_completion_tokens",
    )
    discarded_completion_tokens = _nonnegative_integer(
        result.get("discarded_sampled_completion_tokens"),
        f"matched comparison {role} discarded_sampled_completion_tokens",
    )
    if discarded_completion_tokens:
        raise ValueError(
            f"the matched comparison {role} has discarded completion tokens without branch evidence"
        )
    if total_completion_tokens != persisted_completion_tokens:
        raise ValueError(
            f"the matched comparison {role} completion-token counter disagrees with evidence"
        )
    schedule_digest = "sha256:" + hashlib.sha256(_canonical_json(schedule).encode()).hexdigest()
    return {
        "branch_snapshot_count": len(snapshots),
        "total_task_groups": total_task_groups,
        "excluded_task_groups": excluded_task_groups,
        "discarded_task_groups": discarded_task_groups,
        "completed_task_groups": completed_task_groups,
        "sampled_trajectory_count": sampled_trajectory_count,
        "terminal_completion_count": terminal_completion_count,
        "total_sampled_actions": total_sampled_actions,
        "total_sampled_completion_tokens": total_completion_tokens,
        "discarded_sampled_completion_tokens": discarded_completion_tokens,
        "task_seed_schedule_digest": schedule_digest,
        "topology_counters": {
            "total_post_prefix_actions": total_post_branch_actions,
            "shared_prefix_completion_tokens": shared_prefix_completion_tokens,
            "sibling_completion_tokens": sibling_completion_tokens,
            "persisted_completion_tokens": persisted_completion_tokens,
        },
    }


_EXACT_REALIZED_BUDGET_FIELDS = (
    "total_task_groups",
    "excluded_task_groups",
    "discarded_task_groups",
    "completed_task_groups",
    "sampled_trajectory_count",
    "terminal_completion_count",
    "total_sampled_actions",
    "total_sampled_completion_tokens",
    "discarded_sampled_completion_tokens",
    "task_seed_schedule_digest",
)


def build_matched_budget_contract(shared_result: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze exact realized budgets from a complete shared-prefix source run."""

    if (
        shared_result.get("branch_width") != STATIC_BRANCH_WIDTH
        or shared_result.get("static_branch_width") != STATIC_BRANCH_WIDTH
        or shared_result.get("restored_continuations") is not True
        or shared_result.get("prefix_gradient") is not False
    ):
        raise ValueError("the matched-budget source is not a shared-prefix K=4 result")
    target_seconds = shared_result.get("target_runtime_seconds")
    if (
        not isinstance(target_seconds, int)
        or isinstance(target_seconds, bool)
        or target_seconds < 1
    ):
        raise ValueError("the matched-budget source has no valid runtime ceiling")
    source_budget = _realized_budget(
        shared_result,
        topology="shared_prefix",
        role="source",
    )
    exact_budget = {name: source_budget[name] for name in _EXACT_REALIZED_BUDGET_FIELDS}
    canonical_source = _canonical_json(dict(shared_result)).encode()
    return {
        "schema_version": 2,
        "contract_id": MATCHED_BUDGET_CONTRACT,
        "source_result_digest": "sha256:" + hashlib.sha256(canonical_source).hexdigest(),
        "comparison_profile_id": COMPARISON_PROFILE_ID,
        "comparison_condition": comparison_condition(),
        "static_group_width": STATIC_BRANCH_WIDTH,
        "budget_basis": [
            "exact_task_groups",
            "exact_terminal_completions",
            "exact_total_sampled_actions",
            "exact_total_sampled_completion_tokens",
            "exact_task_and_sibling_seed_schedule",
        ],
        "exact_realized_budget": exact_budget,
        "source_topology_counters": source_budget["topology_counters"],
        "target_runtime_seconds": target_seconds,
        "matched_identity": _matched_identity(shared_result),
    }


def validate_matched_budget_result(
    contract: Mapping[str, Any],
    comparison_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return admissibility evidence or reject an unmatched comparison result."""

    if (
        contract.get("contract_id") != MATCHED_BUDGET_CONTRACT
        or contract.get("comparison_profile_id") != COMPARISON_PROFILE_ID
        or contract.get("comparison_condition") != comparison_condition()
        or contract.get("static_group_width") != STATIC_BRANCH_WIDTH
        or not isinstance(contract.get("source_result_digest"), str)
        or len(contract["source_result_digest"]) != 71
        or not contract["source_result_digest"].startswith("sha256:")
        or contract.get("schema_version") != 2
    ):
        raise ValueError("the independent-prefix matched-budget contract is invalid")
    expected_identity = contract.get("matched_identity")
    if not isinstance(expected_identity, Mapping):
        raise ValueError("the independent-prefix matched identity is invalid")
    observed_identity = _matched_identity(comparison_result)
    mismatched = {
        key: {
            "expected": expected,
            "observed": observed_identity.get(key),
        }
        for key, expected in expected_identity.items()
        if observed_identity.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "the independent-prefix comparison identity does not match: "
            + ", ".join(sorted(mismatched))
        )
    if (
        comparison_result.get("profile_id") != COMPARISON_PROFILE_ID
        or comparison_result.get("comparison_condition") != comparison_condition()
        or comparison_result.get("rollout_topology") != ROLLOUT_TOPOLOGY
        or comparison_result.get("branch_width") != STATIC_BRANCH_WIDTH
        or comparison_result.get("static_branch_width") != STATIC_BRANCH_WIDTH
        or comparison_result.get("restored_continuations") is not False
        or comparison_result.get("shared_prefix") is not False
        or comparison_result.get("prefix_gradient") is not True
    ):
        raise ValueError("the comparison result is not faithful independent-prefix K=4")
    try:
        validate_independent_comparison_result(comparison_result)
    except RuntimeError as error:
        raise ValueError(
            f"the comparison result has invalid independent-prefix evidence: {error}"
        ) from error
    exact_budget = contract.get("exact_realized_budget")
    if not isinstance(exact_budget, Mapping) or set(exact_budget) != set(
        _EXACT_REALIZED_BUDGET_FIELDS
    ):
        raise ValueError("the independent-prefix exact realized budget is invalid")
    comparison_budget = _realized_budget(
        comparison_result,
        topology="independent_prefix",
        role="result",
    )
    budget_mismatches = {
        name: {
            "expected": exact_budget[name],
            "observed": comparison_budget[name],
        }
        for name in _EXACT_REALIZED_BUDGET_FIELDS
        if comparison_budget[name] != exact_budget[name]
    }
    if budget_mismatches:
        if "task_seed_schedule_digest" in budget_mismatches:
            raise ValueError("the independent-prefix result did not match the task/seed schedule")
        raise ValueError(
            "the independent-prefix result did not match realized budgets: "
            + ", ".join(sorted(budget_mismatches))
        )
    if comparison_result.get("target_runtime_seconds") != contract.get("target_runtime_seconds"):
        raise ValueError("the independent-prefix result did not match the runtime ceiling")
    return {
        "schema_version": 2,
        "contract_id": MATCHED_BUDGET_CONTRACT,
        "source_result_digest": contract["source_result_digest"],
        "comparison_profile_id": COMPARISON_PROFILE_ID,
        "comparison_condition": comparison_condition(),
        "static_group_width": STATIC_BRANCH_WIDTH,
        "exact_realized_budget": dict(exact_budget),
        "comparison_topology_counters": comparison_budget["topology_counters"],
        "target_runtime_seconds": comparison_result["target_runtime_seconds"],
        "identity_matched": True,
        "budget_matched": True,
        "admissible": True,
    }


__all__ = [
    "COMPARISON_CONDITION_ID",
    "COMPARISON_CONDITION_LABEL",
    "COMPARISON_PROFILE_ID",
    "GROUP_ADVANTAGE_ESTIMATOR",
    "MATCHED_BUDGET_CONTRACT",
    "OBJECTIVE_ID",
    "POLICY_CREDIT_SCOPE",
    "ROLLOUT_TOPOLOGY",
    "STATIC_BRANCH_WIDTH",
    "WORKLOAD_REVISION",
    "augment_comparison_result",
    "build_matched_budget_contract",
    "collect_independent_prefix_group",
    "comparison_condition",
    "independent_prefix_branch_evidence_completion_token_count",
    "independent_prefix_policy_examples",
    "independent_prefix_reference_actions",
    "independent_rollout_prompt",
    "install_independent_prefix_comparison",
    "main",
    "serialize_independent_prefix_group",
    "validate_independent_comparison_result",
    "validate_matched_budget_result",
]


if __name__ == "__main__":
    main()
