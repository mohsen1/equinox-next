"""Confirmatory conditions around the byte-frozen revision-30 trainer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import repository_repair_env as repair_environment
    import repository_repair_rl as frozen
except ModuleNotFoundError:
    from . import repository_repair_env as repair_environment
    from . import repository_repair_rl as frozen


WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@30"
OBJECTIVE_ID = "verified-fix-coverage-retention-policy-gradient@15"
STUDY_ID = "repository-repair-confirmatory-study@1"
FROZEN_SOURCE_COMMIT = "e6a139375a4e6ec92df362873237b398aa0041c0"
FROZEN_SOURCE_SHA256 = {
    "repository_repair_rl.py": ("449da958b75d41f5a641782980e0f0f8301122a68e011629faa4b7cc90bf7997"),
    "repository_repair_env.py": (
        "527050c5444a3731119773d4e0932840068fda9664654dce9bf0350edf56e40f"
    ),
}
CONDITIONS = frozenset({"k4_train", "k4_no_update", "k1_train"})
K1_ADVANTAGE_ESTIMATOR = "verified-success-positive-only@1"
K4_ADVANTAGE_ESTIMATOR = "leave-one-sibling-out-correctness-contrast@1"
FROZEN_VALIDATION_SEED_BASE = 40_000
FROZEN_TEST_SEED_BASE = 190_000


@dataclass(frozen=True)
class StudyConfiguration:
    condition: str
    optimization_seed: int
    validation_seed_base: int
    test_seed_base: int
    branch_width: int
    policy_mutation_enabled: bool
    advantage_estimator: str
    completion_budget: int | None
    training_tasks_per_update: int


@dataclass
class RuntimeEvidence:
    sampled_completions: int = 0
    optimizer_step_calls: int = 0
    restored_parameter_tensors: int = 0
    parameter_restore_verified: bool = True


def positive_environment_integer(name: str) -> int:
    raw_value = os.environ.get(name, "")
    if not raw_value.isdigit() or int(raw_value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw_value)


def study_configuration_from_environment() -> StudyConfiguration:
    condition = os.environ.get("EQUINOX_STUDY_CONDITION", "")
    if condition not in CONDITIONS:
        raise ValueError("EQUINOX_STUDY_CONDITION must be k4_train, k4_no_update, or k1_train")
    optimization_seed = positive_environment_integer("EQUINOX_RL_SEED")
    validation_seed_base = positive_environment_integer("EQUINOX_STUDY_VALIDATION_SEED_BASE")
    test_seed_base = positive_environment_integer("EQUINOX_STUDY_TEST_SEED_BASE")
    training_tasks_per_update = positive_environment_integer("EQUINOX_RL_TRAINING_TASKS_PER_UPDATE")
    if validation_seed_base == FROZEN_VALIDATION_SEED_BASE:
        raise ValueError("confirmatory validation tasks must be fresh")
    if test_seed_base == FROZEN_TEST_SEED_BASE:
        raise ValueError("confirmatory test tasks must be fresh")
    if validation_seed_base == test_seed_base:
        raise ValueError("validation and test seed bases must be disjoint")

    branch_width = 1 if condition == "k1_train" else 4
    expected_tasks_per_update = 12 if branch_width == 1 else 4
    if training_tasks_per_update != expected_tasks_per_update:
        raise ValueError(
            f"{condition} requires {expected_tasks_per_update} training tasks per update"
        )
    raw_completion_budget = os.environ.get("EQUINOX_STUDY_COMPLETION_BUDGET", "")
    completion_budget = (
        positive_environment_integer("EQUINOX_STUDY_COMPLETION_BUDGET")
        if raw_completion_budget
        else None
    )
    if condition != "k4_train" and completion_budget is None:
        raise ValueError(f"{condition} requires EQUINOX_STUDY_COMPLETION_BUDGET")
    if completion_budget is not None and completion_budget % branch_width:
        raise ValueError("the completion budget must contain whole branch groups")

    return StudyConfiguration(
        condition=condition,
        optimization_seed=optimization_seed,
        validation_seed_base=validation_seed_base,
        test_seed_base=test_seed_base,
        branch_width=branch_width,
        policy_mutation_enabled=condition != "k4_no_update",
        advantage_estimator=(
            K1_ADVANTAGE_ESTIMATOR if branch_width == 1 else K4_ADVANTAGE_ESTIMATOR
        ),
        completion_budget=completion_budget,
        training_tasks_per_update=training_tasks_per_update,
    )


def verify_frozen_sources(root: Path | None = None) -> None:
    source_root = root or Path(__file__).resolve().parent
    for filename, expected_digest in FROZEN_SOURCE_SHA256.items():
        observed_digest = hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise RuntimeError(f"{filename} does not match the frozen revision-30 source digest")
    if frozen.WORKLOAD_REVISION != WORKLOAD_REVISION:
        raise RuntimeError("the frozen workload revision changed")
    if frozen.OBJECTIVE_ID != OBJECTIVE_ID:
        raise RuntimeError("the frozen objective changed")


def k1_verified_success_advantages(returns: list[float]) -> list[float]:
    if len(returns) != 1:
        raise ValueError("the K=1 estimator requires exactly one return")
    if not math.isfinite(returns[0]):
        raise ValueError("the K=1 return must be finite")
    return [1.0 if returns[0] > 0 else 0.0]


def k1_frontier_probe_decision(
    current_level: int,
    prior_probe_level: int,
    collections: list[frozen.BranchCollection],
    *,
    maximum_level: int = frozen.MAXIMUM_COMPLEXITY_LEVEL,
) -> tuple[int, str]:
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    if current_level == maximum_level:
        if prior_probe_level != current_level:
            raise ValueError("the maximum curriculum level cannot have a harder probe")
        return current_level, "maximum_level_reached"
    minimum_probe_level = current_level + 1
    maximum_probe_level = min(
        maximum_level,
        current_level + frozen.MAXIMUM_FRONTIER_PROBE_OFFSET,
    )
    if not minimum_probe_level <= prior_probe_level <= maximum_probe_level:
        raise ValueError("frontier probe level is outside the adaptive probe range")
    probe_collections = [
        collection
        for collection in collections
        if collection.curriculum_role == "adjacent_complexity_probe"
        and collection.task.level == prior_probe_level
        and collection.exclusion_reason is None
    ]
    if not probe_collections:
        return prior_probe_level, "insufficient_probe_evidence"
    if all(collection.solved_siblings == 1 for collection in probe_collections):
        next_level = min(maximum_probe_level, prior_probe_level + 1)
        return (
            next_level,
            (
                "single_trajectory_solved_raise_probe"
                if next_level > prior_probe_level
                else "hardest_probe_solved"
            ),
        )
    if all(collection.solved_siblings == 0 for collection in probe_collections):
        next_level = max(minimum_probe_level, prior_probe_level - 1)
        return (
            next_level,
            (
                "single_trajectory_failed_lower_probe"
                if next_level < prior_probe_level
                else "nearest_probe_failed"
            ),
        )
    return prior_probe_level, "heterogeneous_single_trajectory_outcomes_hold_probe"


def load_runtime_evidence(path: Path) -> RuntimeEvidence:
    if not path.exists():
        return RuntimeEvidence()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeEvidence(
        sampled_completions=int(payload.get("sampled_completions", 0)),
        optimizer_step_calls=int(payload.get("optimizer_step_calls", 0)),
        restored_parameter_tensors=int(payload.get("restored_parameter_tensors", 0)),
        parameter_restore_verified=payload.get("parameter_restore_verified") is True,
    )


def persist_runtime_evidence(path: Path, evidence: RuntimeEvidence) -> None:
    pending_path = path.with_suffix(".pending")
    pending_path.write_text(
        json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pending_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(pending_path, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def exhausted_collection(task: Any, *, replay: bool, curriculum_role: str) -> Any:
    return frozen.BranchCollection(
        task=task,
        snapshot=None,
        prefix=repair_environment.RepositoryRepairEnvironment(task),
        siblings=[],
        generated_by_sibling=[],
        sampling_seeds=[],
        returns=[],
        advantages=[],
        exclusion_reason="STUDY_COMPLETION_BUDGET_REACHED",
        replay=replay,
        generated_prefix=[],
        curriculum_role=curriculum_role,
    )


def install_collection_budget(
    configuration: StudyConfiguration,
    evidence: RuntimeEvidence,
    evidence_path: Path,
) -> None:
    original_collect = frozen.collect_branch_group
    original_stop_decision = frozen.training_stop_decision

    def budgeted_collect(
        task: Any,
        sample_one: Any,
        *,
        stochastic: bool,
        sampling_seed: int,
        replay: bool = False,
        curriculum_role: str = "active_frontier",
        deadline_reached: Callable[[], bool] | None = None,
    ) -> Any:
        if (
            configuration.completion_budget is not None
            and evidence.sampled_completions >= configuration.completion_budget
        ):
            return exhausted_collection(
                task,
                replay=replay,
                curriculum_role=curriculum_role,
            )
        collection = original_collect(
            task,
            sample_one,
            stochastic=stochastic,
            sampling_seed=sampling_seed,
            replay=replay,
            curriculum_role=curriculum_role,
            deadline_reached=deadline_reached,
        )
        evidence.sampled_completions += len(collection.siblings)
        if (
            configuration.completion_budget is not None
            and evidence.sampled_completions > configuration.completion_budget
        ):
            raise RuntimeError("a branch group exceeded the study completion budget")
        persist_runtime_evidence(evidence_path, evidence)
        return collection

    def budgeted_stop_decision(**arguments: Any) -> tuple[bool, str | None, float]:
        stop, reason, deadline = original_stop_decision(**arguments)
        if (
            configuration.completion_budget is not None
            and evidence.sampled_completions >= configuration.completion_budget
        ):
            return True, "study_completion_budget", deadline
        return stop, reason, deadline

    frozen.collect_branch_group = budgeted_collect
    frozen.training_stop_decision = budgeted_stop_decision


def install_matched_compute_no_update(
    evidence: RuntimeEvidence,
    evidence_path: Path,
) -> None:
    import torch

    original_step = torch.optim.AdamW.step

    def restore_after_step(optimizer: Any, closure: Any = None) -> Any:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        snapshots = [parameter.detach().clone() for parameter in parameters]
        outcome = original_step(optimizer, closure)
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots, strict=True):
                parameter.copy_(snapshot)
        restored = all(
            torch.equal(parameter.detach(), snapshot)
            for parameter, snapshot in zip(parameters, snapshots, strict=True)
        )
        evidence.optimizer_step_calls += 1
        evidence.restored_parameter_tensors += len(parameters)
        evidence.parameter_restore_verified = evidence.parameter_restore_verified and restored
        persist_runtime_evidence(evidence_path, evidence)
        if not restored:
            raise RuntimeError("the frozen-policy control failed to restore adapter parameters")
        return outcome

    torch.optim.AdamW.step = restore_after_step


def install_study_condition(
    configuration: StudyConfiguration,
    evidence: RuntimeEvidence,
    evidence_path: Path,
) -> None:
    original_emit_progress = frozen.emit_progress

    def study_emit_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        original_emit_progress(
            phase,
            message,
            runtime_configuration,
            preserve_context=preserve_context,
            study_id=STUDY_ID,
            study_condition=configuration.condition,
            optimization_seed=configuration.optimization_seed,
            policy_mutation_enabled=configuration.policy_mutation_enabled,
            completion_budget=configuration.completion_budget,
            **values,
        )

    frozen.emit_progress = study_emit_progress
    frozen.VALIDATION_SEED_BASE = configuration.validation_seed_base
    frozen.TEST_SEED_BASE = configuration.test_seed_base
    frozen.BRANCH_WIDTH = configuration.branch_width
    repair_environment.BRANCH_WIDTH = configuration.branch_width
    if configuration.branch_width == 1:
        frozen.sibling_advantages = k1_verified_success_advantages
        frozen.correctness_contrast_advantages = k1_verified_success_advantages
        frozen.adaptive_frontier_probe_decision = k1_frontier_probe_decision
    install_collection_budget(configuration, evidence, evidence_path)
    if not configuration.policy_mutation_enabled:
        install_matched_compute_no_update(evidence, evidence_path)


def paired_hypothesis_passed(result: dict[str, Any]) -> bool:
    paired_change = result.get("paired_test_change")
    return (
        isinstance(paired_change, dict)
        and result.get("retention_passed") is True
        and isinstance(result.get("reward_gain"), int | float)
        and not isinstance(result["reward_gain"], bool)
        and result["reward_gain"] > 0
        and paired_change.get("regressed") == 0
        and isinstance(paired_change.get("mcnemar_exact_p_value"), int | float)
        and paired_change["mcnemar_exact_p_value"] < 0.05
        and result.get("final_evaluation_complete") is True
    )


def augment_result(
    result: dict[str, Any],
    configuration: StudyConfiguration,
    evidence: RuntimeEvidence,
) -> dict[str, Any]:
    completed_groups = max(
        0,
        int(result.get("total_task_groups", 0)) - int(result.get("excluded_task_groups", 0)),
    )
    derived_completions = completed_groups * configuration.branch_width
    sampled_completions = max(evidence.sampled_completions, derived_completions)
    if (
        configuration.completion_budget is not None
        and sampled_completions != configuration.completion_budget
    ):
        raise RuntimeError("the completed run did not consume the declared continuation budget")

    raw_policy_updates = int(result.get("policy_update_count", 0))
    raw_optimizer_updates = int(result.get("optimizer_update_count", 0))
    effective_policy_updates = raw_policy_updates if configuration.policy_mutation_enabled else 0
    effective_optimizer_updates = (
        raw_optimizer_updates if configuration.policy_mutation_enabled else 0
    )
    study = {
        "study_id": STUDY_ID,
        "condition": configuration.condition,
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "optimization_seed": configuration.optimization_seed,
        "validation_seed_base": configuration.validation_seed_base,
        "test_seed_base": configuration.test_seed_base,
        "branch_width": configuration.branch_width,
        "policy_mutation_enabled": configuration.policy_mutation_enabled,
        "advantage_estimator": configuration.advantage_estimator,
        "completion_budget": configuration.completion_budget,
        "sampled_completions": sampled_completions,
        "training_tasks_per_update": configuration.training_tasks_per_update,
        "attempted_policy_update_count": raw_policy_updates,
        "attempted_optimizer_update_count": raw_optimizer_updates,
        "effective_policy_update_count": effective_policy_updates,
        "effective_optimizer_update_count": effective_optimizer_updates,
        "optimizer_step_calls": evidence.optimizer_step_calls,
        "restored_parameter_tensors": evidence.restored_parameter_tensors,
        "parameter_restore_verified": evidence.parameter_restore_verified,
    }
    result = {
        **result,
        "total_sampled_completions": sampled_completions,
        "study": study,
        "training_configuration": {
            **result.get("training_configuration", {}),
            "study_id": STUDY_ID,
            "study_condition": configuration.condition,
            "validation_seed_base": configuration.validation_seed_base,
            "test_seed_base": configuration.test_seed_base,
            "policy_mutation_enabled": configuration.policy_mutation_enabled,
            "advantage_estimator": configuration.advantage_estimator,
            "completion_budget": configuration.completion_budget,
        },
        "effective_policy_update_count": effective_policy_updates,
        "effective_optimizer_update_count": effective_optimizer_updates,
    }
    if configuration.condition == "k4_no_update":
        result.update(
            hypothesis_passed=False,
            probative_post_training=False,
            claim_strength="CONTROL_NO_POLICY_MUTATION",
        )
    elif configuration.condition == "k1_train":
        frozen_hypothesis_passed = result.get("hypothesis_passed")
        single_trajectory_observed = any(
            len(snapshot.get("siblings", [])) == 1
            for snapshot in result.get("branch_snapshots", [])
            if isinstance(snapshot, dict)
        )
        result.update(
            algorithm="verified-fix-positive-only-policy-gradient-k1",
            frozen_hypothesis_passed=frozen_hypothesis_passed,
            hypothesis_passed=(
                single_trajectory_observed
                and effective_policy_updates > 0
                and paired_hypothesis_passed(result)
            ),
            probative_post_training=(single_trajectory_observed and effective_policy_updates > 0),
            restored_branching_observed=False,
            single_trajectory_observed=single_trajectory_observed,
            claim_strength="EXPLORATORY_CAUSAL_ABLATION_SINGLE_SEED",
        )
    return result


def run_frozen_and_capture_result() -> dict[str, Any]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        frozen.main()
    lines = [line for line in captured.getvalue().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("the frozen trainer did not emit a result")
    for diagnostic in lines[:-1]:
        print(diagnostic, file=sys.stderr)
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("the frozen trainer result was not an object")
    return result


def main() -> None:
    configuration = study_configuration_from_environment()
    verify_frozen_sources()
    evidence_path = (
        Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp")) / "study-runtime-evidence.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = load_runtime_evidence(evidence_path)
    if "--validate-configuration" in sys.argv:
        frozen.main()
        return
    if "--self-test" in sys.argv:
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "study_id": STUDY_ID,
                    "condition": configuration.condition,
                    "configuration": asdict(configuration),
                },
                sort_keys=True,
            )
        )
        return
    install_study_condition(configuration, evidence, evidence_path)
    result = run_frozen_and_capture_result()
    print(json.dumps(augment_result(result, configuration, evidence), sort_keys=True))


if __name__ == "__main__":
    main()
