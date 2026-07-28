"""Preregistered revision-31 K-by-curriculum causal study."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import repository_repair_env as frozen_environment
    import repository_repair_env_v31 as interface
    import repository_repair_rl as frozen
    import repository_repair_study as revision30_study
except ModuleNotFoundError:
    from . import repository_repair_env as frozen_environment
    from . import repository_repair_env_v31 as interface
    from . import repository_repair_rl as frozen
    from . import repository_repair_study as revision30_study


WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@31"
OBJECTIVE_ID = "verified-fix-coverage-retention-policy-gradient@15"
STUDY_ID = "repository-repair-causal-factorial-study@2"
FROZEN_BASE_SOURCE_COMMIT = "e6a139375a4e6ec92df362873237b398aa0041c0"
FROZEN_INTERFACE_SHA256 = "703c5badc19513cf8a7766a1a1e63fa77dce49a2f0011d78d9f4b94b64132657"
COMPLETION_BUDGET = 320
CONDITIONS = frozenset(
    {
        "k1_adaptive",
        "k1_scheduled_dynamic",
        "k4_adaptive",
        "k4_scheduled_dynamic",
    }
)
CURRICULUM_POLICIES = frozenset({"adaptive", "scheduled_dynamic"})
DISABLED_OUTCOME_STOP_THRESHOLD = 10_000


@dataclass(frozen=True)
class StudyConfiguration:
    condition: str
    optimization_seed: int
    validation_seed_base: int
    test_seed_base: int
    branch_width: int
    curriculum_policy: str
    completion_budget: int
    training_tasks_per_update: int


def positive_environment_integer(name: str) -> int:
    raw_value = os.environ.get(name, "")
    if not raw_value.isdigit() or int(raw_value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw_value)


def study_configuration_from_environment() -> StudyConfiguration:
    condition = os.environ.get("EQUINOX_STUDY_CONDITION", "")
    if condition not in CONDITIONS:
        raise ValueError(
            "EQUINOX_STUDY_CONDITION must be one preregistered revision-31 factorial cell"
        )
    branch_width = 1 if condition.startswith("k1_") else 4
    curriculum_policy = condition.removeprefix(f"k{branch_width}_")
    if curriculum_policy not in CURRICULUM_POLICIES:
        raise ValueError("the study condition has an unknown curriculum policy")
    training_tasks_per_update = positive_environment_integer("EQUINOX_RL_TRAINING_TASKS_PER_UPDATE")
    expected_tasks_per_update = 16 if branch_width == 1 else 4
    if training_tasks_per_update != expected_tasks_per_update:
        raise ValueError(
            f"K={branch_width} requires {expected_tasks_per_update} training tasks per update"
        )
    completion_budget = positive_environment_integer("EQUINOX_STUDY_COMPLETION_BUDGET")
    if completion_budget != COMPLETION_BUDGET:
        raise ValueError(f"revision 31 requires exactly {COMPLETION_BUDGET} completions")
    if completion_budget % branch_width:
        raise ValueError("the completion budget must contain whole branch groups")
    validation_seed_base = positive_environment_integer("EQUINOX_STUDY_VALIDATION_SEED_BASE")
    test_seed_base = positive_environment_integer("EQUINOX_STUDY_TEST_SEED_BASE")
    if validation_seed_base == test_seed_base:
        raise ValueError("validation and test seed bases must be disjoint")
    return StudyConfiguration(
        condition=condition,
        optimization_seed=positive_environment_integer("EQUINOX_RL_SEED"),
        validation_seed_base=validation_seed_base,
        test_seed_base=test_seed_base,
        branch_width=branch_width,
        curriculum_policy=curriculum_policy,
        completion_budget=completion_budget,
        training_tasks_per_update=training_tasks_per_update,
    )


def verify_revision31_sources(root: Path | None = None) -> None:
    revision30_study.verify_frozen_sources(root)
    source_root = root or Path(__file__).resolve().parent
    observed_digest = hashlib.sha256(
        (source_root / "repository_repair_env_v31.py").read_bytes()
    ).hexdigest()
    if observed_digest != FROZEN_INTERFACE_SHA256:
        raise RuntimeError("repository_repair_env_v31.py does not match its frozen digest")


def scheduled_frontier_probe_decision(
    current_level: int,
    prior_probe_level: int,
    collections: list[frozen.BranchCollection],
    *,
    maximum_level: int = frozen.MAXIMUM_COMPLEXITY_LEVEL,
) -> tuple[int, str]:
    del collections
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    if current_level == maximum_level:
        if prior_probe_level != current_level:
            raise ValueError("the maximum curriculum level cannot have a harder probe")
        return current_level, "scheduled_maximum_level"
    return min(maximum_level, current_level + 1), "scheduled_nearest_harder_probe"


def scheduled_mastery_windows(
    current: int,
    *,
    candidate_retained: bool,
    candidate_mastered: bool,
) -> int:
    del candidate_retained, candidate_mastered
    if current < 0:
        raise ValueError("retained mastery windows cannot be negative")
    return current + 1


def install_fixed_budget_stop_policy() -> None:
    original_stop_decision = frozen.training_stop_decision

    def fixed_budget_stop_decision(**arguments: Any) -> tuple[bool, str | None, float]:
        return original_stop_decision(
            **{
                **arguments,
                "maximum_level_mastered": False,
            }
        )

    frozen.training_stop_decision = fixed_budget_stop_decision
    frozen.MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS = DISABLED_OUTCOME_STOP_THRESHOLD
    frozen.MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS = DISABLED_OUTCOME_STOP_THRESHOLD


def install_revision31_condition(
    configuration: StudyConfiguration,
    evidence: revision30_study.RuntimeEvidence,
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
            branch_width=configuration.branch_width,
            curriculum_policy=configuration.curriculum_policy,
            completion_budget=configuration.completion_budget,
            **values,
        )

    frozen.emit_progress = study_emit_progress
    frozen.WORKLOAD_REVISION = WORKLOAD_REVISION
    frozen.OBJECTIVE_ID = OBJECTIVE_ID
    frozen.SYSTEM_PROMPT = interface.SYSTEM_PROMPT
    frozen.ACTION_PROTOCOL_REVISION = interface.ACTION_PROTOCOL_REVISION
    frozen.ENVIRONMENT_REVISION = interface.ENVIRONMENT_REVISION
    frozen.RepositoryRepairEnvironment = interface.RepositoryRepairEnvironment
    frozen.VALIDATION_SEED_BASE = configuration.validation_seed_base
    frozen.TEST_SEED_BASE = configuration.test_seed_base
    frozen.BRANCH_WIDTH = configuration.branch_width
    frozen_environment.BRANCH_WIDTH = configuration.branch_width
    if configuration.branch_width == 1:
        frozen.sibling_advantages = revision30_study.k1_verified_success_advantages
        frozen.correctness_contrast_advantages = revision30_study.k1_verified_success_advantages
        frozen.adaptive_frontier_probe_decision = revision30_study.k1_frontier_probe_decision
    if configuration.curriculum_policy == "scheduled_dynamic":
        frozen.adaptive_frontier_probe_decision = scheduled_frontier_probe_decision
        frozen.next_retained_mastery_windows = scheduled_mastery_windows
    install_fixed_budget_stop_policy()
    revision30_study.install_collection_budget(configuration, evidence, evidence_path)


def augment_result(
    result: dict[str, Any],
    configuration: StudyConfiguration,
    evidence: revision30_study.RuntimeEvidence,
) -> dict[str, Any]:
    completed_groups = max(
        0,
        int(result.get("total_task_groups", 0)) - int(result.get("excluded_task_groups", 0)),
    )
    sampled_completions = max(
        evidence.sampled_completions,
        completed_groups * configuration.branch_width,
    )
    if sampled_completions != configuration.completion_budget:
        raise RuntimeError("the condition did not consume the preregistered completion budget")
    policy_updates = int(result.get("policy_update_count", 0))
    training_configuration = {
        **result.get("training_configuration", {}),
        "study_id": STUDY_ID,
        "study_condition": configuration.condition,
        "branch_width": configuration.branch_width,
        "curriculum_policy": configuration.curriculum_policy,
        "curriculum_schedule_outcome_blinded": (
            configuration.curriculum_policy == "scheduled_dynamic"
        ),
        "completion_budget": configuration.completion_budget,
        "outcome_triggered_early_stops_enabled": False,
        "tool_contract_change": "paths-from-observations-and-edit-recovery-state@1",
    }
    if configuration.curriculum_policy == "scheduled_dynamic":
        training_configuration.update(
            mastery_window_basis="scheduled_completed_validation_windows",
            frontier_probe_routing="scheduled_nearest_harder_probe",
            training_level_allocation="current_and_scheduled_probe_even_split",
        )
    study = {
        "study_id": STUDY_ID,
        "condition": configuration.condition,
        "frozen_base_source_commit": FROZEN_BASE_SOURCE_COMMIT,
        "frozen_base_source_sha256": revision30_study.FROZEN_SOURCE_SHA256,
        "frozen_interface_sha256": FROZEN_INTERFACE_SHA256,
        "optimization_seed": configuration.optimization_seed,
        "validation_seed_base": configuration.validation_seed_base,
        "test_seed_base": configuration.test_seed_base,
        "branch_width": configuration.branch_width,
        "static_branch_width": True,
        "curriculum_policy": configuration.curriculum_policy,
        "curriculum_schedule_outcome_blinded": (
            configuration.curriculum_policy == "scheduled_dynamic"
        ),
        "completion_budget": configuration.completion_budget,
        "sampled_completions": sampled_completions,
        "training_tasks_per_update": configuration.training_tasks_per_update,
        "policy_mutation_enabled": True,
        "effective_policy_update_count": policy_updates,
    }
    frozen_hypothesis_passed = result.get("hypothesis_passed")
    return {
        **result,
        "workload_revision": WORKLOAD_REVISION,
        "algorithm": (
            "verified-fix-positive-only-policy-gradient-k1"
            if configuration.branch_width == 1
            else "verified-fix-group-conditioned-policy-gradient"
        ),
        "complexity_strategy": configuration.curriculum_policy,
        "hypothesis_passed": (
            policy_updates > 0 and revision30_study.paired_hypothesis_passed(result)
        ),
        "frozen_workload_hypothesis_passed": frozen_hypothesis_passed,
        "probative_post_training": policy_updates > 0,
        "claim_strength": "PREREGISTERED_FACTORIAL_CONDITION",
        "total_sampled_completions": sampled_completions,
        "study": study,
        "training_configuration": training_configuration,
        "effective_policy_update_count": policy_updates,
        "effective_optimizer_update_count": int(result.get("optimizer_update_count", 0)),
    }


def main() -> None:
    configuration = study_configuration_from_environment()
    verify_revision31_sources()
    evidence_path = (
        Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp"))
        / "revision31-study-runtime-evidence.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = revision30_study.load_runtime_evidence(evidence_path)
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
    install_revision31_condition(configuration, evidence, evidence_path)
    result = revision30_study.run_frozen_and_capture_result()
    print(json.dumps(augment_result(result, configuration, evidence), sort_keys=True))


if __name__ == "__main__":
    main()
