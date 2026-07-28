"""Restored-continuation causal-credit post-training for repository repair.

This workload collects one policy-generated diagnostic prefix, snapshots the
repository and transcript, restores four continuations, and applies
reward-conditioned credit only to fault-fixing edit tokens from verified
successful siblings in mixed-outcome groups. Candidate content is verified as
inert simulator state and is never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from research.runpod.repository_repair_env import (
        ACTION_PROTOCOL_REVISION,
        BRANCH_WIDTH,
        COMPLEXITY_LEVELS,
        DIAGNOSTIC_TOOLS,
        ENVIRONMENT_REVISION,
        STRUCTURAL_MIRROR_DISCLOSURES,
        SYSTEM_PROMPT,
        TEMPLATE_SPLITS,
        VERIFIER_REVISION,
        EnvironmentSnapshot,
        RepairTask,
        RepositoryRepairEnvironment,
        diagnostic_actions,
        encode_action,
        make_task,
        make_tasks,
        semantic_task_universe_size,
        teacher_continuation_actions,
    )
except ModuleNotFoundError:
    from repository_repair_env import (  # type: ignore[no-redef]
        ACTION_PROTOCOL_REVISION,
        BRANCH_WIDTH,
        COMPLEXITY_LEVELS,
        DIAGNOSTIC_TOOLS,
        ENVIRONMENT_REVISION,
        STRUCTURAL_MIRROR_DISCLOSURES,
        SYSTEM_PROMPT,
        TEMPLATE_SPLITS,
        VERIFIER_REVISION,
        EnvironmentSnapshot,
        RepairTask,
        RepositoryRepairEnvironment,
        diagnostic_actions,
        encode_action,
        make_task,
        make_tasks,
        semantic_task_universe_size,
        teacher_continuation_actions,
    )

DEFAULT_SEED = 73
SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
}
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_VALIDATION_EXAMPLES = 8
DEFAULT_TEST_EXAMPLES = 6
DEFAULT_TRAINING_TASKS_PER_UPDATE = 4
DEFAULT_REPLAY_TASKS_PER_LEVEL = 1
DEFAULT_MAX_UPDATES = 120
DEFAULT_MASTERY_WINDOWS = 2
DEFAULT_TARGET_RUNTIME_SECONDS = 7_200
MAXIMUM_TARGET_RUNTIME_SECONDS = 21_600
DEFAULT_MAXIMUM_RESUME_GAP_SECONDS = 2_700
DEFAULT_MAX_FINAL_EVALUATION_RESERVE_SECONDS = 2_700
WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@22"
OBJECTIVE_ID = "verified-fix-accumulated-retention-policy-gradient@12"
DEPENDENCIES = (
    "transformers==5.14.1",
    "peft==0.19.1",
    "accelerate==1.14.0",
)
DEPENDENCY_VERSIONS = {
    requirement.split("==", 1)[0]: requirement.split("==", 1)[1] for requirement in DEPENDENCIES
}
MAXIMUM_COMPLEXITY_LEVEL = len(COMPLEXITY_LEVELS) - 1
PREFIX_ACCEPTED_ACTIONS = 2
PREFIX_MAX_ATTEMPTS = 5
EVALUATION_INTERVAL = 5
VALIDATION_SEED_BASE = 40_000
VALIDATION_WINDOW_SEED_STRIDE = 1_000_003
TEST_SEED_BASE = 90_000
MAX_INPUT_TOKENS = 4_096
MAX_NEW_TOKENS = 192
ACTION_RESPONSE_PREFIX = '{"tool":'
SIBLING_SAMPLING_TEMPERATURE = 0.6
SIBLING_SAMPLING_TOP_P = 0.9
LEARNING_RATE = 2e-5
REFERENCE_KL_COEFFICIENT = 0.2
REFERENCE_KL_ESTIMATOR = "k3_log_ratio_penalty"
REFERENCE_ANCHOR_SCOPE = "all_accepted_actions_including_greedy_prefix"
POLICY_CREDIT_SCOPE = "fault_fixing_edits_from_verified_successful_siblings"
ADVANTAGE_STANDARD_DEVIATION_FLOOR = 0.1
MAXIMUM_ABSOLUTE_ADVANTAGE = 1.0
MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE = 2
MAXIMUM_FRONTIER_PROBE_OFFSET = 2
TRAINING_MICROBATCH_SIZE = 2
MASTERY_THRESHOLD = 0.50
MINIMUM_PROTOCOL_VALIDITY_RATE = 0.99
MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS = 12
MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS = 2
MAXIMUM_RECENT_MALFORMED_ACTION_RATE = 0.05
MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS = 2
PROGRESS_PATH = os.environ.get("EQUINOX_PROGRESS_PATH")


def render_action_prompt(tokenizer: Any, prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("the tokenizer did not render a text prompt")
    return rendered + ACTION_RESPONSE_PREFIX


def complete_json_object(response: str) -> bool:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


def positive_environment_integer(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit():
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class RuntimeConfiguration:
    model_id: str
    model_revision: str
    optimization_seed: int
    validation_examples: int
    test_examples: int
    training_tasks_per_update: int
    replay_tasks_per_level: int
    maximum_updates: int
    mastery_windows: int
    target_runtime_seconds: int
    maximum_resume_gap_seconds: int
    maximum_final_evaluation_reserve_seconds: int
    workload_attempt: int


DEFAULT_RUNTIME_CONFIGURATION = RuntimeConfiguration(
    model_id=DEFAULT_MODEL_ID,
    model_revision=SUPPORTED_MODELS[DEFAULT_MODEL_ID],
    optimization_seed=DEFAULT_SEED,
    validation_examples=DEFAULT_VALIDATION_EXAMPLES,
    test_examples=DEFAULT_TEST_EXAMPLES,
    training_tasks_per_update=DEFAULT_TRAINING_TASKS_PER_UPDATE,
    replay_tasks_per_level=DEFAULT_REPLAY_TASKS_PER_LEVEL,
    maximum_updates=DEFAULT_MAX_UPDATES,
    mastery_windows=DEFAULT_MASTERY_WINDOWS,
    target_runtime_seconds=DEFAULT_TARGET_RUNTIME_SECONDS,
    maximum_resume_gap_seconds=DEFAULT_MAXIMUM_RESUME_GAP_SECONDS,
    maximum_final_evaluation_reserve_seconds=(DEFAULT_MAX_FINAL_EVALUATION_RESERVE_SECONDS),
    workload_attempt=1,
)


def workload_attempt_from_environment() -> int:
    return positive_environment_integer(
        "EQUINOX_WORKLOAD_ATTEMPT",
        1,
        maximum=2,
    )


def configure_from_environment() -> RuntimeConfiguration:
    workload_attempt = workload_attempt_from_environment()
    model_id = os.environ.get("EQUINOX_RL_MODEL_ID", DEFAULT_MODEL_ID)
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"EQUINOX_RL_MODEL_ID must be one of {sorted(SUPPORTED_MODELS)}")
    optimization_seed = positive_environment_integer(
        "EQUINOX_RL_SEED",
        DEFAULT_SEED,
        minimum=0,
        maximum=2**31 - 1,
    )
    maximum_validation_examples = (
        min(
            semantic_task_universe_size(level, "validation")
            for level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
        )
        // 2
    )
    maximum_test_examples = min(
        semantic_task_universe_size(level, "test") for level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    )
    maximum_training_tasks = min(
        semantic_task_universe_size(level, "train") for level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    )
    validation_examples = positive_environment_integer(
        "EQUINOX_RL_VALIDATION_EXAMPLES",
        DEFAULT_VALIDATION_EXAMPLES,
        minimum=4,
        maximum=maximum_validation_examples,
    )
    test_examples = positive_environment_integer(
        "EQUINOX_RL_TEST_EXAMPLES",
        DEFAULT_TEST_EXAMPLES,
        minimum=4,
        maximum=maximum_test_examples,
    )
    training_tasks_per_update = positive_environment_integer(
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE",
        DEFAULT_TRAINING_TASKS_PER_UPDATE,
        minimum=2,
        maximum=maximum_training_tasks,
    )
    replay_tasks_per_level = positive_environment_integer(
        "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL",
        DEFAULT_REPLAY_TASKS_PER_LEVEL,
        maximum=4,
    )
    maximum_updates = positive_environment_integer(
        "EQUINOX_RL_MAX_UPDATES",
        DEFAULT_MAX_UPDATES,
        maximum=500,
    )
    mastery_windows = positive_environment_integer(
        "EQUINOX_RL_MASTERY_WINDOWS",
        DEFAULT_MASTERY_WINDOWS,
        maximum=2,
    )
    target_runtime_seconds = positive_environment_integer(
        "EQUINOX_RL_TARGET_SECONDS",
        DEFAULT_TARGET_RUNTIME_SECONDS,
        maximum=MAXIMUM_TARGET_RUNTIME_SECONDS,
    )
    maximum_resume_gap_seconds = positive_environment_integer(
        "EQUINOX_RL_MAX_RESUME_GAP_SECONDS",
        DEFAULT_MAXIMUM_RESUME_GAP_SECONDS,
        maximum=7_200,
    )
    maximum_final_evaluation_reserve_seconds = positive_environment_integer(
        "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS",
        DEFAULT_MAX_FINAL_EVALUATION_RESERVE_SECONDS,
        minimum=300,
        maximum=7_200,
    )
    if target_runtime_seconds <= maximum_final_evaluation_reserve_seconds + 60:
        raise ValueError(
            "EQUINOX_RL_TARGET_SECONDS must leave at least 60 seconds "
            "for training before final evaluation"
        )
    return RuntimeConfiguration(
        model_id=model_id,
        model_revision=SUPPORTED_MODELS[model_id],
        optimization_seed=optimization_seed,
        validation_examples=validation_examples,
        test_examples=test_examples,
        training_tasks_per_update=training_tasks_per_update,
        replay_tasks_per_level=replay_tasks_per_level,
        maximum_updates=maximum_updates,
        mastery_windows=mastery_windows,
        target_runtime_seconds=target_runtime_seconds,
        maximum_resume_gap_seconds=maximum_resume_gap_seconds,
        maximum_final_evaluation_reserve_seconds=(maximum_final_evaluation_reserve_seconds),
        workload_attempt=workload_attempt,
    )


@dataclass(frozen=True)
class GeneratedAction:
    response: str
    input_ids: tuple[int, ...] = ()
    attention_mask: tuple[int, ...] = ()
    completion_mask: tuple[int, ...] = ()


@dataclass(frozen=True)
class WeightedAction:
    generated: GeneratedAction
    weight: float


@dataclass
class BranchCollection:
    task: RepairTask
    snapshot: EnvironmentSnapshot | None
    prefix: RepositoryRepairEnvironment
    siblings: list[RepositoryRepairEnvironment]
    generated_by_sibling: list[list[GeneratedAction]]
    sampling_seeds: list[int]
    returns: list[float]
    advantages: list[float]
    exclusion_reason: str | None
    replay: bool
    generated_prefix: list[GeneratedAction] = field(default_factory=list)
    curriculum_role: str = "active_frontier"

    @property
    def solved_siblings(self) -> int:
        return sum(
            sibling.terminal and sibling.terminal_reason == "solved" for sibling in self.siblings
        )

    @property
    def informative(self) -> bool:
        return any(abs(advantage) > 1e-8 for advantage in self.advantages)


SampleOne = Callable[[str, bool, int], GeneratedAction]


def sampled_action_count(collections: list[BranchCollection]) -> int:
    return sum(
        len(collection.prefix.steps)
        + sum(len(sibling.steps) - len(collection.prefix.steps) for sibling in collection.siblings)
        for collection in collections
    )


def post_branch_action_count(collections: list[BranchCollection]) -> int:
    return sum(
        sum(len(actions) for actions in collection.generated_by_sibling)
        for collection in collections
    )


def action_protocol_counts(collection: BranchCollection) -> tuple[int, int]:
    steps = [
        *collection.prefix.steps,
        *(
            step
            for sibling in collection.siblings
            for step in sibling.steps[len(collection.prefix.steps) :]
        ),
    ]
    return len(steps), sum(step.action is None for step in steps)


def recent_action_protocol_summary(
    group_counts: list[tuple[int, int]],
    *,
    maximum_groups: int,
) -> dict[str, Any]:
    recent = group_counts[-maximum_groups:]
    total_actions = sum(total for total, _ in recent)
    malformed_actions = sum(malformed for _, malformed in recent)
    return {
        "groups": len(recent),
        "window_complete": len(recent) == maximum_groups,
        "total_actions": total_actions,
        "malformed_actions": malformed_actions,
        "validity_rate": (
            round((total_actions - malformed_actions) / total_actions, 6) if total_actions else None
        ),
        "malformed_rate": (round(malformed_actions / total_actions, 6) if total_actions else None),
    }


def next_malformed_action_window_streak(
    current_streak: int,
    protocol_summary: dict[str, Any],
    *,
    maximum_malformed_rate: float = MAXIMUM_RECENT_MALFORMED_ACTION_RATE,
) -> int:
    malformed_rate = protocol_summary.get("malformed_rate")
    if (
        protocol_summary.get("window_complete") is not True
        or malformed_rate is None
        or float(malformed_rate) <= maximum_malformed_rate
    ):
        return 0
    return current_streak + 1


def next_uninformative_group_streak(
    current: int,
    collections: list[BranchCollection],
) -> int:
    streak = current
    for collection in collections:
        streak = 0 if collection.informative else streak + 1
    return streak


def uninformative_group_limit_reached(
    current: int,
    collections: list[BranchCollection],
    *,
    maximum_consecutive_groups: int,
) -> bool:
    streak = current
    for collection in collections:
        streak = 0 if collection.informative else streak + 1
        if streak >= maximum_consecutive_groups:
            return True
    return False


def validation_regression_decision(
    *,
    best_exact_successes: int,
    observed_exact_successes: int,
    consecutive_regressions: int,
) -> tuple[bool, int, int]:
    if observed_exact_successes > best_exact_successes:
        return True, observed_exact_successes, 0
    if observed_exact_successes < best_exact_successes:
        return (
            False,
            best_exact_successes,
            consecutive_regressions + 1,
        )
    return False, best_exact_successes, 0


def lightweight_validation_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "update",
        "level",
        "exact_successes",
        "exact_rate",
        "exact_rate_95ci",
        "checkpoint_successes",
        "checkpoint_rate",
        "checkpoint_rate_95ci",
        "examples",
        "mastered",
        "mastery_streak",
        "policy_loss",
        "reinforce_loss",
        "reference_kl",
        "gradient_norm",
        "curriculum_baseline_exact_successes",
        "curriculum_baseline_exact_rate",
        "curriculum_exact_successes",
        "curriculum_exact_rate",
        "curriculum_paired_change",
        "fixed_guard_levels",
        "fixed_guard_paired_change",
        "retention_guard_passed",
        "elapsed_seconds",
        "validation_elapsed_seconds",
        "final_evaluation_reserve_seconds",
        "regression_streak",
        "best_checkpoint",
    )
    return [{key: item[key] for key in keys if key in item} for item in history]


def discarded_collection_accounting(
    collections: list[BranchCollection],
) -> tuple[int, int, int]:
    return (
        len(collections),
        sampled_action_count(collections),
        post_branch_action_count(collections),
    )


def checkpoint_target_disposition(
    target: str,
    *,
    checkpoint_name: str,
    previous_checkpoint: object,
) -> str:
    if os.path.isdir(target) and previous_checkpoint == checkpoint_name:
        return "rewrite_live"
    if os.path.exists(target):
        return "replace_orphan"
    return "create"


def remove_orphan_checkpoint_target(target: str) -> None:
    if os.path.isdir(target) and not os.path.islink(target):
        shutil.rmtree(target)
    else:
        os.remove(target)


def remove_stale_checkpoint_targets(checkpoints_root: str, live_checkpoint: str) -> None:
    for name in os.listdir(checkpoints_root):
        suffix = name.removeprefix("update-")
        if name == live_checkpoint or not name.startswith("update-") or not suffix.isdigit():
            continue
        remove_orphan_checkpoint_target(os.path.join(checkpoints_root, name))


def fsync_file(path: str) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: str) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_tree(root: str) -> None:
    for current_root, _, files in os.walk(root, topdown=False):
        for name in files:
            fsync_file(os.path.join(current_root, name))
        fsync_directory(current_root)


def persist_checkpoint(
    *,
    checkpoints_root: str,
    latest_checkpoint_path: str,
    checkpoint_name: str,
    state: dict[str, Any],
    save_adapter: Callable[[str], None],
    save_state: Callable[[dict[str, Any], str], None],
    after_step: Callable[[str], None] | None = None,
) -> None:
    os.makedirs(checkpoints_root, exist_ok=True)
    target = os.path.join(checkpoints_root, checkpoint_name)
    previous = None
    if os.path.isfile(latest_checkpoint_path):
        with open(latest_checkpoint_path, encoding="utf-8") as handle:
            previous = json.load(handle).get("checkpoint")
    target_disposition = checkpoint_target_disposition(
        target,
        checkpoint_name=checkpoint_name,
        previous_checkpoint=previous,
    )
    rewrite_live_state = target_disposition == "rewrite_live"
    if target_disposition == "replace_orphan":
        remove_orphan_checkpoint_target(target)
    if rewrite_live_state:
        required_adapter_files = (
            os.path.join(target, "adapter_config.json"),
            os.path.join(target, "adapter_model.safetensors"),
            os.path.join(target, "training-state.pt"),
        )
        if not all(os.path.isfile(path) for path in required_adapter_files):
            raise RuntimeError("live checkpoint is incomplete")
    else:
        os.makedirs(target)
        save_adapter(target)
        fsync_tree(target)
        fsync_directory(checkpoints_root)
        if after_step is not None:
            after_step("adapter_saved")
    temporary_state = os.path.join(target, "training-state.pt.pending")
    save_state(state, temporary_state)
    fsync_file(temporary_state)
    os.replace(temporary_state, os.path.join(target, "training-state.pt"))
    fsync_directory(target)
    if after_step is not None:
        after_step("state_persisted")
    if not rewrite_live_state:
        temporary_pointer = latest_checkpoint_path + ".pending"
        with open(temporary_pointer, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": 1, "checkpoint": checkpoint_name},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_pointer, latest_checkpoint_path)
        fsync_directory(checkpoints_root)
        if after_step is not None:
            after_step("pointer_persisted")
    remove_stale_checkpoint_targets(checkpoints_root, checkpoint_name)
    fsync_directory(checkpoints_root)


def persist_named_adapter(
    *,
    checkpoints_root: str,
    name: str,
    metadata: dict[str, Any],
    save_adapter: Callable[[str], None],
) -> None:
    if not name or "/" in name or name.startswith("."):
        raise ValueError("named adapter checkpoint is invalid")
    os.makedirs(checkpoints_root, exist_ok=True)
    target = os.path.join(checkpoints_root, name)
    pending = target + ".pending"
    if os.path.exists(pending):
        remove_orphan_checkpoint_target(pending)
    os.makedirs(pending)
    save_adapter(pending)
    metadata_path = os.path.join(pending, "checkpoint-metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": 1, **metadata},
            handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_tree(pending)
    if os.path.exists(target):
        remove_orphan_checkpoint_target(target)
    os.replace(pending, target)
    fsync_directory(checkpoints_root)


def emit_progress(
    phase: str,
    message: str,
    runtime_configuration: RuntimeConfiguration | None,
    *,
    preserve_context: bool = False,
    **values: Any,
) -> None:
    if not PROGRESS_PATH:
        return
    preserved: dict[str, Any] = {}
    if preserve_context and os.path.isfile(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = {}
        if isinstance(previous, dict):
            for key in (
                "update",
                "current_level",
                "maximum_updates",
                "elapsed_seconds",
                "action_protocol_validity_rate",
                "minimum_protocol_validity_rate",
            ):
                if key in previous:
                    preserved[key] = previous[key]
    payload = {
        "schema_version": 2,
        "phase": phase,
        "message": message,
        "branch_width": BRANCH_WIDTH,
        "complexity_strategy": "adaptive",
        "maximum_level": MAXIMUM_COMPLEXITY_LEVEL,
        "multi_step": True,
        "restored_continuations": True,
        **(
            {
                "maximum_updates": runtime_configuration.maximum_updates,
                "attempt": runtime_configuration.workload_attempt,
            }
            if runtime_configuration is not None
            else {}
        ),
        **preserved,
        "error": None,
        **values,
    }
    temporary_path = f"{PROGRESS_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, PROGRESS_PATH)
    fsync_directory(os.path.dirname(PROGRESS_PATH) or ".")


def sibling_advantages(returns: list[float]) -> list[float]:
    if len(returns) != BRANCH_WIDTH:
        raise ValueError(f"expected {BRANCH_WIDTH} sibling returns")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("sibling returns must be finite")
    pooled_standard_deviation = statistics.pstdev(returns)
    if pooled_standard_deviation <= 1e-8:
        return [0.0] * BRANCH_WIDTH
    advantage_scale = max(
        pooled_standard_deviation,
        ADVANTAGE_STANDARD_DEVIATION_FLOOR,
    )
    total = sum(returns)
    advantages = [
        (value - (total - value) / (BRANCH_WIDTH - 1)) / advantage_scale for value in returns
    ]
    centered = sum(advantages) / BRANCH_WIDTH
    centered_advantages = [value - centered for value in advantages]
    maximum_magnitude = max(abs(value) for value in centered_advantages)
    scale = min(1.0, MAXIMUM_ABSOLUTE_ADVANTAGE / maximum_magnitude)
    return [round(value * scale, 8) for value in centered_advantages]


def correctness_contrast_advantages(returns: list[float]) -> list[float]:
    if len(returns) != BRANCH_WIDTH:
        raise ValueError(f"expected {BRANCH_WIDTH} sibling returns")
    solved = [value > 0 for value in returns]
    if all(solved) or not any(solved):
        return [0.0] * BRANCH_WIDTH
    return sibling_advantages(returns)


def training_level_allocation(
    current_level: int,
    task_count: int,
    *,
    maximum_level: int = MAXIMUM_COMPLEXITY_LEVEL,
) -> list[int]:
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    if task_count < 1:
        raise ValueError("task count must be positive")
    allocation = [current_level] * task_count
    probe_levels = list(
        range(
            current_level + 1,
            min(maximum_level, current_level + MAXIMUM_FRONTIER_PROBE_OFFSET) + 1,
        )
    )
    for index, probe_level in enumerate(
        probe_levels[: task_count // 2],
        start=1,
    ):
        allocation[-index] = probe_level
    return sorted(allocation)


def training_analogue_family_ids(
    failure_family_ids: list[str] | tuple[str, ...],
) -> list[str]:
    """Resolve declared train-split analogues without consulting test outcomes."""
    training_families = {template[0] for template in TEMPLATE_SPLITS["train"]}
    adjacency: dict[str, set[str]] = {}
    for disclosure in STRUCTURAL_MIRROR_DISCLOSURES:
        families = {str(family) for family in disclosure["families"]}
        for family in families:
            adjacency.setdefault(family, set()).update(families - {family})

    analogues = {
        analogue
        for failure_family in failure_family_ids
        for analogue in adjacency.get(failure_family, set())
    }
    return sorted(analogues & training_families)


def failure_directed_training_tasks(
    level: int,
    count: int,
    seed: int,
    *,
    target_family_ids: list[str] | tuple[str, ...],
) -> list[RepairTask]:
    """Sample distinct train semantics that cover declared weak-family analogues."""
    if count < 1:
        raise ValueError("training task count must be positive")
    targets = sorted(set(target_family_ids))
    if not targets:
        return make_tasks(level, count, seed, split="train")

    target_count = min(count, len(targets))
    rotation = seed % len(targets)
    selected_targets = [targets[(rotation + index) % len(targets)] for index in range(target_count)]
    tasks: list[RepairTask] = []
    selected_semantics: set[str] = set()
    for target_index, target in enumerate(selected_targets):
        for candidate_index in range(1_000):
            candidate = make_task(
                level,
                seed + target_index * 100_003 + candidate_index * 7_919,
                split="train",
            )
            if target not in {fault.family_id for fault in candidate.faults}:
                continue
            if candidate.semantic_task_id in selected_semantics:
                continue
            tasks.append(candidate)
            selected_semantics.add(candidate.semantic_task_id)
            break
        else:
            raise RuntimeError(
                f"could not sample a level-{level} training task for analogue {target}"
            )

    remaining = count - len(tasks)
    if remaining:
        tasks.extend(
            make_tasks(
                level,
                remaining,
                seed + 900_001,
                split="train",
                exclude_semantic_task_ids=frozenset(selected_semantics),
            )
        )
    return tasks


def retention_guard_decision(
    *,
    best_fixed_successes: int,
    best_guard_net_improved: int,
    observed_fixed_successes: int,
    guard_change: dict[str, Any],
    consecutive_regressions: int,
) -> tuple[bool, int, int, int]:
    guard_regressions = int(guard_change["regressed"])
    guard_net_improved = int(guard_change["net_improved"])
    regressed = observed_fixed_successes < best_fixed_successes or guard_regressions > 0
    candidate_is_best = guard_regressions == 0 and (
        observed_fixed_successes > best_fixed_successes
        or (
            observed_fixed_successes == best_fixed_successes
            and guard_net_improved > best_guard_net_improved
        )
    )
    if candidate_is_best:
        return (
            True,
            observed_fixed_successes,
            guard_net_improved,
            0,
        )
    return (
        False,
        best_fixed_successes,
        best_guard_net_improved,
        consecutive_regressions + 1 if regressed else 0,
    )


def paired_retention_guard_decision(
    *,
    best_fixed_net_improved: int,
    best_rotating_net_improved: int,
    fixed_change: dict[str, Any],
    rotating_change: dict[str, Any],
    consecutive_regressions: int,
) -> tuple[bool, int, int, int]:
    fixed_net_improved = int(fixed_change["net_improved"])
    rotating_net_improved = int(rotating_change["net_improved"])
    regressed = int(fixed_change["regressed"]) > 0 or int(rotating_change["regressed"]) > 0
    candidate_is_best = not regressed and (
        fixed_net_improved > best_fixed_net_improved
        or (
            fixed_net_improved == best_fixed_net_improved
            and rotating_net_improved > best_rotating_net_improved
        )
    )
    if candidate_is_best:
        return True, fixed_net_improved, rotating_net_improved, 0
    return (
        False,
        best_fixed_net_improved,
        best_rotating_net_improved,
        consecutive_regressions + 1 if regressed else 0,
    )


def fixed_retention_guard_levels(
    current_level: int,
    *,
    maximum_level: int = MAXIMUM_COMPLEXITY_LEVEL,
) -> list[int]:
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    return list(range(min(maximum_level, current_level + 1) + 1))


def sampled_reverse_kl_penalty(log_reference_over_policy: float) -> float:
    return math.expm1(log_reference_over_policy) - log_reference_over_policy


def wilson_interval(successes: int, total: int, *, z_score: float = 1.96) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    radius = (
        z_score
        * math.sqrt(proportion * (1 - proportion) / total + z_squared / (4 * total**2))
        / denominator
    )
    return [
        round(max(0.0, center - radius), 6),
        round(min(1.0, center + radius), 6),
    ]


def validation_window_seed(level: int, update: int) -> int:
    return (
        VALIDATION_SEED_BASE
        + 10_000_000
        + level * 1_000_000
        + update * VALIDATION_WINDOW_SEED_STRIDE
    )


def checkpoint_validation_seed(level: int) -> int:
    return VALIDATION_SEED_BASE + level * 1_000


def bounded_final_evaluation_reserve(
    estimated_seconds: int,
    maximum_seconds: int = DEFAULT_MAX_FINAL_EVALUATION_RESERVE_SECONDS,
) -> tuple[int, bool]:
    reserve_seconds = max(300, estimated_seconds)
    exceeded_ceiling = reserve_seconds > maximum_seconds
    return min(reserve_seconds, maximum_seconds), exceeded_ceiling


def retained_final_evaluation_reserve(
    *,
    current_seconds: int,
    current_maximum_measured_seconds: int,
    current_exceeded_ceiling: bool,
    candidate_measured_seconds: int,
    maximum_seconds: int,
    candidate_retained: bool,
) -> tuple[int, int, bool]:
    if not candidate_retained:
        return (
            current_seconds,
            current_maximum_measured_seconds,
            current_exceeded_ceiling,
        )
    candidate_seconds, candidate_exceeded_ceiling = bounded_final_evaluation_reserve(
        candidate_measured_seconds,
        maximum_seconds,
    )
    return (
        max(current_seconds, candidate_seconds),
        max(current_maximum_measured_seconds, candidate_measured_seconds),
        current_exceeded_ceiling or candidate_exceeded_ceiling,
    )


def training_loop_entry(
    resume_state: dict[str, Any] | None,
    *,
    updates_completed: int,
    maximum_updates: int,
) -> tuple[bool, str, int]:
    training_complete = (
        bool(resume_state.get("training_complete", False)) if resume_state else False
    )
    if not training_complete:
        return False, "maximum_updates", updates_completed + 1
    stop_reason = resume_state.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise RuntimeError("terminal training checkpoint omitted its stop reason")
    return True, stop_reason, maximum_updates + 1


def training_stop_decision(
    *,
    elapsed_seconds: float,
    target_seconds: int,
    final_evaluation_reserve_seconds: int,
    final_evaluation_reserve_exceeded_ceiling: bool = False,
    maximum_level_mastered: bool = False,
) -> tuple[bool, str | None, float]:
    training_deadline_seconds = target_seconds - final_evaluation_reserve_seconds
    if final_evaluation_reserve_exceeded_ceiling:
        return True, "final_evaluation_reserve_ceiling", training_deadline_seconds
    if maximum_level_mastered:
        return True, "maximum_level_mastered", training_deadline_seconds
    if elapsed_seconds + final_evaluation_reserve_seconds + 60 >= target_seconds:
        return True, "final_evaluation_reserve", training_deadline_seconds
    return False, None, training_deadline_seconds


def evaluation_reward_summary(
    initial_by_level: dict[str, dict[str, Any]],
    final_by_level: dict[str, dict[str, Any]],
    *,
    complete: bool,
) -> tuple[float | None, float | None, float | None]:
    if not complete:
        return None, None, None
    initial_rates = [
        observation["exact_rate"]
        for observation in initial_by_level.values()
        if observation["exact_rate"] is not None
    ]
    final_rates = [
        observation["exact_rate"]
        for observation in final_by_level.values()
        if observation["exact_rate"] is not None
    ]
    initial_reward = statistics.mean(initial_rates) if initial_rates else None
    final_reward = statistics.mean(final_rates) if final_rates else None
    reward_gain = (
        final_reward - initial_reward
        if initial_reward is not None and final_reward is not None
        else None
    )
    return initial_reward, final_reward, reward_gain


def post_training_claim_strength(
    *,
    final_evaluation_complete: bool,
    probative_post_training: bool,
) -> str:
    if not final_evaluation_complete:
        return "INCOMPLETE_FINAL_EVALUATION"
    if not probative_post_training:
        return "NONPROBATIVE_RESERVE_STOP"
    return "EXPLORATORY_SINGLE_SEED"


def resumed_crash_tail_actions_unaccounted(
    resume_state: dict[str, Any] | None,
) -> bool:
    return resume_state is not None and not bool(resume_state.get("training_complete"))


def paired_change_summary(
    initial_outcomes: list[dict[str, Any]],
    final_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    initial = {item["task_id"]: bool(item["solved"]) for item in initial_outcomes}
    final = {item["task_id"]: bool(item["solved"]) for item in final_outcomes}
    if initial.keys() != final.keys() or not initial:
        raise ValueError("paired evaluation requires the same non-empty task set")
    improved = sum(not initial[task_id] and final[task_id] for task_id in initial)
    regressed = sum(initial[task_id] and not final[task_id] for task_id in initial)
    discordant = improved + regressed
    if discordant:
        smaller_tail = (
            sum(math.comb(discordant, count) for count in range(min(improved, regressed) + 1))
            / 2**discordant
        )
        exact_p_value = min(1.0, 2 * smaller_tail)
    else:
        exact_p_value = 1.0
    return {
        "examples": len(initial),
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(initial) - discordant,
        "net_improved": improved - regressed,
        "mcnemar_exact_p_value": exact_p_value,
    }


def collect_branch_group(
    task: RepairTask,
    sample_one: SampleOne,
    *,
    stochastic: bool,
    sampling_seed: int,
    replay: bool = False,
    curriculum_role: str = "active_frontier",
    deadline_reached: Callable[[], bool] | None = None,
) -> BranchCollection:
    prefix = RepositoryRepairEnvironment(task)
    generated_prefix: list[GeneratedAction] = []
    accepted_diagnostics = 0
    for attempt in range(PREFIX_MAX_ATTEMPTS):
        if deadline_reached is not None and deadline_reached():
            return BranchCollection(
                task=task,
                snapshot=None,
                prefix=prefix,
                siblings=[],
                generated_by_sibling=[],
                sampling_seeds=[],
                returns=[],
                advantages=[],
                exclusion_reason="TRAINING_DEADLINE_REACHED",
                replay=replay,
                generated_prefix=generated_prefix,
                curriculum_role=curriculum_role,
            )
        generated = sample_one(
            prefix.policy_prompt("shared_prefix"),
            False,
            sampling_seed + attempt,
        )
        generated_prefix.append(generated)
        step = prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if step.accepted:
            accepted_diagnostics += 1
        if accepted_diagnostics >= PREFIX_ACCEPTED_ACTIONS:
            break

    if accepted_diagnostics < PREFIX_ACCEPTED_ACTIONS:
        return BranchCollection(
            task=task,
            snapshot=None,
            prefix=prefix,
            siblings=[],
            generated_by_sibling=[],
            sampling_seeds=[],
            returns=[],
            advantages=[],
            exclusion_reason="PREFIX_CHECKPOINT_NOT_REACHED",
            replay=replay,
            generated_prefix=generated_prefix,
            curriculum_role=curriculum_role,
        )

    snapshot = prefix.capture_snapshot()
    siblings = [RepositoryRepairEnvironment.restore(task, snapshot) for _ in range(BRANCH_WIDTH)]
    generated_by_sibling: list[list[GeneratedAction]] = [[] for _ in range(BRANCH_WIDTH)]
    sampling_seeds = [sampling_seed + 10_000 * (index + 1) for index in range(BRANCH_WIDTH)]
    while not all(sibling.terminal for sibling in siblings):
        for sibling_index, sibling in enumerate(siblings):
            if sibling.terminal:
                continue
            if deadline_reached is not None and deadline_reached():
                return BranchCollection(
                    task=task,
                    snapshot=snapshot,
                    prefix=prefix,
                    siblings=siblings,
                    generated_by_sibling=generated_by_sibling,
                    sampling_seeds=sampling_seeds,
                    returns=[],
                    advantages=[],
                    exclusion_reason="TRAINING_DEADLINE_REACHED",
                    replay=replay,
                    generated_prefix=generated_prefix,
                    curriculum_role=curriculum_role,
                )
            action_index = len(generated_by_sibling[sibling_index])
            generated = sample_one(
                sibling.policy_prompt("continuation"),
                stochastic,
                sampling_seeds[sibling_index] + action_index,
            )
            generated_by_sibling[sibling_index].append(generated)
            sibling.step(generated.response)

    returns = [sibling.terminal_reward for sibling in siblings]
    return BranchCollection(
        task=task,
        snapshot=snapshot,
        prefix=prefix,
        siblings=siblings,
        generated_by_sibling=generated_by_sibling,
        sampling_seeds=sampling_seeds,
        returns=returns,
        advantages=correctness_contrast_advantages(returns),
        exclusion_reason=None,
        replay=replay,
        generated_prefix=generated_prefix,
        curriculum_role=curriculum_role,
    )


def collect_greedy_trajectory(
    task: RepairTask,
    sample_one: SampleOne,
    *,
    sampling_seed: int,
    deadline_reached: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    prefix = RepositoryRepairEnvironment(task)
    accepted_diagnostics = 0
    for attempt in range(PREFIX_MAX_ATTEMPTS):
        if deadline_reached is not None and deadline_reached():
            return None
        generated = sample_one(
            prefix.policy_prompt("shared_prefix"),
            False,
            sampling_seed + attempt,
        )
        step = prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if step.accepted:
            accepted_diagnostics += 1
        if accepted_diagnostics >= PREFIX_ACCEPTED_ACTIONS:
            break
    if accepted_diagnostics < PREFIX_ACCEPTED_ACTIONS:
        return {
            "solved": False,
            "checkpoint_reached": False,
            "actions": len(prefix.steps),
            "accepted_actions": sum(step.accepted for step in prefix.steps),
            "malformed_actions": sum(step.action is None for step in prefix.steps),
            "reward": 0.0,
        }

    continuation = RepositoryRepairEnvironment.restore(task, prefix.capture_snapshot())
    while not continuation.terminal:
        if deadline_reached is not None and deadline_reached():
            return None
        generated = sample_one(
            continuation.policy_prompt("continuation"),
            False,
            sampling_seed + 1_000 + len(continuation.steps),
        )
        continuation.step(generated.response)
    return {
        "solved": continuation.terminal_reason == "solved",
        "checkpoint_reached": True,
        "actions": len(continuation.steps),
        "accepted_actions": sum(step.accepted for step in continuation.steps),
        "malformed_actions": sum(step.action is None for step in continuation.steps),
        "reward": continuation.terminal_reward,
    }


def policy_examples(collection: BranchCollection) -> list[WeightedAction]:
    if collection.exclusion_reason or not collection.informative:
        return []
    solved_sibling_indexes = [
        index
        for index, sibling in enumerate(collection.siblings)
        if sibling.terminal_reason == "solved"
    ]
    if not solved_sibling_indexes:
        return []
    examples: list[WeightedAction] = []
    per_solved_trajectory_weight = 1.0 / len(solved_sibling_indexes)
    for sibling_index in solved_sibling_indexes:
        generated_actions = collection.generated_by_sibling[sibling_index]
        post_branch_steps = collection.siblings[sibling_index].steps[len(collection.prefix.steps) :]
        previous_fixed_faults = (
            int(getattr(collection.prefix.steps[-1], "fixed_faults", 0))
            if collection.prefix.steps
            else 0
        )
        fault_fixing_edits: list[GeneratedAction] = []
        for generated, step in zip(
            generated_actions,
            post_branch_steps,
            strict=True,
        ):
            fixed_faults = int(getattr(step, "fixed_faults", previous_fixed_faults))
            if (
                step.accepted
                and getattr(step, "tool", None) == "edit"
                and fixed_faults > previous_fixed_faults
                and generated.input_ids
            ):
                fault_fixing_edits.append(generated)
            previous_fixed_faults = fixed_faults
        if not fault_fixing_edits:
            continue
        per_action_weight = per_solved_trajectory_weight / len(fault_fixing_edits)
        examples.extend(
            WeightedAction(generated=generated, weight=per_action_weight)
            for generated in fault_fixing_edits
        )
    return examples


def accepted_reference_actions(collection: BranchCollection) -> list[GeneratedAction]:
    accepted = [
        generated
        for generated, step in zip(
            collection.generated_prefix,
            collection.prefix.steps,
            strict=True,
        )
        if step.accepted and generated.input_ids
    ]
    for sibling_index, generated_actions in enumerate(collection.generated_by_sibling):
        post_branch_steps = collection.siblings[sibling_index].steps[len(collection.prefix.steps) :]
        accepted.extend(
            generated
            for generated, step in zip(
                generated_actions,
                post_branch_steps,
                strict=True,
            )
            if step.accepted and generated.input_ids
        )
    return accepted


def reference_anchored_examples(
    collections: list[BranchCollection],
    *,
    minimum_informative_groups: int = 1,
) -> tuple[list[WeightedAction], int]:
    if minimum_informative_groups < 1:
        raise ValueError("minimum informative groups must be positive")
    informative_groups = sum(collection.informative for collection in collections)
    policy_training_examples = (
        [example for collection in collections for example in policy_examples(collection)]
        if informative_groups >= minimum_informative_groups
        else []
    )
    policy_weights_by_identity: dict[int, list[float]] = {}
    for example in policy_training_examples:
        policy_weights_by_identity.setdefault(id(example.generated), []).append(example.weight)
    reference_actions = [
        generated
        for collection in collections
        for generated in accepted_reference_actions(collection)
    ]
    anchored: list[WeightedAction] = []
    for generated in reference_actions:
        weights = policy_weights_by_identity.get(id(generated), [])
        anchored.append(
            WeightedAction(
                generated=generated,
                weight=weights.pop(0) if weights else 0.0,
            )
        )
    if any(weights for weights in policy_weights_by_identity.values()):
        raise RuntimeError("policy training action is missing from the reference anchor")
    return anchored, len(policy_training_examples)


def accumulated_reference_anchored_examples(
    collections: list[BranchCollection],
    pending_training_examples: list[WeightedAction],
    pending_policy_example_count: int,
    pending_informative_group_ids: list[str],
    *,
    minimum_informative_groups: int,
) -> tuple[
    list[WeightedAction],
    int,
    list[WeightedAction],
    int,
    list[str],
    list[str],
]:
    if minimum_informative_groups < 1:
        raise ValueError("minimum informative groups must be positive")
    if pending_policy_example_count < 0:
        raise ValueError("pending policy example count cannot be negative")
    if bool(pending_training_examples) != bool(pending_informative_group_ids):
        raise ValueError("pending training examples and group identities must agree")
    if bool(pending_policy_example_count) != bool(pending_informative_group_ids):
        raise ValueError("pending policy example count and group identities must agree")
    if len(pending_informative_group_ids) >= minimum_informative_groups:
        raise ValueError("a complete pending policy batch should already have been applied")
    if len(set(pending_informative_group_ids)) != len(pending_informative_group_ids):
        raise ValueError("pending informative group identities must be unique")

    current_policy_groups = [
        (collection.task.task_id, examples)
        for collection in collections
        if (examples := policy_examples(collection))
    ]
    current_group_ids = [group_id for group_id, _ in current_policy_groups]
    accumulated_group_ids = [*pending_informative_group_ids, *current_group_ids]
    if len(set(accumulated_group_ids)) != len(accumulated_group_ids):
        raise RuntimeError("an informative task group was sampled more than once")
    current_anchored, current_policy_example_count = reference_anchored_examples(
        collections,
    )
    accumulated_examples = [*pending_training_examples, *current_anchored]
    accumulated_policy_example_count = pending_policy_example_count + current_policy_example_count
    if not accumulated_group_ids:
        return (
            current_anchored,
            0,
            [],
            0,
            [],
            [],
        )
    if len(accumulated_group_ids) < minimum_informative_groups:
        return (
            [],
            0,
            accumulated_examples,
            accumulated_policy_example_count,
            accumulated_group_ids,
            [],
        )

    return (
        accumulated_examples,
        accumulated_policy_example_count,
        [],
        0,
        [],
        accumulated_group_ids,
    )


def serialize_step(step: Any) -> dict[str, Any]:
    value = asdict(step)
    value["step_id"] = f"step-{step.index}"
    return value


def sibling_failure_classification(
    sibling: RepositoryRepairEnvironment,
) -> list[dict[str, str]]:
    classifications: list[dict[str, str]] = []
    if any(step.action is None for step in sibling.steps):
        classifications.append({"category": "malformed_policy_action", "source": "typed"})
    if sibling.terminal_reason == "finished_with_failures":
        classifications.append({"category": "valid_candidate_failure", "source": "typed"})
        last_test = next(
            (
                step
                for step in reversed(sibling.steps[:-1])
                if step.accepted and step.tool == "test"
            ),
            None,
        )
        classifications.append(
            {
                "category": (
                    "strategy_persistence_despite_negative_feedback"
                    if last_test is not None and not last_test.verifier_passed
                    else "insufficient_verification_before_finish"
                ),
                "source": "heuristic",
            }
        )
    elif sibling.terminal_reason == "horizon_exhausted":
        classifications.append({"category": "unproductive_action_loop", "source": "typed"})
    return classifications


def serialize_branch_group(
    collection: BranchCollection,
    *,
    update: int,
    optimizer_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_id = (
        collection.snapshot.snapshot_id
        if collection.snapshot is not None
        else f"unavailable-{collection.task.task_id}"
    )
    best_sibling_index = (
        max(
            range(len(collection.returns)),
            key=lambda index: (collection.returns[index], -index),
        )
        if collection.returns
        else None
    )

    def serialized_sibling(index: int, sibling: RepositoryRepairEnvironment) -> dict[str, Any]:
        sibling_return = collection.returns[index] if index < len(collection.returns) else None
        sibling_advantage = (
            collection.advantages[index] if index < len(collection.advantages) else None
        )
        generated_actions = (
            collection.generated_by_sibling[index]
            if index < len(collection.generated_by_sibling)
            else []
        )
        post_branch_steps = sibling.steps[len(collection.prefix.steps) :]
        accepted_training_actions = sum(
            step.accepted and bool(generated.input_ids)
            for generated, step in zip(
                generated_actions,
                post_branch_steps,
                strict=True,
            )
        )
        solved_sibling_count = sum(
            candidate.terminal_reason == "solved" for candidate in collection.siblings
        )
        policy_signal = (
            collection.informative
            and sibling.terminal_reason == "solved"
            and accepted_training_actions > 0
        )
        reward_components = sibling.reward_components()
        return {
            "index": index,
            "sampling_seed": collection.sampling_seeds[index],
            "return": sibling_return,
            "advantage": sibling_advantage,
            "policy_signal": policy_signal,
            "passed": sibling.terminal_reason == "solved",
            "terminal_reason": sibling.terminal_reason,
            "trajectory_digest": sibling.trajectory_digest,
            "reward_components": reward_components,
            "completion_tokens": sum(
                sum(generated.completion_mask) for generated in generated_actions
            ),
            "failure_classification": sibling_failure_classification(sibling),
            "effective_batch_weight": (
                round(1.0 / solved_sibling_count / accepted_training_actions, 8)
                if policy_signal and solved_sibling_count
                else 0.0
            ),
            "steps": [
                serialize_step(step) for step in sibling.steps[len(collection.prefix.steps) :]
            ],
        }

    return {
        "schema_version": 2,
        "selection": collection.curriculum_role,
        "snapshot_id": f"update-{update}-{snapshot_id}",
        "update": update,
        "level": collection.task.level,
        "domain": "micro_repository",
        "task_id": collection.task.task_id,
        "task": {
            "description": collection.task.description,
            "known_failing_tests": list(collection.task.failing_tests),
            "family_ids": [fault.family_id for fault in collection.task.faults],
            "complexity": asdict(collection.task.complexity),
        },
        "checkpoint": (
            {
                "checkpoint_id": snapshot_id,
                "payload_digest": collection.snapshot.payload_digest,
                "fidelity": collection.snapshot.fidelity,
                "environment_revision": ENVIRONMENT_REVISION,
                "verifier_revision": VERIFIER_REVISION,
                "action_protocol_revision": ACTION_PROTOCOL_REVISION,
                "static_branch_width": BRANCH_WIDTH,
            }
            if collection.snapshot is not None
            else None
        ),
        "shared_prefix": {
            "policy_generated": True,
            "accepted_diagnostic_actions": sum(
                step.accepted and step.tool in DIAGNOSTIC_TOOLS for step in collection.prefix.steps
            ),
            "steps": [serialize_step(step) for step in collection.prefix.steps],
        },
        "best_sibling_index": best_sibling_index,
        "learning_signal": collection.informative,
        "excluded": collection.exclusion_reason is not None,
        "exclusion_reason": collection.exclusion_reason,
        "replay": collection.replay,
        "curriculum_role": collection.curriculum_role,
        "optimizer_update": optimizer_update,
        "siblings": [
            serialized_sibling(index, sibling) for index, sibling in enumerate(collection.siblings)
        ],
    }


def observation_mastered(observation: dict[str, Any]) -> bool:
    return (
        observation["exact_rate_95ci"][0] >= MASTERY_THRESHOLD
        and observation["checkpoint_rate_95ci"][0] >= MASTERY_THRESHOLD
    )


def select_representative_collection(
    collections: list[BranchCollection],
    *,
    current_level: int,
) -> BranchCollection:
    if not collections:
        raise ValueError("cannot select a representative from an empty collection")
    return min(
        collections,
        key=lambda collection: (
            collection.task.level != current_level,
            collection.replay,
            collection.exclusion_reason is not None,
            not collection.informative,
            collection.task.task_id,
        ),
    )


def validate_resume_state(
    state: dict[str, Any],
    *,
    experiment_seed: int,
    training_configuration: dict[str, Any],
    resume_started_at_unix_seconds: float,
    workload_attempt: int,
    model_revision: str,
) -> tuple[float, int, float, float]:
    if (
        state.get("seed") != experiment_seed
        or state.get("workload_revision") != WORKLOAD_REVISION
        or state.get("model_revision") != model_revision
        or state.get("objective_id") != OBJECTIVE_ID
        or state.get("training_configuration") != training_configuration
    ):
        raise RuntimeError("training checkpoint identity does not match this workload")
    if "cumulative_elapsed_seconds" not in state:
        raise RuntimeError("training checkpoint elapsed budget is missing")
    prior_elapsed_seconds = float(state["cumulative_elapsed_seconds"])
    if not math.isfinite(prior_elapsed_seconds) or prior_elapsed_seconds < 0:
        raise RuntimeError("training checkpoint elapsed budget is invalid")
    checkpointed_at = float(state.get("checkpointed_at_unix_seconds", math.nan))
    if not math.isfinite(checkpointed_at) or checkpointed_at < 0:
        raise RuntimeError("training checkpoint wall time is invalid")
    raw_resume_gap_seconds = resume_started_at_unix_seconds - checkpointed_at
    maximum_resume_gap_seconds = float(training_configuration["maximum_resume_gap_seconds"])
    applied_resume_gap_seconds = min(
        maximum_resume_gap_seconds,
        max(0.0, raw_resume_gap_seconds),
    )
    prior_elapsed_seconds += applied_resume_gap_seconds
    attempt_count = int(state.get("attempt_count", 0)) + 1
    if attempt_count != workload_attempt or workload_attempt != 2:
        raise RuntimeError("training checkpoint attempt count does not match the runner")
    return (
        prior_elapsed_seconds,
        attempt_count,
        raw_resume_gap_seconds,
        applied_resume_gap_seconds,
    )


def self_test() -> dict[str, Any]:
    unequal = sibling_advantages([1.0, 0.0, 0.0, 0.0])
    if not unequal[0] > 0 or not all(value < 0 for value in unequal[1:]):
        raise AssertionError("leave-one-out sibling advantages have the wrong sign")
    if abs(sum(unequal)) > 1e-7:
        raise AssertionError("sibling advantages are not centered")
    if sibling_advantages([0.5] * BRANCH_WIDTH) != [0.0] * BRANCH_WIDTH:
        raise AssertionError("equal sibling returns must produce exact zero advantage")
    if wilson_interval(4, 4)[0] >= 1.0:
        raise AssertionError("finite perfect samples must retain statistical uncertainty")

    task = make_task(2, seed=DEFAULT_SEED)
    continuation = teacher_continuation_actions(task)
    calls = 0

    def scripted_sample(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        if calls == 0:
            action = diagnostic_actions(task)[0]
        elif calls == 1:
            action = diagnostic_actions(task)[1]
        else:
            continuation_index = (calls - 2) // BRANCH_WIDTH
            action = continuation[min(continuation_index, len(continuation) - 1)]
        calls += 1
        return GeneratedAction(response=encode_action(action))

    collection = collect_branch_group(
        task,
        scripted_sample,
        stochastic=False,
        sampling_seed=DEFAULT_SEED,
    )
    if collection.snapshot is None or len(collection.siblings) != BRANCH_WIDTH:
        raise AssertionError("collector did not restore four siblings")
    if collection.solved_siblings != BRANCH_WIDTH:
        raise AssertionError("scripted sibling continuations did not solve")
    if collection.advantages != [0.0] * BRANCH_WIDTH:
        raise AssertionError("identical sibling outcomes produced a policy signal")
    serialized = serialize_branch_group(collection, update=1)
    if len(serialized["shared_prefix"]["steps"]) != PREFIX_ACCEPTED_ACTIONS or any(
        not sibling["steps"] for sibling in serialized["siblings"]
    ):
        raise AssertionError("serialized lineage omitted multi-step trajectory evidence")
    result = {
        "self_test_passed": True,
        "workload_revision": WORKLOAD_REVISION,
        "branch_width": BRANCH_WIDTH,
        "shared_prefix_actions": PREFIX_ACCEPTED_ACTIONS,
        "sibling_steps": [len(sibling.steps) for sibling in collection.siblings],
        "environment_revision": ENVIRONMENT_REVISION,
        "structural_mirror_disclosures": STRUCTURAL_MIRROR_DISCLOSURES,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def ensure_dependencies() -> None:
    def versions_match() -> bool:
        try:
            return all(
                importlib.metadata.version(package) == version
                for package, version in DEPENDENCY_VERSIONS.items()
            )
        except importlib.metadata.PackageNotFoundError:
            return False

    if versions_match():
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *DEPENDENCIES],
        check=True,
        stdout=sys.stderr,
    )
    importlib.invalidate_caches()
    if not versions_match():
        raise RuntimeError("pinned research dependency versions were not installed exactly")


def run_experiment(runtime: RuntimeConfiguration) -> None:
    started = time.monotonic()
    attempt_started_at_unix_seconds = time.time()
    emit_progress(
        "dependency_setup",
        "Preparing model runtime.",
        runtime_configuration=runtime,
        elapsed_seconds=0,
    )
    ensure_dependencies()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for repository repair post-training")

    target_seconds = runtime.target_runtime_seconds
    experiment_seed = runtime.optimization_seed
    training_configuration = {
        "maximum_updates": runtime.maximum_updates,
        "validation_examples": runtime.validation_examples,
        "test_examples": runtime.test_examples,
        "training_tasks_per_update": runtime.training_tasks_per_update,
        "replay_tasks_per_level": runtime.replay_tasks_per_level,
        "evaluation_interval": EVALUATION_INTERVAL,
        "mastery_threshold": MASTERY_THRESHOLD,
        "mastery_windows": runtime.mastery_windows,
        "minimum_protocol_validity_rate": MINIMUM_PROTOCOL_VALIDITY_RATE,
        "learning_rate": LEARNING_RATE,
        "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
        "reference_kl_estimator": REFERENCE_KL_ESTIMATOR,
        "reference_policy": "disabled_adapter_base",
        "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
        "failed_sibling_policy_weight": 0.0,
        "advantage_standard_deviation_floor": ADVANTAGE_STANDARD_DEVIATION_FLOOR,
        "maximum_absolute_advantage": MAXIMUM_ABSOLUTE_ADVANTAGE,
        "minimum_informative_groups_per_policy_update": (
            MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE
        ),
        "policy_batching": "durable_cross_update_verified_fix_accumulation",
        "frontier_probe_offset": MAXIMUM_FRONTIER_PROBE_OFFSET,
        "training_level_allocation": "frontier_majority_with_adjacent_probes",
        "task_sampling": "validation_failure_structural_analogues",
        "maximum_consecutive_uninformative_groups": (MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS),
        "maximum_consecutive_regression_windows": (MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS),
        "maximum_recent_malformed_action_rate": (MAXIMUM_RECENT_MALFORMED_ACTION_RATE),
        "maximum_consecutive_malformed_windows": (MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS),
        "shared_prefix_sampling": "greedy",
        "sibling_sampling_temperature": SIBLING_SAMPLING_TEMPERATURE,
        "sibling_sampling_top_p": SIBLING_SAMPLING_TOP_P,
        "policy_prompt_roles": ["system", "user"],
        "learning_signal": "verified_fault_fixing_edits_from_mixed_correctness_sibling_group",
        "checkpoint_selection_window": "fixed_paired_validation",
        "checkpoint_retention_guard": "paired_active_adjacent_and_rotating_zero_regressions",
        "final_evaluation_reserve_source": "retained_checkpoint",
        "curriculum_validation_window": "rotating_disjoint",
        "reward_contract_revision": "correctness-gated-efficiency@1",
        "maximum_final_evaluation_reserve_seconds": (
            runtime.maximum_final_evaluation_reserve_seconds
        ),
        "target_runtime_seconds": target_seconds,
        "maximum_resume_gap_seconds": runtime.maximum_resume_gap_seconds,
        "validation_window_seed_stride": VALIDATION_WINDOW_SEED_STRIDE,
    }
    random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    emit_progress(
        "model_loading",
        "Loading model and LoRA adapter.",
        runtime_configuration=runtime,
        elapsed_seconds=round(time.monotonic() - started, 3),
        gpu_name=torch.cuda.get_device_name(0),
        model_id=runtime.model_id,
        validation_examples=runtime.validation_examples,
        test_examples=runtime.test_examples,
        mastery_windows=runtime.mastery_windows,
        teacher_data_used=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        runtime.model_id,
        revision=runtime.model_revision,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        runtime.model_id,
        revision=runtime.model_revision,
        dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device)
    adapter_path = os.environ.get("EQUINOX_ADAPTER_PATH")
    checkpoints_root = os.path.join(adapter_path, "checkpoints") if adapter_path else None
    latest_checkpoint_path = (
        os.path.join(checkpoints_root, "latest.json") if checkpoints_root else None
    )
    checkpoint_directory: str | None = None
    if latest_checkpoint_path and os.path.isfile(latest_checkpoint_path):
        with open(latest_checkpoint_path, encoding="utf-8") as handle:
            checkpoint_metadata = json.load(handle)
        checkpoint_name = checkpoint_metadata.get("checkpoint")
        if (
            not isinstance(checkpoint_name, str)
            or not checkpoint_name.startswith("update-")
            or "/" in checkpoint_name
        ):
            raise RuntimeError("training checkpoint pointer is invalid")
        checkpoint_directory = os.path.join(checkpoints_root, checkpoint_name)
    if checkpoint_directory:
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_directory,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=(
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ),
                task_type="CAUSAL_LM",
            ),
        )
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )
    resume_state: dict[str, Any] | None = None
    if checkpoint_directory:
        resume_state = torch.load(
            os.path.join(checkpoint_directory, "training-state.pt"),
            map_location=device,
            weights_only=False,
        )
        (
            prior_elapsed_seconds,
            attempt_count,
            raw_resume_gap_seconds,
            applied_resume_gap_seconds,
        ) = validate_resume_state(
            resume_state,
            experiment_seed=experiment_seed,
            training_configuration=training_configuration,
            resume_started_at_unix_seconds=attempt_started_at_unix_seconds,
            workload_attempt=runtime.workload_attempt,
            model_revision=runtime.model_revision,
        )
        optimizer.load_state_dict(resume_state["optimizer"])
    else:
        if runtime.workload_attempt != 1:
            raise RuntimeError("a retry requires a valid training checkpoint")
        prior_elapsed_seconds = 0.0
        attempt_count = 1
        raw_resume_gap_seconds = 0.0
        applied_resume_gap_seconds = 0.0

    def cumulative_elapsed_seconds() -> float:
        return prior_elapsed_seconds + time.monotonic() - started

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    def render_prompt(prompt: str) -> str:
        return render_action_prompt(tokenizer, prompt)

    def sample_one(prompt: str, stochastic: bool, sampling_seed: int) -> GeneratedAction:
        rendered = render_prompt(prompt)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)
        input_width = encoded.input_ids.shape[1]
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        model.eval()
        model.config.use_cache = True
        generation_options: dict[str, Any] = {
            "do_sample": stochastic,
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if stochastic:
            generation_options.update(
                temperature=SIBLING_SAMPLING_TEMPERATURE,
                top_p=SIBLING_SAMPLING_TOP_P,
            )

        class CompleteJsonObjectCriteria(StoppingCriteria):
            def __call__(
                self,
                input_ids: Any,
                scores: Any,
                **kwargs: Any,
            ) -> Any:
                del scores, kwargs
                completed = [
                    complete_json_object(
                        ACTION_RESPONSE_PREFIX
                        + tokenizer.decode(row[input_width:], skip_special_tokens=True)
                    )
                    for row in input_ids
                ]
                return torch.tensor(completed, device=input_ids.device, dtype=torch.bool)

        with torch.no_grad():
            sequence = model.generate(
                **encoded,
                **generation_options,
                stopping_criteria=StoppingCriteriaList([CompleteJsonObjectCriteria()]),
            )
        model.config.use_cache = False
        continuation = sequence[0, input_width:]
        response = (
            ACTION_RESPONSE_PREFIX
            + tokenizer.decode(continuation, skip_special_tokens=True).strip()
        )
        eos_hits = continuation.eq(tokenizer.eos_token_id)
        eos_count = eos_hits.cumsum(dim=0)
        continuation_mask = ((eos_count == 0) | (eos_hits & eos_count.eq(1))).long()
        completion_mask = torch.cat(
            (
                torch.zeros(input_width, device=device, dtype=torch.long),
                continuation_mask,
            )
        )
        attention_mask = torch.cat((encoded.attention_mask[0], continuation_mask))
        generated = GeneratedAction(
            response=response,
            input_ids=tuple(int(value) for value in sequence[0].detach().cpu().tolist()),
            attention_mask=tuple(int(value) for value in attention_mask.detach().cpu().tolist()),
            completion_mask=tuple(int(value) for value in completion_mask.detach().cpu().tolist()),
        )
        del encoded, sequence, continuation
        return generated

    def evaluate_tasks(
        tasks: list[RepairTask],
        *,
        level: int,
        seed: int,
        split: str,
        expected_examples: int | None = None,
        deadline_seconds: float | None = None,
        progress_phase: str | None = None,
    ) -> dict[str, Any]:
        expected_examples = len(tasks) if expected_examples is None else expected_examples
        outcomes = []
        for index, task in enumerate(tasks):
            if deadline_seconds is not None and cumulative_elapsed_seconds() >= deadline_seconds:
                break
            outcome = collect_greedy_trajectory(
                task,
                sample_one,
                sampling_seed=seed + index * 101,
                deadline_reached=(
                    None
                    if deadline_seconds is None
                    else lambda: cumulative_elapsed_seconds() >= deadline_seconds
                ),
            )
            if outcome is None:
                break
            outcomes.append(
                {
                    "task_id": task.task_id,
                    "semantic_task_id": task.semantic_task_id,
                    "family_ids": [fault.family_id for fault in task.faults],
                    **outcome,
                }
            )
            if progress_phase is not None:
                emit_progress(
                    progress_phase,
                    (
                        f"Evaluating {split} level {level}: "
                        f"{len(outcomes)} of {expected_examples} tasks."
                    ),
                    runtime_configuration=runtime,
                    elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
                    current_level=level,
                    evaluation_completed=len(outcomes),
                    evaluation_total=expected_examples,
                    evaluation_split=split,
                )
        successes = sum(outcome["solved"] for outcome in outcomes)
        checkpoint_successes = sum(outcome["checkpoint_reached"] for outcome in outcomes)
        total_actions = sum(outcome["actions"] for outcome in outcomes)
        malformed_actions = sum(outcome["malformed_actions"] for outcome in outcomes)
        semantic_task_ids = {outcome["semantic_task_id"] for outcome in outcomes}
        if len(semantic_task_ids) != len(outcomes):
            raise RuntimeError("evaluation task set contains duplicate semantic repairs")
        examples = len(outcomes)
        return {
            "level": level,
            "split": split,
            "seed": seed,
            "examples": examples,
            "expected_examples": expected_examples,
            "complete": examples == expected_examples,
            "distinct_semantic_examples": len(semantic_task_ids),
            "semantic_universe_size": semantic_task_universe_size(
                level,
                split,
            ),
            "exact_successes": successes,
            "exact_rate": round(successes / examples, 6) if examples else None,
            "exact_rate_95ci": wilson_interval(successes, examples) if examples else None,
            "checkpoint_successes": checkpoint_successes,
            "checkpoint_rate": (round(checkpoint_successes / examples, 6) if examples else None),
            "checkpoint_rate_95ci": (
                wilson_interval(checkpoint_successes, examples) if examples else None
            ),
            "total_actions": total_actions,
            "accepted_actions": sum(outcome["accepted_actions"] for outcome in outcomes),
            "malformed_actions": malformed_actions,
            "action_protocol_validity_rate": (
                round((total_actions - malformed_actions) / total_actions, 6)
                if total_actions
                else None
            ),
            "mean_reward": round(
                sum(outcome["reward"] for outcome in outcomes) / examples,
                6,
            )
            if examples
            else None,
            "mean_actions": round(
                sum(outcome["actions"] for outcome in outcomes) / examples,
                6,
            )
            if examples
            else None,
            "task_outcomes": outcomes,
        }

    def evaluate(
        level: int,
        count: int,
        seed: int,
        *,
        split: str,
        exclude_semantic_task_ids: frozenset[str] = frozenset(),
        progress_phase: str | None = None,
    ) -> dict[str, Any]:
        tasks = make_tasks(
            level,
            count,
            seed,
            split=split,
            exclude_semantic_task_ids=exclude_semantic_task_ids,
        )
        return evaluate_tasks(
            tasks,
            level=level,
            seed=seed,
            split=split,
            progress_phase=progress_phase,
        )

    def train_policy(
        examples: list[WeightedAction],
        denominator: int,
    ) -> tuple[float, float, float]:
        if not examples:
            return 0.0, 0.0, 0.0
        reinforce_loss_value = 0.0
        reference_kl_value = 0.0
        total_loss_value = 0.0
        for start in range(0, len(examples), TRAINING_MICROBATCH_SIZE):
            batch = examples[start : start + TRAINING_MICROBATCH_SIZE]
            width = max(len(example.generated.input_ids) for example in batch)
            input_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            completion_rows: list[list[int]] = []
            weights: list[float] = []
            for example in batch:
                padding = width - len(example.generated.input_ids)
                input_rows.append(
                    list(example.generated.input_ids) + [tokenizer.pad_token_id] * padding
                )
                attention_rows.append(list(example.generated.attention_mask) + [0] * padding)
                completion_rows.append(list(example.generated.completion_mask) + [0] * padding)
                weights.append(example.weight)
            input_ids = torch.tensor(input_rows, device=device)
            attention_mask = torch.tensor(attention_rows, device=device)
            completion_mask = torch.tensor(
                completion_rows,
                device=device,
                dtype=torch.float32,
            )
            target_ids = input_ids[:, 1:]
            target_mask = completion_mask[:, 1:]
            completion_token_count = target_mask.sum(dim=1).clamp_min(1.0)
            with torch.no_grad(), model.disable_adapter():
                reference_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                reference_logits = reference_output.logits[:, :-1].float()
                reference_token_log_probabilities = (
                    torch.log_softmax(reference_logits, dim=-1)
                    .gather(-1, target_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
            del reference_output, reference_logits
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits[:, :-1].float()
            token_log_probabilities = (
                torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            )
            sequence_log_probability = (token_log_probabilities * target_mask).sum(
                dim=1
            ) / completion_token_count
            weight_tensor = torch.tensor(weights, device=device)
            reinforce_loss = (
                -(weight_tensor.detach() * sequence_log_probability).sum() / denominator
            )
            token_reference_over_policy_log_ratio = torch.where(
                target_mask.bool(),
                reference_token_log_probabilities - token_log_probabilities,
                torch.zeros_like(token_log_probabilities),
            )
            token_reference_kl = (
                torch.expm1(token_reference_over_policy_log_ratio)
                - token_reference_over_policy_log_ratio
            )
            sequence_reference_kl = (token_reference_kl * target_mask).sum(
                dim=1
            ) / completion_token_count
            reference_kl = sequence_reference_kl.sum() / len(examples)
            total_loss = reinforce_loss + REFERENCE_KL_COEFFICIENT * reference_kl
            total_loss.backward()
            reinforce_loss_value += float(reinforce_loss.detach().item())
            reference_kl_value += float(reference_kl.detach().item())
            total_loss_value += float(total_loss.detach().item())
            del (
                output,
                logits,
                token_log_probabilities,
                reference_token_log_probabilities,
                token_reference_over_policy_log_ratio,
                token_reference_kl,
            )
        return reinforce_loss_value, reference_kl_value, total_loss_value

    def capture_trainable_state() -> dict[str, Any]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def restore_trainable_state(state: dict[str, Any]) -> None:
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if trainable.keys() != state.keys():
            raise RuntimeError("best validation checkpoint parameter identity changed")
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    emit_progress(
        "resuming" if resume_state else "baseline_evaluation",
        (
            "Restoring the validation baseline from the checkpoint."
            if resume_state
            else "Evaluating the validation split."
        ),
        runtime_configuration=runtime,
        elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
        current_level=0,
    )
    baseline_evaluation_started = time.monotonic()
    validation_baseline_by_level = (
        resume_state["validation_baseline_by_level"]
        if resume_state
        else {
            "0": evaluate(
                0,
                runtime.validation_examples,
                VALIDATION_SEED_BASE,
                split="validation",
                progress_phase="baseline_evaluation",
            )
        }
    )
    baseline_protocol_validity = validation_baseline_by_level["0"].get(
        "action_protocol_validity_rate"
    )
    baseline_failure_family_ids = sorted(
        {
            str(family_id)
            for outcome in validation_baseline_by_level["0"]["task_outcomes"]
            if not outcome["solved"]
            for family_id in outcome["family_ids"]
        }
    )
    targeted_training_family_ids = training_analogue_family_ids(baseline_failure_family_ids)
    emit_progress(
        "protocol_evaluation",
        "Action protocol evaluated on the held-out active-level baseline.",
        runtime_configuration=runtime,
        elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
        current_level=0,
        evaluation_split="validation",
        evaluation_examples=runtime.validation_examples,
        evaluation_completed=None,
        evaluation_total=None,
        action_protocol_validity_rate=baseline_protocol_validity,
        minimum_protocol_validity_rate=MINIMUM_PROTOCOL_VALIDITY_RATE,
        baseline_failure_family_ids=baseline_failure_family_ids,
        targeted_training_family_ids=targeted_training_family_ids,
    )
    if (
        baseline_protocol_validity is None
        or baseline_protocol_validity < MINIMUM_PROTOCOL_VALIDITY_RATE
    ):
        raise RuntimeError("ACTION_PROTOCOL_VALIDITY_BELOW_THRESHOLD")
    maximum_final_evaluation_actions = (
        2 * runtime.test_examples * sum(item.repair_horizon for item in COMPLEXITY_LEVELS)
    )
    if resume_state:
        final_evaluation_reserve_seconds = int(resume_state["final_evaluation_reserve_seconds"])
        final_evaluation_reserve_exceeded_ceiling = bool(
            resume_state["final_evaluation_reserve_exceeded_ceiling"]
        )
        maximum_measured_final_evaluation_reserve_seconds = int(
            resume_state["maximum_measured_final_evaluation_reserve_seconds"]
        )
    else:
        baseline_evaluation_seconds = time.monotonic() - baseline_evaluation_started
        baseline_evaluation_actions = sum(
            outcome["actions"]
            for observation in validation_baseline_by_level.values()
            for outcome in observation["task_outcomes"]
        )
        maximum_measured_final_evaluation_reserve_seconds = math.ceil(
            baseline_evaluation_seconds
            / max(1, baseline_evaluation_actions)
            * maximum_final_evaluation_actions
            * 1.5
        )
        (
            final_evaluation_reserve_seconds,
            final_evaluation_reserve_exceeded_ceiling,
        ) = bounded_final_evaluation_reserve(
            maximum_measured_final_evaluation_reserve_seconds,
            runtime.maximum_final_evaluation_reserve_seconds,
        )

    (
        initial_stop,
        initial_stop_reason,
        training_deadline_seconds,
    ) = training_stop_decision(
        elapsed_seconds=cumulative_elapsed_seconds(),
        target_seconds=target_seconds,
        final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
        final_evaluation_reserve_exceeded_ceiling=(final_evaluation_reserve_exceeded_ceiling),
    )
    if (
        not resume_state
        and initial_stop
        and initial_stop_reason != "final_evaluation_reserve_ceiling"
    ):
        raise RuntimeError("target runtime is too short for measured final-evaluation reserve")
    level = int(resume_state["level"]) if resume_state else 0
    history = list(resume_state["history"]) if resume_state else []
    previous_validation_semantic_task_ids = (
        set(resume_state["previous_validation_semantic_task_ids"]) if resume_state else set()
    )
    promotions = list(resume_state["promotions"]) if resume_state else []
    branch_snapshots = list(resume_state["branch_snapshots"]) if resume_state else []
    mastery_streak = int(resume_state["mastery_streak"]) if resume_state else 0
    updates_completed = int(resume_state["updates_completed"]) if resume_state else 0
    optimizer_update_count = int(resume_state["optimizer_update_count"]) if resume_state else 0
    policy_update_count = int(resume_state["policy_update_count"]) if resume_state else 0
    pending_training_examples = (
        list(resume_state.get("pending_training_examples", [])) if resume_state else []
    )
    pending_policy_example_count = (
        int(resume_state.get("pending_policy_example_count", 0)) if resume_state else 0
    )
    pending_informative_group_ids = (
        list(resume_state.get("pending_informative_group_ids", [])) if resume_state else []
    )
    if (
        bool(pending_training_examples) != bool(pending_informative_group_ids)
        or bool(pending_policy_example_count) != bool(pending_informative_group_ids)
        or pending_policy_example_count < 0
        or pending_policy_example_count > len(pending_training_examples)
        or len(pending_informative_group_ids) >= MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE
        or len(set(pending_informative_group_ids)) != len(pending_informative_group_ids)
        or not all(isinstance(group_id, str) for group_id in pending_informative_group_ids)
    ):
        raise RuntimeError("training checkpoint pending policy batch is invalid")
    total_task_groups = int(resume_state["total_task_groups"]) if resume_state else 0
    replay_task_groups = int(resume_state["replay_task_groups"]) if resume_state else 0
    frontier_probe_task_groups = (
        int(resume_state.get("frontier_probe_task_groups", 0)) if resume_state else 0
    )
    maximum_sampled_complexity_level = (
        int(resume_state.get("maximum_sampled_complexity_level", level)) if resume_state else level
    )
    informative_task_groups = int(resume_state["informative_task_groups"]) if resume_state else 0
    excluded_task_groups = int(resume_state["excluded_task_groups"]) if resume_state else 0
    discarded_task_groups = int(resume_state.get("discarded_task_groups", 0)) if resume_state else 0
    discarded_sampled_actions = (
        int(resume_state.get("discarded_sampled_actions", 0)) if resume_state else 0
    )
    discarded_post_branch_actions = (
        int(resume_state.get("discarded_post_branch_actions", 0)) if resume_state else 0
    )
    total_sampled_actions = int(resume_state["total_sampled_actions"]) if resume_state else 0
    total_post_branch_actions = (
        int(resume_state["total_post_branch_actions"]) if resume_state else 0
    )
    recent_action_protocol_groups = (
        [tuple(item) for item in resume_state.get("recent_action_protocol_groups", [])]
        if resume_state
        else []
    )
    total_malformed_actions = (
        int(resume_state.get("total_malformed_actions", 0)) if resume_state else 0
    )
    consecutive_uninformative_groups = (
        int(resume_state.get("consecutive_uninformative_groups", 0)) if resume_state else 0
    )
    consecutive_regression_windows = (
        int(resume_state.get("consecutive_regression_windows", 0)) if resume_state else 0
    )
    consecutive_malformed_windows = (
        int(resume_state.get("consecutive_malformed_windows", 0)) if resume_state else 0
    )
    training_complete, stop_reason, first_update = training_loop_entry(
        resume_state,
        updates_completed=updates_completed,
        maximum_updates=runtime.maximum_updates,
    )
    persist_initial_terminal_checkpoint = initial_stop and not training_complete
    if persist_initial_terminal_checkpoint:
        training_complete = True
        stop_reason = initial_stop_reason or "final_evaluation_reserve"
        first_update = runtime.maximum_updates + 1
    last_observation = (
        resume_state["last_observation"] if resume_state else validation_baseline_by_level["0"]
    )
    best_validation = (
        dict(resume_state["best_validation"])
        if resume_state
        else {
            "update": 0,
            "level": 0,
            "exact_successes": int(last_observation["exact_successes"]),
            "exact_rate": last_observation["exact_rate"],
            "exact_rate_95ci": last_observation["exact_rate_95ci"],
            "checkpoint_rate": last_observation["checkpoint_rate"],
            "checkpoint_rate_95ci": last_observation["checkpoint_rate_95ci"],
            "fixed_guard_levels": [0],
            "fixed_guard_net_improved": 0,
            "fixed_guard_regressions": 0,
            "rotating_guard_net_improved": 0,
            "rotating_guard_regressions": 0,
            "guard_net_improved": 0,
            "guard_regressions": 0,
            "source": "active_level_baseline",
        }
    )
    best_trainable_state = (
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in resume_state["best_trainable_state"].items()
        }
        if resume_state
        else capture_trainable_state()
    )
    rollback_applied = bool(resume_state.get("rollback_applied", False)) if resume_state else False
    if checkpoints_root and not resume_state:
        persist_named_adapter(
            checkpoints_root=checkpoints_root,
            name="best-validation",
            metadata={
                "kind": "best_validation",
                "update": 0,
                "level": 0,
                "exact_successes": best_validation["exact_successes"],
            },
            save_adapter=lambda target: model.save_pretrained(
                target,
                safe_serialization=True,
            ),
        )
    if resume_state:
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])

    def persist_training_checkpoint(update: int) -> None:
        if not checkpoints_root or not latest_checkpoint_path:
            return
        checkpoint_name = f"update-{update:04d}"
        state = {
            "schema_version": 1,
            "workload_revision": WORKLOAD_REVISION,
            "model_revision": runtime.model_revision,
            "objective_id": OBJECTIVE_ID,
            "seed": experiment_seed,
            "training_configuration": training_configuration,
            "final_evaluation_reserve_seconds": final_evaluation_reserve_seconds,
            "final_evaluation_reserve_exceeded_ceiling": (
                final_evaluation_reserve_exceeded_ceiling
            ),
            "maximum_measured_final_evaluation_reserve_seconds": (
                maximum_measured_final_evaluation_reserve_seconds
            ),
            "cumulative_elapsed_seconds": cumulative_elapsed_seconds(),
            "checkpointed_at_unix_seconds": time.time(),
            "attempt_count": attempt_count,
            "raw_resume_gap_seconds": raw_resume_gap_seconds,
            "applied_resume_gap_seconds": applied_resume_gap_seconds,
            "training_complete": training_complete,
            "stop_reason": stop_reason,
            "updates_completed": updates_completed,
            "level": level,
            "history": history,
            "previous_validation_semantic_task_ids": sorted(previous_validation_semantic_task_ids),
            "promotions": promotions,
            "branch_snapshots": branch_snapshots,
            "mastery_streak": mastery_streak,
            "optimizer_update_count": optimizer_update_count,
            "policy_update_count": policy_update_count,
            "pending_training_examples": pending_training_examples,
            "pending_policy_example_count": pending_policy_example_count,
            "pending_informative_group_ids": pending_informative_group_ids,
            "total_task_groups": total_task_groups,
            "replay_task_groups": replay_task_groups,
            "frontier_probe_task_groups": frontier_probe_task_groups,
            "maximum_sampled_complexity_level": maximum_sampled_complexity_level,
            "informative_task_groups": informative_task_groups,
            "excluded_task_groups": excluded_task_groups,
            "discarded_task_groups": discarded_task_groups,
            "discarded_sampled_actions": discarded_sampled_actions,
            "discarded_post_branch_actions": discarded_post_branch_actions,
            "total_sampled_actions": total_sampled_actions,
            "total_post_branch_actions": total_post_branch_actions,
            "recent_action_protocol_groups": recent_action_protocol_groups,
            "total_malformed_actions": total_malformed_actions,
            "consecutive_uninformative_groups": consecutive_uninformative_groups,
            "consecutive_regression_windows": consecutive_regression_windows,
            "consecutive_malformed_windows": consecutive_malformed_windows,
            "best_validation": best_validation,
            "best_trainable_state": best_trainable_state,
            "rollback_applied": rollback_applied,
            "last_observation": last_observation,
            "validation_baseline_by_level": validation_baseline_by_level,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        persist_checkpoint(
            checkpoints_root=checkpoints_root,
            latest_checkpoint_path=latest_checkpoint_path,
            checkpoint_name=checkpoint_name,
            state=state,
            save_adapter=lambda target: model.save_pretrained(
                target,
                safe_serialization=True,
            ),
            save_state=torch.save,
        )

    if persist_initial_terminal_checkpoint:
        persist_training_checkpoint(updates_completed)

    for update in range(first_update, runtime.maximum_updates + 1):
        stop_before_collection, decision_reason, training_deadline_seconds = training_stop_decision(
            elapsed_seconds=cumulative_elapsed_seconds(),
            target_seconds=target_seconds,
            final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
            final_evaluation_reserve_exceeded_ceiling=(final_evaluation_reserve_exceeded_ceiling),
        )
        if stop_before_collection:
            stop_reason = decision_reason or "final_evaluation_reserve"
            training_complete = True
            persist_training_checkpoint(updates_completed)
            break
        training_allocation = training_level_allocation(
            level,
            runtime.training_tasks_per_update,
        )
        tasks_by_level: dict[int, list[RepairTask]] = {}
        for task_level in sorted(set(training_allocation)):
            tasks_by_level[task_level] = failure_directed_training_tasks(
                task_level,
                training_allocation.count(task_level),
                experiment_seed * 100_000 + update * 17 + task_level * 10_000_019,
                target_family_ids=targeted_training_family_ids,
            )
        current_tasks = [
            (
                task,
                ("active_frontier" if task_level == level else "adjacent_complexity_probe"),
            )
            for task_level, tasks in sorted(tasks_by_level.items())
            for task in tasks
        ]
        replay_tasks = [
            task
            for replay_level in range(level)
            for task in make_tasks(
                replay_level,
                runtime.replay_tasks_per_level,
                experiment_seed * 1_000_000 + update * 101 + replay_level,
                split="train",
            )
        ]
        task_specs = [(task, False, role) for task, role in current_tasks] + [
            (task, True, "mastered_level_replay") for task in replay_tasks
        ]
        random.Random(experiment_seed + update).shuffle(task_specs)
        collections = []
        deadline_reached_during_collection = False
        for index, (task, replay, curriculum_role) in enumerate(task_specs):
            if cumulative_elapsed_seconds() >= training_deadline_seconds:
                deadline_reached_during_collection = True
                break
            collections.append(
                collect_branch_group(
                    task,
                    sample_one,
                    stochastic=True,
                    sampling_seed=(experiment_seed * 10_000_000 + update * 100_000 + index * 1_000),
                    replay=replay,
                    curriculum_role=curriculum_role,
                    deadline_reached=lambda deadline=training_deadline_seconds: (
                        cumulative_elapsed_seconds() >= deadline
                    ),
                )
            )
        deadline_reached_during_collection = (
            deadline_reached_during_collection
            or cumulative_elapsed_seconds() >= training_deadline_seconds
        )
        if deadline_reached_during_collection:
            (
                discarded_groups,
                discarded_actions,
                discarded_post_branch_action_count,
            ) = discarded_collection_accounting(collections)
            discarded_task_groups += discarded_groups
            discarded_sampled_actions += discarded_actions
            discarded_post_branch_actions += discarded_post_branch_action_count
            total_sampled_actions += discarded_actions
            total_post_branch_actions += discarded_post_branch_action_count
            stop_reason = "final_evaluation_reserve"
            training_complete = True
            persist_training_checkpoint(updates_completed)
            break
        (
            anchored_training_examples,
            policy_training_example_count,
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            policy_signal_group_ids,
        ) = accumulated_reference_anchored_examples(
            collections,
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            minimum_informative_groups=MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE,
        )
        informative_collections = sum(collection.informative for collection in collections)
        policy_signal_accumulating = (
            bool(pending_informative_group_ids) and not policy_signal_group_ids
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        reinforce_loss, reference_kl, policy_loss = train_policy(
            anchored_training_examples,
            max(1, len(policy_signal_group_ids)),
        )
        if anchored_training_examples:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer_update_count += 1
        else:
            gradient_norm = 0.0
        if policy_training_example_count:
            policy_update_count += 1

        updates_completed = update
        group_protocol_counts = [action_protocol_counts(collection) for collection in collections]
        recent_action_protocol_groups.extend(group_protocol_counts)
        recent_action_protocol_groups = recent_action_protocol_groups[
            -runtime.validation_examples :
        ]
        total_malformed_actions += sum(malformed for _, malformed in group_protocol_counts)
        recent_protocol = recent_action_protocol_summary(
            recent_action_protocol_groups,
            maximum_groups=runtime.validation_examples,
        )
        consecutive_malformed_windows = next_malformed_action_window_streak(
            consecutive_malformed_windows,
            recent_protocol,
        )
        uninformative_group_stop = uninformative_group_limit_reached(
            consecutive_uninformative_groups,
            collections,
            maximum_consecutive_groups=MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS,
        )
        consecutive_uninformative_groups = next_uninformative_group_streak(
            consecutive_uninformative_groups,
            collections,
        )
        total_task_groups += len(collections)
        replay_task_groups += sum(collection.replay for collection in collections)
        frontier_probe_task_groups += sum(
            collection.curriculum_role == "adjacent_complexity_probe" for collection in collections
        )
        maximum_sampled_complexity_level = max(
            maximum_sampled_complexity_level,
            *(collection.task.level for collection in collections),
        )
        informative_task_groups += informative_collections
        excluded_task_groups += sum(
            collection.exclusion_reason is not None for collection in collections
        )
        total_sampled_actions += sampled_action_count(collections)
        total_post_branch_actions += post_branch_action_count(collections)
        representative_collection = select_representative_collection(
            collections,
            current_level=level,
        )
        optimizer_evidence = {
            "applied": bool(anchored_training_examples),
            "policy_signal_applied": bool(policy_training_example_count),
            "reference_anchor_applied": bool(anchored_training_examples),
            "objective_id": OBJECTIVE_ID,
            "adapter_revision": f"update-{update}",
            "learning_rate": LEARNING_RATE,
            "policy_loss": round(policy_loss, 6),
            "reinforce_loss": round(reinforce_loss, 6),
            "reference_kl": round(reference_kl, 6),
            "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
            "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "failed_sibling_policy_weight": 0.0,
            "gradient_norm": round(float(gradient_norm), 6),
            "training_examples": policy_training_example_count,
            "reference_examples": len(anchored_training_examples),
            "minimum_informative_groups": (MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE),
            "policy_signal_group_count": len(policy_signal_group_ids),
            "policy_signal_group_ids": policy_signal_group_ids,
            "pending_informative_group_count": len(pending_informative_group_ids),
            "pending_informative_group_ids": pending_informative_group_ids,
            "pending_policy_examples": pending_policy_example_count,
            "pending_training_examples": len(pending_training_examples),
            "policy_signal_suppressed_reason": (
                "ACCUMULATING_INDEPENDENT_INFORMATIVE_GROUPS"
                if policy_signal_accumulating
                else None
            ),
            "effective_batch_weight": round(
                sum(abs(example.weight) for example in anchored_training_examples),
                8,
            ),
            "informative_group_count": informative_collections,
        }
        if policy_signal_group_ids:
            for snapshot in branch_snapshots:
                if snapshot.get("task_id") in policy_signal_group_ids:
                    prior_optimizer = snapshot.get("optimizer_update")
                    if isinstance(prior_optimizer, dict):
                        prior_optimizer["policy_signal_consumed_by_update"] = update
        latest_snapshot = serialize_branch_group(
            representative_collection,
            update=update,
            optimizer_update=optimizer_evidence,
        )
        snapshots_for_update = [latest_snapshot]
        probe_collections = [
            collection
            for collection in collections
            if collection.curriculum_role == "adjacent_complexity_probe"
        ]
        if probe_collections:
            representative_probe = min(
                probe_collections,
                key=lambda collection: (
                    collection.exclusion_reason is not None,
                    not collection.informative,
                    -collection.task.level,
                    collection.task.task_id,
                ),
            )
            snapshots_for_update.append(
                serialize_branch_group(
                    representative_probe,
                    update=update,
                    optimizer_update=optimizer_evidence,
                )
            )
        branch_snapshots.extend(snapshots_for_update)
        branch_snapshots = branch_snapshots[-40:]

        emit_progress(
            "training",
            "Collecting restored continuations and updating the adapter.",
            runtime_configuration=runtime,
            update=update,
            current_level=level,
            promotion_count=len(promotions),
            active_complexity=asdict(COMPLEXITY_LEVELS[level]),
            curriculum_decision={
                "reason": "active_level_frontier_with_mastered_level_replay",
                "training_level_allocation": training_allocation,
                "frontier_probe_levels": sorted(
                    candidate_level
                    for candidate_level in set(training_allocation)
                    if candidate_level > level
                ),
                "baseline_failure_family_ids": baseline_failure_family_ids,
                "targeted_training_family_ids": targeted_training_family_ids,
                "replay_probability": round(
                    len(replay_tasks) / max(1, len(task_specs)),
                    6,
                ),
                "minimum_level": 0,
                "maximum_level": MAXIMUM_COMPLEXITY_LEVEL,
                "mastery_streak": mastery_streak,
            },
            exact_rate=last_observation["exact_rate"],
            exact_rate_95ci=last_observation["exact_rate_95ci"],
            checkpoint_rate=last_observation["checkpoint_rate"],
            checkpoint_rate_95ci=last_observation["checkpoint_rate_95ci"],
            evaluation_examples=runtime.validation_examples,
            evaluation_split="validation",
            validation_examples=runtime.validation_examples,
            test_examples=runtime.test_examples,
            mastery_windows=runtime.mastery_windows,
            mastery_streak=mastery_streak,
            teacher_data_used=False,
            baseline_failure_family_ids=baseline_failure_family_ids,
            targeted_training_family_ids=targeted_training_family_ids,
            informative_group_rate=round(
                informative_task_groups / total_task_groups,
                6,
            ),
            action_protocol_validity_rate=(
                round(
                    (total_sampled_actions - total_malformed_actions) / total_sampled_actions,
                    6,
                )
                if total_sampled_actions
                else None
            ),
            recent_malformed_action_rate=recent_protocol["malformed_rate"],
            recent_action_protocol_groups=recent_protocol["groups"],
            recent_action_protocol_window_complete=recent_protocol["window_complete"],
            consecutive_malformed_windows=consecutive_malformed_windows,
            consecutive_uninformative_groups=consecutive_uninformative_groups,
            total_sampled_actions=total_sampled_actions,
            policy_update_count=policy_update_count,
            optimizer_update_count=optimizer_update_count,
            pending_informative_group_count=len(pending_informative_group_ids),
            pending_policy_example_count=pending_policy_example_count,
            pending_training_example_count=len(pending_training_examples),
            frontier_probe_task_groups=frontier_probe_task_groups,
            maximum_sampled_complexity_level=maximum_sampled_complexity_level,
            baseline_validation={
                key: validation_baseline_by_level["0"].get(key)
                for key in (
                    "level",
                    "examples",
                    "exact_successes",
                    "exact_rate",
                    "exact_rate_95ci",
                    "checkpoint_successes",
                    "checkpoint_rate",
                    "checkpoint_rate_95ci",
                    "action_protocol_validity_rate",
                )
            },
            validation_history=lightweight_validation_history(history),
            curriculum_history=promotions,
            best_validation=best_validation,
            regression_streak=consecutive_regression_windows,
            training_remaining_seconds=round(
                max(0.0, training_deadline_seconds - cumulative_elapsed_seconds()),
                3,
            ),
            final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
            provider_remaining_seconds=round(
                max(
                    0.0,
                    target_seconds
                    + runtime.maximum_final_evaluation_reserve_seconds
                    - cumulative_elapsed_seconds(),
                ),
                3,
            ),
            latest_branch_snapshot=latest_snapshot,
            elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
            evaluation_completed=None,
            evaluation_total=None,
        )

        (
            stop_after_checkpoint,
            decision_reason,
            training_deadline_seconds,
        ) = training_stop_decision(
            elapsed_seconds=cumulative_elapsed_seconds(),
            target_seconds=target_seconds,
            final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
            final_evaluation_reserve_exceeded_ceiling=(final_evaluation_reserve_exceeded_ceiling),
        )
        if decision_reason is not None:
            stop_reason = decision_reason
        if uninformative_group_stop:
            stop_after_checkpoint = True
            stop_reason = "consecutive_uninformative_groups"
        if consecutive_malformed_windows >= MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS:
            stop_after_checkpoint = True
            stop_reason = "recent_malformed_action_rate"
        if update % EVALUATION_INTERVAL == 0 and not stop_after_checkpoint:
            if checkpoints_root:
                persist_named_adapter(
                    checkpoints_root=checkpoints_root,
                    name="prevalidation",
                    metadata={
                        "kind": "prevalidation",
                        "update": update,
                        "level": level,
                    },
                    save_adapter=lambda target: model.save_pretrained(
                        target,
                        safe_serialization=True,
                    ),
                )
            validation_seed = validation_window_seed(level, update)
            validation_tasks = make_tasks(
                level,
                runtime.validation_examples,
                validation_seed,
                split="validation",
                exclude_semantic_task_ids=frozenset(previous_validation_semantic_task_ids),
            )
            validation_semantic_task_ids = {task.semantic_task_id for task in validation_tasks}
            if not validation_semantic_task_ids.isdisjoint(previous_validation_semantic_task_ids):
                raise RuntimeError(
                    "validation generator reused a semantic task from its predecessor"
                )
            (
                stop_after_checkpoint,
                decision_reason,
                training_deadline_seconds,
            ) = training_stop_decision(
                elapsed_seconds=cumulative_elapsed_seconds(),
                target_seconds=target_seconds,
                final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
                final_evaluation_reserve_exceeded_ceiling=(
                    final_evaluation_reserve_exceeded_ceiling
                ),
            )
            if decision_reason is not None:
                stop_reason = decision_reason
            if stop_after_checkpoint:
                training_complete = True
                persist_training_checkpoint(update)
                break
            validation_started = time.monotonic()
            guard_levels = fixed_retention_guard_levels(level)
            fixed_baseline_observations: dict[str, dict[str, Any]] = {}
            fixed_candidate_observations: dict[str, dict[str, Any]] = {}
            evaluations_performed: list[dict[str, Any]] = []
            fixed_guard_complete = True
            for guard_level in guard_levels:
                guard_key = str(guard_level)
                guard_seed = checkpoint_validation_seed(guard_level)
                guard_tasks = make_tasks(
                    guard_level,
                    runtime.validation_examples,
                    guard_seed,
                    split="validation",
                )
                baseline_observation = validation_baseline_by_level.get(guard_key)
                if baseline_observation is None:
                    with model.disable_adapter():
                        baseline_observation = evaluate_tasks(
                            guard_tasks,
                            level=guard_level,
                            seed=guard_seed,
                            split="validation",
                            deadline_seconds=training_deadline_seconds,
                            progress_phase="checkpoint_baseline_evaluation",
                        )
                    evaluations_performed.append(baseline_observation)
                    if not baseline_observation["complete"]:
                        fixed_guard_complete = False
                        break
                    validation_baseline_by_level[guard_key] = baseline_observation
                fixed_baseline_observations[guard_key] = baseline_observation
                candidate_observation = evaluate_tasks(
                    guard_tasks,
                    level=guard_level,
                    seed=guard_seed,
                    split="validation",
                    deadline_seconds=training_deadline_seconds,
                    progress_phase="checkpoint_validation_evaluation",
                )
                evaluations_performed.append(candidate_observation)
                fixed_candidate_observations[guard_key] = candidate_observation
                if not candidate_observation["complete"]:
                    fixed_guard_complete = False
                    break
            checkpoint_observation = fixed_candidate_observations.get(str(level))
            if not fixed_guard_complete or checkpoint_observation is None:
                history.append(
                    {
                        "update": update,
                        **(checkpoint_observation or {}),
                        "fixed_guard_levels": guard_levels,
                        "fixed_guard_complete": False,
                        "mastery_streak": mastery_streak,
                        "mastered": False,
                        "selection_window": "paired_active_adjacent_fixed_guard",
                        "validation_elapsed_seconds": round(
                            time.monotonic() - validation_started,
                            3,
                        ),
                        "incomplete_reason": "training_deadline",
                        "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                    }
                )
                stop_reason = "final_evaluation_reserve"
                training_complete = True
                persist_training_checkpoint(update)
                break
            fixed_guard_paired_change = paired_change_summary(
                [
                    outcome
                    for guard_level in guard_levels
                    for outcome in fixed_baseline_observations[str(guard_level)]["task_outcomes"]
                ],
                [
                    outcome
                    for guard_level in guard_levels
                    for outcome in fixed_candidate_observations[str(guard_level)]["task_outcomes"]
                ],
            )
            with model.disable_adapter():
                curriculum_baseline_observation = evaluate_tasks(
                    validation_tasks,
                    level=level,
                    seed=validation_seed,
                    split="validation",
                    deadline_seconds=training_deadline_seconds,
                    progress_phase="retention_guard_baseline_evaluation",
                )
            evaluations_performed.append(curriculum_baseline_observation)
            if not curriculum_baseline_observation["complete"]:
                history.append(
                    {
                        "update": update,
                        **checkpoint_observation,
                        "curriculum_baseline_complete": False,
                        "curriculum_baseline_examples": (
                            curriculum_baseline_observation["examples"]
                        ),
                        "mastery_streak": mastery_streak,
                        "mastered": False,
                        "selection_window": "paired_active_adjacent_and_rotating_guard",
                        "validation_elapsed_seconds": round(
                            time.monotonic() - validation_started,
                            3,
                        ),
                        "incomplete_reason": "training_deadline",
                        "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                    }
                )
                stop_reason = "final_evaluation_reserve"
                training_complete = True
                persist_training_checkpoint(update)
                break
            curriculum_observation = evaluate_tasks(
                validation_tasks,
                level=level,
                seed=validation_seed,
                split="validation",
                deadline_seconds=training_deadline_seconds,
                progress_phase="validation_evaluation",
            )
            evaluations_performed.append(curriculum_observation)
            validation_elapsed_seconds = time.monotonic() - validation_started
            if not curriculum_observation["complete"]:
                history.append(
                    {
                        "update": update,
                        **checkpoint_observation,
                        "curriculum_complete": False,
                        "curriculum_examples": curriculum_observation["examples"],
                        "mastery_streak": mastery_streak,
                        "mastered": False,
                        "selection_window": "paired_active_adjacent_and_rotating_guard",
                        "validation_elapsed_seconds": round(
                            validation_elapsed_seconds,
                            3,
                        ),
                        "incomplete_reason": "training_deadline",
                        "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                    }
                )
                stop_reason = "final_evaluation_reserve"
                training_complete = True
                persist_training_checkpoint(update)
                break
            previous_validation_semantic_task_ids = validation_semantic_task_ids
            curriculum_paired_change = paired_change_summary(
                curriculum_baseline_observation["task_outcomes"],
                curriculum_observation["task_outcomes"],
            )
            observed_validation_actions = sum(
                outcome["actions"]
                for evaluation in evaluations_performed
                for outcome in evaluation["task_outcomes"]
            )
            measured_trained_policy_reserve = math.ceil(
                validation_elapsed_seconds
                / max(1, observed_validation_actions)
                * maximum_final_evaluation_actions
                * 1.5
            )
            (
                candidate_final_evaluation_reserve_seconds,
                reserve_exceeded_this_window,
            ) = bounded_final_evaluation_reserve(
                measured_trained_policy_reserve,
                runtime.maximum_final_evaluation_reserve_seconds,
            )
            retention_guard_passed = (
                fixed_guard_paired_change["regressed"] == 0
                and curriculum_paired_change["regressed"] == 0
            )
            mastered = observation_mastered(curriculum_observation) and retention_guard_passed
            mastery_streak = mastery_streak + 1 if mastered else 0
            same_fixed_guard = best_validation.get("fixed_guard_levels", [0]) == guard_levels
            (
                candidate_is_best,
                best_fixed_guard_net_improved,
                best_rotating_guard_net_improved,
                consecutive_regression_windows,
            ) = paired_retention_guard_decision(
                best_fixed_net_improved=int(
                    best_validation.get("fixed_guard_net_improved", 0) if same_fixed_guard else 0
                ),
                best_rotating_net_improved=int(
                    best_validation.get("rotating_guard_net_improved", 0) if same_fixed_guard else 0
                ),
                fixed_change=fixed_guard_paired_change,
                rotating_change=curriculum_paired_change,
                consecutive_regressions=consecutive_regression_windows,
            )
            (
                final_evaluation_reserve_seconds,
                maximum_measured_final_evaluation_reserve_seconds,
                final_evaluation_reserve_exceeded_ceiling,
            ) = retained_final_evaluation_reserve(
                current_seconds=final_evaluation_reserve_seconds,
                current_maximum_measured_seconds=(
                    maximum_measured_final_evaluation_reserve_seconds
                ),
                current_exceeded_ceiling=final_evaluation_reserve_exceeded_ceiling,
                candidate_measured_seconds=measured_trained_policy_reserve,
                maximum_seconds=runtime.maximum_final_evaluation_reserve_seconds,
                candidate_retained=candidate_is_best,
            )
            if candidate_is_best:
                best_validation = {
                    "update": update,
                    "level": level,
                    "exact_successes": int(checkpoint_observation["exact_successes"]),
                    "exact_rate": checkpoint_observation["exact_rate"],
                    "exact_rate_95ci": checkpoint_observation["exact_rate_95ci"],
                    "checkpoint_rate": checkpoint_observation["checkpoint_rate"],
                    "checkpoint_rate_95ci": checkpoint_observation["checkpoint_rate_95ci"],
                    "fixed_guard_levels": guard_levels,
                    "fixed_guard_net_improved": best_fixed_guard_net_improved,
                    "fixed_guard_regressions": fixed_guard_paired_change["regressed"],
                    "rotating_guard_net_improved": best_rotating_guard_net_improved,
                    "rotating_guard_regressions": curriculum_paired_change["regressed"],
                    "guard_net_improved": best_rotating_guard_net_improved,
                    "guard_regressions": (
                        fixed_guard_paired_change["regressed"]
                        + curriculum_paired_change["regressed"]
                    ),
                    "source": "paired_active_adjacent_and_rotating_retention_guard",
                }
                best_trainable_state = capture_trainable_state()
                if checkpoints_root:
                    persist_named_adapter(
                        checkpoints_root=checkpoints_root,
                        name="best-validation",
                        metadata={
                            "kind": "best_validation",
                            "update": update,
                            "level": level,
                            "exact_successes": checkpoint_observation["exact_successes"],
                        },
                        save_adapter=lambda target: model.save_pretrained(
                            target,
                            safe_serialization=True,
                        ),
                    )
            history.append(
                {
                    "update": update,
                    **checkpoint_observation,
                    "selection_window": "paired_active_adjacent_and_rotating_guard",
                    "fixed_guard_levels": guard_levels,
                    "fixed_guard_paired_change": fixed_guard_paired_change,
                    "curriculum_seed": validation_seed,
                    "curriculum_baseline_exact_successes": (
                        curriculum_baseline_observation["exact_successes"]
                    ),
                    "curriculum_baseline_exact_rate": (
                        curriculum_baseline_observation["exact_rate"]
                    ),
                    "curriculum_examples": curriculum_observation["examples"],
                    "curriculum_exact_successes": curriculum_observation["exact_successes"],
                    "curriculum_exact_rate": curriculum_observation["exact_rate"],
                    "curriculum_exact_rate_95ci": curriculum_observation["exact_rate_95ci"],
                    "curriculum_checkpoint_successes": (
                        curriculum_observation["checkpoint_successes"]
                    ),
                    "curriculum_checkpoint_rate": curriculum_observation["checkpoint_rate"],
                    "curriculum_checkpoint_rate_95ci": (
                        curriculum_observation["checkpoint_rate_95ci"]
                    ),
                    "curriculum_paired_change": curriculum_paired_change,
                    "retention_guard_passed": retention_guard_passed,
                    "policy_loss": round(policy_loss, 6),
                    "reinforce_loss": round(reinforce_loss, 6),
                    "reference_kl": round(reference_kl, 6),
                    "gradient_norm": round(float(gradient_norm), 6),
                    "informative_group_rate": round(
                        informative_task_groups / total_task_groups,
                        6,
                    ),
                    "mastery_streak": mastery_streak,
                    "mastered": mastered,
                    "regression_streak": consecutive_regression_windows,
                    "best_checkpoint": best_validation["update"],
                    "validation_elapsed_seconds": round(
                        validation_elapsed_seconds,
                        3,
                    ),
                    "final_evaluation_reserve_seconds": (final_evaluation_reserve_seconds),
                    "measured_final_evaluation_reserve_seconds": (measured_trained_policy_reserve),
                    "candidate_final_evaluation_reserve_seconds": (
                        candidate_final_evaluation_reserve_seconds
                    ),
                    "final_evaluation_reserve_exceeded_ceiling": (reserve_exceeded_this_window),
                    "checkpoint_candidate_retained": candidate_is_best,
                    "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                }
            )
            last_observation = checkpoint_observation
            if mastery_streak >= runtime.mastery_windows and level < MAXIMUM_COMPLEXITY_LEVEL:
                current_complexity = asdict(COMPLEXITY_LEVELS[level])
                next_complexity = asdict(COMPLEXITY_LEVELS[level + 1])
                promotions.append(
                    {
                        "update": update,
                        "from_level": level,
                        "to_level": level + 1,
                        "reason": "two_disjoint_mastery_windows",
                        "from_complexity": current_complexity,
                        "to_complexity": next_complexity,
                        "changed_dimensions": {
                            key: {
                                "from": current_complexity[key],
                                "to": next_complexity[key],
                            }
                            for key in current_complexity
                            if current_complexity[key] != next_complexity[key]
                        },
                        "replay_probability": round(
                            len(replay_tasks) / max(1, len(task_specs)),
                            6,
                        ),
                        "minimum_level": 0,
                        "maximum_level": MAXIMUM_COMPLEXITY_LEVEL,
                        "exact_rate": curriculum_observation["exact_rate"],
                        "exact_rate_95ci": curriculum_observation["exact_rate_95ci"],
                        "checkpoint_rate": curriculum_observation["checkpoint_rate"],
                        "checkpoint_rate_95ci": (curriculum_observation["checkpoint_rate_95ci"]),
                        "validation_examples": runtime.validation_examples,
                        "validation_seed": validation_seed,
                        "mastery_windows": mastery_streak,
                    }
                )
                level += 1
                mastery_streak = 0
            maximum_level_mastered = (
                level == MAXIMUM_COMPLEXITY_LEVEL and mastery_streak >= runtime.mastery_windows
            )
            regression_stop = (
                consecutive_regression_windows >= MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS
            )
            if regression_stop:
                stop_reason = "validation_regression"
                restore_trainable_state(best_trainable_state)
                rollback_applied = True
            (
                provider_stop_after_checkpoint,
                decision_reason,
                training_deadline_seconds,
            ) = training_stop_decision(
                elapsed_seconds=cumulative_elapsed_seconds(),
                target_seconds=target_seconds,
                final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
                final_evaluation_reserve_exceeded_ceiling=(
                    final_evaluation_reserve_exceeded_ceiling
                ),
                maximum_level_mastered=maximum_level_mastered,
            )
            stop_after_checkpoint = regression_stop or provider_stop_after_checkpoint
            if decision_reason is not None and not regression_stop:
                stop_reason = decision_reason
        training_complete = stop_after_checkpoint
        persist_training_checkpoint(update)
        if stop_after_checkpoint:
            break

    if best_validation["update"] != updates_completed:
        restore_trainable_state(best_trainable_state)
        rollback_applied = True

    emit_progress(
        "finalizing",
        "Evaluating retained levels.",
        runtime_configuration=runtime,
        update=updates_completed,
        current_level=level,
        promotion_count=len(promotions),
        best_validation=best_validation,
        rollback_applied=rollback_applied,
        stop_reason=stop_reason,
        pending_informative_group_count=len(pending_informative_group_ids),
        pending_policy_example_count=pending_policy_example_count,
        pending_training_example_count=len(pending_training_examples),
        validation_history=lightweight_validation_history(history),
        curriculum_history=promotions,
        final_evaluation_reserve_seconds=final_evaluation_reserve_seconds,
        provider_remaining_seconds=round(
            max(
                0.0,
                target_seconds
                + runtime.maximum_final_evaluation_reserve_seconds
                - cumulative_elapsed_seconds(),
            ),
            3,
        ),
        elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
    )
    final_evaluation_deadline_seconds = (
        target_seconds + runtime.maximum_final_evaluation_reserve_seconds
    )
    test_tasks_by_level = {
        str(candidate_level): make_tasks(
            candidate_level,
            runtime.test_examples,
            TEST_SEED_BASE + candidate_level * 1_000,
            split="test",
        )
        for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    }
    initial_by_level: dict[str, dict[str, Any]] = {}
    with model.disable_adapter():
        for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1):
            level_key = str(candidate_level)
            initial_by_level[level_key] = evaluate_tasks(
                test_tasks_by_level[level_key],
                level=candidate_level,
                seed=TEST_SEED_BASE + candidate_level * 1_000,
                split="test",
                expected_examples=runtime.test_examples,
                deadline_seconds=final_evaluation_deadline_seconds,
                progress_phase="baseline_test_evaluation",
            )
    final_by_level: dict[str, dict[str, Any]] = {}
    for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1):
        level_key = str(candidate_level)
        baseline_task_ids = {
            outcome["task_id"] for outcome in initial_by_level[level_key]["task_outcomes"]
        }
        paired_tasks = [
            task for task in test_tasks_by_level[level_key] if task.task_id in baseline_task_ids
        ]
        final_by_level[level_key] = evaluate_tasks(
            paired_tasks,
            level=candidate_level,
            seed=TEST_SEED_BASE + candidate_level * 1_000,
            split="test",
            expected_examples=runtime.test_examples,
            deadline_seconds=final_evaluation_deadline_seconds,
            progress_phase="final_test_evaluation",
        )
    final_evaluation_complete = all(
        observation["complete"]
        for observation in [*initial_by_level.values(), *final_by_level.values()]
    )
    initial_reward, final_reward, reward_gain = evaluation_reward_summary(
        initial_by_level,
        final_by_level,
        complete=final_evaluation_complete,
    )
    initial_outcomes = [
        outcome
        for observation in initial_by_level.values()
        for outcome in observation["task_outcomes"]
    ]
    final_outcomes = [
        outcome
        for observation in final_by_level.values()
        for outcome in observation["task_outcomes"]
    ]
    paired_task_ids = {outcome["task_id"] for outcome in initial_outcomes} & {
        outcome["task_id"] for outcome in final_outcomes
    }
    if paired_task_ids:
        paired_test_change = paired_change_summary(
            [outcome for outcome in initial_outcomes if outcome["task_id"] in paired_task_ids],
            [outcome for outcome in final_outcomes if outcome["task_id"] in paired_task_ids],
        )
    else:
        paired_test_change = {
            "examples": 0,
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
            "net_improved": 0,
            "mcnemar_exact_p_value": 1.0,
        }
    paired_test_change_by_level = {
        level_key: paired_change_summary(
            initial_by_level[level_key]["task_outcomes"],
            final_by_level[level_key]["task_outcomes"],
        )
        for level_key in initial_by_level
        if initial_by_level[level_key]["task_outcomes"]
        and {outcome["task_id"] for outcome in initial_by_level[level_key]["task_outcomes"]}
        == {outcome["task_id"] for outcome in final_by_level[level_key]["task_outcomes"]}
    }
    initial_outcomes_by_id = {outcome["task_id"]: outcome for outcome in initial_outcomes}
    final_outcomes_by_id = {outcome["task_id"]: outcome for outcome in final_outcomes}
    paired_test_transitions = [
        {
            "task_id": task_id,
            "semantic_task_id": initial_outcomes_by_id[task_id]["semantic_task_id"],
            "family_ids": initial_outcomes_by_id[task_id]["family_ids"],
            "initial_solved": bool(initial_outcomes_by_id[task_id]["solved"]),
            "final_solved": bool(final_outcomes_by_id[task_id]["solved"]),
        }
        for task_id in sorted(paired_task_ids)
        if bool(initial_outcomes_by_id[task_id]["solved"])
        != bool(final_outcomes_by_id[task_id]["solved"])
    ]
    mastered_level_count = level + int(stop_reason == "maximum_level_mastered")
    retention_passed = final_evaluation_complete and all(
        observation_mastered(final_by_level[str(candidate_level)])
        for candidate_level in range(mastered_level_count)
    )
    restored_branching_observed = any(
        snapshot["checkpoint"] is not None
        and len(snapshot["siblings"]) == BRANCH_WIDTH
        and all(sibling["steps"] for sibling in snapshot["siblings"])
        for snapshot in branch_snapshots
    )
    hypothesis_passed = (
        restored_branching_observed
        and informative_task_groups > 0
        and policy_update_count > 0
        and retention_passed
        and reward_gain is not None
        and reward_gain > 0
        and paired_test_change["regressed"] == 0
        and paired_test_change["mcnemar_exact_p_value"] < 0.05
        and final_evaluation_complete
    )
    probative_post_training = (
        updates_completed >= 1 and policy_update_count >= 1 and restored_branching_observed
    )
    adapter_manifest: dict[str, Any] | None = None
    if adapter_path:
        model.save_pretrained(adapter_path, safe_serialization=True)
        files = []
        for directory, child_directories, names in os.walk(adapter_path):
            child_directories[:] = [name for name in child_directories if name != "checkpoints"]
            for name in sorted(names):
                path = os.path.join(directory, name)
                relative_path = os.path.relpath(path, adapter_path)
                if relative_path == "adapter-manifest.json":
                    continue
                with open(path, "rb") as handle:
                    payload = handle.read()
                files.append(
                    {
                        "path": relative_path,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        if not files or not any(item["size_bytes"] > 0 for item in files):
            raise RuntimeError("adapter persistence produced no durable bytes")
        adapter_manifest_content = {
            "schema_version": 1,
            "model_id": runtime.model_id,
            "model_revision": runtime.model_revision,
            "workload_revision": WORKLOAD_REVISION,
            "objective_id": OBJECTIVE_ID,
            "training_configuration": training_configuration,
            "files": files,
        }
        manifest_bytes = json.dumps(
            adapter_manifest_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        adapter_manifest = {
            **adapter_manifest_content,
            "digest": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        }
        manifest_path = os.path.join(adapter_path, "adapter-manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(adapter_manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        with open(manifest_path, encoding="utf-8") as handle:
            if json.load(handle) != adapter_manifest:
                raise RuntimeError("adapter manifest failed its read-after-write check")
    result = {
        "schema_version": 2,
        "experiment_completed": True,
        "post_training_completed": True,
        "hypothesis_passed": hypothesis_passed,
        "workload": "repository-repair-restored-continuation-post-training",
        "workload_revision": WORKLOAD_REVISION,
        "algorithm": "verified-fix-group-conditioned-policy-gradient",
        "objective_id": OBJECTIVE_ID,
        "objective_sequence_reduction": "mean_completion_token_log_probabilities",
        "reward_contract": {
            "revision": "correctness-gated-efficiency@1",
            "hidden_correctness_reward": 1.0,
            "incorrect_terminal_reward": 0.0,
            "efficiency_adjustment_bounds": [-0.05, 0.0],
            "accepted_action_penalty": 0.005,
        },
        "optimizer_contract": {
            "learning_rate": LEARNING_RATE,
            "maximum_gradient_norm": 1.0,
            "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
            "reference_kl_estimator": REFERENCE_KL_ESTIMATOR,
            "reference_policy": "disabled_adapter_base",
            "reference_anchor_scope": REFERENCE_ANCHOR_SCOPE,
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "failed_sibling_policy_weight": 0.0,
            "advantage_standard_deviation_floor": (ADVANTAGE_STANDARD_DEVIATION_FLOOR),
            "maximum_absolute_advantage": MAXIMUM_ABSOLUTE_ADVANTAGE,
            "minimum_informative_groups_per_policy_update": (
                MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE
            ),
            "policy_batching": "durable_cross_update_verified_fix_accumulation",
            "sequence_reduction": "mean_completion_token_log_probabilities",
        },
        "teacher_data_used": False,
        "branch_width": BRANCH_WIDTH,
        "static_branch_width": BRANCH_WIDTH,
        "multi_step": True,
        "restored_continuations": True,
        "prefix_gradient": False,
        "complexity_strategy": "adaptive",
        "replay_enabled": True,
        "task_domains": ["micro_repository"],
        "environment_revision": ENVIRONMENT_REVISION,
        "structural_mirror_disclosures": STRUCTURAL_MIRROR_DISCLOSURES,
        "verifier_revision": VERIFIER_REVISION,
        "action_protocol_revision": ACTION_PROTOCOL_REVISION,
        "snapshot_fidelity": "logical_restore",
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "seed": experiment_seed,
        "determinism": {
            "cuda_seeded_all_devices": True,
            "deterministic_algorithms": "warn_on_unavailable_kernel",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
        },
        "mastery_threshold": MASTERY_THRESHOLD,
        "mastery_windows": runtime.mastery_windows,
        "maximum_complexity_level": MAXIMUM_COMPLEXITY_LEVEL,
        "complexity_levels": [asdict(item) for item in COMPLEXITY_LEVELS],
        "validation_examples": runtime.validation_examples,
        "test_examples": runtime.test_examples,
        "evaluation_interval": EVALUATION_INTERVAL,
        "evaluation_interval_method": "wilson-score-95",
        "training_tasks_per_update": runtime.training_tasks_per_update,
        "replay_tasks_per_level": runtime.replay_tasks_per_level,
        "training_configuration": training_configuration,
        "reached_complexity_level": level,
        "maximum_sampled_complexity_level": maximum_sampled_complexity_level,
        "promotion_count": len(promotions),
        "promotions": promotions,
        "best_validation": best_validation,
        "rollback_applied": rollback_applied,
        "validation_regression_limit": MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS,
        "maximum_consecutive_uninformative_groups": (MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS),
        "maximum_recent_malformed_action_rate": (MAXIMUM_RECENT_MALFORMED_ACTION_RATE),
        "maximum_consecutive_malformed_windows": (MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS),
        "initial_reward": round(initial_reward, 6) if initial_reward is not None else None,
        "final_reward": round(final_reward, 6) if final_reward is not None else None,
        "reward_gain": round(reward_gain, 6) if reward_gain is not None else None,
        "initial_by_level": initial_by_level,
        "validation_baseline_by_level": validation_baseline_by_level,
        "baseline_failure_family_ids": baseline_failure_family_ids,
        "targeted_training_family_ids": targeted_training_family_ids,
        "final_by_level": final_by_level,
        "paired_test_change": paired_test_change,
        "paired_test_change_by_level": paired_test_change_by_level,
        "paired_test_transitions": paired_test_transitions,
        "history": history,
        "branch_snapshots": branch_snapshots,
        "updates_completed": updates_completed,
        "optimizer_update_count": optimizer_update_count,
        "policy_update_count": policy_update_count,
        "pending_informative_group_count": len(pending_informative_group_ids),
        "pending_informative_group_ids": pending_informative_group_ids,
        "pending_policy_example_count": pending_policy_example_count,
        "pending_training_example_count": len(pending_training_examples),
        "total_task_groups": total_task_groups,
        "replay_task_groups": replay_task_groups,
        "frontier_probe_task_groups": frontier_probe_task_groups,
        "informative_task_groups": informative_task_groups,
        "excluded_task_groups": excluded_task_groups,
        "discarded_task_groups": discarded_task_groups,
        "discarded_sampled_actions": discarded_sampled_actions,
        "discarded_post_branch_actions": discarded_post_branch_actions,
        "informative_group_rate": round(
            informative_task_groups / max(1, total_task_groups),
            6,
        ),
        "total_sampled_actions": total_sampled_actions,
        "total_malformed_actions": total_malformed_actions,
        "action_protocol_validity_rate": (
            round(
                (total_sampled_actions - total_malformed_actions) / total_sampled_actions,
                6,
            )
            if total_sampled_actions
            else None
        ),
        "recent_action_protocol": recent_action_protocol_summary(
            recent_action_protocol_groups,
            maximum_groups=runtime.validation_examples,
        ),
        "consecutive_uninformative_groups": consecutive_uninformative_groups,
        "consecutive_regression_windows": consecutive_regression_windows,
        "consecutive_malformed_windows": consecutive_malformed_windows,
        "total_post_branch_actions": total_post_branch_actions,
        "stop_reason": stop_reason,
        "target_runtime_seconds": target_seconds,
        "final_evaluation_reserve_seconds": final_evaluation_reserve_seconds,
        "maximum_measured_final_evaluation_reserve_seconds": (
            maximum_measured_final_evaluation_reserve_seconds
        ),
        "final_evaluation_reserve_exceeded_ceiling": (final_evaluation_reserve_exceeded_ceiling),
        "final_evaluation_complete": final_evaluation_complete,
        "final_evaluation_partial": not final_evaluation_complete,
        "final_evaluation_deadline_seconds": final_evaluation_deadline_seconds,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dependency_versions": {
            package: importlib.metadata.version(package) for package in sorted(DEPENDENCY_VERSIONS)
        },
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
        "attempt_elapsed_seconds": round(time.monotonic() - started, 3),
        "attempt_count": attempt_count,
        "raw_resume_gap_seconds": round(raw_resume_gap_seconds, 3),
        "applied_resume_gap_seconds": round(applied_resume_gap_seconds, 3),
        "crash_tail_actions_unaccounted": resumed_crash_tail_actions_unaccounted(resume_state),
        "crash_tail_cost_accounting": (
            "wall_clock_only" if resume_state is not None else "not_applicable"
        ),
        "cumulative_elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
        "adapter_persisted": adapter_manifest is not None,
        "adapter_manifest": adapter_manifest,
        "resumed_from_checkpoint": resume_state is not None,
        "training_state_checkpointed": bool(
            latest_checkpoint_path and os.path.isfile(latest_checkpoint_path)
        ),
        "optimization_seed_count": 1,
        "probative_post_training": probative_post_training,
        "claim_strength": post_training_claim_strength(
            final_evaluation_complete=final_evaluation_complete,
            probative_post_training=probative_post_training,
        ),
        "retention_passed": retention_passed,
        "restored_branching_observed": restored_branching_observed,
    }
    reached = final_by_level[str(level)]
    emit_progress(
        "finalizing",
        "Result ready for persistence and teardown.",
        runtime_configuration=runtime,
        update=updates_completed,
        current_level=level,
        promotion_count=len(promotions),
        exact_rate=reached["exact_rate"],
        exact_rate_95ci=reached["exact_rate_95ci"],
        checkpoint_rate=reached["checkpoint_rate"],
        checkpoint_rate_95ci=reached["checkpoint_rate_95ci"],
        evaluation_examples=reached["examples"],
        evaluation_completed=reached["examples"],
        evaluation_total=runtime.test_examples,
        expected_evaluation_examples=runtime.test_examples,
        evaluation_complete=reached["complete"],
        evaluation_split="test",
        validation_examples=runtime.validation_examples,
        test_examples=runtime.test_examples,
        mastery_windows=runtime.mastery_windows,
        teacher_data_used=False,
        informative_group_rate=result["informative_group_rate"],
        action_protocol_validity_rate=result["action_protocol_validity_rate"],
        recent_malformed_action_rate=result["recent_action_protocol"]["malformed_rate"],
        recent_action_protocol_groups=result["recent_action_protocol"]["groups"],
        recent_action_protocol_window_complete=result["recent_action_protocol"]["window_complete"],
        consecutive_malformed_windows=consecutive_malformed_windows,
        best_validation=best_validation,
        rollback_applied=rollback_applied,
        validation_history=lightweight_validation_history(history),
        curriculum_history=promotions,
        total_sampled_actions=total_sampled_actions,
        optimizer_update_count=optimizer_update_count,
        policy_update_count=policy_update_count,
        pending_informative_group_count=len(pending_informative_group_ids),
        pending_policy_example_count=pending_policy_example_count,
        pending_training_example_count=len(pending_training_examples),
        frontier_probe_task_groups=frontier_probe_task_groups,
        maximum_sampled_complexity_level=maximum_sampled_complexity_level,
        hypothesis_passed=hypothesis_passed,
        paired_test_change=paired_test_change,
        claim_strength=result["claim_strength"],
        elapsed_seconds=result["elapsed_seconds"],
        stop_reason=stop_reason,
        latest_branch_snapshot=branch_snapshots[-1] if branch_snapshots else None,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-configuration", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.validate_configuration:
        configure_from_environment()
        return
    runtime: RuntimeConfiguration | None = None
    try:
        runtime = configure_from_environment()
        run_experiment(runtime)
    except Exception as exc:
        raw_attempt = os.environ.get("EQUINOX_WORKLOAD_ATTEMPT", "1")
        reported_attempt = int(raw_attempt) if raw_attempt in {"1", "2"} else 1
        emit_progress(
            "failed",
            "Repository repair workload failed.",
            runtime_configuration=runtime,
            preserve_context=True,
            attempt=reported_attempt,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
