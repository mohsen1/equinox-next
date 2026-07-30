"""Transactional larger-model post-training for repository repair.

This workload collects one policy-generated diagnostic prefix, snapshots the
repository and transcript, restores four continuations, and applies
reward-conditioned credit only to fault-fixing edit tokens from verified
successful siblings in mixed-outcome groups. Candidate content is verified as
inert simulator state and is never executed. Its transaction mode remains
disabled until the receipt-authorized larger-model pilot installs its revision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
import os
import random
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
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
CHECKPOINT_AUTHENTICATION_REVISION = "launch-bound-checkpoint-manifest@1"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint-authentication.json"
ADAPTER_ROOT_DESCRIPTOR_ENVIRONMENT = "EQUINOX_ADAPTER_ROOT_FD"
CHECKPOINT_AUTHENTICATION_MECHANISM = {
    "revision": CHECKPOINT_AUTHENTICATION_REVISION,
    "manifest_schema_version": 1,
    "pointer_schema_version": 2,
    "file_validation": "dirfd-nofollow-bounded-single-link-regular@1",
    "private_materialization": "digest-verified-private-copy@1",
    "training_state_load": "torch-weights-only@1",
    "replay_binding": "runner-private-generation-and-manifest-digest@1",
}
MAXIMUM_CHECKPOINT_FILES = 32
MAXIMUM_CHECKPOINT_FILE_BYTES = 16 * 1024 * 1024 * 1024
MAXIMUM_CHECKPOINT_TOTAL_BYTES = 24 * 1024 * 1024 * 1024
MAXIMUM_PENDING_TRAINING_EXAMPLES = 8_192
MAXIMUM_GENERATED_ACTION_RESPONSE_BYTES = 1024 * 1024
DETERMINISTIC_RUNTIME_REVISION = "eager-math-sdp-deterministic@1"
ATTENTION_IMPLEMENTATION = "eager"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
WORKLOAD_REVISION = "runpod-repository-repair-transactional-retention@2"
OBJECTIVE_ID = "verified-repair-chain-transactional-retention-policy-gradient@19"
DEPENDENCIES = (
    "transformers==5.14.1",
    "peft==0.19.1",
    "accelerate==1.14.0",
)
DEPENDENCY_VERSIONS = {
    requirement.split("==", 1)[0]: requirement.split("==", 1)[1] for requirement in DEPENDENCIES
}
MAXIMUM_COMPLEXITY_LEVEL = len(COMPLEXITY_LEVELS) - 1
MINIMUM_PREFIX_ACCEPTED_ACTIONS = 2
PREFIX_MAX_ATTEMPTS = 8
EVALUATION_INTERVAL = 5
VALIDATION_SEED_BASE = 40_000
VALIDATION_WINDOW_SEED_STRIDE = 1_000_003
TEST_SEED_BASE = 190_000
FIXED_RETENTION_GUARD_MULTIPLIER = 2
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
SHARED_PREFIX_CHECKPOINT_STRATEGY = "all_fault_sources_observed"
ADVANTAGE_STANDARD_DEVIATION_FLOOR = 0.1
MAXIMUM_ABSOLUTE_ADVANTAGE = 1.0
MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE = 2
MAXIMUM_FRONTIER_PROBE_OFFSET = 2
TRAINING_LEVEL_ALLOCATION_REVISION = "retained-promotion-3-1-to-2-2@1"
TRAINING_MICROBATCH_SIZE = 2
MASTERY_THRESHOLD = 0.50
MINIMUM_PROTOCOL_VALIDITY_RATE = 0.99
MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS = 12
MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS = 4
MAXIMUM_RECENT_MALFORMED_ACTION_RATE = 0.05
MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS = 2
TRANSACTIONAL_RETENTION_REVISION = "adapter-optimizer-policy-lineage@1"
ACTIVE_RETENTION_TRANSACTION_REVISION: str | None = None
MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS = 1_024
MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES = 16 * 1_024 * 1_024
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


def maximum_branch_groups_for_configuration(
    *,
    maximum_updates: int,
    training_tasks_per_update: int,
    replay_tasks_per_level: int,
) -> int:
    if maximum_updates < 1 or training_tasks_per_update < 1 or replay_tasks_per_level < 0:
        raise ValueError("branch evidence configuration is invalid")
    return maximum_updates * (
        training_tasks_per_update + MAXIMUM_COMPLEXITY_LEVEL * replay_tasks_per_level
    )


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


def configure_from_environment(
    *,
    allow_zero_test_examples: bool = False,
) -> RuntimeConfiguration:
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
        minimum=0 if allow_zero_test_examples else 4,
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
    maximum_branch_groups = maximum_branch_groups_for_configuration(
        maximum_updates=maximum_updates,
        training_tasks_per_update=training_tasks_per_update,
        replay_tasks_per_level=replay_tasks_per_level,
    )
    if maximum_branch_groups > MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS:
        raise ValueError(
            "the configured run can exceed the bounded branch-evidence capacity "
            f"({maximum_branch_groups} > {MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS})"
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
    optimizer_input_group_id: str | None = None


def serialize_weighted_action_for_checkpoint(example: WeightedAction) -> dict[str, Any]:
    """Encode one pending optimizer example using only weights-only-safe primitives."""

    if not isinstance(example, WeightedAction):
        raise TypeError("checkpoint pending example must be a WeightedAction")
    return {
        "response": example.generated.response,
        "input_ids": list(example.generated.input_ids),
        "attention_mask": list(example.generated.attention_mask),
        "completion_mask": list(example.generated.completion_mask),
        "weight": example.weight,
        "optimizer_input_group_id": example.optimizer_input_group_id,
    }


def deserialize_weighted_action_from_checkpoint(value: object) -> WeightedAction:
    """Validate and decode one primitive-only pending optimizer example."""

    if not isinstance(value, dict) or set(value) != {
        "response",
        "input_ids",
        "attention_mask",
        "completion_mask",
        "weight",
        "optimizer_input_group_id",
    }:
        raise ValueError("checkpoint pending example field set is invalid")
    response = value["response"]
    weight = value["weight"]
    group_id = value["optimizer_input_group_id"]
    if (
        not isinstance(response, str)
        or len(response.encode("utf-8")) > MAXIMUM_GENERATED_ACTION_RESPONSE_BYTES
        or not isinstance(weight, int | float)
        or isinstance(weight, bool)
        or not math.isfinite(float(weight))
        or not 0.0 <= float(weight) <= 1.0
        or not isinstance(group_id, str)
        or not group_id
        or len(group_id) > 512
    ):
        raise ValueError("checkpoint pending example scalar values are invalid")

    token_vectors: dict[str, tuple[int, ...]] = {}
    for name in ("input_ids", "attention_mask", "completion_mask"):
        raw_vector = value[name]
        if (
            not isinstance(raw_vector, list)
            or len(raw_vector) > MAX_INPUT_TOKENS + MAX_NEW_TOKENS
            or any(type(token) is not int or token < 0 or token > 2**31 - 1 for token in raw_vector)
        ):
            raise ValueError(f"checkpoint pending example {name} is invalid")
        token_vectors[name] = tuple(raw_vector)
    if not (
        len(token_vectors["input_ids"])
        == len(token_vectors["attention_mask"])
        == len(token_vectors["completion_mask"])
    ):
        raise ValueError("checkpoint pending example token vectors do not align")
    return WeightedAction(
        generated=GeneratedAction(
            response=response,
            input_ids=token_vectors["input_ids"],
            attention_mask=token_vectors["attention_mask"],
            completion_mask=token_vectors["completion_mask"],
        ),
        weight=float(weight),
        optimizer_input_group_id=group_id,
    )


def serialize_pending_training_examples(
    examples: list[WeightedAction],
) -> list[dict[str, Any]]:
    if len(examples) > MAXIMUM_PENDING_TRAINING_EXAMPLES:
        raise ValueError("too many pending training examples for one checkpoint")
    return [serialize_weighted_action_for_checkpoint(example) for example in examples]


def deserialize_pending_training_examples(value: object) -> list[WeightedAction]:
    if not isinstance(value, list) or len(value) > MAXIMUM_PENDING_TRAINING_EXAMPLES:
        raise ValueError("checkpoint pending training examples are invalid")
    return [deserialize_weighted_action_from_checkpoint(example) for example in value]


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


def branch_checkpoint_diagnostic_actions(task: RepairTask) -> int:
    """Keep harder tasks shared until the policy has gathered proportionate evidence."""
    return min(
        PREFIX_MAX_ATTEMPTS,
        max(MINIMUM_PREFIX_ACCEPTED_ACTIONS, len(task.faults) + 1),
    )


def branch_checkpoint_fault_source_reads(
    task: RepairTask,
    prefix: RepositoryRepairEnvironment,
) -> list[str]:
    fault_paths = {fault.path for fault in task.faults}
    return sorted(
        {
            str(step.action["path"])
            for step in prefix.steps
            if step.accepted
            and step.tool == "read"
            and step.action is not None
            and step.action.get("path") in fault_paths
        }
    )


def branch_checkpoint_reached(
    task: RepairTask,
    prefix: RepositoryRepairEnvironment,
) -> bool:
    accepted_diagnostics = sum(
        step.accepted and step.tool in DIAGNOSTIC_TOOLS for step in prefix.steps
    )
    return accepted_diagnostics >= branch_checkpoint_diagnostic_actions(task) and len(
        branch_checkpoint_fault_source_reads(task, prefix)
    ) == len(task.faults)


def sampled_action_count(collections: list[BranchCollection]) -> int:
    return sum(
        len(collection.prefix.steps)
        + sum(len(sibling.steps) - len(collection.prefix.steps) for sibling in collection.siblings)
        for collection in collections
    )


def generated_action_completion_tokens(generated: GeneratedAction) -> int:
    if any(type(value) is not int or value not in {0, 1} for value in generated.completion_mask):
        raise RuntimeError("generated action completion mask is invalid")
    return sum(generated.completion_mask)


def sampled_completion_token_count(collections: list[BranchCollection]) -> int:
    return sum(
        sum(
            generated_action_completion_tokens(generated)
            for generated in collection.generated_prefix
        )
        + sum(
            generated_action_completion_tokens(generated)
            for generated_actions in collection.generated_by_sibling
            for generated in generated_actions
        )
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


def defer_uninformative_stop_for_validation(
    *,
    update: int,
    evaluation_interval: int,
    limit_reached: bool,
) -> bool:
    if update < 1 or evaluation_interval < 1:
        raise ValueError("update and evaluation interval must be positive")
    return limit_reached and update % evaluation_interval == 0


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
        "curriculum_failure_family_ids",
        "targeted_training_family_ids",
        "newly_targeted_training_family_ids",
        "curriculum_target_expansion_count",
        "priority_training_family_ids",
        "fixed_guard_levels",
        "fixed_guard_paired_change",
        "retention_guard_passed",
        "elapsed_seconds",
        "validation_elapsed_seconds",
        "final_evaluation_reserve_seconds",
        "regression_streak",
        "best_checkpoint",
        "retention_transaction_disposition",
        "attempted_policy_update_count",
        "effective_policy_update_count",
        "retained_policy_update_count",
        "retention_rollback_count",
    )
    return [{key: item[key] for key in keys if key in item} for item in history]


def discarded_collection_accounting(
    collections: list[BranchCollection],
) -> tuple[int, int, int, int]:
    return (
        len(collections),
        sampled_action_count(collections),
        post_branch_action_count(collections),
        sampled_completion_token_count(collections),
    )


@dataclass(frozen=True)
class CheckpointCommit:
    checkpoint_name: str
    generation: int
    manifest_digest: str


@dataclass(frozen=True)
class AuthenticatedCheckpoint:
    checkpoint_name: str
    generation: int
    manifest_digest: str
    private_directory: str
    source_training_state_device: int
    source_training_state_inode: int
    source_training_state_size_bytes: int
    source_training_state_digest: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _tagged_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def checkpoint_authentication_mechanism_digest() -> str:
    return _tagged_sha256(_canonical_json_bytes(CHECKPOINT_AUTHENTICATION_MECHANISM))


def _checkpoint_hmac(key: bytes, material: dict[str, Any]) -> str:
    return (
        "hmac-sha256:"
        + hmac.new(
            key,
            _canonical_json_bytes(material),
            hashlib.sha256,
        ).hexdigest()
    )


def _validate_checkpoint_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != 32:
        raise RuntimeError("checkpoint authentication key must contain 32 bytes")
    return key


def checkpoint_authentication_key_from_environment() -> bytes:
    raw = os.environ.get("EQUINOX_CHECKPOINT_AUTHENTICATION_KEY", "")
    if not re.fullmatch(r"[0-9a-f]{64}", raw):
        raise RuntimeError("checkpoint authentication key is missing or malformed")
    return bytes.fromhex(raw)


def checkpoint_authentication_identity(
    *,
    model_revision: str,
) -> dict[str, str]:
    names = {
        "run_identity": "EQUINOX_RUN_IDENTITY",
        "profile_id": "EQUINOX_LARGER_MODEL_PROFILE_ID",
        "authorization_digest": "EQUINOX_LARGER_MODEL_AUTHORIZATION_DIGEST",
        "source_head_commit": "EQUINOX_SOURCE_HEAD_COMMIT",
        "workload_bundle_digest": "EQUINOX_BUNDLE_SHA256",
    }
    identity = {key: os.environ.get(environment, "") for key, environment in names.items()}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,191}", identity["run_identity"]):
        raise RuntimeError("checkpoint run identity is missing or malformed")
    if not re.fullmatch(r"[A-Za-z0-9._@-]{1,191}", identity["profile_id"]):
        raise RuntimeError("checkpoint profile identity is missing or malformed")
    for name in ("authorization_digest", "workload_bundle_digest"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity[name]):
            raise RuntimeError(f"checkpoint {name} is missing or malformed")
    if not re.fullmatch(r"[0-9a-f]{40}", identity["source_head_commit"]):
        raise RuntimeError("checkpoint source commit is missing or malformed")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise RuntimeError("checkpoint model revision is malformed")
    return {
        **identity,
        "workload_revision": WORKLOAD_REVISION,
        "model_revision": model_revision,
        "objective_id": OBJECTIVE_ID,
    }


def expected_checkpoint_from_environment(
    workload_attempt: int,
) -> tuple[int | None, str | None]:
    raw_generation = os.environ.get("EQUINOX_EXPECTED_CHECKPOINT_GENERATION")
    raw_digest = os.environ.get("EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256")
    if workload_attempt == 1:
        if raw_generation is not None or raw_digest is not None:
            raise RuntimeError("first attempt received unexpected checkpoint replay state")
        return None, None
    if workload_attempt != 2:
        raise RuntimeError("checkpoint resume is restricted to the sole retry")
    if (
        not isinstance(raw_generation, str)
        or not raw_generation.isascii()
        or not raw_generation.isdigit()
        or int(raw_generation) < 1
        or not isinstance(raw_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", raw_digest)
    ):
        raise RuntimeError("retry checkpoint replay binding is missing or malformed")
    return int(raw_generation), raw_digest


def authenticated_adapter_root_from_environment(adapter_path: str) -> tuple[str, int]:
    """Bind the runtime adapter path to the inherited bootstrap-owned descriptor."""

    raw_descriptor = os.environ.get(ADAPTER_ROOT_DESCRIPTOR_ENVIRONMENT, "")
    if (
        not isinstance(adapter_path, str)
        or not raw_descriptor.isascii()
        or not raw_descriptor.isdigit()
    ):
        raise RuntimeError("authenticated adapter root descriptor is missing or malformed")
    descriptor = int(raw_descriptor)
    if descriptor < 3 or adapter_path != f"/proc/self/fd/{descriptor}":
        raise RuntimeError("adapter path is not the authenticated descriptor path")
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = os.stat(adapter_path)
    except OSError as error:
        raise RuntimeError("authenticated adapter root is unavailable") from error
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or descriptor_metadata.st_dev != path_metadata.st_dev
        or descriptor_metadata.st_ino != path_metadata.st_ino
    ):
        raise RuntimeError("adapter path does not match its authenticated descriptor")
    return adapter_path, descriptor


def _open_nofollow_directory(path: str) -> int:
    return os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


def _read_regular_at_with_metadata(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, os.stat_result]:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise RuntimeError("checkpoint file name is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise RuntimeError("checkpoint entry is not a bounded single-link regular file")
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed_bytes))
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise RuntimeError("checkpoint entry exceeds its byte limit")
        after = os.fstat(descriptor)
        if (
            observed_bytes != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise RuntimeError("checkpoint entry changed while it was read")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> bytes:
    payload, _ = _read_regular_at_with_metadata(
        directory_descriptor,
        name,
        maximum_bytes=maximum_bytes,
    )
    return payload


def _read_canonical_json_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    payload = _read_regular_at(
        directory_descriptor,
        name,
        maximum_bytes=maximum_bytes,
    )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"checkpoint {name} is not valid JSON") from error
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value) + b"\n":
        raise RuntimeError(f"checkpoint {name} is not canonical JSON")
    return value


def _checkpoint_file_inventory(
    directory_descriptor: int,
    *,
    require_training_state: bool = True,
) -> list[dict[str, Any]]:
    with os.scandir(directory_descriptor) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in entries:
        if entry.name == CHECKPOINT_MANIFEST_FILENAME:
            continue
        metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("checkpoint directory contains a non-regular entry")
        payload = _read_regular_at(
            directory_descriptor,
            entry.name,
            maximum_bytes=MAXIMUM_CHECKPOINT_FILE_BYTES,
        )
        total_bytes += len(payload)
        if (
            len(inventory) + 1 > MAXIMUM_CHECKPOINT_FILES
            or total_bytes > MAXIMUM_CHECKPOINT_TOTAL_BYTES
        ):
            raise RuntimeError("checkpoint file inventory exceeds its limits")
        inventory.append(
            {
                "path": entry.name,
                "size_bytes": len(payload),
                "sha256": _tagged_sha256(payload),
            }
        )
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    if require_training_state:
        required.add("training-state.pt")
    if not required.issubset({item["path"] for item in inventory}):
        raise RuntimeError("checkpoint file inventory is incomplete")
    return inventory


def _atomic_canonical_json(path: str, value: dict[str, Any]) -> None:
    pending = path + f".pending-{secrets.token_hex(16)}"
    payload = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        pending,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def _checkpoint_manifest(
    *,
    key: bytes,
    identity: dict[str, str],
    checkpoint_name: str,
    generation: int,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    material: dict[str, Any] = {
        "schema_version": 1,
        "revision": CHECKPOINT_AUTHENTICATION_REVISION,
        "identity": identity,
        "checkpoint": checkpoint_name,
        "generation": generation,
        "files": files,
    }
    return {**material, "hmac": _checkpoint_hmac(key, material)}


def _checkpoint_pointer(
    *,
    key: bytes,
    identity: dict[str, str],
    checkpoint_name: str,
    generation: int,
    manifest_digest: str,
) -> dict[str, Any]:
    material: dict[str, Any] = {
        "schema_version": 2,
        "revision": CHECKPOINT_AUTHENTICATION_REVISION,
        "run_identity": identity["run_identity"],
        "checkpoint": checkpoint_name,
        "generation": generation,
        "checkpoint_manifest_digest": manifest_digest,
    }
    return {**material, "hmac": _checkpoint_hmac(key, material)}


def _verify_hmac_object(
    value: dict[str, Any],
    *,
    key: bytes,
    description: str,
) -> None:
    observed = value.get("hmac")
    material = {name: item for name, item in value.items() if name != "hmac"}
    if (
        not isinstance(observed, str)
        or not re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", observed)
        or not hmac.compare_digest(observed, _checkpoint_hmac(key, material))
    ):
        raise RuntimeError(f"{description} authentication failed")


def _copy_checkpoint_to_private(
    directory_descriptor: int,
    *,
    files: list[dict[str, Any]],
    run_identity: str,
    manifest_digest: str,
) -> tuple[str, os.stat_result]:
    private_base = tempfile.mkdtemp(prefix=f"equinox-checkpoint-{run_identity}-")
    training_state_metadata: os.stat_result | None = None
    try:
        for expected in files:
            payload, source_metadata = _read_regular_at_with_metadata(
                directory_descriptor,
                expected["path"],
                maximum_bytes=MAXIMUM_CHECKPOINT_FILE_BYTES,
            )
            if len(payload) != expected["size_bytes"] or not hmac.compare_digest(
                _tagged_sha256(payload), expected["sha256"]
            ):
                raise RuntimeError("checkpoint changed before private materialization")
            destination = os.path.join(private_base, expected["path"])
            descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if expected["path"] == "training-state.pt":
                training_state_metadata = source_metadata
        if training_state_metadata is None:
            raise RuntimeError("checkpoint training state was not materialized")
        marker = {
            "schema_version": 1,
            "checkpoint_manifest_digest": manifest_digest,
        }
        _atomic_canonical_json(
            os.path.join(private_base, CHECKPOINT_MANIFEST_FILENAME),
            marker,
        )
        fsync_directory(private_base)
        os.chmod(private_base, 0o500)
        return private_base, training_state_metadata
    except BaseException:
        shutil.rmtree(private_base, ignore_errors=True)
        raise


def load_authenticated_checkpoint(
    *,
    checkpoints_root: str,
    authentication_key: bytes,
    authentication_identity: dict[str, str],
    expected_generation: int,
    expected_manifest_digest: str,
) -> AuthenticatedCheckpoint:
    key = _validate_checkpoint_key(authentication_key)
    if (
        type(expected_generation) is not int
        or expected_generation < 1
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_manifest_digest)
    ):
        raise RuntimeError("checkpoint replay binding is invalid")
    root_descriptor = _open_nofollow_directory(checkpoints_root)
    checkpoint_descriptor: int | None = None
    try:
        pointer = _read_canonical_json_at(
            root_descriptor,
            "latest.json",
            maximum_bytes=16 * 1024,
        )
        if set(pointer) != {
            "schema_version",
            "revision",
            "run_identity",
            "checkpoint",
            "generation",
            "checkpoint_manifest_digest",
            "hmac",
        }:
            raise RuntimeError("checkpoint pointer field set is invalid")
        _verify_hmac_object(pointer, key=key, description="checkpoint pointer")
        checkpoint_name = pointer.get("checkpoint")
        if (
            pointer.get("schema_version") != 2
            or pointer.get("revision") != CHECKPOINT_AUTHENTICATION_REVISION
            or pointer.get("run_identity") != authentication_identity["run_identity"]
            or not isinstance(checkpoint_name, str)
            or not re.fullmatch(r"update-[0-9]{4,12}", checkpoint_name)
            or pointer.get("generation") != expected_generation
            or pointer.get("checkpoint_manifest_digest") != expected_manifest_digest
        ):
            raise RuntimeError("checkpoint pointer identity or generation is invalid")
        checkpoint_descriptor = os.open(
            checkpoint_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        manifest_payload = _read_regular_at(
            checkpoint_descriptor,
            CHECKPOINT_MANIFEST_FILENAME,
            maximum_bytes=1024 * 1024,
        )
        if _tagged_sha256(manifest_payload) != expected_manifest_digest:
            raise RuntimeError("checkpoint manifest digest is invalid")
        try:
            manifest = json.loads(manifest_payload)
        except json.JSONDecodeError as error:
            raise RuntimeError("checkpoint manifest is not valid JSON") from error
        if (
            not isinstance(manifest, dict)
            or manifest_payload != _canonical_json_bytes(manifest) + b"\n"
            or set(manifest)
            != {
                "schema_version",
                "revision",
                "identity",
                "checkpoint",
                "generation",
                "files",
                "hmac",
            }
        ):
            raise RuntimeError("checkpoint manifest field set is invalid")
        _verify_hmac_object(manifest, key=key, description="checkpoint manifest")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("revision") != CHECKPOINT_AUTHENTICATION_REVISION
            or manifest.get("identity") != authentication_identity
            or manifest.get("checkpoint") != checkpoint_name
            or manifest.get("generation") != expected_generation
            or not isinstance(manifest.get("files"), list)
        ):
            raise RuntimeError("checkpoint manifest identity or generation is invalid")
        observed_files = _checkpoint_file_inventory(checkpoint_descriptor)
        if observed_files != manifest["files"]:
            raise RuntimeError("checkpoint file inventory does not match its manifest")
        private_directory, training_state_metadata = _copy_checkpoint_to_private(
            checkpoint_descriptor,
            files=observed_files,
            run_identity=authentication_identity["run_identity"],
            manifest_digest=expected_manifest_digest,
        )
        return AuthenticatedCheckpoint(
            checkpoint_name=checkpoint_name,
            generation=expected_generation,
            manifest_digest=expected_manifest_digest,
            private_directory=private_directory,
            source_training_state_device=training_state_metadata.st_dev,
            source_training_state_inode=training_state_metadata.st_ino,
            source_training_state_size_bytes=training_state_metadata.st_size,
            source_training_state_digest=next(
                item["sha256"] for item in observed_files if item["path"] == "training-state.pt"
            ),
        )
    finally:
        if checkpoint_descriptor is not None:
            os.close(checkpoint_descriptor)
        os.close(root_descriptor)


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
    generation: int,
    authentication_key: bytes,
    authentication_identity: dict[str, str],
    expected_previous_generation: int | None = None,
    expected_previous_manifest_digest: str | None = None,
    after_step: Callable[[str], None] | None = None,
) -> CheckpointCommit:
    key = _validate_checkpoint_key(authentication_key)
    if (
        not re.fullmatch(r"update-[0-9]{4,12}", checkpoint_name)
        or type(generation) is not int
        or generation < 1
        or latest_checkpoint_path != os.path.join(checkpoints_root, "latest.json")
    ):
        raise RuntimeError("checkpoint persistence identity is invalid")
    os.makedirs(checkpoints_root, exist_ok=True)
    target = os.path.join(checkpoints_root, checkpoint_name)
    previous = None
    pointer_exists = os.path.lexists(latest_checkpoint_path)
    if pointer_exists:
        if (
            expected_previous_generation is None
            or expected_previous_manifest_digest is None
            or generation != expected_previous_generation + 1
        ):
            raise RuntimeError("checkpoint persistence replay predecessor is missing")
        previous_checkpoint = load_authenticated_checkpoint(
            checkpoints_root=checkpoints_root,
            authentication_key=key,
            authentication_identity=authentication_identity,
            expected_generation=expected_previous_generation,
            expected_manifest_digest=expected_previous_manifest_digest,
        )
        previous = previous_checkpoint.checkpoint_name
        shutil.rmtree(previous_checkpoint.private_directory, ignore_errors=True)
    elif (
        expected_previous_generation is not None
        or expected_previous_manifest_digest is not None
        or generation != 1
    ):
        raise RuntimeError("checkpoint persistence predecessor is inconsistent")
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
        target_descriptor = _open_nofollow_directory(target)
        try:
            _checkpoint_file_inventory_without_state = {
                item["path"]
                for item in _checkpoint_file_inventory(
                    target_descriptor,
                    require_training_state=False,
                )
            }
        finally:
            os.close(target_descriptor)
        if not {
            "adapter_config.json",
            "adapter_model.safetensors",
        }.issubset(_checkpoint_file_inventory_without_state):
            raise RuntimeError("adapter checkpoint persistence is incomplete")
        fsync_tree(target)
        fsync_directory(checkpoints_root)
        if after_step is not None:
            after_step("adapter_saved")
    temporary_state = os.path.join(target, "training-state.pt.pending")
    if os.path.lexists(temporary_state):
        raise RuntimeError("checkpoint pending state path already exists")
    save_state(state, temporary_state)
    fsync_file(temporary_state)
    os.replace(temporary_state, os.path.join(target, "training-state.pt"))
    fsync_directory(target)
    if after_step is not None:
        after_step("state_persisted")
    target_descriptor = _open_nofollow_directory(target)
    try:
        files = _checkpoint_file_inventory(target_descriptor)
    finally:
        os.close(target_descriptor)
    manifest = _checkpoint_manifest(
        key=key,
        identity=authentication_identity,
        checkpoint_name=checkpoint_name,
        generation=generation,
        files=files,
    )
    manifest_path = os.path.join(target, CHECKPOINT_MANIFEST_FILENAME)
    _atomic_canonical_json(manifest_path, manifest)
    manifest_payload = _canonical_json_bytes(manifest) + b"\n"
    manifest_digest = _tagged_sha256(manifest_payload)
    fsync_directory(target)
    pointer = _checkpoint_pointer(
        key=key,
        identity=authentication_identity,
        checkpoint_name=checkpoint_name,
        generation=generation,
        manifest_digest=manifest_digest,
    )
    _atomic_canonical_json(latest_checkpoint_path, pointer)
    fsync_directory(checkpoints_root)
    if after_step is not None:
        after_step("pointer_persisted")
    remove_stale_checkpoint_targets(checkpoints_root, checkpoint_name)
    fsync_directory(checkpoints_root)
    return CheckpointCommit(
        checkpoint_name=checkpoint_name,
        generation=generation,
        manifest_digest=manifest_digest,
    )


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
            preserved = previous
    payload = {
        **preserved,
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


def workload_remote_error(exception: Exception) -> dict[str, str]:
    return {
        "code": "REMOTE_WORKLOAD_FAILURE",
        "message": f"{type(exception).__name__}: {exception}",
    }


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


def adaptive_frontier_probe_decision(
    current_level: int,
    prior_probe_level: int,
    collections: list[BranchCollection],
    *,
    maximum_level: int = MAXIMUM_COMPLEXITY_LEVEL,
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
        current_level + MAXIMUM_FRONTIER_PROBE_OFFSET,
    )
    if not minimum_probe_level <= prior_probe_level <= maximum_probe_level:
        raise ValueError("frontier probe level is outside the adaptive probe range")

    probe_collections = [
        collection
        for collection in collections
        if collection.curriculum_role == "adjacent_complexity_probe"
        and collection.task.level == prior_probe_level
    ]
    if not probe_collections:
        return prior_probe_level, "insufficient_probe_evidence"
    if any(collection.informative for collection in probe_collections):
        return prior_probe_level, "mixed_correctness_contrast_retained"
    if all(collection.solved_siblings == BRANCH_WIDTH for collection in probe_collections):
        next_level = min(maximum_probe_level, prior_probe_level + 1)
        return (
            next_level,
            (
                "all_siblings_solved_raise_probe"
                if next_level > prior_probe_level
                else "hardest_probe_all_solved"
            ),
        )
    if all(collection.solved_siblings == 0 for collection in probe_collections):
        next_level = max(minimum_probe_level, prior_probe_level - 1)
        return (
            next_level,
            (
                "no_siblings_solved_lower_probe"
                if next_level < prior_probe_level
                else "nearest_probe_all_failed"
            ),
        )
    return prior_probe_level, "heterogeneous_saturation_hold_probe"


def training_level_allocation_contract() -> dict[str, Any]:
    return {
        "revision": TRAINING_LEVEL_ALLOCATION_REVISION,
        "before_first_retained_promotion": {
            "active_frontier_tasks": 3,
            "nearest_probe_tasks": 1,
        },
        "after_first_retained_promotion": {
            "active_frontier_tasks": 2,
            "adaptive_probe_tasks": 2,
        },
    }


def training_level_allocation_transition_evidence(
    promotions: list[dict[str, Any]],
    *,
    observed_updates: int,
) -> dict[str, Any]:
    first_promotion_update: int | None = None
    if promotions:
        raw_update = promotions[0].get("update")
        if type(raw_update) is not int or raw_update < 1:
            raise RuntimeError("the first retained promotion update is invalid")
        first_promotion_update = raw_update
    first_post_promotion_allocation_update = (
        first_promotion_update + 1
        if first_promotion_update is not None and observed_updates >= first_promotion_update + 1
        else None
    )
    return {
        "revision": TRAINING_LEVEL_ALLOCATION_REVISION,
        "first_retained_promotion_update": first_promotion_update,
        "first_post_promotion_allocation_update": (first_post_promotion_allocation_update),
        "transition_observed": first_post_promotion_allocation_update is not None,
    }


def training_level_allocation(
    current_level: int,
    task_count: int,
    *,
    probe_level: int | None = None,
    retained_promotion_count: int,
    maximum_level: int = MAXIMUM_COMPLEXITY_LEVEL,
) -> list[int]:
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    if task_count < 1:
        raise ValueError("task count must be positive")
    if type(retained_promotion_count) is not int or retained_promotion_count < 0:
        raise ValueError("retained promotion count must be non-negative")
    if current_level == maximum_level:
        if probe_level not in (None, current_level):
            raise ValueError("the maximum curriculum level cannot have a harder probe")
        return [current_level] * task_count
    if probe_level is None:
        probe_level = current_level + 1
    if retained_promotion_count == 0 and probe_level != current_level + 1:
        raise ValueError("pre-promotion training must use the nearest probe")
    if not (
        current_level
        < probe_level
        <= min(
            maximum_level,
            current_level + MAXIMUM_FRONTIER_PROBE_OFFSET,
        )
    ):
        raise ValueError("frontier probe level is outside the adaptive probe range")
    allocation = [current_level] * task_count
    if retained_promotion_count == 0:
        probe_task_count = min(max(1, task_count // 4), task_count - 1)
    else:
        probe_task_count = min(task_count // 2, task_count - 1)
    if probe_task_count:
        allocation[-probe_task_count:] = [probe_level] * probe_task_count
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

    reachable = set(failure_family_ids)
    frontier = list(reachable)
    while frontier:
        family = frontier.pop()
        for analogue in adjacency.get(family, set()):
            if analogue in reachable:
                continue
            reachable.add(analogue)
            frontier.append(analogue)
    return sorted(reachable & training_families)


def expanded_validation_training_targets(
    current_failure_family_ids: list[str] | tuple[str, ...],
    observations: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Accumulate validation-only weaknesses and report newly reachable train families."""
    expanded_failure_family_ids = sorted(
        {
            *current_failure_family_ids,
            *(
                str(family_id)
                for observation in observations
                for outcome in observation["task_outcomes"]
                if not outcome["solved"]
                for family_id in outcome["family_ids"]
            ),
        }
    )
    previous_targets = set(training_analogue_family_ids(current_failure_family_ids))
    targeted_training_family_ids = training_analogue_family_ids(expanded_failure_family_ids)
    newly_targeted_training_family_ids = sorted(
        set(targeted_training_family_ids) - previous_targets
    )
    return (
        expanded_failure_family_ids,
        targeted_training_family_ids,
        newly_targeted_training_family_ids,
    )


def failure_directed_training_tasks(
    level: int,
    count: int,
    seed: int,
    *,
    target_family_ids: list[str] | tuple[str, ...],
    priority_family_ids: list[str] | tuple[str, ...] = (),
) -> list[RepairTask]:
    """Sample distinct train semantics that cover declared weak-family analogues."""
    if count < 1:
        raise ValueError("training task count must be positive")
    targets = sorted(set(target_family_ids))
    if not targets:
        if priority_family_ids:
            raise ValueError("priority families require a non-empty target set")
        return make_tasks(level, count, seed, split="train")

    priorities = sorted(set(priority_family_ids))
    if not set(priorities).issubset(targets):
        raise ValueError("priority families must be active training targets")
    selected_targets = priorities[:count]
    remaining_targets = [target for target in targets if target not in priorities]
    remaining_count = min(count - len(selected_targets), len(remaining_targets))
    if remaining_count:
        rotation = seed % len(remaining_targets)
        selected_targets.extend(
            remaining_targets[(rotation + index) % len(remaining_targets)]
            for index in range(remaining_count)
        )
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


def transactional_retention_enabled() -> bool:
    """Keep historical revision-30 behavior unless a named pilot opts in."""

    if ACTIVE_RETENTION_TRANSACTION_REVISION is None:
        return False
    if ACTIVE_RETENTION_TRANSACTION_REVISION != TRANSACTIONAL_RETENTION_REVISION:
        raise RuntimeError("unknown retention transaction revision")
    return True


def retention_validation_due(
    update: int,
    *,
    policy_update_applied: bool,
    evaluation_interval: int = EVALUATION_INTERVAL,
) -> bool:
    if update < 1 or evaluation_interval < 1:
        raise ValueError("update and evaluation interval must be positive")
    return update % evaluation_interval == 0 or (
        transactional_retention_enabled() and policy_update_applied
    )


def retention_transaction_disposition(
    *,
    candidate_is_best: bool,
    fixed_change: dict[str, Any],
    rotating_change: dict[str, Any],
) -> str:
    regressed = int(fixed_change["regressed"]) > 0 or int(rotating_change["regressed"]) > 0
    if regressed and candidate_is_best:
        raise RuntimeError("a regressed retention candidate cannot be retained")
    if regressed:
        return "rollback"
    return "retain" if candidate_is_best else "provisional"


def policy_lineage_counters_from_resume(
    state: dict[str, Any] | None,
    *,
    policy_update_count: int,
) -> tuple[int, int, int, int]:
    attempted = int(state.get("attempted_policy_update_count", policy_update_count)) if state else 0
    effective = int(state.get("effective_policy_update_count", policy_update_count)) if state else 0
    retained = int(state.get("retained_policy_update_count", 0)) if state else 0
    rollback_count = int(state.get("retention_rollback_count", 0)) if state else 0
    if (
        attempted != policy_update_count
        or not 0 <= retained <= effective <= attempted
        or rollback_count < 0
    ):
        raise RuntimeError("training checkpoint policy lineage counters are invalid")
    return attempted, effective, retained, rollback_count


def completion_token_counters_from_resume(
    state: dict[str, Any] | None,
) -> tuple[int, int]:
    if state is None:
        return 0, 0
    total = state.get("total_sampled_completion_tokens")
    discarded = state.get("discarded_sampled_completion_tokens")
    if (
        type(total) is not int
        or total < 0
        or type(discarded) is not int
        or discarded < 0
        or discarded > total
    ):
        raise RuntimeError("training checkpoint completion-token counters are invalid")
    return total, discarded


def validate_retained_transaction_state(
    transaction: object,
    *,
    best_validation_update: int,
    retained_policy_update_count: int,
) -> dict[str, Any]:
    if not isinstance(transaction, dict):
        raise RuntimeError("training checkpoint retained transaction is missing")
    retained_update = transaction.get("update")
    retained_effective_count = transaction.get("effective_policy_update_count")
    if (
        retained_update != best_validation_update
        or retained_effective_count != retained_policy_update_count
        or not isinstance(transaction.get("trainable_state"), dict)
        or not isinstance(transaction.get("optimizer_state"), dict)
        or not isinstance(transaction.get("retained_observation"), dict)
    ):
        raise RuntimeError("training checkpoint retained transaction lineage is invalid")
    return copy.deepcopy(transaction)


def capture_retention_transaction(
    *,
    capture_trainable_state: Callable[[], dict[str, Any]],
    optimizer: Any,
    effective_policy_update_count: int,
    update: int,
    retained_observation: dict[str, Any],
) -> dict[str, Any]:
    if effective_policy_update_count < 0 or update < 0:
        raise ValueError("retained transaction counters cannot be negative")
    return {
        "update": update,
        "trainable_state": capture_trainable_state(),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "effective_policy_update_count": effective_policy_update_count,
        "retained_observation": copy.deepcopy(retained_observation),
    }


def restore_retention_transaction(
    transaction: dict[str, Any],
    *,
    restore_trainable_state: Callable[[dict[str, Any]], None],
    optimizer: Any,
) -> tuple[int, list[WeightedAction], int, list[str], list[str], dict[str, Any]]:
    retained_update = transaction.get("update")
    retained_effective_count = transaction.get("effective_policy_update_count")
    trainable_state = transaction.get("trainable_state")
    optimizer_state = transaction.get("optimizer_state")
    retained_observation = transaction.get("retained_observation")
    if (
        type(retained_update) is not int
        or retained_update < 0
        or type(retained_effective_count) is not int
        or retained_effective_count < 0
        or not isinstance(trainable_state, dict)
        or not isinstance(optimizer_state, dict)
        or not isinstance(retained_observation, dict)
    ):
        raise RuntimeError("retained transaction state is invalid")
    restore_trainable_state(trainable_state)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    return retained_effective_count, [], 0, [], [], copy.deepcopy(retained_observation)


def annotate_retention_window_disposition(
    branch_snapshots: list[dict[str, Any]],
    *,
    update: int,
    disposition: str,
) -> None:
    if disposition not in {"retain", "provisional", "rollback"}:
        raise ValueError("unknown retention-window disposition")
    for snapshot in branch_snapshots:
        optimizer_update = snapshot.get("optimizer_update")
        if (
            isinstance(optimizer_update, dict)
            and optimizer_update.get(
                "update",
                snapshot.get("update"),
            )
            == update
        ):
            optimizer_update["retention_transaction_disposition"] = disposition


def resolve_pending_branch_snapshot_lineage(
    branch_snapshots: list[dict[str, Any]],
    *,
    status: str,
    resolution_update: int,
    reason: str,
    effective_policy_update_count: int | None = None,
    retained_policy_update_count: int | None = None,
    retention_rollback_count: int | None = None,
) -> int:
    if status not in {"retained", "rolled_back"}:
        raise ValueError("retention lineage can only resolve as retained or rolled back")
    if resolution_update < 0 or not reason:
        raise ValueError("retention lineage resolution metadata is invalid")
    resolution_counters = (
        effective_policy_update_count,
        retained_policy_update_count,
        retention_rollback_count,
    )
    if any(value is not None for value in resolution_counters) and (
        any(type(value) is not int or value < 0 for value in resolution_counters)
        or not retained_policy_update_count <= effective_policy_update_count
    ):
        raise ValueError("retention lineage resolution counters are invalid")
    resolved = 0
    for snapshot in branch_snapshots:
        optimizer_update = snapshot.get("optimizer_update")
        if not isinstance(optimizer_update, dict):
            continue
        if optimizer_update.get("retention_lineage_status") != "pending":
            continue
        optimizer_update["retention_lineage_status"] = status
        optimizer_update["retention_resolution_update"] = resolution_update
        optimizer_update["retention_resolution_reason"] = reason
        if effective_policy_update_count is not None:
            optimizer_update["effective_policy_update_count_after_resolution"] = (
                effective_policy_update_count
            )
            optimizer_update["retained_policy_update_count_after_resolution"] = (
                retained_policy_update_count
            )
            optimizer_update["retention_rollback_count_after_resolution"] = retention_rollback_count
        resolved += 1
    return resolved


def next_retained_mastery_windows(
    current: int,
    *,
    candidate_retained: bool,
    candidate_mastered: bool,
) -> int:
    if current < 0:
        raise ValueError("retained mastery windows cannot be negative")
    if not candidate_retained:
        return current
    return current + 1 if candidate_mastered else 0


def fixed_retention_guard_levels(
    current_level: int,
    *,
    maximum_level: int = MAXIMUM_COMPLEXITY_LEVEL,
) -> list[int]:
    if not 0 <= current_level <= maximum_level:
        raise ValueError("current level is outside the curriculum")
    return list(range(min(maximum_level, current_level + 1) + 1))


def fixed_retention_guard_example_count(level: int, validation_examples: int) -> int:
    if validation_examples < 1:
        raise ValueError("validation examples must be positive")
    return min(
        semantic_task_universe_size(level, "validation"),
        validation_examples * FIXED_RETENTION_GUARD_MULTIPLIER,
    )


def family_balanced_validation_tasks(
    level: int,
    count: int,
    seed: int,
) -> list[RepairTask]:
    """Select a deterministic validation suite with balanced family coverage."""
    candidates = make_tasks(
        level,
        semantic_task_universe_size(level, "validation"),
        seed,
        split="validation",
    )
    if not 1 <= count <= len(candidates):
        raise ValueError("validation task count is outside the semantic universe")

    family_counts = {template[0]: 0 for template in TEMPLATE_SPLITS["validation"]}
    selected: list[RepairTask] = []
    remaining = list(candidates)
    while len(selected) < count:
        candidate_index = max(
            range(len(remaining)),
            key=lambda index: (
                sum(family_counts[fault.family_id] == 0 for fault in remaining[index].faults),
                -sum(family_counts[fault.family_id] for fault in remaining[index].faults),
                -max(family_counts[fault.family_id] for fault in remaining[index].faults),
                -index,
            ),
        )
        task = remaining.pop(candidate_index)
        selected.append(task)
        for fault in task.faults:
            family_counts[fault.family_id] += 1
    return selected


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
    hypothesis_passed: bool | None = None,
) -> str:
    if not final_evaluation_complete:
        return "INCOMPLETE_FINAL_EVALUATION"
    if not probative_post_training:
        return "NONPROBATIVE_RESERVE_STOP"
    if hypothesis_passed is False:
        return "NEGATIVE_RESULT"
    return "EXPLORATORY_SINGLE_SEED"


def post_training_outcome_classification(
    *,
    hypothesis_passed: bool,
    final_evaluation_complete: bool,
    probative_post_training: bool,
    promotion_count: int,
) -> dict[str, Any]:
    """Separate operational completion from meaningful learning and curriculum claims."""

    if (
        type(hypothesis_passed) is not bool
        or type(final_evaluation_complete) is not bool
        or type(probative_post_training) is not bool
    ):
        raise TypeError("post-training classification flags must be booleans")
    if type(promotion_count) is not int or promotion_count < 0:
        raise ValueError("post-training promotion count must be non-negative")
    if hypothesis_passed and (not final_evaluation_complete or not probative_post_training):
        raise ValueError(
            "meaningful post-training requires a complete final evaluation that is probative"
        )
    return {
        "meaningful_post_training": hypothesis_passed,
        "post_training_outcome": (
            "MEANINGFUL_POST_TRAINING"
            if hypothesis_passed
            else (
                "NEGATIVE_EXPERIMENT_COMPLETED"
                if final_evaluation_complete and probative_post_training
                else "INCONCLUSIVE_EXPERIMENT_COMPLETED"
            )
        ),
        "dynamic_complexity_progressed": promotion_count >= 1,
    }


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
        prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if branch_checkpoint_reached(task, prefix):
            break

    if not branch_checkpoint_reached(task, prefix):
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
    for attempt in range(PREFIX_MAX_ATTEMPTS):
        if deadline_reached is not None and deadline_reached():
            return None
        generated = sample_one(
            prefix.policy_prompt("shared_prefix"),
            False,
            sampling_seed + attempt,
        )
        prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if branch_checkpoint_reached(task, prefix):
            break
    if not branch_checkpoint_reached(task, prefix):
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


def fault_fixing_edit_actions(
    collection: BranchCollection,
    sibling_index: int,
) -> list[GeneratedAction]:
    if not 0 <= sibling_index < len(collection.siblings):
        raise IndexError("sibling index is outside the branch group")
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
    return fault_fixing_edits


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
        fault_fixing_edits = fault_fixing_edit_actions(collection, sibling_index)
        if not fault_fixing_edits:
            continue
        per_action_weight = per_solved_trajectory_weight / len(fault_fixing_edits)
        examples.extend(
            WeightedAction(
                generated=generated,
                weight=per_action_weight,
                optimizer_input_group_id=collection.task.task_id,
            )
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
        (collection.task.task_id, generated)
        for collection in collections
        for generated in accepted_reference_actions(collection)
    ]
    anchored: list[WeightedAction] = []
    for group_id, generated in reference_actions:
        weights = policy_weights_by_identity.get(id(generated), [])
        anchored.append(
            WeightedAction(
                generated=generated,
                weight=weights.pop(0) if weights else 0.0,
                optimizer_input_group_id=group_id,
            )
        )
    if any(weights for weights in policy_weights_by_identity.values()):
        raise RuntimeError("policy training action is missing from the reference anchor")
    return anchored, len(policy_training_examples)


def optimizer_input_group_ids_from_examples(
    training_examples: list[WeightedAction],
) -> list[str]:
    group_ids: list[str] = []
    for example in training_examples:
        group_id = example.optimizer_input_group_id
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("optimizer input example is missing its task group identity")
        if not group_ids or group_ids[-1] != group_id:
            if group_id in group_ids:
                raise ValueError("optimizer input examples are not grouped by task identity")
            group_ids.append(group_id)
    return group_ids


def validate_pending_optimizer_batch(
    pending_training_examples: list[WeightedAction],
    pending_policy_example_count: int,
    pending_informative_group_ids: list[str],
    pending_optimizer_input_group_ids: list[str],
    *,
    minimum_informative_groups: int,
) -> None:
    if minimum_informative_groups < 1:
        raise ValueError("minimum informative groups must be positive")
    if (
        type(pending_policy_example_count) is not int
        or pending_policy_example_count < 0
        or pending_policy_example_count > len(pending_training_examples)
    ):
        raise ValueError("pending policy example count is invalid")
    if not all(isinstance(example, WeightedAction) for example in pending_training_examples):
        raise ValueError("pending training examples are invalid")
    if bool(pending_training_examples) != bool(pending_optimizer_input_group_ids):
        raise ValueError("pending training examples and optimizer input identities must agree")
    if (
        optimizer_input_group_ids_from_examples(pending_training_examples)
        != pending_optimizer_input_group_ids
    ):
        raise ValueError("pending optimizer input identities do not match the pending examples")
    if bool(pending_policy_example_count) != bool(pending_informative_group_ids):
        raise ValueError("pending policy example count and signal group identities must agree")
    if len(pending_informative_group_ids) >= minimum_informative_groups:
        raise ValueError("a complete pending policy batch should already have been applied")
    if len(set(pending_informative_group_ids)) != len(pending_informative_group_ids) or not all(
        isinstance(group_id, str) and group_id for group_id in pending_informative_group_ids
    ):
        raise ValueError("pending informative group identities must be unique non-empty strings")
    if len(set(pending_optimizer_input_group_ids)) != len(
        pending_optimizer_input_group_ids
    ) or not all(
        isinstance(group_id, str) and group_id for group_id in pending_optimizer_input_group_ids
    ):
        raise ValueError("pending optimizer input identities must be unique non-empty strings")
    if not set(pending_informative_group_ids).issubset(pending_optimizer_input_group_ids):
        raise ValueError("pending policy signal groups must be pending optimizer inputs")


def pending_optimizer_batch_from_resume(
    state: dict[str, Any] | None,
    *,
    minimum_informative_groups: int,
) -> tuple[list[WeightedAction], int, list[str], list[str]]:
    if state is None:
        return [], 0, [], []
    try:
        pending_training_examples = deserialize_pending_training_examples(
            state.get("pending_training_examples", [])
        )
        pending_policy_example_count = state.get("pending_policy_example_count", 0)
        pending_informative_group_ids = list(state.get("pending_informative_group_ids", []))
        pending_optimizer_input_group_ids = list(state.get("pending_optimizer_input_group_ids", []))
        validate_pending_optimizer_batch(
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            pending_optimizer_input_group_ids,
            minimum_informative_groups=minimum_informative_groups,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("training checkpoint pending optimizer batch is invalid") from exc
    return (
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        pending_optimizer_input_group_ids,
    )


def accumulated_reference_anchored_examples(
    collections: list[BranchCollection],
    pending_training_examples: list[WeightedAction],
    pending_policy_example_count: int,
    pending_informative_group_ids: list[str],
    pending_optimizer_input_group_ids: list[str],
    *,
    minimum_informative_groups: int,
) -> tuple[
    list[WeightedAction],
    int,
    list[WeightedAction],
    int,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    validate_pending_optimizer_batch(
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        pending_optimizer_input_group_ids,
        minimum_informative_groups=minimum_informative_groups,
    )

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
    current_optimizer_input_group_ids = optimizer_input_group_ids_from_examples(current_anchored)
    accumulated_optimizer_input_group_ids = [
        *pending_optimizer_input_group_ids,
        *current_optimizer_input_group_ids,
    ]
    if len(set(accumulated_optimizer_input_group_ids)) != len(
        accumulated_optimizer_input_group_ids
    ):
        raise RuntimeError("an optimizer input task group was sampled more than once")
    if not set(accumulated_group_ids).issubset(accumulated_optimizer_input_group_ids):
        raise RuntimeError("a policy signal group is missing accepted reference actions")
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
            [],
            current_optimizer_input_group_ids,
        )
    if len(accumulated_group_ids) < minimum_informative_groups:
        return (
            [],
            0,
            accumulated_examples,
            accumulated_policy_example_count,
            accumulated_group_ids,
            [],
            accumulated_optimizer_input_group_ids,
            [],
        )

    return (
        accumulated_examples,
        accumulated_policy_example_count,
        [],
        0,
        [],
        accumulated_group_ids,
        [],
        accumulated_optimizer_input_group_ids,
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
        credited_actions = fault_fixing_edit_actions(collection, index)
        credited_action_ids = {id(generated) for generated in credited_actions}
        solved_sibling_count = sum(
            candidate.terminal_reason == "solved" for candidate in collection.siblings
        )
        policy_signal = (
            collection.informative
            and sibling.terminal_reason == "solved"
            and bool(credited_actions)
        )
        per_action_weight = (
            1.0 / solved_sibling_count / len(credited_actions)
            if policy_signal and solved_sibling_count
            else 0.0
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
            "effective_batch_weight": (round(per_action_weight * len(credited_actions), 8)),
            "steps": [
                {
                    **serialize_step(step),
                    "policy_signal": (policy_signal and id(generated) in credited_action_ids),
                    "effective_batch_weight": (
                        round(per_action_weight, 8)
                        if policy_signal and id(generated) in credited_action_ids
                        else 0.0
                    ),
                }
                for generated, step in zip(
                    generated_actions,
                    post_branch_steps,
                    strict=True,
                )
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
            "completion_tokens": sum(
                generated_action_completion_tokens(generated)
                for generated in collection.generated_prefix
            ),
            "checkpoint_strategy": SHARED_PREFIX_CHECKPOINT_STRATEGY,
            "required_diagnostic_actions": branch_checkpoint_diagnostic_actions(collection.task),
            "required_fault_source_reads": len(collection.task.faults),
            "observed_fault_source_paths": branch_checkpoint_fault_source_reads(
                collection.task,
                collection.prefix,
            ),
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
        "sampled_completion_tokens": sampled_completion_token_count([collection]),
        "optimizer_update": optimizer_update,
        "siblings": [
            serialized_sibling(index, sibling) for index, sibling in enumerate(collection.siblings)
        ],
    }


def append_complete_branch_evidence(
    branch_snapshots: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    *,
    maximum_snapshots: int = MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
    maximum_payload_bytes: int = MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
) -> None:
    if maximum_snapshots < 1 or maximum_payload_bytes < 2:
        raise ValueError("branch evidence capacity must be positive")
    existing_ids = {
        snapshot.get("snapshot_id")
        for snapshot in branch_snapshots
        if isinstance(snapshot.get("snapshot_id"), str)
    }
    incoming_ids = [snapshot.get("snapshot_id") for snapshot in snapshots]
    if (
        any(not isinstance(snapshot_id, str) or not snapshot_id for snapshot_id in incoming_ids)
        or len(set(incoming_ids)) != len(incoming_ids)
        or existing_ids.intersection(incoming_ids)
    ):
        raise RuntimeError("branch evidence snapshot identities must be unique")
    if len(branch_snapshots) + len(snapshots) > maximum_snapshots:
        raise RuntimeError("BRANCH_EVIDENCE_CAPACITY_EXCEEDED")
    candidate_snapshots = [*branch_snapshots, *snapshots]
    if branch_evidence_payload_size_bytes(candidate_snapshots) > maximum_payload_bytes:
        raise RuntimeError("BRANCH_EVIDENCE_PAYLOAD_CAPACITY_EXCEEDED")
    branch_snapshots.extend(snapshots)


def branch_evidence_payload_size_bytes(
    branch_snapshots: list[dict[str, Any]],
) -> int:
    return len(
        json.dumps(
            branch_snapshots,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


def branch_evidence_completion_token_count(
    branch_snapshots: list[dict[str, Any]],
) -> int:
    total = 0
    for snapshot in branch_snapshots:
        sampled_tokens = snapshot.get("sampled_completion_tokens")
        shared_prefix = snapshot.get("shared_prefix")
        siblings = snapshot.get("siblings")
        if (
            type(sampled_tokens) is not int
            or sampled_tokens < 0
            or not isinstance(shared_prefix, dict)
            or type(shared_prefix.get("completion_tokens")) is not int
            or shared_prefix["completion_tokens"] < 0
            or not isinstance(siblings, list)
        ):
            raise RuntimeError("branch evidence completion-token accounting is invalid")
        reconstructed_tokens = shared_prefix["completion_tokens"]
        for sibling in siblings:
            if (
                not isinstance(sibling, dict)
                or type(sibling.get("completion_tokens")) is not int
                or sibling["completion_tokens"] < 0
            ):
                raise RuntimeError("branch evidence sibling token accounting is invalid")
            reconstructed_tokens += sibling["completion_tokens"]
        if reconstructed_tokens != sampled_tokens:
            raise RuntimeError("branch evidence completion-token total is inconsistent")
        total += sampled_tokens
    return total


def validate_sampled_completion_token_accounting(
    branch_snapshots: list[dict[str, Any]],
    *,
    total_sampled_completion_tokens: int,
    discarded_sampled_completion_tokens: int,
) -> int:
    if (
        total_sampled_completion_tokens < 0
        or discarded_sampled_completion_tokens < 0
        or discarded_sampled_completion_tokens > total_sampled_completion_tokens
    ):
        raise RuntimeError("sampled completion-token counters are invalid")
    persisted_tokens = branch_evidence_completion_token_count(branch_snapshots)
    if persisted_tokens + discarded_sampled_completion_tokens != total_sampled_completion_tokens:
        raise RuntimeError("sampled completion tokens do not reconcile with branch evidence")
    return persisted_tokens


def bind_optimizer_input_groups_to_update(
    branch_snapshots: list[dict[str, Any]],
    *,
    optimizer_input_group_ids: list[str],
    optimizer_update: dict[str, Any],
) -> None:
    declared_optimizer_input_group_ids = optimizer_update.get("optimizer_input_group_ids")
    declared_optimizer_input_group_count = optimizer_update.get("optimizer_input_group_count")
    policy_signal_group_ids = optimizer_update.get("policy_signal_group_ids")
    if (
        not isinstance(declared_optimizer_input_group_ids, list)
        or declared_optimizer_input_group_ids != optimizer_input_group_ids
        or declared_optimizer_input_group_count != len(optimizer_input_group_ids)
        or len(set(optimizer_input_group_ids)) != len(optimizer_input_group_ids)
        or not all(isinstance(group_id, str) and group_id for group_id in optimizer_input_group_ids)
    ):
        raise RuntimeError("optimizer input group identities are invalid")
    if (
        not isinstance(policy_signal_group_ids, list)
        or len(set(policy_signal_group_ids)) != len(policy_signal_group_ids)
        or not all(isinstance(group_id, str) and group_id for group_id in policy_signal_group_ids)
        or not set(policy_signal_group_ids).issubset(optimizer_input_group_ids)
    ):
        raise RuntimeError("policy signal group identities are invalid")
    if not optimizer_input_group_ids:
        return
    update = optimizer_update.get("update")
    attempted_index = optimizer_update.get("attempted_policy_update_index")
    if (
        type(update) is not int
        or update < 1
        or (
            attempted_index is not None
            and (type(attempted_index) is not int or attempted_index < 1)
        )
        or (policy_signal_group_ids and attempted_index is None)
    ):
        raise RuntimeError("optimizer input lineage is incomplete")
    snapshots_by_group_id = {
        snapshot.get("task_id"): snapshot
        for snapshot in branch_snapshots
        if snapshot.get("task_id") in optimizer_input_group_ids
    }
    if set(snapshots_by_group_id) != set(optimizer_input_group_ids):
        raise RuntimeError("optimizer input group is missing full branch evidence")
    for group_id in optimizer_input_group_ids:
        snapshot = snapshots_by_group_id[group_id]
        previous_optimizer_update = snapshot.get("optimizer_update")
        previous_consumed_update = (
            previous_optimizer_update.get("optimizer_input_consumed_by_update")
            if isinstance(previous_optimizer_update, dict)
            else None
        )
        if previous_consumed_update not in {None, update}:
            raise RuntimeError("optimizer input group was consumed by more than one update")
        snapshot["optimizer_update"] = {
            **copy.deepcopy(optimizer_update),
            "optimizer_input_consumed_by_update": update,
            **(
                {"policy_signal_consumed_by_update": update}
                if group_id in policy_signal_group_ids
                else {}
            ),
        }


def policy_update_lineage_from_branch_evidence(
    branch_snapshots: list[dict[str, Any]],
    *,
    attempted_policy_update_count: int,
    effective_policy_update_count: int,
    retained_policy_update_count: int,
) -> list[dict[str, Any]]:
    if not 0 <= retained_policy_update_count <= effective_policy_update_count:
        raise RuntimeError("policy update lineage counters are invalid")
    if not effective_policy_update_count <= attempted_policy_update_count:
        raise RuntimeError("policy update lineage counters are invalid")
    records: dict[int, dict[str, Any]] = {}
    stable_keys = (
        "update",
        "adapter_revision",
        "objective_id",
        "retention_transaction_revision",
        "retention_lineage_status",
        "retention_transaction_disposition",
        "retention_resolution_update",
        "retention_resolution_reason",
        "effective_policy_update_count_after_apply",
        "retained_policy_update_count_before_validation",
        "effective_policy_update_count_after_resolution",
        "retained_policy_update_count_after_resolution",
        "retention_rollback_count_after_resolution",
        "policy_signal_group_count",
        "policy_signal_group_ids",
        "optimizer_input_group_count",
        "optimizer_input_group_ids",
        "training_examples",
        "reference_examples",
        "policy_loss",
        "reinforce_loss",
        "reference_kl",
        "gradient_norm",
    )
    for snapshot in branch_snapshots:
        optimizer_update = snapshot.get("optimizer_update")
        if not isinstance(optimizer_update, dict):
            continue
        attempted_index = optimizer_update.get("attempted_policy_update_index")
        if attempted_index is None:
            continue
        if type(attempted_index) is not int or attempted_index < 1:
            raise RuntimeError("policy update attempt identity is invalid")
        task_id = snapshot.get("task_id")
        snapshot_id = snapshot.get("snapshot_id")
        if not isinstance(task_id, str) or not isinstance(snapshot_id, str):
            raise RuntimeError("policy update input is missing branch identity")
        if optimizer_update.get("optimizer_input_consumed_by_update") != optimizer_update.get(
            "update"
        ):
            raise RuntimeError("optimizer input consumption is missing exact update identity")
        material = {key: copy.deepcopy(optimizer_update.get(key)) for key in stable_keys}
        existing = records.get(attempted_index)
        if existing is None:
            records[attempted_index] = {
                "schema_version": 2,
                "attempted_policy_update_index": attempted_index,
                **material,
                "_reconstructed_optimizer_input_group_ids": [task_id],
                "branch_snapshot_ids": [snapshot_id],
            }
            continue
        if any(existing.get(key) != material[key] for key in stable_keys):
            raise RuntimeError("policy update input groups disagree on optimizer lineage")
        existing["_reconstructed_optimizer_input_group_ids"].append(task_id)
        existing["branch_snapshot_ids"].append(snapshot_id)

    expected_attempts = set(range(1, attempted_policy_update_count + 1))
    if set(records) != expected_attempts:
        raise RuntimeError("attempted policy update is missing full branch evidence")
    lineage = [records[index] for index in sorted(records)]
    branch_task_ids = {
        snapshot.get("task_id")
        for snapshot in branch_snapshots
        if isinstance(snapshot.get("task_id"), str)
    }
    for record in lineage:
        policy_signal_group_ids = record.get("policy_signal_group_ids")
        optimizer_input_group_ids = record.get("optimizer_input_group_ids")
        reconstructed_optimizer_input_group_ids = record.pop(
            "_reconstructed_optimizer_input_group_ids"
        )
        if (
            not isinstance(policy_signal_group_ids, list)
            or not policy_signal_group_ids
            or len(set(policy_signal_group_ids)) != len(policy_signal_group_ids)
            or not all(isinstance(group_id, str) for group_id in policy_signal_group_ids)
            or record.get("policy_signal_group_count") != len(policy_signal_group_ids)
            or not isinstance(optimizer_input_group_ids, list)
            or not optimizer_input_group_ids
            or len(set(optimizer_input_group_ids)) != len(optimizer_input_group_ids)
            or not all(isinstance(group_id, str) for group_id in optimizer_input_group_ids)
            or record.get("optimizer_input_group_count") != len(optimizer_input_group_ids)
            or reconstructed_optimizer_input_group_ids != optimizer_input_group_ids
            or not set(optimizer_input_group_ids).issubset(branch_task_ids)
            or not set(policy_signal_group_ids).issubset(optimizer_input_group_ids)
        ):
            raise RuntimeError(
                "optimizer input groups are not exactly reconstructable from branch evidence"
            )
        signal_group_ids = set(policy_signal_group_ids)
        for snapshot in branch_snapshots:
            optimizer_update = snapshot.get("optimizer_update")
            if (
                not isinstance(optimizer_update, dict)
                or optimizer_update.get("attempted_policy_update_index")
                != record["attempted_policy_update_index"]
            ):
                continue
            task_id = snapshot.get("task_id")
            signal_consumed_by_update = optimizer_update.get("policy_signal_consumed_by_update")
            if (task_id in signal_group_ids) != (signal_consumed_by_update == record.get("update")):
                raise RuntimeError("policy signal subset is not exactly bound to optimizer inputs")

    if transactional_retention_enabled():
        status_counts = {
            status: sum(record.get("retention_lineage_status") == status for record in lineage)
            for status in ("pending", "retained", "rolled_back")
        }
        if (
            status_counts["retained"] != retained_policy_update_count
            or status_counts["pending"]
            != effective_policy_update_count - retained_policy_update_count
            or status_counts["rolled_back"]
            != attempted_policy_update_count - effective_policy_update_count
        ):
            raise RuntimeError("transactional policy update dispositions disagree with counters")
    return lineage


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
        state.get("schema_version") != 5
        or state.get("branch_evidence_complete") is not True
        or state.get("branch_evidence_limit") != MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS
        or state.get("branch_evidence_payload_limit_bytes") != MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES
        or state.get("seed") != experiment_seed
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
    diagnostics = diagnostic_actions(task)
    continuation = teacher_continuation_actions(task)
    calls = 0

    def scripted_sample(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        if calls < len(diagnostics):
            action = diagnostics[calls]
        else:
            continuation_index = (calls - len(diagnostics)) // BRANCH_WIDTH
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
    expected_prefix_actions = branch_checkpoint_diagnostic_actions(task)
    if len(serialized["shared_prefix"]["steps"]) != expected_prefix_actions or any(
        not sibling["steps"] for sibling in serialized["siblings"]
    ):
        raise AssertionError("serialized lineage omitted multi-step trajectory evidence")
    result = {
        "self_test_passed": True,
        "workload_revision": WORKLOAD_REVISION,
        "branch_width": BRANCH_WIDTH,
        "shared_prefix_actions": expected_prefix_actions,
        "shared_prefix_strategy": SHARED_PREFIX_CHECKPOINT_STRATEGY,
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


def require_cublas_workspace_config() -> str:
    """Require the deterministic cuBLAS workspace policy before CUDA is used."""

    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set to "
            f"{CUBLAS_WORKSPACE_CONFIG!r} before the paid runtime starts"
        )
    return observed


def configure_deterministic_torch_runtime(torch_module: Any) -> dict[str, Any]:
    """Select and prove the fail-closed eager/math-SDP runtime."""

    cublas_workspace_config = require_cublas_workspace_config()
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.enable_flash_sdp(False)
    torch_module.backends.cuda.enable_mem_efficient_sdp(False)
    torch_module.backends.cuda.enable_math_sdp(True)
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    evidence = {
        "revision": DETERMINISTIC_RUNTIME_REVISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "cublas_workspace_config": cublas_workspace_config,
        "deterministic_algorithms": (torch_module.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": (
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "flash_sdp_enabled": torch_module.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp_enabled": (torch_module.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": torch_module.backends.cuda.math_sdp_enabled(),
        "cudnn_benchmark": torch_module.backends.cudnn.benchmark,
        "cudnn_deterministic": torch_module.backends.cudnn.deterministic,
        "tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32 or torch_module.backends.cudnn.allow_tf32
        ),
    }
    expected = {
        "revision": DETERMINISTIC_RUNTIME_REVISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "flash_sdp_enabled": False,
        "memory_efficient_sdp_enabled": False,
        "math_sdp_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
    }
    if evidence != expected:
        raise RuntimeError("the deterministic Torch runtime could not be established")
    return evidence


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

    deterministic_runtime = configure_deterministic_torch_runtime(torch)
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
        "mastery_window_basis": "retained_zero_regression_checkpoints",
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
        "frontier_probe_routing": "mixed_correctness_branch_contrast_feedback",
        "training_level_allocation": training_level_allocation_contract(),
        "task_sampling": "cumulative_validation_failure_structural_analogues",
        "curriculum_feedback_source": ("disabled_adapter_fixed_and_disjoint_rotating_validation"),
        "contrast_streak_reset": "new_validation_supported_training_family",
        "new_target_scheduling": "priority_active_frontier_before_rotation",
        "fixed_retention_guard_sampling": "deterministic_family_balanced_validation",
        "fixed_retention_guard_multiplier": FIXED_RETENTION_GUARD_MULTIPLIER,
        "test_seed_base": TEST_SEED_BASE,
        "maximum_consecutive_uninformative_groups": (MAXIMUM_CONSECUTIVE_UNINFORMATIVE_GROUPS),
        "maximum_consecutive_regression_windows": (MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS),
        "maximum_recent_malformed_action_rate": (MAXIMUM_RECENT_MALFORMED_ACTION_RATE),
        "maximum_consecutive_malformed_windows": (MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS),
        "shared_prefix_sampling": "greedy",
        "shared_prefix_checkpoint": SHARED_PREFIX_CHECKPOINT_STRATEGY,
        "minimum_shared_prefix_actions": MINIMUM_PREFIX_ACCEPTED_ACTIONS,
        "maximum_shared_prefix_actions": PREFIX_MAX_ATTEMPTS,
        "sibling_sampling_temperature": SIBLING_SAMPLING_TEMPERATURE,
        "sibling_sampling_top_p": SIBLING_SAMPLING_TOP_P,
        "policy_prompt_roles": ["system", "user"],
        "learning_signal": "verified_fault_fixing_edits_from_mixed_correctness_sibling_group",
        "checkpoint_selection_window": "fixed_paired_validation",
        "checkpoint_retention_guard": "paired_active_adjacent_and_rotating_zero_regressions",
        "retention_transaction_revision": ACTIVE_RETENTION_TRANSACTION_REVISION,
        "final_evaluation_reserve_source": "retained_checkpoint",
        "curriculum_validation_window": "rotating_disjoint",
        "reward_contract_revision": "correctness-gated-efficiency@1",
        "maximum_final_evaluation_reserve_seconds": (
            runtime.maximum_final_evaluation_reserve_seconds
        ),
        "target_runtime_seconds": target_seconds,
        "maximum_resume_gap_seconds": runtime.maximum_resume_gap_seconds,
        "validation_window_seed_stride": VALIDATION_WINDOW_SEED_STRIDE,
        "deterministic_runtime": deterministic_runtime,
    }
    random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
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
        deterministic_runtime=deterministic_runtime,
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
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(device)
    adapter_path = os.environ.get("EQUINOX_ADAPTER_PATH")
    if adapter_path:
        adapter_path, _ = authenticated_adapter_root_from_environment(adapter_path)
    checkpoints_root = os.path.join(adapter_path, "checkpoints") if adapter_path else None
    latest_checkpoint_path = (
        os.path.join(checkpoints_root, "latest.json") if checkpoints_root else None
    )
    checkpoint_directory: str | None = None
    authenticated_checkpoint: AuthenticatedCheckpoint | None = None
    checkpoint_authentication_key: bytes | None = None
    checkpoint_identity: dict[str, str] | None = None
    checkpoint_generation = 0
    checkpoint_manifest_digest: str | None = None
    if checkpoints_root:
        checkpoint_authentication_key = checkpoint_authentication_key_from_environment()
        checkpoint_identity = checkpoint_authentication_identity(
            model_revision=runtime.model_revision,
        )
        expected_generation, expected_manifest_digest = expected_checkpoint_from_environment(
            runtime.workload_attempt
        )
        pointer_exists = os.path.lexists(latest_checkpoint_path)
        if runtime.workload_attempt == 1:
            if pointer_exists:
                raise RuntimeError("first attempt found an unexpected checkpoint")
        else:
            if (
                not pointer_exists
                or expected_generation is None
                or expected_manifest_digest is None
            ):
                raise RuntimeError("retry requires an authenticated checkpoint")
            authenticated_checkpoint = load_authenticated_checkpoint(
                checkpoints_root=checkpoints_root,
                authentication_key=checkpoint_authentication_key,
                authentication_identity=checkpoint_identity,
                expected_generation=expected_generation,
                expected_manifest_digest=expected_manifest_digest,
            )
            checkpoint_directory = authenticated_checkpoint.private_directory
            checkpoint_generation = authenticated_checkpoint.generation
            checkpoint_manifest_digest = authenticated_checkpoint.manifest_digest
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
            weights_only=True,
        )
        if (
            not isinstance(resume_state, dict)
            or resume_state.get("checkpoint_generation") != checkpoint_generation
        ):
            raise RuntimeError("training checkpoint generation is invalid")
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
        shutil.rmtree(checkpoint_directory, ignore_errors=True)
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
            "0": evaluate_tasks(
                family_balanced_validation_tasks(
                    0,
                    fixed_retention_guard_example_count(
                        0,
                        runtime.validation_examples,
                    ),
                    checkpoint_validation_seed(0),
                ),
                level=0,
                seed=checkpoint_validation_seed(0),
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
    curriculum_failure_family_ids = (
        list(resume_state["curriculum_failure_family_ids"])
        if resume_state
        else list(baseline_failure_family_ids)
    )
    targeted_training_family_ids = (
        list(resume_state["targeted_training_family_ids"])
        if resume_state
        else training_analogue_family_ids(curriculum_failure_family_ids)
    )
    curriculum_target_expansion_count = (
        int(resume_state["curriculum_target_expansion_count"]) if resume_state else 0
    )
    priority_training_family_ids = (
        list(resume_state["priority_training_family_ids"]) if resume_state else []
    )
    if len(set(priority_training_family_ids)) != len(priority_training_family_ids) or not set(
        priority_training_family_ids
    ).issubset(targeted_training_family_ids):
        raise RuntimeError("training checkpoint priority target queue is invalid")
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
        curriculum_failure_family_ids=curriculum_failure_family_ids,
        targeted_training_family_ids=targeted_training_family_ids,
        curriculum_target_expansion_count=curriculum_target_expansion_count,
        priority_training_family_ids=priority_training_family_ids,
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
    branch_snapshot_ids = [snapshot.get("snapshot_id") for snapshot in branch_snapshots]
    branch_task_ids = [snapshot.get("task_id") for snapshot in branch_snapshots]
    branch_evidence_payload_bytes = branch_evidence_payload_size_bytes(branch_snapshots)
    if (
        len(branch_snapshots) > MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS
        or branch_evidence_payload_bytes > MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES
        or (
            resume_state
            and resume_state.get("branch_evidence_payload_bytes") != branch_evidence_payload_bytes
        )
        or any(not isinstance(snapshot_id, str) for snapshot_id in branch_snapshot_ids)
        or len(set(branch_snapshot_ids)) != len(branch_snapshot_ids)
        or any(not isinstance(task_id, str) for task_id in branch_task_ids)
        or len(set(branch_task_ids)) != len(branch_task_ids)
    ):
        raise RuntimeError("training checkpoint branch evidence is invalid")
    mastery_streak = int(resume_state["mastery_streak"]) if resume_state else 0
    updates_completed = int(resume_state["updates_completed"]) if resume_state else 0
    optimizer_update_count = int(resume_state["optimizer_update_count"]) if resume_state else 0
    policy_update_count = int(resume_state["policy_update_count"]) if resume_state else 0
    (
        attempted_policy_update_count,
        effective_policy_update_count,
        retained_policy_update_count,
        retention_rollback_count,
    ) = policy_lineage_counters_from_resume(
        resume_state,
        policy_update_count=policy_update_count,
    )
    policy_update_lineage_from_branch_evidence(
        branch_snapshots,
        attempted_policy_update_count=attempted_policy_update_count,
        effective_policy_update_count=effective_policy_update_count,
        retained_policy_update_count=retained_policy_update_count,
    )
    (
        pending_training_examples,
        pending_policy_example_count,
        pending_informative_group_ids,
        pending_optimizer_input_group_ids,
    ) = pending_optimizer_batch_from_resume(
        resume_state,
        minimum_informative_groups=MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE,
    )
    total_task_groups = int(resume_state["total_task_groups"]) if resume_state else 0
    if total_task_groups != len(branch_snapshots):
        raise RuntimeError("training checkpoint omitted collected branch evidence")
    replay_task_groups = int(resume_state["replay_task_groups"]) if resume_state else 0
    frontier_probe_task_groups = (
        int(resume_state.get("frontier_probe_task_groups", 0)) if resume_state else 0
    )
    frontier_probe_level = (
        int(
            resume_state.get(
                "frontier_probe_level",
                min(MAXIMUM_COMPLEXITY_LEVEL, level + 1),
            )
        )
        if resume_state
        else min(MAXIMUM_COMPLEXITY_LEVEL, level + 1)
    )
    frontier_probe_decision = (
        str(resume_state.get("frontier_probe_decision", "nearest_unmeasured_probe"))
        if resume_state
        else "nearest_unmeasured_probe"
    )
    if (level == MAXIMUM_COMPLEXITY_LEVEL and frontier_probe_level != level) or (
        level < MAXIMUM_COMPLEXITY_LEVEL
        and not (
            level
            < frontier_probe_level
            <= min(
                MAXIMUM_COMPLEXITY_LEVEL,
                level + MAXIMUM_FRONTIER_PROBE_OFFSET,
            )
        )
    ):
        raise RuntimeError("training checkpoint frontier probe level is invalid")
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
    (
        total_sampled_completion_tokens,
        discarded_sampled_completion_tokens,
    ) = completion_token_counters_from_resume(
        resume_state,
    )
    persisted_branch_completion_tokens = validate_sampled_completion_token_accounting(
        branch_snapshots,
        total_sampled_completion_tokens=total_sampled_completion_tokens,
        discarded_sampled_completion_tokens=discarded_sampled_completion_tokens,
    )
    if (
        resume_state
        and resume_state.get("persisted_branch_completion_tokens")
        != persisted_branch_completion_tokens
    ):
        raise RuntimeError("training checkpoint completion-token evidence is invalid")
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
    retained_transaction: dict[str, Any] | None = None
    if transactional_retention_enabled():
        if resume_state:
            retained_transaction = validate_retained_transaction_state(
                resume_state.get("retained_transaction"),
                best_validation_update=best_validation["update"],
                retained_policy_update_count=retained_policy_update_count,
            )
        else:
            retained_transaction = {
                "update": 0,
                "trainable_state": best_trainable_state,
                "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                "effective_policy_update_count": 0,
                "retained_observation": copy.deepcopy(last_observation),
            }
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
        nonlocal checkpoint_generation
        nonlocal checkpoint_manifest_digest
        if not checkpoints_root or not latest_checkpoint_path:
            return
        if checkpoint_authentication_key is None or checkpoint_identity is None:
            raise RuntimeError("checkpoint authentication context is unavailable")
        persisted_branch_completion_tokens = validate_sampled_completion_token_accounting(
            branch_snapshots,
            total_sampled_completion_tokens=total_sampled_completion_tokens,
            discarded_sampled_completion_tokens=discarded_sampled_completion_tokens,
        )
        checkpoint_name = f"update-{update:04d}"
        next_checkpoint_generation = checkpoint_generation + 1
        state = {
            "schema_version": 5,
            "checkpoint_generation": next_checkpoint_generation,
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
            "branch_evidence_complete": len(branch_snapshots) == total_task_groups,
            "branch_evidence_limit": MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
            "branch_evidence_payload_bytes": branch_evidence_payload_size_bytes(branch_snapshots),
            "branch_evidence_payload_limit_bytes": MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
            "mastery_streak": mastery_streak,
            "optimizer_update_count": optimizer_update_count,
            "policy_update_count": policy_update_count,
            "attempted_policy_update_count": attempted_policy_update_count,
            "effective_policy_update_count": effective_policy_update_count,
            "retained_policy_update_count": retained_policy_update_count,
            "retention_rollback_count": retention_rollback_count,
            "pending_training_examples": serialize_pending_training_examples(
                pending_training_examples
            ),
            "pending_policy_example_count": pending_policy_example_count,
            "pending_informative_group_ids": pending_informative_group_ids,
            "pending_optimizer_input_group_ids": pending_optimizer_input_group_ids,
            "total_task_groups": total_task_groups,
            "replay_task_groups": replay_task_groups,
            "frontier_probe_task_groups": frontier_probe_task_groups,
            "frontier_probe_level": frontier_probe_level,
            "frontier_probe_decision": frontier_probe_decision,
            "curriculum_failure_family_ids": curriculum_failure_family_ids,
            "targeted_training_family_ids": targeted_training_family_ids,
            "curriculum_target_expansion_count": curriculum_target_expansion_count,
            "priority_training_family_ids": priority_training_family_ids,
            "maximum_sampled_complexity_level": maximum_sampled_complexity_level,
            "informative_task_groups": informative_task_groups,
            "excluded_task_groups": excluded_task_groups,
            "discarded_task_groups": discarded_task_groups,
            "discarded_sampled_actions": discarded_sampled_actions,
            "discarded_post_branch_actions": discarded_post_branch_actions,
            "discarded_sampled_completion_tokens": (discarded_sampled_completion_tokens),
            "total_sampled_actions": total_sampled_actions,
            "total_post_branch_actions": total_post_branch_actions,
            "total_sampled_completion_tokens": total_sampled_completion_tokens,
            "persisted_branch_completion_tokens": (persisted_branch_completion_tokens),
            "recent_action_protocol_groups": recent_action_protocol_groups,
            "total_malformed_actions": total_malformed_actions,
            "consecutive_uninformative_groups": consecutive_uninformative_groups,
            "consecutive_regression_windows": consecutive_regression_windows,
            "consecutive_malformed_windows": consecutive_malformed_windows,
            "best_validation": best_validation,
            "best_trainable_state": best_trainable_state,
            "retained_transaction": retained_transaction,
            "rollback_applied": rollback_applied,
            "last_observation": last_observation,
            "validation_baseline_by_level": validation_baseline_by_level,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        commit = persist_checkpoint(
            checkpoints_root=checkpoints_root,
            latest_checkpoint_path=latest_checkpoint_path,
            checkpoint_name=checkpoint_name,
            state=state,
            save_adapter=lambda target: model.save_pretrained(
                target,
                safe_serialization=True,
            ),
            save_state=torch.save,
            generation=next_checkpoint_generation,
            authentication_key=checkpoint_authentication_key,
            authentication_identity=checkpoint_identity,
            expected_previous_generation=(
                checkpoint_generation if checkpoint_generation > 0 else None
            ),
            expected_previous_manifest_digest=checkpoint_manifest_digest,
        )
        checkpoint_generation = commit.generation
        checkpoint_manifest_digest = commit.manifest_digest
        emit_progress(
            "checkpointing",
            "Authenticated checkpoint committed.",
            runtime_configuration=runtime,
            preserve_context=True,
            checkpoint_generation=checkpoint_generation,
            checkpoint_manifest_digest=checkpoint_manifest_digest,
            checkpoint_name=commit.checkpoint_name,
            update=update,
            current_level=level,
            elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
        )

    def rollback_unvalidated_retention(update: int, *, reason: str) -> bool:
        nonlocal effective_policy_update_count
        nonlocal pending_training_examples
        nonlocal pending_policy_example_count
        nonlocal pending_informative_group_ids
        nonlocal pending_optimizer_input_group_ids
        nonlocal retention_rollback_count
        nonlocal rollback_applied
        nonlocal last_observation

        pending_lineage = any(
            isinstance(snapshot.get("optimizer_update"), dict)
            and snapshot["optimizer_update"].get("retention_lineage_status") == "pending"
            for snapshot in branch_snapshots
        )
        if not transactional_retention_enabled() or not pending_lineage:
            return False
        if retained_transaction is None:
            raise RuntimeError("retained transaction state is missing")
        annotate_retention_window_disposition(
            branch_snapshots,
            update=update,
            disposition="rollback",
        )
        (
            effective_policy_update_count,
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            pending_optimizer_input_group_ids,
            last_observation,
        ) = restore_retention_transaction(
            retained_transaction,
            restore_trainable_state=restore_trainable_state,
            optimizer=optimizer,
        )
        resolve_pending_branch_snapshot_lineage(
            branch_snapshots,
            status="rolled_back",
            resolution_update=update,
            reason=reason,
            effective_policy_update_count=effective_policy_update_count,
            retained_policy_update_count=retained_policy_update_count,
            retention_rollback_count=retention_rollback_count + 1,
        )
        retention_rollback_count += 1
        rollback_applied = True
        return True

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
        pre_first_retained_promotion = len(promotions) == 0
        allocation_probe_level = (
            min(MAXIMUM_COMPLEXITY_LEVEL, level + 1)
            if pre_first_retained_promotion
            else frontier_probe_level
        )
        training_allocation = training_level_allocation(
            level,
            runtime.training_tasks_per_update,
            probe_level=allocation_probe_level,
            retained_promotion_count=len(promotions),
        )
        frontier_probe_level_used = allocation_probe_level
        training_allocation_phase = (
            "before_first_retained_promotion_3_to_1"
            if pre_first_retained_promotion
            else "after_first_retained_promotion_2_to_2"
        )
        tasks_by_level: dict[int, list[RepairTask]] = {}
        for task_level in sorted(set(training_allocation)):
            tasks_by_level[task_level] = failure_directed_training_tasks(
                task_level,
                training_allocation.count(task_level),
                experiment_seed * 100_000 + update * 17 + task_level * 10_000_019,
                target_family_ids=targeted_training_family_ids,
                priority_family_ids=(priority_training_family_ids if task_level == level else ()),
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
                discarded_completion_tokens,
            ) = discarded_collection_accounting(collections)
            discarded_task_groups += discarded_groups
            discarded_sampled_actions += discarded_actions
            discarded_post_branch_actions += discarded_post_branch_action_count
            discarded_sampled_completion_tokens += discarded_completion_tokens
            total_sampled_actions += discarded_actions
            total_post_branch_actions += discarded_post_branch_action_count
            total_sampled_completion_tokens += discarded_completion_tokens
            stop_reason = "final_evaluation_reserve"
            training_complete = True
            persist_training_checkpoint(updates_completed)
            break
        attempted_priority_training_family_ids = sorted(
            set(priority_training_family_ids)
            & {
                fault.family_id
                for collection in collections
                if collection.curriculum_role == "active_frontier"
                for fault in collection.task.faults
            }
        )
        priority_training_family_ids = sorted(
            set(priority_training_family_ids) - set(attempted_priority_training_family_ids)
        )
        (
            anchored_training_examples,
            policy_training_example_count,
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            policy_signal_group_ids,
            pending_optimizer_input_group_ids,
            optimizer_input_group_ids,
        ) = accumulated_reference_anchored_examples(
            collections,
            pending_training_examples,
            pending_policy_example_count,
            pending_informative_group_ids,
            pending_optimizer_input_group_ids,
            minimum_informative_groups=MINIMUM_INFORMATIVE_GROUPS_PER_POLICY_UPDATE,
        )
        informative_collections = sum(collection.informative for collection in collections)
        (
            frontier_probe_level,
            frontier_probe_decision,
        ) = adaptive_frontier_probe_decision(
            level,
            frontier_probe_level_used,
            collections,
        )
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
        policy_update_applied = bool(policy_training_example_count)
        if policy_update_applied:
            policy_update_count += 1
            attempted_policy_update_count += 1
            effective_policy_update_count += 1

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
        total_sampled_completion_tokens += sampled_completion_token_count(collections)
        representative_collection = select_representative_collection(
            collections,
            current_level=level,
        )
        optimizer_evidence = {
            "update": update,
            "applied": bool(anchored_training_examples),
            "policy_signal_applied": bool(policy_training_example_count),
            "reference_anchor_applied": bool(anchored_training_examples),
            "objective_id": OBJECTIVE_ID,
            "adapter_revision": f"update-{update}",
            "retention_transaction_revision": ACTIVE_RETENTION_TRANSACTION_REVISION,
            "retention_lineage_status": (
                "pending" if transactional_retention_enabled() and policy_update_applied else None
            ),
            "retention_transaction_disposition": None,
            "attempted_policy_update_index": (
                attempted_policy_update_count if policy_update_applied else None
            ),
            "effective_policy_update_count_after_apply": (
                effective_policy_update_count if policy_update_applied else None
            ),
            "retained_policy_update_count_before_validation": (
                retained_policy_update_count if policy_update_applied else None
            ),
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
            "optimizer_input_group_count": len(optimizer_input_group_ids),
            "optimizer_input_group_ids": optimizer_input_group_ids,
            "pending_informative_group_count": len(pending_informative_group_ids),
            "pending_informative_group_ids": pending_informative_group_ids,
            "pending_optimizer_input_group_count": len(pending_optimizer_input_group_ids),
            "pending_optimizer_input_group_ids": pending_optimizer_input_group_ids,
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
        snapshots_for_update = [
            {
                **serialize_branch_group(
                    collection,
                    update=update,
                    optimizer_update=(
                        copy.deepcopy(optimizer_evidence)
                        if collection.task.task_id in optimizer_input_group_ids
                        else (
                            copy.deepcopy(optimizer_evidence)
                            if not optimizer_input_group_ids
                            else None
                        )
                    ),
                ),
                "collection_index": index,
                "collection_count": len(collections),
            }
            for index, collection in enumerate(collections, start=1)
        ]
        append_complete_branch_evidence(
            branch_snapshots,
            snapshots_for_update,
        )
        bind_optimizer_input_groups_to_update(
            branch_snapshots,
            optimizer_input_group_ids=optimizer_input_group_ids,
            optimizer_update=optimizer_evidence,
        )
        latest_snapshot = next(
            snapshot
            for snapshot in snapshots_for_update
            if snapshot.get("task_id") == representative_collection.task.task_id
        )
        policy_update_lineage = policy_update_lineage_from_branch_evidence(
            branch_snapshots,
            attempted_policy_update_count=attempted_policy_update_count,
            effective_policy_update_count=effective_policy_update_count,
            retained_policy_update_count=retained_policy_update_count,
        )

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
                "training_level_allocation_revision": (TRAINING_LEVEL_ALLOCATION_REVISION),
                "training_level_allocation_phase": training_allocation_phase,
                "training_level_allocation_transition": (
                    training_level_allocation_transition_evidence(
                        promotions,
                        observed_updates=update,
                    )
                ),
                "training_level_allocation": training_allocation,
                "frontier_probe_levels": sorted(
                    candidate_level
                    for candidate_level in set(training_allocation)
                    if candidate_level > level
                ),
                "frontier_probe_level_used": frontier_probe_level_used,
                "next_frontier_probe_level": frontier_probe_level,
                "frontier_probe_decision": frontier_probe_decision,
                "baseline_failure_family_ids": baseline_failure_family_ids,
                "curriculum_failure_family_ids": curriculum_failure_family_ids,
                "targeted_training_family_ids": targeted_training_family_ids,
                "curriculum_target_expansion_count": curriculum_target_expansion_count,
                "attempted_priority_training_family_ids": (attempted_priority_training_family_ids),
                "priority_training_family_ids": priority_training_family_ids,
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
            curriculum_failure_family_ids=curriculum_failure_family_ids,
            targeted_training_family_ids=targeted_training_family_ids,
            curriculum_target_expansion_count=curriculum_target_expansion_count,
            attempted_priority_training_family_ids=attempted_priority_training_family_ids,
            priority_training_family_ids=priority_training_family_ids,
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
            total_sampled_completion_tokens=total_sampled_completion_tokens,
            discarded_sampled_completion_tokens=(discarded_sampled_completion_tokens),
            policy_update_count=policy_update_count,
            attempted_policy_update_count=attempted_policy_update_count,
            effective_policy_update_count=effective_policy_update_count,
            retained_policy_update_count=retained_policy_update_count,
            retention_rollback_count=retention_rollback_count,
            optimizer_update_count=optimizer_update_count,
            pending_informative_group_count=len(pending_informative_group_ids),
            pending_optimizer_input_group_count=len(pending_optimizer_input_group_ids),
            pending_optimizer_input_group_ids=pending_optimizer_input_group_ids,
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
            branch_snapshots=branch_snapshots,
            branch_evidence_complete=len(branch_snapshots) == total_task_groups,
            branch_evidence_group_count=len(branch_snapshots),
            branch_evidence_limit=MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
            branch_evidence_payload_bytes=branch_evidence_payload_size_bytes(branch_snapshots),
            branch_evidence_payload_limit_bytes=MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
            policy_update_lineage=policy_update_lineage,
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
        contrast_stop_deferred = defer_uninformative_stop_for_validation(
            update=update,
            evaluation_interval=EVALUATION_INTERVAL,
            limit_reached=uninformative_group_stop,
        )
        if uninformative_group_stop and not contrast_stop_deferred:
            stop_after_checkpoint = True
            stop_reason = "consecutive_uninformative_groups"
        if consecutive_malformed_windows >= MAXIMUM_CONSECUTIVE_MALFORMED_WINDOWS:
            stop_after_checkpoint = True
            stop_reason = "recent_malformed_action_rate"
        completed_retention_window = False
        retention_disposition: str | None = None
        restored_retained_observation: dict[str, Any] | None = None
        if (
            retention_validation_due(
                update,
                policy_update_applied=policy_update_applied,
            )
            and not stop_after_checkpoint
        ):
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
                if rollback_unvalidated_retention(
                    update,
                    reason="validation_not_started_before_deadline",
                ):
                    retention_disposition = "rollback"
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
                guard_examples = fixed_retention_guard_example_count(
                    guard_level,
                    runtime.validation_examples,
                )
                guard_tasks = family_balanced_validation_tasks(
                    guard_level,
                    guard_examples,
                    guard_seed,
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
                if rollback_unvalidated_retention(
                    update,
                    reason="incomplete_fixed_retention_guard",
                ):
                    retention_disposition = "rollback"
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
                        "retention_transaction_disposition": retention_disposition,
                        "attempted_policy_update_count": attempted_policy_update_count,
                        "effective_policy_update_count": effective_policy_update_count,
                        "retained_policy_update_count": retained_policy_update_count,
                        "retention_rollback_count": retention_rollback_count,
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
                if rollback_unvalidated_retention(
                    update,
                    reason="incomplete_rotating_retention_baseline",
                ):
                    retention_disposition = "rollback"
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
                        "retention_transaction_disposition": retention_disposition,
                        "attempted_policy_update_count": attempted_policy_update_count,
                        "effective_policy_update_count": effective_policy_update_count,
                        "retained_policy_update_count": retained_policy_update_count,
                        "retention_rollback_count": retention_rollback_count,
                        "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                    }
                )
                stop_reason = "final_evaluation_reserve"
                training_complete = True
                persist_training_checkpoint(update)
                break
            (
                curriculum_failure_family_ids,
                targeted_training_family_ids,
                newly_targeted_training_family_ids,
            ) = expanded_validation_training_targets(
                curriculum_failure_family_ids,
                [
                    *fixed_baseline_observations.values(),
                    curriculum_baseline_observation,
                ],
            )
            if newly_targeted_training_family_ids:
                consecutive_uninformative_groups = 0
                curriculum_target_expansion_count += 1
                priority_training_family_ids = sorted(
                    set(priority_training_family_ids) | set(newly_targeted_training_family_ids)
                )
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
                if rollback_unvalidated_retention(
                    update,
                    reason="incomplete_rotating_retention_candidate",
                ):
                    retention_disposition = "rollback"
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
                        "retention_transaction_disposition": retention_disposition,
                        "attempted_policy_update_count": attempted_policy_update_count,
                        "effective_policy_update_count": effective_policy_update_count,
                        "retained_policy_update_count": retained_policy_update_count,
                        "retention_rollback_count": retention_rollback_count,
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
            if transactional_retention_enabled():
                retention_disposition = retention_transaction_disposition(
                    candidate_is_best=candidate_is_best,
                    fixed_change=fixed_guard_paired_change,
                    rotating_change=curriculum_paired_change,
                )
                annotate_retention_window_disposition(
                    branch_snapshots,
                    update=update,
                    disposition=retention_disposition,
                )
                if retention_disposition == "rollback":
                    if retained_transaction is None:
                        raise RuntimeError("retained transaction state is missing")
                    (
                        effective_policy_update_count,
                        pending_training_examples,
                        pending_policy_example_count,
                        pending_informative_group_ids,
                        pending_optimizer_input_group_ids,
                        restored_retained_observation,
                    ) = restore_retention_transaction(
                        retained_transaction,
                        restore_trainable_state=restore_trainable_state,
                        optimizer=optimizer,
                    )
                    resolve_pending_branch_snapshot_lineage(
                        branch_snapshots,
                        status="rolled_back",
                        resolution_update=update,
                        reason="retention_guard_regression",
                        effective_policy_update_count=effective_policy_update_count,
                        retained_policy_update_count=retained_policy_update_count,
                        retention_rollback_count=retention_rollback_count + 1,
                    )
                    retention_rollback_count += 1
                    rollback_applied = True
            mastery_streak = next_retained_mastery_windows(
                mastery_streak,
                candidate_retained=candidate_is_best,
                candidate_mastered=mastered,
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
                    "fixed_guard_examples_by_level": {
                        str(guard_level): fixed_retention_guard_example_count(
                            guard_level,
                            runtime.validation_examples,
                        )
                        for guard_level in guard_levels
                    },
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
                if transactional_retention_enabled():
                    retained_policy_update_count = effective_policy_update_count
                    resolve_pending_branch_snapshot_lineage(
                        branch_snapshots,
                        status="retained",
                        resolution_update=update,
                        reason="retention_guard_improvement",
                        effective_policy_update_count=effective_policy_update_count,
                        retained_policy_update_count=retained_policy_update_count,
                        retention_rollback_count=retention_rollback_count,
                    )
                    retained_transaction = capture_retention_transaction(
                        capture_trainable_state=capture_trainable_state,
                        optimizer=optimizer,
                        effective_policy_update_count=effective_policy_update_count,
                        update=update,
                        retained_observation=checkpoint_observation,
                    )
                    best_trainable_state = retained_transaction["trainable_state"]
                else:
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
                    "fixed_guard_examples_by_level": {
                        str(guard_level): fixed_retention_guard_example_count(
                            guard_level,
                            runtime.validation_examples,
                        )
                        for guard_level in guard_levels
                    },
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
                    "curriculum_failure_family_ids": curriculum_failure_family_ids,
                    "targeted_training_family_ids": targeted_training_family_ids,
                    "newly_targeted_training_family_ids": (newly_targeted_training_family_ids),
                    "curriculum_target_expansion_count": (curriculum_target_expansion_count),
                    "priority_training_family_ids": priority_training_family_ids,
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
                    "retention_transaction_disposition": retention_disposition,
                    "attempted_policy_update_count": attempted_policy_update_count,
                    "effective_policy_update_count": effective_policy_update_count,
                    "retained_policy_update_count": retained_policy_update_count,
                    "retention_rollback_count": retention_rollback_count,
                    "elapsed_seconds": round(cumulative_elapsed_seconds(), 3),
                }
            )
            completed_retention_window = True
            last_observation = (
                restored_retained_observation
                if restored_retained_observation is not None
                else checkpoint_observation
            )
            if mastery_streak >= runtime.mastery_windows and level < MAXIMUM_COMPLEXITY_LEVEL:
                current_complexity = asdict(COMPLEXITY_LEVELS[level])
                next_complexity = asdict(COMPLEXITY_LEVELS[level + 1])
                promotions.append(
                    {
                        "update": update,
                        "from_level": level,
                        "to_level": level + 1,
                        "reason": "two_retained_zero_regression_mastery_windows",
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
                frontier_probe_level = min(
                    MAXIMUM_COMPLEXITY_LEVEL,
                    level + 1,
                )
                frontier_probe_decision = "curriculum_promotion_reset_to_nearest_probe"
            maximum_level_mastered = (
                level == MAXIMUM_COMPLEXITY_LEVEL and mastery_streak >= runtime.mastery_windows
            )
            regression_stop = (
                consecutive_regression_windows >= MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS
            )
            contrast_stop_after_validation = (
                contrast_stop_deferred and not newly_targeted_training_family_ids
            )
            if regression_stop:
                stop_reason = "validation_regression"
                if not transactional_retention_enabled():
                    restore_trainable_state(best_trainable_state)
                rollback_applied = True
            elif contrast_stop_after_validation:
                stop_reason = "consecutive_uninformative_groups"
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
            stop_after_checkpoint = (
                regression_stop or contrast_stop_after_validation or provider_stop_after_checkpoint
            )
            if (
                decision_reason is not None
                and not regression_stop
                and not contrast_stop_after_validation
            ):
                stop_reason = decision_reason
        if not completed_retention_window and rollback_unvalidated_retention(
            update,
            reason="checkpoint_without_complete_retention_window",
        ):
            retention_disposition = "rollback"
        if completed_retention_window:
            policy_update_lineage = policy_update_lineage_from_branch_evidence(
                branch_snapshots,
                attempted_policy_update_count=attempted_policy_update_count,
                effective_policy_update_count=effective_policy_update_count,
                retained_policy_update_count=retained_policy_update_count,
            )
            emit_progress(
                "checkpointing",
                "Persisting completed retention window.",
                runtime_configuration=runtime,
                preserve_context=True,
                checkpoint_stage="post_validation",
                update=update,
                current_level=level,
                best_validation=best_validation,
                rollback_applied=rollback_applied,
                retention_transaction_disposition=retention_disposition,
                attempted_policy_update_count=attempted_policy_update_count,
                effective_policy_update_count=effective_policy_update_count,
                retained_policy_update_count=retained_policy_update_count,
                retention_rollback_count=retention_rollback_count,
                policy_update_count=policy_update_count,
                optimizer_update_count=optimizer_update_count,
                pending_informative_group_count=len(pending_informative_group_ids),
                pending_optimizer_input_group_count=len(pending_optimizer_input_group_ids),
                pending_optimizer_input_group_ids=pending_optimizer_input_group_ids,
                pending_policy_example_count=pending_policy_example_count,
                pending_training_example_count=len(pending_training_examples),
                total_sampled_completion_tokens=total_sampled_completion_tokens,
                discarded_sampled_completion_tokens=(discarded_sampled_completion_tokens),
                validation_history=lightweight_validation_history(history),
                curriculum_history=promotions,
                latest_branch_snapshot=branch_snapshots[-1] if branch_snapshots else None,
                branch_snapshots=branch_snapshots,
                branch_evidence_complete=len(branch_snapshots) == total_task_groups,
                branch_evidence_group_count=len(branch_snapshots),
                branch_evidence_limit=MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
                branch_evidence_payload_bytes=branch_evidence_payload_size_bytes(branch_snapshots),
                branch_evidence_payload_limit_bytes=MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
                policy_update_lineage=policy_update_lineage,
                elapsed_seconds=round(cumulative_elapsed_seconds(), 3),
                stop_reason=stop_reason,
            )
        training_complete = stop_after_checkpoint
        persist_training_checkpoint(update)
        if stop_after_checkpoint:
            break

    if best_validation["update"] != updates_completed:
        if transactional_retention_enabled():
            if retained_transaction is None:
                raise RuntimeError("retained transaction state is missing")
            (
                effective_policy_update_count,
                pending_training_examples,
                pending_policy_example_count,
                pending_informative_group_ids,
                pending_optimizer_input_group_ids,
                last_observation,
            ) = restore_retention_transaction(
                retained_transaction,
                restore_trainable_state=restore_trainable_state,
                optimizer=optimizer,
            )
            finalization_rollbacks = resolve_pending_branch_snapshot_lineage(
                branch_snapshots,
                status="rolled_back",
                resolution_update=updates_completed,
                reason="finalization_restore",
                effective_policy_update_count=effective_policy_update_count,
                retained_policy_update_count=retained_policy_update_count,
                retention_rollback_count=retention_rollback_count + 1,
            )
            if finalization_rollbacks:
                retention_rollback_count += 1
        else:
            restore_trainable_state(best_trainable_state)
        rollback_applied = True

    policy_update_lineage = policy_update_lineage_from_branch_evidence(
        branch_snapshots,
        attempted_policy_update_count=attempted_policy_update_count,
        effective_policy_update_count=effective_policy_update_count,
        retained_policy_update_count=retained_policy_update_count,
    )
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
        pending_optimizer_input_group_count=len(pending_optimizer_input_group_ids),
        pending_optimizer_input_group_ids=pending_optimizer_input_group_ids,
        pending_policy_example_count=pending_policy_example_count,
        pending_training_example_count=len(pending_training_examples),
        attempted_policy_update_count=attempted_policy_update_count,
        effective_policy_update_count=effective_policy_update_count,
        retained_policy_update_count=retained_policy_update_count,
        retention_rollback_count=retention_rollback_count,
        total_sampled_completion_tokens=total_sampled_completion_tokens,
        discarded_sampled_completion_tokens=(discarded_sampled_completion_tokens),
        validation_history=lightweight_validation_history(history),
        curriculum_history=promotions,
        branch_snapshots=branch_snapshots,
        branch_evidence_complete=len(branch_snapshots) == total_task_groups,
        branch_evidence_group_count=len(branch_snapshots),
        branch_evidence_limit=MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
        branch_evidence_payload_bytes=branch_evidence_payload_size_bytes(branch_snapshots),
        branch_evidence_payload_limit_bytes=MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
        policy_update_lineage=policy_update_lineage,
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
    policy_updates_in_retained_lineage = (
        retained_policy_update_count if transactional_retention_enabled() else policy_update_count
    )
    hypothesis_passed = (
        restored_branching_observed
        and informative_task_groups > 0
        and policy_updates_in_retained_lineage > 0
        and retention_passed
        and reward_gain is not None
        and reward_gain > 0
        and paired_test_change["regressed"] == 0
        and paired_test_change["mcnemar_exact_p_value"] < 0.05
        and final_evaluation_complete
    )
    probative_post_training = (
        updates_completed >= 1
        and policy_updates_in_retained_lineage >= 1
        and restored_branching_observed
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
    if len(branch_snapshots) != total_task_groups:
        raise RuntimeError("completed result omitted collected branch evidence")
    persisted_branch_completion_tokens = validate_sampled_completion_token_accounting(
        branch_snapshots,
        total_sampled_completion_tokens=total_sampled_completion_tokens,
        discarded_sampled_completion_tokens=discarded_sampled_completion_tokens,
    )
    policy_update_lineage = policy_update_lineage_from_branch_evidence(
        branch_snapshots,
        attempted_policy_update_count=attempted_policy_update_count,
        effective_policy_update_count=effective_policy_update_count,
        retained_policy_update_count=retained_policy_update_count,
    )
    if transactional_retention_enabled() and any(
        item.get("retention_lineage_status") == "pending" for item in policy_update_lineage
    ):
        raise RuntimeError("completed result contains unresolved policy update lineage")
    result = {
        "schema_version": 2,
        "experiment_completed": True,
        "post_training_completed": True,
        "hypothesis_passed": hypothesis_passed,
        **post_training_outcome_classification(
            hypothesis_passed=hypothesis_passed,
            final_evaluation_complete=final_evaluation_complete,
            probative_post_training=probative_post_training,
            promotion_count=len(promotions),
        ),
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
            "retention_transaction_revision": ACTIVE_RETENTION_TRANSACTION_REVISION,
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
            **deterministic_runtime,
            "cuda_seeded_all_devices": True,
        },
        "mastery_threshold": MASTERY_THRESHOLD,
        "mastery_windows": runtime.mastery_windows,
        "maximum_complexity_level": MAXIMUM_COMPLEXITY_LEVEL,
        "complexity_levels": [asdict(item) for item in COMPLEXITY_LEVELS],
        "validation_examples": runtime.validation_examples,
        "test_examples": runtime.test_examples,
        "test_seed_base": TEST_SEED_BASE,
        "evaluation_interval": EVALUATION_INTERVAL,
        "evaluation_interval_method": "wilson-score-95",
        "training_tasks_per_update": runtime.training_tasks_per_update,
        "replay_tasks_per_level": runtime.replay_tasks_per_level,
        "training_configuration": training_configuration,
        "reached_complexity_level": level,
        "maximum_sampled_complexity_level": maximum_sampled_complexity_level,
        "promotion_count": len(promotions),
        "promotions": promotions,
        "training_level_allocation_contract": training_level_allocation_contract(),
        "training_level_allocation_transition": (
            training_level_allocation_transition_evidence(
                promotions,
                observed_updates=updates_completed,
            )
        ),
        "best_validation": best_validation,
        "rollback_applied": rollback_applied,
        "retention_transaction_revision": ACTIVE_RETENTION_TRANSACTION_REVISION,
        "retention_rollback_count": retention_rollback_count,
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
        "curriculum_failure_family_ids": curriculum_failure_family_ids,
        "targeted_training_family_ids": targeted_training_family_ids,
        "curriculum_target_expansion_count": curriculum_target_expansion_count,
        "priority_training_family_ids": priority_training_family_ids,
        "final_by_level": final_by_level,
        "paired_test_change": paired_test_change,
        "paired_test_change_by_level": paired_test_change_by_level,
        "paired_test_transitions": paired_test_transitions,
        "history": history,
        "branch_snapshots": branch_snapshots,
        "branch_evidence_complete": True,
        "branch_evidence_group_count": len(branch_snapshots),
        "branch_evidence_limit": MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
        "branch_evidence_payload_bytes": branch_evidence_payload_size_bytes(branch_snapshots),
        "branch_evidence_payload_limit_bytes": MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
        "policy_update_lineage": policy_update_lineage,
        "updates_completed": updates_completed,
        "optimizer_update_count": optimizer_update_count,
        "policy_update_count": policy_update_count,
        "attempted_policy_update_count": attempted_policy_update_count,
        "effective_policy_update_count": effective_policy_update_count,
        "retained_policy_update_count": retained_policy_update_count,
        "retained_checkpoint_update": best_validation["update"],
        "pending_informative_group_count": len(pending_informative_group_ids),
        "pending_informative_group_ids": pending_informative_group_ids,
        "pending_optimizer_input_group_count": len(pending_optimizer_input_group_ids),
        "pending_optimizer_input_group_ids": pending_optimizer_input_group_ids,
        "pending_policy_example_count": pending_policy_example_count,
        "pending_training_example_count": len(pending_training_examples),
        "total_task_groups": total_task_groups,
        "replay_task_groups": replay_task_groups,
        "frontier_probe_task_groups": frontier_probe_task_groups,
        "frontier_probe_level": frontier_probe_level,
        "frontier_probe_decision": frontier_probe_decision,
        "informative_task_groups": informative_task_groups,
        "excluded_task_groups": excluded_task_groups,
        "discarded_task_groups": discarded_task_groups,
        "discarded_sampled_actions": discarded_sampled_actions,
        "discarded_post_branch_actions": discarded_post_branch_actions,
        "discarded_sampled_completion_tokens": discarded_sampled_completion_tokens,
        "informative_group_rate": round(
            informative_task_groups / max(1, total_task_groups),
            6,
        ),
        "total_sampled_actions": total_sampled_actions,
        "total_sampled_completion_tokens": total_sampled_completion_tokens,
        "persisted_branch_completion_tokens": persisted_branch_completion_tokens,
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
        "crash_tail_completion_tokens_unaccounted": (
            resumed_crash_tail_actions_unaccounted(resume_state)
        ),
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
        "checkpoint_authentication_revision": (
            CHECKPOINT_AUTHENTICATION_REVISION if checkpoints_root else None
        ),
        "checkpoint_authentication_mechanism_digest": (
            checkpoint_authentication_mechanism_digest() if checkpoints_root else None
        ),
        "checkpoint_generation": checkpoint_generation if checkpoints_root else None,
        "checkpoint_manifest_digest": (checkpoint_manifest_digest if checkpoints_root else None),
        "optimization_seed_count": 1,
        "probative_post_training": probative_post_training,
        "claim_strength": post_training_claim_strength(
            final_evaluation_complete=final_evaluation_complete,
            probative_post_training=probative_post_training,
            hypothesis_passed=hypothesis_passed,
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
        total_sampled_completion_tokens=total_sampled_completion_tokens,
        discarded_sampled_completion_tokens=(discarded_sampled_completion_tokens),
        optimizer_update_count=optimizer_update_count,
        policy_update_count=policy_update_count,
        attempted_policy_update_count=attempted_policy_update_count,
        effective_policy_update_count=effective_policy_update_count,
        retained_policy_update_count=retained_policy_update_count,
        retention_rollback_count=retention_rollback_count,
        pending_informative_group_count=len(pending_informative_group_ids),
        pending_optimizer_input_group_count=len(pending_optimizer_input_group_ids),
        pending_optimizer_input_group_ids=pending_optimizer_input_group_ids,
        pending_policy_example_count=pending_policy_example_count,
        pending_training_example_count=len(pending_training_examples),
        frontier_probe_task_groups=frontier_probe_task_groups,
        maximum_sampled_complexity_level=maximum_sampled_complexity_level,
        baseline_failure_family_ids=baseline_failure_family_ids,
        curriculum_failure_family_ids=curriculum_failure_family_ids,
        targeted_training_family_ids=targeted_training_family_ids,
        curriculum_target_expansion_count=curriculum_target_expansion_count,
        priority_training_family_ids=priority_training_family_ids,
        hypothesis_passed=hypothesis_passed,
        paired_test_change=paired_test_change,
        claim_strength=result["claim_strength"],
        elapsed_seconds=result["elapsed_seconds"],
        stop_reason=stop_reason,
        latest_branch_snapshot=branch_snapshots[-1] if branch_snapshots else None,
        branch_snapshots=branch_snapshots,
        branch_evidence_complete=True,
        branch_evidence_group_count=len(branch_snapshots),
        branch_evidence_limit=MAXIMUM_BRANCH_EVIDENCE_SNAPSHOTS,
        branch_evidence_payload_bytes=branch_evidence_payload_size_bytes(branch_snapshots),
        branch_evidence_payload_limit_bytes=MAXIMUM_BRANCH_EVIDENCE_PAYLOAD_BYTES,
        policy_update_lineage=policy_update_lineage,
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
        remote_error = workload_remote_error(exc)
        emit_progress(
            "failed",
            "Repository repair workload failed.",
            runtime_configuration=runtime,
            preserve_context=True,
            attempt=reported_attempt,
            remote_error=remote_error,
            error=remote_error["message"],
        )
        raise


if __name__ == "__main__":
    main()
