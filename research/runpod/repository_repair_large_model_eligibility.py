"""Bounded, non-mutating eligibility screen for the pinned 7B repair policy.

The frozen revision-30 trainer remains byte-identical. This wrapper replaces its
policy-facing environment with revision 32, measures a balanced all-level
validation baseline, restores every optimizer and adapter mutation, and stops
immediately after one eight-group K=4 collection.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import larger_model_gate as gate
    import repository_repair_env as frozen_environment
    import repository_repair_env_v32 as revision32
    import repository_repair_rl as frozen
except ModuleNotFoundError:
    from . import larger_model_gate as gate
    from . import repository_repair_env as frozen_environment
    from . import repository_repair_env_v32 as revision32
    from . import repository_repair_rl as frozen


SCREEN_SCHEMA_VERSION = 1
SCREEN_LEVELS = ("0", "1", "2", "3")
REQUIRED_SNAPSHOT_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)


class EligibilityScreenComplete(BaseException):
    """Leave the frozen trainer without entering final/test evaluation."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__("the larger-model eligibility screen completed")


@dataclass
class BaselineOutcome:
    level: int
    solved: bool
    checkpoint_reached: bool


@dataclass
class ScreenEvidence:
    """Evidence accumulated by runtime hooks without retaining model outputs."""

    started_at: float = field(default_factory=time.monotonic)
    baseline_outcomes: list[BaselineOutcome] = field(default_factory=list)
    branch_collections: list[Any] = field(default_factory=list)
    total_actions: int = 0
    accepted_actions: int = 0
    schema_valid_actions: int = 0
    repeated_rejected_loop_count: int = 0
    optimizer_step_calls: int = 0
    restored_parameter_tensors: int = 0
    policy_mutation_detected: bool = False
    optimizer_state_restored: bool = True
    capacity_smoke_completed: bool = False
    gradient_checkpointing_enabled: bool = False
    pinned_snapshot_ready: bool = False
    pinned_snapshot_path: str | None = None
    cache_free_bytes: int = 0
    offline_mode_active: bool = False
    test_split_accessed: bool = False
    early_stop_reason: str | None = None
    gpu_name: str | None = None
    gpu_total_memory_bytes: int = 0
    reserved_vram_bytes: int = 0
    peak_reserved_vram_bytes: int = 0
    bf16_supported: bool = False
    base_parameter_count: int = 0
    base_bf16_parameter_count: int = 0
    trainable_parameter_count: int = 0
    model_parameter_count_with_adapter: int = 0
    runtime_configuration: Any | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _minimum_successes(rate: float, total: int) -> int:
    return math.ceil(rate * total)


def _maximum_successes(rate: float, total: int) -> int:
    return math.floor(rate * total)


def _snapshot_candidates(manifest: dict[str, Any]) -> tuple[Path, ...]:
    cache_root = Path(manifest["artifact_readiness"]["cache_directory"])
    model_cache_name = "models--" + manifest["model"]["id"].replace("/", "--")
    revision = manifest["model"]["revision"]
    return (
        cache_root / "hub" / model_cache_name / "snapshots" / revision,
        cache_root / model_cache_name / "snapshots" / revision,
    )


def verify_pinned_snapshot(manifest: dict[str, Any]) -> tuple[bool, str | None]:
    """Verify the exact four local weight shards without contacting the hub."""

    expected_bytes = manifest["model"]["safetensors_bytes"]
    expected_names = {f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)}
    for snapshot in _snapshot_candidates(manifest):
        if not snapshot.is_dir():
            continue
        shards = sorted(snapshot.glob("*.safetensors"))
        if {item.name for item in shards} != expected_names:
            continue
        if not all((snapshot / name).is_file() for name in REQUIRED_SNAPSHOT_FILES):
            continue
        try:
            observed_bytes = sum(item.stat().st_size for item in shards)
            index = json.loads(
                (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
        except OSError:
            continue
        except json.JSONDecodeError:
            continue
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or set(weight_map.values()) != expected_names
        ):
            continue
        if observed_bytes == expected_bytes:
            return True, str(snapshot)
    return False, None


def offline_mode_active() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def verify_prewarmed_dependencies(
    manifest: dict[str, Any],
    *,
    version_reader: Any = importlib.metadata.version,
) -> dict[str, str]:
    """Require exact preinstalled packages; a paid screen never runs pip."""

    observed: dict[str, str] = {}
    for package, expected_version in manifest["runtime"]["dependencies"].items():
        try:
            observed_version = version_reader(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"prewarmed dependency {package} is unavailable") from error
        if observed_version != expected_version:
            raise RuntimeError(
                f"prewarmed dependency {package}={observed_version} "
                f"does not match {expected_version}"
            )
        observed[package] = observed_version
    return observed


def capture_cuda_evidence(evidence: ScreenEvidence, torch_module: Any) -> None:
    """Capture hardware facts after synchronizing outstanding CUDA work."""

    if not torch_module.cuda.is_available():
        return
    torch_module.cuda.synchronize()
    properties = torch_module.cuda.get_device_properties(0)
    evidence.gpu_name = str(torch_module.cuda.get_device_name(0))
    evidence.gpu_total_memory_bytes = int(properties.total_memory)
    evidence.reserved_vram_bytes = int(torch_module.cuda.memory_reserved(0))
    evidence.peak_reserved_vram_bytes = int(torch_module.cuda.max_memory_reserved(0))
    evidence.bf16_supported = bool(torch_module.cuda.is_bf16_supported())


def _nested_equal(left: Any, right: Any, torch_module: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key], torch_module) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_equal(a, b, torch_module) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _nested_equal(a, b, torch_module) for a, b in zip(left, right, strict=True)
        )
    if hasattr(left, "shape") and hasattr(right, "shape"):
        return bool(torch_module.equal(left, right))
    return left == right


def install_no_persistent_mutation(
    evidence: ScreenEvidence,
    torch_module: Any,
) -> None:
    """Run AdamW normally, then restore both parameters and optimizer state."""

    original_step = torch_module.optim.AdamW.step

    def restored_step(optimizer: Any, closure: Any = None) -> Any:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        parameter_snapshots = [parameter.detach().clone() for parameter in parameters]
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        outcome = original_step(optimizer, closure)
        with torch_module.no_grad():
            for parameter, snapshot in zip(parameters, parameter_snapshots, strict=True):
                parameter.copy_(snapshot)
        optimizer.load_state_dict(optimizer_snapshot)
        parameters_restored = all(
            bool(torch_module.equal(parameter.detach(), snapshot))
            for parameter, snapshot in zip(parameters, parameter_snapshots, strict=True)
        )
        optimizer_restored = _nested_equal(
            optimizer.state_dict(),
            optimizer_snapshot,
            torch_module,
        )
        evidence.optimizer_step_calls += 1
        evidence.restored_parameter_tensors += len(parameters)
        evidence.policy_mutation_detected = (
            evidence.policy_mutation_detected or not parameters_restored
        )
        evidence.optimizer_state_restored = evidence.optimizer_state_restored and optimizer_restored
        if not parameters_restored or not optimizer_restored:
            raise RuntimeError("eligibility screen failed to restore optimizer mutation")
        return outcome

    torch_module.optim.AdamW.step = restored_step


def run_capacity_smoke(
    model: Any,
    optimizer: Any,
    *,
    maximum_input_tokens: int,
    torch_module: Any,
) -> None:
    """Exercise the screen's exact reference/policy/backward/Adam memory shape."""

    parameter = next(model.parameters())
    device = parameter.device
    input_ids = torch_module.zeros(
        (1, maximum_input_tokens),
        device=device,
        dtype=torch_module.long,
    )
    attention_mask = torch_module.ones_like(input_ids)
    target_ids = input_ids[:, 1:]
    with torch_module.no_grad(), model.disable_adapter():
        reference_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        reference_logits = reference_output.logits[:, :-1].float()
        reference_log_probabilities = (
            torch_module.log_softmax(reference_logits, dim=-1)
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
    policy_log_probabilities = (
        torch_module.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    )
    log_ratio = reference_log_probabilities - policy_log_probabilities
    reverse_kl = torch_module.expm1(log_ratio) - log_ratio
    loss = -policy_log_probabilities.mean() + frozen.REFERENCE_KL_COEFFICIENT * reverse_kl.mean()
    loss.backward()
    torch_module.nn.utils.clip_grad_norm_(
        (item for item in model.parameters() if item.requires_grad),
        max_norm=1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del (
        attention_mask,
        input_ids,
        logits,
        log_ratio,
        loss,
        output,
        policy_log_probabilities,
        reference_log_probabilities,
        reverse_kl,
        target_ids,
    )
    torch_module.cuda.empty_cache()


def _baseline_counts(evidence: ScreenEvidence) -> dict[str, dict[str, int]]:
    counts = {level: {"examples": 0, "checkpoints": 0, "solved": 0} for level in SCREEN_LEVELS}
    for outcome in evidence.baseline_outcomes:
        level = str(outcome.level)
        counts[level]["examples"] += 1
        counts[level]["checkpoints"] += int(outcome.checkpoint_reached)
        counts[level]["solved"] += int(outcome.solved)
    return counts


def baseline_impossible(evidence: ScreenEvidence, manifest: dict[str, Any]) -> str | None:
    """Return a fail-fast reason only after the preregistered minimum sample."""

    screen = manifest["screen"]
    completed = len(evidence.baseline_outcomes)
    if completed < screen["minimum_completed_baseline_examples"]:
        return None
    expected_per_level = screen["baseline_examples_per_level"]
    expected_total = expected_per_level * len(SCREEN_LEVELS)
    thresholds = screen["thresholds"]
    counts = _baseline_counts(evidence)
    required_per_level = _minimum_successes(
        thresholds["minimum_per_level_checkpoint_rate"],
        expected_per_level,
    )
    for level, values in counts.items():
        remaining = expected_per_level - values["examples"]
        if values["checkpoints"] + remaining < required_per_level:
            return f"LEVEL_{level}_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE"

    checkpoints = sum(item.checkpoint_reached for item in evidence.baseline_outcomes)
    remaining = expected_total - completed
    required_overall = _minimum_successes(
        thresholds["minimum_baseline_checkpoint_rate"],
        expected_total,
    )
    if checkpoints + remaining < required_overall:
        return "OVERALL_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE"

    solved = sum(item.solved for item in evidence.baseline_outcomes)
    minimum_exact = _minimum_successes(
        thresholds["minimum_baseline_exact_rate"],
        expected_total,
    )
    maximum_exact = _maximum_successes(
        thresholds["maximum_baseline_exact_rate"],
        expected_total,
    )
    if solved + remaining < minimum_exact or solved > maximum_exact:
        return "BASELINE_EXACT_HEADROOM_GATE_MATHEMATICALLY_IMPOSSIBLE"
    if evidence.repeated_rejected_loop_count:
        return "REPEATED_REJECTED_ACTION_LOOP"
    return None


def _hardware_verified(evidence: ScreenEvidence, manifest: dict[str, Any]) -> bool:
    minimum_bytes = manifest["hardware"]["minimum_gpu_memory_gb"] * 1024**3
    return (
        evidence.gpu_name == manifest["hardware"]["gpu_id"]
        and evidence.gpu_total_memory_bytes >= minimum_bytes
        and evidence.bf16_supported
        and evidence.base_parameter_count == manifest["model"]["parameter_count"]
        and evidence.base_bf16_parameter_count == manifest["model"]["parameter_count"]
    )


def build_screen_result(
    evidence: ScreenEvidence,
    manifest: dict[str, Any],
    *,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build the exact authorization input; failed metric gates remain successful work."""

    screen = manifest["screen"]
    thresholds = screen["thresholds"]
    expected_per_level = screen["baseline_examples_per_level"]
    expected_baseline = expected_per_level * len(SCREEN_LEVELS)
    baseline_counts = _baseline_counts(evidence)
    completed_baseline = len(evidence.baseline_outcomes)
    baseline_checkpoints = sum(outcome.checkpoint_reached for outcome in evidence.baseline_outcomes)
    baseline_solved = sum(outcome.solved for outcome in evidence.baseline_outcomes)
    per_level_rates = {
        level: _safe_rate(values["checkpoints"], values["examples"])
        for level, values in baseline_counts.items()
    }
    baseline_checkpoint_rate = _safe_rate(baseline_checkpoints, completed_baseline)
    baseline_exact_rate = _safe_rate(baseline_solved, completed_baseline)
    action_protocol_validity = _safe_rate(
        evidence.accepted_actions,
        evidence.total_actions,
    )

    expected_groups = screen["branch_groups"]
    branch_groups = evidence.branch_collections
    checkpointed_groups = [
        collection
        for collection in branch_groups
        if collection.snapshot is not None
        and collection.exclusion_reason is None
        and not collection.prefix.terminal
    ]
    informative_groups = sum(collection.informative for collection in branch_groups)
    solved_siblings = sum(collection.solved_siblings for collection in checkpointed_groups)
    completed_siblings = sum(len(collection.siblings) for collection in checkpointed_groups)
    failed_siblings = completed_siblings - solved_siblings
    branch_checkpoint_rate = _safe_rate(len(checkpointed_groups), expected_groups)
    informative_group_rate = _safe_rate(informative_groups, expected_groups)
    solved_sibling_rate = _safe_rate(solved_siblings, completed_siblings)
    peak_fraction = _safe_rate(
        evidence.peak_reserved_vram_bytes,
        evidence.gpu_total_memory_bytes,
    )

    baseline_gate = (
        completed_baseline == expected_baseline
        and baseline_checkpoint_rate >= thresholds["minimum_baseline_checkpoint_rate"]
        and all(
            rate >= thresholds["minimum_per_level_checkpoint_rate"]
            for rate in per_level_rates.values()
        )
    )
    action_gate = (
        evidence.total_actions > 0
        and action_protocol_validity >= thresholds["minimum_action_protocol_validity"]
        and evidence.repeated_rejected_loop_count == 0
    )
    branch_gate = (
        len(branch_groups) == expected_groups
        and branch_checkpoint_rate >= thresholds["minimum_branch_checkpoint_rate"]
    )
    informative_gate = (
        informative_groups >= thresholds["minimum_informative_groups"]
        and informative_group_rate >= thresholds["minimum_informative_group_rate"]
    )
    sibling_gate = (
        solved_siblings >= thresholds["minimum_solved_siblings"]
        and failed_siblings >= thresholds["minimum_failed_siblings"]
        and thresholds["minimum_solved_sibling_rate"]
        <= solved_sibling_rate
        <= thresholds["maximum_solved_sibling_rate"]
    )
    exact_gate = (
        completed_baseline == expected_baseline
        and thresholds["minimum_baseline_exact_rate"]
        <= baseline_exact_rate
        <= thresholds["maximum_baseline_exact_rate"]
    )
    peak_gate = (
        evidence.capacity_smoke_completed
        and evidence.gpu_total_memory_bytes > 0
        and peak_fraction <= manifest["hardware"]["maximum_peak_reserved_vram_fraction"]
    )
    policy_unchanged = (
        evidence.capacity_smoke_completed
        and evidence.optimizer_step_calls >= 1
        and not evidence.policy_mutation_detected
    )
    gate_results = {
        "baseline_checkpoint_rate": baseline_gate,
        "action_protocol_validity": action_gate,
        "branch_checkpoint_rate": branch_gate,
        "informative_group_rate": informative_gate,
        "solved_sibling_rate": sibling_gate,
        "baseline_exact_rate": exact_gate,
        "peak_reserved_vram_within_limit": peak_gate,
        "pinned_snapshot_ready": evidence.pinned_snapshot_ready,
        "offline_mode_active": evidence.offline_mode_active,
        "hardware_verified": _hardware_verified(evidence, manifest),
        "policy_unchanged": policy_unchanged,
        "optimizer_state_restored": evidence.optimizer_state_restored,
        "test_split_isolated": not evidence.test_split_accessed,
    }
    if set(gate_results) != set(gate.REQUIRED_GATE_RESULTS):
        raise RuntimeError("eligibility result gates drifted from the authorization contract")
    eligible = all(gate_results.values())
    runtime = evidence.runtime_configuration
    result = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "workload": screen["workload"],
        "workload_revision": screen["workload_revision"],
        "profile_id": manifest["profile_id"],
        "larger_model_profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "device": "cuda" if evidence.gpu_name is not None else "unavailable",
        "parameter_count": evidence.base_parameter_count,
        "base_bfloat16_parameter_count": evidence.base_bf16_parameter_count,
        "model_parameter_count_with_adapter": evidence.model_parameter_count_with_adapter,
        "trainable_parameter_count": evidence.trainable_parameter_count,
        "screen_completed": True,
        "eligible": eligible,
        "training_started": False,
        "persistent_policy_updates": 0,
        "policy_mutation_enabled": False,
        "policy_mutation_detected": evidence.policy_mutation_detected,
        "optimizer_state_restored": evidence.optimizer_state_restored,
        "optimizer_step_calls": evidence.optimizer_step_calls,
        "restored_parameter_tensors": evidence.restored_parameter_tensors,
        "capacity_smoke_completed": evidence.capacity_smoke_completed,
        "gradient_checkpointing_enabled": evidence.gradient_checkpointing_enabled,
        "test_split_accessed": evidence.test_split_accessed,
        "branch_width": screen["branch_width"],
        "branch_groups": len(branch_groups),
        "completed_baseline_examples": completed_baseline,
        "expected_baseline_examples": expected_baseline,
        "per_level_checkpoint_rates": per_level_rates,
        "baseline_checkpoint_rate": baseline_checkpoint_rate,
        "baseline_exact_rate": baseline_exact_rate,
        "action_protocol_validity": action_protocol_validity,
        "accepted_action_rate": action_protocol_validity,
        "schema_valid_action_rate": _safe_rate(
            evidence.schema_valid_actions,
            evidence.total_actions,
        ),
        "repeated_rejected_loop_count": evidence.repeated_rejected_loop_count,
        "checkpoint_admission_rate": branch_checkpoint_rate,
        "branch_checkpoint_rate": branch_checkpoint_rate,
        "informative_groups": informative_groups,
        "informative_group_rate": informative_group_rate,
        "solved_siblings": solved_siblings,
        "failed_siblings": failed_siblings,
        "solved_sibling_rate": solved_sibling_rate,
        "gpu_name": evidence.gpu_name,
        "gpu_id": evidence.gpu_name,
        "gpu_total_memory_bytes": evidence.gpu_total_memory_bytes,
        "gpu_total_memory_gb": round(evidence.gpu_total_memory_bytes / 1024**3, 3),
        "gpu_memory_gb": round(evidence.gpu_total_memory_bytes / 1024**3, 3),
        "reserved_vram_bytes": evidence.reserved_vram_bytes,
        "reserved_vram_gb": round(evidence.reserved_vram_bytes / 1024**3, 3),
        "peak_reserved_vram_bytes": evidence.peak_reserved_vram_bytes,
        "peak_reserved_vram_gb": round(evidence.peak_reserved_vram_bytes / 1024**3, 3),
        "peak_reserved_vram_fraction": peak_fraction,
        "bf16_supported": evidence.bf16_supported,
        "pinned_snapshot_ready": evidence.pinned_snapshot_ready,
        "pinned_snapshot_path": evidence.pinned_snapshot_path,
        "cache_free_bytes": evidence.cache_free_bytes,
        "offline_mode_active": evidence.offline_mode_active,
        "training_microbatch_size": screen["training_microbatch_size"],
        "maximum_input_tokens": screen["maximum_input_tokens"],
        "capacity_smoke_sequence_tokens": (screen["maximum_input_tokens"] + frozen.MAX_NEW_TOKENS),
        "validation_examples": (
            runtime.validation_examples if runtime is not None else screen["validation_examples"]
        ),
        "test_examples_accessed": 0,
        "early_stop_reason": evidence.early_stop_reason,
        "ineligibility_reasons": sorted(key for key, passed in gate_results.items() if not passed),
        "ineligible_reasons": sorted(key for key, passed in gate_results.items() if not passed),
        "gate_results": gate_results,
        "elapsed_seconds": round(time.monotonic() - evidence.started_at, 3),
        "completed_at": completed_at or utc_now(),
        "environment_revision": revision32.ENVIRONMENT_REVISION,
        "action_protocol_revision": revision32.ACTION_PROTOCOL_REVISION,
    }
    result["digest"] = gate.result_digest(result)
    return result


def validate_runtime_configuration(runtime: Any, manifest: dict[str, Any]) -> None:
    screen = manifest["screen"]
    expected = {
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "validation_examples": screen["validation_examples"],
        "training_tasks_per_update": screen["training_tasks_per_update"],
        "maximum_updates": screen["maximum_updates"],
        "workload_attempt": 1,
    }
    for name, value in expected.items():
        if getattr(runtime, name) != value:
            raise ValueError(f"larger-model screen requires {name}={value!r}")
    if os.environ.get("EQUINOX_ADAPTER_PATH"):
        raise ValueError("larger-model eligibility must not load or persist an adapter")


def disarm_empty_adapter_path() -> None:
    """Ignore the runner's empty conventional path, but reject resumable state."""

    raw_path = os.environ.get("EQUINOX_ADAPTER_PATH")
    if not raw_path:
        return
    adapter_path = Path(raw_path)
    if adapter_path.exists() and (not adapter_path.is_dir() or any(adapter_path.iterdir())):
        raise ValueError("larger-model eligibility found pre-existing adapter state")
    os.environ.pop("EQUINOX_ADAPTER_PATH", None)


def install_environment_hooks(
    evidence: ScreenEvidence,
    manifest: dict[str, Any],
) -> None:
    """Install revision-32 semantics, balanced baseline, K4, and isolation guards."""

    screen = manifest["screen"]
    if screen["branch_width"] != 4:
        raise RuntimeError("larger-model eligibility is defined only for static K=4")
    frozen.SUPPORTED_MODELS[manifest["model"]["id"]] = manifest["model"]["revision"]
    frozen.SYSTEM_PROMPT = revision32.SYSTEM_PROMPT
    frozen.ENVIRONMENT_REVISION = revision32.ENVIRONMENT_REVISION
    frozen.ACTION_PROTOCOL_REVISION = revision32.ACTION_PROTOCOL_REVISION
    frozen.BRANCH_WIDTH = screen["branch_width"]
    frozen_environment.BRANCH_WIDTH = screen["branch_width"]
    frozen.TRAINING_MICROBATCH_SIZE = screen["training_microbatch_size"]
    frozen.MAX_INPUT_TOKENS = screen["maximum_input_tokens"]

    class AuditedEnvironment(revision32.RepositoryRepairEnvironment):
        def step(
            self,
            response: str,
            *,
            allowed_tools: frozenset[str] | None = None,
        ) -> Any:
            result = super().step(response, allowed_tools=allowed_tools)
            evidence.total_actions += 1
            evidence.accepted_actions += int(result.accepted)
            evidence.schema_valid_actions += int(result.action is not None)
            if len(self.steps) >= 2:
                previous = self.steps[-2]
                if (
                    not previous.accepted
                    and not result.accepted
                    and previous.action == result.action
                ):
                    evidence.repeated_rejected_loop_count += 1
            return result

    frozen.RepositoryRepairEnvironment = AuditedEnvironment
    original_checkpoint = frozen.branch_checkpoint_reached

    def nonterminal_checkpoint(task: Any, prefix: Any) -> bool:
        return not prefix.terminal and original_checkpoint(task, prefix)

    frozen.branch_checkpoint_reached = nonterminal_checkpoint
    original_make_tasks = frozen.make_tasks

    def isolated_make_tasks(
        level: int,
        count: int,
        seed: int,
        *,
        split: str = "train",
        exclude_semantic_task_ids: frozenset[str] = frozenset(),
    ) -> Any:
        if split == "test":
            evidence.test_split_accessed = True
            raise RuntimeError("eligibility screen attempted to access the test split")
        return original_make_tasks(
            level,
            count,
            seed,
            split=split,
            exclude_semantic_task_ids=exclude_semantic_task_ids,
        )

    frozen.make_tasks = isolated_make_tasks
    original_family_balanced = frozen.family_balanced_validation_tasks
    expected_baseline = screen["baseline_examples_per_level"] * len(SCREEN_LEVELS)
    original_fixed_guard_count = frozen.fixed_retention_guard_example_count

    def screen_baseline_count(level: int, validation_examples: int) -> int:
        if level == 0 and validation_examples == screen["validation_examples"]:
            return expected_baseline
        return original_fixed_guard_count(level, validation_examples)

    def all_level_baseline(level: int, count: int, seed: int) -> list[Any]:
        if level != 0 or count != expected_baseline:
            raise RuntimeError("frozen baseline shape drifted from the eligibility profile")
        return [
            task
            for baseline_level in range(len(SCREEN_LEVELS))
            for task in original_family_balanced(
                baseline_level,
                screen["baseline_examples_per_level"],
                seed + baseline_level * 1_000_003,
            )
        ]

    frozen.fixed_retention_guard_example_count = screen_baseline_count
    frozen.family_balanced_validation_tasks = all_level_baseline
    original_greedy = frozen.collect_greedy_trajectory

    def observed_greedy(task: Any, *args: Any, **kwargs: Any) -> Any:
        outcome = original_greedy(task, *args, **kwargs)
        if outcome is not None and len(evidence.baseline_outcomes) < expected_baseline:
            evidence.baseline_outcomes.append(
                BaselineOutcome(
                    level=int(task.level),
                    solved=bool(outcome["solved"]),
                    checkpoint_reached=bool(outcome["checkpoint_reached"]),
                )
            )
            reason = baseline_impossible(evidence, manifest)
            if reason is not None:
                evidence.early_stop_reason = reason
                raise EligibilityScreenComplete(build_screen_result(evidence, manifest))
        return outcome

    frozen.collect_greedy_trajectory = observed_greedy
    original_collect = frozen.collect_branch_group

    def observed_collection(task: Any, *args: Any, **kwargs: Any) -> Any:
        if len(evidence.branch_collections) >= screen["branch_groups"]:
            raise RuntimeError("eligibility screen exceeded its branch-group budget")
        collection = original_collect(task, *args, **kwargs)
        evidence.branch_collections.append(collection)
        return collection

    frozen.collect_branch_group = observed_collection


def install_model_hooks(evidence: ScreenEvidence, manifest: dict[str, Any]) -> None:
    """Force offline exact-revision loads and run one reversible capacity step."""

    hooks_installed = False

    def ensure_and_install() -> None:
        nonlocal hooks_installed
        verify_prewarmed_dependencies(manifest)
        if hooks_installed:
            return
        hooks_installed = True
        import peft
        import torch
        import transformers

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for larger-model eligibility")
        torch.cuda.reset_peak_memory_stats(0)
        capture_cuda_evidence(evidence, torch)
        install_no_persistent_mutation(evidence, torch)

        original_tokenizer_load = transformers.AutoTokenizer.from_pretrained
        original_model_load = transformers.AutoModelForCausalLM.from_pretrained

        def require_identity(model_id: str, revision: Any) -> None:
            if model_id != manifest["model"]["id"] or revision != manifest["model"]["revision"]:
                raise RuntimeError("model loader identity drifted from eligibility profile")

        def offline_tokenizer_load(model_id: str, *args: Any, **kwargs: Any) -> Any:
            require_identity(model_id, kwargs.get("revision"))
            if kwargs.get("local_files_only") is False:
                raise RuntimeError("network tokenizer loading is forbidden")
            kwargs["local_files_only"] = True
            kwargs["cache_dir"] = manifest["artifact_readiness"]["cache_directory"]
            return original_tokenizer_load(model_id, *args, **kwargs)

        def offline_model_load(model_id: str, *args: Any, **kwargs: Any) -> Any:
            require_identity(model_id, kwargs.get("revision"))
            if kwargs.get("local_files_only") is False:
                raise RuntimeError("network model loading is forbidden")
            kwargs["local_files_only"] = True
            kwargs["cache_dir"] = manifest["artifact_readiness"]["cache_directory"]
            model = original_model_load(model_id, *args, **kwargs)
            evidence.base_parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            evidence.base_bf16_parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.dtype == torch.bfloat16
            )
            return model

        transformers.AutoTokenizer.from_pretrained = offline_tokenizer_load
        transformers.AutoModelForCausalLM.from_pretrained = offline_model_load
        original_get_peft_model = peft.get_peft_model
        latest_peft_model: Any | None = None

        def checkpointed_peft_model(*args: Any, **kwargs: Any) -> Any:
            nonlocal latest_peft_model
            model = original_get_peft_model(*args, **kwargs)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            evidence.gradient_checkpointing_enabled = True
            evidence.model_parameter_count_with_adapter = sum(
                parameter.numel() for parameter in model.parameters()
            )
            evidence.trainable_parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            latest_peft_model = model
            return model

        peft.get_peft_model = checkpointed_peft_model
        original_adam_init = torch.optim.AdamW.__init__

        def smoke_after_adam_init(optimizer: Any, *args: Any, **kwargs: Any) -> None:
            original_adam_init(optimizer, *args, **kwargs)
            if latest_peft_model is None:
                raise RuntimeError("capacity smoke did not observe the PEFT model")
            run_capacity_smoke(
                latest_peft_model,
                optimizer,
                maximum_input_tokens=(
                    manifest["screen"]["maximum_input_tokens"] + frozen.MAX_NEW_TOKENS
                ),
                torch_module=torch,
            )
            evidence.capacity_smoke_completed = True
            capture_cuda_evidence(evidence, torch)

        torch.optim.AdamW.__init__ = smoke_after_adam_init

    frozen.ensure_dependencies = ensure_and_install


def install_progress_hook(evidence: ScreenEvidence, manifest: dict[str, Any]) -> None:
    original_emit = frozen.emit_progress

    def screen_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        original_emit(
            phase,
            message,
            runtime_configuration,
            preserve_context=preserve_context,
            larger_model_profile_id=manifest["profile_id"],
            larger_model_screen_revision=manifest["screen"]["workload_revision"],
            policy_mutation_enabled=False,
            test_split_accessed=False,
            **values,
        )
        if phase != "training":
            return
        if len(evidence.branch_collections) != manifest["screen"]["branch_groups"]:
            raise RuntimeError("training progress arrived before all screen groups completed")
        try:
            import torch

            capture_cuda_evidence(evidence, torch)
        except (ImportError, RuntimeError):
            pass
        raise EligibilityScreenComplete(build_screen_result(evidence, manifest))

    frozen.emit_progress = screen_progress


def prepare_runtime(
    evidence: ScreenEvidence,
    manifest: dict[str, Any],
) -> Any:
    disarm_empty_adapter_path()
    install_environment_hooks(evidence, manifest)
    runtime = frozen.configure_from_environment()
    validate_runtime_configuration(runtime, manifest)
    evidence.runtime_configuration = runtime
    return runtime


def self_test(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["screen"]["branch_width"] != 4:
        raise AssertionError("eligibility screen is not K=4")
    if manifest["screen"]["training_microbatch_size"] != 1:
        raise AssertionError("eligibility screen microbatch drifted")
    result = {
        "self_test_passed": True,
        "workload": manifest["screen"]["workload"],
        "workload_revision": manifest["screen"]["workload_revision"],
        "profile_id": manifest["profile_id"],
        "environment_revision": revision32.ENVIRONMENT_REVISION,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    manifest = gate.load_manifest()
    if "--self-test" in sys.argv:
        self_test(manifest)
        return
    evidence = ScreenEvidence()
    runtime = prepare_runtime(evidence, manifest)
    if "--validate-configuration" in sys.argv:
        return

    evidence.offline_mode_active = offline_mode_active()
    try:
        import torch

        capture_cuda_evidence(evidence, torch)
    except ImportError:
        pass
    (
        evidence.pinned_snapshot_ready,
        evidence.pinned_snapshot_path,
    ) = verify_pinned_snapshot(manifest)
    if evidence.pinned_snapshot_path is not None:
        evidence.cache_free_bytes = shutil.disk_usage(evidence.pinned_snapshot_path).free
        evidence.pinned_snapshot_ready = (
            evidence.pinned_snapshot_ready
            and evidence.cache_free_bytes >= manifest["hardware"]["minimum_free_cache_bytes"]
        )
    if not evidence.offline_mode_active or not evidence.pinned_snapshot_ready:
        print(json.dumps(build_screen_result(evidence, manifest), sort_keys=True), flush=True)
        return

    install_model_hooks(evidence, manifest)
    install_progress_hook(evidence, manifest)
    try:
        frozen.run_experiment(runtime)
    except EligibilityScreenComplete as completed:
        print(json.dumps(completed.result, sort_keys=True), flush=True)
        return
    raise RuntimeError("larger-model eligibility ended without a screen result")


if __name__ == "__main__":
    main()
