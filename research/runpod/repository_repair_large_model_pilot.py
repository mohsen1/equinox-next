"""Receipt-authorized 7B K=4 adaptive repository-repair pilot."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import larger_model_gate as gate
    import repository_repair_env as frozen_environment
    import repository_repair_env_v33 as interface
    import repository_repair_large_model_eligibility as eligibility
    import repository_repair_large_model_study as study
    import repository_repair_large_model_trainer as frozen
except ModuleNotFoundError:
    from . import larger_model_gate as gate
    from . import repository_repair_env as frozen_environment
    from . import repository_repair_env_v33 as interface
    from . import repository_repair_large_model_eligibility as eligibility
    from . import repository_repair_large_model_study as study
    from . import repository_repair_large_model_trainer as frozen


WORKLOAD_REVISION = "runpod-repository-repair-large-model-pilot@7"
OBJECTIVE_ID = "verified-repair-chain-transactional-retention-policy-gradient@19"
REWARD_CONTRACT_REVISION = "correctness-gated-efficiency@1"
RETENTION_TRANSACTION_REVISION = "adapter-optimizer-policy-lineage@1"
DETERMINISTIC_RUNTIME_REVISION = "eager-math-sdp-deterministic@1"
ATTENTION_IMPLEMENTATION = "eager"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS = 4
TRAINING_LEVEL_ALLOCATION_REVISION = "retained-promotion-3-1-to-2-2@1"
LEARNING_RATE = 1e-5
REFERENCE_KL_COEFFICIENT = 1.0
SHARED_PREFIX_CHECKPOINT_STRATEGY = "repository_root_observed@1"
LOCALIZATION_TELEMETRY_STRATEGY = "all_fault_sources_observed"
POLICY_CREDIT_SCOPE = (
    "fault_fixing_edits_and_immediately_upstream_fresh_reads_from_verified_successful_siblings"
)
LEARNING_SIGNAL = (
    "verified_fresh_read_and_fault_fixing_edit_chains_from_mixed_correctness_sibling_groups"
)
MODEL_REVISION = gate.MODEL_REVISION
PILOT_WORKLOAD = "repository-repair-restored-continuation-post-training"
OBSERVED_POLICY_UPDATE_COUNT = 0
POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE: dict[int, int] = {}
RETAINED_CHECKPOINT_UPDATE = 0
RETAINED_POLICY_UPDATE_COUNT = 0
VERIFIED_SNAPSHOT_PATH: Path | None = None
_FROZEN_SERIALIZE_BRANCH_GROUP = frozen.serialize_branch_group
_FROZEN_FAULT_SOURCE_READS = frozen.branch_checkpoint_fault_source_reads


def deterministic_runtime_contract() -> dict[str, Any]:
    return {
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


def require_deterministic_runtime_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = deterministic_runtime_contract()
    if manifest["pilot"].get("determinism") != expected:
        raise RuntimeError("larger-model pilot deterministic runtime contract drifted")
    observed_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed_cublas != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be supplied by the paid worker as "
            f"{CUBLAS_WORKSPACE_CONFIG!r}"
        )
    return expected


class PilotRepositoryRepairEnvironment(interface.RepositoryRepairEnvironment):
    """Use one root listing as the shared prefix before sibling localization."""

    def policy_prompt(self, phase: str) -> str:
        prompt = super().policy_prompt(phase)
        if phase != "shared_prefix":
            return prompt
        pilot_instruction = (
            "Expose the repository filename inventory for the branch checkpoint. "
            'Return exactly {"tool":"list","path":""}; do not read, search, test, edit, '
            "or finish before the checkpoint."
        )
        prompt_header = "PHASE INSTRUCTION\n"
        environment_boundary = "\n\n<untrusted-environment-data>\n"
        if not prompt.startswith(prompt_header) or environment_boundary not in prompt:
            raise RuntimeError("the structured phase prompt contract changed")
        _, environment_payload = prompt.split(environment_boundary, 1)
        return prompt_header + pilot_instruction + environment_boundary + environment_payload


def repository_root_observed_checkpoint(task: Any, prefix: Any) -> bool:
    """Admit branching after an accepted, complete root listing."""

    if prefix.terminal:
        return False
    expected_paths = sorted(task.files)
    for step in prefix.steps:
        if not step.accepted or step.tool != "list" or step.action != {"tool": "list", "path": ""}:
            continue
        try:
            observation = json.loads(step.observation)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(observation, dict) and observation.get("files") == expected_paths:
            return True
    return False


def _localization_telemetry(task: Any, environment: Any) -> dict[str, Any]:
    observed_paths = _FROZEN_FAULT_SOURCE_READS(task, environment)
    required_reads = len(task.faults)
    return {
        "required_fault_source_reads": required_reads,
        "observed_fault_source_paths": observed_paths,
        "all_fault_sources_observed": len(observed_paths) == required_reads,
    }


def fault_fixing_edit_and_fresh_read_actions(
    collection: Any,
    sibling_index: int,
) -> list[Any]:
    """Credit only causal reads and edits from a verified successful sibling."""

    if not 0 <= sibling_index < len(collection.siblings):
        raise IndexError("sibling index is outside the branch group")
    sibling = collection.siblings[sibling_index]
    if sibling.terminal_reason != "solved":
        return []
    generated_actions = collection.generated_by_sibling[sibling_index]
    post_branch_steps = sibling.steps[len(collection.prefix.steps) :]
    previous_fixed_faults = (
        int(getattr(collection.prefix.steps[-1], "fixed_faults", 0))
        if collection.prefix.steps
        else 0
    )
    credited: list[Any] = []
    for action_index, (generated, step) in enumerate(
        zip(generated_actions, post_branch_steps, strict=True)
    ):
        fixed_faults = int(getattr(step, "fixed_faults", previous_fixed_faults))
        is_fault_fixing_edit = (
            step.accepted
            and step.tool == "edit"
            and fixed_faults > previous_fixed_faults
            and bool(generated.input_ids)
        )
        if is_fault_fixing_edit:
            if action_index:
                preceding_generated = generated_actions[action_index - 1]
                preceding_step = post_branch_steps[action_index - 1]
                edited_path = step.action.get("path") if isinstance(step.action, dict) else None
                if (
                    preceding_step.accepted
                    and preceding_step.tool == "read"
                    and isinstance(preceding_step.action, dict)
                    and preceding_step.action.get("path") == edited_path
                    and preceding_generated.input_ids
                ):
                    credited.append(preceding_generated)
            credited.append(generated)
        previous_fixed_faults = fixed_faults
    return credited


def serialize_pilot_branch_group(
    collection: Any,
    *,
    update: int,
    optimizer_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep localization coverage visible without using it for branch admission."""

    serialized = _FROZEN_SERIALIZE_BRANCH_GROUP(
        collection,
        update=update,
        optimizer_update=optimizer_update,
    )
    shared_prefix = serialized["shared_prefix"]
    prefix_localization = _localization_telemetry(collection.task, collection.prefix)
    shared_prefix.update(
        {
            "checkpoint_strategy": SHARED_PREFIX_CHECKPOINT_STRATEGY,
            "required_diagnostic_actions": 1,
            "required_fault_source_reads": 0,
            "observed_fault_source_paths": [],
            "repository_root_observed": repository_root_observed_checkpoint(
                collection.task,
                collection.prefix,
            ),
        }
    )
    serialized["localization_telemetry"] = {
        "strategy": LOCALIZATION_TELEMETRY_STRATEGY,
        "branch_admission_gate": False,
        "prefix": prefix_localization,
        "siblings": [
            {
                "index": index,
                **_localization_telemetry(collection.task, sibling),
            }
            for index, sibling in enumerate(collection.siblings)
        ],
    }
    serialized["policy_credit_scope"] = POLICY_CREDIT_SCOPE
    return serialized


def reset_retained_update_evidence() -> None:
    """Reset process-local evidence before installing the one-shot pilot hooks."""

    global OBSERVED_POLICY_UPDATE_COUNT
    global RETAINED_CHECKPOINT_UPDATE
    global RETAINED_POLICY_UPDATE_COUNT

    OBSERVED_POLICY_UPDATE_COUNT = 0
    POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE.clear()
    RETAINED_CHECKPOINT_UPDATE = 0
    RETAINED_POLICY_UPDATE_COUNT = 0


def observe_retained_update_evidence(phase: str, values: dict[str, Any]) -> None:
    """Bind cumulative policy updates to the checkpoint retained for evaluation."""

    global OBSERVED_POLICY_UPDATE_COUNT
    global RETAINED_CHECKPOINT_UPDATE
    global RETAINED_POLICY_UPDATE_COUNT

    observed_update = values.get("update")
    observed_policy_updates = values.get(
        "effective_policy_update_count",
        values.get("policy_update_count"),
    )
    if (
        type(observed_update) is int
        and observed_update >= 1
        and type(observed_policy_updates) is int
        and observed_policy_updates >= 0
    ):
        POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE[observed_update] = observed_policy_updates
        OBSERVED_POLICY_UPDATE_COUNT = max(
            OBSERVED_POLICY_UPDATE_COUNT,
            observed_policy_updates,
        )

    observed_retained_policy_updates = values.get("retained_policy_update_count")
    if type(observed_retained_policy_updates) is int and observed_retained_policy_updates >= 0:
        RETAINED_POLICY_UPDATE_COUNT = observed_retained_policy_updates

    if phase != "finalizing":
        return
    best_validation = values.get("best_validation")
    retained_update = best_validation.get("update") if isinstance(best_validation, dict) else None
    if type(retained_update) is not int or retained_update < 0:
        raise RuntimeError("PILOT_RETAINED_CHECKPOINT_EVIDENCE_INVALID")
    RETAINED_CHECKPOINT_UPDATE = retained_update
    if type(observed_retained_policy_updates) is not int:
        RETAINED_POLICY_UPDATE_COUNT = POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE.get(
            retained_update,
            0,
        )


def require_retained_policy_update(manifest: dict[str, Any]) -> tuple[int, int]:
    """Return retained update evidence or refuse to spend on final evaluation."""

    minimum = manifest["pilot"]["minimum_effective_policy_updates"]
    if RETAINED_CHECKPOINT_UPDATE < 1 or minimum > RETAINED_POLICY_UPDATE_COUNT:
        raise RuntimeError("PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE")
    return RETAINED_CHECKPOINT_UPDATE, RETAINED_POLICY_UPDATE_COUNT


def require_offline_snapshot(manifest: dict[str, Any]) -> tuple[Path, str]:
    """Verify and privately materialize the pinned snapshot before model loading."""

    global VERIFIED_SNAPSHOT_PATH
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("larger-model workloads require Hugging Face offline mode")
    cache_root = Path(os.environ.get("HF_HOME", ""))
    if cache_root.as_posix() != manifest["artifact_readiness"]["cache_directory"]:
        raise RuntimeError("HF_HOME does not match the immutable larger-model profile")
    snapshot = (
        cache_root
        / "hub"
        / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
        / "snapshots"
        / gate.MODEL_REVISION
    )
    expected_digest = os.environ.get("EQUINOX_LARGER_MODEL_EXPECTED_SNAPSHOT_DIGEST")
    if expected_digest != gate.expected_snapshot_digest(manifest):
        raise RuntimeError("the pilot snapshot receipt does not match the immutable profile")
    free_bytes = os.statvfs(cache_root).f_bavail * os.statvfs(cache_root).f_frsize
    if free_bytes < manifest["hardware"]["minimum_free_cache_bytes"]:
        raise RuntimeError("the larger-model cache volume does not have the required free space")
    try:
        verification = gate.materialize_verified_snapshot(manifest, snapshot)
    except gate.GateError as error:
        raise RuntimeError(
            "the complete pinned 7B snapshot is not ready in the offline cache"
        ) from error
    if verification["snapshot_digest"] != expected_digest:
        raise RuntimeError("the pilot snapshot does not match the screened artifact digest")
    VERIFIED_SNAPSHOT_PATH = Path(verification["snapshot_path"])
    return VERIFIED_SNAPSHOT_PATH, verification["snapshot_digest"]


def require_pre_model_readiness(
    manifest: dict[str, Any],
    *,
    torch_module: Any | None = None,
    snapshot_verifier: Any | None = None,
) -> tuple[Path, str, gate.CUDAHardware]:
    """Reject a misallocated GPU before hashing the pinned 15 GB snapshot."""

    if torch_module is None:
        import torch

        torch_module = torch
    hardware = gate.require_cuda_hardware(manifest, torch_module)
    snapshot, snapshot_digest = (snapshot_verifier or require_offline_snapshot)(manifest)
    return snapshot, snapshot_digest, hardware


def require_authorization_digest() -> str:
    digest = os.environ.get("EQUINOX_LARGER_MODEL_AUTHORIZATION_DIGEST", "")
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise RuntimeError("the pilot requires a verified eligibility authorization digest")
    return digest


def require_private_materialization(manifest: dict[str, Any]) -> dict[str, str]:
    """Verify that pilot code and non-Torch dependencies come from private trees."""

    required_environment = (
        "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH",
        "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256",
        "EQUINOX_DEPENDENCY_LOCK_SHA256",
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
        "EQUINOX_DEPENDENCY_ROOT",
        "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH",
        "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256",
        "EQUINOX_CODE_PRIVATE_TREE_SHA256",
        "EQUINOX_CODE_ROOT",
        "EQUINOX_BUNDLE_SHA256",
        "EQUINOX_BUNDLE_SIZE_BYTES",
        "EQUINOX_BUNDLE_VOLUME_PATH",
    )
    environment = {name: os.environ.get(name, "") for name in required_environment}
    missing = sorted(name for name, value in environment.items() if not value)
    if missing:
        raise RuntimeError(
            "pilot private materialization evidence is missing: " + ", ".join(missing)
        )
    try:
        bundle_size_bytes = int(environment["EQUINOX_BUNDLE_SIZE_BYTES"])
        dependency_evidence = json.loads(
            Path(environment["EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH"]).read_text(
                encoding="utf-8"
            )
        )
        code_evidence = json.loads(
            Path(environment["EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("pilot private materialization evidence is unreadable") from error
    if not isinstance(dependency_evidence, dict) or not isinstance(code_evidence, dict):
        raise RuntimeError("pilot private materialization evidence is invalid")
    dependency_verification = gate.verify_dependency_quarantine_evidence(
        manifest,
        dependency_evidence,
        workload_bundle_path=environment["EQUINOX_BUNDLE_VOLUME_PATH"],
    )
    code_verification = gate.verify_code_materialization_evidence(
        manifest,
        code_evidence,
        workload_bundle_digest=environment["EQUINOX_BUNDLE_SHA256"],
        workload_bundle_size_bytes=bundle_size_bytes,
        workload_bundle_path=environment["EQUINOX_BUNDLE_VOLUME_PATH"],
    )
    expected = {
        "dependency_lock_digest": environment["EQUINOX_DEPENDENCY_LOCK_SHA256"],
        "dependency_quarantine_evidence_digest": environment[
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256"
        ],
        "dependency_private_tree_digest": environment["EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256"],
        "code_materialization_evidence_digest": environment[
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256"
        ],
        "code_private_tree_digest": environment["EQUINOX_CODE_PRIVATE_TREE_SHA256"],
    }
    observed = {
        "dependency_lock_digest": manifest["materialization"]["dependency_lock"]["digest"],
        "dependency_quarantine_evidence_digest": dependency_verification["evidence_digest"],
        "dependency_private_tree_digest": dependency_verification["private_tree_digest"],
        "code_materialization_evidence_digest": code_verification["evidence_digest"],
        "code_private_tree_digest": code_verification["private_tree_digest"],
    }
    if observed != expected:
        raise RuntimeError("pilot private materialization digest binding is invalid")
    try:
        code_root = Path(environment["EQUINOX_CODE_ROOT"]).resolve(strict=True)
        dependency_root = Path(environment["EQUINOX_DEPENDENCY_ROOT"]).resolve(strict=True)
        current_code_root = Path(__file__).resolve(strict=True).parent
        expected_code_root = Path(code_evidence["private_root"]).resolve(strict=True)
        expected_dependency_root = Path(dependency_evidence["private_root"]).resolve(strict=True)
    except OSError as error:
        raise RuntimeError("pilot private code root is unavailable") from error
    if (
        code_root != expected_code_root
        or current_code_root != expected_code_root
        or dependency_root != expected_dependency_root
    ):
        raise RuntimeError("pilot is not executing from the private materialization")
    gate.verify_dependency_import_smoke(
        manifest,
        dependency_root=Path(dependency_evidence["private_root"]),
    )
    return observed


def install_fail_closed_tokenizer_loader(transformers_module: Any) -> None:
    """Load model assets only from the immutable copy, guarding tokenizer width."""

    original_tokenizer_load = transformers_module.AutoTokenizer.from_pretrained
    original_model_load = transformers_module.AutoModelForCausalLM.from_pretrained

    def verified_call(
        original_load: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not args or args[0] != gate.MODEL_ID or kwargs.get("revision") != gate.MODEL_REVISION:
            raise RuntimeError("pilot model loader identity drifted from the profile")
        if kwargs.get("local_files_only") is False:
            raise RuntimeError("network model loading is forbidden")
        if VERIFIED_SNAPSHOT_PATH is None:
            raise RuntimeError("the verified local snapshot is unavailable")
        local_args = (str(VERIFIED_SNAPSHOT_PATH), *args[1:])
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        local_kwargs.pop("cache_dir", None)
        local_kwargs.pop("revision", None)
        return original_load(*local_args, **local_kwargs)

    def fail_closed_tokenizer_load(*args: Any, **kwargs: Any) -> Any:
        return eligibility.fail_closed_prompt_tokenizer(
            verified_call(original_tokenizer_load, args, kwargs)
        )

    def fail_closed_model_load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("attn_implementation") != ATTENTION_IMPLEMENTATION:
            raise RuntimeError("pilot model attention implementation drifted")
        return verified_call(original_model_load, args, kwargs)

    transformers_module.AutoTokenizer.from_pretrained = fail_closed_tokenizer_load
    transformers_module.AutoModelForCausalLM.from_pretrained = fail_closed_model_load


def install_memory_profile(manifest: dict[str, Any]) -> None:
    """Apply the screened 7B memory envelope without changing frozen sources."""

    def ensure_dependencies_and_patch_peft() -> None:
        observed_versions = {
            package: importlib.metadata.version(package)
            for package in manifest["runtime"]["dependencies"]
        }
        if observed_versions != manifest["runtime"]["dependencies"]:
            raise RuntimeError("the paid worker does not contain the manifest-pinned dependencies")
        import peft
        import transformers

        original_get_peft_model = peft.get_peft_model
        original_lora_config = peft.LoraConfig

        def pinned_lora_config(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("revision", gate.MODEL_REVISION)
            return original_lora_config(*args, **kwargs)

        def memory_bounded_get_peft_model(*args: Any, **kwargs: Any) -> Any:
            model = original_get_peft_model(*args, **kwargs)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            return model

        peft.LoraConfig = pinned_lora_config
        peft.get_peft_model = memory_bounded_get_peft_model
        install_fail_closed_tokenizer_loader(transformers)

    frozen.ensure_dependencies = ensure_dependencies_and_patch_peft
    frozen.TRAINING_MICROBATCH_SIZE = manifest["pilot"]["training_microbatch_size"]
    frozen.MAX_INPUT_TOKENS = manifest["pilot"]["maximum_input_tokens"]


def validate_runtime_configuration(runtime: Any, manifest: dict[str, Any]) -> None:
    pilot = manifest["pilot"]
    expected = {
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "target_runtime_seconds": pilot["target_runtime_seconds"],
        "maximum_updates": pilot["maximum_updates"],
        "validation_examples": pilot["validation_examples"],
        "test_examples": pilot["test_examples"],
        "training_tasks_per_update": pilot["training_tasks_per_update"],
        "replay_tasks_per_level": pilot["replay_tasks_per_level"],
        "mastery_windows": pilot["mastery_windows"],
        "maximum_final_evaluation_reserve_seconds": pilot[
            "maximum_final_evaluation_reserve_seconds"
        ],
        "optimization_seed": manifest["pilot_limits"]["optimization_seed"],
        "workload_attempt": 1,
    }
    for name, expected_value in expected.items():
        if getattr(runtime, name) != expected_value:
            raise ValueError(f"larger-model pilot requires {name}={expected_value!r}")


def install_bootstrap_checkpoint_contract(branch_width: int) -> None:
    """Install the K=4 root checkpoint and its narrowly causal policy credit."""

    if branch_width != 4:
        raise ValueError("the larger-model pilot requires static K=4")
    frozen.RepositoryRepairEnvironment = PilotRepositoryRepairEnvironment
    frozen.BRANCH_WIDTH = branch_width
    frozen_environment.BRANCH_WIDTH = branch_width
    frozen.SHARED_PREFIX_CHECKPOINT_STRATEGY = SHARED_PREFIX_CHECKPOINT_STRATEGY
    frozen.MINIMUM_PREFIX_ACCEPTED_ACTIONS = 1
    frozen.branch_checkpoint_reached = repository_root_observed_checkpoint
    frozen.fault_fixing_edit_actions = fault_fixing_edit_and_fresh_read_actions
    frozen.POLICY_CREDIT_SCOPE = POLICY_CREDIT_SCOPE
    frozen.serialize_branch_group = serialize_pilot_branch_group


def install_v33_contract(manifest: dict[str, Any], authorization_digest: str) -> None:
    """Install the new model/interface identity around the byte-frozen trainer."""

    pilot = manifest["pilot"]
    expected_identity = {
        "workload_revision": WORKLOAD_REVISION,
        "objective_id": OBJECTIVE_ID,
        "shared_prefix_checkpoint_strategy": SHARED_PREFIX_CHECKPOINT_STRATEGY,
        "localization_telemetry_strategy": LOCALIZATION_TELEMETRY_STRATEGY,
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
        "reward_contract_revision": REWARD_CONTRACT_REVISION,
        "retention_transaction_revision": RETENTION_TRANSACTION_REVISION,
        "learning_rate": LEARNING_RATE,
        "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
        "maximum_consecutive_regression_windows": (MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS),
        "determinism": deterministic_runtime_contract(),
        "training_level_allocation": training_level_allocation_contract(),
    }
    for name, expected in expected_identity.items():
        if pilot[name] != expected:
            raise RuntimeError(f"larger-model pilot {name} drifted")
    reset_retained_update_evidence()
    original_emit_progress = frozen.emit_progress
    original_training_stop_decision = frozen.training_stop_decision
    original_make_tasks = frozen.make_tasks

    def pilot_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        observe_retained_update_evidence(phase, values)
        values.setdefault("retention_transaction_revision", RETENTION_TRANSACTION_REVISION)
        values.setdefault("learning_rate", LEARNING_RATE)
        values.setdefault("reference_kl_coefficient", REFERENCE_KL_COEFFICIENT)
        if phase == "model_loading":
            import torch

            hardware = gate.require_cuda_hardware(manifest, torch)
            values.update(
                gpu_id=hardware.gpu_name,
                gpu_name=hardware.gpu_name,
                gpu_total_memory_bytes=hardware.total_memory_bytes,
                gpu_memory_gb=round(hardware.total_memory_bytes / 1_000_000_000, 3),
                gpu_memory_gib=round(hardware.total_memory_bytes / 2**30, 3),
            )
        original_emit_progress(
            phase,
            message,
            runtime_configuration,
            preserve_context=preserve_context,
            profile_id=manifest["profile_id"],
            model_revision=gate.MODEL_REVISION,
            authorization_digest=authorization_digest,
            **values,
        )

    frozen.emit_progress = pilot_progress
    frozen.SUPPORTED_MODELS[gate.MODEL_ID] = gate.MODEL_REVISION
    if frozen.TRANSACTIONAL_RETENTION_REVISION != RETENTION_TRANSACTION_REVISION:
        raise RuntimeError("the trainer retention transaction contract drifted")
    if (
        frozen.DETERMINISTIC_RUNTIME_REVISION != DETERMINISTIC_RUNTIME_REVISION
        or frozen.ATTENTION_IMPLEMENTATION != ATTENTION_IMPLEMENTATION
        or frozen.CUBLAS_WORKSPACE_CONFIG != CUBLAS_WORKSPACE_CONFIG
        or frozen.TRAINING_LEVEL_ALLOCATION_REVISION != TRAINING_LEVEL_ALLOCATION_REVISION
        or frozen.training_level_allocation_contract() != training_level_allocation_contract()
    ):
        raise RuntimeError("the trainer paid runtime contract drifted")
    frozen.WORKLOAD_REVISION = WORKLOAD_REVISION
    frozen.OBJECTIVE_ID = OBJECTIVE_ID
    frozen.ACTIVE_RETENTION_TRANSACTION_REVISION = RETENTION_TRANSACTION_REVISION
    frozen.LEARNING_RATE = LEARNING_RATE
    frozen.REFERENCE_KL_COEFFICIENT = REFERENCE_KL_COEFFICIENT
    frozen.MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS = MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS
    frozen.SYSTEM_PROMPT = interface.SYSTEM_PROMPT
    frozen.ACTION_PROTOCOL_REVISION = interface.ACTION_PROTOCOL_REVISION
    frozen.ENVIRONMENT_REVISION = interface.ENVIRONMENT_REVISION
    frozen_environment.ENVIRONMENT_REVISION = interface.ENVIRONMENT_REVISION
    frozen_environment.ACTION_PROTOCOL_REVISION = interface.ACTION_PROTOCOL_REVISION
    install_bootstrap_checkpoint_contract(manifest["screen"]["branch_width"])

    def useful_training_stop_decision(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("final_evaluation_reserve_exceeded_ceiling") is True:
            raise RuntimeError("PILOT_RUNTIME_INFEASIBLE_BEFORE_TRAINING")
        return original_training_stop_decision(*args, **kwargs)

    def useful_pilot_tasks(
        level: int,
        count: int,
        seed: int,
        *,
        split: str = "train",
        exclude_semantic_task_ids: frozenset[str] = frozenset(),
    ) -> Any:
        if split == "test":
            require_retained_policy_update(manifest)
        return original_make_tasks(
            level,
            count,
            seed,
            split=split,
            exclude_semantic_task_ids=exclude_semantic_task_ids,
        )

    frozen.training_stop_decision = useful_training_stop_decision
    frozen.make_tasks = useful_pilot_tasks
    install_memory_profile(manifest)


def augment_result(
    result: dict[str, Any],
    *,
    manifest: dict[str, Any],
    authorization_digest: str,
    snapshot: Path,
    materialization_evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch

    hardware = gate.require_cuda_hardware(manifest, torch)
    retained_update, retained_policy_updates = require_retained_policy_update(manifest)
    best_validation = result.get("best_validation")
    result_retained_update = (
        best_validation.get("update") if isinstance(best_validation, dict) else None
    )
    total_policy_updates = result.get("policy_update_count")
    attempted_policy_updates = result.get("attempted_policy_update_count")
    effective_policy_updates = result.get("effective_policy_update_count")
    result_retained_policy_updates = result.get("retained_policy_update_count")
    optimization_seed = manifest["pilot_limits"]["optimization_seed"]
    reward_contract = result.get("reward_contract")
    if (
        not isinstance(reward_contract, dict)
        or reward_contract.get("revision") != REWARD_CONTRACT_REVISION
    ):
        raise RuntimeError("the larger-model pilot reward contract drifted")
    if result.get("seed") != optimization_seed:
        raise RuntimeError("the larger-model pilot optimization seed drifted")
    expected_determinism = deterministic_runtime_contract()
    if result.get("determinism") != {
        **expected_determinism,
        "cuda_seeded_all_devices": True,
    }:
        raise RuntimeError("the larger-model pilot deterministic runtime evidence drifted")
    promotions = result.get("promotions")
    updates_completed = result.get("updates_completed")
    if not isinstance(promotions, list) or type(updates_completed) is not int:
        raise RuntimeError("the larger-model pilot allocation transition evidence is absent")
    expected_allocation_contract = training_level_allocation_contract()
    expected_allocation_transition = frozen.training_level_allocation_transition_evidence(
        promotions,
        observed_updates=updates_completed,
    )
    if (
        result.get("training_level_allocation_contract") != expected_allocation_contract
        or result.get("training_level_allocation_transition") != expected_allocation_transition
    ):
        raise RuntimeError("the larger-model pilot allocation transition evidence drifted")
    if (
        result_retained_update != retained_update
        or type(total_policy_updates) is not int
        or total_policy_updates < retained_policy_updates
        or attempted_policy_updates != total_policy_updates
        or effective_policy_updates != retained_policy_updates
        or result_retained_policy_updates != retained_policy_updates
        or result.get("retention_transaction_revision") != RETENTION_TRANSACTION_REVISION
    ):
        raise RuntimeError("the larger-model pilot retained-update evidence is inconsistent")
    training_configuration = result.get("training_configuration")
    if (
        not isinstance(training_configuration, dict)
        or training_configuration.get("deterministic_runtime") != expected_determinism
        or training_configuration.get("maximum_consecutive_regression_windows")
        != MAXIMUM_CONSECUTIVE_REGRESSION_WINDOWS
        or training_configuration.get("training_level_allocation") != expected_allocation_contract
    ):
        raise RuntimeError("the larger-model pilot training runtime evidence drifted")
    result = {
        **result,
        "training_configuration": {
            **training_configuration,
            "shared_prefix_checkpoint": SHARED_PREFIX_CHECKPOINT_STRATEGY,
            "minimum_shared_prefix_actions": 1,
            "policy_credit_scope": POLICY_CREDIT_SCOPE,
            "learning_signal": LEARNING_SIGNAL,
            "retention_transaction_revision": RETENTION_TRANSACTION_REVISION,
            "learning_rate": LEARNING_RATE,
            "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
        },
    }
    private_evidence = materialization_evidence or {}
    return {
        **result,
        "workload": PILOT_WORKLOAD,
        "workload_revision": WORKLOAD_REVISION,
        "objective_id": OBJECTIVE_ID,
        "profile_id": manifest["profile_id"],
        "model_id": gate.MODEL_ID,
        "model_revision": gate.MODEL_REVISION,
        "optimization_seed": optimization_seed,
        "source_contract_digest": gate.expected_source_contract_digest(manifest),
        "authorization_digest": authorization_digest,
        "environment_revision": interface.ENVIRONMENT_REVISION,
        "action_protocol_revision": interface.ACTION_PROTOCOL_REVISION,
        "terminal_submission_contract": interface.TERMINAL_SUBMISSION_CONTRACT,
        "gpu_id": hardware.gpu_name,
        "gpu_name": hardware.gpu_name,
        "gpu_total_memory_bytes": hardware.total_memory_bytes,
        "gpu_memory_gb": round(hardware.total_memory_bytes / 1_000_000_000, 3),
        "gpu_memory_gib": round(hardware.total_memory_bytes / 2**30, 3),
        "offline_snapshot": str(snapshot),
        "pinned_snapshot_digest": os.environ["EQUINOX_LARGER_MODEL_EXPECTED_SNAPSHOT_DIGEST"],
        "network_volume_id": os.environ["EQUINOX_RUNPOD_NETWORK_VOLUME_ID"],
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_lock_digest": private_evidence.get("dependency_lock_digest"),
        "dependency_quarantine_evidence_digest": private_evidence.get(
            "dependency_quarantine_evidence_digest"
        ),
        "dependency_private_tree_digest": private_evidence.get("dependency_private_tree_digest"),
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": private_evidence.get(
            "code_materialization_evidence_digest"
        ),
        "code_private_tree_digest": private_evidence.get("code_private_tree_digest"),
        "offline_mode_active": True,
        "training_microbatch_size": frozen.TRAINING_MICROBATCH_SIZE,
        "maximum_input_tokens": frozen.MAX_INPUT_TOKENS,
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_use_reentrant": False,
        "deterministic_runtime_revision": DETERMINISTIC_RUNTIME_REVISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "training_level_allocation_revision": (TRAINING_LEVEL_ALLOCATION_REVISION),
        "retained_checkpoint_update": retained_update,
        "effective_policy_update_count": retained_policy_updates,
        "retained_policy_update_count": retained_policy_updates,
        "retention_transaction_revision": RETENTION_TRANSACTION_REVISION,
        "learning_rate": LEARNING_RATE,
        "reference_kl_coefficient": REFERENCE_KL_COEFFICIENT,
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
    }


def main() -> None:
    manifest = gate.load_manifest()
    gate.verify_source_contract(manifest)
    if os.environ.get("EQUINOX_RL_MODEL_ID") != gate.MODEL_ID:
        raise ValueError("the larger-model pilot requires the manifest-pinned model")
    materialization_evidence = require_private_materialization(manifest)
    authorization_digest = require_authorization_digest()
    require_deterministic_runtime_environment(manifest)
    study.verify_transactional_sources()
    install_v33_contract(manifest, authorization_digest)
    runtime = frozen.configure_from_environment()
    validate_runtime_configuration(runtime, manifest)
    if "--validate-configuration" in sys.argv:
        return
    if "--self-test" in sys.argv:
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "profile_id": manifest["profile_id"],
                    "workload_revision": WORKLOAD_REVISION,
                },
                sort_keys=True,
            )
        )
        return
    snapshot, _snapshot_digest, _hardware = require_pre_model_readiness(manifest)
    result = study.run_trainer_and_capture_result()
    print(
        json.dumps(
            augment_result(
                result,
                manifest=manifest,
                authorization_digest=authorization_digest,
                snapshot=snapshot,
                materialization_evidence=materialization_evidence,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
