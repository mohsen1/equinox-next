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
    import repository_repair_env_v32 as interface
    import repository_repair_rl as frozen
    import repository_repair_study as study
except ModuleNotFoundError:
    from . import larger_model_gate as gate
    from . import repository_repair_env as frozen_environment
    from . import repository_repair_env_v32 as interface
    from . import repository_repair_rl as frozen
    from . import repository_repair_study as study


WORKLOAD_REVISION = "runpod-repository-repair-large-model-pilot@1"
OBJECTIVE_ID = "verified-fix-coverage-retention-policy-gradient@15"
SHARED_PREFIX_CHECKPOINT_STRATEGY = "all_fault_sources_observed"
MODEL_REVISION = gate.MODEL_REVISION
PILOT_WORKLOAD = "repository-repair-restored-continuation-post-training"
OBSERVED_POLICY_UPDATE_COUNT = 0
POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE: dict[int, int] = {}
RETAINED_CHECKPOINT_UPDATE = 0
RETAINED_POLICY_UPDATE_COUNT = 0


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
    observed_policy_updates = values.get("policy_update_count")
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

    if phase != "finalizing":
        return
    best_validation = values.get("best_validation")
    retained_update = best_validation.get("update") if isinstance(best_validation, dict) else None
    if type(retained_update) is not int or retained_update < 0:
        raise RuntimeError("PILOT_RETAINED_CHECKPOINT_EVIDENCE_INVALID")
    RETAINED_CHECKPOINT_UPDATE = retained_update
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
    """Verify the exact pinned four-shard snapshot before model initialization."""

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
    try:
        verification = gate.verify_local_snapshot(manifest, snapshot)
    except gate.GateError as error:
        raise RuntimeError(
            "the complete pinned 7B snapshot is not ready in the offline cache"
        ) from error
    expected_digest = os.environ.get("EQUINOX_LARGER_MODEL_EXPECTED_SNAPSHOT_DIGEST")
    if verification["snapshot_digest"] != expected_digest:
        raise RuntimeError("the pilot snapshot does not match the screened artifact digest")
    free_bytes = os.statvfs(cache_root).f_bavail * os.statvfs(cache_root).f_frsize
    if free_bytes < manifest["hardware"]["minimum_free_cache_bytes"]:
        raise RuntimeError("the larger-model cache volume does not have the required free space")
    return snapshot, verification["snapshot_digest"]


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


def install_v32_contract(manifest: dict[str, Any], authorization_digest: str) -> None:
    """Install the new model/interface identity around the byte-frozen trainer."""

    reset_retained_update_evidence()
    original_emit_progress = frozen.emit_progress
    original_checkpoint_reached = frozen.branch_checkpoint_reached
    original_training_stop_decision = frozen.training_stop_decision
    original_make_tasks = frozen.make_tasks

    def nonterminal_checkpoint(task: Any, prefix: Any) -> bool:
        return not prefix.terminal and original_checkpoint_reached(task, prefix)

    def pilot_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        observe_retained_update_evidence(phase, values)
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
    frozen.WORKLOAD_REVISION = WORKLOAD_REVISION
    frozen.OBJECTIVE_ID = OBJECTIVE_ID
    frozen.SYSTEM_PROMPT = interface.SYSTEM_PROMPT
    frozen.ACTION_PROTOCOL_REVISION = interface.ACTION_PROTOCOL_REVISION
    frozen.ENVIRONMENT_REVISION = interface.ENVIRONMENT_REVISION
    frozen.RepositoryRepairEnvironment = interface.RepositoryRepairEnvironment
    frozen_environment.ENVIRONMENT_REVISION = interface.ENVIRONMENT_REVISION
    frozen_environment.ACTION_PROTOCOL_REVISION = interface.ACTION_PROTOCOL_REVISION
    frozen.BRANCH_WIDTH = manifest["screen"]["branch_width"]
    frozen_environment.BRANCH_WIDTH = manifest["screen"]["branch_width"]
    frozen.branch_checkpoint_reached = nonterminal_checkpoint

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
) -> dict[str, Any]:
    import torch

    hardware = gate.require_cuda_hardware(manifest, torch)
    retained_update, retained_policy_updates = require_retained_policy_update(manifest)
    best_validation = result.get("best_validation")
    result_retained_update = (
        best_validation.get("update") if isinstance(best_validation, dict) else None
    )
    total_policy_updates = result.get("policy_update_count")
    optimization_seed = manifest["pilot_limits"]["optimization_seed"]
    if result.get("seed") != optimization_seed:
        raise RuntimeError("the larger-model pilot optimization seed drifted")
    if (
        result_retained_update != retained_update
        or type(total_policy_updates) is not int
        or total_policy_updates < retained_policy_updates
    ):
        raise RuntimeError("the larger-model pilot retained-update evidence is inconsistent")
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
        "gpu_id": hardware.gpu_name,
        "gpu_name": hardware.gpu_name,
        "gpu_total_memory_bytes": hardware.total_memory_bytes,
        "gpu_memory_gb": round(hardware.total_memory_bytes / 1_000_000_000, 3),
        "gpu_memory_gib": round(hardware.total_memory_bytes / 2**30, 3),
        "offline_snapshot": str(snapshot),
        "pinned_snapshot_digest": os.environ["EQUINOX_LARGER_MODEL_EXPECTED_SNAPSHOT_DIGEST"],
        "network_volume_id": os.environ["EQUINOX_RUNPOD_NETWORK_VOLUME_ID"],
        "offline_mode_active": True,
        "training_microbatch_size": frozen.TRAINING_MICROBATCH_SIZE,
        "maximum_input_tokens": frozen.MAX_INPUT_TOKENS,
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_use_reentrant": False,
        "retained_checkpoint_update": retained_update,
        "effective_policy_update_count": retained_policy_updates,
    }


def main() -> None:
    manifest = gate.load_manifest()
    gate.verify_source_contract(manifest)
    if os.environ.get("EQUINOX_RL_MODEL_ID") != gate.MODEL_ID:
        raise ValueError("the larger-model pilot requires the manifest-pinned model")
    authorization_digest = require_authorization_digest()
    study.verify_frozen_sources()
    install_v32_contract(manifest, authorization_digest)
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
    result = study.run_frozen_and_capture_result()
    print(
        json.dumps(
            augment_result(
                result,
                manifest=manifest,
                authorization_digest=authorization_digest,
                snapshot=snapshot,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
