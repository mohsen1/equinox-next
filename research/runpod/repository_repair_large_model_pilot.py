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
    import repository_repair_rl as frozen
    import repository_repair_study as study
except ModuleNotFoundError:
    from . import larger_model_gate as gate
    from . import repository_repair_env as frozen_environment
    from . import repository_repair_env_v33 as interface
    from . import repository_repair_large_model_eligibility as eligibility
    from . import repository_repair_rl as frozen
    from . import repository_repair_study as study


WORKLOAD_REVISION = "runpod-repository-repair-large-model-pilot@4"
OBJECTIVE_ID = "verified-repair-chain-root-branch-retention-policy-gradient@17"
REWARD_CONTRACT_REVISION = "correctness-gated-efficiency@1"
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
_FROZEN_SERIALIZE_BRANCH_GROUP = frozen.serialize_branch_group
_FROZEN_FAULT_SOURCE_READS = frozen.branch_checkpoint_fault_source_reads


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


def install_fail_closed_tokenizer_loader(transformers_module: Any) -> None:
    """Wrap every pilot tokenizer with the shared no-truncation guard."""

    original_tokenizer_load = transformers_module.AutoTokenizer.from_pretrained

    def fail_closed_tokenizer_load(*args: Any, **kwargs: Any) -> Any:
        return eligibility.fail_closed_prompt_tokenizer(original_tokenizer_load(*args, **kwargs))

    transformers_module.AutoTokenizer.from_pretrained = fail_closed_tokenizer_load


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
    reward_contract = result.get("reward_contract")
    if (
        not isinstance(reward_contract, dict)
        or reward_contract.get("revision") != REWARD_CONTRACT_REVISION
    ):
        raise RuntimeError("the larger-model pilot reward contract drifted")
    if result.get("seed") != optimization_seed:
        raise RuntimeError("the larger-model pilot optimization seed drifted")
    if (
        result_retained_update != retained_update
        or type(total_policy_updates) is not int
        or total_policy_updates < retained_policy_updates
    ):
        raise RuntimeError("the larger-model pilot retained-update evidence is inconsistent")
    training_configuration = result.get("training_configuration")
    if isinstance(training_configuration, dict):
        result = {
            **result,
            "training_configuration": {
                **training_configuration,
                "shared_prefix_checkpoint": SHARED_PREFIX_CHECKPOINT_STRATEGY,
                "minimum_shared_prefix_actions": 1,
                "policy_credit_scope": POLICY_CREDIT_SCOPE,
                "learning_signal": LEARNING_SIGNAL,
            },
        }
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
        "offline_mode_active": True,
        "training_microbatch_size": frozen.TRAINING_MICROBATCH_SIZE,
        "maximum_input_tokens": frozen.MAX_INPUT_TOKENS,
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_use_reentrant": False,
        "retained_checkpoint_update": retained_update,
        "effective_policy_update_count": retained_policy_updates,
        "policy_credit_scope": POLICY_CREDIT_SCOPE,
    }


def main() -> None:
    manifest = gate.load_manifest()
    gate.verify_source_contract(manifest)
    if os.environ.get("EQUINOX_RL_MODEL_ID") != gate.MODEL_ID:
        raise ValueError("the larger-model pilot requires the manifest-pinned model")
    authorization_digest = require_authorization_digest()
    study.verify_frozen_sources()
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
