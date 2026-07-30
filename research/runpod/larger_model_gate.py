"""Fail-closed eligibility contract for the first 7B RunPod pilot.

The profile is intentionally immutable. Changing a model, runtime image, dependency,
hardware class, budget, screen size, or threshold requires a new profile identity and
an accompanying code change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@10"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
MODEL_PARAMETER_COUNT = 7_615_616_512
MODEL_SAFETENSORS_BYTES = 15_231_271_864
SNAPSHOT_FILES = {
    ".gitattributes": {
        "size_bytes": 1_519,
        "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    },
    "LICENSE": {
        "size_bytes": 11_343,
        "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    },
    "README.md": {
        "size_bytes": 6_392,
        "sha256": "3c090be37f829adc1e4cdb78733667437732541470430ab2cd785d7f0d460077",
    },
    "config.json": {
        "size_bytes": 663,
        "sha256": "c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28",
    },
    "generation_config.json": {
        "size_bytes": 242,
        "sha256": "1a628a5775bc69cde01c6749a531150ca4d3189652c618a174f7077923acf3b1",
    },
    "merges.txt": {
        "size_bytes": 1_671_839,
        "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    "model-00001-of-00004.safetensors": {
        "size_bytes": 4_877_660_776,
        "sha256": "0b6f069918b07c064cbba8ae4f00f529aa9bbf84b7cdfcb7fc2694a40f6aa8ef",
    },
    "model-00002-of-00004.safetensors": {
        "size_bytes": 4_932_751_008,
        "sha256": "c3d46733e7aa054ea7b063fbccd0a5a08446e7bd1814bef26936c5aa1331da62",
    },
    "model-00003-of-00004.safetensors": {
        "size_bytes": 4_330_865_200,
        "sha256": "9fe45dacee087385b3d2d6dd27a7413a8a56d95f145772facc148fa86fc73446",
    },
    "model-00004-of-00004.safetensors": {
        "size_bytes": 1_089_994_880,
        "sha256": "5aa6e5cbe642377fd441fb4e60e83cca96b2bcd9820e245b9ea06d94653f17f2",
    },
    "model.safetensors.index.json": {
        "size_bytes": 27_752,
        "sha256": "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb",
    },
    "tokenizer.json": {
        "size_bytes": 7_031_645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    "tokenizer_config.json": {
        "size_bytes": 7_305,
        "sha256": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    },
    "vocab.json": {
        "size_bytes": 2_776_833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
SNAPSHOT_AUXILIARY_FILES = frozenset({".gitattributes", "LICENSE", "README.md"})
PREDECESSOR_SNAPSHOT_FILES = {
    name: metadata
    for name, metadata in SNAPSHOT_FILES.items()
    if name not in SNAPSHOT_AUXILIARY_FILES
}
SOURCE_CONTRACT_SHA256 = {
    "larger-model-dependencies.lock": (
        "bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
    ),
    "repository_repair_env.py": (
        "527050c5444a3731119773d4e0932840068fda9664654dce9bf0350edf56e40f"
    ),
    "repository_repair_env_v31.py": (
        "703c5badc19513cf8a7766a1a1e63fa77dce49a2f0011d78d9f4b94b64132657"
    ),
    "repository_repair_env_v32.py": (
        "fefade752e6bd82530835a5de493fffe7629a6c8963ae6f5fe7e3073dd1d2557"
    ),
    "repository_repair_env_v33.py": (
        "02b57b074c72e81ae3c72ab82121d226656378ab47c8aa51215091a913c41285"
    ),
    "repository_repair_large_model_eligibility.py": (
        "8a1dc7161760a7cf8a2ca738bcfe4fa0e513b80df6ee073c874002d289800a18"
    ),
    "repository_repair_large_model_pilot.py": (
        "4f67ee3a3c6b90c9fc3b53f477f3981359a4d67d241ebf959999d9a22f41dd80"
    ),
    "repository_repair_large_model_study.py": (
        "9535783495e7584ad650e280a80e6aa03df6f91f27b74b6ef7859878d6f981f4"
    ),
    "repository_repair_large_model_trainer.py": (
        "50e71174e4a4c897009e99730e2c7bfd2abd5897b89196cdfc53ae0396791ee2"
    ),
    "retention_checkpoint_probe.py": (
        "4277e92188b53f41a5201ff6b01ffb9d4fdac925eb38462d0d9d930b62aa2e38"
    ),
}
SCREEN_WORKLOAD = "repository-repair-larger-model-eligibility-screen"
SCREEN_WORKLOAD_REVISION = "larger-model-eligibility-screen@10"
CAPPED_GENERATION_TOKENS = 192
CLEANUP_COST_RESERVE_SECONDS = 120
TEARDOWN_RESERVE_SECONDS = 120
BUNDLE_HANDOFF_REVISION = "runpod-volume-bundle-handoff@1"
LIVE_STAGE_ACTIVATION_REVISION = "authenticated-proxy-stage-activation@2"
RETENTION_CHECKPOINT_EVIDENCE_REVISION = "real-adamw-persist-restore-advance@7"
CHECKPOINT_AUTHENTICATION_REVISION = "launch-bound-checkpoint-manifest@1"
CHECKPOINT_AUTHENTICATION_MECHANISM = {
    "revision": CHECKPOINT_AUTHENTICATION_REVISION,
    "manifest_schema_version": 1,
    "pointer_schema_version": 2,
    "file_validation": "dirfd-nofollow-bounded-single-link-regular@1",
    "private_materialization": "digest-verified-private-copy@1",
    "training_state_load": "torch-weights-only@1",
    "replay_binding": "runner-private-generation-and-manifest-digest@1",
}
VOLUME_READINESS_ATTESTATION_REVISION = "runpod-h100-volume-readiness@2"
DEPENDENCY_QUARANTINE_REVISION = "hash-locked-private-dependencies@1"
CODE_MATERIALIZATION_REVISION = "private-code-materialization@1"
DEPENDENCY_INSTALLER_REVISION = "isolated-pip-binary-hash-lock@1"
DEPENDENCY_LOCK_MEMBER = "larger-model-dependencies.lock"
DEPENDENCY_LOCK_SIZE_BYTES = 30_866
DEPENDENCY_LOCK_SHA256 = "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
PREDECESSOR_READINESS_BRIDGE_REVISION = "exact-predecessor-readiness-seed@1"
PREDECESSOR_PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@6"
PREDECESSOR_MANIFEST_COMMIT = "f0b83d62591b9b2a43ca1aa29a554b23e1a31300"
PREDECESSOR_MANIFEST_DIGEST = (
    "sha256:e3637e7ba95eaeb23caa3747e592918b1d48659a3fa8f30c287982f4872029f2"
)
PREDECESSOR_SNAPSHOT_DIGEST = (
    "sha256:54c5fe22c88f933cd11f35f6da3146376f3d3741e952e385c7a96931ab1058ba"
)
MAXIMUM_WORKLOAD_BUNDLE_BYTES = 2 * 1024 * 1024
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "studies/larger-model-eligibility.json"
)

REQUIRED_GATE_RESULTS = (
    "baseline_checkpoint_rate",
    "action_protocol_validity",
    "branch_checkpoint_rate",
    "informative_group_rate",
    "solved_sibling_rate",
    "baseline_exact_rate",
    "peak_reserved_vram_within_limit",
    "pinned_snapshot_ready",
    "offline_mode_active",
    "hardware_verified",
    "deterministic_runtime_verified",
    "pilot_runtime_feasible",
    "policy_unchanged",
    "optimizer_state_restored",
    "test_split_isolated",
    "preparation_evidence_complete",
)

_EXPECTED_MANIFEST: dict[str, Any] = {
    "schema_version": 1,
    "profile_id": PROFILE_ID,
    "model": {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "safetensors_bytes": MODEL_SAFETENSORS_BYTES,
        "snapshot_files": SNAPSHOT_FILES,
        "dtype": "bfloat16",
    },
    "runtime": {
        "template_id": "runpod-torch-v280",
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "image_digest": "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851",
        "torch_version": "2.8.0+cu128",
        "dependencies": {
            "accelerate": "1.14.0",
            "peft": "0.19.1",
            "transformers": "5.14.1",
        },
    },
    "interface": {
        "environment_revision": "repository-repair-simulator@9",
        "action_protocol_revision": "repository-repair-json-tools@8",
        "terminal_submission_contract": "accepted-passing-test-or-finish@1",
    },
    "materialization": {
        "code": {
            "revision": CODE_MATERIALIZATION_REVISION,
            "evidence_schema_version": 1,
        },
        "dependency_lock": {
            "path": DEPENDENCY_LOCK_MEMBER,
            "digest": DEPENDENCY_LOCK_SHA256,
            "size_bytes": DEPENDENCY_LOCK_SIZE_BYTES,
            "revision": DEPENDENCY_QUARANTINE_REVISION,
            "evidence_schema_version": 2,
            "installer_revision": DEPENDENCY_INSTALLER_REVISION,
            "python_version": "3.12",
            "platform_tag": "manylinux_2_28_x86_64",
            "index_url": "https://pypi.org/simple",
            "network_policy": ("hash-locked-binary-wheels-during-authenticated-preparation-only"),
        },
    },
    "source_contract": {
        "algorithm": "sha256",
        "files": SOURCE_CONTRACT_SHA256,
    },
    "hardware": {
        "provider": "runpod",
        "cloud_type": "SECURE",
        "gpu_id": "NVIDIA H100 80GB HBM3",
        "gpu_display_name": "H100 SXM",
        "minimum_gpu_memory_gb": 80,
        "minimum_cuda_memory_bytes": 78_000_000_000,
        "container_disk_gb": 50,
        "volume_disk_gb": 50,
        "minimum_free_cache_bytes": 25_000_000_000,
        "maximum_peak_reserved_vram_fraction": 0.85,
    },
    "artifact_readiness": {
        "cache_directory": "/workspace/equinox-state/huggingface",
        "require_complete_pinned_snapshot_before_screen": True,
        "require_offline_mode_after_readiness": True,
        "allow_network_model_download_during_screen": False,
    },
    "provider_safety": {
        "maximum_storage_only_hourly_spend_usd": 0.01,
    },
    "screen_limits": {
        "maximum_hourly_cost_usd": 4.0,
        "maximum_total_cost_usd": 3.35,
        "maximum_lifetime_seconds": 2_880,
        "conservative_billing_seconds": 3_000,
        "boot_timeout_seconds": 600,
        "model_load_timeout_seconds": 1_200,
        "stale_progress_timeout_seconds": 600,
        "maximum_workload_attempts": 1,
        "target_runtime_seconds": 1_500,
        "retry_reserve_seconds": 300,
        "maximum_final_evaluation_reserve_seconds": 300,
        "artifact_retrieval_reserve_seconds": 60,
        "optimization_seed": 137,
    },
    "pilot_limits": {
        "maximum_hourly_cost_usd": 3.25,
        "maximum_total_cost_usd": 13.0,
        "maximum_lifetime_seconds": 14_280,
        "boot_timeout_seconds": 360,
        "model_load_timeout_seconds": 1_200,
        "stale_progress_timeout_seconds": 900,
        "maximum_workload_attempts": 1,
        "target_runtime_seconds": 9_000,
        "retry_reserve_seconds": 1_800,
        "maximum_final_evaluation_reserve_seconds": 1_800,
        "optimization_seed": 137,
    },
    "screen": {
        "workload": SCREEN_WORKLOAD,
        "workload_revision": SCREEN_WORKLOAD_REVISION,
        "admission_levels": [0],
        "shared_prefix_checkpoint_strategy": "repository_root_observed@1",
        "localization_telemetry_strategy": "all_fault_sources_observed",
        "branch_width": 4,
        "live_stage_activation_revision": LIVE_STAGE_ACTIVATION_REVISION,
        "validation_examples": 8,
        "training_tasks_per_update": 8,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 2_048,
        "maximum_updates": 1,
        "test_examples": 0,
        "baseline_examples_per_level": 8,
        "minimum_completed_baseline_examples": 8,
        "branch_groups": 8,
        "thresholds": {
            "minimum_baseline_checkpoint_rate": 0.8,
            "minimum_per_level_checkpoint_rate": 0.75,
            "minimum_action_protocol_validity": 0.99,
            "minimum_branch_checkpoint_rate": 0.75,
            "minimum_informative_group_rate": 0.1,
            "minimum_informative_groups": 2,
            "minimum_solved_siblings": 2,
            "minimum_failed_siblings": 2,
            "minimum_solved_sibling_rate": 0.05,
            "maximum_solved_sibling_rate": 0.8,
            "minimum_baseline_exact_rate": 0.0,
            "maximum_baseline_exact_rate": 0.75,
            "maximum_predicted_final_evaluation_seconds": 1_440,
        },
    },
    "pilot": {
        "workload_revision": "runpod-repository-repair-large-model-pilot@7",
        "objective_id": "verified-repair-chain-transactional-retention-policy-gradient@19",
        "reward_contract_revision": "correctness-gated-efficiency@1",
        "shared_prefix_checkpoint_strategy": "repository_root_observed@1",
        "localization_telemetry_strategy": "all_fault_sources_observed",
        "policy_credit_scope": (
            "fault_fixing_edits_and_immediately_upstream_fresh_reads_"
            "from_verified_successful_siblings"
        ),
        "target_runtime_seconds": 9_000,
        "maximum_updates": 40,
        "validation_examples": 8,
        "test_examples": 12,
        "training_tasks_per_update": 4,
        "replay_tasks_per_level": 1,
        "mastery_windows": 2,
        "maximum_final_evaluation_reserve_seconds": 1_800,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 2_048,
        "retention_transaction_revision": "adapter-optimizer-policy-lineage@1",
        "learning_rate": 1e-5,
        "reference_kl_coefficient": 1.0,
        "maximum_consecutive_regression_windows": 4,
        "determinism": {
            "revision": "eager-math-sdp-deterministic@1",
            "attention_implementation": "eager",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "flash_sdp_enabled": False,
            "memory_efficient_sdp_enabled": False,
            "math_sdp_enabled": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
        },
        "training_level_allocation": {
            "revision": "retained-promotion-3-1-to-2-2@1",
            "before_first_retained_promotion": {
                "active_frontier_tasks": 3,
                "nearest_probe_tasks": 1,
            },
            "after_first_retained_promotion": {
                "active_frontier_tasks": 2,
                "adaptive_probe_tasks": 2,
            },
        },
        "minimum_effective_policy_updates": 1,
        "final_evaluation_safety_factor": 1.5,
    },
    "authorization": {
        "maximum_age_hours": 168,
        "require_teardown_confirmation": True,
        "require_no_policy_mutation": True,
        "require_optimizer_state_restored": True,
        "require_test_split_isolation": True,
        "require_all_gate_results": True,
        "required_gate_results": list(REQUIRED_GATE_RESULTS),
    },
}


class GateError(ValueError):
    """Raised when a larger-model safety or eligibility check fails."""


@dataclass(frozen=True)
class RunPodGPU:
    """The RunPod inventory fields relevant to an allocation decision."""

    gpu_id: str
    display_name: str
    memory_gb: int
    available: bool
    community_cloud: bool
    secure_cloud: bool
    stock_status: str


@dataclass(frozen=True)
class CUDAHardware:
    """Actual CUDA device facts checked before paid artifact verification."""

    gpu_name: str
    total_memory_bytes: int
    bf16_supported: bool


MetadataFetcher = Callable[[str], Mapping[str, Any] | str | bytes]


def _expect_exact(observed: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise GateError(f"{path} must be an object")
        observed_keys = set(observed)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)
            extra = sorted(observed_keys - expected_keys)
            raise GateError(f"{path} keys are invalid (missing={missing}, extra={extra})")
        for key, value in expected.items():
            _expect_exact(observed[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise GateError(f"{path} must be the pinned {len(expected)}-item list")
        for index, value in enumerate(expected):
            _expect_exact(observed[index], value, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int | float)
            or not math.isfinite(float(observed))
            or float(observed) != expected
        ):
            raise GateError(f"{path} must equal {expected!r}")
        return
    if type(observed) is not type(expected) or observed != expected:
        raise GateError(f"{path} must equal {expected!r}")


def resolve_manifest_path() -> Path:
    """Resolve the profile in the repository or a flattened remote bundle."""

    configured = os.environ.get("EQUINOX_LARGER_MODEL_PROFILE_PATH")
    if configured:
        return Path(configured)
    if DEFAULT_MANIFEST_PATH.is_file():
        return DEFAULT_MANIFEST_PATH
    return Path(__file__).resolve().with_name("larger-model-eligibility.json")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the immutable larger-model profile, rejecting any drift."""

    path = path or resolve_manifest_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GateError(f"could not read larger-model manifest: {error}") from error
    except json.JSONDecodeError as error:
        raise GateError(f"larger-model manifest is not valid JSON: {error}") from error
    _expect_exact(payload, _EXPECTED_MANIFEST, "manifest")
    screen_limits = payload["screen_limits"]
    exact_screen_lifetime = (
        screen_limits["boot_timeout_seconds"]
        + screen_limits["target_runtime_seconds"]
        + screen_limits["retry_reserve_seconds"]
        + screen_limits["maximum_final_evaluation_reserve_seconds"]
        + screen_limits["artifact_retrieval_reserve_seconds"]
        + TEARDOWN_RESERVE_SECONDS
    )
    if screen_limits["maximum_lifetime_seconds"] != exact_screen_lifetime:
        raise GateError(
            "manifest.screen_limits maximum lifetime does not cover the exact "
            "readiness, science, retrieval, and teardown reserves"
        )
    if screen_limits["conservative_billing_seconds"] != (
        screen_limits["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS
    ):
        raise GateError(
            "manifest.screen_limits conservative billing reserve does not match "
            "the exact maximum lifetime plus cleanup reserve"
        )
    for name in ("screen_limits", "pilot_limits"):
        limits = payload[name]
        billed_seconds = (
            limits["conservative_billing_seconds"]
            if name == "screen_limits"
            else limits["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS
        )
        bound = lifetime_cost_bound(
            limits["maximum_hourly_cost_usd"],
            billed_seconds,
        )
        if bound > Decimal(str(limits["maximum_total_cost_usd"])):
            raise GateError(f"manifest.{name} lifetime cost exceeds its total-cost ceiling")
    return payload


def _positive_decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise GateError(f"{name} must be a positive finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise GateError(f"{name} must be a positive finite number") from error
    if not result.is_finite() or result <= 0:
        raise GateError(f"{name} must be a positive finite number")
    return result


def lifetime_cost_bound(hourly_cost_usd: Any, lifetime_seconds: Any) -> Decimal:
    """Return the worst-case compute cost for a fixed lifetime and hourly rate."""

    hourly = _positive_decimal(hourly_cost_usd, "hourly_cost_usd")
    seconds = _positive_decimal(lifetime_seconds, "lifetime_seconds")
    return hourly * seconds / Decimal(3_600)


def _json_value(value: Any, name: str) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GateError(f"{name} is not UTF-8") from error
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise GateError(f"{name} is not valid JSON") from error
    return value


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def verify_registry_image_index(
    manifest: Mapping[str, Any],
    value: Any,
) -> dict[str, str]:
    """Resolve one exact linux/amd64 image manifest from a registry index."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    payload = _json_value(value, "container registry image index")
    if not isinstance(payload, Mapping):
        raise GateError("container registry image index must be an object")
    if type(payload.get("schemaVersion")) is not int or payload["schemaVersion"] != 2:
        raise GateError("container registry image index schemaVersion must be 2")
    if payload.get("mediaType") not in {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }:
        raise GateError("container registry image index mediaType is invalid")
    index_digest = payload.get("digest")
    if not _is_sha256_digest(index_digest):
        raise GateError("container registry image index digest is invalid")
    descriptors = payload.get("manifests")
    if isinstance(descriptors, str | bytes) or not isinstance(descriptors, Sequence):
        raise GateError("container registry image index manifests must be an array")

    candidates: list[str] = []
    valid_manifest_media_types = {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise GateError(f"container registry image descriptor {index} must be an object")
        platform = descriptor.get("platform")
        if not isinstance(platform, Mapping):
            raise GateError(
                f"container registry image descriptor {index} platform must be an object"
            )
        operating_system = platform.get("os")
        architecture = platform.get("architecture")
        if not isinstance(operating_system, str) or not isinstance(architecture, str):
            raise GateError(f"container registry image descriptor {index} platform is invalid")
        if operating_system != "linux" or architecture != "amd64":
            continue
        variant = platform.get("variant")
        if variant not in (None, ""):
            raise GateError("container registry linux/amd64 image variant is unsupported")
        if descriptor.get("mediaType") not in valid_manifest_media_types:
            raise GateError("container registry linux/amd64 descriptor is not an image manifest")
        digest = descriptor.get("digest")
        if not _is_sha256_digest(digest):
            raise GateError("container registry linux/amd64 image digest is invalid")
        candidates.append(digest)

    if len(candidates) != 1:
        raise GateError("container registry must return exactly one linux/amd64 image manifest")
    expected_digest = manifest["runtime"]["image_digest"]
    if candidates[0] != expected_digest:
        raise GateError(
            "container registry linux/amd64 image digest does not match the immutable profile"
        )
    return {
        "image": manifest["runtime"]["image"],
        "image_digest": candidates[0],
        "index_digest": index_digest,
        "platform": "linux/amd64",
    }


def parse_runpod_inventory(value: Any) -> tuple[RunPodGPU, ...]:
    """Parse ``runpodctl gpu list`` JSON without accepting ambiguous entries."""

    payload = _json_value(value, "RunPod GPU inventory")
    if isinstance(payload, dict):
        if set(payload) != {"gpus"}:
            raise GateError("RunPod GPU inventory wrapper must contain only 'gpus'")
        payload = payload["gpus"]
    if isinstance(payload, str | bytes) or not isinstance(payload, Sequence):
        raise GateError("RunPod GPU inventory must be a JSON array")

    parsed: list[RunPodGPU] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise GateError(f"RunPod GPU inventory item {index} must be an object")
        required = {
            "gpuId": str,
            "displayName": str,
            "memoryInGb": int,
            "available": bool,
            "communityCloud": bool,
            "secureCloud": bool,
            "stockStatus": str,
        }
        for key, expected_type in required.items():
            field = item.get(key)
            if type(field) is not expected_type:
                raise GateError(f"RunPod GPU inventory item {index}.{key} has an invalid type")
        if not item["gpuId"] or not item["displayName"] or item["memoryInGb"] <= 0:
            raise GateError(f"RunPod GPU inventory item {index} has an invalid identity")
        parsed.append(
            RunPodGPU(
                gpu_id=item["gpuId"],
                display_name=item["displayName"],
                memory_gb=item["memoryInGb"],
                available=item["available"],
                community_cloud=item["communityCloud"],
                secure_cloud=item["secureCloud"],
                stock_status=item["stockStatus"],
            )
        )
    if not parsed:
        raise GateError("RunPod GPU inventory is empty")
    identities = [(item.gpu_id, item.display_name) for item in parsed]
    if len(set(identities)) != len(identities):
        raise GateError("RunPod GPU inventory contains duplicate identities")
    return tuple(parsed)


def matching_runpod_gpus(
    manifest: Mapping[str, Any],
    inventory: Any,
) -> tuple[RunPodGPU, ...]:
    """Return available inventory rows that satisfy the pinned hardware profile."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    hardware = manifest["hardware"]
    rows = (
        inventory
        if isinstance(inventory, tuple) and all(isinstance(item, RunPodGPU) for item in inventory)
        else parse_runpod_inventory(inventory)
    )
    secure_required = hardware["cloud_type"] == "SECURE"
    return tuple(
        item
        for item in rows
        if item.gpu_id == hardware["gpu_id"]
        and item.display_name == hardware["gpu_display_name"]
        and item.memory_gb >= hardware["minimum_gpu_memory_gb"]
        and item.available
        and (not secure_required or item.secure_cloud)
    )


def require_runpod_gpu(manifest: Mapping[str, Any], inventory: Any) -> RunPodGPU:
    """Return the sole eligible GPU inventory row or fail before allocation."""

    matches = matching_runpod_gpus(manifest, inventory)
    if not matches:
        raise GateError("the pinned RunPod H100 80GB HBM3 secure-cloud profile is unavailable")
    if len(matches) != 1:
        raise GateError("RunPod returned an ambiguous pinned GPU inventory")
    return matches[0]


def require_cuda_hardware(
    manifest: Mapping[str, Any],
    torch_module: Any,
) -> CUDAHardware:
    """Require the exact paid CUDA profile before hashing or loading model bytes."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    observed_torch_version = getattr(torch_module, "__version__", None)
    expected_torch_version = manifest["runtime"]["torch_version"]
    if observed_torch_version != expected_torch_version:
        raise GateError(
            "the paid larger-model worker torch build does not match the immutable profile "
            f"(observed={observed_torch_version!r}, expected={expected_torch_version!r})"
        )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise GateError("the paid larger-model worker has no available CUDA device")
    try:
        gpu_name = str(cuda.get_device_name(0))
        total_memory_bytes = int(cuda.get_device_properties(0).total_memory)
        bf16_supported = bool(cuda.is_bf16_supported())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise GateError("the paid larger-model worker CUDA profile could not be read") from error

    hardware = manifest["hardware"]
    failures: list[str] = []
    if gpu_name != hardware["gpu_id"]:
        failures.append(f"gpu_name={gpu_name!r}")
    if total_memory_bytes < hardware["minimum_cuda_memory_bytes"]:
        failures.append(f"total_memory_bytes={total_memory_bytes}")
    if not bf16_supported:
        failures.append("bf16_supported=false")
    if failures:
        raise GateError(
            "the actual paid CUDA device does not match the immutable profile ("
            + ", ".join(failures)
            + ")"
        )
    return CUDAHardware(
        gpu_name=gpu_name,
        total_memory_bytes=total_memory_bytes,
        bf16_supported=bf16_supported,
    )


def _default_metadata_fetcher(url: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "equinox-larger-model-gate/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read()
    except OSError as error:
        raise GateError(f"could not verify Hugging Face model metadata: {error}") from error
    payload = _json_value(body, "Hugging Face model metadata")
    if not isinstance(payload, dict):
        raise GateError("Hugging Face model metadata must be an object")
    return payload


def verify_huggingface_metadata(
    manifest: Mapping[str, Any],
    *,
    fetcher: MetadataFetcher | None = None,
) -> dict[str, Any]:
    """Verify the pinned revision, parameter count, and snapshot byte total."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    model = manifest["model"]
    encoded_id = urllib.parse.quote(model["id"], safe="/")
    encoded_revision = urllib.parse.quote(model["revision"], safe="")
    url = f"https://huggingface.co/api/models/{encoded_id}/revision/{encoded_revision}?blobs=true"
    raw = (fetcher or _default_metadata_fetcher)(url)
    payload = _json_value(raw, "Hugging Face model metadata")
    if not isinstance(payload, dict):
        raise GateError("Hugging Face model metadata must be an object")
    if payload.get("id") != model["id"]:
        raise GateError("Hugging Face returned a different model identity")
    if payload.get("sha") != model["revision"]:
        raise GateError("Hugging Face returned a different model revision")

    safetensors = payload.get("safetensors")
    parameters = safetensors.get("parameters") if isinstance(safetensors, dict) else None
    if (
        not isinstance(parameters, dict)
        or type(parameters.get("BF16")) is not int
        or parameters["BF16"] != model["parameter_count"]
        or type(safetensors.get("total")) is not int
        or safetensors["total"] != model["parameter_count"]
    ):
        raise GateError("Hugging Face parameter metadata does not match the pinned model")

    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise GateError("Hugging Face metadata omitted the snapshot file list")
    shard_sizes: list[int] = []
    shard_names: list[str] = []
    for index, sibling in enumerate(siblings):
        if not isinstance(sibling, dict):
            raise GateError(f"Hugging Face snapshot item {index} must be an object")
        name = sibling.get("rfilename")
        if isinstance(name, str) and name.startswith("model-") and name.endswith(".safetensors"):
            size = sibling.get("size")
            expected_file = model["snapshot_files"].get(name)
            lfs = sibling.get("lfs")
            if (
                not isinstance(expected_file, dict)
                or type(size) is not int
                or size != expected_file["size_bytes"]
                or not isinstance(lfs, dict)
                or lfs.get("sha256") != expected_file["sha256"]
            ):
                raise GateError(f"Hugging Face snapshot shard {name!r} has no valid size")
            shard_names.append(name)
            shard_sizes.append(size)
    if len(shard_names) != 4 or len(set(shard_names)) != 4:
        raise GateError("Hugging Face snapshot must contain four unique model shards")
    observed_bytes = sum(shard_sizes)
    if observed_bytes != model["safetensors_bytes"]:
        raise GateError("Hugging Face snapshot size does not match the pinned artifact")
    return {
        "model_id": model["id"],
        "model_revision": model["revision"],
        "parameter_count": model["parameter_count"],
        "safetensors_bytes": observed_bytes,
        "shard_count": len(shard_names),
        "metadata_url": url,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise GateError(f"could not hash pinned file {path.name}: {error}") from error
    return digest.hexdigest()


def _snapshot_entries(snapshot: Path) -> dict[str, Path]:
    try:
        entries = {entry.name: entry for entry in snapshot.iterdir()}
    except OSError as error:
        raise GateError(f"could not enumerate the pinned snapshot: {error}") from error
    return entries


def _open_snapshot_regular_file(path: Path) -> tuple[Any, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GateError(f"could not open pinned snapshot file {path.name}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GateError(f"the pinned snapshot entry {path.name} is not a regular file")
        return os.fdopen(descriptor, "rb"), metadata
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot_content_path(snapshot: Path, path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GateError(f"the pinned snapshot is missing {path.name}") from error
    if stat.S_ISREG(metadata.st_mode):
        return path
    if not stat.S_ISLNK(metadata.st_mode):
        raise GateError(f"the pinned snapshot entry {path.name} has an unsafe file type")

    try:
        target_text = os.readlink(path)
    except OSError as error:
        raise GateError(f"could not read pinned snapshot symlink {path.name}") from error
    if Path(target_text).is_absolute():
        raise GateError(f"the pinned snapshot symlink {path.name} must be relative")

    model_cache = snapshot.parent.parent
    blobs = model_cache / "blobs"
    target = path.parent / target_text
    try:
        resolved_blobs = blobs.resolve(strict=True)
        resolved_target = target.resolve(strict=True)
        target_metadata = target.lstat()
    except OSError as error:
        raise GateError(f"the pinned snapshot symlink {path.name} is broken") from error
    if (
        stat.S_ISLNK(target_metadata.st_mode)
        or not stat.S_ISREG(target_metadata.st_mode)
        or resolved_target.parent != resolved_blobs
        or len(resolved_target.name) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in resolved_target.name)
    ):
        raise GateError(
            f"the pinned snapshot symlink {path.name} does not target a direct cache blob"
        )
    return target


def _hash_snapshot_entry(
    snapshot: Path,
    path: Path,
    *,
    destination: Path | None = None,
    capture_bytes: bool = False,
) -> tuple[int, str, bytes | None]:
    content_path = _snapshot_content_path(snapshot, path)
    handle, before = _open_snapshot_regular_file(content_path)
    digest = hashlib.sha256()
    captured = bytearray() if capture_bytes else None
    output: Any | None = None
    try:
        if destination is not None:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(destination, flags, 0o400)
            output = os.fdopen(descriptor, "wb")
        with handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
                if output is not None:
                    output.write(chunk)
            after = os.fstat(handle.fileno())
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            copied = os.fstat(output.fileno())
            if not stat.S_ISREG(copied.st_mode) or copied.st_size != before.st_size:
                raise GateError(f"the verified snapshot copy {path.name} is incomplete")
    except OSError as error:
        raise GateError(f"could not hash pinned snapshot file {path.name}: {error}") from error
    finally:
        if output is not None:
            output.close()
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise GateError(f"the pinned snapshot file {path.name} changed while being hashed")
    return before.st_size, digest.hexdigest(), bytes(captured) if captured is not None else None


def _verify_snapshot(
    manifest: Mapping[str, Any],
    snapshot: Path,
    *,
    destination: Path | None,
) -> dict[str, Any]:
    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    try:
        snapshot_metadata = snapshot.lstat()
    except OSError as error:
        raise GateError("the pinned local model snapshot directory is unavailable") from error
    if not stat.S_ISDIR(snapshot_metadata.st_mode):
        raise GateError("the pinned local model snapshot directory is unavailable")
    expected_files = manifest["model"]["snapshot_files"]
    entries = _snapshot_entries(snapshot)
    if set(entries) != set(expected_files):
        raise GateError("the pinned snapshot entry set does not match the exact file contract")
    if destination is not None:
        try:
            destination_metadata = destination.lstat()
        except OSError as error:
            raise GateError("the verified snapshot destination is unavailable") from error
        if not stat.S_ISDIR(destination_metadata.st_mode) or _snapshot_entries(destination):
            raise GateError("the verified snapshot destination must be an empty directory")
    observed_files: dict[str, dict[str, Any]] = {}
    index_bytes: bytes | None = None
    for name, expected in expected_files.items():
        path = entries[name]
        size, observed_sha256, captured = _hash_snapshot_entry(
            snapshot,
            path,
            destination=destination / name if destination is not None else None,
            capture_bytes=name == "model.safetensors.index.json",
        )
        if size != expected["size_bytes"]:
            raise GateError(f"the pinned snapshot file {name} has an invalid size")
        if observed_sha256 != expected["sha256"]:
            raise GateError(f"the pinned snapshot file {name} failed SHA-256 verification")
        if captured is not None:
            index_bytes = captured
        observed_files[name] = {
            "size_bytes": size,
            "sha256": observed_sha256,
        }
    if set(_snapshot_entries(snapshot)) != set(expected_files):
        raise GateError("the pinned snapshot entry set changed while being verified")
    if destination is not None:
        copied_entries = _snapshot_entries(destination)
        if set(copied_entries) != set(expected_files) or any(
            not stat.S_ISREG(path.lstat().st_mode) for path in copied_entries.values()
        ):
            raise GateError("the verified snapshot copy has an invalid entry set")

    try:
        if index_bytes is None:
            raise ValueError("missing captured model index")
        index = json.loads(index_bytes)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise GateError("the pinned model index is invalid") from error
    expected_shards = {name for name in expected_files if name.endswith(".safetensors")}
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or set(weight_map.values()) != expected_shards
    ):
        raise GateError("the pinned model index does not cover the exact shard set")

    snapshot_material = {
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "files": observed_files,
    }
    return {
        **snapshot_material,
        "snapshot_path": str(destination if destination is not None else snapshot),
        "snapshot_digest": "sha256:"
        + hashlib.sha256(canonical_json(snapshot_material)).hexdigest(),
    }


def verify_local_snapshot(
    manifest: Mapping[str, Any],
    snapshot: Path,
) -> dict[str, Any]:
    """Hash every required local artifact before any model initialization."""

    return _verify_snapshot(manifest, snapshot, destination=None)


def materialize_verified_snapshot(
    manifest: Mapping[str, Any],
    snapshot: Path,
    *,
    destination_parent: Path | None = None,
) -> dict[str, Any]:
    """Verify once while copying into a private immutable container-local snapshot."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    parent = (destination_parent or Path(tempfile.gettempdir())).resolve(strict=True)
    try:
        parent_metadata = parent.lstat()
    except OSError as error:
        raise GateError("the verified snapshot parent is unavailable") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise GateError("the verified snapshot parent must be a directory")
    required_bytes = sum(
        item["size_bytes"] for item in manifest["model"]["snapshot_files"].values()
    )
    if shutil.disk_usage(parent).free < required_bytes + 2_000_000_000:
        raise GateError("container-local storage cannot hold the verified model snapshot")

    destination = Path(tempfile.mkdtemp(prefix="equinox-verified-snapshot-", dir=str(parent)))
    try:
        evidence = _verify_snapshot(manifest, snapshot, destination=destination)
        directory_descriptor = os.open(
            destination,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(destination, 0o500)
        return evidence
    except BaseException:
        try:
            os.chmod(destination, 0o700)
            shutil.rmtree(destination)
        except OSError:
            pass
        raise


def canonical_json(value: Any) -> bytes:
    """Serialize a JSON value for stable hashing and receipts."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GateError(f"value cannot be represented as canonical JSON: {error}") from error


def checkpoint_authentication_mechanism_digest() -> str:
    """Return the stable cross-launch checkpoint mechanism identity."""

    return (
        "sha256:" + hashlib.sha256(canonical_json(CHECKPOINT_AUTHENTICATION_MECHANISM)).hexdigest()
    )


def expected_source_contract_digest(manifest: Mapping[str, Any]) -> str:
    """Return the stable identity of the manifest-pinned scientific sources."""

    material = {
        "profile_id": manifest["profile_id"],
        "source_contract": manifest["source_contract"],
    }
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def verify_source_contract(
    manifest: Mapping[str, Any],
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Hash the exact local or flattened-bundle sources used by paid workloads."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    root = source_root or Path(__file__).resolve().parent
    expected_files = manifest["source_contract"]["files"]
    observed_files: dict[str, str] = {}
    for name, expected_sha256 in expected_files.items():
        path = root / name
        if not path.is_file():
            raise GateError(f"the pinned scientific source {name} is unavailable")
        observed_sha256 = _sha256_path(path)
        if observed_sha256 != expected_sha256:
            raise GateError(f"the pinned scientific source {name} failed SHA-256 verification")
        observed_files[name] = observed_sha256
    return {
        "profile_id": manifest["profile_id"],
        "algorithm": manifest["source_contract"]["algorithm"],
        "files": observed_files,
        "source_contract_digest": expected_source_contract_digest(manifest),
    }


def result_digest(result: Mapping[str, Any]) -> str:
    """Digest a screen result, excluding its optional self-reported digest."""

    if not isinstance(result, Mapping):
        raise GateError("screen result must be an object")
    content = {key: value for key, value in result.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


def expected_snapshot_digest(manifest: Mapping[str, Any]) -> str:
    material = {
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "files": manifest["model"]["snapshot_files"],
    }
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def _required_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise GateError(f"{name} must be a finite number")
    return float(value)


def _required_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise GateError(f"{name} must be a non-negative integer")
    return value


def _utc_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GateError(f"{name} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        timestamp = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise GateError(f"{name} must be an RFC 3339 UTC timestamp") from error
    if timestamp.tzinfo != UTC:
        raise GateError(f"{name} must use UTC")
    return timestamp


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    content = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != len("sha256:") + 64
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise GateError(f"{name} must be a SHA-256 digest")
    return value


def materialization_evidence_digest(evidence: Mapping[str, Any]) -> str:
    """Digest one private-materialization receipt without its self digest."""

    material = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def _verify_private_tree_identity(
    evidence: Mapping[str, Any],
    *,
    tree_kind: str,
    source_digest_key: str = "source_tree_digest",
) -> tuple[str, str]:
    source_tree_digest = _required_sha256(
        evidence.get(source_digest_key),
        f"{tree_kind} source tree digest",
    )
    private_tree_digest = _required_sha256(
        evidence.get("private_tree_digest"),
        f"{tree_kind} private tree digest",
    )
    if private_tree_digest != source_tree_digest:
        raise GateError(f"{tree_kind} private tree does not match its source")
    expected_private_root = (
        f"/tmp/equinox-quarantine/{tree_kind}/{private_tree_digest.removeprefix('sha256:')}"
    )
    if evidence.get("private_root") != expected_private_root:
        raise GateError(f"{tree_kind} private root is not content-addressed")
    evidence_digest = _required_sha256(
        evidence.get("evidence_digest"),
        f"{tree_kind} evidence digest",
    )
    if evidence_digest != materialization_evidence_digest(evidence):
        raise GateError(f"{tree_kind} evidence digest is invalid")
    return evidence_digest, private_tree_digest


def verify_dependency_quarantine_evidence(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    workload_bundle_path: str,
) -> dict[str, Any]:
    """Verify one hash-locked, binary-only private dependency installation."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    required_keys = {
        "schema_version",
        "revision",
        "profile_id",
        "lock_path",
        "lock_digest",
        "python_version",
        "platform_tag",
        "installer_revision",
        "index_url",
        "private_root",
        "install_tree_digest",
        "private_tree_digest",
        "record_closure_digest",
        "distributions",
        "installed_file_count",
        "installed_bytes",
        "ready",
        "evidence_digest",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_keys:
        raise GateError("dependency quarantine evidence has an invalid field set")
    expected_identity = {
        "schema_version": 2,
        "revision": manifest["materialization"]["dependency_lock"]["revision"],
        "profile_id": manifest["profile_id"],
        "lock_path": (
            f"{workload_bundle_path}::{manifest['materialization']['dependency_lock']['path']}"
        ),
        "lock_digest": manifest["materialization"]["dependency_lock"]["digest"],
        "python_version": manifest["materialization"]["dependency_lock"]["python_version"],
        "platform_tag": manifest["materialization"]["dependency_lock"]["platform_tag"],
        "installer_revision": manifest["materialization"]["dependency_lock"]["installer_revision"],
        "index_url": manifest["materialization"]["dependency_lock"]["index_url"],
        "ready": True,
    }
    for key, expected in expected_identity.items():
        if evidence.get(key) != expected:
            raise GateError(f"dependency quarantine evidence {key} is invalid")
    distributions = evidence.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise GateError("dependency quarantine distributions are invalid")
    if distributions != sorted(distributions, key=lambda item: str(item.get("name", ""))):
        raise GateError("dependency quarantine distributions are not sorted")
    observed_versions: dict[str, str] = {}
    file_count = 0
    for distribution in distributions:
        if not isinstance(distribution, Mapping) or set(distribution) != {
            "name",
            "version",
            "record_path",
            "record_digest",
            "file_count",
            "files_digest",
        }:
            raise GateError("dependency quarantine distribution has an invalid field set")
        name = distribution.get("name")
        version = distribution.get("version")
        record_path = distribution.get("record_path")
        if (
            not isinstance(name, str)
            or not name
            or name in observed_versions
            or not isinstance(version, str)
            or not version
            or not isinstance(record_path, str)
            or not record_path.endswith(".dist-info/RECORD")
            or record_path.startswith("/")
            or ".." in PurePosixPath(record_path).parts
        ):
            raise GateError("dependency quarantine distribution identity is invalid")
        observed_versions[name] = version
        _required_sha256(distribution.get("record_digest"), "dependency RECORD digest")
        _required_sha256(distribution.get("files_digest"), "dependency files digest")
        count = _required_integer(
            distribution.get("file_count"),
            "dependency distribution file count",
        )
        if count == 0:
            raise GateError("dependency distribution file count must be positive")
        file_count += count
    lock_path = (
        Path(__file__).resolve().with_name(manifest["materialization"]["dependency_lock"]["path"])
    )
    try:
        lock_payload = lock_path.read_bytes()
    except OSError as error:
        raise GateError("dependency lock is unavailable beside the gate source") from error
    if (
        len(lock_payload) != manifest["materialization"]["dependency_lock"]["size_bytes"]
        or "sha256:" + hashlib.sha256(lock_payload).hexdigest()
        != manifest["materialization"]["dependency_lock"]["digest"]
    ):
        raise GateError("dependency lock bytes do not match the immutable profile")
    locked_versions: dict[str, str] = {}
    for raw_line in lock_payload.decode("utf-8").splitlines():
        match = re.fullmatch(r"([a-z0-9][a-z0-9._-]*)==([^ \\]+) \\", raw_line)
        if match is None:
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in locked_versions:
            raise GateError("dependency lock contains a duplicate distribution")
        locked_versions[name] = match.group(2)
    if (
        len(locked_versions) != 30
        or "torch" in locked_versions
        or observed_versions != locked_versions
    ):
        raise GateError("dependency installation does not exactly match the hash lock")
    installed_file_count = _required_integer(
        evidence.get("installed_file_count"),
        "dependency installed file count",
    )
    installed_bytes = _required_integer(
        evidence.get("installed_bytes"),
        "dependency installed bytes",
    )
    if installed_file_count == 0 or installed_bytes == 0 or file_count != installed_file_count:
        raise GateError("dependency quarantine installed totals are invalid")
    record_closure_digest = _required_sha256(
        evidence.get("record_closure_digest"),
        "dependency record closure digest",
    )
    expected_closure_digest = "sha256:" + hashlib.sha256(canonical_json(distributions)).hexdigest()
    if record_closure_digest != expected_closure_digest:
        raise GateError("dependency RECORD closure digest is invalid")
    install_tree_digest = _required_sha256(
        evidence.get("install_tree_digest"),
        "dependency install tree digest",
    )
    if evidence.get("private_tree_digest") != install_tree_digest:
        raise GateError("dependency private tree does not match the installed tree")
    evidence_digest, private_tree_digest = _verify_private_tree_identity(
        evidence,
        tree_kind="dependencies",
        source_digest_key="install_tree_digest",
    )
    return {
        "evidence_digest": evidence_digest,
        "private_tree_digest": private_tree_digest,
        "private_root": evidence["private_root"],
    }


def verify_code_materialization_evidence(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    workload_bundle_digest: str,
    workload_bundle_size_bytes: int,
    workload_bundle_path: str,
) -> dict[str, Any]:
    """Verify the exact allowlisted private code materialization receipt."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    required_keys = {
        "schema_version",
        "revision",
        "profile_id",
        "bundle_digest",
        "bundle_size_bytes",
        "source_contract_digest",
        "source_root",
        "private_root",
        "source_tree_digest",
        "private_tree_digest",
        "files",
        "installed_file_count",
        "installed_bytes",
        "ready",
        "evidence_digest",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_keys:
        raise GateError("code materialization evidence has an invalid field set")
    expected_identity = {
        "schema_version": 1,
        "revision": manifest["materialization"]["code"]["revision"],
        "profile_id": manifest["profile_id"],
        "bundle_digest": _required_sha256(
            workload_bundle_digest,
            "workload bundle digest",
        ),
        "bundle_size_bytes": workload_bundle_size_bytes,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "source_root": workload_bundle_path,
        "ready": True,
    }
    for key, expected in expected_identity.items():
        if evidence.get(key) != expected:
            raise GateError(f"code materialization evidence {key} is invalid")
    if (
        type(workload_bundle_size_bytes) is not int
        or not 0 < workload_bundle_size_bytes <= MAXIMUM_WORKLOAD_BUNDLE_BYTES
    ):
        raise GateError("code materialization bundle size is invalid")
    files = evidence.get("files")
    if not isinstance(files, list) or not files:
        raise GateError("code materialization file inventory is invalid")
    observed_paths: set[str] = set()
    installed_bytes = 0
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise GateError("code materialization file entry has an invalid field set")
        path = entry.get("path")
        size_bytes = _required_integer(
            entry.get("size_bytes"),
            "code materialization file size",
        )
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or ".." in PurePosixPath(path).parts
            or path in observed_paths
        ):
            raise GateError("code materialization file path is invalid")
        observed_paths.add(path)
        _required_sha256(entry.get("sha256"), "code materialization file digest")
        installed_bytes += size_bytes
    if files != sorted(files, key=lambda entry: str(entry["path"])):
        raise GateError("code materialization files are not sorted")
    if not set(manifest["source_contract"]["files"]).issubset(observed_paths):
        raise GateError("code materialization omits a pinned scientific source")
    entries_by_path = {str(entry["path"]): entry for entry in files}
    for path, expected_digest in manifest["source_contract"]["files"].items():
        if entries_by_path[path]["sha256"] != f"sha256:{expected_digest}":
            raise GateError(f"code materialization source {path} digest is invalid")
    dependency_lock_entry = entries_by_path[manifest["materialization"]["dependency_lock"]["path"]]
    if (
        dependency_lock_entry["size_bytes"]
        != manifest["materialization"]["dependency_lock"]["size_bytes"]
        or dependency_lock_entry["sha256"]
        != manifest["materialization"]["dependency_lock"]["digest"]
    ):
        raise GateError("code materialization dependency lock identity is invalid")
    if (
        evidence.get("installed_file_count") != len(files)
        or evidence.get("installed_bytes") != installed_bytes
    ):
        raise GateError("code materialization installed totals are invalid")
    expected_tree_digest = "sha256:" + hashlib.sha256(canonical_json(files)).hexdigest()
    if evidence.get("source_tree_digest") != expected_tree_digest:
        raise GateError("code materialization source tree digest is invalid")
    evidence_digest, private_tree_digest = _verify_private_tree_identity(
        evidence,
        tree_kind="code",
    )
    return {
        "evidence_digest": evidence_digest,
        "private_tree_digest": private_tree_digest,
        "private_root": evidence["private_root"],
    }


def expected_cuda_version(manifest: Mapping[str, Any]) -> str:
    """Return the exact dotted CUDA version encoded by the pinned Torch build."""

    torch_version = manifest["runtime"]["torch_version"]
    _, separator, cuda_suffix = torch_version.partition("+cu")
    if (
        not separator
        or len(cuda_suffix) != 3
        or not cuda_suffix.isascii()
        or not cuda_suffix.isdigit()
    ):
        raise GateError("the immutable Torch build does not encode an exact CUDA version")
    return f"{int(cuda_suffix[:-1])}.{int(cuda_suffix[-1])}"


def verify_dependency_import_smoke(
    manifest: Mapping[str, Any],
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    dependency_root: Path | None = None,
) -> tuple[str, ...]:
    """Import every pinned volume dependency before declaring the cache ready."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    imported: list[str] = []
    for package, expected_version in manifest["runtime"]["dependencies"].items():
        try:
            module = importer(package)
        except Exception as error:
            raise GateError(f"prewarmed dependency {package} could not be imported") from error
        observed_version = getattr(module, "__version__", None)
        if observed_version != expected_version:
            raise GateError(
                f"prewarmed dependency {package} import reported {observed_version!r}, "
                f"expected {expected_version!r}"
            )
        if dependency_root is not None:
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise GateError(f"prewarmed dependency {package} did not expose its import origin")
            try:
                Path(module_file).resolve(strict=False).relative_to(
                    dependency_root.resolve(strict=False)
                )
            except (OSError, ValueError) as error:
                raise GateError(
                    f"prewarmed dependency {package} was not imported from "
                    "the private hash-locked tree"
                ) from error
        imported.append(package)
    return tuple(imported)


def build_attested_volume_readiness_receipt(
    manifest: Mapping[str, Any],
    *,
    snapshot_digest: str,
    volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
    dependency_versions: Mapping[str, str],
    torch_version: str,
    torch_cuda_version: str,
    hardware: CUDAHardware,
    dependency_lock_digest: str,
    dependency_quarantine_evidence: Mapping[str, Any],
    dependency_quarantine_evidence_digest: str,
    dependency_private_tree_digest: str,
    code_materialization_evidence: Mapping[str, Any],
    code_materialization_evidence_digest: str,
    code_private_tree_digest: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a receipt from exact evidence already verified on the paid H100."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise GateError("volume readiness dependency lock digest is invalid")
    if (
        not isinstance(volume_id, str)
        or not volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
    ):
        raise GateError("volume readiness identity or size is invalid")
    if (
        not isinstance(dependency_versions, Mapping)
        or dict(dependency_versions) != manifest["runtime"]["dependencies"]
    ):
        raise GateError("volume readiness dependencies do not match the immutable profile")
    if snapshot_digest != expected_snapshot_digest(manifest):
        raise GateError("volume readiness snapshot does not match the immutable profile")
    if torch_version != manifest["runtime"]["torch_version"]:
        raise GateError("volume readiness Torch build does not match the immutable profile")
    if torch_cuda_version != expected_cuda_version(manifest):
        raise GateError("volume readiness CUDA build does not match the immutable profile")
    if (
        not isinstance(hardware, CUDAHardware)
        or hardware.gpu_name != manifest["hardware"]["gpu_id"]
        or hardware.total_memory_bytes < manifest["hardware"]["minimum_cuda_memory_bytes"]
        or hardware.bf16_supported is not True
    ):
        raise GateError("volume readiness hardware does not match the immutable profile")
    prepared_at = now or datetime.now(UTC)
    if prepared_at.tzinfo is None or prepared_at.utcoffset() is None:
        raise GateError("volume readiness time must be timezone-aware")
    prepared_at = prepared_at.astimezone(UTC)
    code_bundle_digest = code_materialization_evidence.get("bundle_digest")
    code_bundle_size_bytes = code_materialization_evidence.get("bundle_size_bytes")
    code_source_root = code_materialization_evidence.get("source_root")
    verified_dependencies = verify_dependency_quarantine_evidence(
        manifest,
        dependency_quarantine_evidence,
        workload_bundle_path=code_source_root,
    )
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise GateError("volume readiness dependency lock digest is invalid")
    verified_code = verify_code_materialization_evidence(
        manifest,
        code_materialization_evidence,
        workload_bundle_digest=code_bundle_digest,
        workload_bundle_size_bytes=code_bundle_size_bytes,
        workload_bundle_path=code_source_root,
    )
    if (
        verified_dependencies["evidence_digest"]
        != _required_sha256(
            dependency_quarantine_evidence_digest,
            "dependency quarantine evidence digest",
        )
        or verified_dependencies["private_tree_digest"]
        != _required_sha256(
            dependency_private_tree_digest,
            "dependency private tree digest",
        )
        or verified_code["evidence_digest"]
        != _required_sha256(
            code_materialization_evidence_digest,
            "code materialization evidence digest",
        )
        or verified_code["private_tree_digest"]
        != _required_sha256(
            code_private_tree_digest,
            "code private tree digest",
        )
    ):
        raise GateError("volume readiness private materialization digests do not match")
    receipt = {
        "schema_version": 3,
        "attestation_revision": VOLUME_READINESS_ATTESTATION_REVISION,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "network_volume_id": volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
        "snapshot_digest": snapshot_digest,
        "dependencies": dict(dependency_versions),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "gpu_name": hardware.gpu_name,
        "gpu_total_memory_bytes": hardware.total_memory_bytes,
        "bf16_supported": hardware.bf16_supported,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_evidence": dict(dependency_quarantine_evidence),
        "dependency_quarantine_evidence_digest": verified_dependencies["evidence_digest"],
        "dependency_private_tree_digest": verified_dependencies["private_tree_digest"],
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence": dict(code_materialization_evidence),
        "code_materialization_evidence_digest": verified_code["evidence_digest"],
        "code_private_tree_digest": verified_code["private_tree_digest"],
        "prepared_at": prepared_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def build_volume_readiness_receipt(
    manifest: Mapping[str, Any],
    snapshot: Path,
    *,
    volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
    dependency_versions: Mapping[str, str],
    dependency_lock_digest: str,
    dependency_quarantine_evidence: Mapping[str, Any],
    dependency_quarantine_evidence_digest: str,
    dependency_private_tree_digest: str,
    code_materialization_evidence: Mapping[str, Any],
    code_materialization_evidence_digest: str,
    code_private_tree_digest: str,
    torch_module: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the mounted snapshot and runtime, then build an H100-attested receipt."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise GateError("the pinned Torch runtime is unavailable") from error
    hardware = require_cuda_hardware(manifest, torch_module)
    verify_dependency_import_smoke(manifest)
    snapshot_evidence = verify_local_snapshot(manifest, snapshot)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not isinstance(cuda_version, str):
        raise GateError("the pinned Torch runtime did not report its CUDA build")
    return build_attested_volume_readiness_receipt(
        manifest,
        snapshot_digest=snapshot_evidence["snapshot_digest"],
        volume_id=volume_id,
        data_center_id=data_center_id,
        volume_size_gb=volume_size_gb,
        dependency_versions=dependency_versions,
        torch_version=str(torch_module.__version__),
        torch_cuda_version=cuda_version,
        hardware=hardware,
        dependency_lock_digest=dependency_lock_digest,
        dependency_quarantine_evidence=dependency_quarantine_evidence,
        dependency_quarantine_evidence_digest=dependency_quarantine_evidence_digest,
        dependency_private_tree_digest=dependency_private_tree_digest,
        code_materialization_evidence=code_materialization_evidence,
        code_materialization_evidence_digest=code_materialization_evidence_digest,
        code_private_tree_digest=code_private_tree_digest,
        now=now,
    )


def verify_volume_readiness_receipt(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    provider_volume: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify the CPU-prewarm receipt against current provider volume identity."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    required_keys = {
        "schema_version",
        "attestation_revision",
        "profile_id",
        "model_id",
        "model_revision",
        "manifest_digest",
        "network_volume_id",
        "network_volume_data_center_id",
        "network_volume_size_gb",
        "snapshot_digest",
        "dependencies",
        "torch_version",
        "torch_cuda_version",
        "gpu_name",
        "gpu_total_memory_bytes",
        "bf16_supported",
        "dependency_quarantine_revision",
        "dependency_lock_digest",
        "dependency_quarantine_evidence",
        "dependency_quarantine_evidence_digest",
        "dependency_private_tree_digest",
        "code_materialization_revision",
        "code_materialization_evidence",
        "code_materialization_evidence_digest",
        "code_private_tree_digest",
        "prepared_at",
        "ready",
        "receipt_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required_keys:
        raise GateError("volume readiness receipt has an invalid field set")
    expected_identity = {
        "schema_version": 3,
        "attestation_revision": VOLUME_READINESS_ATTESTATION_REVISION,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "snapshot_digest": expected_snapshot_digest(manifest),
        "dependencies": manifest["runtime"]["dependencies"],
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": expected_cuda_version(manifest),
        "gpu_name": manifest["hardware"]["gpu_id"],
        "bf16_supported": True,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_lock_digest": manifest["materialization"]["dependency_lock"]["digest"],
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "ready": True,
    }
    for key, expected in expected_identity.items():
        try:
            _expect_exact(receipt.get(key), expected, f"volume readiness receipt {key}")
        except GateError as error:
            raise GateError(f"volume readiness receipt {key} does not match the profile") from error
    volume_id = receipt.get("network_volume_id")
    data_center_id = receipt.get("network_volume_data_center_id")
    volume_size_gb = receipt.get("network_volume_size_gb")
    gpu_total_memory_bytes = receipt.get("gpu_total_memory_bytes")
    for key in (
        "dependency_quarantine_evidence_digest",
        "dependency_private_tree_digest",
        "code_materialization_evidence_digest",
        "code_private_tree_digest",
    ):
        _required_sha256(receipt.get(key), f"volume readiness receipt {key}")
    dependency_evidence = receipt.get("dependency_quarantine_evidence")
    code_evidence = receipt.get("code_materialization_evidence")
    if not isinstance(dependency_evidence, Mapping) or not isinstance(
        code_evidence,
        Mapping,
    ):
        raise GateError("volume readiness private materialization evidence is invalid")
    verified_dependencies = verify_dependency_quarantine_evidence(
        manifest,
        dependency_evidence,
        workload_bundle_path=code_evidence.get("source_root"),
    )
    verified_code = verify_code_materialization_evidence(
        manifest,
        code_evidence,
        workload_bundle_digest=code_evidence.get("bundle_digest"),
        workload_bundle_size_bytes=code_evidence.get("bundle_size_bytes"),
        workload_bundle_path=code_evidence.get("source_root"),
    )
    for key, expected in {
        "dependency_quarantine_evidence_digest": verified_dependencies["evidence_digest"],
        "dependency_private_tree_digest": verified_dependencies["private_tree_digest"],
        "code_materialization_evidence_digest": verified_code["evidence_digest"],
        "code_private_tree_digest": verified_code["private_tree_digest"],
    }.items():
        if receipt.get(key) != expected:
            raise GateError(f"volume readiness receipt {key} does not match its evidence")
    if (
        not isinstance(volume_id, str)
        or not volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
        or type(gpu_total_memory_bytes) is not int
        or gpu_total_memory_bytes < manifest["hardware"]["minimum_cuda_memory_bytes"]
    ):
        raise GateError("volume readiness receipt volume identity or size or hardware is invalid")
    receipt_digest = receipt.get("receipt_digest")
    if not isinstance(receipt_digest, str) or receipt_digest != _receipt_digest(receipt):
        raise GateError("volume readiness receipt digest is invalid")
    prepared_at = _utc_timestamp(receipt.get("prepared_at"), "volume receipt prepared_at")
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise GateError("volume receipt verification time must be timezone-aware")
    current_time = current_time.astimezone(UTC)
    age_seconds = (current_time - prepared_at).total_seconds()
    if age_seconds < 0 or age_seconds > 7 * 24 * 3_600:
        raise GateError("volume readiness receipt is not fresh")

    payload = _json_value(provider_volume, "RunPod network volume")
    if not isinstance(payload, Mapping):
        raise GateError("RunPod network volume must be an object")
    volume = payload.get("networkVolume", payload)
    if not isinstance(volume, Mapping):
        raise GateError("RunPod network volume payload is invalid")
    provider_id = volume.get("id", volume.get("networkVolumeId"))
    provider_size = volume.get("size")
    provider_data_center = volume.get("dataCenterId")
    if (
        not isinstance(provider_id, str)
        or not isinstance(provider_data_center, str)
        or type(provider_size) is not int
        or provider_id != volume_id
        or provider_data_center != data_center_id
        or provider_size < volume_size_gb
    ):
        raise GateError("RunPod network volume no longer matches its readiness receipt")
    return {
        "profile_id": manifest["profile_id"],
        "network_volume_id": provider_id,
        "network_volume_data_center_id": provider_data_center,
        "network_volume_size_gb": provider_size,
        "snapshot_digest": receipt["snapshot_digest"],
        "torch_version": receipt["torch_version"],
        "torch_cuda_version": receipt["torch_cuda_version"],
        "gpu_name": receipt["gpu_name"],
        "dependency_lock_digest": receipt["dependency_lock_digest"],
        "dependency_quarantine_evidence_digest": receipt["dependency_quarantine_evidence_digest"],
        "dependency_private_tree_digest": receipt["dependency_private_tree_digest"],
        "code_materialization_evidence_digest": receipt["code_materialization_evidence_digest"],
        "code_private_tree_digest": receipt["code_private_tree_digest"],
        "gpu_total_memory_bytes": receipt["gpu_total_memory_bytes"],
        "bf16_supported": receipt["bf16_supported"],
        "receipt_digest": receipt["receipt_digest"],
    }


def verify_predecessor_readiness_bridge(
    manifest: Mapping[str, Any],
    predecessor_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    provider_volume: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Allow one screen allocation from only the exact @6 volume seed receipt."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if not isinstance(predecessor_manifest, Mapping):
        raise GateError("predecessor manifest must be an object")
    observed_predecessor_digest = (
        "sha256:" + hashlib.sha256(canonical_json(predecessor_manifest)).hexdigest()
    )
    if observed_predecessor_digest != PREDECESSOR_MANIFEST_DIGEST:
        raise GateError("predecessor manifest is not the pinned committed @6 profile")
    if predecessor_manifest.get("profile_id") != PREDECESSOR_PROFILE_ID:
        raise GateError("predecessor manifest profile identity is invalid")
    for section in ("model", "runtime"):
        predecessor_section = predecessor_manifest.get(section)
        current_section = manifest.get(section)
        if section == "model":
            expected = {
                key: (
                    PREDECESSOR_SNAPSHOT_FILES if key == "snapshot_files" else current_section[key]
                )
                for key in (
                    "id",
                    "revision",
                    "parameter_count",
                    "safetensors_bytes",
                    "snapshot_files",
                    "dtype",
                )
            }
        else:
            expected = {
                key: current_section[key]
                for key in (
                    "template_id",
                    "image",
                    "image_digest",
                    "torch_version",
                    "dependencies",
                )
            }
        if predecessor_section != expected:
            raise GateError(f"predecessor manifest {section} no longer matches the current profile")

    required_keys = {
        "schema_version",
        "profile_id",
        "model_id",
        "model_revision",
        "manifest_digest",
        "network_volume_id",
        "network_volume_data_center_id",
        "network_volume_size_gb",
        "snapshot_digest",
        "dependencies",
        "prepared_at",
        "ready",
        "receipt_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required_keys:
        raise GateError("predecessor volume readiness receipt has an invalid field set")
    expected_receipt = {
        "schema_version": 1,
        "profile_id": PREDECESSOR_PROFILE_ID,
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": PREDECESSOR_MANIFEST_DIGEST,
        "snapshot_digest": PREDECESSOR_SNAPSHOT_DIGEST,
        "dependencies": manifest["runtime"]["dependencies"],
        "ready": True,
    }
    for key, expected in expected_receipt.items():
        try:
            _expect_exact(receipt.get(key), expected, f"predecessor receipt {key}")
        except GateError as error:
            raise GateError(f"predecessor receipt {key} is invalid") from error
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise GateError("predecessor volume readiness receipt digest is invalid")
    prepared_at = _utc_timestamp(
        receipt.get("prepared_at"),
        "predecessor receipt prepared_at",
    )
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise GateError("predecessor bridge verification time must be timezone-aware")
    current_time = current_time.astimezone(UTC)
    age_seconds = (current_time - prepared_at).total_seconds()
    if age_seconds < 0 or age_seconds > 7 * 24 * 3_600:
        raise GateError("predecessor volume readiness receipt is not fresh")

    payload = _json_value(provider_volume, "RunPod network volume")
    if not isinstance(payload, Mapping):
        raise GateError("RunPod network volume must be an object")
    volume = payload.get("networkVolume", payload)
    if not isinstance(volume, Mapping):
        raise GateError("RunPod network volume payload is invalid")
    volume_id = receipt.get("network_volume_id")
    data_center_id = receipt.get("network_volume_data_center_id")
    volume_size_gb = receipt.get("network_volume_size_gb")
    provider_id = volume.get("id", volume.get("networkVolumeId"))
    provider_data_center = volume.get("dataCenterId")
    provider_size = volume.get("size")
    if (
        not isinstance(volume_id, str)
        or not volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
        or provider_id != volume_id
        or provider_data_center != data_center_id
        or type(provider_size) is not int
        or provider_size < volume_size_gb
    ):
        raise GateError("RunPod network volume does not match the predecessor seed")
    bridge = {
        "schema_version": 1,
        "bridge_revision": PREDECESSOR_READINESS_BRIDGE_REVISION,
        "current_profile_id": manifest["profile_id"],
        "predecessor_profile_id": PREDECESSOR_PROFILE_ID,
        "predecessor_manifest_commit": PREDECESSOR_MANIFEST_COMMIT,
        "predecessor_manifest_digest": PREDECESSOR_MANIFEST_DIGEST,
        "predecessor_receipt_digest": receipt["receipt_digest"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "snapshot_file_contract": manifest["model"]["snapshot_files"],
        "dependencies": manifest["runtime"]["dependencies"],
        "network_volume_id": provider_id,
        "network_volume_data_center_id": provider_data_center,
        "network_volume_size_gb": provider_size,
        "seed_only": True,
        "remote_current_attestation_required": True,
        "verified_at": current_time.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    bridge["bridge_digest"] = _receipt_digest(bridge)
    return bridge


def retention_checkpoint_evidence_digest(evidence: Mapping[str, Any]) -> str:
    """Return the canonical digest of integrated checkpoint evidence."""

    material = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


RETENTION_CHECKPOINT_STORAGE_EVIDENCE_KEYS = (
    "checkpoint_storage_scope",
    "checkpoint_storage_root",
    "checkpoint_storage_root_device",
    "checkpoint_storage_checkpoint_device",
    "checkpoint_storage_root_inode",
    "checkpoint_inode_before_reopen",
    "checkpoint_inode_after_reopen",
    "checkpoint_persist_process_pid",
    "checkpoint_resume_process_pid",
    "checkpoint_resume_parent_process_pid",
    "checkpoint_reopened_after_fsync",
    "checkpoint_source_device",
    "checkpoint_authentication_revision",
    "checkpoint_authentication_mechanism_digest",
    "checkpoint_generation",
    "checkpoint_authenticated_private_resume",
)


def retention_checkpoint_storage_evidence_digest(
    evidence: Mapping[str, Any],
) -> str:
    """Digest the exact persistent-storage and fresh-process checkpoint facts."""

    material = {key: evidence.get(key) for key in RETENTION_CHECKPOINT_STORAGE_EVIDENCE_KEYS}
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def build_live_stage_activation(
    manifest: Mapping[str, Any],
    *,
    head_commit: str,
    workload_bundle_digest: str,
    workload_bundle_size_bytes: int,
    workload_bundle_path: str,
    bundle_stage_receipt_digest: str,
    bootstrap_source_digest: str,
    volume_readiness_receipt_digest: str,
    torch_retention_evidence_digest: str,
    dependency_lock_digest: str,
    dependency_quarantine_evidence_digest: str,
    dependency_private_tree_digest: str,
    code_materialization_evidence_digest: str,
    code_private_tree_digest: str,
    network_volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
) -> dict[str, Any]:
    """Build the exact activation body that releases a staged screen workload."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise GateError("live-stage dependency lock digest is invalid")
    if (
        not isinstance(head_commit, str)
        or len(head_commit) != 40
        or any(character not in "0123456789abcdef" for character in head_commit)
    ):
        raise GateError("live-stage activation HEAD commit is invalid")
    expected_bundle_path = (
        f"/workspace/equinox-state/workload-bundles/{manifest['profile_id']}/"
        f"{_required_sha256(workload_bundle_digest, 'workload bundle digest')[7:]}.tar.xz"
    )
    if (
        workload_bundle_path != expected_bundle_path
        or type(workload_bundle_size_bytes) is not int
        or not 0 < workload_bundle_size_bytes <= MAXIMUM_WORKLOAD_BUNDLE_BYTES
        or not isinstance(network_volume_id, str)
        or not network_volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
    ):
        raise GateError("live-stage activation handoff identity is invalid")
    body = {
        "revision": manifest["screen"]["live_stage_activation_revision"],
        "profile_id": manifest["profile_id"],
        "head_commit": head_commit,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "bootstrap_source_digest": _required_sha256(
            bootstrap_source_digest,
            "bootstrap source digest",
        ),
        "workload_bundle_digest": workload_bundle_digest,
        "workload_bundle_size_bytes": workload_bundle_size_bytes,
        "workload_bundle_path": workload_bundle_path,
        "bundle_stage_receipt_digest": _required_sha256(
            bundle_stage_receipt_digest,
            "bundle stage receipt digest",
        ),
        "volume_readiness_receipt_digest": _required_sha256(
            volume_readiness_receipt_digest,
            "volume readiness receipt digest",
        ),
        "torch_retention_evidence_digest": _required_sha256(
            torch_retention_evidence_digest,
            "Torch retention evidence digest",
        ),
        "dependency_lock_digest": _required_sha256(
            dependency_lock_digest,
            "dependency lock digest",
        ),
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_quarantine_evidence_digest": _required_sha256(
            dependency_quarantine_evidence_digest,
            "dependency quarantine evidence digest",
        ),
        "dependency_private_tree_digest": _required_sha256(
            dependency_private_tree_digest,
            "dependency private tree digest",
        ),
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": _required_sha256(
            code_materialization_evidence_digest,
            "code materialization evidence digest",
        ),
        "code_private_tree_digest": _required_sha256(
            code_private_tree_digest,
            "code private tree digest",
        ),
        "network_volume_id": network_volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
    }
    body["activation_digest"] = "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def verify_retention_checkpoint_evidence(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    head_commit: str,
    workload_bundle_digest: str,
    workload_bundle_size_bytes: int,
    workload_bundle_path: str,
    bundle_stage_receipt_digest: str,
    bootstrap_source_digest: str,
    volume_readiness_receipt_digest: str,
    dependency_lock_digest: str,
    dependency_quarantine_evidence_digest: str,
    dependency_private_tree_digest: str,
    code_materialization_evidence_digest: str,
    code_private_tree_digest: str,
    network_volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
) -> dict[str, Any]:
    """Verify a flat-bundle real AdamW checkpoint proof against one exact handoff."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise GateError("retention checkpoint dependency lock digest is invalid")
    required_keys = {
        "schema_version",
        "evidence_revision",
        "status",
        "test_id",
        "profile_id",
        "head_commit",
        "source_contract_digest",
        "workload_bundle_digest",
        "workload_bundle_size_bytes",
        "workload_bundle_path",
        "bundle_stage_receipt_digest",
        "bootstrap_source_digest",
        "volume_readiness_receipt_digest",
        "dependency_lock_digest",
        "dependency_quarantine_revision",
        "dependency_quarantine_evidence_digest",
        "dependency_private_tree_digest",
        "code_materialization_revision",
        "code_materialization_evidence_digest",
        "code_private_tree_digest",
        "network_volume_id",
        "network_volume_data_center_id",
        "network_volume_size_gb",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "gpu_name",
        "gpu_total_memory_bytes",
        "bf16_supported",
        "probe_sha256",
        "trainer_sha256",
        "environment_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "restored_weight_before_resume_step",
        "advanced_weight_after_resume_step",
        "effective_policy_update_count_before_resume_step",
        "effective_policy_update_count_after_resume_step",
        "retained_observation_after_resume_step",
        "optimizer_state_entries_after_resume_step",
        "optimizer_state_digest_before_persist",
        "optimizer_state_digest_after_restore",
        "optimizer_state_digest_after_resume_step",
        "optimizer_parameter_device",
        "optimizer_state_devices_before_persist",
        "optimizer_state_devices_after_restore",
        "checkpoint_storage_scope",
        "checkpoint_storage_root",
        "checkpoint_storage_root_device",
        "checkpoint_storage_checkpoint_device",
        "checkpoint_storage_root_inode",
        "checkpoint_inode_before_reopen",
        "checkpoint_inode_after_reopen",
        "checkpoint_source_device",
        "checkpoint_persist_process_pid",
        "checkpoint_resume_process_pid",
        "checkpoint_resume_parent_process_pid",
        "checkpoint_reopened_after_fsync",
        "checkpoint_storage_evidence_digest",
        "checkpoint_authentication_revision",
        "checkpoint_authentication_mechanism_digest",
        "checkpoint_generation",
        "checkpoint_manifest_digest",
        "checkpoint_authenticated_private_resume",
        "evidence_digest",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_keys:
        raise GateError("retention checkpoint evidence has an invalid field set")
    if (
        not isinstance(head_commit, str)
        or len(head_commit) != 40
        or any(character not in "0123456789abcdef" for character in head_commit)
    ):
        raise GateError("retention checkpoint expected HEAD commit is invalid")
    expected_bundle_path = (
        f"/workspace/equinox-state/workload-bundles/{manifest['profile_id']}/"
        f"{_required_sha256(workload_bundle_digest, 'workload bundle digest')[7:]}.tar.xz"
    )
    if workload_bundle_path != expected_bundle_path:
        raise GateError("retention checkpoint workload bundle path is invalid")
    if (
        type(workload_bundle_size_bytes) is not int
        or not 0 < workload_bundle_size_bytes <= MAXIMUM_WORKLOAD_BUNDLE_BYTES
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
        or not isinstance(network_volume_id, str)
        or not network_volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
    ):
        raise GateError("retention checkpoint handoff identity is invalid")
    expected_identity = {
        "schema_version": 3,
        "evidence_revision": RETENTION_CHECKPOINT_EVIDENCE_REVISION,
        "status": "passed",
        "test_id": (
            "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        ),
        "profile_id": manifest["profile_id"],
        "head_commit": head_commit,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "workload_bundle_digest": workload_bundle_digest,
        "workload_bundle_size_bytes": workload_bundle_size_bytes,
        "workload_bundle_path": workload_bundle_path,
        "bundle_stage_receipt_digest": _required_sha256(
            bundle_stage_receipt_digest,
            "bundle stage receipt digest",
        ),
        "bootstrap_source_digest": _required_sha256(
            bootstrap_source_digest,
            "bootstrap source digest",
        ),
        "volume_readiness_receipt_digest": _required_sha256(
            volume_readiness_receipt_digest,
            "volume readiness receipt digest",
        ),
        "dependency_lock_digest": _required_sha256(
            dependency_lock_digest,
            "dependency lock digest",
        ),
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_quarantine_evidence_digest": _required_sha256(
            dependency_quarantine_evidence_digest,
            "dependency quarantine evidence digest",
        ),
        "dependency_private_tree_digest": _required_sha256(
            dependency_private_tree_digest,
            "dependency private tree digest",
        ),
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": _required_sha256(
            code_materialization_evidence_digest,
            "code materialization evidence digest",
        ),
        "code_private_tree_digest": _required_sha256(
            code_private_tree_digest,
            "code private tree digest",
        ),
        "network_volume_id": network_volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": expected_cuda_version(manifest),
        "cuda_available": True,
        "gpu_name": manifest["hardware"]["gpu_id"],
        "bf16_supported": True,
        "probe_sha256": (
            "sha256:" + manifest["source_contract"]["files"]["retention_checkpoint_probe.py"]
        ),
        "trainer_sha256": (
            "sha256:"
            + manifest["source_contract"]["files"]["repository_repair_large_model_trainer.py"]
        ),
        "environment_sha256": (
            "sha256:" + manifest["source_contract"]["files"]["repository_repair_env.py"]
        ),
        "source_sha256": {
            name: f"sha256:{digest}"
            for name, digest in manifest["source_contract"]["files"].items()
        },
        "effective_policy_update_count_before_resume_step": 1,
        "effective_policy_update_count_after_resume_step": 2,
        "retained_observation_after_resume_step": {"exact_rate": 0.75},
        "optimizer_parameter_device": "cuda:0",
        "optimizer_state_devices_before_persist": {
            "exp_avg": ["cuda:0"],
            "exp_avg_sq": ["cuda:0"],
            "step": ["cpu"],
        },
        "optimizer_state_devices_after_restore": {
            "exp_avg": ["cuda:0"],
            "exp_avg_sq": ["cuda:0"],
            "step": ["cpu"],
        },
        "checkpoint_storage_scope": "runpod-network-volume",
        "checkpoint_reopened_after_fsync": True,
        "checkpoint_authentication_revision": CHECKPOINT_AUTHENTICATION_REVISION,
        "checkpoint_authentication_mechanism_digest": (
            checkpoint_authentication_mechanism_digest()
        ),
        "checkpoint_generation": 1,
        "checkpoint_authenticated_private_resume": True,
    }
    for key, expected in expected_identity.items():
        try:
            _expect_exact(evidence.get(key), expected, f"retention checkpoint evidence {key}")
        except GateError as error:
            raise GateError(f"retention checkpoint evidence {key} is invalid") from error
    if (
        type(evidence.get("gpu_total_memory_bytes")) is not int
        or evidence["gpu_total_memory_bytes"] < manifest["hardware"]["minimum_cuda_memory_bytes"]
    ):
        raise GateError("retention checkpoint evidence GPU memory is invalid")
    for key in (
        "checkpoint_sha256",
        "checkpoint_manifest_digest",
        "optimizer_state_digest_before_persist",
        "optimizer_state_digest_after_restore",
        "optimizer_state_digest_after_resume_step",
        "checkpoint_storage_evidence_digest",
        "evidence_digest",
    ):
        _required_sha256(evidence.get(key), f"retention checkpoint evidence {key}")
    checkpoint_size = evidence.get("checkpoint_size_bytes")
    storage_root = evidence.get("checkpoint_storage_root")
    storage_root_parts = PurePosixPath(storage_root).parts if isinstance(storage_root, str) else ()
    storage_root_device = evidence.get("checkpoint_storage_root_device")
    checkpoint_device = evidence.get("checkpoint_storage_checkpoint_device")
    checkpoint_source_device = evidence.get("checkpoint_source_device")
    storage_root_inode = evidence.get("checkpoint_storage_root_inode")
    checkpoint_inode_before = evidence.get("checkpoint_inode_before_reopen")
    checkpoint_inode_after = evidence.get("checkpoint_inode_after_reopen")
    persist_pid = evidence.get("checkpoint_persist_process_pid")
    resume_pid = evidence.get("checkpoint_resume_process_pid")
    resume_parent_pid = evidence.get("checkpoint_resume_parent_process_pid")
    optimizer_entries = evidence.get("optimizer_state_entries_after_resume_step")
    before = evidence.get("restored_weight_before_resume_step")
    after = evidence.get("advanced_weight_after_resume_step")
    if (
        type(checkpoint_size) is not int
        or checkpoint_size <= 0
        or len(storage_root_parts) != 4
        or storage_root_parts[:3] != ("/", "workspace", "equinox-runs")
        or storage_root_parts[3] in {"", ".", ".."}
        or type(storage_root_device) is not int
        or storage_root_device <= 0
        or checkpoint_device != storage_root_device
        or checkpoint_source_device != checkpoint_device
        or type(storage_root_inode) is not int
        or storage_root_inode <= 0
        or type(checkpoint_inode_before) is not int
        or checkpoint_inode_before <= 0
        or checkpoint_inode_after != checkpoint_inode_before
        or type(persist_pid) is not int
        or persist_pid <= 1
        or type(resume_pid) is not int
        or resume_pid <= 1
        or resume_pid == persist_pid
        or resume_parent_pid != persist_pid
        or type(optimizer_entries) is not int
        or optimizer_entries <= 0
        or not isinstance(before, list)
        or len(before) != 1
        or not isinstance(before[0], int | float)
        or isinstance(before[0], bool)
        or not math.isfinite(float(before[0]))
        or not isinstance(after, list)
        or len(after) != 1
        or not isinstance(after[0], int | float)
        or isinstance(after[0], bool)
        or not math.isfinite(float(after[0]))
        or before == after
        or evidence["optimizer_state_digest_before_persist"]
        != evidence["optimizer_state_digest_after_restore"]
        or evidence["optimizer_state_digest_after_restore"]
        == evidence["optimizer_state_digest_after_resume_step"]
    ):
        raise GateError("retention checkpoint persistence evidence is invalid")
    if evidence[
        "checkpoint_storage_evidence_digest"
    ] != retention_checkpoint_storage_evidence_digest(evidence):
        raise GateError("retention checkpoint storage evidence digest is invalid")
    if evidence["evidence_digest"] != retention_checkpoint_evidence_digest(evidence):
        raise GateError("retention checkpoint evidence digest is invalid")
    return {
        "profile_id": manifest["profile_id"],
        "head_commit": evidence["head_commit"],
        "source_contract_digest": evidence["source_contract_digest"],
        "workload_bundle_digest": evidence["workload_bundle_digest"],
        "bundle_stage_receipt_digest": evidence["bundle_stage_receipt_digest"],
        "bootstrap_source_digest": evidence["bootstrap_source_digest"],
        "volume_readiness_receipt_digest": evidence["volume_readiness_receipt_digest"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "checkpoint_authentication_mechanism_digest": evidence[
            "checkpoint_authentication_mechanism_digest"
        ],
        "evidence_digest": evidence["evidence_digest"],
    }


def verify_pilot_authorization(
    manifest: Mapping[str, Any],
    screen_result: Mapping[str, Any],
    provider_receipt: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Authorize one pilot only from a fresh, exact, fully torn-down screen."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if not isinstance(screen_result, Mapping):
        raise GateError("screen result must be an object")
    if not isinstance(provider_receipt, Mapping):
        raise GateError("provider receipt must be an object")
    expected_result_identity = {
        "workload": manifest["screen"]["workload"],
        "workload_revision": manifest["screen"]["workload_revision"],
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "environment_revision": manifest["interface"]["environment_revision"],
        "action_protocol_revision": manifest["interface"]["action_protocol_revision"],
        "terminal_submission_contract": manifest["interface"]["terminal_submission_contract"],
        "screen_levels": manifest["screen"]["admission_levels"],
        "shared_prefix_checkpoint_strategy": manifest["screen"][
            "shared_prefix_checkpoint_strategy"
        ],
        "branch_width": manifest["screen"]["branch_width"],
        "training_microbatch_size": manifest["screen"]["training_microbatch_size"],
        "maximum_input_tokens": manifest["screen"]["maximum_input_tokens"],
        "capacity_smoke_sequence_tokens": (
            max(
                manifest["screen"]["maximum_input_tokens"],
                manifest["pilot"]["maximum_input_tokens"],
            )
            + CAPPED_GENERATION_TOKENS
        ),
        "optimization_seed": manifest["screen_limits"]["optimization_seed"],
        "capacity_smoke_completed": True,
        "gradient_checkpointing_enabled": True,
        "determinism": manifest["pilot"]["determinism"],
        "pinned_snapshot_digest": expected_snapshot_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
        "live_stage_activation_revision": manifest["screen"]["live_stage_activation_revision"],
        "preparation_evidence_complete": True,
    }
    for key, expected in expected_result_identity.items():
        if screen_result.get(key) != expected:
            raise GateError(f"screen result {key} does not match the larger-model profile")
    if screen_result.get("screen_completed") is not True:
        raise GateError("screen result is incomplete")
    if screen_result.get("eligible") is not True:
        raise GateError("screen result did not authorize a pilot")
    if screen_result.get("policy_mutation_detected") is not False:
        raise GateError("screen result did not prove that policy weights were unchanged")
    if screen_result.get("optimizer_state_restored") is not True:
        raise GateError("screen result did not prove optimizer-state restoration")
    if screen_result.get("test_split_accessed") is not False:
        raise GateError("screen result did not preserve test-split isolation")
    if screen_result.get("training_started") is not False:
        raise GateError("screen result training_started must be false")
    if screen_result.get("policy_mutation_enabled") is not False:
        raise GateError("screen result policy_mutation_enabled must be false")

    expected_level_keys = {str(level) for level in manifest["screen"]["admission_levels"]}
    expected_baseline_examples = manifest["screen"]["baseline_examples_per_level"] * len(
        expected_level_keys
    )
    exact_screen_counts = {
        "persistent_policy_updates": 0,
        "test_examples_accessed": 0,
        "branch_groups": manifest["screen"]["branch_groups"],
        "expected_baseline_examples": expected_baseline_examples,
        "completed_baseline_examples": expected_baseline_examples,
    }
    for key, expected in exact_screen_counts.items():
        observed = _required_integer(screen_result.get(key), f"screen result {key}")
        if observed != expected:
            raise GateError(f"screen result {key} does not match the eligibility profile")

    gate_results = screen_result.get("gate_results")
    required_gates = tuple(manifest["authorization"]["required_gate_results"])
    if not isinstance(gate_results, dict) or set(gate_results) != set(required_gates):
        raise GateError("screen result gate set does not match the eligibility profile")
    failed_gates = sorted(key for key in required_gates if gate_results.get(key) is not True)
    if failed_gates:
        raise GateError(f"screen result has failed gates: {failed_gates}")

    thresholds = manifest["screen"]["thresholds"]
    metric_limits = {
        "baseline_checkpoint_rate": (
            thresholds["minimum_baseline_checkpoint_rate"],
            1.0,
        ),
        "action_protocol_validity": (
            thresholds["minimum_action_protocol_validity"],
            1.0,
        ),
        "branch_checkpoint_rate": (
            thresholds["minimum_branch_checkpoint_rate"],
            1.0,
        ),
        "informative_group_rate": (
            thresholds["minimum_informative_group_rate"],
            1.0,
        ),
        "solved_sibling_rate": (
            thresholds["minimum_solved_sibling_rate"],
            thresholds["maximum_solved_sibling_rate"],
        ),
        "baseline_exact_rate": (
            thresholds["minimum_baseline_exact_rate"],
            thresholds["maximum_baseline_exact_rate"],
        ),
        "peak_reserved_vram_fraction": (
            0.0,
            manifest["hardware"]["maximum_peak_reserved_vram_fraction"],
        ),
        "predicted_final_evaluation_seconds": (
            0.0,
            thresholds["maximum_predicted_final_evaluation_seconds"],
        ),
    }
    for key, (minimum, maximum) in metric_limits.items():
        observed = _required_number(screen_result.get(key), f"screen result {key}")
        if not minimum <= observed <= maximum:
            raise GateError(f"screen result {key}={observed} is outside [{minimum}, {maximum}]")
    if (
        _required_number(
            screen_result.get("predicted_final_evaluation_seconds"),
            "screen result predicted_final_evaluation_seconds",
        )
        <= 0
    ):
        raise GateError("screen result did not measure positive pilot runtime")

    per_level_rates = screen_result.get("per_level_checkpoint_rates")
    if not isinstance(per_level_rates, dict) or set(per_level_rates) != expected_level_keys:
        raise GateError("screen result checkpoint rates do not match the admission levels")
    per_level_minimum = thresholds["minimum_per_level_checkpoint_rate"]
    for level, raw_rate in per_level_rates.items():
        rate = _required_number(
            raw_rate,
            f"screen result per_level_checkpoint_rates[{level}]",
        )
        if not per_level_minimum <= rate <= 1.0:
            raise GateError(f"screen result level {level} checkpoint rate is below threshold")

    count_limits = {
        "informative_groups": (
            thresholds["minimum_informative_groups"],
            manifest["screen"]["branch_groups"],
        ),
        "solved_siblings": (
            thresholds["minimum_solved_siblings"],
            manifest["screen"]["branch_groups"] * manifest["screen"]["branch_width"],
        ),
        "failed_siblings": (
            thresholds["minimum_failed_siblings"],
            manifest["screen"]["branch_groups"] * manifest["screen"]["branch_width"],
        ),
    }
    for key, (minimum, maximum) in count_limits.items():
        observed = _required_integer(screen_result.get(key), f"screen result {key}")
        if not minimum <= observed <= maximum:
            raise GateError(f"screen result {key}={observed} is outside [{minimum}, {maximum}]")
    if screen_result["solved_siblings"] + screen_result["failed_siblings"] > (
        manifest["screen"]["branch_groups"] * manifest["screen"]["branch_width"]
    ):
        raise GateError("screen result sibling outcome counts exceed the sampled budget")

    observed_digest = result_digest(screen_result)
    embedded_digest = screen_result.get("digest")
    if embedded_digest is not None and embedded_digest != observed_digest:
        raise GateError("screen result self-reported digest does not match its content")

    expected_receipt_keys = {
        "provider_name",
        "provider_handle",
        "provider_cli_version",
        "resource_profile",
        "workload",
        "result",
        "started_at",
        "completed_at",
        "teardown_confirmed",
    }
    if set(provider_receipt) != expected_receipt_keys:
        raise GateError("provider receipt has an invalid field set")
    if provider_receipt.get("provider_name") != "RunPod":
        raise GateError("provider receipt provider_name does not identify RunPod")
    provider_version = provider_receipt.get("provider_cli_version")
    if not isinstance(provider_version, str) or not provider_version:
        raise GateError("provider receipt provider_cli_version is invalid")
    receipt_result = provider_receipt.get("result")
    if not isinstance(receipt_result, Mapping) or dict(receipt_result) != dict(screen_result):
        raise GateError("provider receipt does not cover the exact screen result")
    if provider_receipt.get("teardown_confirmed") is not True:
        raise GateError("provider teardown is not confirmed")
    workload = provider_receipt.get("workload")
    if not isinstance(workload, Mapping):
        raise GateError("provider receipt workload is invalid")
    expected_workload_identity = {
        "id": manifest["screen"]["workload"],
        "revision": manifest["screen"]["workload_revision"],
        "static_branch_width": manifest["screen"]["branch_width"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
    }
    for key, expected in expected_workload_identity.items():
        if workload.get(key) != expected:
            raise GateError(
                f"provider receipt workload {key} does not match the larger-model profile"
            )
    resource_profile = provider_receipt.get("resource_profile")
    if not isinstance(resource_profile, Mapping):
        raise GateError("provider receipt resource profile is invalid")
    expected_resource_identity = {
        "profile_id": manifest["profile_id"],
        "gpu_id": manifest["hardware"]["gpu_id"],
        "image": manifest["runtime"]["image"],
        "image_digest": manifest["runtime"]["image_digest"],
        "manifest_digest": ("sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest()),
        "source_contract_digest": expected_source_contract_digest(manifest),
    }
    for key, expected in expected_resource_identity.items():
        if resource_profile.get(key) != expected:
            raise GateError(
                f"provider receipt resource profile {key} does not match the larger-model profile"
            )
    network_volume_id = resource_profile.get("network_volume_id")
    if not isinstance(network_volume_id, str) or not network_volume_id:
        raise GateError("provider receipt does not bind a RunPod network volume")
    network_volume_data_center_id = resource_profile.get("network_volume_data_center_id")
    network_volume_size_gb = resource_profile.get("network_volume_size_gb")
    source_head_commit = resource_profile.get("source_head_commit")
    if (
        not isinstance(network_volume_data_center_id, str)
        or not network_volume_data_center_id
        or type(network_volume_size_gb) is not int
        or network_volume_size_gb < manifest["hardware"]["volume_disk_gb"]
        or not isinstance(source_head_commit, str)
        or len(source_head_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_head_commit)
    ):
        raise GateError("provider receipt live-stage source or volume identity is invalid")
    bundle_handoff_revision = resource_profile.get("bundle_handoff_revision")
    if bundle_handoff_revision != BUNDLE_HANDOFF_REVISION:
        raise GateError("provider receipt bundle handoff revision is invalid")
    workload_bundle_digest = _required_sha256(
        resource_profile.get("workload_bundle_digest"),
        "provider receipt workload bundle digest",
    )
    bundle_stage_receipt_digest = _required_sha256(
        resource_profile.get("bundle_stage_receipt_digest"),
        "provider receipt bundle stage receipt digest",
    )
    bootstrap_source_digest = _required_sha256(
        resource_profile.get("bootstrap_source_digest"),
        "provider receipt bootstrap source digest",
    )
    volume_readiness_receipt_digest = _required_sha256(
        resource_profile.get("volume_readiness_receipt_digest"),
        "provider receipt volume readiness receipt digest",
    )
    torch_retention_evidence_digest = _required_sha256(
        resource_profile.get("torch_retention_evidence_digest"),
        "provider receipt Torch retention evidence digest",
    )
    dependency_lock_digest = _required_sha256(
        resource_profile.get("dependency_lock_digest"),
        "provider receipt dependency lock digest",
    )
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise GateError("provider receipt dependency lock digest is invalid")
    dependency_quarantine_evidence_digest = _required_sha256(
        resource_profile.get("dependency_quarantine_evidence_digest"),
        "provider receipt dependency quarantine evidence digest",
    )
    dependency_private_tree_digest = _required_sha256(
        resource_profile.get("dependency_private_tree_digest"),
        "provider receipt dependency private tree digest",
    )
    code_materialization_evidence_digest = _required_sha256(
        resource_profile.get("code_materialization_evidence_digest"),
        "provider receipt code materialization evidence digest",
    )
    code_private_tree_digest = _required_sha256(
        resource_profile.get("code_private_tree_digest"),
        "provider receipt code private tree digest",
    )
    if (
        resource_profile.get("dependency_quarantine_revision")
        != manifest["materialization"]["dependency_lock"]["revision"]
        or resource_profile.get("code_materialization_revision")
        != manifest["materialization"]["code"]["revision"]
    ):
        raise GateError("provider receipt private materialization revision is invalid")
    workload_bundle_size_bytes = resource_profile.get("workload_bundle_size_bytes")
    if (
        type(workload_bundle_size_bytes) is not int
        or not 0 < workload_bundle_size_bytes <= MAXIMUM_WORKLOAD_BUNDLE_BYTES
    ):
        raise GateError("provider receipt workload bundle size is invalid")
    if resource_profile.get("workload_bundle_compression") != "xz":
        raise GateError("provider receipt workload bundle compression is invalid")
    workload_bundle_path = resource_profile.get("workload_bundle_path")
    expected_bundle_path = (
        f"/workspace/equinox-state/workload-bundles/{manifest['profile_id']}/"
        f"{workload_bundle_digest.removeprefix('sha256:')}.tar.xz"
    )
    if workload_bundle_path != expected_bundle_path:
        raise GateError("provider receipt workload bundle path is invalid")
    activation = build_live_stage_activation(
        manifest,
        head_commit=source_head_commit,
        workload_bundle_digest=workload_bundle_digest,
        workload_bundle_size_bytes=workload_bundle_size_bytes,
        workload_bundle_path=workload_bundle_path,
        bundle_stage_receipt_digest=bundle_stage_receipt_digest,
        bootstrap_source_digest=bootstrap_source_digest,
        volume_readiness_receipt_digest=volume_readiness_receipt_digest,
        torch_retention_evidence_digest=torch_retention_evidence_digest,
        dependency_lock_digest=dependency_lock_digest,
        dependency_quarantine_evidence_digest=(dependency_quarantine_evidence_digest),
        dependency_private_tree_digest=dependency_private_tree_digest,
        code_materialization_evidence_digest=code_materialization_evidence_digest,
        code_private_tree_digest=code_private_tree_digest,
        network_volume_id=network_volume_id,
        data_center_id=network_volume_data_center_id,
        volume_size_gb=network_volume_size_gb,
    )
    if (
        resource_profile.get("live_stage_activation_revision") != activation["revision"]
        or resource_profile.get("live_stage_activation_digest") != activation["activation_digest"]
    ):
        raise GateError("provider receipt live-stage activation evidence is invalid")
    for key, expected in {
        "source_head_commit": source_head_commit,
        "live_stage_activation_digest": activation["activation_digest"],
        "volume_readiness_receipt_digest": volume_readiness_receipt_digest,
        "torch_retention_evidence_digest": torch_retention_evidence_digest,
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_quarantine_evidence_digest": (dependency_quarantine_evidence_digest),
        "dependency_private_tree_digest": dependency_private_tree_digest,
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": code_materialization_evidence_digest,
        "code_private_tree_digest": code_private_tree_digest,
    }.items():
        if screen_result.get(key) != expected:
            raise GateError(f"screen result {key} does not match the provider handoff")

    volume_readiness_receipt = screen_result.get("volume_readiness_receipt")
    if not isinstance(volume_readiness_receipt, Mapping):
        raise GateError("screen result volume readiness receipt is invalid")
    verified_volume = verify_volume_readiness_receipt(
        manifest,
        volume_readiness_receipt,
        {
            "id": network_volume_id,
            "dataCenterId": network_volume_data_center_id,
            "size": network_volume_size_gb,
        },
        now=_utc_timestamp(
            provider_receipt.get("completed_at"),
            "receipt completed_at",
        ),
    )
    if verified_volume["receipt_digest"] != volume_readiness_receipt_digest:
        raise GateError("screen result volume readiness receipt digest is invalid")
    for key, expected in {
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_evidence_digest": (dependency_quarantine_evidence_digest),
        "dependency_private_tree_digest": dependency_private_tree_digest,
        "code_materialization_evidence_digest": code_materialization_evidence_digest,
        "code_private_tree_digest": code_private_tree_digest,
    }.items():
        if verified_volume.get(key) != expected:
            raise GateError(f"volume readiness receipt {key} does not match the provider handoff")
    retention_checkpoint_evidence = screen_result.get("retention_checkpoint_evidence")
    if not isinstance(retention_checkpoint_evidence, Mapping):
        raise GateError("screen result retention checkpoint evidence is invalid")
    verified_retention = verify_retention_checkpoint_evidence(
        manifest,
        retention_checkpoint_evidence,
        head_commit=source_head_commit,
        workload_bundle_digest=workload_bundle_digest,
        workload_bundle_size_bytes=workload_bundle_size_bytes,
        workload_bundle_path=workload_bundle_path,
        bundle_stage_receipt_digest=bundle_stage_receipt_digest,
        bootstrap_source_digest=bootstrap_source_digest,
        volume_readiness_receipt_digest=volume_readiness_receipt_digest,
        dependency_lock_digest=dependency_lock_digest,
        dependency_quarantine_evidence_digest=(dependency_quarantine_evidence_digest),
        dependency_private_tree_digest=dependency_private_tree_digest,
        code_materialization_evidence_digest=code_materialization_evidence_digest,
        code_private_tree_digest=code_private_tree_digest,
        network_volume_id=network_volume_id,
        data_center_id=network_volume_data_center_id,
        volume_size_gb=network_volume_size_gb,
    )
    if verified_retention["evidence_digest"] != torch_retention_evidence_digest:
        raise GateError("screen result Torch retention evidence digest is invalid")
    checkpoint_mechanism_digest = verified_retention["checkpoint_authentication_mechanism_digest"]
    if (
        screen_result.get("checkpoint_authentication_mechanism_digest")
        != checkpoint_mechanism_digest
    ):
        raise GateError("screen result checkpoint authentication mechanism digest is invalid")
    provider_handle = provider_receipt.get("provider_handle")
    if not isinstance(provider_handle, str) or not provider_handle.startswith("runpod://pods/"):
        raise GateError("provider receipt has no valid RunPod handle")

    started_at = _utc_timestamp(provider_receipt.get("started_at"), "receipt started_at")
    completed_at = _utc_timestamp(provider_receipt.get("completed_at"), "receipt completed_at")
    if started_at > completed_at:
        raise GateError("provider receipt completion precedes its start")
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise GateError("authorization time must be timezone-aware")
    current_time = current_time.astimezone(UTC)
    age_seconds = (current_time - completed_at).total_seconds()
    maximum_age_seconds = manifest["authorization"]["maximum_age_hours"] * 3_600
    if age_seconds < 0:
        raise GateError("provider receipt completion time is in the future")
    if age_seconds > maximum_age_seconds:
        raise GateError("provider receipt is too old to authorize a pilot")

    authorization = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "screen_workload_revision": manifest["screen"]["workload_revision"],
        "optimization_seed": manifest["pilot_limits"]["optimization_seed"],
        "screen_result_digest": observed_digest,
        "pinned_snapshot_digest": expected_snapshot_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
        "terminal_submission_contract": manifest["interface"]["terminal_submission_contract"],
        "image": manifest["runtime"]["image"],
        "image_digest": manifest["runtime"]["image_digest"],
        "network_volume_id": network_volume_id,
        "network_volume_data_center_id": network_volume_data_center_id,
        "network_volume_size_gb": network_volume_size_gb,
        "bundle_handoff_revision": bundle_handoff_revision,
        "workload_bundle_digest": workload_bundle_digest,
        "workload_bundle_size_bytes": workload_bundle_size_bytes,
        "workload_bundle_compression": "xz",
        "workload_bundle_path": workload_bundle_path,
        "bundle_stage_receipt_digest": bundle_stage_receipt_digest,
        "bootstrap_source_digest": bootstrap_source_digest,
        "source_head_commit": source_head_commit,
        "live_stage_activation_revision": activation["revision"],
        "live_stage_activation_digest": activation["activation_digest"],
        "volume_readiness_receipt_digest": volume_readiness_receipt_digest,
        "torch_retention_evidence_digest": torch_retention_evidence_digest,
        "checkpoint_authentication_mechanism_digest": checkpoint_mechanism_digest,
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_quarantine_evidence_digest": (dependency_quarantine_evidence_digest),
        "dependency_private_tree_digest": dependency_private_tree_digest,
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": code_materialization_evidence_digest,
        "code_private_tree_digest": code_private_tree_digest,
        "retention_checkpoint_sha256": verified_retention["checkpoint_sha256"],
        "provider_handle": provider_handle,
        "screen_completed_at": provider_receipt["completed_at"],
        "authorized_at": current_time.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    authorization["digest"] = "sha256:" + hashlib.sha256(canonical_json(authorization)).hexdigest()
    return authorization


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GateError(f"could not read {name}: {error}") from error
    except json.JSONDecodeError as error:
        raise GateError(f"{name} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{name} must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("verify-sources")
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("path", type=Path)
    image_parser = subparsers.add_parser("verify-image-index")
    image_parser.add_argument("path", type=Path)
    subparsers.add_parser("verify-model")
    cost_parser = subparsers.add_parser("cost")
    cost_parser.add_argument("hourly_cost_usd")
    cost_parser.add_argument("lifetime_seconds")
    authorize_parser = subparsers.add_parser("authorize")
    authorize_parser.add_argument("screen_result", type=Path)
    authorize_parser.add_argument("provider_receipt", type=Path)
    for command in ("create-volume-receipt", "create-attested-volume-receipt"):
        create_volume_parser = subparsers.add_parser(command)
        create_volume_parser.add_argument("--volume-id", required=True)
        create_volume_parser.add_argument("--data-center-id", required=True)
        create_volume_parser.add_argument("--volume-size-gb", required=True, type=int)
        create_volume_parser.add_argument("--snapshot", type=Path)
        create_volume_parser.add_argument(
            "--dependency-quarantine-evidence",
            required=True,
            type=Path,
        )
        create_volume_parser.add_argument(
            "--dependency-lock-digest",
            required=True,
        )
        create_volume_parser.add_argument(
            "--dependency-quarantine-evidence-digest",
            required=True,
        )
        create_volume_parser.add_argument(
            "--dependency-private-tree-digest",
            required=True,
        )
        create_volume_parser.add_argument(
            "--code-materialization-evidence",
            required=True,
            type=Path,
        )
        create_volume_parser.add_argument(
            "--code-materialization-evidence-digest",
            required=True,
        )
        create_volume_parser.add_argument(
            "--code-private-tree-digest",
            required=True,
        )
    verify_volume_parser = subparsers.add_parser("verify-volume-receipt")
    verify_volume_parser.add_argument("receipt", type=Path)
    verify_volume_parser.add_argument("provider_volume", type=Path)
    bridge_parser = subparsers.add_parser("bridge-predecessor-volume-receipt")
    bridge_parser.add_argument("receipt", type=Path)
    bridge_parser.add_argument("provider_volume", type=Path)
    bridge_parser.add_argument("--predecessor-manifest", required=True, type=Path)
    verify_retention_parser = subparsers.add_parser("verify-retention-evidence")
    verify_retention_parser.add_argument("evidence", type=Path)
    verify_retention_parser.add_argument("--head-commit", required=True)
    verify_retention_parser.add_argument("--workload-bundle-digest", required=True)
    verify_retention_parser.add_argument(
        "--workload-bundle-size-bytes",
        required=True,
        type=int,
    )
    verify_retention_parser.add_argument("--workload-bundle-path", required=True)
    verify_retention_parser.add_argument(
        "--bundle-stage-receipt-digest",
        required=True,
    )
    verify_retention_parser.add_argument("--bootstrap-source-digest", required=True)
    verify_retention_parser.add_argument(
        "--volume-readiness-receipt-digest",
        required=True,
    )
    verify_retention_parser.add_argument(
        "--dependency-quarantine-evidence-digest",
        required=True,
    )
    verify_retention_parser.add_argument(
        "--dependency-lock-digest",
        required=True,
    )
    verify_retention_parser.add_argument(
        "--dependency-private-tree-digest",
        required=True,
    )
    verify_retention_parser.add_argument(
        "--code-materialization-evidence-digest",
        required=True,
    )
    verify_retention_parser.add_argument(
        "--code-private-tree-digest",
        required=True,
    )
    verify_retention_parser.add_argument("--volume-id", required=True)
    verify_retention_parser.add_argument("--data-center-id", required=True)
    verify_retention_parser.add_argument("--volume-size-gb", required=True, type=int)
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == "manifest":
            output: Any = manifest
        elif arguments.command == "verify-sources":
            output = verify_source_contract(manifest)
        elif arguments.command == "inventory":
            raw = (
                sys.stdin.buffer.read()
                if str(arguments.path) == "-"
                else arguments.path.read_bytes()
            )
            output = asdict(require_runpod_gpu(manifest, raw))
        elif arguments.command == "verify-image-index":
            raw = (
                sys.stdin.buffer.read()
                if str(arguments.path) == "-"
                else arguments.path.read_bytes()
            )
            output = verify_registry_image_index(manifest, raw)
        elif arguments.command == "verify-model":
            output = verify_huggingface_metadata(manifest)
        elif arguments.command == "cost":
            output = {
                "maximum_cost_usd": str(
                    lifetime_cost_bound(
                        arguments.hourly_cost_usd,
                        arguments.lifetime_seconds,
                    )
                )
            }
        elif arguments.command == "authorize":
            verify_source_contract(manifest)
            output = verify_pilot_authorization(
                manifest,
                _read_json_object(arguments.screen_result, "screen result"),
                _read_json_object(arguments.provider_receipt, "provider receipt"),
            )
        elif arguments.command in {
            "create-volume-receipt",
            "create-attested-volume-receipt",
        }:
            snapshot = arguments.snapshot or (
                Path(manifest["artifact_readiness"]["cache_directory"])
                / "hub"
                / ("models--" + manifest["model"]["id"].replace("/", "--"))
                / "snapshots"
                / manifest["model"]["revision"]
            )
            try:
                versions = {
                    package: importlib.metadata.version(package)
                    for package in manifest["runtime"]["dependencies"]
                }
            except importlib.metadata.PackageNotFoundError as error:
                raise GateError(
                    f"prewarmed dependency {error.name or 'unknown'} is unavailable"
                ) from error
            output = build_volume_readiness_receipt(
                manifest,
                snapshot,
                volume_id=arguments.volume_id,
                data_center_id=arguments.data_center_id,
                volume_size_gb=arguments.volume_size_gb,
                dependency_versions=versions,
                dependency_lock_digest=arguments.dependency_lock_digest,
                dependency_quarantine_evidence=_read_json_object(
                    arguments.dependency_quarantine_evidence,
                    "dependency quarantine evidence",
                ),
                dependency_quarantine_evidence_digest=(
                    arguments.dependency_quarantine_evidence_digest
                ),
                dependency_private_tree_digest=(arguments.dependency_private_tree_digest),
                code_materialization_evidence=_read_json_object(
                    arguments.code_materialization_evidence,
                    "code materialization evidence",
                ),
                code_materialization_evidence_digest=(
                    arguments.code_materialization_evidence_digest
                ),
                code_private_tree_digest=arguments.code_private_tree_digest,
            )
        elif arguments.command in {
            "verify-volume-receipt",
            "bridge-predecessor-volume-receipt",
        }:
            raw_volume = (
                sys.stdin.buffer.read()
                if str(arguments.provider_volume) == "-"
                else arguments.provider_volume.read_bytes()
            )
            if arguments.command == "verify-volume-receipt":
                output = verify_volume_readiness_receipt(
                    manifest,
                    _read_json_object(arguments.receipt, "volume readiness receipt"),
                    raw_volume,
                )
            else:
                output = verify_predecessor_readiness_bridge(
                    manifest,
                    _read_json_object(
                        arguments.predecessor_manifest,
                        "predecessor manifest",
                    ),
                    _read_json_object(
                        arguments.receipt,
                        "predecessor volume readiness receipt",
                    ),
                    raw_volume,
                )
        else:
            output = verify_retention_checkpoint_evidence(
                manifest,
                _read_json_object(
                    arguments.evidence,
                    "Torch retention checkpoint evidence",
                ),
                head_commit=arguments.head_commit,
                workload_bundle_digest=arguments.workload_bundle_digest,
                workload_bundle_size_bytes=arguments.workload_bundle_size_bytes,
                workload_bundle_path=arguments.workload_bundle_path,
                bundle_stage_receipt_digest=arguments.bundle_stage_receipt_digest,
                bootstrap_source_digest=arguments.bootstrap_source_digest,
                volume_readiness_receipt_digest=(arguments.volume_readiness_receipt_digest),
                dependency_lock_digest=arguments.dependency_lock_digest,
                dependency_quarantine_evidence_digest=(
                    arguments.dependency_quarantine_evidence_digest
                ),
                dependency_private_tree_digest=(arguments.dependency_private_tree_digest),
                code_materialization_evidence_digest=(
                    arguments.code_materialization_evidence_digest
                ),
                code_private_tree_digest=arguments.code_private_tree_digest,
                network_volume_id=arguments.volume_id,
                data_center_id=arguments.data_center_id,
                volume_size_gb=arguments.volume_size_gb,
            )
    except (GateError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_json(output) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
