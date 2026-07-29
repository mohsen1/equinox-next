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
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@1"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
MODEL_PARAMETER_COUNT = 7_615_616_512
MODEL_SAFETENSORS_BYTES = 15_231_271_864
SNAPSHOT_FILES = {
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
SOURCE_CONTRACT_SHA256 = {
    "repository_repair_env_v31.py": (
        "703c5badc19513cf8a7766a1a1e63fa77dce49a2f0011d78d9f4b94b64132657"
    ),
    "repository_repair_env_v32.py": (
        "3ebc2ced3fa7b790dfda0b3788ddd4a71383644e4c79902b0aec13add49aec30"
    ),
    "repository_repair_large_model_eligibility.py": (
        "b1d2ab48e702d05b4b3b3ca7c0c7eb74d8caa228eae20e4f5b4c094ecaec23af"
    ),
    "repository_repair_large_model_pilot.py": (
        "d64fbf8bb4c7e62b434b5214c9a773e9aa247d860e88a42ef01fcd336d805cdf"
    ),
    "repository_repair_study.py": (
        "7799ff8969d67ad620db0f8c6bc9ee66bd0ef0300866cd0160696ff80fae7a0c"
    ),
}
SCREEN_WORKLOAD = "repository-repair-larger-model-eligibility-screen"
SCREEN_WORKLOAD_REVISION = "larger-model-eligibility-screen@1"
CLEANUP_COST_RESERVE_SECONDS = 120
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
    "pilot_runtime_feasible",
    "policy_unchanged",
    "optimizer_state_restored",
    "test_split_isolated",
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
        "environment_revision": "repository-repair-simulator@6",
        "action_protocol_revision": "repository-repair-json-tools@5",
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
        "maximum_total_cost_usd": 3.0,
        "maximum_lifetime_seconds": 2_580,
        "boot_timeout_seconds": 360,
        "model_load_timeout_seconds": 1_200,
        "stale_progress_timeout_seconds": 600,
        "maximum_workload_attempts": 1,
        "target_runtime_seconds": 1_500,
        "retry_reserve_seconds": 300,
        "maximum_final_evaluation_reserve_seconds": 300,
        "optimization_seed": 137,
    },
    "pilot_limits": {
        "maximum_hourly_cost_usd": 4.0,
        "maximum_total_cost_usd": 16.0,
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
        "branch_width": 4,
        "validation_examples": 8,
        "training_tasks_per_update": 8,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 1_536,
        "maximum_updates": 1,
        "test_examples": 0,
        "baseline_examples_per_level": 8,
        "minimum_completed_baseline_examples": 12,
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
            "minimum_baseline_exact_rate": 0.02,
            "maximum_baseline_exact_rate": 0.75,
            "maximum_predicted_final_evaluation_seconds": 1_440,
        },
    },
    "pilot": {
        "target_runtime_seconds": 9_000,
        "maximum_updates": 40,
        "validation_examples": 8,
        "test_examples": 12,
        "training_tasks_per_update": 4,
        "replay_tasks_per_level": 1,
        "mastery_windows": 2,
        "maximum_final_evaluation_reserve_seconds": 1_800,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 1_536,
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
    for name in ("screen_limits", "pilot_limits"):
        limits = payload[name]
        bound = lifetime_cost_bound(
            limits["maximum_hourly_cost_usd"],
            limits["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS,
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


def verify_local_snapshot(
    manifest: Mapping[str, Any],
    snapshot: Path,
) -> dict[str, Any]:
    """Hash every required local artifact before any model initialization."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
    if not snapshot.is_dir():
        raise GateError("the pinned local model snapshot directory is unavailable")
    expected_files = manifest["model"]["snapshot_files"]
    observed_files: dict[str, dict[str, Any]] = {}
    for name, expected in expected_files.items():
        path = snapshot / name
        try:
            size = path.stat().st_size
        except OSError as error:
            raise GateError(f"the pinned snapshot is missing {name}") from error
        if not path.is_file() or size != expected["size_bytes"]:
            raise GateError(f"the pinned snapshot file {name} has an invalid size")
        observed_sha256 = _sha256_path(path)
        if observed_sha256 != expected["sha256"]:
            raise GateError(f"the pinned snapshot file {name} failed SHA-256 verification")
        observed_files[name] = {
            "size_bytes": size,
            "sha256": observed_sha256,
        }

    try:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
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
        "snapshot_path": str(snapshot),
        "snapshot_digest": "sha256:"
        + hashlib.sha256(canonical_json(snapshot_material)).hexdigest(),
    }


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


def verify_dependency_import_smoke(
    manifest: Mapping[str, Any],
    *,
    importer: Callable[[str], Any] = importlib.import_module,
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
        imported.append(package)
    return tuple(imported)


def build_volume_readiness_receipt(
    manifest: Mapping[str, Any],
    snapshot: Path,
    *,
    volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
    dependency_versions: Mapping[str, str],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a local receipt on a CPU-prewarmed mounted network volume."""

    _expect_exact(dict(manifest), _EXPECTED_MANIFEST, "manifest")
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
    verify_dependency_import_smoke(manifest)
    snapshot_evidence = verify_local_snapshot(manifest, snapshot)
    prepared_at = now or datetime.now(UTC)
    if prepared_at.tzinfo is None or prepared_at.utcoffset() is None:
        raise GateError("volume readiness time must be timezone-aware")
    prepared_at = prepared_at.astimezone(UTC)
    receipt = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "network_volume_id": volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
        "snapshot_digest": snapshot_evidence["snapshot_digest"],
        "dependencies": dict(dependency_versions),
        "prepared_at": prepared_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


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
        raise GateError("volume readiness receipt has an invalid field set")
    expected_identity = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "snapshot_digest": expected_snapshot_digest(manifest),
        "dependencies": manifest["runtime"]["dependencies"],
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
    if (
        not isinstance(volume_id, str)
        or not volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
    ):
        raise GateError("volume readiness receipt volume identity or size is invalid")
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
        "receipt_digest": receipt["receipt_digest"],
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
        "branch_width": manifest["screen"]["branch_width"],
        "training_microbatch_size": manifest["screen"]["training_microbatch_size"],
        "maximum_input_tokens": manifest["screen"]["maximum_input_tokens"],
        "optimization_seed": manifest["screen_limits"]["optimization_seed"],
        "capacity_smoke_completed": True,
        "gradient_checkpointing_enabled": True,
        "pinned_snapshot_digest": expected_snapshot_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
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

    completed_baseline_examples = _required_integer(
        screen_result.get("completed_baseline_examples"),
        "screen result completed_baseline_examples",
    )
    per_level_rates = screen_result.get("per_level_checkpoint_rates")
    if not isinstance(per_level_rates, dict) or set(per_level_rates) != {"0", "1", "2", "3"}:
        raise GateError("screen result per-level checkpoint rates must cover levels 0 through 3")
    expected_baseline_examples = manifest["screen"]["baseline_examples_per_level"] * len(
        per_level_rates
    )
    if completed_baseline_examples != expected_baseline_examples:
        raise GateError("eligible screen did not complete the full baseline")
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

    expected_receipt_identity = {
        "provider": manifest["hardware"]["provider"],
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "screen_workload_revision": manifest["screen"]["workload_revision"],
        "gpu_id": manifest["hardware"]["gpu_id"],
    }
    for key, expected in expected_receipt_identity.items():
        if provider_receipt.get(key) != expected:
            raise GateError(f"provider receipt {key} does not match the larger-model profile")
    if provider_receipt.get("result_digest") != observed_digest:
        raise GateError("provider receipt does not cover the exact screen result")
    if provider_receipt.get("teardown_confirmed") is not True:
        raise GateError("provider teardown is not confirmed")
    resource_profile = provider_receipt.get("resource_profile")
    if not isinstance(resource_profile, Mapping):
        raise GateError("provider receipt resource profile is invalid")
    expected_image_identity = {
        "image": manifest["runtime"]["image"],
        "image_digest": manifest["runtime"]["image_digest"],
    }
    for key, expected in expected_image_identity.items():
        if resource_profile.get(key) != expected:
            raise GateError(
                f"provider receipt resource profile {key} does not match the larger-model profile"
            )
    network_volume_id = resource_profile.get("network_volume_id")
    if not isinstance(network_volume_id, str) or not network_volume_id:
        raise GateError("provider receipt does not bind a RunPod network volume")
    provider_handle = provider_receipt.get("provider_handle")
    if not isinstance(provider_handle, str) or not provider_handle.startswith("runpod://pods/"):
        raise GateError("provider receipt has no valid RunPod handle")

    completed_at = _utc_timestamp(provider_receipt.get("completed_at"), "receipt completed_at")
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
        "image": manifest["runtime"]["image"],
        "image_digest": manifest["runtime"]["image_digest"],
        "network_volume_id": network_volume_id,
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
    create_volume_parser = subparsers.add_parser("create-volume-receipt")
    create_volume_parser.add_argument("--volume-id", required=True)
    create_volume_parser.add_argument("--data-center-id", required=True)
    create_volume_parser.add_argument("--volume-size-gb", required=True, type=int)
    create_volume_parser.add_argument("--snapshot", type=Path)
    verify_volume_parser = subparsers.add_parser("verify-volume-receipt")
    verify_volume_parser.add_argument("receipt", type=Path)
    verify_volume_parser.add_argument("provider_volume", type=Path)
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
        elif arguments.command == "create-volume-receipt":
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
            )
        else:
            raw_volume = (
                sys.stdin.buffer.read()
                if str(arguments.provider_volume) == "-"
                else arguments.provider_volume.read_bytes()
            )
            output = verify_volume_readiness_receipt(
                manifest,
                _read_json_object(arguments.receipt, "volume readiness receipt"),
                raw_volume,
            )
    except (GateError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_json(output) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
