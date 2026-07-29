"""Fail-closed eligibility contract for the first 7B RunPod pilot.

The profile is intentionally immutable. Changing a model, runtime image, dependency,
hardware class, budget, screen size, or threshold requires a new profile identity and
an accompanying code change.
"""

from __future__ import annotations

import argparse
import hashlib
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

PROFILE_ID = "qwen2.5-coder-7b-runpod-l40@1"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
MODEL_PARAMETER_COUNT = 7_615_616_512
MODEL_SAFETENSORS_BYTES = 15_231_271_864
SCREEN_WORKLOAD = "repository-repair-larger-model-eligibility-screen"
SCREEN_WORKLOAD_REVISION = "larger-model-eligibility-screen@1"
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
        "dtype": "bfloat16",
    },
    "runtime": {
        "template_id": "runpod-torch-v280",
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "dependencies": {
            "accelerate": "1.14.0",
            "peft": "0.19.1",
            "transformers": "5.14.1",
        },
    },
    "hardware": {
        "provider": "runpod",
        "cloud_type": "SECURE",
        "gpu_id": "NVIDIA L40",
        "gpu_display_name": "L40",
        "minimum_gpu_memory_gb": 48,
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
    "screen_limits": {
        "maximum_hourly_cost_usd": 1.0,
        "maximum_total_cost_usd": 0.75,
        "maximum_lifetime_seconds": 2_700,
        "model_load_timeout_seconds": 1_200,
        "maximum_workload_attempts": 1,
    },
    "pilot_limits": {
        "maximum_hourly_cost_usd": 1.0,
        "maximum_total_cost_usd": 4.0,
        "maximum_lifetime_seconds": 14_400,
        "model_load_timeout_seconds": 1_200,
        "maximum_workload_attempts": 1,
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
        },
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
            limits["maximum_lifetime_seconds"],
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
        raise GateError("the pinned RunPod L40 48 GB secure-cloud profile is unavailable")
    if len(matches) != 1:
        raise GateError("RunPod returned an ambiguous pinned GPU inventory")
    return matches[0]


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
            if type(size) is not int or size <= 0:
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


def result_digest(result: Mapping[str, Any]) -> str:
    """Digest a screen result, excluding its optional self-reported digest."""

    if not isinstance(result, Mapping):
        raise GateError("screen result must be an object")
    content = {key: value for key, value in result.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


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
    }
    for key, (minimum, maximum) in metric_limits.items():
        observed = _required_number(screen_result.get(key), f"screen result {key}")
        if not minimum <= observed <= maximum:
            raise GateError(f"screen result {key}={observed} is outside [{minimum}, {maximum}]")

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
        "screen_result_digest": observed_digest,
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
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("path", type=Path)
    subparsers.add_parser("verify-model")
    cost_parser = subparsers.add_parser("cost")
    cost_parser.add_argument("hourly_cost_usd")
    cost_parser.add_argument("lifetime_seconds")
    authorize_parser = subparsers.add_parser("authorize")
    authorize_parser.add_argument("screen_result", type=Path)
    authorize_parser.add_argument("provider_receipt", type=Path)
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == "manifest":
            output: Any = manifest
        elif arguments.command == "inventory":
            raw = (
                sys.stdin.buffer.read()
                if str(arguments.path) == "-"
                else arguments.path.read_bytes()
            )
            output = asdict(require_runpod_gpu(manifest, raw))
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
        else:
            output = verify_pilot_authorization(
                manifest,
                _read_json_object(arguments.screen_result, "screen result"),
                _read_json_object(arguments.provider_receipt, "provider receipt"),
            )
    except (GateError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_json(output) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
