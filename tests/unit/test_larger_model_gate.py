from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from research.runpod.larger_model_gate import (
    DEFAULT_MANIFEST_PATH,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_SAFETENSORS_BYTES,
    PROFILE_ID,
    REQUIRED_GATE_RESULTS,
    SCREEN_WORKLOAD,
    SCREEN_WORKLOAD_REVISION,
    GateError,
    canonical_json,
    lifetime_cost_bound,
    load_manifest,
    matching_runpod_gpus,
    parse_runpod_inventory,
    require_runpod_gpu,
    result_digest,
    verify_huggingface_metadata,
    verify_pilot_authorization,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def inventory_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "available": True,
        "communityCloud": True,
        "displayName": "L40",
        "gpuId": "NVIDIA L40",
        "memoryInGb": 48,
        "secureCloud": True,
        "stockStatus": "Low",
    }
    row.update(overrides)
    return row


def huggingface_metadata(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": MODEL_ID,
        "sha": MODEL_REVISION,
        "safetensors": {
            "parameters": {"BF16": MODEL_PARAMETER_COUNT},
            "total": MODEL_PARAMETER_COUNT,
        },
        "siblings": [
            {"rfilename": "config.json", "size": 663},
            {"rfilename": "model-00001-of-00004.safetensors", "size": 4_877_660_776},
            {"rfilename": "model-00002-of-00004.safetensors", "size": 4_932_751_008},
            {"rfilename": "model-00003-of-00004.safetensors", "size": 4_330_865_200},
            {"rfilename": "model-00004-of-00004.safetensors", "size": 1_089_994_880},
        ],
    }
    payload.update(overrides)
    return payload


def eligible_screen(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "workload": SCREEN_WORKLOAD,
        "workload_revision": SCREEN_WORKLOAD_REVISION,
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "screen_completed": True,
        "eligible": True,
        "policy_mutation_detected": False,
        "optimizer_state_restored": True,
        "test_split_accessed": False,
        "baseline_checkpoint_rate": 0.9,
        "action_protocol_validity": 1.0,
        "branch_checkpoint_rate": 0.875,
        "informative_group_rate": 0.25,
        "solved_sibling_rate": 0.25,
        "baseline_exact_rate": 0.2,
        "peak_reserved_vram_fraction": 0.8,
        "completed_baseline_examples": 32,
        "per_level_checkpoint_rates": {
            "0": 0.875,
            "1": 0.875,
            "2": 0.875,
            "3": 0.875,
        },
        "informative_groups": 2,
        "solved_siblings": 8,
        "failed_siblings": 24,
        "gate_results": dict.fromkeys(REQUIRED_GATE_RESULTS, True),
    }
    result.update(overrides)
    return result


def receipt_for(result: dict[str, object], **overrides: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "provider": "runpod",
        "provider_handle": "runpod://pods/pod-123",
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "screen_workload_revision": SCREEN_WORKLOAD_REVISION,
        "gpu_id": "NVIDIA L40",
        "result_digest": result_digest(result),
        "teardown_confirmed": True,
        "completed_at": "2026-07-29T11:00:00Z",
    }
    receipt.update(overrides)
    return receipt


def write_manifest(tmp_path: Path, transform: callable) -> Path:
    payload = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    transform(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repository_manifest_is_the_exact_bounded_profile() -> None:
    manifest = load_manifest()

    assert manifest["model"] == {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "safetensors_bytes": MODEL_SAFETENSORS_BYTES,
        "dtype": "bfloat16",
    }
    assert manifest["hardware"]["gpu_id"] == "NVIDIA L40"
    assert manifest["hardware"]["minimum_gpu_memory_gb"] == 48
    assert manifest["hardware"]["maximum_peak_reserved_vram_fraction"] == 0.85
    assert manifest["artifact_readiness"] == {
        "cache_directory": "/workspace/equinox-state/huggingface",
        "require_complete_pinned_snapshot_before_screen": True,
        "require_offline_mode_after_readiness": True,
        "allow_network_model_download_during_screen": False,
    }
    assert lifetime_cost_bound(
        manifest["screen_limits"]["maximum_hourly_cost_usd"],
        manifest["screen_limits"]["maximum_lifetime_seconds"],
    ) == Decimal("0.75")
    assert lifetime_cost_bound(
        manifest["pilot_limits"]["maximum_hourly_cost_usd"],
        manifest["pilot_limits"]["maximum_lifetime_seconds"],
    ) == Decimal("4.0")


def test_manifest_path_can_be_supplied_to_a_flattened_remote_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "larger-model-eligibility.json"
    bundled.write_bytes(DEFAULT_MANIFEST_PATH.read_bytes())
    monkeypatch.setenv("EQUINOX_LARGER_MODEL_PROFILE_PATH", str(bundled))

    assert load_manifest()["profile_id"] == PROFILE_ID


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda value: value["model"].update(revision="main"), "manifest.model.revision"),
        (lambda value: value.update(profile_id="unreviewed@1"), "manifest.profile_id"),
        (
            lambda value: value["artifact_readiness"].update(
                allow_network_model_download_during_screen=True
            ),
            "allow_network_model_download_during_screen",
        ),
        (
            lambda value: value["screen_limits"].update(maximum_workload_attempts=2),
            "maximum_workload_attempts",
        ),
        (lambda value: value.update(unreviewed=True), "keys are invalid"),
    ],
)
def test_manifest_validation_fails_closed(
    tmp_path: Path,
    transform: callable,
    message: str,
) -> None:
    path = write_manifest(tmp_path, transform)

    with pytest.raises(GateError, match=message):
        load_manifest(path)


@pytest.mark.parametrize("hourly,lifetime", [(0, 60), (1, 0), (True, 60), ("nan", 60)])
def test_lifetime_cost_rejects_unbounded_inputs(hourly: object, lifetime: object) -> None:
    with pytest.raises(GateError, match="positive finite"):
        lifetime_cost_bound(hourly, lifetime)


def test_runpod_inventory_requires_exact_available_secure_l40() -> None:
    manifest = load_manifest()
    inventory = json.dumps(
        [
            inventory_row(gpuId="NVIDIA L40S", displayName="L40S", memoryInGb=48),
            inventory_row(),
        ]
    )

    parsed = parse_runpod_inventory(inventory)
    matches = matching_runpod_gpus(manifest, parsed)

    assert len(parsed) == 2
    assert [match.gpu_id for match in matches] == ["NVIDIA L40"]
    assert require_runpod_gpu(manifest, inventory).memory_gb == 48


@pytest.mark.parametrize(
    "overrides",
    [
        {"memoryInGb": 47},
        {"available": False},
        {"secureCloud": False},
        {"gpuId": "NVIDIA A40", "displayName": "A40"},
    ],
)
def test_runpod_inventory_rejects_ineligible_hardware(overrides: dict[str, object]) -> None:
    with pytest.raises(GateError, match="unavailable"):
        require_runpod_gpu(load_manifest(), [inventory_row(**overrides)])


def test_runpod_inventory_rejects_malformed_or_duplicate_rows() -> None:
    with pytest.raises(GateError, match="invalid type"):
        parse_runpod_inventory([inventory_row(memoryInGb="48")])
    with pytest.raises(GateError, match="duplicate"):
        parse_runpod_inventory([inventory_row(), inventory_row()])


def test_huggingface_metadata_verifies_revision_parameters_and_bytes() -> None:
    requested_urls: list[str] = []

    def fetch(url: str) -> dict[str, object]:
        requested_urls.append(url)
        return huggingface_metadata()

    observed = verify_huggingface_metadata(load_manifest(), fetcher=fetch)

    assert requested_urls == [
        "https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-7B-Instruct/"
        f"revision/{MODEL_REVISION}?blobs=true"
    ]
    assert observed["parameter_count"] == MODEL_PARAMETER_COUNT
    assert observed["safetensors_bytes"] == MODEL_SAFETENSORS_BYTES
    assert observed["shard_count"] == 4


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (huggingface_metadata(sha="different"), "different model revision"),
        (
            huggingface_metadata(
                safetensors={
                    "parameters": {"BF16": 3_000_000_000},
                    "total": 3_000_000_000,
                }
            ),
            "parameter metadata",
        ),
        (
            huggingface_metadata(
                siblings=[
                    {
                        "rfilename": "model-00001-of-00004.safetensors",
                        "size": MODEL_SAFETENSORS_BYTES,
                    }
                ]
            ),
            "four unique",
        ),
    ],
)
def test_huggingface_metadata_fails_closed(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GateError, match=message):
        verify_huggingface_metadata(load_manifest(), fetcher=lambda _url: payload)


def test_result_digest_is_canonical_and_ignores_only_self_digest() -> None:
    left = {"z": [1, 2], "a": "same"}
    right = {"a": "same", "z": [1, 2], "digest": "ignored"}

    assert canonical_json(left) == b'{"a":"same","z":[1,2]}'
    assert result_digest(left) == result_digest(right)
    with pytest.raises(GateError, match="canonical JSON"):
        canonical_json({"not_finite": float("nan")})


def test_exact_fresh_screen_and_teardown_receipt_authorize_pilot() -> None:
    screen = eligible_screen()
    receipt = receipt_for(screen)

    authorization = verify_pilot_authorization(
        load_manifest(),
        screen,
        receipt,
        now=NOW,
    )

    assert authorization["profile_id"] == PROFILE_ID
    assert authorization["model_id"] == MODEL_ID
    assert authorization["model_revision"] == MODEL_REVISION
    assert authorization["screen_result_digest"] == result_digest(screen)
    assert authorization["provider_handle"] == "runpod://pods/pod-123"
    assert authorization["digest"].startswith("sha256:")


@pytest.mark.parametrize(
    ("result_overrides", "receipt_overrides", "message"),
    [
        ({"profile_id": "other-profile@1"}, {}, "result profile_id"),
        ({"model_revision": "main"}, {}, "result model_revision"),
        ({"eligible": False}, {}, "did not authorize"),
        ({"policy_mutation_detected": True}, {}, "policy weights"),
        ({"optimizer_state_restored": False}, {}, "optimizer-state"),
        ({"test_split_accessed": True}, {}, "test-split"),
        ({"peak_reserved_vram_fraction": 0.86}, {}, "outside"),
        ({"completed_baseline_examples": 12}, {}, "full baseline"),
        (
            {
                "per_level_checkpoint_rates": {
                    "0": 0.875,
                    "1": 0.875,
                    "2": 0.625,
                    "3": 0.875,
                }
            },
            {},
            "level 2 checkpoint rate",
        ),
        ({"informative_groups": 1}, {}, "informative_groups"),
        ({"solved_siblings": 1}, {}, "solved_siblings"),
        ({"failed_siblings": 1}, {}, "failed_siblings"),
        ({}, {"teardown_confirmed": False}, "teardown"),
        ({}, {"result_digest": "sha256:" + "0" * 64}, "exact screen result"),
        ({}, {"profile_id": "other-profile@1"}, "receipt profile_id"),
        ({}, {"model_revision": "main"}, "receipt model_revision"),
        (
            {},
            {
                "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
                "model_revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
            },
            "receipt model_id",
        ),
        ({}, {"completed_at": "2026-07-20T11:00:00Z"}, "too old"),
    ],
)
def test_pilot_authorization_fails_closed(
    result_overrides: dict[str, object],
    receipt_overrides: dict[str, object],
    message: str,
) -> None:
    screen = eligible_screen(**result_overrides)
    receipt = receipt_for(screen, **receipt_overrides)

    with pytest.raises(GateError, match=message):
        verify_pilot_authorization(load_manifest(), screen, receipt, now=NOW)


def test_failed_or_missing_gate_cannot_authorize() -> None:
    failed = dict.fromkeys(REQUIRED_GATE_RESULTS, True)
    failed["peak_reserved_vram_within_limit"] = False
    screen = eligible_screen(gate_results=failed)
    with pytest.raises(GateError, match="failed gates"):
        verify_pilot_authorization(load_manifest(), screen, receipt_for(screen), now=NOW)

    missing = copy.deepcopy(failed)
    missing.pop("peak_reserved_vram_within_limit")
    screen = eligible_screen(gate_results=missing)
    with pytest.raises(GateError, match="gate set"):
        verify_pilot_authorization(load_manifest(), screen, receipt_for(screen), now=NOW)


def test_future_receipt_cannot_authorize() -> None:
    screen = eligible_screen()
    future = (NOW + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    with pytest.raises(GateError, match="future"):
        verify_pilot_authorization(
            load_manifest(),
            screen,
            receipt_for(screen, completed_at=future),
            now=NOW,
        )
