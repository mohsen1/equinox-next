from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from research.runpod.larger_model_gate import (
    CLEANUP_COST_RESERVE_SECONDS,
    DEFAULT_MANIFEST_PATH,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_SAFETENSORS_BYTES,
    PROFILE_ID,
    REQUIRED_GATE_RESULTS,
    SCREEN_WORKLOAD,
    SCREEN_WORKLOAD_REVISION,
    SNAPSHOT_FILES,
    GateError,
    canonical_json,
    expected_snapshot_digest,
    lifetime_cost_bound,
    load_manifest,
    matching_runpod_gpus,
    parse_runpod_inventory,
    require_runpod_gpu,
    result_digest,
    verify_huggingface_metadata,
    verify_pilot_authorization,
    verify_volume_readiness_receipt,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
EXPECTED_SNAPSHOT_HASHES = {
    "config.json": "c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28",
    "generation_config.json": "1a628a5775bc69cde01c6749a531150ca4d3189652c618a174f7077923acf3b1",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00004.safetensors": (
        "0b6f069918b07c064cbba8ae4f00f529aa9bbf84b7cdfcb7fc2694a40f6aa8ef"
    ),
    "model-00002-of-00004.safetensors": (
        "c3d46733e7aa054ea7b063fbccd0a5a08446e7bd1814bef26936c5aa1331da62"
    ),
    "model-00003-of-00004.safetensors": (
        "9fe45dacee087385b3d2d6dd27a7413a8a56d95f145772facc148fa86fc73446"
    ),
    "model-00004-of-00004.safetensors": (
        "5aa6e5cbe642377fd441fb4e60e83cca96b2bcd9820e245b9ea06d94653f17f2"
    ),
    "model.safetensors.index.json": (
        "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb"
    ),
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": ("959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4"),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


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
            *[
                {
                    "rfilename": name,
                    "size": metadata["size_bytes"],
                    "lfs": {"sha256": metadata["sha256"]},
                }
                for name, metadata in SNAPSHOT_FILES.items()
                if name.endswith(".safetensors")
            ],
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
        "environment_revision": "repository-repair-simulator@6",
        "action_protocol_revision": "repository-repair-json-tools@5",
        "branch_width": 4,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 1_536,
        "capacity_smoke_completed": True,
        "gradient_checkpointing_enabled": True,
        "pinned_snapshot_digest": expected_snapshot_digest(load_manifest()),
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
        "predicted_final_evaluation_seconds": 1_200.0,
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
        "resource_profile": {"network_volume_id": "network-volume-123"},
    }
    receipt.update(overrides)
    return receipt


def volume_receipt(**overrides: object) -> dict[str, object]:
    manifest = load_manifest()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "network_volume_id": "network-volume-123",
        "network_volume_data_center_id": "EU-RO-1",
        "network_volume_size_gb": 50,
        "snapshot_digest": expected_snapshot_digest(manifest),
        "dependencies": manifest["runtime"]["dependencies"],
        "prepared_at": "2026-07-29T11:00:00Z",
        "ready": True,
    }
    receipt.update(overrides)
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    return receipt


def write_manifest(tmp_path: Path, transform: callable) -> Path:
    payload = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    transform(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repository_manifest_is_the_exact_bounded_profile() -> None:
    manifest = load_manifest()

    assert {
        key: manifest["model"][key]
        for key in ("id", "revision", "parameter_count", "safetensors_bytes", "dtype")
    } == {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "safetensors_bytes": MODEL_SAFETENSORS_BYTES,
        "dtype": "bfloat16",
    }
    assert {
        name: metadata["sha256"] for name, metadata in manifest["model"]["snapshot_files"].items()
    } == EXPECTED_SNAPSHOT_HASHES
    assert manifest["model"]["snapshot_files"] == SNAPSHOT_FILES
    assert manifest["hardware"]["gpu_id"] == "NVIDIA L40"
    assert manifest["hardware"]["minimum_gpu_memory_gb"] == 48
    assert manifest["hardware"]["minimum_cuda_memory_bytes"] == 47_000_000_000
    assert manifest["hardware"]["maximum_peak_reserved_vram_fraction"] == 0.85
    assert manifest["interface"] == {
        "environment_revision": "repository-repair-simulator@6",
        "action_protocol_revision": "repository-repair-json-tools@5",
    }
    assert manifest["screen"]["validation_examples"] == 8
    assert manifest["screen"]["baseline_examples_per_level"] == 8
    assert manifest["screen"]["training_microbatch_size"] == 1
    assert manifest["screen"]["maximum_input_tokens"] == 1_536
    assert manifest["screen"]["thresholds"]["maximum_predicted_final_evaluation_seconds"] == 1_440
    assert manifest["pilot"] == {
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
    }
    assert manifest["artifact_readiness"] == {
        "cache_directory": "/workspace/equinox-state/huggingface",
        "require_complete_pinned_snapshot_before_screen": True,
        "require_offline_mode_after_readiness": True,
        "allow_network_model_download_during_screen": False,
    }
    assert lifetime_cost_bound(
        manifest["screen_limits"]["maximum_hourly_cost_usd"],
        manifest["screen_limits"]["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS,
    ) == Decimal("0.75")
    assert lifetime_cost_bound(
        manifest["pilot_limits"]["maximum_hourly_cost_usd"],
        manifest["pilot_limits"]["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS,
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
        (
            lambda value: value["model"]["snapshot_files"]["config.json"].update(sha256="0" * 64),
            "manifest.model.snapshot_files.config.json.sha256",
        ),
        (
            lambda value: value["interface"].update(environment_revision="unreviewed@1"),
            "manifest.interface.environment_revision",
        ),
        (
            lambda value: value["pilot"].update(maximum_input_tokens=4_096),
            "manifest.pilot.maximum_input_tokens",
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
                        "size": SNAPSHOT_FILES["model-00001-of-00004.safetensors"]["size_bytes"],
                        "lfs": {
                            "sha256": SNAPSHOT_FILES["model-00001-of-00004.safetensors"]["sha256"]
                        },
                    }
                ]
            ),
            "four unique",
        ),
        (
            huggingface_metadata(
                siblings=[
                    {
                        **sibling,
                        "lfs": {"sha256": "0" * 64},
                    }
                    if sibling.get("rfilename") == "model-00001-of-00004.safetensors"
                    else sibling
                    for sibling in huggingface_metadata()["siblings"]
                ]
            ),
            "no valid size",
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


def test_exact_volume_readiness_receipt_matches_current_provider_volume() -> None:
    receipt = volume_receipt()

    evidence = verify_volume_readiness_receipt(
        load_manifest(),
        receipt,
        {
            "networkVolume": {
                "id": "network-volume-123",
                "dataCenterId": "EU-RO-1",
                "size": 50,
            }
        },
        now=NOW,
    )

    assert evidence == {
        "profile_id": PROFILE_ID,
        "network_volume_id": "network-volume-123",
        "network_volume_data_center_id": "EU-RO-1",
        "network_volume_size_gb": 50,
        "snapshot_digest": expected_snapshot_digest(load_manifest()),
        "receipt_digest": receipt["receipt_digest"],
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"network_volume_id": 123}, "identity or size"),
        ({"network_volume_id": ""}, "identity or size"),
        ({"network_volume_data_center_id": None}, "identity or size"),
        ({"network_volume_size_gb": "50"}, "identity or size"),
        ({"network_volume_size_gb": True}, "identity or size"),
        ({"network_volume_size_gb": 49}, "identity or size"),
        ({"dependencies": []}, "dependencies"),
        ({"ready": 1}, "ready"),
        ({"prepared_at": 123}, "prepared_at"),
    ],
)
def test_malformed_volume_readiness_receipt_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GateError, match=message):
        verify_volume_readiness_receipt(
            load_manifest(),
            volume_receipt(**overrides),
            {
                "id": "network-volume-123",
                "dataCenterId": "EU-RO-1",
                "size": 50,
            },
            now=NOW,
        )


@pytest.mark.parametrize(
    "provider_volume",
    [
        [],
        {"networkVolume": []},
        {"id": 123, "dataCenterId": "EU-RO-1", "size": 50},
        {"id": "network-volume-123", "dataCenterId": 123, "size": 50},
        {"id": "network-volume-123", "dataCenterId": "EU-RO-1", "size": "50"},
        {"id": "another-volume", "dataCenterId": "EU-RO-1", "size": 50},
    ],
)
def test_malformed_or_changed_provider_volume_fails_closed(provider_volume: object) -> None:
    with pytest.raises(GateError, match="network volume"):
        verify_volume_readiness_receipt(
            load_manifest(),
            volume_receipt(),
            provider_volume,
            now=NOW,
        )


def test_volume_readiness_receipt_requires_valid_digest_and_freshness() -> None:
    malformed_digest = volume_receipt()
    malformed_digest["receipt_digest"] = 123
    with pytest.raises(GateError, match="digest"):
        verify_volume_readiness_receipt(
            load_manifest(),
            malformed_digest,
            {"id": "network-volume-123", "dataCenterId": "EU-RO-1", "size": 50},
            now=NOW,
        )

    with pytest.raises(GateError, match="not fresh"):
        verify_volume_readiness_receipt(
            load_manifest(),
            volume_receipt(prepared_at="2026-07-21T11:59:59Z"),
            {"id": "network-volume-123", "dataCenterId": "EU-RO-1", "size": 50},
            now=NOW,
        )


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
    assert authorization["pinned_snapshot_digest"] == expected_snapshot_digest(load_manifest())
    assert authorization["network_volume_id"] == "network-volume-123"
    assert authorization["provider_handle"] == "runpod://pods/pod-123"
    assert authorization["digest"].startswith("sha256:")


@pytest.mark.parametrize(
    ("result_overrides", "receipt_overrides", "message"),
    [
        ({"profile_id": "other-profile@1"}, {}, "result profile_id"),
        ({"model_revision": "main"}, {}, "result model_revision"),
        ({"environment_revision": "simulator@old"}, {}, "environment_revision"),
        ({"action_protocol_revision": "tools@old"}, {}, "action_protocol_revision"),
        ({"branch_width": 1}, {}, "branch_width"),
        ({"training_microbatch_size": 2}, {}, "training_microbatch_size"),
        ({"maximum_input_tokens": 4_096}, {}, "maximum_input_tokens"),
        ({"capacity_smoke_completed": False}, {}, "capacity_smoke_completed"),
        ({"gradient_checkpointing_enabled": False}, {}, "gradient_checkpointing_enabled"),
        ({"pinned_snapshot_digest": "sha256:" + "0" * 64}, {}, "pinned_snapshot_digest"),
        ({"eligible": False}, {}, "did not authorize"),
        ({"policy_mutation_detected": True}, {}, "policy weights"),
        ({"optimizer_state_restored": False}, {}, "optimizer-state"),
        ({"test_split_accessed": True}, {}, "test-split"),
        ({"peak_reserved_vram_fraction": 0.86}, {}, "outside"),
        ({"predicted_final_evaluation_seconds": 0.0}, {}, "positive pilot runtime"),
        ({"predicted_final_evaluation_seconds": 1_441.0}, {}, "outside"),
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
        ({}, {"resource_profile": None}, "network volume"),
        ({}, {"resource_profile": {}}, "network volume"),
        (
            {},
            {"resource_profile": {"network_volume_id": 123}},
            "network volume",
        ),
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
