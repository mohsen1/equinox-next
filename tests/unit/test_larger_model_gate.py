from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from research.runpod import larger_model_gate as gate_module
from research.runpod.larger_model_gate import (
    BUNDLE_HANDOFF_REVISION,
    CLEANUP_COST_RESERVE_SECONDS,
    DEFAULT_MANIFEST_PATH,
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_SAFETENSORS_BYTES,
    PREDECESSOR_MANIFEST_COMMIT,
    PREDECESSOR_MANIFEST_DIGEST,
    PREDECESSOR_PROFILE_ID,
    PREDECESSOR_SNAPSHOT_DIGEST,
    PROFILE_ID,
    REQUIRED_GATE_RESULTS,
    SCREEN_WORKLOAD,
    SCREEN_WORKLOAD_REVISION,
    SNAPSHOT_FILES,
    SOURCE_CONTRACT_SHA256,
    VOLUME_READINESS_ATTESTATION_REVISION,
    CUDAHardware,
    GateError,
    build_attested_volume_readiness_receipt,
    build_live_stage_activation,
    canonical_json,
    expected_cuda_version,
    expected_snapshot_digest,
    expected_source_contract_digest,
    lifetime_cost_bound,
    load_manifest,
    matching_runpod_gpus,
    materialize_verified_snapshot,
    parse_runpod_inventory,
    require_cuda_hardware,
    require_runpod_gpu,
    result_digest,
    retention_checkpoint_evidence_digest,
    retention_checkpoint_storage_evidence_digest,
    verify_dependency_import_smoke,
    verify_huggingface_metadata,
    verify_local_snapshot,
    verify_pilot_authorization,
    verify_predecessor_readiness_bridge,
    verify_registry_image_index,
    verify_retention_checkpoint_evidence,
    verify_source_contract,
    verify_volume_readiness_receipt,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
IMAGE_TAG = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
IMAGE_DIGEST = "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
IMAGE_INDEX_DIGEST = "sha256:" + "1" * 64
WORKLOAD_BUNDLE_DIGEST = "sha256:" + "2" * 64
BUNDLE_STAGE_RECEIPT_DIGEST = "sha256:" + "3" * 64
BOOTSTRAP_SOURCE_DIGEST = "sha256:" + "4" * 64
DEPENDENCY_TREE_DIGEST = "sha256:" + "9" * 64
CODE_TREE_DIGEST = "sha256:" + "a" * 64
SOURCE_HEAD_COMMIT = "5" * 40
NETWORK_VOLUME_ID = "network-volume-123"
NETWORK_VOLUME_DATA_CENTER_ID = "EU-RO-1"
NETWORK_VOLUME_SIZE_GB = 50
WORKLOAD_BUNDLE_SIZE_BYTES = 80_000
WORKLOAD_BUNDLE_PATH = (
    f"/workspace/equinox-state/workload-bundles/{PROFILE_ID}/"
    f"{WORKLOAD_BUNDLE_DIGEST.removeprefix('sha256:')}.tar.xz"
)
EXPECTED_SNAPSHOT_HASHES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "3c090be37f829adc1e4cdb78733667437732541470430ab2cd785d7f0d460077",
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
        "displayName": "H100 SXM",
        "gpuId": "NVIDIA H100 80GB HBM3",
        "memoryInGb": 80,
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
    manifest = load_manifest()
    readiness = volume_receipt()
    retention = retention_evidence(readiness)
    activation = live_stage_activation(readiness, retention)
    result: dict[str, object] = {
        "workload": SCREEN_WORKLOAD,
        "workload_revision": SCREEN_WORKLOAD_REVISION,
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "optimization_seed": 137,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "source_head_commit": SOURCE_HEAD_COMMIT,
        "live_stage_activation_revision": manifest["screen"]["live_stage_activation_revision"],
        "live_stage_activation_digest": activation["activation_digest"],
        "volume_readiness_receipt_digest": readiness["receipt_digest"],
        "torch_retention_evidence_digest": retention["evidence_digest"],
        "checkpoint_authentication_mechanism_digest": retention[
            "checkpoint_authentication_mechanism_digest"
        ],
        "dependency_quarantine_revision": readiness["dependency_quarantine_revision"],
        "dependency_lock_digest": readiness["dependency_lock_digest"],
        "dependency_quarantine_evidence_digest": readiness["dependency_quarantine_evidence_digest"],
        "dependency_private_tree_digest": readiness["dependency_private_tree_digest"],
        "code_materialization_revision": readiness["code_materialization_revision"],
        "code_materialization_evidence_digest": readiness["code_materialization_evidence_digest"],
        "code_private_tree_digest": readiness["code_private_tree_digest"],
        "volume_readiness_receipt": readiness,
        "retention_checkpoint_evidence": retention,
        "preparation_evidence_complete": True,
        "environment_revision": manifest["interface"]["environment_revision"],
        "action_protocol_revision": manifest["interface"]["action_protocol_revision"],
        "terminal_submission_contract": manifest["interface"]["terminal_submission_contract"],
        "screen_levels": manifest["screen"]["admission_levels"],
        "shared_prefix_checkpoint_strategy": manifest["screen"][
            "shared_prefix_checkpoint_strategy"
        ],
        "branch_width": 4,
        "training_microbatch_size": 1,
        "maximum_input_tokens": 2_048,
        "capacity_smoke_sequence_tokens": 2_240,
        "capacity_smoke_completed": True,
        "gradient_checkpointing_enabled": True,
        "determinism": manifest["pilot"]["determinism"],
        "pinned_snapshot_digest": expected_snapshot_digest(manifest),
        "screen_completed": True,
        "eligible": True,
        "training_started": False,
        "persistent_policy_updates": 0,
        "policy_mutation_enabled": False,
        "policy_mutation_detected": False,
        "optimizer_state_restored": True,
        "test_split_accessed": False,
        "test_examples_accessed": 0,
        "baseline_checkpoint_rate": 0.9,
        "action_protocol_validity": 1.0,
        "branch_checkpoint_rate": 0.875,
        "informative_group_rate": 0.25,
        "solved_sibling_rate": 0.25,
        "baseline_exact_rate": 0.2,
        "peak_reserved_vram_fraction": 0.8,
        "predicted_final_evaluation_seconds": 1_200.0,
        "branch_groups": 8,
        "completed_baseline_examples": 8,
        "expected_baseline_examples": 8,
        "per_level_checkpoint_rates": {"0": 0.875},
        "informative_groups": 2,
        "solved_siblings": 8,
        "failed_siblings": 24,
        "gate_results": dict.fromkeys(REQUIRED_GATE_RESULTS, True),
    }
    result.update(overrides)
    return result


def receipt_for(result: dict[str, object], **overrides: object) -> dict[str, object]:
    manifest = load_manifest()
    readiness = result["volume_readiness_receipt"]
    retention = result["retention_checkpoint_evidence"]
    activation = live_stage_activation(readiness, retention)
    receipt: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/pod-123",
        "provider_cli_version": "1.14.0",
        "teardown_confirmed": True,
        "started_at": "2026-07-29T10:00:00Z",
        "completed_at": "2026-07-29T11:00:00Z",
        "resource_profile": {
            "profile_id": PROFILE_ID,
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "image": IMAGE_TAG,
            "image_digest": IMAGE_DIGEST,
            "manifest_digest": ("sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest()),
            "source_contract_digest": expected_source_contract_digest(manifest),
            "network_volume_id": NETWORK_VOLUME_ID,
            "network_volume_data_center_id": NETWORK_VOLUME_DATA_CENTER_ID,
            "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
            "source_head_commit": SOURCE_HEAD_COMMIT,
            "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
            "workload_bundle_digest": WORKLOAD_BUNDLE_DIGEST,
            "workload_bundle_size_bytes": WORKLOAD_BUNDLE_SIZE_BYTES,
            "workload_bundle_compression": "xz",
            "workload_bundle_path": WORKLOAD_BUNDLE_PATH,
            "bundle_stage_receipt_digest": BUNDLE_STAGE_RECEIPT_DIGEST,
            "bootstrap_source_digest": BOOTSTRAP_SOURCE_DIGEST,
            "volume_readiness_receipt_digest": readiness["receipt_digest"],
            "torch_retention_evidence_digest": retention["evidence_digest"],
            "dependency_quarantine_revision": readiness["dependency_quarantine_revision"],
            "dependency_lock_digest": readiness["dependency_lock_digest"],
            "dependency_quarantine_evidence_digest": readiness[
                "dependency_quarantine_evidence_digest"
            ],
            "dependency_private_tree_digest": readiness["dependency_private_tree_digest"],
            "code_materialization_revision": readiness["code_materialization_revision"],
            "code_materialization_evidence_digest": readiness[
                "code_materialization_evidence_digest"
            ],
            "code_private_tree_digest": readiness["code_private_tree_digest"],
            "live_stage_activation_revision": activation["revision"],
            "live_stage_activation_digest": activation["activation_digest"],
        },
        "workload": {
            "id": SCREEN_WORKLOAD,
            "revision": SCREEN_WORKLOAD_REVISION,
            "algorithm": None,
            "static_branch_width": 4,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        "result": result,
    }
    receipt.update(overrides)
    return receipt


def volume_receipt(**overrides: object) -> dict[str, object]:
    manifest = load_manifest()
    dependency_evidence = private_dependency_evidence()
    code_evidence = private_code_evidence()
    receipt: dict[str, object] = {
        "schema_version": 3,
        "attestation_revision": VOLUME_READINESS_ATTESTATION_REVISION,
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "network_volume_id": NETWORK_VOLUME_ID,
        "network_volume_data_center_id": NETWORK_VOLUME_DATA_CENTER_ID,
        "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
        "snapshot_digest": expected_snapshot_digest(manifest),
        "dependencies": manifest["runtime"]["dependencies"],
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": expected_cuda_version(manifest),
        "gpu_name": manifest["hardware"]["gpu_id"],
        "gpu_total_memory_bytes": manifest["hardware"]["minimum_cuda_memory_bytes"],
        "bf16_supported": True,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_lock_digest": manifest["materialization"]["dependency_lock"]["digest"],
        "dependency_quarantine_evidence": dependency_evidence,
        "dependency_quarantine_evidence_digest": dependency_evidence["evidence_digest"],
        "dependency_private_tree_digest": dependency_evidence["private_tree_digest"],
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence": code_evidence,
        "code_materialization_evidence_digest": code_evidence["evidence_digest"],
        "code_private_tree_digest": code_evidence["private_tree_digest"],
        "prepared_at": "2026-07-29T11:00:00Z",
        "ready": True,
    }
    receipt.update(overrides)
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    return receipt


def private_dependency_evidence() -> dict[str, object]:
    manifest = load_manifest()
    versions: dict[str, str] = {}
    for line in (
        (Path(gate_module.__file__).with_name("larger-model-dependencies.lock"))
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if "==" not in line or line.startswith((" ", "#")):
            continue
        name, remainder = line.split("==", 1)
        versions[name.replace("_", "-").replace(".", "-").lower()] = remainder.split()[0]
    distributions = [
        {
            "name": name,
            "version": version,
            "record_path": f"{name}.dist-info/RECORD",
            "record_digest": "sha256:" + "b" * 64,
            "file_count": 1,
            "files_digest": "sha256:" + "c" * 64,
        }
        for name, version in sorted(versions.items())
    ]
    evidence: dict[str, object] = {
        "schema_version": 2,
        "revision": manifest["materialization"]["dependency_lock"]["revision"],
        "profile_id": PROFILE_ID,
        "lock_path": f"{WORKLOAD_BUNDLE_PATH}::larger-model-dependencies.lock",
        "lock_digest": manifest["materialization"]["dependency_lock"]["digest"],
        "python_version": "3.12",
        "platform_tag": "manylinux_2_28_x86_64",
        "installer_revision": "isolated-pip-binary-hash-lock@1",
        "index_url": "https://pypi.org/simple",
        "private_root": (
            "/tmp/equinox-quarantine/dependencies/" + DEPENDENCY_TREE_DIGEST.removeprefix("sha256:")
        ),
        "install_tree_digest": DEPENDENCY_TREE_DIGEST,
        "private_tree_digest": DEPENDENCY_TREE_DIGEST,
        "record_closure_digest": (
            "sha256:" + hashlib.sha256(canonical_json(distributions)).hexdigest()
        ),
        "distributions": distributions,
        "installed_file_count": len(distributions),
        "installed_bytes": 1,
        "ready": True,
    }
    evidence["evidence_digest"] = gate_module.materialization_evidence_digest(evidence)
    return evidence


def private_code_evidence() -> dict[str, object]:
    manifest = load_manifest()
    files = [
        {
            "path": name,
            "size_bytes": (
                manifest["materialization"]["dependency_lock"]["size_bytes"]
                if name == "larger-model-dependencies.lock"
                else 1
            ),
            "sha256": f"sha256:{digest}",
        }
        for name, digest in sorted(manifest["source_contract"]["files"].items())
    ]
    tree_digest = "sha256:" + hashlib.sha256(canonical_json(files)).hexdigest()
    evidence: dict[str, object] = {
        "schema_version": 1,
        "revision": manifest["materialization"]["code"]["revision"],
        "profile_id": PROFILE_ID,
        "bundle_digest": WORKLOAD_BUNDLE_DIGEST,
        "bundle_size_bytes": WORKLOAD_BUNDLE_SIZE_BYTES,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "source_root": WORKLOAD_BUNDLE_PATH,
        "private_root": ("/tmp/equinox-quarantine/code/" + tree_digest.removeprefix("sha256:")),
        "source_tree_digest": tree_digest,
        "private_tree_digest": tree_digest,
        "files": files,
        "installed_file_count": len(files),
        "installed_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "ready": True,
    }
    evidence["evidence_digest"] = gate_module.materialization_evidence_digest(evidence)
    return evidence


def retention_evidence(readiness: dict[str, object]) -> dict[str, object]:
    manifest = load_manifest()
    evidence: dict[str, object] = {
        "schema_version": 3,
        "evidence_revision": "real-adamw-persist-restore-advance@7",
        "status": "passed",
        "test_id": (
            "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        ),
        "profile_id": PROFILE_ID,
        "head_commit": SOURCE_HEAD_COMMIT,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "workload_bundle_digest": WORKLOAD_BUNDLE_DIGEST,
        "workload_bundle_size_bytes": WORKLOAD_BUNDLE_SIZE_BYTES,
        "workload_bundle_path": WORKLOAD_BUNDLE_PATH,
        "bundle_stage_receipt_digest": BUNDLE_STAGE_RECEIPT_DIGEST,
        "bootstrap_source_digest": BOOTSTRAP_SOURCE_DIGEST,
        "volume_readiness_receipt_digest": readiness["receipt_digest"],
        "dependency_quarantine_revision": readiness["dependency_quarantine_revision"],
        "dependency_lock_digest": readiness["dependency_lock_digest"],
        "dependency_quarantine_evidence_digest": readiness["dependency_quarantine_evidence_digest"],
        "dependency_private_tree_digest": readiness["dependency_private_tree_digest"],
        "code_materialization_revision": readiness["code_materialization_revision"],
        "code_materialization_evidence_digest": readiness["code_materialization_evidence_digest"],
        "code_private_tree_digest": readiness["code_private_tree_digest"],
        "network_volume_id": NETWORK_VOLUME_ID,
        "network_volume_data_center_id": NETWORK_VOLUME_DATA_CENTER_ID,
        "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": expected_cuda_version(manifest),
        "cuda_available": True,
        "gpu_name": manifest["hardware"]["gpu_id"],
        "gpu_total_memory_bytes": manifest["hardware"]["minimum_cuda_memory_bytes"],
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
        "checkpoint_sha256": "sha256:" + "6" * 64,
        "checkpoint_size_bytes": 1024,
        "restored_weight_before_resume_step": [0.99],
        "advanced_weight_after_resume_step": [0.98],
        "effective_policy_update_count_before_resume_step": 1,
        "effective_policy_update_count_after_resume_step": 2,
        "retained_observation_after_resume_step": {"exact_rate": 0.75},
        "optimizer_state_entries_after_resume_step": 1,
        "optimizer_state_digest_before_persist": "sha256:" + "7" * 64,
        "optimizer_state_digest_after_restore": "sha256:" + "7" * 64,
        "optimizer_state_digest_after_resume_step": "sha256:" + "8" * 64,
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
        "checkpoint_storage_root": "/workspace/equinox-runs/proof-test",
        "checkpoint_storage_root_device": 42,
        "checkpoint_storage_checkpoint_device": 42,
        "checkpoint_source_device": 42,
        "checkpoint_storage_root_inode": 84,
        "checkpoint_inode_before_reopen": 85,
        "checkpoint_inode_after_reopen": 85,
        "checkpoint_persist_process_pid": 100,
        "checkpoint_resume_process_pid": 101,
        "checkpoint_resume_parent_process_pid": 100,
        "checkpoint_reopened_after_fsync": True,
        "checkpoint_authentication_revision": "launch-bound-checkpoint-manifest@1",
        "checkpoint_authentication_mechanism_digest": (
            gate_module.checkpoint_authentication_mechanism_digest()
        ),
        "checkpoint_generation": 1,
        "checkpoint_manifest_digest": "sha256:" + "9" * 64,
        "checkpoint_authenticated_private_resume": True,
    }
    evidence["checkpoint_storage_evidence_digest"] = retention_checkpoint_storage_evidence_digest(
        evidence
    )
    evidence["evidence_digest"] = retention_checkpoint_evidence_digest(evidence)
    return evidence


def live_stage_activation(
    readiness: dict[str, object],
    retention: dict[str, object],
) -> dict[str, object]:
    return build_live_stage_activation(
        load_manifest(),
        head_commit=SOURCE_HEAD_COMMIT,
        workload_bundle_digest=WORKLOAD_BUNDLE_DIGEST,
        workload_bundle_size_bytes=WORKLOAD_BUNDLE_SIZE_BYTES,
        workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        bundle_stage_receipt_digest=BUNDLE_STAGE_RECEIPT_DIGEST,
        bootstrap_source_digest=BOOTSTRAP_SOURCE_DIGEST,
        volume_readiness_receipt_digest=str(readiness["receipt_digest"]),
        torch_retention_evidence_digest=str(retention["evidence_digest"]),
        dependency_lock_digest=str(readiness["dependency_lock_digest"]),
        dependency_quarantine_evidence_digest=str(
            readiness["dependency_quarantine_evidence_digest"]
        ),
        dependency_private_tree_digest=str(readiness["dependency_private_tree_digest"]),
        code_materialization_evidence_digest=str(readiness["code_materialization_evidence_digest"]),
        code_private_tree_digest=str(readiness["code_private_tree_digest"]),
        network_volume_id=NETWORK_VOLUME_ID,
        data_center_id=NETWORK_VOLUME_DATA_CENTER_ID,
        volume_size_gb=NETWORK_VOLUME_SIZE_GB,
    )


def registry_image_index(*descriptors: object, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "digest": IMAGE_INDEX_DIGEST,
        "manifests": list(descriptors)
        or [
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": IMAGE_DIGEST,
                "platform": {"architecture": "amd64", "os": "linux"},
            },
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": "sha256:" + "2" * 64,
                "platform": {"architecture": "unknown", "os": "unknown"},
            },
        ],
    }
    payload.update(overrides)
    return payload


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
    assert manifest["runtime"]["image_digest"] == (
        "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
    )
    assert manifest["runtime"]["torch_version"] == "2.8.0+cu128"
    assert manifest["hardware"]["gpu_id"] == "NVIDIA H100 80GB HBM3"
    assert manifest["hardware"]["gpu_display_name"] == "H100 SXM"
    assert manifest["hardware"]["minimum_gpu_memory_gb"] == 80
    assert manifest["hardware"]["minimum_cuda_memory_bytes"] == 78_000_000_000
    assert manifest["hardware"]["maximum_peak_reserved_vram_fraction"] == 0.85
    assert manifest["interface"] == {
        "environment_revision": "repository-repair-simulator@9",
        "action_protocol_revision": "repository-repair-json-tools@8",
        "terminal_submission_contract": "accepted-passing-test-or-finish@1",
    }
    assert manifest["source_contract"] == {
        "algorithm": "sha256",
        "files": SOURCE_CONTRACT_SHA256,
    }
    assert manifest["screen"]["validation_examples"] == 8
    assert manifest["screen"]["admission_levels"] == [0]
    assert manifest["screen"]["shared_prefix_checkpoint_strategy"] == "repository_root_observed@1"
    assert manifest["screen"]["minimum_completed_baseline_examples"] == 8
    assert manifest["screen"]["baseline_examples_per_level"] == 8
    assert manifest["screen"]["training_microbatch_size"] == 1
    assert manifest["screen"]["maximum_input_tokens"] == 2_048
    assert manifest["screen"]["live_stage_activation_revision"] == (
        "authenticated-proxy-stage-activation@2"
    )
    assert manifest["materialization"]["dependency_lock"]["digest"] == (
        "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
    )
    assert manifest["screen"]["thresholds"]["minimum_baseline_exact_rate"] == 0.0
    assert manifest["screen"]["thresholds"]["maximum_baseline_exact_rate"] == 0.75
    assert {
        key: manifest["screen"]["thresholds"][key]
        for key in (
            "minimum_informative_group_rate",
            "minimum_informative_groups",
            "minimum_solved_siblings",
            "minimum_failed_siblings",
            "minimum_solved_sibling_rate",
            "maximum_solved_sibling_rate",
        )
    } == {
        "minimum_informative_group_rate": 0.1,
        "minimum_informative_groups": 2,
        "minimum_solved_siblings": 2,
        "minimum_failed_siblings": 2,
        "minimum_solved_sibling_rate": 0.05,
        "maximum_solved_sibling_rate": 0.8,
    }
    assert manifest["screen"]["thresholds"]["maximum_predicted_final_evaluation_seconds"] == 1_440
    assert manifest["pilot"] == {
        "workload_revision": "runpod-repository-repair-large-model-pilot@7",
        "objective_id": ("verified-repair-chain-transactional-retention-policy-gradient@19"),
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
    }
    assert manifest["artifact_readiness"] == {
        "cache_directory": "/workspace/equinox-state/huggingface",
        "require_complete_pinned_snapshot_before_screen": True,
        "require_offline_mode_after_readiness": True,
        "allow_network_model_download_during_screen": False,
    }
    assert manifest["provider_safety"] == {
        "maximum_storage_only_hourly_spend_usd": 0.01,
    }
    assert manifest["screen_limits"]["maximum_lifetime_seconds"] == 2_880
    assert manifest["screen_limits"]["conservative_billing_seconds"] == 3_000
    assert manifest["screen_limits"]["maximum_total_cost_usd"] == 3.35
    assert lifetime_cost_bound(
        manifest["screen_limits"]["maximum_hourly_cost_usd"],
        manifest["screen_limits"]["conservative_billing_seconds"],
    ) == Decimal("3.333333333333333333333333333")
    assert lifetime_cost_bound(
        manifest["pilot_limits"]["maximum_hourly_cost_usd"],
        manifest["pilot_limits"]["maximum_lifetime_seconds"] + CLEANUP_COST_RESERVE_SECONDS,
    ) == Decimal("13.0")


def test_registry_image_index_resolves_one_exact_linux_amd64_manifest() -> None:
    evidence = verify_registry_image_index(load_manifest(), registry_image_index())

    assert evidence == {
        "image": IMAGE_TAG,
        "image_digest": IMAGE_DIGEST,
        "index_digest": IMAGE_INDEX_DIGEST,
        "platform": "linux/amd64",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "not valid JSON"),
        ([], "must be an object"),
        (
            registry_image_index(mediaType="application/vnd.oci.image.manifest.v1+json"),
            "index mediaType",
        ),
        (
            registry_image_index(digest="sha256:invalid"),
            "index digest",
        ),
        (
            registry_image_index(
                {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "digest": IMAGE_DIGEST,
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ),
            "not an image manifest",
        ),
        (
            registry_image_index(
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": "sha256:" + "A" * 64,
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ),
            "image digest is invalid",
        ),
        (
            registry_image_index(
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": "sha256:" + "0" * 64,
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ),
            "does not match",
        ),
        (
            registry_image_index(
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": IMAGE_DIGEST,
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": IMAGE_DIGEST,
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
            ),
            "exactly one",
        ),
        (
            registry_image_index(
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": IMAGE_DIGEST,
                    "platform": {
                        "architecture": "amd64",
                        "os": "linux",
                        "variant": "v2",
                    },
                }
            ),
            "variant is unsupported",
        ),
    ],
    ids=(
        "invalid-json",
        "not-object",
        "wrong-index-media-type",
        "invalid-index-digest",
        "wrong-child-media-type",
        "invalid-child-digest",
        "mismatched-child-digest",
        "ambiguous-platform",
        "unsupported-variant",
    ),
)
def test_registry_image_index_parser_fails_closed(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(GateError, match=message):
        verify_registry_image_index(load_manifest(), payload)


@pytest.mark.parametrize("tampered_name", tuple(SOURCE_CONTRACT_SHA256))
def test_source_contract_verifies_exact_sources_and_rejects_tampering(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    manifest = load_manifest()
    repository_sources = DEFAULT_MANIFEST_PATH.parents[1] / "runpod"
    for name in SOURCE_CONTRACT_SHA256:
        shutil.copy2(repository_sources / name, tmp_path / name)

    verification = verify_source_contract(manifest, tmp_path)

    assert verification["files"] == SOURCE_CONTRACT_SHA256
    assert verification["source_contract_digest"] == expected_source_contract_digest(manifest)

    tampered = tmp_path / tampered_name
    tampered.write_bytes(tampered.read_bytes() + b"\n# tampered\n")
    with pytest.raises(GateError, match="failed SHA-256"):
        verify_source_contract(manifest, tmp_path)


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
            lambda value: value["provider_safety"].update(
                maximum_storage_only_hourly_spend_usd=0.02
            ),
            "maximum_storage_only_hourly_spend_usd",
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


def test_runpod_inventory_requires_exact_available_secure_h100() -> None:
    manifest = load_manifest()
    inventory = json.dumps(
        [
            inventory_row(
                gpuId="NVIDIA H100 PCIe",
                displayName="H100 PCIe",
                memoryInGb=80,
            ),
            inventory_row(),
        ]
    )

    parsed = parse_runpod_inventory(inventory)
    matches = matching_runpod_gpus(manifest, parsed)

    assert len(parsed) == 2
    assert [match.gpu_id for match in matches] == ["NVIDIA H100 80GB HBM3"]
    assert require_runpod_gpu(manifest, inventory).memory_gb == 80


@pytest.mark.parametrize(
    "overrides",
    [
        {"memoryInGb": 79},
        {"available": False},
        {"secureCloud": False},
        {"displayName": "H100 PCIe"},
        {"gpuId": "NVIDIA A40", "displayName": "A40"},
    ],
)
def test_runpod_inventory_rejects_ineligible_hardware(overrides: dict[str, object]) -> None:
    with pytest.raises(GateError, match="unavailable"):
        require_runpod_gpu(load_manifest(), [inventory_row(**overrides)])


def test_runpod_inventory_rejects_malformed_or_duplicate_rows() -> None:
    with pytest.raises(GateError, match="invalid type"):
        parse_runpod_inventory([inventory_row(memoryInGb="80")])
    with pytest.raises(GateError, match="duplicate"):
        parse_runpod_inventory([inventory_row(), inventory_row()])


def test_actual_cuda_hardware_requires_exact_identity_byte_floor_and_bf16() -> None:
    class FakeCuda:
        def __init__(
            self,
            *,
            name: str = "NVIDIA H100 80GB HBM3",
            total_memory: int = 78_000_000_000,
            bf16: bool = True,
        ) -> None:
            self.name = name
            self.total_memory = total_memory
            self.bf16 = bf16

        def is_available(self) -> bool:
            return True

        def get_device_name(self, _index: int) -> str:
            return self.name

        def get_device_properties(self, _index: int) -> object:
            return type("Properties", (), {"total_memory": self.total_memory})()

        def is_bf16_supported(self) -> bool:
            return self.bf16

    manifest = load_manifest()
    observed = require_cuda_hardware(
        manifest,
        type("Torch", (), {"__version__": "2.8.0+cu128", "cuda": FakeCuda()})(),
    )
    assert observed.gpu_name == "NVIDIA H100 80GB HBM3"
    assert observed.total_memory_bytes == 78_000_000_000
    assert observed.bf16_supported is True

    for cuda in (
        FakeCuda(name="NVIDIA A40"),
        FakeCuda(total_memory=77_999_999_999),
        FakeCuda(bf16=False),
    ):
        with pytest.raises(GateError, match="does not match"):
            require_cuda_hardware(
                manifest,
                type("Torch", (), {"__version__": "2.8.0+cu128", "cuda": cuda})(),
            )
    with pytest.raises(GateError, match="torch build"):
        require_cuda_hardware(
            manifest,
            type("Torch", (), {"__version__": "2.9.0+cu128", "cuda": FakeCuda()})(),
        )


def test_dependency_import_smoke_requires_importable_exact_versions() -> None:
    manifest = load_manifest()

    def matching_import(package: str) -> object:
        return type(
            "Dependency", (), {"__version__": manifest["runtime"]["dependencies"][package]}
        )()

    assert verify_dependency_import_smoke(manifest, importer=matching_import) == (
        "accelerate",
        "peft",
        "transformers",
    )

    with pytest.raises(GateError, match="could not be imported"):
        verify_dependency_import_smoke(
            manifest,
            importer=lambda _package: (_ for _ in ()).throw(ImportError("broken")),
        )
    with pytest.raises(GateError, match="expected"):
        verify_dependency_import_smoke(
            manifest,
            importer=lambda _package: type("Dependency", (), {"__version__": "wrong"})(),
        )


def test_attested_volume_receipt_is_built_only_from_exact_h100_runtime_facts() -> None:
    manifest = load_manifest()
    dependency_evidence = private_dependency_evidence()
    code_evidence = private_code_evidence()
    receipt = build_attested_volume_readiness_receipt(
        manifest,
        snapshot_digest=expected_snapshot_digest(manifest),
        volume_id=NETWORK_VOLUME_ID,
        data_center_id=NETWORK_VOLUME_DATA_CENTER_ID,
        volume_size_gb=NETWORK_VOLUME_SIZE_GB,
        dependency_versions=manifest["runtime"]["dependencies"],
        torch_version=manifest["runtime"]["torch_version"],
        torch_cuda_version=expected_cuda_version(manifest),
        hardware=CUDAHardware(
            gpu_name=manifest["hardware"]["gpu_id"],
            total_memory_bytes=manifest["hardware"]["minimum_cuda_memory_bytes"],
            bf16_supported=True,
        ),
        dependency_lock_digest=manifest["materialization"]["dependency_lock"]["digest"],
        dependency_quarantine_evidence=dependency_evidence,
        dependency_quarantine_evidence_digest=str(dependency_evidence["evidence_digest"]),
        dependency_private_tree_digest=str(dependency_evidence["private_tree_digest"]),
        code_materialization_evidence=code_evidence,
        code_materialization_evidence_digest=str(code_evidence["evidence_digest"]),
        code_private_tree_digest=str(code_evidence["private_tree_digest"]),
        now=NOW,
    )

    assert receipt["schema_version"] == 3
    assert receipt["attestation_revision"] == VOLUME_READINESS_ATTESTATION_REVISION
    assert receipt["snapshot_digest"] == expected_snapshot_digest(manifest)
    assert receipt["torch_version"] == "2.8.0+cu128"
    assert receipt["torch_cuda_version"] == "12.8"
    assert receipt["receipt_digest"].startswith("sha256:")

    with pytest.raises(GateError, match="snapshot"):
        build_attested_volume_readiness_receipt(
            manifest,
            snapshot_digest="sha256:" + "0" * 64,
            volume_id=NETWORK_VOLUME_ID,
            data_center_id=NETWORK_VOLUME_DATA_CENTER_ID,
            volume_size_gb=NETWORK_VOLUME_SIZE_GB,
            dependency_versions=manifest["runtime"]["dependencies"],
            torch_version=manifest["runtime"]["torch_version"],
            torch_cuda_version=expected_cuda_version(manifest),
            hardware=CUDAHardware(
                gpu_name=manifest["hardware"]["gpu_id"],
                total_memory_bytes=manifest["hardware"]["minimum_cuda_memory_bytes"],
                bf16_supported=True,
            ),
            dependency_lock_digest=manifest["materialization"]["dependency_lock"]["digest"],
            dependency_quarantine_evidence=dependency_evidence,
            dependency_quarantine_evidence_digest=str(dependency_evidence["evidence_digest"]),
            dependency_private_tree_digest=str(dependency_evidence["private_tree_digest"]),
            code_materialization_evidence=code_evidence,
            code_materialization_evidence_digest=str(code_evidence["evidence_digest"]),
            code_private_tree_digest=str(code_evidence["private_tree_digest"]),
        )


def test_private_materialization_evidence_is_exact_and_tamper_evident() -> None:
    manifest = load_manifest()
    dependency_evidence = private_dependency_evidence()
    assert (
        gate_module.verify_dependency_quarantine_evidence(
            manifest,
            dependency_evidence,
            workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        )["private_tree_digest"]
        == dependency_evidence["private_tree_digest"]
    )

    tampered_dependency = {
        **dependency_evidence,
        "lock_digest": "sha256:" + "0" * 64,
    }
    tampered_dependency["evidence_digest"] = gate_module.materialization_evidence_digest(
        tampered_dependency
    )
    with pytest.raises(GateError, match="lock_digest"):
        gate_module.verify_dependency_quarantine_evidence(
            manifest,
            tampered_dependency,
            workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        )

    code_evidence = private_code_evidence()
    assert (
        gate_module.verify_code_materialization_evidence(
            manifest,
            code_evidence,
            workload_bundle_digest=WORKLOAD_BUNDLE_DIGEST,
            workload_bundle_size_bytes=WORKLOAD_BUNDLE_SIZE_BYTES,
            workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        )["private_tree_digest"]
        == code_evidence["private_tree_digest"]
    )
    tampered_files = copy.deepcopy(code_evidence["files"])
    source_entry = next(
        entry
        for entry in tampered_files
        if entry["path"] == "repository_repair_large_model_trainer.py"
    )
    source_entry["sha256"] = "sha256:" + "0" * 64
    tampered_tree_digest = "sha256:" + hashlib.sha256(canonical_json(tampered_files)).hexdigest()
    tampered_code = {
        **code_evidence,
        "files": tampered_files,
        "source_tree_digest": tampered_tree_digest,
        "private_tree_digest": tampered_tree_digest,
        "private_root": (
            "/tmp/equinox-quarantine/code/" + tampered_tree_digest.removeprefix("sha256:")
        ),
    }
    tampered_code["evidence_digest"] = gate_module.materialization_evidence_digest(tampered_code)
    with pytest.raises(GateError, match=r"trainer\.py digest"):
        gate_module.verify_code_materialization_evidence(
            manifest,
            tampered_code,
            workload_bundle_digest=WORKLOAD_BUNDLE_DIGEST,
            workload_bundle_size_bytes=WORKLOAD_BUNDLE_SIZE_BYTES,
            workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        )


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


def tiny_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    symlinked: bool,
    base_manifest: dict[str, object] | None = None,
) -> tuple[dict[str, object], Path]:
    manifest = copy.deepcopy(base_manifest if base_manifest is not None else load_manifest())
    payloads = {
        "config.json": b'{"model_type":"tiny"}',
        "model-00001-of-00001.safetensors": b"tiny-weights",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    manifest["model"]["snapshot_files"] = {
        name: {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    monkeypatch.setattr(gate_module, "_EXPECTED_MANIFEST", manifest)
    model_cache = tmp_path / "models--test--tiny"
    snapshot = model_cache / "snapshots" / str(manifest["model"]["revision"])
    snapshot.mkdir(parents=True)
    blobs = model_cache / "blobs"
    blobs.mkdir()
    for name, payload in payloads.items():
        if symlinked:
            blob_name = hashlib.sha256(payload).hexdigest()
            (blobs / blob_name).write_bytes(payload)
            (snapshot / name).symlink_to(Path("../../blobs") / blob_name)
        else:
            (snapshot / name).write_bytes(payload)
    return manifest, snapshot


def test_local_snapshot_accepts_only_exact_regular_or_huggingface_symlink_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_manifest = load_manifest()
    regular_manifest, regular_snapshot = tiny_snapshot(
        tmp_path / "regular",
        monkeypatch,
        symlinked=False,
        base_manifest=base_manifest,
    )
    regular = verify_local_snapshot(regular_manifest, regular_snapshot)

    symlink_manifest, symlink_snapshot = tiny_snapshot(
        tmp_path / "symlink",
        monkeypatch,
        symlinked=True,
        base_manifest=base_manifest,
    )
    symlink = verify_local_snapshot(symlink_manifest, symlink_snapshot)

    assert regular["snapshot_digest"] == symlink["snapshot_digest"]
    assert regular["files"] == symlink["files"]


def test_verified_snapshot_is_a_private_regular_file_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, snapshot = tiny_snapshot(
        tmp_path / "source",
        monkeypatch,
        symlinked=True,
    )
    destination_parent = tmp_path / "local"
    destination_parent.mkdir()

    evidence = materialize_verified_snapshot(
        manifest,
        snapshot,
        destination_parent=destination_parent,
    )
    materialized = Path(evidence["snapshot_path"])

    assert materialized.parent == destination_parent
    assert materialized.stat().st_mode & 0o777 == 0o500
    assert set(path.name for path in materialized.iterdir()) == set(
        manifest["model"]["snapshot_files"]
    )
    assert all(path.is_file() and not path.is_symlink() for path in materialized.iterdir())
    source_blob = (snapshot / "config.json").resolve()
    source_blob.write_bytes(b"mutated after verification")
    assert (materialized / "config.json").read_bytes() == b'{"model_type":"tiny"}'


def test_local_snapshot_rejects_extra_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, snapshot = tiny_snapshot(tmp_path, monkeypatch, symlinked=False)
    (snapshot / "special_tokens_map.json").write_text("{}", encoding="utf-8")

    with pytest.raises(GateError, match="entry set"):
        verify_local_snapshot(manifest, snapshot)


@pytest.mark.parametrize("target_kind", ["absolute", "outside", "link-chain"])
def test_local_snapshot_rejects_unsafe_symlink_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    manifest, snapshot = tiny_snapshot(tmp_path, monkeypatch, symlinked=True)
    entry = snapshot / "config.json"
    entry.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b'{"model_type":"tiny"}')
    if target_kind == "absolute":
        entry.symlink_to(outside)
    elif target_kind == "outside":
        entry.symlink_to(Path("../../../outside"))
    else:
        chain = snapshot.parent.parent / "blobs" / ("a" * 64)
        chain.symlink_to(outside)
        entry.symlink_to(Path("../../blobs") / chain.name)

    with pytest.raises(GateError, match="symlink"):
        verify_local_snapshot(manifest, snapshot)


def test_local_snapshot_rejects_a_symlinked_leaf_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, snapshot = tiny_snapshot(tmp_path, monkeypatch, symlinked=False)
    alias = snapshot.parent / "aliased-revision"
    alias.symlink_to(snapshot, target_is_directory=True)

    with pytest.raises(GateError, match="directory is unavailable"):
        verify_local_snapshot(manifest, alias)


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
        "network_volume_id": NETWORK_VOLUME_ID,
        "network_volume_data_center_id": NETWORK_VOLUME_DATA_CENTER_ID,
        "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
        "snapshot_digest": expected_snapshot_digest(load_manifest()),
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_total_memory_bytes": 78_000_000_000,
        "bf16_supported": True,
        "dependency_lock_digest": receipt["dependency_lock_digest"],
        "dependency_quarantine_evidence_digest": receipt["dependency_quarantine_evidence_digest"],
        "dependency_private_tree_digest": receipt["dependency_private_tree_digest"],
        "code_materialization_evidence_digest": receipt["code_materialization_evidence_digest"],
        "code_private_tree_digest": receipt["code_private_tree_digest"],
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


def test_profile5_volume_receipt_fails_closed_under_profile6() -> None:
    previous_receipt = volume_receipt(profile_id="qwen2.5-coder-7b-runpod-h100@5")

    with pytest.raises(GateError, match="profile_id"):
        verify_volume_readiness_receipt(
            load_manifest(),
            previous_receipt,
            {"id": "network-volume-123", "dataCenterId": "EU-RO-1", "size": 50},
            now=NOW,
        )


def test_integrated_retention_evidence_binds_the_exact_paid_handoff() -> None:
    readiness = volume_receipt()
    evidence = retention_evidence(readiness)
    verification = verify_retention_checkpoint_evidence(
        load_manifest(),
        evidence,
        head_commit=SOURCE_HEAD_COMMIT,
        workload_bundle_digest=WORKLOAD_BUNDLE_DIGEST,
        workload_bundle_size_bytes=WORKLOAD_BUNDLE_SIZE_BYTES,
        workload_bundle_path=WORKLOAD_BUNDLE_PATH,
        bundle_stage_receipt_digest=BUNDLE_STAGE_RECEIPT_DIGEST,
        bootstrap_source_digest=BOOTSTRAP_SOURCE_DIGEST,
        volume_readiness_receipt_digest=str(readiness["receipt_digest"]),
        dependency_lock_digest=str(readiness["dependency_lock_digest"]),
        dependency_quarantine_evidence_digest=str(
            readiness["dependency_quarantine_evidence_digest"]
        ),
        dependency_private_tree_digest=str(readiness["dependency_private_tree_digest"]),
        code_materialization_evidence_digest=str(readiness["code_materialization_evidence_digest"]),
        code_private_tree_digest=str(readiness["code_private_tree_digest"]),
        network_volume_id=NETWORK_VOLUME_ID,
        data_center_id=NETWORK_VOLUME_DATA_CENTER_ID,
        volume_size_gb=NETWORK_VOLUME_SIZE_GB,
    )

    assert verification["head_commit"] == SOURCE_HEAD_COMMIT
    assert verification["checkpoint_sha256"] == "sha256:" + "6" * 64
    assert verification["evidence_digest"] == evidence["evidence_digest"]

    for field, value in (
        ("head_commit", "0" * 40),
        ("source_contract_digest", "sha256:" + "0" * 64),
        ("workload_bundle_digest", "sha256:" + "0" * 64),
        ("bootstrap_source_digest", "sha256:" + "0" * 64),
        ("volume_readiness_receipt_digest", "sha256:" + "0" * 64),
        ("torch_version", "2.8.0"),
        ("torch_cuda_version", "12.1"),
        ("cuda_available", False),
        ("gpu_name", "NVIDIA A40"),
        ("bf16_supported", False),
        ("checkpoint_size_bytes", 0),
        ("advanced_weight_after_resume_step", [0.99]),
        ("optimizer_state_entries_after_resume_step", 0),
        ("optimizer_state_digest_before_persist", "sha256:bad"),
        ("optimizer_state_digest_after_restore", "sha256:" + "9" * 64),
        ("optimizer_state_digest_after_resume_step", "sha256:" + "7" * 64),
        ("optimizer_parameter_device", "cpu"),
        ("optimizer_state_devices_after_restore", {"exp_avg": ["cpu"]}),
        ("checkpoint_storage_root", "/tmp/not-the-runpod-volume"),
        ("checkpoint_storage_checkpoint_device", 43),
        ("checkpoint_reopened_after_fsync", False),
        ("checkpoint_authentication_revision", "legacy-checkpoint@1"),
        ("checkpoint_authentication_mechanism_digest", "sha256:" + "0" * 64),
        ("checkpoint_generation", 0),
        ("checkpoint_manifest_digest", "not-a-digest"),
        ("checkpoint_authenticated_private_resume", False),
    ):
        tampered = {**evidence, field: value}
        tampered["evidence_digest"] = retention_checkpoint_evidence_digest(tampered)
        with pytest.raises(GateError, match="retention checkpoint"):
            verify_retention_checkpoint_evidence(
                load_manifest(),
                tampered,
                head_commit=SOURCE_HEAD_COMMIT,
                workload_bundle_digest=WORKLOAD_BUNDLE_DIGEST,
                workload_bundle_size_bytes=WORKLOAD_BUNDLE_SIZE_BYTES,
                workload_bundle_path=WORKLOAD_BUNDLE_PATH,
                bundle_stage_receipt_digest=BUNDLE_STAGE_RECEIPT_DIGEST,
                bootstrap_source_digest=BOOTSTRAP_SOURCE_DIGEST,
                volume_readiness_receipt_digest=str(readiness["receipt_digest"]),
                dependency_lock_digest=str(readiness["dependency_lock_digest"]),
                dependency_quarantine_evidence_digest=str(
                    readiness["dependency_quarantine_evidence_digest"]
                ),
                dependency_private_tree_digest=str(readiness["dependency_private_tree_digest"]),
                code_materialization_evidence_digest=str(
                    readiness["code_materialization_evidence_digest"]
                ),
                code_private_tree_digest=str(readiness["code_private_tree_digest"]),
                network_volume_id=NETWORK_VOLUME_ID,
                data_center_id=NETWORK_VOLUME_DATA_CENTER_ID,
                volume_size_gb=NETWORK_VOLUME_SIZE_GB,
            )


def test_cross_launch_checkpoint_authorization_uses_stable_mechanism_digest() -> None:
    readiness = volume_receipt()
    screen_evidence = retention_evidence(readiness)
    fresh_pilot_evidence = {
        **screen_evidence,
        "checkpoint_manifest_digest": "sha256:" + "a" * 64,
    }
    fresh_pilot_evidence["evidence_digest"] = retention_checkpoint_evidence_digest(
        fresh_pilot_evidence
    )

    assert (
        screen_evidence["checkpoint_authentication_mechanism_digest"]
        == fresh_pilot_evidence["checkpoint_authentication_mechanism_digest"]
        == gate_module.checkpoint_authentication_mechanism_digest()
    )
    assert screen_evidence["evidence_digest"] != fresh_pilot_evidence["evidence_digest"]


def test_exact_historical_profile6_receipt_is_only_a_screen_seed_bridge() -> None:
    root = DEFAULT_MANIFEST_PATH.parents[2]
    predecessor = json.loads(
        subprocess.run(
            [
                "git",
                "show",
                (f"{PREDECESSOR_MANIFEST_COMMIT}:research/studies/larger-model-eligibility.json"),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "profile_id": PREDECESSOR_PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "manifest_digest": PREDECESSOR_MANIFEST_DIGEST,
        "network_volume_id": NETWORK_VOLUME_ID,
        "network_volume_data_center_id": NETWORK_VOLUME_DATA_CENTER_ID,
        "network_volume_size_gb": NETWORK_VOLUME_SIZE_GB,
        "snapshot_digest": PREDECESSOR_SNAPSHOT_DIGEST,
        "dependencies": load_manifest()["runtime"]["dependencies"],
        "prepared_at": "2026-07-29T11:00:00Z",
        "ready": True,
    }
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    provider_volume = {
        "id": NETWORK_VOLUME_ID,
        "dataCenterId": NETWORK_VOLUME_DATA_CENTER_ID,
        "size": NETWORK_VOLUME_SIZE_GB,
    }

    bridge = verify_predecessor_readiness_bridge(
        load_manifest(),
        predecessor,
        receipt,
        provider_volume,
        now=NOW,
    )

    assert bridge["predecessor_manifest_commit"] == PREDECESSOR_MANIFEST_COMMIT
    assert bridge["seed_only"] is True
    assert bridge["remote_current_attestation_required"] is True
    assert bridge["current_profile_id"] == PROFILE_ID

    tampered_predecessor = copy.deepcopy(predecessor)
    tampered_predecessor["runtime"]["dependencies"]["peft"] = "0.0.0"
    with pytest.raises(GateError, match="pinned committed"):
        verify_predecessor_readiness_bridge(
            load_manifest(),
            tampered_predecessor,
            receipt,
            provider_volume,
            now=NOW,
        )
    with pytest.raises(GateError, match="predecessor seed"):
        verify_predecessor_readiness_bridge(
            load_manifest(),
            predecessor,
            receipt,
            {**provider_volume, "id": "other-volume"},
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
    assert authorization["optimization_seed"] == 137
    assert authorization["source_contract_digest"] == expected_source_contract_digest(
        load_manifest()
    )
    assert authorization["terminal_submission_contract"] == ("accepted-passing-test-or-finish@1")
    assert authorization["screen_result_digest"] == result_digest(screen)
    assert authorization["pinned_snapshot_digest"] == expected_snapshot_digest(load_manifest())
    assert authorization["image"] == IMAGE_TAG
    assert authorization["image_digest"] == IMAGE_DIGEST
    assert authorization["network_volume_id"] == "network-volume-123"
    assert authorization["bundle_handoff_revision"] == BUNDLE_HANDOFF_REVISION
    assert authorization["workload_bundle_digest"] == WORKLOAD_BUNDLE_DIGEST
    assert authorization["workload_bundle_size_bytes"] == 80_000
    assert authorization["workload_bundle_compression"] == "xz"
    assert authorization["bundle_stage_receipt_digest"] == BUNDLE_STAGE_RECEIPT_DIGEST
    assert authorization["bootstrap_source_digest"] == BOOTSTRAP_SOURCE_DIGEST
    assert authorization["provider_handle"] == "runpod://pods/pod-123"
    assert authorization["digest"].startswith("sha256:")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bundle_handoff_revision", "inline-bundle@1", "handoff revision"),
        ("workload_bundle_digest", "sha256:" + "0" * 64, "bundle path"),
        ("workload_bundle_digest", "not-a-digest", "SHA-256"),
        ("workload_bundle_size_bytes", 0, "bundle size"),
        ("workload_bundle_compression", "gzip", "compression"),
        ("workload_bundle_path", "/workspace/wrong.tar.xz", "bundle path"),
        ("bundle_stage_receipt_digest", "not-a-digest", "SHA-256"),
        ("bootstrap_source_digest", "not-a-digest", "SHA-256"),
    ],
)
def test_pilot_authorization_binds_the_exact_operational_handoff(
    field: str,
    value: object,
    message: str,
) -> None:
    screen = eligible_screen()
    receipt = receipt_for(screen)
    resource_profile = dict(receipt["resource_profile"])
    resource_profile[field] = value
    receipt["resource_profile"] = resource_profile

    with pytest.raises(GateError, match=message):
        verify_pilot_authorization(load_manifest(), screen, receipt, now=NOW)


def test_profile5_screen_and_provider_receipt_fail_closed_under_profile6() -> None:
    previous_screen = eligible_screen(
        profile_id="qwen2.5-coder-7b-runpod-h100@5",
        workload_revision="larger-model-eligibility-screen@5",
    )
    previous_receipt = receipt_for(previous_screen)

    with pytest.raises(GateError, match="result workload_revision"):
        verify_pilot_authorization(
            load_manifest(),
            previous_screen,
            previous_receipt,
            now=NOW,
        )

    current_screen = eligible_screen()
    old_profile_receipt = receipt_for(current_screen)
    old_resource_profile = dict(old_profile_receipt["resource_profile"])
    old_resource_profile["profile_id"] = "qwen2.5-coder-7b-runpod-h100@5"
    old_profile_receipt["resource_profile"] = old_resource_profile
    with pytest.raises(GateError, match="resource profile profile_id"):
        verify_pilot_authorization(
            load_manifest(),
            current_screen,
            old_profile_receipt,
            now=NOW,
        )


@pytest.mark.parametrize(
    ("result_overrides", "receipt_overrides", "message"),
    [
        ({"profile_id": "other-profile@1"}, {}, "result profile_id"),
        ({"model_revision": "main"}, {}, "result model_revision"),
        ({"optimization_seed": 138}, {}, "optimization_seed"),
        ({"source_contract_digest": "sha256:" + "0" * 64}, {}, "source_contract_digest"),
        ({"environment_revision": "simulator@old"}, {}, "environment_revision"),
        ({"action_protocol_revision": "tools@old"}, {}, "action_protocol_revision"),
        (
            {"terminal_submission_contract": "accepted-finish-only@1"},
            {},
            "terminal_submission_contract",
        ),
        ({"branch_width": 1}, {}, "branch_width"),
        ({"training_microbatch_size": 2}, {}, "training_microbatch_size"),
        ({"maximum_input_tokens": 4_096}, {}, "maximum_input_tokens"),
        ({"capacity_smoke_completed": False}, {}, "capacity_smoke_completed"),
        ({"gradient_checkpointing_enabled": False}, {}, "gradient_checkpointing_enabled"),
        (
            {"determinism": {"revision": "non-deterministic@1"}},
            {},
            "result determinism",
        ),
        ({"pinned_snapshot_digest": "sha256:" + "0" * 64}, {}, "pinned_snapshot_digest"),
        ({"eligible": False}, {}, "did not authorize"),
        ({"training_started": True}, {}, "training_started"),
        ({"persistent_policy_updates": 1}, {}, "persistent_policy_updates"),
        ({"policy_mutation_enabled": True}, {}, "policy_mutation_enabled"),
        ({"policy_mutation_detected": True}, {}, "policy weights"),
        ({"optimizer_state_restored": False}, {}, "optimizer-state"),
        ({"test_split_accessed": True}, {}, "test-split"),
        ({"test_examples_accessed": 1}, {}, "test_examples_accessed"),
        ({"peak_reserved_vram_fraction": 0.86}, {}, "outside"),
        ({"predicted_final_evaluation_seconds": 0.0}, {}, "positive pilot runtime"),
        ({"predicted_final_evaluation_seconds": 1_441.0}, {}, "outside"),
        ({"branch_groups": 7}, {}, "branch_groups"),
        ({"expected_baseline_examples": 7}, {}, "expected_baseline_examples"),
        ({"completed_baseline_examples": 7}, {}, "completed_baseline_examples"),
        (
            {"per_level_checkpoint_rates": {"0": 0.625}},
            {},
            "level 0 checkpoint rate",
        ),
        ({"informative_groups": 1}, {}, "informative_groups"),
        ({"solved_siblings": 1}, {}, "solved_siblings"),
        ({"failed_siblings": 1}, {}, "failed_siblings"),
        ({}, {"teardown_confirmed": False}, "teardown"),
        ({}, {"resource_profile": None}, "resource profile"),
        (
            {},
            {
                "resource_profile": {
                    "image": IMAGE_TAG,
                    "image_digest": IMAGE_DIGEST,
                    "network_volume_id": 123,
                }
            },
            "network volume",
        ),
        (
            {},
            {
                "resource_profile": {
                    "image": "runpod/pytorch:latest",
                    "image_digest": IMAGE_DIGEST,
                    "network_volume_id": "network-volume-123",
                }
            },
            "resource profile image",
        ),
        (
            {},
            {
                "resource_profile": {
                    "image": IMAGE_TAG,
                    "image_digest": "sha256:" + "0" * 64,
                    "network_volume_id": "network-volume-123",
                }
            },
            "resource profile image_digest",
        ),
        ({}, {"result": {}}, "exact screen result"),
        (
            {},
            {"resource_profile": {"profile_id": "other-profile@1"}},
            "resource profile profile_id",
        ),
        ({}, {"workload": {"model_revision": "main"}}, "workload model_revision"),
        (
            {},
            {
                "workload": {
                    "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
                    "model_revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
                },
            },
            "workload model_id",
        ),
        ({}, {"provider_name": "runpod"}, "provider_name"),
        ({}, {"provider_cli_version": ""}, "provider_cli_version"),
        ({}, {"provider": "runpod"}, "invalid field set"),
        (
            {},
            {"started_at": "2026-07-29T11:00:01Z"},
            "completion precedes",
        ),
        ({}, {"completed_at": "2026-07-20T11:00:00Z"}, "not fresh"),
    ],
)
def test_pilot_authorization_fails_closed(
    result_overrides: dict[str, object],
    receipt_overrides: dict[str, object],
    message: str,
) -> None:
    screen = eligible_screen(**result_overrides)
    receipt = receipt_for(screen)
    for key, value in receipt_overrides.items():
        if key in {"resource_profile", "workload"} and isinstance(value, dict):
            nested = dict(receipt[key])
            nested.update(value)
            receipt[key] = nested
        else:
            receipt[key] = value

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
