from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import research.runpod.artifact_publication as artifact_publication
import research.runpod.artifact_republisher as artifact_republisher
from packages.equinox_core.canonical import canonical_digest as api_canonical_digest
from research.runpod.artifact_publication import (
    ARCHIVABLE_PRIOR_ROLES,
    ARTIFACT_FILENAMES,
    PILOT_ARTIFACT_ROLES,
    SCREEN_ARTIFACT_ROLES,
    ArtifactPublicationBusy,
    ArtifactPublicationError,
    PublicationRequest,
    build_artifact_publication_envelope,
    build_committed_proof_replay_payload,
    load_committed_artifact_set,
    load_committed_provider_receipt,
    publish_artifact_set,
    recover_artifact_publication,
)
from research.runpod.artifact_republisher import (
    ProofReplayHttpError,
    dry_run_committed_proof,
    dry_run_committed_result,
    republish_committed_proof,
    republish_committed_screen,
)
from research.runpod.larger_model_gate import canonical_json, result_digest

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_ID = "runpod-proof-20260729T220000Z-deadbeef"
PILOT_PUBLICATION_ID = "runpod-proof-20260729T230000Z-cafefeed"
PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@10"
PRIOR_PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@6"


class InjectedFailure(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _with_self_digest(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    return {
        **value,
        digest_field: ("sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()),
    }


def _artifact_values(
    publication_id: str,
    *,
    publication_type: str,
) -> dict[str, dict[str, Any]]:
    manifest_digest = "sha256:" + "1" * 64
    source_contract_digest = "sha256:" + "2" * 64
    source_head_commit = "3" * 40
    bootstrap_source_digest = "sha256:" + "4" * 64
    workload_bundle_digest = "sha256:" + "5" * 64
    workload_bundle_size_bytes = 4096
    workload_bundle_path = f"/workspace/equinox-state/{workload_bundle_digest[7:]}.tar.xz"
    bundle_handoff_revision = "network-volume-bundle-handoff@3"
    live_stage_activation_revision = "live-stage-activation@3"
    live_stage_activation_digest = "sha256:" + "6" * 64
    materialization_identity = {
        "dependency_lock_digest": "sha256:" + "7" * 64,
        "dependency_quarantine_revision": "dependency-quarantine@1",
        "dependency_quarantine_evidence_digest": "sha256:" + "8" * 64,
        "dependency_private_tree_digest": "sha256:" + "9" * 64,
        "code_materialization_revision": "code-materialization@1",
        "code_materialization_evidence_digest": "sha256:" + "a" * 64,
        "code_private_tree_digest": "sha256:" + "b" * 64,
    }
    volume_identity = {
        "network_volume_id": "network-volume-test",
        "network_volume_data_center_id": "EU-RO-1",
        "network_volume_size_gb": 50,
    }
    deletion_identity = {
        "network_volume_deletion_required": publication_type == "pilot",
        "network_volume_deletion_deadline": (
            "2026-07-29T23:45:00Z" if publication_type == "pilot" else None
        ),
        "network_volume_deletion_attempted": publication_type == "pilot",
        "network_volume_deletion_confirmed": publication_type == "pilot",
        "network_volume_deletion_deadline_met": publication_type == "pilot",
        "network_volume_deleted_at": (
            "2026-07-29T23:31:00Z" if publication_type == "pilot" else None
        ),
    }
    readiness_receipt = _with_self_digest(
        {
            "schema_version": 1,
            "role": "readiness_receipt",
            "profile_id": PROFILE_ID,
            "proof_id": publication_id,
            "manifest_digest": manifest_digest,
            **materialization_identity,
            **volume_identity,
        },
        "receipt_digest",
    )
    bundle_receipt = _with_self_digest(
        {
            "schema_version": 1,
            "role": "bundle_receipt",
            "profile_id": PROFILE_ID,
            "proof_id": publication_id,
            "manifest_digest": manifest_digest,
            "source_contract_digest": source_contract_digest,
            "bundle_digest": workload_bundle_digest,
            "bundle_size_bytes": workload_bundle_size_bytes,
            "bundle_path": workload_bundle_path,
            "bundle_handoff_revision": bundle_handoff_revision,
            **volume_identity,
        },
        "receipt_digest",
    )
    torch_evidence = _with_self_digest(
        {
            "schema_version": 1,
            "role": "torch_evidence",
            "profile_id": PROFILE_ID,
            "proof_id": publication_id,
            "head_commit": source_head_commit,
            "source_contract_digest": source_contract_digest,
            "workload_bundle_digest": workload_bundle_digest,
            "workload_bundle_size_bytes": workload_bundle_size_bytes,
            "workload_bundle_path": workload_bundle_path,
            "bootstrap_source_digest": bootstrap_source_digest,
            "volume_readiness_receipt_digest": readiness_receipt["receipt_digest"],
            "bundle_stage_receipt_digest": bundle_receipt["receipt_digest"],
            **materialization_identity,
            **volume_identity,
        },
        "evidence_digest",
    )
    evidence_values = {
        "readiness_receipt": readiness_receipt,
        "bundle_receipt": bundle_receipt,
        "torch_evidence": torch_evidence,
    }
    run_result: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "proof_id": publication_id,
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "workload": (
            "repository-repair-larger-model-eligibility-screen"
            if publication_type == "screen"
            else "repository-repair-larger-model-pilot"
        ),
    }
    if publication_type == "screen":
        run_result.update(
            {
                "volume_readiness_receipt": evidence_values["readiness_receipt"],
                "retention_checkpoint_evidence": evidence_values["torch_evidence"],
                "source_head_commit": source_head_commit,
                "source_contract_digest": source_contract_digest,
                "live_stage_activation_revision": live_stage_activation_revision,
                "live_stage_activation_digest": live_stage_activation_digest,
                "volume_readiness_receipt_digest": evidence_values["readiness_receipt"][
                    "receipt_digest"
                ],
                "torch_retention_evidence_digest": evidence_values["torch_evidence"][
                    "evidence_digest"
                ],
                **materialization_identity,
            }
        )
    if publication_type == "pilot":
        run_result.update(
            {
                "adapter_persisted": True,
                "adapter_manifest": {"digest": "sha256:" + "d" * 64},
                "source_contract_digest": source_contract_digest,
                "network_volume_id": volume_identity["network_volume_id"],
                **materialization_identity,
            }
        )
    digest = result_digest(run_result)
    evidence_digests = {
        "volume_readiness_receipt_digest": evidence_values["readiness_receipt"]["receipt_digest"],
        "bundle_stage_receipt_digest": evidence_values["bundle_receipt"]["receipt_digest"],
        "torch_retention_evidence_digest": evidence_values["torch_evidence"]["evidence_digest"],
    }
    values = dict(evidence_values)
    values.update(
        {
            "run_result": run_result,
            "provider_receipt": {
                "provider_name": "RunPod",
                "provider_handle": f"runpod://pods/{publication_id}",
                "provider_cli_version": "2.7.2",
                "resource_profile": {
                    "gpu_id": "NVIDIA H100 80GB HBM3",
                    "gpu_count": 1,
                    "image": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
                    "image_digest": "sha256:" + "c" * 64,
                    "hourly_cost_usd": 2.69,
                    "cloud_type": "SECURE",
                    "data_center_ids": ["EU-RO-1"],
                    "persistent_volume_in_gb": 0,
                    "profile_id": PROFILE_ID,
                    "manifest_digest": manifest_digest,
                    "source_head_commit": source_head_commit,
                    "source_contract_digest": source_contract_digest,
                    "bootstrap_source_digest": bootstrap_source_digest,
                    "workload_bundle_digest": workload_bundle_digest,
                    "workload_bundle_size_bytes": workload_bundle_size_bytes,
                    "workload_bundle_compression": "xz",
                    "workload_bundle_path": workload_bundle_path,
                    "bundle_handoff_revision": bundle_handoff_revision,
                    "live_stage_activation_revision": live_stage_activation_revision,
                    "live_stage_activation_digest": live_stage_activation_digest,
                    "bundle_activation_digest": live_stage_activation_digest,
                    **materialization_identity,
                    **evidence_digests,
                    **volume_identity,
                    **deletion_identity,
                },
                "workload": {
                    "id": run_result["workload"],
                    "revision": run_result.get("workload_revision"),
                    "algorithm": run_result.get("algorithm"),
                    "static_branch_width": run_result["branch_width"],
                    "complexity_strategy": run_result["complexity_strategy"],
                    "scale_case_count": run_result.get("test_case_count"),
                    "maximum_horizon": run_result.get("maximum_horizon"),
                    "total_action_decisions": run_result.get("total_action_decisions"),
                    "model_id": run_result["model_id"],
                    "model_revision": run_result.get("model_revision"),
                    "task_domains": run_result.get("task_domains"),
                    "maximum_complexity_level": run_result.get("maximum_complexity_level"),
                    "reached_complexity_level": run_result.get("reached_complexity_level"),
                    "total_sampled_completions": run_result.get("total_sampled_completions"),
                    "total_sampled_actions": run_result.get("total_sampled_actions"),
                    "multi_step": run_result.get("multi_step"),
                    "restored_continuations": run_result.get("restored_continuations"),
                    "snapshot_fidelity": run_result.get("snapshot_fidelity"),
                    "replay_enabled": run_result.get("replay_enabled"),
                },
                "result": run_result,
                "started_at": "2026-07-29T22:00:00Z",
                "completed_at": "2026-07-29T22:30:00Z",
                "teardown_confirmed": True,
            },
            "attempt_record": {
                "schema_version": 1,
                "profile_id": PROFILE_ID,
                "proof_id": publication_id,
                "publication_type": publication_type,
                "provider_handle": f"runpod://pods/{publication_id}",
                "source_head_commit": source_head_commit,
                "source_contract_digest": source_contract_digest,
                "bootstrap_source_digest": bootstrap_source_digest,
                "workload_bundle_digest": workload_bundle_digest,
                **materialization_identity,
                "activation_digest": live_stage_activation_digest,
                "result_digest": digest,
                "started_at": "2026-07-29T22:00:00Z",
                "completed_at": "2026-07-29T22:30:00Z",
                "teardown_confirmed": True,
                **evidence_digests,
                **deletion_identity,
            },
        }
    )
    return values


def _publication_request(
    tmp_path: Path,
    *,
    preserve_prior: bool = True,
) -> tuple[PublicationRequest, dict[str, bytes], dict[str, bytes]]:
    proof_directory = tmp_path / "proofs"
    state_directory = tmp_path / "state"
    staged_directory = tmp_path / "staged"
    proof_directory.mkdir()
    state_directory.mkdir()
    staged_directory.mkdir()
    artifacts: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    values = _artifact_values(PUBLICATION_ID, publication_type="screen")
    for role in SCREEN_ARTIFACT_ROLES:
        path = staged_directory / ARTIFACT_FILENAMES[role]
        artifacts[role] = path
        payloads[role] = _write_json(path, values[role])
    prior_receipts: dict[str, Path] = {}
    prior_payloads: dict[str, bytes] = {}
    if preserve_prior:
        for role in ARCHIVABLE_PRIOR_ROLES:
            path = proof_directory / f"old-{role}.json"
            prior_receipts[role] = path
            prior_payloads[role] = _write_json(
                path,
                {
                    "schema_version": 1,
                    "role": role,
                    "profile_id": PRIOR_PROFILE_ID,
                },
            )
    request = PublicationRequest(
        publication_id=PUBLICATION_ID,
        publication_type="screen",
        profile_id=PROFILE_ID,
        proof_directory=proof_directory,
        state_directory=state_directory,
        artifacts=artifacts,
        teardown_confirmed=True,
        prior_profile_id=PRIOR_PROFILE_ID if preserve_prior else None,
        prior_receipts=prior_receipts,
    )
    return request, payloads, prior_payloads


def _rewrite_screen_evidence_coherently(
    request: PublicationRequest,
    *,
    role: str,
    field: str,
) -> None:
    evidence_values = {
        evidence_role: json.loads(request.artifacts[evidence_role].read_bytes())
        for evidence_role in (
            "readiness_receipt",
            "bundle_receipt",
            "torch_evidence",
        )
    }
    digest_fields = {
        "readiness_receipt": "receipt_digest",
        "bundle_receipt": "receipt_digest",
        "torch_evidence": "evidence_digest",
    }
    evidence = evidence_values[role]
    evidence[field] = evidence[field] + 1 if type(evidence.get(field)) is int else "different"
    digest_field = digest_fields[role]
    evidence_values[role] = _with_self_digest(
        {key: value for key, value in evidence.items() if key != digest_field},
        digest_field,
    )
    if role in {"readiness_receipt", "bundle_receipt"}:
        pointer_field = (
            "volume_readiness_receipt_digest"
            if role == "readiness_receipt"
            else "bundle_stage_receipt_digest"
        )
        torch_evidence = evidence_values["torch_evidence"]
        torch_evidence[pointer_field] = evidence_values[role]["receipt_digest"]
        evidence_values["torch_evidence"] = _with_self_digest(
            {key: value for key, value in torch_evidence.items() if key != "evidence_digest"},
            "evidence_digest",
        )

    run_result = json.loads(request.artifacts["run_result"].read_bytes())
    run_result["volume_readiness_receipt"] = evidence_values["readiness_receipt"]
    run_result["volume_readiness_receipt_digest"] = evidence_values["readiness_receipt"][
        "receipt_digest"
    ]
    run_result["retention_checkpoint_evidence"] = evidence_values["torch_evidence"]
    run_result["torch_retention_evidence_digest"] = evidence_values["torch_evidence"][
        "evidence_digest"
    ]
    observed_result_digest = result_digest(run_result)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["result"] = run_result
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt["result_digest"] = observed_result_digest
    for binding_field, evidence_role, evidence_digest_field in (
        (
            "volume_readiness_receipt_digest",
            "readiness_receipt",
            "receipt_digest",
        ),
        ("bundle_stage_receipt_digest", "bundle_receipt", "receipt_digest"),
        ("torch_retention_evidence_digest", "torch_evidence", "evidence_digest"),
    ):
        digest = evidence_values[evidence_role][evidence_digest_field]
        receipt["resource_profile"][binding_field] = digest
        attempt[binding_field] = digest
    for evidence_role, value in evidence_values.items():
        _write_json(request.artifacts[evidence_role], value)
    _write_json(request.artifacts["run_result"], run_result)
    _write_json(request.artifacts["provider_receipt"], receipt)
    _write_json(request.artifacts["attempt_record"], attempt)


def _write_model_archive(
    path: Path,
    *,
    unsafe_member: str | None = None,
) -> dict[str, Any]:
    file_payloads = {
        "adapter_config.json": b'{"r":8}\n',
        "adapter_model.safetensors": b"SAFE" * 1024,
    }
    manifest_content: dict[str, Any] = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "revision",
        "workload_revision": "runpod-repository-repair-large-model-pilot@6",
        "objective_id": "repository-repair",
        "training_configuration": {"rank": 8},
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in file_payloads.items()
        ],
    }
    manifest = {
        **manifest_content,
        "digest": ("sha256:" + hashlib.sha256(canonical_json(manifest_content)).hexdigest()),
    }
    if unsafe_member == "manifest_digest":
        manifest["digest"] = "sha256:" + "f" * 64
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo("adapter")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        archive.addfile(directory)
        archived_files = dict(file_payloads)
        if unsafe_member == "digest":
            archived_files["adapter_model.safetensors"] = b"FAIL" * 1024
        archive_payloads = {
            "adapter/adapter-manifest.json": canonical_json(manifest) + b"\n",
            **{f"adapter/{name}": payload for name, payload in archived_files.items()},
        }
        for name, payload in archive_payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o600
            archive.addfile(member, io.BytesIO(payload))
        if unsafe_member == "extra":
            payload = b"extra"
            member = tarfile.TarInfo("adapter/extra.txt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        elif unsafe_member == "traversal":
            payload = b"escape"
            member = tarfile.TarInfo("adapter/../escape")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        elif unsafe_member == "link":
            member = tarfile.TarInfo("adapter/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape"
            archive.addfile(member)
        elif unsafe_member == "special":
            member = tarfile.TarInfo("adapter/device")
            member.type = tarfile.CHRTYPE
            archive.addfile(member)
    return manifest


def _bind_model_artifact_attempt(
    attempt_path: Path,
    model_path: Path,
) -> bytes:
    attempt = json.loads(attempt_path.read_bytes())
    payload = model_path.read_bytes()
    attempt.update(
        {
            "model_artifact_sha256": ("sha256:" + hashlib.sha256(payload).hexdigest()),
            "model_artifact_size_bytes": len(payload),
        }
    )
    return _write_json(attempt_path, attempt)


def _pilot_publication_request(
    tmp_path: Path,
    source_screen: Any,
) -> tuple[PublicationRequest, dict[str, bytes]]:
    staged_directory = tmp_path / "pilot-staged"
    staged_directory.mkdir()
    values = _artifact_values(PILOT_PUBLICATION_ID, publication_type="pilot")
    artifacts: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    for role in PILOT_ARTIFACT_ROLES:
        path = staged_directory / ARTIFACT_FILENAMES[role]
        artifacts[role] = path
        if role == "model_artifact":
            adapter_manifest = _write_model_archive(path)
            values["run_result"]["adapter_manifest"] = adapter_manifest
            digest = result_digest(values["run_result"])
            values["provider_receipt"]["result"] = values["run_result"]
            values["attempt_record"]["result_digest"] = digest
            payload = path.read_bytes()
        else:
            payload = _write_json(path, values[role])
        payloads[role] = payload
    payloads["run_result"] = _write_json(
        artifacts["run_result"],
        values["run_result"],
    )
    payloads["provider_receipt"] = _write_json(
        artifacts["provider_receipt"],
        values["provider_receipt"],
    )
    _write_json(
        artifacts["attempt_record"],
        values["attempt_record"],
    )
    payloads["attempt_record"] = _bind_model_artifact_attempt(
        artifacts["attempt_record"],
        artifacts["model_artifact"],
    )
    request = PublicationRequest(
        publication_id=PILOT_PUBLICATION_ID,
        publication_type="pilot",
        profile_id=PROFILE_ID,
        proof_directory=source_screen.manifest_path.parent,
        state_directory=tmp_path / "state",
        artifacts=artifacts,
        teardown_confirmed=True,
        source_screen_publication_id=source_screen.publication_id,
        source_screen_manifest_digest=source_screen.artifact_set_manifest_digest,
    )
    return request, payloads


def _plan(request: PublicationRequest) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "publication_id": request.publication_id,
        "publication_type": request.publication_type,
        "profile_id": request.profile_id,
        "proof_directory": str(request.proof_directory),
        "state_directory": str(request.state_directory),
        "teardown_confirmed": request.teardown_confirmed,
        "artifacts": {
            role: str(request.artifacts[role])
            for role in (
                SCREEN_ARTIFACT_ROLES
                if request.publication_type == "screen"
                else PILOT_ARTIFACT_ROLES
            )
        },
        "source_screen": (
            {
                "publication_id": request.source_screen_publication_id,
                "artifact_set_manifest_digest": request.source_screen_manifest_digest,
            }
            if request.source_screen_publication_id is not None
            else None
        ),
        "prior_profile_id": request.prior_profile_id,
        "prior_receipts": {
            role: str(request.prior_receipts[role])
            for role in ARCHIVABLE_PRIOR_ROLES
            if role in request.prior_receipts
        },
    }


def _transaction_directory(request: PublicationRequest) -> Path:
    return (
        request.state_directory / "artifact-publications" / f"{request.publication_id}.transaction"
    )


def _run_crashing_publisher(
    request: PublicationRequest,
    tmp_path: Path,
    phase: str,
    *,
    sleep_at_phase: bool = False,
) -> subprocess.Popen[bytes]:
    plan_path = tmp_path / f"plan-{phase.replace(':', '-')}.json"
    marker_path = tmp_path / f"marker-{phase.replace(':', '-')}"
    _write_json(plan_path, _plan(request))
    action = (
        "marker.write_text('ready'); time.sleep(60)"
        if sleep_at_phase
        else "os.kill(os.getpid(), signal.SIGKILL)"
    )
    source = (
        "import os,signal,sys,time\n"
        "from pathlib import Path\n"
        "from research.runpod.artifact_publication import "
        "_request_from_plan,publish_artifact_set\n"
        "plan=Path(sys.argv[1]); marker=Path(sys.argv[2]); target=sys.argv[3]\n"
        f"def inject(phase):\n"
        f"    if phase == target: {action}\n"
        "publish_artifact_set(_request_from_plan(plan), fault_injector=inject)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        [sys.executable, "-c", source, str(plan_path), str(marker_path), phase],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.marker_path = marker_path  # type: ignore[attr-defined]
    return process


def test_publication_commits_one_verified_set_and_preserves_profile_v6(
    tmp_path: Path,
) -> None:
    request, payloads, prior_payloads = _publication_request(tmp_path)

    result = publish_artifact_set(request)

    assert result.publication_id == PUBLICATION_ID
    assert result.profile_id == PROFILE_ID
    assert result.manifest_path.name == f"{PUBLICATION_ID}.artifact-set.json"
    assert result.set_digest.startswith("sha256:")
    assert result.artifact_set_manifest_digest.startswith("sha256:")
    assert result.publication_type == "screen"
    assert set(result.artifact_paths) == set(SCREEN_ARTIFACT_ROLES)
    assert {path.name for path in result.generation_directory.iterdir()} == {
        *(ARTIFACT_FILENAMES[role] for role in SCREEN_ARTIFACT_ROLES),
        *(f"old-{role}.profile-v6.json" for role in ARCHIVABLE_PRIOR_ROLES),
    }
    for role, path in result.artifact_paths.items():
        assert path.read_bytes() == payloads[role]
        assert request.artifacts[role].read_bytes() == payloads[role]
    for role, path in result.archive_paths.items():
        assert path.name == f"old-{role}.profile-v6.json"
        assert path.parent == result.generation_directory
        assert path.read_bytes() == prior_payloads[role]
        assert request.prior_receipts[role].read_bytes() == prior_payloads[role]
    manifest = json.loads(result.manifest_path.read_bytes())
    assert manifest["artifact_set_committed"] is True
    assert manifest["artifact_set_manifest_digest"] == result.artifact_set_manifest_digest
    assert manifest["teardown_confirmed"] is True
    assert manifest["publication_type"] == "screen"
    assert [item["role"] for item in manifest["artifacts"]] == list(SCREEN_ARTIFACT_ROLES)
    assert all(
        item["path"].startswith(f".artifact-generations/{result.generation_directory.name}/")
        for item in manifest["prior_profile_archives"]
    )
    assert all(
        item["canonical_json_sha256"].startswith("sha256:") for item in manifest["artifacts"]
    )
    assert not _transaction_directory(request).exists()

    loaded = load_committed_artifact_set(
        request.proof_directory,
        PUBLICATION_ID,
        expected_profile_id=PROFILE_ID,
    )
    assert loaded == result


def test_commit_marker_is_never_visible_while_generation_is_partial(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    observed: list[str] = []
    manifest_path = request.proof_directory / f"{PUBLICATION_ID}.artifact-set.json"

    def observe(phase: str) -> None:
        observed.append(phase)
        if phase != "after_manifest_commit":
            assert not manifest_path.exists()
        if phase == "after_generation_rename":
            generations = [
                path
                for path in (request.proof_directory / ".artifact-generations").iterdir()
                if not path.name.startswith(".pending-")
            ]
            assert len(generations) == 1
            assert {item.name for item in generations[0].iterdir()} == {
                ARTIFACT_FILENAMES[role] for role in SCREEN_ARTIFACT_ROLES
            }

    result = publish_artifact_set(request, fault_injector=observe)

    assert "after_generation_rename" in observed
    assert "after_manifest_commit" in observed
    assert result.manifest_path == manifest_path


def test_prior_profile_archives_remain_behind_the_commit_manifest(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    manifest_path = request.proof_directory / f"{PUBLICATION_ID}.artifact-set.json"

    def observe(phase: str) -> None:
        assert not list(request.proof_directory.glob("*.profile-v6.json"))
        if phase != "after_manifest_commit":
            assert not manifest_path.exists()

    result = publish_artifact_set(request, fault_injector=observe)

    assert manifest_path.is_file()
    assert not list(request.proof_directory.glob("*.profile-v6.json"))
    assert set(result.archive_paths) == set(ARCHIVABLE_PRIOR_ROLES)
    assert all(path.parent == result.generation_directory for path in result.archive_paths.values())


@pytest.mark.parametrize(
    "phase",
    (
        "after_journal_fsync",
        "after_archive:bundle_receipt",
        "after_artifact_fsync:run_result",
        "after_generation_rename",
        "before_manifest_commit",
    ),
)
def test_precommit_injected_failures_roll_back_without_partial_visibility(
    tmp_path: Path,
    phase: str,
) -> None:
    request, _, prior_payloads = _publication_request(tmp_path)

    def inject(observed: str) -> None:
        if observed == phase:
            raise InjectedFailure(phase)

    with pytest.raises(InjectedFailure, match=phase):
        publish_artifact_set(request, fault_injector=inject)

    assert not (request.proof_directory / f"{PUBLICATION_ID}.artifact-set.json").exists()
    assert not _transaction_directory(request).exists()
    generation_root = request.proof_directory / ".artifact-generations"
    assert not generation_root.exists() or not list(generation_root.iterdir())
    assert not list(request.proof_directory.glob("*.profile-v6.json"))
    for role, payload in prior_payloads.items():
        assert request.prior_receipts[role].read_bytes() == payload
    with pytest.raises(ArtifactPublicationError, match="unavailable"):
        load_committed_artifact_set(request.proof_directory, PUBLICATION_ID)


def test_postcommit_injected_failure_finishes_as_success(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)

    def inject(phase: str) -> None:
        if phase == "after_manifest_commit":
            raise InjectedFailure(phase)

    result = publish_artifact_set(request, fault_injector=inject)

    assert result.as_dict()["artifact_set_committed"] is True
    assert not _transaction_directory(request).exists()
    assert load_committed_artifact_set(request.proof_directory, PUBLICATION_ID) == result


def test_publish_is_idempotent_and_conflicting_republication_fails_closed(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    first = publish_artifact_set(request)
    manifest_bytes = first.manifest_path.read_bytes()
    generation = first.generation_directory

    second = publish_artifact_set(request)

    assert second == first
    assert second.manifest_path.read_bytes() == manifest_bytes
    assert second.generation_directory == generation
    _write_json(
        request.artifacts["attempt_record"],
        {
            **_artifact_values(PUBLICATION_ID, publication_type="screen")["attempt_record"],
            "changed": True,
        },
    )
    with pytest.raises(ArtifactPublicationError, match="different artifacts"):
        publish_artifact_set(request)
    assert first.manifest_path.read_bytes() == manifest_bytes
    assert load_committed_artifact_set(request.proof_directory, PUBLICATION_ID) == first


def test_publication_requires_teardown_and_exact_json_artifact_roles(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    unconfirmed = PublicationRequest(
        **{
            **request.__dict__,
            "teardown_confirmed": False,
        }
    )
    with pytest.raises(ArtifactPublicationError, match="confirmed provider teardown"):
        publish_artifact_set(unconfirmed)

    missing = PublicationRequest(
        **{
            **request.__dict__,
            "artifacts": {
                role: path for role, path in request.artifacts.items() if role != "torch_evidence"
            },
        }
    )
    with pytest.raises(ArtifactPublicationError, match="declared artifact roles"):
        publish_artifact_set(missing)

    _write_json(
        request.artifacts["provider_receipt"],
        {
            "profile_id": PROFILE_ID,
            "proof_id": PUBLICATION_ID,
            "teardown_confirmed": False,
        },
    )
    with pytest.raises(ArtifactPublicationError, match="does not confirm"):
        publish_artifact_set(request)


def test_mismatched_payload_identity_and_non_object_json_are_rejected(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    _write_json(
        request.artifacts["run_result"],
        {
            **_artifact_values(PUBLICATION_ID, publication_type="screen")["run_result"],
            "profile_id": "different-profile@1",
        },
    )
    with pytest.raises(ArtifactPublicationError, match="different or missing profile_id"):
        publish_artifact_set(request)

    _write_json(request.artifacts["run_result"], [])
    with pytest.raises(ArtifactPublicationError, match="one JSON object"):
        publish_artifact_set(request)

    request.artifacts["run_result"].write_text(
        '{"profile_id":"' + PROFILE_ID + '","proof_id":"' + PUBLICATION_ID + '","nonfinite":NaN}\n'
    )
    with pytest.raises(ArtifactPublicationError, match="canonical JSON"):
        publish_artifact_set(request)


def test_preexisting_top_level_archive_is_never_used_or_overwritten(
    tmp_path: Path,
) -> None:
    request, _, prior_payloads = _publication_request(tmp_path)
    collision = request.proof_directory / "old-readiness_receipt.profile-v6.json"
    collision_payload = _write_json(
        collision,
        {"profile_id": PRIOR_PROFILE_ID, "different": True},
    )

    result = publish_artifact_set(request)

    assert collision.read_bytes() == collision_payload
    assert (
        result.archive_paths["readiness_receipt"].read_bytes()
        == prior_payloads["readiness_receipt"]
    )
    assert result.archive_paths["readiness_receipt"].parent == result.generation_directory
    assert (
        request.prior_receipts["readiness_receipt"].read_bytes()
        == prior_payloads["readiness_receipt"]
    )


@pytest.mark.parametrize("tamper", ("artifact", "extra", "manifest"))
def test_reader_fails_closed_on_any_committed_set_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    result = publish_artifact_set(request)
    if tamper == "artifact":
        result.artifact_paths["run_result"].write_bytes(b"{}\n")
        expected = "digest or size"
    elif tamper == "extra":
        (result.generation_directory / "unexpected.json").write_text("{}\n")
        expected = "unexpected"
    else:
        manifest = json.loads(result.manifest_path.read_bytes())
        manifest["profile_id"] = "different-profile@1"
        result.manifest_path.write_text(json.dumps(manifest))
        expected = "identity|digest"

    with pytest.raises(ArtifactPublicationError, match=expected):
        load_committed_artifact_set(
            request.proof_directory,
            PUBLICATION_ID,
            expected_profile_id=PROFILE_ID,
        )


def test_publication_rejects_group_or_world_writable_evidence_boundaries(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    request.proof_directory.chmod(0o777)
    try:
        with pytest.raises(
            ArtifactPublicationError,
            match="group- or world-writable",
        ):
            publish_artifact_set(request)
    finally:
        request.proof_directory.chmod(0o700)

    result = publish_artifact_set(request)
    generation_root = result.generation_directory.parent
    generation_root.chmod(0o770)
    try:
        with pytest.raises(
            ArtifactPublicationError,
            match="group- or world-writable",
        ):
            load_committed_artifact_set(
                request.proof_directory,
                PUBLICATION_ID,
            )
    finally:
        generation_root.chmod(0o700)

    result.generation_directory.chmod(0o770)
    try:
        with pytest.raises(
            ArtifactPublicationError,
            match="group- or world-writable",
        ):
            load_committed_artifact_set(
                request.proof_directory,
                PUBLICATION_ID,
            )
    finally:
        result.generation_directory.chmod(0o700)


def test_sigkill_before_commit_is_recovered_as_a_full_rollback(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    process = _run_crashing_publisher(request, tmp_path, "after_generation_rename")
    process.wait(timeout=15)

    assert process.returncode == -signal.SIGKILL
    assert _transaction_directory(request).is_dir()
    assert not (request.proof_directory / f"{PUBLICATION_ID}.artifact-set.json").exists()

    assert (
        recover_artifact_publication(
            request.proof_directory,
            request.state_directory,
            PUBLICATION_ID,
        )
        is None
    )
    assert not _transaction_directory(request).exists()
    generation_root = request.proof_directory / ".artifact-generations"
    assert not list(generation_root.iterdir())
    assert not list(request.proof_directory.glob("*.profile-v6.json"))


def test_sigkill_after_commit_is_recovered_as_idempotent_completion(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    process = _run_crashing_publisher(request, tmp_path, "after_manifest_commit")
    process.wait(timeout=15)

    assert process.returncode == -signal.SIGKILL
    before = load_committed_artifact_set(request.proof_directory, PUBLICATION_ID)
    assert _transaction_directory(request).is_dir()

    recovered = recover_artifact_publication(
        request.proof_directory,
        request.state_directory,
        PUBLICATION_ID,
    )

    assert recovered == before
    assert not _transaction_directory(request).exists()
    assert (
        recover_artifact_publication(
            request.proof_directory,
            request.state_directory,
            PUBLICATION_ID,
        )
        == before
    )


def test_live_concurrent_publisher_is_rejected_but_stale_lock_recovers(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    process = _run_crashing_publisher(
        request,
        tmp_path,
        "after_journal_fsync",
        sleep_at_phase=True,
    )
    marker = process.marker_path  # type: ignore[attr-defined]
    deadline = time.monotonic() + 10
    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists(), process.stderr.read().decode() if process.stderr else ""
    competing_request = PublicationRequest(
        **{
            **request.__dict__,
            "state_directory": tmp_path / "different-state-root",
        }
    )
    try:
        with pytest.raises(ArtifactPublicationBusy, match="is active"):
            publish_artifact_set(competing_request)
    finally:
        process.kill()
        process.wait(timeout=10)

    assert (
        recover_artifact_publication(
            request.proof_directory,
            request.state_directory,
            PUBLICATION_ID,
        )
        is None
    )
    lock_path = request.proof_directory / f".{PUBLICATION_ID}.artifact-publication.lock"
    assert lock_path.is_file()
    assert not _transaction_directory(request).exists()


def test_cli_publishes_and_verifies_only_through_commit_manifest(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path)
    plan_path = tmp_path / "publication-plan.json"
    _write_json(plan_path, _plan(request))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)

    published = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.runpod.artifact_publication",
            "publish",
            "--plan",
            str(plan_path),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    assert published.returncode == 0, published.stderr.decode()
    published_result = json.loads(published.stdout)
    assert published_result["artifact_set_committed"] is True
    assert published_result["artifact_set_manifest_digest"].startswith("sha256:")

    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.runpod.artifact_publication",
            "verify",
            "--proof-directory",
            str(request.proof_directory),
            "--publication-id",
            PUBLICATION_ID,
            "--expected-profile-id",
            PROFILE_ID,
            "--expected-publication-type",
            "screen",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    assert verified.returncode == 0, verified.stderr.decode()
    verified_result = json.loads(verified.stdout)
    assert verified_result["verified"] is True
    assert (
        verified_result["artifact_set_manifest_digest"]
        == published_result["artifact_set_manifest_digest"]
    )
    assert verified_result["publication_type"] == "screen"


def test_pilot_commits_binary_model_and_exact_source_screen_lineage(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, pilot_payloads = _pilot_publication_request(tmp_path, screen)

    pilot = publish_artifact_set(pilot_request)

    assert pilot.publication_type == "pilot"
    assert set(pilot.artifact_paths) == set(PILOT_ARTIFACT_ROLES)
    assert pilot.source_screen_publication_id == screen.publication_id
    assert pilot.source_screen_manifest_digest == screen.artifact_set_manifest_digest
    assert pilot.artifact_paths["model_artifact"].read_bytes() == pilot_payloads["model_artifact"]
    model_descriptor = pilot.artifact_descriptors["model_artifact"]
    assert model_descriptor == {
        "role": "model_artifact",
        "media_type": "application/vnd.equinox.lora-adapter+gzip",
        "filename": "model-artifact.tgz",
        "sha256": ("sha256:" + hashlib.sha256(pilot_payloads["model_artifact"]).hexdigest()),
        "size_bytes": len(pilot_payloads["model_artifact"]),
    }
    for role in ("run_result", "provider_receipt", "attempt_record"):
        descriptor = pilot.artifact_descriptors[role]
        assert descriptor["sha256"] == (
            "sha256:" + hashlib.sha256(pilot_payloads[role]).hexdigest()
        )
        assert descriptor["canonical_json_sha256"] == (
            "sha256:" + hashlib.sha256(canonical_json(json.loads(pilot_payloads[role]))).hexdigest()
        )
        assert descriptor["canonical_json_sha256"] != descriptor["sha256"]

    manifest = json.loads(pilot.manifest_path.read_bytes())
    assert manifest["publication_type"] == "pilot"
    assert manifest["source_screen"] == {
        "publication_id": screen.publication_id,
        "artifact_set_manifest_digest": screen.artifact_set_manifest_digest,
    }
    assert manifest["prior_profile_archives"] == []
    assert (
        load_committed_artifact_set(
            pilot_request.proof_directory,
            PILOT_PUBLICATION_ID,
            expected_profile_id=PROFILE_ID,
            expected_publication_type="pilot",
        )
        == pilot
    )
    with pytest.raises(ArtifactPublicationError, match="identity"):
        load_committed_artifact_set(
            pilot_request.proof_directory,
            PILOT_PUBLICATION_ID,
            expected_publication_type="screen",
        )


def test_pilot_requires_exact_committed_screen_lineage(tmp_path: Path) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    wrong_digest = PublicationRequest(
        **{
            **pilot_request.__dict__,
            "source_screen_manifest_digest": "sha256:" + "e" * 64,
        }
    )

    with pytest.raises(ArtifactPublicationError, match="does not match"):
        publish_artifact_set(wrong_digest)

    missing_source = PublicationRequest(
        **{
            **pilot_request.__dict__,
            "source_screen_publication_id": None,
            "source_screen_manifest_digest": None,
        }
    )
    with pytest.raises(ArtifactPublicationError, match="requires a committed"):
        publish_artifact_set(missing_source)


def test_pilot_reader_reverifies_source_screen_transitively(tmp_path: Path) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    pilot = publish_artifact_set(pilot_request)

    screen.artifact_paths["run_result"].write_bytes(b"{}\n")

    with pytest.raises(ArtifactPublicationError, match="digest or size"):
        load_committed_artifact_set(
            pilot_request.proof_directory,
            pilot.publication_id,
            expected_publication_type="pilot",
        )


def test_pilot_model_artifact_is_nonempty_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    pilot_request.artifacts["model_artifact"].write_bytes(b"")

    with pytest.raises(ArtifactPublicationError, match="must not be empty"):
        publish_artifact_set(pilot_request)

    pilot_request.artifacts["model_artifact"].write_bytes(b"x" * 65)
    monkeypatch.setattr(artifact_publication, "MAXIMUM_MODEL_ARTIFACT_BYTES", 64)
    with pytest.raises(ArtifactPublicationError, match="size limit"):
        publish_artifact_set(pilot_request)


def test_pilot_model_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    model_path = pilot_request.artifacts["model_artifact"]
    target = tmp_path / "untrusted-adapter.tgz"
    target.write_bytes(b"untrusted adapter")
    model_path.unlink()
    model_path.symlink_to(target)

    with pytest.raises(ArtifactPublicationError, match="must not be a symlink"):
        publish_artifact_set(pilot_request)


@pytest.mark.parametrize(
    ("unsafe_member", "expected_error"),
    (
        ("extra", "member sets differ"),
        ("traversal", "unsafe member path"),
        ("link", "link or special"),
        ("special", "link or special"),
        ("digest", "file digest does not match"),
        ("manifest_digest", "manifest identity is invalid"),
    ),
)
def test_pilot_model_archive_rejects_unsafe_or_unbound_contents(
    tmp_path: Path,
    unsafe_member: str,
    expected_error: str,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    model_path = pilot_request.artifacts["model_artifact"]
    archive_manifest = _write_model_archive(model_path, unsafe_member=unsafe_member)
    if unsafe_member == "manifest_digest":
        run_result_path = pilot_request.artifacts["run_result"]
        provider_receipt_path = pilot_request.artifacts["provider_receipt"]
        attempt_path = pilot_request.artifacts["attempt_record"]
        run_result = json.loads(run_result_path.read_bytes())
        run_result["adapter_manifest"] = archive_manifest
        digest = result_digest(run_result)
        provider_receipt = json.loads(provider_receipt_path.read_bytes())
        provider_receipt["result"] = run_result
        attempt = json.loads(attempt_path.read_bytes())
        attempt["result_digest"] = digest
        _write_json(run_result_path, run_result)
        _write_json(provider_receipt_path, provider_receipt)
        _write_json(attempt_path, attempt)
    _bind_model_artifact_attempt(
        pilot_request.artifacts["attempt_record"],
        model_path,
    )

    with pytest.raises(ArtifactPublicationError, match=expected_error):
        publish_artifact_set(pilot_request)


def test_pilot_attempt_record_must_bind_exact_model_bytes(tmp_path: Path) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    attempt_path = pilot_request.artifacts["attempt_record"]
    attempt = json.loads(attempt_path.read_bytes())
    attempt["model_artifact_sha256"] = "sha256:" + "e" * 64
    _write_json(attempt_path, attempt)

    with pytest.raises(ArtifactPublicationError, match="exact model_artifact"):
        publish_artifact_set(pilot_request)


def test_pilot_attempt_record_binds_the_provider_handoff(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    attempt_path = pilot_request.artifacts["attempt_record"]
    attempt = json.loads(attempt_path.read_bytes())
    attempt["dependency_lock_digest"] = "sha256:" + "e" * 64
    _write_json(attempt_path, attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=(
            "attempt_record dependency_lock_digest does not match "
            "provider resource_profile dependency_lock_digest"
        ),
    ):
        publish_artifact_set(pilot_request)


def test_pilot_result_binds_the_provider_materialization(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    run_result = json.loads(pilot_request.artifacts["run_result"].read_bytes())
    run_result["dependency_private_tree_digest"] = "sha256:" + "e" * 64
    receipt = json.loads(pilot_request.artifacts["provider_receipt"].read_bytes())
    receipt["result"] = run_result
    attempt = json.loads(pilot_request.artifacts["attempt_record"].read_bytes())
    attempt["result_digest"] = result_digest(run_result)
    _write_json(pilot_request.artifacts["run_result"], run_result)
    _write_json(pilot_request.artifacts["provider_receipt"], receipt)
    _write_json(pilot_request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=(
            "pilot run_result dependency_private_tree_digest does not match "
            "provider resource_profile dependency_private_tree_digest"
        ),
    ):
        publish_artifact_set(pilot_request)


def test_pilot_sigkill_recovery_preserves_source_and_removes_partial_binary_set(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    process = _run_crashing_publisher(
        pilot_request,
        tmp_path,
        "after_generation_rename",
    )
    process.wait(timeout=15)

    assert process.returncode == -signal.SIGKILL
    assert (
        recover_artifact_publication(
            pilot_request.proof_directory,
            pilot_request.state_directory,
            pilot_request.publication_id,
        )
        is None
    )
    assert not (
        pilot_request.proof_directory / f"{pilot_request.publication_id}.artifact-set.json"
    ).exists()
    assert (
        load_committed_artifact_set(
            screen_request.proof_directory,
            screen.publication_id,
            expected_publication_type="screen",
        )
        == screen
    )


def test_publication_type_and_cross_artifact_result_are_fail_closed(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    values = _artifact_values(PUBLICATION_ID, publication_type="screen")
    _write_json(
        request.artifacts["attempt_record"],
        {**values["attempt_record"], "publication_type": "pilot"},
    )
    with pytest.raises(ArtifactPublicationError, match="publication_type"):
        publish_artifact_set(request)

    _write_json(request.artifacts["attempt_record"], values["attempt_record"])
    _write_json(
        request.artifacts["provider_receipt"],
        {
            **values["provider_receipt"],
            "result": {**values["run_result"], "uncommitted_overlay": True},
        },
    )
    with pytest.raises(ArtifactPublicationError, match="exact committed run_result"):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", "different-workload"),
        ("model_id", "Different/Model"),
        ("static_branch_width", 8),
        ("complexity_strategy", "static"),
    ),
)
def test_provider_workload_must_match_the_exact_run_result(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["workload"][field] = value
    _write_json(request.artifacts["provider_receipt"], receipt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"provider_receipt workload {field} does not match run_result",
    ):
        publish_artifact_set(request)


def test_provider_receipt_rejects_completion_before_start(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["completed_at"] = "2026-07-29T21:59:59Z"
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt["completed_at"] = receipt["completed_at"]
    _write_json(request.artifacts["provider_receipt"], receipt)
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match="completed_at precedes started_at",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    "field",
    ("provider_handle", "started_at", "completed_at"),
)
def test_attempt_record_binds_the_exact_provider_receipt_identity(
    tmp_path: Path,
    field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt[field] = "different"
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"attempt_record does not match provider_receipt {field}",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("role", "digest_field"),
    (
        ("readiness_receipt", "receipt_digest"),
        ("bundle_receipt", "receipt_digest"),
        ("torch_evidence", "evidence_digest"),
    ),
)
def test_screen_evidence_requires_a_valid_exact_self_digest(
    tmp_path: Path,
    role: str,
    digest_field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    evidence = json.loads(request.artifacts[role].read_bytes())
    evidence[digest_field] = "sha256:" + "e" * 64
    _write_json(request.artifacts[role], evidence)

    with pytest.raises(ArtifactPublicationError, match="invalid self digest"):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("role", "binding_field"),
    (
        ("readiness_receipt", "volume_readiness_receipt_digest"),
        ("bundle_receipt", "bundle_stage_receipt_digest"),
        ("torch_evidence", "torch_retention_evidence_digest"),
    ),
)
def test_screen_evidence_is_bound_to_the_attempt_record(
    tmp_path: Path,
    role: str,
    binding_field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt[binding_field] = "sha256:" + "e" * 64
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"{role} does not match attempt_record {binding_field}",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("role", "binding_field"),
    (
        ("readiness_receipt", "volume_readiness_receipt_digest"),
        ("bundle_receipt", "bundle_stage_receipt_digest"),
        ("torch_evidence", "torch_retention_evidence_digest"),
    ),
)
def test_screen_evidence_is_bound_to_the_provider_resource_profile(
    tmp_path: Path,
    role: str,
    binding_field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["resource_profile"][binding_field] = "sha256:" + "e" * 64
    _write_json(request.artifacts["provider_receipt"], receipt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"{role} does not match provider resource_profile {binding_field}",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("role", "field", "resource_field"),
    (
        ("readiness_receipt", "manifest_digest", "manifest_digest"),
        ("bundle_receipt", "bundle_digest", "workload_bundle_digest"),
        ("torch_evidence", "head_commit", "source_head_commit"),
        ("readiness_receipt", "network_volume_id", "network_volume_id"),
        ("bundle_receipt", "network_volume_id", "network_volume_id"),
        ("torch_evidence", "network_volume_id", "network_volume_id"),
    ),
)
def test_screen_evidence_cannot_mix_shared_handoff_identities(
    tmp_path: Path,
    role: str,
    field: str,
    resource_field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    _rewrite_screen_evidence_coherently(
        request,
        role=role,
        field=field,
    )

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"{role}.*provider resource_profile {resource_field}",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("role", "result_field"),
    (
        ("readiness_receipt", "volume_readiness_receipt"),
        ("torch_evidence", "retention_checkpoint_evidence"),
    ),
)
def test_screen_evidence_is_the_exact_object_committed_in_the_result(
    tmp_path: Path,
    role: str,
    result_field: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    run_result = json.loads(request.artifacts["run_result"].read_bytes())
    run_result[result_field] = {**run_result[result_field], "uncommitted_overlay": True}
    digest = result_digest(run_result)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["result"] = run_result
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt["result_digest"] = digest
    _write_json(request.artifacts["run_result"], run_result)
    _write_json(request.artifacts["provider_receipt"], receipt)
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"{role} does not match the exact committed run_result",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("torch_binding", "expected_role"),
    (
        ("volume_readiness_receipt_digest", "readiness_receipt"),
        ("bundle_stage_receipt_digest", "bundle_receipt"),
    ),
)
def test_screen_torch_evidence_binds_the_other_exact_evidence_objects(
    tmp_path: Path,
    torch_binding: str,
    expected_role: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    torch_evidence = json.loads(request.artifacts["torch_evidence"].read_bytes())
    torch_evidence[torch_binding] = "sha256:" + "e" * 64
    torch_evidence = _with_self_digest(
        {key: value for key, value in torch_evidence.items() if key != "evidence_digest"},
        "evidence_digest",
    )
    torch_digest = torch_evidence["evidence_digest"]
    run_result = json.loads(request.artifacts["run_result"].read_bytes())
    run_result["retention_checkpoint_evidence"] = torch_evidence
    run_result["torch_retention_evidence_digest"] = torch_digest
    result_sha256 = result_digest(run_result)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["result"] = run_result
    receipt["resource_profile"]["torch_retention_evidence_digest"] = torch_digest
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt["result_digest"] = result_sha256
    attempt["torch_retention_evidence_digest"] = torch_digest
    _write_json(request.artifacts["torch_evidence"], torch_evidence)
    _write_json(request.artifacts["run_result"], run_result)
    _write_json(request.artifacts["provider_receipt"], receipt)
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"torch_evidence does not bind the exact {expected_role}",
    ):
        publish_artifact_set(request)


@pytest.mark.parametrize(
    ("result_binding", "expected_role"),
    (
        ("volume_readiness_receipt_digest", "readiness_receipt"),
        ("torch_retention_evidence_digest", "torch_evidence"),
    ),
)
def test_screen_result_digest_fields_bind_the_exact_evidence(
    tmp_path: Path,
    result_binding: str,
    expected_role: str,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    run_result = json.loads(request.artifacts["run_result"].read_bytes())
    run_result[result_binding] = "sha256:" + "e" * 64
    digest = result_digest(run_result)
    receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    receipt["result"] = run_result
    attempt = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt["result_digest"] = digest
    _write_json(request.artifacts["run_result"], run_result)
    _write_json(request.artifacts["provider_receipt"], receipt)
    _write_json(request.artifacts["attempt_record"], attempt)

    with pytest.raises(
        ArtifactPublicationError,
        match=rf"run_result {result_binding} does not match {expected_role}",
    ):
        publish_artifact_set(request)


def test_json_descriptor_uses_the_same_canonical_digest_as_the_api(
    tmp_path: Path,
) -> None:
    request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    run_result = json.loads(request.artifacts["run_result"].read_bytes())
    run_result["unicode_label"] = "e\u0301"
    digest = result_digest(run_result)
    provider_receipt = json.loads(request.artifacts["provider_receipt"].read_bytes())
    provider_receipt["result"] = run_result
    attempt_record = json.loads(request.artifacts["attempt_record"].read_bytes())
    attempt_record["result_digest"] = digest
    _write_json(request.artifacts["run_result"], run_result)
    _write_json(request.artifacts["provider_receipt"], provider_receipt)
    _write_json(request.artifacts["attempt_record"], attempt_record)

    result = publish_artifact_set(request)

    assert result.artifact_descriptors["run_result"][
        "canonical_json_sha256"
    ] == api_canonical_digest(run_result)
    assert result.artifact_descriptors["provider_receipt"][
        "canonical_json_sha256"
    ] == api_canonical_digest(provider_receipt)
    assert api_canonical_digest(run_result) != (
        "sha256:" + hashlib.sha256(canonical_json(run_result)).hexdigest()
    )


def _committed_pilot_for_replay(tmp_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    pilot_request, _ = _pilot_publication_request(tmp_path, screen)
    pilot = publish_artifact_set(pilot_request)
    _, proof_payload = build_committed_proof_replay_payload(
        pilot_request.proof_directory,
        pilot.publication_id,
    )
    return screen, pilot, proof_payload


def _execution_for_replay(
    publication: Any,
    proof_payload: dict[str, Any],
    *,
    status: str = "FINALIZING",
    artifact_publication: dict[str, Any] | None = None,
    proof_id: str | None = None,
    receipt_digest: str | None = None,
    teardown_confirmed: bool = False,
    failure_message: str | None = None,
) -> dict[str, Any]:
    progress: dict[str, Any] = {"profile_id": PROFILE_ID}
    if failure_message is not None:
        progress["operator_error"] = {"message": failure_message}
    return {
        "execution_id": publication.publication_id,
        "name": "7B branch-aware pilot",
        "provider_name": "RunPod",
        "provider_handle": proof_payload["provider_handle"],
        "workload_id": proof_payload["workload"]["id"],
        "model_id": proof_payload["workload"]["model_id"],
        "started_at": proof_payload["started_at"],
        "resource_profile": proof_payload["resource_profile"],
        "progress": progress,
        "branch_width": proof_payload["workload"]["static_branch_width"],
        "complexity_strategy": "adaptive",
        "artifact_publication_required": True,
        "artifact_publication": artifact_publication,
        "status": status,
        "teardown_confirmed": teardown_confirmed,
        "completed_at": (
            proof_payload["completed_at"] if status in {"SUCCEEDED", "FAILED"} else None
        ),
        "failure_receipt_digest": ("sha256:" + "9" * 64 if status == "FAILED" else None),
        "proof_id": proof_id,
        "receipt_digest": receipt_digest,
    }


def _source_execution_for_replay(source: Any) -> dict[str, Any]:
    return {
        "execution_id": source.publication_id,
        "status": "SUCCEEDED",
        "teardown_confirmed": True,
        "artifact_publication_required": True,
        "artifact_publication": build_artifact_publication_envelope(source),
        "resource_profile": {"profile_id": PROFILE_ID},
    }


def _screen_execution_for_replay(
    screen: Any,
    provider_receipt: dict[str, Any],
    *,
    status: str = "FINALIZING",
    envelope: dict[str, Any] | None = None,
    teardown_confirmed: bool = False,
    failure_message: str | None = None,
) -> dict[str, Any]:
    progress: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "phase": "finalizing",
    }
    if failure_message is not None:
        progress["operator_error"] = {"message": failure_message}
    resource_profile = dict(provider_receipt["resource_profile"])
    # These receipt-only fields are learned after the execution registration
    # profile is frozen. Recovery binds the full shared handoff identity without
    # pretending that the earlier observer row contained later receipt fields.
    resource_profile.pop("network_volume_size_gb", None)
    resource_profile.pop("bundle_activation_digest", None)
    return {
        "execution_id": screen.publication_id,
        "name": "7B eligibility screen",
        "provider_name": "RunPod",
        "provider_handle": provider_receipt["provider_handle"],
        "workload_id": provider_receipt["workload"]["id"],
        "model_id": provider_receipt["workload"]["model_id"],
        "started_at": provider_receipt["started_at"].replace("Z", "+00:00"),
        "completed_at": (provider_receipt["completed_at"] if status == "SUCCEEDED" else None),
        "resource_profile": {
            **resource_profile,
            "maximum_total_cost_usd": 3.35,
        },
        "progress": progress,
        "branch_width": provider_receipt["workload"]["static_branch_width"],
        "complexity_strategy": "adaptive",
        "artifact_publication_required": True,
        "artifact_publication": envelope,
        "status": status,
        "teardown_confirmed": teardown_confirmed,
        "failure_receipt_digest": ("sha256:" + "9" * 64 if status == "FAILED" else None),
        "proof_id": None,
        "receipt_digest": None,
    }


def _proof_response_for_replay(
    pilot: Any,
    proof_payload: dict[str, Any],
    proof_id: str,
    receipt_digest: str,
) -> dict[str, Any]:
    return {
        "proof_id": proof_id,
        "execution_id": pilot.publication_id,
        "teardown_confirmed": True,
        "provider": {"handle": proof_payload["provider_handle"]},
        "evidence": {
            "receipt_digest": receipt_digest,
            "artifact_set_committed": True,
            "artifact_set_manifest_digest": pilot.artifact_set_manifest_digest,
        },
    }


@pytest.mark.parametrize(
    ("initial_status", "failure_message"),
    (
        ("FINALIZING", None),
        (
            "FAILED",
            "The completed larger-model eligibility screen could not be published.",
        ),
    ),
)
def test_committed_screen_finalization_replays_one_exact_zero_provider_put(
    tmp_path: Path,
    initial_status: str,
    failure_message: str | None,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    provider_receipt = load_committed_provider_receipt(screen)
    initial = _screen_execution_for_replay(
        screen,
        provider_receipt,
        status=initial_status,
        envelope=(
            build_artifact_publication_envelope(screen) if initial_status == "FAILED" else None
        ),
        teardown_confirmed=initial_status == "FAILED",
        failure_message=failure_message,
    )
    put_payloads: list[dict[str, Any]] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        if method == "GET":
            if not put_payloads:
                return dict(initial)
            return {
                **initial,
                **put_payloads[-1],
                "execution_id": screen.publication_id,
                "failure_receipt_digest": initial["failure_receipt_digest"],
            }
        assert method == "PUT"
        assert token == "t" * 40
        assert payload is not None
        put_payload = json.loads(payload)
        put_payloads.append(put_payload)
        return {
            **initial,
            **put_payload,
            "execution_id": screen.publication_id,
            "failure_receipt_digest": initial["failure_receipt_digest"],
        }

    result = republish_committed_screen(
        screen.manifest_path.parent,
        screen.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        requester=requester,
        sleeper=lambda _: None,
    )
    replayed = republish_committed_screen(
        screen.manifest_path.parent,
        screen.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        requester=requester,
        sleeper=lambda _: None,
    )

    assert result["screen_finalized"] is True
    assert replayed["screen_finalized"] is True
    assert result["provider_compute_used"] is False
    assert result["authorization_consumed"] is False
    assert len(put_payloads) == 2
    terminal = put_payloads[-1]
    assert terminal["status"] == "SUCCEEDED"
    assert terminal["teardown_confirmed"] is True
    assert terminal["artifact_publication"] == build_artifact_publication_envelope(screen)
    assert terminal["resource_profile"] == initial["resource_profile"]
    assert terminal["name"] == initial["name"]
    assert "error" not in terminal["progress"]
    assert "remote_error" not in terminal["progress"]
    assert "operator_error" not in terminal["progress"]


def test_committed_screen_running_recovery_advances_before_terminal_put(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    provider_receipt = load_committed_provider_receipt(screen)
    state = _screen_execution_for_replay(
        screen,
        provider_receipt,
        status="RUNNING",
    )
    state["progress"]["phase"] = "screening"
    put_payloads: list[dict[str, Any]] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal state
        if method == "GET":
            return dict(state)
        assert method == "PUT"
        assert token == "t" * 40
        assert payload is not None
        update = json.loads(payload)
        put_payloads.append(update)
        state = {
            **state,
            **update,
            "execution_id": screen.publication_id,
        }
        return dict(state)

    result = republish_committed_screen(
        screen.manifest_path.parent,
        screen.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        requester=requester,
        sleeper=lambda _: None,
    )

    assert result["screen_finalized"] is True
    assert [payload["status"] for payload in put_payloads] == [
        "FINALIZING",
        "SUCCEEDED",
    ]
    advance = put_payloads[0]
    assert advance["completed_at"] is None
    assert advance["teardown_confirmed"] is False
    assert advance["failure_receipt_digest"] is None
    assert advance["artifact_publication"] == build_artifact_publication_envelope(screen)
    assert advance["progress"]["phase"] == "finalizing"
    assert advance["progress"]["message"] == (
        "Committed artifacts verified; replaying finalization."
    )


def test_committed_screen_refuses_running_row_with_terminal_evidence_before_put(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    provider_receipt = load_committed_provider_receipt(screen)
    execution = _screen_execution_for_replay(
        screen,
        provider_receipt,
        status="RUNNING",
        envelope=build_artifact_publication_envelope(screen),
    )
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        return execution

    with pytest.raises(
        ArtifactPublicationError,
        match="RUNNING screen is not safe",
    ):
        republish_committed_screen(
            screen.manifest_path.parent,
            screen.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert methods == ["GET"]


def test_committed_screen_refuses_nonrecoverable_stale_demotion_before_put(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    provider_receipt = load_committed_provider_receipt(screen)
    execution = _screen_execution_for_replay(
        screen,
        provider_receipt,
        status="FAILED",
        teardown_confirmed=True,
        failure_message="unrelated stale reconciliation failure",
    )
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        return execution

    with pytest.raises(
        ArtifactPublicationError,
        match="not publication-recoverable",
    ):
        republish_committed_screen(
            screen.manifest_path.parent,
            screen.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert methods == ["GET"]


def test_committed_screen_refuses_shared_handoff_mutation_before_put(
    tmp_path: Path,
) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)
    provider_receipt = load_committed_provider_receipt(screen)
    execution = _screen_execution_for_replay(screen, provider_receipt)
    execution["resource_profile"]["workload_bundle_digest"] = "sha256:" + "0" * 64
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        return execution

    with pytest.raises(
        ArtifactPublicationError,
        match="does not match the committed screen",
    ):
        republish_committed_screen(
            screen.manifest_path.parent,
            screen.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert methods == ["GET"]


def test_committed_screen_dry_run_is_local_and_typed(tmp_path: Path) -> None:
    screen_request, _, _ = _publication_request(tmp_path, preserve_prior=False)
    screen = publish_artifact_set(screen_request)

    result = dry_run_committed_result(
        screen.manifest_path.parent,
        screen.publication_id,
    )

    assert result["publication_type"] == "screen"
    assert result["network_attempted"] is False
    assert result["registered_execution_required"] is True
    assert result["artifact_publication"] == build_artifact_publication_envelope(screen)


def test_committed_pilot_proof_replay_is_exact_idempotent_and_zero_provider(
    tmp_path: Path,
) -> None:
    screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    proof_id = "research_proof_replayed"
    receipt_digest = api_canonical_digest(proof_payload)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    authorization_directory = tmp_path / "authorization-consumption"
    authorization_directory.mkdir()
    authorization_marker = authorization_directory / "sentinel"
    authorization_marker.write_text("unchanged", encoding="utf-8")
    posted_payloads: list[dict[str, Any]] = []
    requests: list[tuple[str, str, str | None]] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        requests.append((method, url, token))
        assert timeout == 12.0
        if url.endswith(f"/{screen.publication_id}"):
            return _source_execution_for_replay(screen)
        if url.endswith(f"/{pilot.publication_id}"):
            if posted_payloads:
                return _execution_for_replay(
                    pilot,
                    proof_payload,
                    status="SUCCEEDED",
                    artifact_publication=proof_payload["artifact_publication"],
                    proof_id=proof_id,
                    receipt_digest=receipt_digest,
                    teardown_confirmed=True,
                )
            execution = _execution_for_replay(pilot, proof_payload)
            execution["started_at"] = proof_payload["started_at"].replace(
                "Z",
                "+00:00",
            )
            return execution
        if url.endswith("/internal/research-compute-proofs"):
            assert method == "POST"
            assert token == "t" * 40
            assert payload is not None
            posted = json.loads(payload)
            posted_payloads.append(posted)
            assert posted == proof_payload
            return {
                "proof_id": proof_id,
                "receipt_digest": api_canonical_digest(posted),
                "already_recorded": len(posted_payloads) > 1,
            }
        if url.endswith(f"/v1/proofs/{proof_id}"):
            return _proof_response_for_replay(
                pilot,
                proof_payload,
                proof_id,
                receipt_digest,
            )
        raise AssertionError(f"unexpected request {method} {url}")

    first = republish_committed_proof(
        pilot.manifest_path.parent,
        pilot.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        timeout_seconds=12,
        requester=requester,
        sleeper=lambda _: None,
    )
    second = republish_committed_proof(
        pilot.manifest_path.parent,
        pilot.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        timeout_seconds=12,
        requester=requester,
        sleeper=lambda _: None,
    )

    assert first["replayed"] is True
    assert first["already_recorded"] is False
    assert second["already_recorded"] is True
    assert first["provider_compute_used"] is False
    assert first["authorization_consumed"] is False
    assert len(posted_payloads) == 2
    assert all(token is None for method, _, token in requests if method == "GET")
    assert authorization_marker.read_text(encoding="utf-8") == "unchanged"
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and path != authorization_marker
    }
    assert after == before


def test_committed_pilot_refuses_registered_handoff_mutation_before_post(
    tmp_path: Path,
) -> None:
    _screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        assert method == "GET"
        assert url.endswith(f"/{pilot.publication_id}")
        execution = _execution_for_replay(pilot, proof_payload)
        execution["resource_profile"] = {
            **execution["resource_profile"],
            "workload_bundle_digest": "sha256:" + "0" * 64,
        }
        return execution

    with pytest.raises(
        ArtifactPublicationError,
        match="does not match the committed pilot proof",
    ):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert methods == ["GET"]


def test_committed_pilot_running_recovery_advances_before_proof_post(
    tmp_path: Path,
) -> None:
    screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    proof_id = "research_proof_running_recovery"
    receipt_digest = api_canonical_digest(proof_payload)
    state = _execution_for_replay(
        pilot,
        proof_payload,
        status="RUNNING",
    )
    state["progress"]["phase"] = "training"
    put_payloads: list[dict[str, Any]] = []
    requests: list[tuple[str, str]] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal state
        requests.append((method, url))
        if url.endswith(f"/{screen.publication_id}"):
            assert method == "GET"
            return _source_execution_for_replay(screen)
        if url.endswith(f"/{pilot.publication_id}"):
            if method == "PUT":
                assert token == "t" * 40
                assert payload is not None
                update = json.loads(payload)
                put_payloads.append(update)
                state = {
                    **state,
                    **update,
                    "execution_id": pilot.publication_id,
                }
            else:
                assert method == "GET"
            return dict(state)
        if url.endswith("/internal/research-compute-proofs"):
            assert method == "POST"
            assert token == "t" * 40
            assert payload is not None
            assert json.loads(payload) == proof_payload
            assert state["status"] == "FINALIZING"
            state = {
                **state,
                "status": "SUCCEEDED",
                "artifact_publication": proof_payload["artifact_publication"],
                "proof_id": proof_id,
                "receipt_digest": receipt_digest,
                "completed_at": proof_payload["completed_at"],
                "teardown_confirmed": True,
            }
            return {
                "proof_id": proof_id,
                "receipt_digest": receipt_digest,
                "already_recorded": False,
            }
        if url.endswith(f"/v1/proofs/{proof_id}"):
            assert method == "GET"
            return _proof_response_for_replay(
                pilot,
                proof_payload,
                proof_id,
                receipt_digest,
            )
        raise AssertionError(f"unexpected request {method} {url}")

    result = republish_committed_proof(
        pilot.manifest_path.parent,
        pilot.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        requester=requester,
        sleeper=lambda _: None,
    )

    assert result["replayed"] is True
    assert result["proof_id"] == proof_id
    assert len(put_payloads) == 1
    advance = put_payloads[0]
    assert advance["status"] == "FINALIZING"
    assert advance["completed_at"] is None
    assert advance["teardown_confirmed"] is False
    assert advance["artifact_publication"] == proof_payload["artifact_publication"]
    assert advance["progress"]["phase"] == "finalizing"
    assert [method for method, _ in requests] == [
        "GET",
        "GET",
        "PUT",
        "POST",
        "GET",
        "GET",
    ]


def test_committed_pilot_refuses_running_row_with_terminal_evidence_before_put(
    tmp_path: Path,
) -> None:
    _screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    execution = _execution_for_replay(
        pilot,
        proof_payload,
        status="RUNNING",
        artifact_publication=proof_payload["artifact_publication"],
    )
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        return execution

    with pytest.raises(
        ArtifactPublicationError,
        match="RUNNING execution is not safe",
    ):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert methods == ["GET"]


@pytest.mark.parametrize(
    ("status", "failure_message", "accepted"),
    (
        (
            "FAILED",
            "A verified local proof receipt is pending ingestion.",
            True,
        ),
        ("FAILED", "unrelated deterministic failure", False),
    ),
)
def test_proof_replay_accepts_only_registered_recoverable_states(
    tmp_path: Path,
    status: str,
    failure_message: str | None,
    accepted: bool,
) -> None:
    screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    methods: list[str] = []

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        methods.append(method)
        if url.endswith(f"/{pilot.publication_id}"):
            return _execution_for_replay(
                pilot,
                proof_payload,
                status=status,
                teardown_confirmed=status == "FAILED",
                failure_message=failure_message,
            )
        if url.endswith(f"/{screen.publication_id}"):
            return _source_execution_for_replay(screen)
        raise ProofReplayHttpError("stop after safe preflight")

    with pytest.raises(ArtifactPublicationError):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
            sleeper=lambda _: None,
        )

    assert ("GET" in methods) is True
    assert ("POST" in methods) is accepted


def test_proof_replay_fails_before_post_on_execution_or_source_mismatch(
    tmp_path: Path,
) -> None:
    screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)

    for mismatch in ("execution", "source"):
        methods: list[str] = []

        def requester(
            method: str,
            url: str,
            payload: bytes | None,
            token: str | None,
            timeout: float,
            current_mismatch: str = mismatch,
            current_methods: list[str] = methods,
        ) -> dict[str, Any]:
            current_methods.append(method)
            if url.endswith(f"/{pilot.publication_id}"):
                execution = _execution_for_replay(pilot, proof_payload)
                if current_mismatch == "execution":
                    execution["provider_handle"] = "runpod://pods/different"
                return execution
            if url.endswith(f"/{screen.publication_id}"):
                source = _source_execution_for_replay(screen)
                if current_mismatch == "source":
                    source["artifact_publication"] = None
                return source
            raise AssertionError("POST must not be attempted")

        with pytest.raises(ArtifactPublicationError, match="registered"):
            republish_committed_proof(
                pilot.manifest_path.parent,
                pilot.publication_id,
                api_root="http://127.0.0.1:8180",
                internal_token="t" * 40,
                requester=requester,
                sleeper=lambda _: None,
            )
        assert "POST" not in methods


def test_proof_replay_requires_local_internal_token_before_any_http(
    tmp_path: Path,
) -> None:
    _, pilot, _ = _committed_pilot_for_replay(tmp_path)
    calls = 0

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP must not start without the local token")

    with pytest.raises(ArtifactPublicationError, match="EQUINOX_INTERNAL_TOKEN"):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="",
            requester=requester,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "api_root",
    (
        "http://api.example.invalid:8180",
        "http://localhost:8180",
    ),
)
def test_proof_replay_never_sends_token_over_unverified_http(
    tmp_path: Path,
    api_root: str,
) -> None:
    _, pilot, _ = _committed_pilot_for_replay(tmp_path)
    calls = 0

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP must not start for an insecure API root")

    with pytest.raises(
        ArtifactPublicationError,
        match="requires HTTPS",
    ):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root=api_root,
            internal_token="t" * 40,
            requester=requester,
        )
    assert calls == 0


def test_proof_replay_rechecks_source_manifest_after_second_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _screen, pilot, _proof_payload = _committed_pilot_for_replay(tmp_path)
    real_loader = artifact_republisher.load_committed_artifact_set

    def changed_source_loader(*args: Any, **kwargs: Any) -> Any:
        publication = real_loader(*args, **kwargs)
        if kwargs.get("expected_publication_type") == "screen":
            return replace(
                publication,
                artifact_set_manifest_digest="sha256:" + "0" * 64,
            )
        return publication

    monkeypatch.setattr(
        artifact_republisher,
        "load_committed_artifact_set",
        changed_source_loader,
    )
    with pytest.raises(
        ArtifactPublicationError,
        match="source_screen changed during proof replay",
    ):
        republish_committed_proof(
            pilot.manifest_path.parent,
            pilot.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=lambda *_: pytest.fail("network must not start"),
        )


def test_proof_replay_dry_run_and_screen_refusal_use_no_network(
    tmp_path: Path,
) -> None:
    screen, pilot, _ = _committed_pilot_for_replay(tmp_path)

    dry_run = dry_run_committed_proof(
        pilot.manifest_path.parent,
        pilot.publication_id,
    )

    assert dry_run["dry_run"] is True
    assert dry_run["network_attempted"] is False
    assert dry_run["request_receipt_digest"].startswith("sha256:")
    calls = 0

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("screen replay must fail locally")

    with pytest.raises(ArtifactPublicationError, match="identity"):
        republish_committed_proof(
            screen.manifest_path.parent,
            screen.publication_id,
            api_root="http://127.0.0.1:8180",
            internal_token="t" * 40,
            requester=requester,
        )
    assert calls == 0
    wrapper = (ROOT / "scripts/republish-runpod-proof").read_text(encoding="utf-8")
    assert "artifact_republisher" in wrapper
    assert "runpodctl" not in wrapper
    assert "run-larger-model-pilot" not in wrapper
    assert "authorize" not in wrapper


def test_zero_provider_republisher_cli_dry_run(
    tmp_path: Path,
) -> None:
    _, pilot, _ = _committed_pilot_for_replay(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment.pop("EQUINOX_INTERNAL_TOKEN", None)

    completed = subprocess.run(
        [
            str(ROOT / "scripts/republish-runpod-proof"),
            "--proof-directory",
            str(pilot.manifest_path.parent),
            "--publication-id",
            pilot.publication_id,
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert result["network_attempted"] is False
    assert result["publication_type"] == "pilot"
    assert result["profile_id"] == PROFILE_ID


def test_proof_replay_retries_only_retryable_transport_failures(
    tmp_path: Path,
) -> None:
    screen, pilot, proof_payload = _committed_pilot_for_replay(tmp_path)
    receipt_digest = api_canonical_digest(proof_payload)
    proof_id = "research_proof_retry"
    post_attempts = 0
    posted = False

    def requester(
        method: str,
        url: str,
        payload: bytes | None,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal post_attempts, posted
        if url.endswith(f"/{screen.publication_id}"):
            return _source_execution_for_replay(screen)
        if url.endswith(f"/{pilot.publication_id}"):
            if posted:
                return _execution_for_replay(
                    pilot,
                    proof_payload,
                    status="SUCCEEDED",
                    artifact_publication=proof_payload["artifact_publication"],
                    proof_id=proof_id,
                    receipt_digest=receipt_digest,
                    teardown_confirmed=True,
                )
            return _execution_for_replay(pilot, proof_payload)
        if url.endswith("/internal/research-compute-proofs"):
            post_attempts += 1
            if post_attempts == 1:
                raise ProofReplayHttpError(
                    "temporary outage",
                    status=503,
                    retryable=True,
                )
            posted = True
            return {
                "proof_id": proof_id,
                "receipt_digest": receipt_digest,
                "already_recorded": False,
            }
        if url.endswith(f"/v1/proofs/{proof_id}"):
            return _proof_response_for_replay(
                pilot,
                proof_payload,
                proof_id,
                receipt_digest,
            )
        raise AssertionError("unexpected replay request")

    delays: list[float] = []
    result = republish_committed_proof(
        pilot.manifest_path.parent,
        pilot.publication_id,
        api_root="http://127.0.0.1:8180",
        internal_token="t" * 40,
        requester=requester,
        sleeper=delays.append,
    )

    assert result["replayed"] is True
    assert post_attempts == 2
    assert delays == [0.5]
