from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from research.runpod.bootstrap_server import (
    EXTERNAL_EVALUATION_WORKLOAD,
    install_bundle,
)
from research.runpod.external_eval_launcher import (
    LaunchFailure,
    build_bundle,
    load_frozen_pack,
    paired_change_summary,
    proxy_put_file,
    ssh_endpoint_from_pod,
    verify_result_contract,
)
from research.runpod.revision30_external_eval import (
    ADAPTER_SET,
    MODEL_ID,
    MODEL_REVISION,
    PACK_ID,
    STUDY_ID,
    WORKLOAD,
    WORKLOAD_REVISION,
    canonical_json,
)


def manifest() -> dict:
    archive_sha = "a" * 64
    return {
        "schema_version": 1,
        "evaluation_id": "revision30-external-eval-test",
        "study_id": STUDY_ID,
        "adapter_set": ADAPTER_SET,
        "pack_id": PACK_ID,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "adapters": [
            {
                "adapter_id": "k4_train_seed113",
                "condition_id": "k4_train_seed113",
                "role": "frozen_reference",
                "optimization_seed": 113,
                "source_execution_id": "runpod-proof-20260728T142404Z",
                "source_result_sha256": "b" * 64,
                "archive_filename": f"adapter-{archive_sha}.tgz",
                "size_bytes": 123,
                "sha256": archive_sha,
                "adapter_manifest_digest": "sha256:" + "c" * 64,
            }
        ],
    }


def policy(policy_id: str, frozen_pack: dict) -> dict:
    outcomes = [
        {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "solved": False,
            "failed_checks": ["still_failing"],
            "terminal_reason": "action_limit",
            "actions": 0,
            "accepted_actions": 0,
            "malformed_actions": 0,
            "trajectory": [],
        }
        for task in frozen_pack["tasks"]
    ]
    return {
        "policy_id": policy_id,
        "examples": len(outcomes),
        "exact_successes": 0,
        "exact_rate": 0.0,
        "domain_successes": {
            "micro_repository": 0,
            "sqlite_data_repair": 0,
            "filesystem_cli": 0,
        },
        "total_actions": 0,
        "accepted_actions": 0,
        "malformed_actions": 0,
        "action_protocol_validity_rate": None,
        "task_outcomes": outcomes,
    }


def result(repository_root: Path) -> tuple[dict, dict, dict]:
    input_manifest = manifest()
    frozen_pack = load_frozen_pack(repository_root)
    base = policy("disabled_adapter_base", frozen_pack)
    adapter_policy = {
        **input_manifest["adapters"][0],
        **policy("k4_train_seed113", frozen_pack),
        "paired_change_vs_base": paired_change_summary(
            base["task_outcomes"],
            policy("k4_train_seed113", frozen_pack)["task_outcomes"],
        ),
    }
    content = {
        "schema_version": 1,
        "workload": WORKLOAD,
        "workload_revision": WORKLOAD_REVISION,
        "external_evaluation_completed": True,
        "device": "cuda",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "pack": frozen_pack,
        "input_manifest": input_manifest,
        "base": base,
        "adapters": [adapter_policy],
        "adapter_count": 1,
        "every_adapter_reported": True,
        "task_count": frozen_pack["task_count"],
        "domain_task_counts": frozen_pack["domains"],
        "elapsed_seconds": 12.5,
    }
    complete = {
        **content,
        "result_digest": "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest(),
    }
    return complete, input_manifest, frozen_pack


def test_external_bundle_round_trips_through_the_bootstrap_allowlist(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    bundle = build_bundle(repository_root)

    install_bundle(
        bundle,
        work_directory=tmp_path,
        workload_file=EXTERNAL_EVALUATION_WORKLOAD,
    )

    assert (tmp_path / EXTERNAL_EVALUATION_WORKLOAD).is_file()
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        assert all(member.isfile() for member in archive.getmembers())


def test_external_result_contract_requires_every_policy_and_task() -> None:
    complete, input_manifest, frozen_pack = result(Path(__file__).resolve().parents[2])
    verify_result_contract(complete, input_manifest, frozen_pack)

    incomplete = dict(complete)
    incomplete["adapters"] = []
    content = {key: value for key, value in incomplete.items() if key != "result_digest"}
    incomplete["result_digest"] = "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()
    with pytest.raises(LaunchFailure, match="proof contract"):
        verify_result_contract(incomplete, input_manifest, frozen_pack)


def test_proxy_upload_reports_remote_http_error_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "manifest.json"
    source.write_text("{}", encoding="utf-8")

    def failed_upload(*_arguments: object, **_keywords: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["curl"],
            returncode=0,
            stdout=b'{"error":"NOT_FOUND"}\n404',
            stderr=b"",
        )

    monkeypatch.setattr(
        "research.runpod.external_eval_launcher.run_command",
        failed_upload,
    )

    with pytest.raises(
        LaunchFailure,
        match=r'HTTP 404: /inputs/manifest.json: \{"error":"NOT_FOUND"\}',
    ):
        proxy_put_file(
            "https://example.test",
            "/inputs/manifest.json",
            "token",
            source,
            timeout=10,
        )


def test_proxy_upload_parses_success_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "manifest.json"
    source.write_text("{}", encoding="utf-8")
    response = {"stored": "manifest.json", "size_bytes": 2, "sha256": "digest"}

    def successful_upload(
        *_arguments: object,
        **_keywords: object,
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["curl"],
            returncode=0,
            stdout=json.dumps(response).encode() + b"\n201",
            stderr=b"",
        )

    monkeypatch.setattr(
        "research.runpod.external_eval_launcher.run_command",
        successful_upload,
    )

    assert (
        proxy_put_file(
            "https://example.test",
            "/inputs/manifest.json",
            "token",
            source,
            timeout=10,
        )
        == response
    )


def test_ssh_endpoint_requires_a_public_port_22_mapping() -> None:
    pod = {
        "runtime": {
            "ports": [
                {
                    "ip": "203.0.113.17",
                    "isIpPublic": True,
                    "privatePort": 22,
                    "publicPort": 32061,
                }
            ]
        }
    }

    assert ssh_endpoint_from_pod(pod) == ("203.0.113.17", 32061)
    pod["runtime"]["ports"][0]["isIpPublic"] = False
    assert ssh_endpoint_from_pod(pod) is None
