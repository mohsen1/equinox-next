from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research.runpod.larger_model_gate import (
    PREDECESSOR_MANIFEST_DIGEST,
    PREDECESSOR_PROFILE_ID,
    PREDECESSOR_SNAPSHOT_DIGEST,
    CUDAHardware,
    build_attested_volume_readiness_receipt,
    build_live_stage_activation,
    canonical_json,
    checkpoint_authentication_mechanism_digest,
    expected_cuda_version,
    expected_snapshot_digest,
    expected_source_contract_digest,
    load_manifest,
    materialization_evidence_digest,
    retention_checkpoint_evidence_digest,
    retention_checkpoint_storage_evidence_digest,
)
from research.runpod.workload_bundle import (
    build_larger_model_bundle,
    stage_bundle_on_mounted_volume,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/runpod-rl-proof"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@10"
VOLUME_GUARD = REPOSITORY_ROOT / "scripts/runpod-volume-expiry-guard"
VOLUME_ID = "network-volume-123"
DATA_CENTER_ID = "EU-RO-1"
IMAGE_TAG = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
IMAGE_DIGEST = "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"


def _repository_head_commit() -> str:
    override = os.environ.get("EQUINOX_TEST_HEAD_COMMIT")
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except FileNotFoundError:
        # The containerized test image intentionally excludes Git and `.git`.
        # Its fake Git binary returns this same valid sentinel during launcher tests.
        return "0" * 40


HEAD_COMMIT = _repository_head_commit()
SCREEN_FAIL_FAST_REASONS = [
    "BRANCH_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "SOLVED_SIBLING_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "FAILED_SIBLING_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "ACTION_PROTOCOL_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "SOLVED_SIBLING_RATE_GATE_MATHEMATICALLY_IMPOSSIBLE",
]
SCREEN_BASELINE_FAIL_FAST_REASONS = [
    "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "OVERALL_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
    "BASELINE_EXACT_HEADROOM_GATE_MATHEMATICALLY_IMPOSSIBLE",
]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _image_index(*digests: str) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "digest": "sha256:" + "1" * 64,
        "manifests": [
            {
                "digest": digest,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": {"architecture": "amd64", "os": "linux"},
            }
            for digest in digests
        ],
    }


def _install_fake_commands(tmp_path: Path) -> tuple[Path, Path]:
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    command_log = tmp_path / "runpodctl.jsonl"
    _write_executable(
        binary_directory / "runpodctl",
        """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
with open(os.environ["FAKE_RUNPOD_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments, separators=(",", ":")) + "\\n")
scenario = json.loads(os.environ["FAKE_RUNPOD_SCENARIO"])
with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
    prior_commands = [json.loads(line) for line in handle if line.strip()]
volume_deleted = any(
    command[:2] == ["network-volume", "delete"] for command in prior_commands
)

if arguments == ["pod", "create", "--help"]:
    print("--env string")
    print("environment variables as json object")
elif arguments[:2] == ["template", "get"]:
    print(json.dumps({
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    }))
elif arguments[:2] == ["network-volume", "get"]:
    if volume_deleted:
        raise SystemExit(1)
    print(json.dumps(scenario.get("network_volume", {
        "id": "network-volume-123",
        "dataCenterId": "EU-RO-1",
        "size": 50
    })))
elif arguments == ["network-volume", "list"]:
    print(json.dumps([] if volume_deleted else scenario.get("network_volumes", [{
        "id": "network-volume-123",
        "name": "equinox-qwen25-coder-7b",
        "dataCenterId": "EU-RO-1",
        "size": 50
    }])))
elif arguments == ["datacenter", "list"]:
    print(json.dumps([{
        "id": "EU-RO-1",
        "gpuAvailability": [{
            "gpuId": "NVIDIA H100 80GB HBM3",
            "stockStatus": "Low"
        }]
    }]))
elif arguments == ["gpu", "list", "--include-unavailable"]:
    print(json.dumps(scenario.get("gpu_inventory", [{
        "available": True,
        "communityCloud": False,
        "displayName": "H100 SXM",
        "gpuId": "NVIDIA H100 80GB HBM3",
        "memoryInGb": 80,
        "secureCloud": True,
        "stockStatus": "Low"
    }])))
elif arguments == ["user"]:
    print(json.dumps({
        "clientBalance": scenario.get("client_balance", 45.00),
        "currentSpendPerHr": 0 if volume_deleted else scenario.get("spend", 0),
    }))
elif arguments == ["version"]:
    print("runpodctl 2.7.2")
elif arguments == ["pod", "list", "--all"]:
    print(json.dumps(scenario.get("pods", [])))
elif arguments[:2] == ["pod", "create"]:
    print(json.dumps({"id": "fake-paid-pod"}))
    if not scenario.get("pod_create_succeeds"):
        raise SystemExit(97)
elif arguments[:2] == ["pod", "get"]:
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        create = next(
            entry
            for line in handle
            if (entry := json.loads(line))[0:2] == ["pod", "create"]
            and entry != ["pod", "create", "--help"]
        )
    print(json.dumps({
        "id": "fake-paid-pod",
        "adjustedCostPerHr": scenario.get("pod_hourly_cost", 2.99),
        "imageName": scenario.get(
            "pod_image",
            "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
            "@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851",
        ),
        "gpuId": scenario.get("pod_gpu_id", "NVIDIA H100 80GB HBM3"),
        "gpuCount": scenario.get("pod_gpu_count", 1),
        "cloudType": scenario.get("pod_cloud_type", "SECURE"),
        "networkVolumeId": scenario.get("pod_network_volume_id", "network-volume-123"),
        "dataCenterId": scenario.get("pod_data_center_id", "EU-RO-1"),
        "volumeMountPath": scenario.get("pod_volume_mount_path", "/workspace"),
        "terminateAfter": scenario.get(
            "pod_terminate_after",
            create[create.index("--terminate-after") + 1],
        ),
        "containerDiskInGb": scenario.get(
            "pod_container_disk_in_gb",
            int(create[create.index("--container-disk-in-gb") + 1]),
        ),
    }))
elif arguments[:2] == ["pod", "delete"]:
    pass
elif arguments[:2] == ["network-volume", "delete"]:
    if arguments != ["network-volume", "delete", "network-volume-123"]:
        raise SystemExit(96)
else:
    print(f"unexpected fake runpodctl arguments: {arguments!r}", file=sys.stderr)
    raise SystemExit(98)
""",
    )
    _write_executable(
        binary_directory / "curl",
        """#!/usr/bin/env python3
import hashlib
import hmac
import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
with open(os.environ["FAKE_RUNPOD_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(["curl", *arguments], separators=(",", ":")) + "\\n")
scenario = json.loads(os.environ["FAKE_RUNPOD_SCENARIO"])
transport = scenario.get("transport")
url = next(
    (argument for argument in arguments if argument.startswith(("http://", "https://"))),
    arguments[-1],
)

def handoff_attestation():
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        creates = [
            entry
            for line in handle
            if (entry := json.loads(line))[0:2] == ["pod", "create"]
            and entry != ["pod", "create", "--help"]
        ]
    if not creates:
        return {}
    create = creates[-1]
    environment = json.loads(create[create.index("--env") + 1])
    evidence = scenario.get("live_evidence", {})
    activation = evidence.get("activation", {})
    return {
        **{
            key: value
            for key, value in activation.items()
            if key not in {"revision", "activation_digest"}
        },
        "live_stage_activation_revision": activation.get("revision"),
        "bundle_handoff_revision": "runpod-volume-bundle-handoff@1",
        "bundle_activation_digest": activation.get("activation_digest"),
        "workload_bundle_digest": environment.get("EQUINOX_BUNDLE_SHA256"),
        "workload_bundle_size_bytes": int(
            environment.get("EQUINOX_BUNDLE_SIZE_BYTES", "0")
        ),
        "workload_bundle_path": environment.get("EQUINOX_BUNDLE_VOLUME_PATH"),
        "bootstrap_source_digest": environment.get("EQUINOX_BOOTSTRAP_SOURCE_SHA256"),
        "network_volume_id": "network-volume-123",
    }

def health_payload(status):
    evidence = scenario["live_evidence"]
    metadata = evidence["bundle_metadata"]
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        create = next(
            entry
            for line in handle
            if (entry := json.loads(line))[0:2] == ["pod", "create"]
            and entry != ["pod", "create", "--help"]
        )
    environment = json.loads(create[create.index("--env") + 1])
    ready = status != "awaiting_bundle"
    activated = status == "activated"
    return {
        "revision": "authenticated-proxy-bundle-bootstrap@1",
        "status": status,
        "profile_id": metadata["profile_id"],
        "head_commit": """
        + repr(HEAD_COMMIT)
        + """,
        "manifest_digest": evidence["manifest_digest"],
        "source_contract_digest": evidence["source_contract_digest"],
        "bootstrap_source_digest": evidence["bootstrap_source_digest"],
        "network_volume_id": "network-volume-123",
        "network_volume_data_center_id": "EU-RO-1",
        "network_volume_size_gb": 50,
        "workload_bundle_digest": metadata["bundle_digest"],
        "workload_bundle_size_bytes": metadata["bundle_size_bytes"],
        "workload_bundle_path": metadata["bundle_path"],
        "readiness_deadline_epoch": int(
            environment["EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH"]
        ),
        "bundle_stage_receipt_digest": (
            evidence["stage_receipt"]["receipt_digest"] if ready else None
        ),
        "volume_readiness_receipt_digest": (
            evidence["readiness_receipt"]["receipt_digest"] if ready else None
        ),
        "torch_retention_evidence_digest": (
            evidence["retention_evidence"]["evidence_digest"] if ready else None
        ),
        "dependency_quarantine_evidence_digest": (
            evidence["dependency_evidence"]["evidence_digest"] if ready else None
        ),
        "dependency_private_tree_digest": (
            evidence["dependency_evidence"]["private_tree_digest"] if ready else None
        ),
        "code_materialization_evidence_digest": (
            evidence["code_evidence"]["evidence_digest"] if ready else None
        ),
        "code_private_tree_digest": (
            evidence["code_evidence"]["private_tree_digest"] if ready else None
        ),
        "activation_digest": (
            evidence["activation"]["activation_digest"] if activated else None
        ),
    }

def sequenced_response(key, suffix, default):
    responses = scenario.get(key)
    if responses is None:
        return default
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        call_count = sum(
            1
            for line in handle
            if (entry := json.loads(line))[0:1] == ["curl"]
            and any(argument.endswith(suffix) for argument in entry[1:])
        )
    return responses[min(call_count - 1, len(responses) - 1)]

def canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

def integrity_manifest():
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        commands = [json.loads(line) for line in handle if line.strip()]
    create = next(
        entry
        for entry in commands
        if entry[0:2] == ["pod", "create"]
        and entry != ["pod", "create", "--help"]
    )
    environment = json.loads(create[create.index("--env") + 1])
    token = environment["EQUINOX_RESULT_TOKEN"].encode()
    identity = {
        "proof_id": environment["EQUINOX_PROOF_ID"],
        "workload_file": "repository_repair_large_model_eligibility.py",
        "model_id": """
        + repr(MODEL_ID)
        + """,
        "optimization_seed": "137",
        "study_condition": "",
        "bundle_digest": environment["EQUINOX_BUNDLE_SHA256"],
        "bundle_activation_digest": environment["EQUINOX_BUNDLE_ACTIVATION_DIGEST"],
        "source_contract_digest": environment[
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256"
        ],
        "code_private_tree_digest": environment[
            "EQUINOX_CODE_PRIVATE_TREE_SHA256"
        ],
        "dependency_private_tree_digest": environment[
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256"
        ],
        "dependency_lock_digest": environment["EQUINOX_DEPENDENCY_LOCK_SHA256"],
    }
    identity_digest = "sha256:" + hashlib.sha256(canonical_json(identity)).hexdigest()

    def build(generation, parent, artifacts):
        material = {
            "schema_version": 1,
            "revision": "token-bound-runner-resume@1",
            "workload_file": identity["workload_file"],
            "launch_identity_digest": identity_digest,
            "generation": generation,
            "parent_manifest_sha256": parent,
            "directories": [],
            "artifacts": artifacts,
        }
        material["hmac"] = (
            "hmac-sha256:"
            + hmac.new(
                token,
                b"equinox/runner-resume-manifest/v1\\0" + canonical_json(material),
                hashlib.sha256,
            ).hexdigest()
        )
        return canonical_json(material) + b"\\n"

    initial = build(0, None, [])
    result_calls = sum(
        command[:1] == ["curl"]
        and any(argument.endswith("/result.json") for argument in command[1:])
        for command in commands
    )
    if result_calls == 0:
        return initial
    result_responses = scenario.get("result_responses") or []
    if not result_responses:
        return initial
    response = result_responses[min(result_calls - 1, len(result_responses) - 1)]
    result_payload = json.dumps(response, separators=(",", ":")).encode() + b"\\n"
    artifact_material = {
        "domain": "equinox/result/v1",
        "launch_identity_digest": identity_digest,
        "path": "result.json",
        "sha256": "sha256:" + hashlib.sha256(result_payload).hexdigest(),
        "size_bytes": len(result_payload),
    }
    artifact = {
        **artifact_material,
        "hmac": (
            "hmac-sha256:"
            + hmac.new(
                token,
                b"equinox/result/v1\\0" + canonical_json(artifact_material),
                hashlib.sha256,
            ).hexdigest()
        ),
    }
    return build(
        1,
        "sha256:" + hashlib.sha256(initial).hexdigest(),
        [artifact],
    )

if transport and url.endswith("/bootstrap-health"):
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        uploaded = any(
            (entry := json.loads(line))[0:1] == ["curl"]
            and any(argument.endswith("/bundle") for argument in entry[1:])
            and "-X" in entry
            for line in handle
        )
    print(json.dumps(
        health_payload("awaiting_stage_activation" if uploaded else "awaiting_bundle"),
        separators=(",", ":"),
    ))
elif transport and url.endswith("/bundle"):
    source = pathlib.Path(arguments[arguments.index("--data-binary") + 1].removeprefix("@"))
    if not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit(95)
    if transport == "reject":
        print(json.dumps({"error": "INVALID_BUNDLE"}, separators=(",", ":")))
        print("400")
    elif transport == "drop":
        raise SystemExit(52)
    else:
        print(json.dumps(
            health_payload("awaiting_stage_activation"),
            separators=(",", ":"),
        ))
        print("202")
elif transport and url.endswith("/bundle-stage-receipt.json"):
    print(json.dumps(scenario["live_evidence"]["stage_receipt"], separators=(",", ":")))
elif transport and url.endswith("/volume-readiness-receipt.json"):
    print(json.dumps(
        scenario["live_evidence"]["readiness_receipt"],
        separators=(",", ":"),
    ))
elif transport and url.endswith("/torch-retention-evidence.json"):
    print(json.dumps(
        scenario["live_evidence"]["retention_evidence"],
        separators=(",", ":"),
    ))
elif transport and url.endswith("/dependency-quarantine-evidence.json"):
    print(json.dumps(
        scenario["live_evidence"]["dependency_evidence"],
        separators=(",", ":"),
    ))
elif transport and url.endswith("/code-materialization-evidence.json"):
    print(json.dumps(
        scenario["live_evidence"]["code_evidence"],
        separators=(",", ":"),
    ))
elif transport and url.endswith("/activate-staged-bundle"):
    source = pathlib.Path(arguments[arguments.index("--data-binary") + 1].removeprefix("@"))
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != scenario["live_evidence"]["activation"]:
        print(json.dumps({"error": "INVALID_ACTIVATION"}, separators=(",", ":")))
        print("400")
    elif scenario.get("activation_transport") == "reject":
        print(json.dumps({"error": "ACTIVATION_REJECTED"}, separators=(",", ":")))
        print("409")
    elif scenario.get("activation_transport") == "drop":
        raise SystemExit(52)
    else:
        print(json.dumps(health_payload("activated"), separators=(",", ":")))
        print("202")
elif transport and url.endswith("/progress.json"):
    response = sequenced_response(
        "progress_responses",
        "/progress.json",
        {"phase": "running"},
    )
    if response is not None:
        print(json.dumps({**handoff_attestation(), **response}, separators=(",", ":")))
elif transport and url.endswith("/exit_code"):
    response = sequenced_response("exit_code_responses", "/exit_code", "1")
    if response is not None:
        print(response)
elif transport and url.endswith("/integrity.json"):
    sys.stdout.buffer.write(integrity_manifest())
elif transport and url.endswith("/result.json"):
    response = sequenced_response("result_responses", "/result.json", None)
    if response is not None:
        print(json.dumps(response, separators=(",", ":")))
elif url == "http://operator-test.invalid/v1/research-compute-executions":
    print(json.dumps(
        scenario.get("research_executions", {"items": []}),
        separators=(",", ":"),
    ))
elif url.startswith("http://operator-test.invalid/internal/"):
    if arguments[arguments.index("--data-binary") + 1] == "@-":
        payload = sys.stdin.read()
        with open(os.environ["FAKE_RUNPOD_PAYLOAD_LOG"], "a", encoding="utf-8") as handle:
            handle.write(payload.strip() + "\\n")
elif "--fail" not in arguments:
    raise SystemExit(96)
""",
    )
    _write_executable(
        binary_directory / "docker",
        """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
with open(os.environ["FAKE_RUNPOD_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(["docker", *arguments], separators=(",", ":")) + "\\n")
scenario = json.loads(os.environ["FAKE_RUNPOD_SCENARIO"])
if scenario.get("image_inspection_error"):
    print("registry unavailable", file=sys.stderr)
    raise SystemExit(95)
if arguments != [
    "buildx",
    "imagetools",
    "inspect",
    "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "--format",
    "{{json .Manifest}}",
]:
    raise SystemExit(94)
image_indexes = scenario.get("image_indexes")
if image_indexes is not None:
    with open(os.environ["FAKE_RUNPOD_LOG"], encoding="utf-8") as handle:
        inspection_number = sum(
            1
            for line in handle
            if json.loads(line)[0:1] == ["docker"]
        )
    if inspection_number > len(image_indexes):
        print("missing sequenced image inspection", file=sys.stderr)
        raise SystemExit(93)
    selected_index = image_indexes[inspection_number - 1]
    if selected_index is None:
        print("registry unavailable", file=sys.stderr)
        raise SystemExit(95)
else:
    selected_index = scenario.get("image_index", {
    "schemaVersion": 2,
    "mediaType": "application/vnd.oci.image.index.v1+json",
    "digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "manifests": [{
        "digest": "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851",
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "platform": {"architecture": "amd64", "os": "linux"}
    }, {
        "digest": "sha256:d78392453cbad3551b6330b058d224912b349dc5783759aa998fa091ee9b8826",
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "platform": {"architecture": "unknown", "os": "unknown"}
    }]
})
print(json.dumps(selected_index))
""",
    )
    _write_executable(
        binary_directory / "sleep",
        """#!/usr/bin/env python3
""",
    )
    _write_executable(
        binary_directory / "git",
        """#!/usr/bin/env python3
import os
import sys

arguments = sys.argv[1:]
if "diff" in arguments and "--quiet" in arguments:
    raise SystemExit(0)
os.execv("/usr/bin/git", ["/usr/bin/git", *arguments])
""",
    )
    return binary_directory, command_log


def _write_volume_receipt(path: Path) -> None:
    manifest = load_manifest()
    receipt = {
        "schema_version": 1,
        "profile_id": PREDECESSOR_PROFILE_ID,
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": PREDECESSOR_MANIFEST_DIGEST,
        "network_volume_id": VOLUME_ID,
        "network_volume_data_center_id": DATA_CENTER_ID,
        "network_volume_size_gb": 50,
        "snapshot_digest": PREDECESSOR_SNAPSHOT_DIGEST,
        "dependencies": manifest["runtime"]["dependencies"],
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    path.write_bytes(canonical_json(receipt) + b"\n")


def _private_materialization_evidence(
    manifest: dict[str, object],
    bundle_metadata: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    materialization = manifest["materialization"]
    assert isinstance(materialization, dict)
    lock = materialization["dependency_lock"]
    assert isinstance(lock, dict)
    lock_path = REPOSITORY_ROOT / "research/runpod" / str(lock["path"])
    locked_versions: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-z0-9][a-z0-9._-]*)==([^ \\]+) \\", raw_line)
        if match is not None:
            locked_versions[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    distributions: list[dict[str, object]] = []
    for name, version in sorted(locked_versions.items()):
        identity = canonical_json({"name": name, "version": version})
        identity_digest = "sha256:" + hashlib.sha256(identity).hexdigest()
        distributions.append(
            {
                "name": name,
                "version": version,
                "record_path": f"{name.replace('-', '_')}-{version}.dist-info/RECORD",
                "record_digest": identity_digest,
                "file_count": 1,
                "files_digest": identity_digest,
            }
        )
    install_tree_digest = (
        "sha256:" + hashlib.sha256(canonical_json({"distributions": distributions})).hexdigest()
    )
    dependency_evidence: dict[str, object] = {
        "schema_version": 2,
        "revision": lock["revision"],
        "profile_id": manifest["profile_id"],
        "lock_path": f"{bundle_metadata['bundle_path']}::{lock['path']}",
        "lock_digest": lock["digest"],
        "python_version": lock["python_version"],
        "platform_tag": lock["platform_tag"],
        "installer_revision": lock["installer_revision"],
        "index_url": lock["index_url"],
        "private_root": f"/tmp/equinox-quarantine/dependencies/{install_tree_digest[7:]}",
        "install_tree_digest": install_tree_digest,
        "private_tree_digest": install_tree_digest,
        "record_closure_digest": "sha256:"
        + hashlib.sha256(canonical_json(distributions)).hexdigest(),
        "distributions": distributions,
        "installed_file_count": len(distributions),
        "installed_bytes": len(distributions) * 128,
        "ready": True,
    }
    dependency_evidence["evidence_digest"] = materialization_evidence_digest(dependency_evidence)

    files: list[dict[str, object]] = []
    for name in sorted(str(item) for item in bundle_metadata["bundle_files"]):
        source = (
            REPOSITORY_ROOT / "research/studies" / name
            if name == "larger-model-eligibility.json"
            else REPOSITORY_ROOT / "research/runpod" / name
        )
        payload = source.read_bytes()
        files.append(
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    code_tree_digest = "sha256:" + hashlib.sha256(canonical_json(files)).hexdigest()
    code_evidence: dict[str, object] = {
        "schema_version": 1,
        "revision": materialization["code"]["revision"],
        "profile_id": manifest["profile_id"],
        "bundle_digest": bundle_metadata["bundle_digest"],
        "bundle_size_bytes": bundle_metadata["bundle_size_bytes"],
        "source_contract_digest": bundle_metadata["source_contract_digest"],
        "source_root": bundle_metadata["bundle_path"],
        "private_root": f"/tmp/equinox-quarantine/code/{code_tree_digest[7:]}",
        "source_tree_digest": code_tree_digest,
        "private_tree_digest": code_tree_digest,
        "files": files,
        "installed_file_count": len(files),
        "installed_bytes": sum(int(item["size_bytes"]) for item in files),
        "ready": True,
    }
    code_evidence["evidence_digest"] = materialization_evidence_digest(code_evidence)
    return dependency_evidence, code_evidence


def _live_stage_evidence(
    manifest: dict[str, object],
    bundle_metadata: dict[str, object],
    stage_receipt: dict[str, object],
) -> dict[str, object]:
    materialization = manifest["materialization"]
    dependency_evidence, code_evidence = _private_materialization_evidence(
        manifest,
        bundle_metadata,
    )
    readiness = build_attested_volume_readiness_receipt(
        manifest,
        snapshot_digest=expected_snapshot_digest(manifest),
        volume_id=VOLUME_ID,
        data_center_id=DATA_CENTER_ID,
        volume_size_gb=50,
        dependency_versions=manifest["runtime"]["dependencies"],
        torch_version=str(manifest["runtime"]["torch_version"]),
        torch_cuda_version=expected_cuda_version(manifest),
        hardware=CUDAHardware(
            gpu_name=str(manifest["hardware"]["gpu_id"]),
            total_memory_bytes=int(manifest["hardware"]["minimum_cuda_memory_bytes"]),
            bf16_supported=True,
        ),
        dependency_lock_digest=str(manifest["materialization"]["dependency_lock"]["digest"]),
        dependency_quarantine_evidence=dependency_evidence,
        dependency_quarantine_evidence_digest=str(dependency_evidence["evidence_digest"]),
        dependency_private_tree_digest=str(dependency_evidence["private_tree_digest"]),
        code_materialization_evidence=code_evidence,
        code_materialization_evidence_digest=str(code_evidence["evidence_digest"]),
        code_private_tree_digest=str(code_evidence["private_tree_digest"]),
    )
    bootstrap_digest = (
        "sha256:"
        + hashlib.sha256(
            (REPOSITORY_ROOT / "research/runpod/bootstrap_server.py").read_bytes()
        ).hexdigest()
    )
    retention: dict[str, object] = {
        "schema_version": 3,
        "evidence_revision": "real-adamw-persist-restore-advance@7",
        "status": "passed",
        "test_id": (
            "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        ),
        "profile_id": manifest["profile_id"],
        "head_commit": HEAD_COMMIT,
        "source_contract_digest": expected_source_contract_digest(manifest),
        "workload_bundle_digest": bundle_metadata["bundle_digest"],
        "workload_bundle_size_bytes": bundle_metadata["bundle_size_bytes"],
        "workload_bundle_path": bundle_metadata["bundle_path"],
        "bundle_stage_receipt_digest": stage_receipt["receipt_digest"],
        "bootstrap_source_digest": bootstrap_digest,
        "volume_readiness_receipt_digest": readiness["receipt_digest"],
        "dependency_quarantine_revision": materialization["dependency_lock"]["revision"],
        "dependency_lock_digest": materialization["dependency_lock"]["digest"],
        "dependency_quarantine_evidence_digest": dependency_evidence["evidence_digest"],
        "dependency_private_tree_digest": dependency_evidence["private_tree_digest"],
        "code_materialization_revision": materialization["code"]["revision"],
        "code_materialization_evidence_digest": code_evidence["evidence_digest"],
        "code_private_tree_digest": code_evidence["private_tree_digest"],
        "network_volume_id": VOLUME_ID,
        "network_volume_data_center_id": DATA_CENTER_ID,
        "network_volume_size_gb": 50,
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": expected_cuda_version(manifest),
        "cuda_available": True,
        "gpu_name": manifest["hardware"]["gpu_id"],
        "gpu_total_memory_bytes": manifest["hardware"]["minimum_cuda_memory_bytes"],
        "bf16_supported": True,
        "probe_sha256": (
            "sha256:" + str(manifest["source_contract"]["files"]["retention_checkpoint_probe.py"])
        ),
        "trainer_sha256": (
            "sha256:"
            + str(manifest["source_contract"]["files"]["repository_repair_large_model_trainer.py"])
        ),
        "environment_sha256": (
            "sha256:" + str(manifest["source_contract"]["files"]["repository_repair_env.py"])
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
            checkpoint_authentication_mechanism_digest()
        ),
        "checkpoint_generation": 1,
        "checkpoint_manifest_digest": "sha256:" + "9" * 64,
        "checkpoint_authenticated_private_resume": True,
    }
    retention["checkpoint_storage_evidence_digest"] = retention_checkpoint_storage_evidence_digest(
        retention
    )
    retention["evidence_digest"] = retention_checkpoint_evidence_digest(retention)
    activation = build_live_stage_activation(
        manifest,
        head_commit=HEAD_COMMIT,
        workload_bundle_digest=str(bundle_metadata["bundle_digest"]),
        workload_bundle_size_bytes=int(bundle_metadata["bundle_size_bytes"]),
        workload_bundle_path=str(bundle_metadata["bundle_path"]),
        bundle_stage_receipt_digest=str(stage_receipt["receipt_digest"]),
        bootstrap_source_digest=bootstrap_digest,
        volume_readiness_receipt_digest=str(readiness["receipt_digest"]),
        torch_retention_evidence_digest=str(retention["evidence_digest"]),
        dependency_lock_digest=str(manifest["materialization"]["dependency_lock"]["digest"]),
        dependency_quarantine_evidence_digest=str(dependency_evidence["evidence_digest"]),
        dependency_private_tree_digest=str(dependency_evidence["private_tree_digest"]),
        code_materialization_evidence_digest=str(code_evidence["evidence_digest"]),
        code_private_tree_digest=str(code_evidence["private_tree_digest"]),
        network_volume_id=VOLUME_ID,
        data_center_id=DATA_CENTER_ID,
        volume_size_gb=50,
    )
    return {
        "manifest_digest": bundle_metadata["manifest_digest"],
        "source_contract_digest": bundle_metadata["source_contract_digest"],
        "bootstrap_source_digest": bootstrap_digest,
        "bundle_metadata": bundle_metadata,
        "stage_receipt": stage_receipt,
        "readiness_receipt": readiness,
        "retention_evidence": retention,
        "dependency_evidence": dependency_evidence,
        "code_evidence": code_evidence,
        "activation": activation,
    }


def _preflight_environment(
    tmp_path: Path,
    binary_directory: Path,
    command_log: Path,
    scenario: dict[str, object],
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("EQUINOX_LARGER_MODEL_", "EQUINOX_RL_", "EQUINOX_RUNPOD_")):
            environment.pop(key)
    receipt = tmp_path / "volume-receipt.json"
    _write_volume_receipt(receipt)
    manifest = load_manifest()
    bundle, bundle_metadata = build_larger_model_bundle(REPOSITORY_ROOT, manifest)
    stage_receipt = stage_bundle_on_mounted_volume(
        bundle,
        manifest,
        mount_root=tmp_path / "mounted-volume",
        volume_id=VOLUME_ID,
        data_center_id=DATA_CENTER_ID,
        volume_size_gb=50,
    )
    stage_receipt_path = tmp_path / "bundle-stage-receipt.json"
    stage_receipt_path.write_bytes(canonical_json(stage_receipt) + b"\n")
    market_price_fixture = tmp_path / "market-price.json"
    market_price_fixture.write_text(
        json.dumps(
            {
                "data": {
                    "gpuTypes": [
                        {
                            "lowestPrice": {
                                "gpuTypeId": "NVIDIA H100 80GB HBM3",
                                "stockStatus": "Low",
                                "uninterruptablePrice": 2.99,
                                "availableGpuCounts": [1],
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    scenario["live_evidence"] = _live_stage_evidence(
        manifest,
        bundle_metadata,
        stage_receipt,
    )
    environment.update(
        {
            "PATH": f"{binary_directory}{os.pathsep}{environment['PATH']}",
            "FAKE_RUNPOD_LOG": str(command_log),
            "FAKE_RUNPOD_PAYLOAD_LOG": str(tmp_path / "observer-payloads.jsonl"),
            "FAKE_RUNPOD_SCENARIO": json.dumps(scenario),
            "EQUINOX_INTERNAL_TOKEN": "test-internal-token-" + "x" * 40,
            "EQUINOX_API_ROOT": "http://operator-test.invalid",
            "EQUINOX_DASHBOARD_ROOT": "http://dashboard-test.invalid",
            "EQUINOX_RESEARCH_PROOF_DIR": str(tmp_path / "research-proofs"),
            "EQUINOX_LARGER_MODEL_MODE": "screen",
            "EQUINOX_LARGER_MODEL_BUNDLE_STAGE_RECEIPT": str(stage_receipt_path),
            "EQUINOX_LARGER_MODEL_VOLUME_RECEIPT": str(receipt),
            "EQUINOX_RUNPOD_OPERATOR_STATE_DIR": str(tmp_path / "operator-state"),
            "EQUINOX_RUNPOD_MARKET_PRICE_FIXTURE": str(market_price_fixture),
            "EQUINOX_RUNPOD_VOLUME_GUARD_SERVICE_MODE": "process",
            "EQUINOX_RUNPOD_PREFLIGHT_ONLY": "1",
            "EQUINOX_RUNPOD_NETWORK_VOLUME_ID": VOLUME_ID,
            "EQUINOX_RUNPOD_DATA_CENTER_IDS": DATA_CENTER_ID,
            "EQUINOX_RUNPOD_GPU": "NVIDIA H100 80GB HBM3",
            "EQUINOX_RUNPOD_CLOUD_TYPE": "SECURE",
            "EQUINOX_RUNPOD_CONTAINER_DISK_GB": "50",
            "EQUINOX_RUNPOD_VOLUME_GB": "50",
            "EQUINOX_RUNPOD_MIN_GPU_MEMORY_GB": "80",
            "EQUINOX_RUNPOD_MAX_HOURLY_COST": "4.00",
            "EQUINOX_RUNPOD_MAX_TOTAL_COST": "3.35",
            "EQUINOX_RUNPOD_MAX_LIFETIME_MINUTES": "48",
            "EQUINOX_RUNPOD_BOOT_TIMEOUT_SECONDS": "600",
            "EQUINOX_RUNPOD_MODEL_LOAD_TIMEOUT_SECONDS": "1200",
            "EQUINOX_RUNPOD_STALE_PROGRESS_SECONDS": "600",
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1",
            "EQUINOX_RUNPOD_RETRY_RESERVE_SECONDS": "300",
            "EQUINOX_RL_MODEL_ID": MODEL_ID,
            "EQUINOX_RL_SEED": "137",
            "EQUINOX_RL_TARGET_SECONDS": "1500",
            "EQUINOX_RL_MAX_UPDATES": "1",
            "EQUINOX_RL_VALIDATION_EXAMPLES": "8",
            "EQUINOX_RL_TEST_EXAMPLES": "0",
            "EQUINOX_RL_MASTERY_WINDOWS": "1",
            "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": "8",
            "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL": "1",
            "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS": "300",
            "EQUINOX_LARGER_MODEL_LIVE_STAGE_REVISION": ("authenticated-proxy-stage-activation@2"),
            "EQUINOX_RUNPOD_CAMPAIGN_BASELINE_BALANCE_USD": "45.2834247943",
            "EQUINOX_RUNPOD_CAMPAIGN_AUTHORIZED_BUDGET_USD": "25.00",
            "EQUINOX_RUNPOD_CAMPAIGN_FUTURE_PILOT_RESERVE_USD": "13.00",
            "EQUINOX_RUNPOD_CAMPAIGN_STORAGE_RESERVE_USD": "0.30",
        }
    )
    return environment


def _run_preflight(
    tmp_path: Path,
    *,
    scenario: dict[str, object] | None = None,
    missing_volume: bool = False,
    missing_receipt: bool = False,
    missing_stage_receipt: bool = False,
    mismatched_stage_receipt: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary_directory, command_log = _install_fake_commands(tmp_path)
    environment = _preflight_environment(
        tmp_path,
        binary_directory,
        command_log,
        scenario or {},
    )
    if missing_volume:
        environment.pop("EQUINOX_RUNPOD_NETWORK_VOLUME_ID")
    if missing_receipt:
        environment["EQUINOX_LARGER_MODEL_VOLUME_RECEIPT"] = str(tmp_path / "missing.json")
    if missing_stage_receipt:
        environment["EQUINOX_LARGER_MODEL_BUNDLE_STAGE_RECEIPT"] = str(
            tmp_path / "missing-stage.json"
        )
    if mismatched_stage_receipt:
        stage_path = Path(environment["EQUINOX_LARGER_MODEL_BUNDLE_STAGE_RECEIPT"])
        stage_receipt = json.loads(stage_path.read_text(encoding="utf-8"))
        stage_receipt["bundle_handoff_revision"] = "old-handoff@1"
        stage_content = {
            key: value for key, value in stage_receipt.items() if key != "receipt_digest"
        }
        stage_receipt["receipt_digest"] = (
            "sha256:" + hashlib.sha256(canonical_json(stage_content)).hexdigest()
        )
        stage_path.write_bytes(canonical_json(stage_receipt) + b"\n")
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    commands = []
    if command_log.is_file():
        commands = (
            [
                json.loads(line)
                for line in command_log.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if command_log.is_file()
            else []
        )
    return result, commands


def _run_launch(
    tmp_path: Path,
    *,
    scenario: dict[str, object],
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary_directory, command_log = _install_fake_commands(tmp_path)
    environment = _preflight_environment(
        tmp_path,
        binary_directory,
        command_log,
        scenario,
    )
    environment.pop("EQUINOX_RUNPOD_PREFLIGHT_ONLY")
    try:
        result = subprocess.run(
            [str(LAUNCHER)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    finally:
        for process_path in (tmp_path / "operator-state" / "volume-expiry-guards").glob(
            "*/guard-process.json"
        ):
            try:
                process = json.loads(process_path.read_text(encoding="utf-8"))
                os.kill(int(process["pid"]), signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError, KeyError, ValueError):
                pass
    commands = [
        json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines() if line
    ]
    return result, commands


def _assert_no_paid_create(commands: list[list[str]]) -> None:
    assert not [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]


def _shell_function(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def _write_volume_guard_state(
    tmp_path: Path,
    binary_directory: Path,
    owner_pid: int,
    *,
    heartbeat_age_seconds: int = 0,
) -> Path:
    root = tmp_path / "guard"
    root.mkdir(mode=0o700)
    now = int(time.time())
    guard_id = "a" * 64
    guard_program = root / "guard-program.py"
    guard_program.write_bytes(VOLUME_GUARD.read_bytes())
    guard_program.chmod(0o500)
    contract = {
        "schema_version": 1,
        "revision": "runpod-network-volume-expiry@1",
        "guard_id": guard_id,
        "screen_proof_id": "runpod-proof-20260730T000000Z-" + "b" * 32,
        "volume_id": VOLUME_ID,
        "data_center_id": DATA_CENTER_ID,
        "size_gb": 50,
        "storage_hourly_cap_usd": 0.01,
        "storage_reserve_usd": 0.30,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": datetime.fromtimestamp(now + 3600, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "expires_at_epoch": now + 3600,
        "runpodctl_path": str(binary_directory / "runpodctl"),
        "guard_program_path": str(guard_program),
        "guard_program_digest": hashlib.sha256(guard_program.read_bytes()).hexdigest(),
    }
    state = {
        "schema_version": 1,
        "revision": "runpod-network-volume-expiry-state@1",
        "guard_id": guard_id,
        "phase": "screen-running",
        "owner_pid": owner_pid,
        "proof_id": contract["screen_proof_id"],
        "pod_id": None,
        "heartbeat_epoch": now - heartbeat_age_seconds,
        "updated_at": contract["created_at"],
    }
    (root / "contract.json").write_bytes(canonical_json(contract) + b"\n")
    (root / "state.json").write_bytes(canonical_json(state) + b"\n")
    return root


def _guard_environment(
    tmp_path: Path,
    binary_directory: Path,
    command_log: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": (
                f"{binary_directory}{os.pathsep}"
                f"{REPOSITORY_ROOT / '.venv/bin'}{os.pathsep}{environment['PATH']}"
            ),
            "FAKE_RUNPOD_LOG": str(command_log),
            "FAKE_RUNPOD_SCENARIO": json.dumps({"spend": 0.005}),
        }
    )
    return environment


def test_volume_guard_deletes_exact_volume_after_sigkill_owner(tmp_path: Path) -> None:
    binary_directory, command_log = _install_fake_commands(tmp_path)
    owner = subprocess.Popen(["/bin/sleep", "60"])
    guard_root = _write_volume_guard_state(tmp_path, binary_directory, owner.pid)
    guard = subprocess.Popen(
        [sys.executable, str(guard_root / "guard-program.py"), str(guard_root)],
        env=_guard_environment(tmp_path, binary_directory, command_log),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=5)
        evidence_path = guard_root / "guard-evidence.json"
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and not evidence_path.is_file():
            time.sleep(0.25)
        guard.wait(timeout=max(1, deadline - time.monotonic()))
        assert guard.returncode == 0, guard.stderr.read() if guard.stderr else ""
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["status"] == "deletion-confirmed"
        assert evidence["volume_id"] == VOLUME_ID
        commands = [
            json.loads(line)
            for line in command_log.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert ["network-volume", "delete", VOLUME_ID] in commands
        assert not [
            command
            for command in commands
            if command[:2] == ["network-volume", "delete"]
            and command != ["network-volume", "delete", VOLUME_ID]
        ]
    finally:
        if owner.poll() is None:
            owner.kill()
        if guard.poll() is None:
            guard.terminate()
            guard.wait(timeout=5)


def test_volume_guard_does_not_delete_during_slow_ambiguous_create(
    tmp_path: Path,
) -> None:
    binary_directory, command_log = _install_fake_commands(tmp_path)
    owner = subprocess.Popen(["/bin/sleep", "60"])
    guard_root = _write_volume_guard_state(
        tmp_path,
        binary_directory,
        owner.pid,
        heartbeat_age_seconds=150,
    )
    guard = subprocess.Popen(
        [sys.executable, str(guard_root / "guard-program.py"), str(guard_root)],
        env=_guard_environment(tmp_path, binary_directory, command_log),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(1)
        commands = (
            [
                json.loads(line)
                for line in command_log.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if command_log.is_file()
            else []
        )
        assert ["network-volume", "delete", VOLUME_ID] not in commands
        assert guard.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)
        guard.terminate()
        guard.wait(timeout=5)


def _monotonic_progress_helper() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index("accept_monotonic_remote_progress() {")
    end = source.index("\nfetch_and_verify_remote_integrity() {", start)
    return source[start:end]


def test_progress_phase_only_toggle_is_cosmetic_and_replay_is_rejected(
    tmp_path: Path,
) -> None:
    common = {
        "attempt": 1,
        "update": 1,
        "elapsed_seconds": 10,
        "branch_groups_completed": 1,
        "total_sampled_actions": 4,
    }
    training = json.dumps(
        {**common, "phase": "training"},
        separators=(",", ":"),
        sort_keys=True,
    )
    checkpointing = json.dumps(
        {**common, "phase": "checkpointing"},
        separators=(",", ":"),
        sort_keys=True,
    )
    harness = f"""set -euo pipefail
progress_floor_path={shlex.quote(str(tmp_path / "progress-floor.json"))}
proof_id=runpod-proof-20260730T000000Z-{"c" * 32}
integrity_generation=0
maximum_workload_attempts=2
maximum_updates=40
maximum_lifetime_minutes=238
maximum_observed_branch_groups=280
maximum_observed_sampled_completions=22400
{_monotonic_progress_helper()}
accept_monotonic_remote_progress {shlex.quote(training)}
[[ "$remote_progress_freshness" == advanced ]]
accept_monotonic_remote_progress {shlex.quote(checkpointing)}
[[ "$remote_progress_freshness" == cosmetic ]]
if accept_monotonic_remote_progress {shlex.quote(training)}; then
  exit 91
fi
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "replayed an older snapshot" in result.stderr


@pytest.mark.parametrize(
    "progress",
    (
        {
            "phase": "eligibility_branch_collection",
            "attempt": 1,
            "branch_groups_completed": 9,
        },
        {
            "phase": "eligibility_branch_collection",
            "attempt": 1,
            "total_sampled_actions": 321,
        },
    ),
    ids=("branch-group-cap", "sampled-action-cap"),
)
def test_progress_rejects_values_above_screen_contract_bounds(
    tmp_path: Path,
    progress: dict[str, object],
) -> None:
    payload = json.dumps(progress, separators=(",", ":"), sort_keys=True)
    harness = f"""set -euo pipefail
progress_floor_path={shlex.quote(str(tmp_path / "progress-floor.json"))}
proof_id=runpod-proof-20260730T000000Z-{"d" * 32}
integrity_generation=0
maximum_workload_attempts=2
maximum_updates=1
maximum_lifetime_minutes=48
maximum_observed_branch_groups=8
maximum_observed_sampled_completions=320
{_monotonic_progress_helper()}
if accept_monotonic_remote_progress {shlex.quote(payload)}; then
  exit 92
fi
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "exceed the immutable workload bound" in result.stderr


def test_terminal_screen_progress_preserves_live_branch_observer_fields() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'finalize_larger_model_screen_progress "$metrics" <<<"$last_progress"' in source
    helper = _shell_function(source, "finalize_larger_model_screen_progress")
    latest_branch_snapshot = {
        "snapshot_id": "update-1-snapshot-root",
        "update": 1,
        "level": 0,
        "siblings": [{"index": index, "steps": []} for index in range(4)],
    }
    progress = {
        "phase": "training",
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "multi_step": True,
        "restored_continuations": True,
        "maximum_level": 3,
        "maximum_updates": 1,
        "current_level": 0,
        "latest_branch_snapshot": latest_branch_snapshot,
    }
    metrics = {
        "branch_width": 4,
        "elapsed_seconds": 93.5,
        "eligible": True,
        "profile_id": load_manifest()["profile_id"],
        "model_revision": "model-revision",
        "branch_checkpoint_rate": 1.0,
        "informative_group_rate": 0.5,
        "peak_reserved_vram_fraction": 0.42,
        "policy_mutation_detected": False,
        "gate_results": {"branch_checkpoint_rate": True},
        "ineligibility_reasons": [],
    }
    harness = f"""set -euo pipefail
{helper}
finalize_larger_model_screen_progress \
  '{json.dumps(metrics, separators=(",", ":"))}' \
  <<'JSON'
{json.dumps(progress, separators=(",", ":"))}
JSON
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    finalized = json.loads(result.stdout)
    assert finalized["phase"] == "finalizing"
    assert finalized["latest_branch_snapshot"] == latest_branch_snapshot
    assert finalized["multi_step"] is True
    assert finalized["restored_continuations"] is True
    assert finalized["maximum_level"] == 3
    assert finalized["maximum_updates"] == 1
    assert finalized["eligible"] is True
    assert finalized["test_split_accessed"] is False


@pytest.mark.parametrize(
    ("scenario", "missing_volume", "missing_receipt", "expected_error"),
    [
        (
            {},
            True,
            False,
            "Larger-model runs require EQUINOX_RUNPOD_NETWORK_VOLUME_ID",
        ),
        (
            {},
            False,
            True,
            "The configured network volume has no local CPU-prewarm readiness receipt.",
        ),
        (
            {"gpu_inventory": {"unexpected": []}},
            False,
            False,
            "RunPod does not currently report an available profile-matching H100 80GB HBM3.",
        ),
        (
            {
                "gpu_inventory": [
                    {
                        "available": True,
                        "communityCloud": False,
                        "displayName": "H100 PCIe",
                        "gpuId": "NVIDIA H100 80GB HBM3",
                        "memoryInGb": 80,
                        "secureCloud": True,
                        "stockStatus": "Medium",
                    }
                ]
            },
            False,
            False,
            "RunPod does not currently report an available profile-matching H100 80GB HBM3.",
        ),
        (
            {"pods": {"unexpected": []}},
            False,
            False,
            "Preflight failed: the RunPod account pod inventory is not empty and valid.",
        ),
        (
            {"spend": 0.011},
            False,
            False,
            "Preflight failed: hourly spend exceeds the permitted idle baseline or could not be verified.",
        ),
        (
            {"pods": [{"id": "pod-active", "name": "another-run"}]},
            False,
            False,
            "Preflight failed: the RunPod account pod inventory is not empty and valid.",
        ),
        (
            {
                "network_volumes": [
                    {
                        "id": VOLUME_ID,
                        "name": "configured",
                        "dataCenterId": DATA_CENTER_ID,
                        "size": 50,
                    },
                    {
                        "id": "network-volume-extra",
                        "name": "unexpected",
                        "dataCenterId": DATA_CENTER_ID,
                        "size": 50,
                    },
                ]
            },
            False,
            False,
            "Preflight failed: the RunPod network-volume inventory is not exactly the verified configured volume.",
        ),
        (
            {"network_volumes": {"items": []}},
            False,
            False,
            "Preflight failed: the RunPod network-volume inventory is not exactly the verified configured volume.",
        ),
    ],
    ids=[
        "missing-volume",
        "missing-receipt",
        "malformed-gpu-inventory",
        "wrong-gpu-display-name",
        "malformed-pod-inventory",
        "active-spend",
        "active-pod",
        "extra-network-volume",
        "malformed-network-volume-inventory",
    ],
)
def test_failed_preflight_never_creates_a_paid_pod(
    tmp_path: Path,
    scenario: dict[str, object],
    missing_volume: bool,
    missing_receipt: bool,
    expected_error: str,
) -> None:
    result, commands = _run_preflight(
        tmp_path,
        scenario=scenario,
        missing_volume=missing_volume,
        missing_receipt=missing_receipt,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert expected_error in result.stderr
    _assert_no_paid_create(commands)


def test_valid_preflight_is_read_only_and_reports_the_pinned_profile(tmp_path: Path) -> None:
    result, commands = _run_preflight(tmp_path, scenario={"spend": 0.005})

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["preflight_passed"] is True
    assert payload["preflight_scope"] == "preallocation_host"
    assert payload["larger_model_mode"] == "screen"
    assert payload["network_volume_id"] == VOLUME_ID
    assert payload["model_id"] == MODEL_ID
    assert payload["image"] == IMAGE_TAG
    assert payload["image_digest"] == IMAGE_DIGEST
    assert payload["optimization_seed"] == 137
    assert payload["workload_bundle_compression"] == "xz"
    assert 0 < payload["workload_bundle_size_bytes"] <= 2 * 1024 * 1024
    assert payload["workload_bundle_digest"].startswith("sha256:")
    assert len(payload["workload_bundle_digest"]) == len("sha256:") + 64
    assert payload["workload_bundle_path"].endswith(
        f"/{payload['workload_bundle_digest'].removeprefix('sha256:')}.tar.xz"
    )
    assert payload["bundle_stage_receipt_digest"] is None
    assert payload["bootstrap_source_digest"].startswith("sha256:")
    assert payload["bundle_handoff_revision"] is None
    assert payload["live_stage_activation_revision"] == ("authenticated-proxy-stage-activation@2")
    assert payload["source_head_commit"] == HEAD_COMMIT
    _assert_no_paid_create(commands)


def test_screen_cannot_disable_authenticated_live_staging(tmp_path: Path) -> None:
    binary_directory, command_log = _install_fake_commands(tmp_path)
    environment = _preflight_environment(
        tmp_path,
        binary_directory,
        command_log,
        {"spend": 0.005},
    )
    environment.pop("EQUINOX_LARGER_MODEL_LIVE_STAGE_REVISION")

    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    commands = [
        json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines() if line
    ]

    assert result.returncode != 0
    assert "requires the immutable authenticated live-stage activation" in result.stderr
    _assert_no_paid_create(commands)


def test_campaign_guard_applies_the_full_pilot_cap_without_live_stage() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "provider_live_stage_campaign_is_allowed")
    harness = f"""set -euo pipefail
{helper}
larger_model_mode=pilot
campaign_baseline_balance_usd=45.2834247943
campaign_authorized_budget_usd=25.00
campaign_future_pilot_reserve_usd=0.00
campaign_storage_reserve_usd=0.30
maximum_total_cost=13.00
maximum_storage_only_hourly_spend=0.01
account_balance=33.5834247943
run_provider_command() {{
  printf '{{"clientBalance":%s,"currentSpendPerHr":0.005}}\\n' "$account_balance"
}}
provider_live_stage_campaign_is_allowed
account_balance=33.58
if provider_live_stage_campaign_is_allowed; then
  exit 91
fi
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_pilot_environment_carries_the_authorized_live_handoff_identity() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index('elif [[ -n "$larger_model_mode" ]]; then\n  pod_environment="$(')
    end = source.index('\nelse\n  pod_environment="$(', start)
    pilot_environment = source[start:end]

    for name in (
        "EQUINOX_LARGER_MODEL_PROFILE_ID",
        "EQUINOX_SOURCE_HEAD_COMMIT",
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
        "EQUINOX_VOLUME_READINESS_RECEIPT_SHA256",
        "EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_ID",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB",
        "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
    ):
        assert name in pilot_environment
    assert 'bundle_activation_digest="$authorized_live_stage_activation_digest"' in source
    assert '"$authorized_source_head_commit" != "$source_head_commit"' in source
    assert "verify-retention-evidence" in source


@pytest.mark.parametrize(
    ("missing", "mismatched", "message"),
    [
        (True, False, "no local workload-bundle stage receipt"),
        (False, True, "does not match the profile or provider volume"),
    ],
)
def test_explicit_live_stage_does_not_trust_an_unborn_local_stage_receipt(
    tmp_path: Path,
    missing: bool,
    mismatched: bool,
    message: str,
) -> None:
    result, commands = _run_preflight(
        tmp_path,
        scenario={"spend": 0.005},
        missing_stage_receipt=missing,
        mismatched_stage_receipt=mismatched,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert message not in result.stderr
    _assert_no_paid_create(commands)


@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    [
        (
            {"image_index": _image_index("sha256:" + "0" * 64)},
            "registry linux/amd64 image digest does not match",
        ),
        (
            {"image_inspection_error": True},
            "pinned container image could not be inspected",
        ),
        (
            {"image_index": _image_index(IMAGE_DIGEST, IMAGE_DIGEST)},
            "must return exactly one linux/amd64 image manifest",
        ),
    ],
    ids=("digest-mismatch", "inspection-failure", "ambiguous-amd64"),
)
def test_image_verification_failure_never_creates_a_paid_pod(
    tmp_path: Path,
    scenario: dict[str, object],
    expected_error: str,
) -> None:
    result, commands = _run_preflight(tmp_path, scenario=scenario)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        IMAGE_TAG,
        "--format",
        "{{json .Manifest}}",
    ] in commands
    _assert_no_paid_create(commands)


@pytest.mark.parametrize(
    "second_resolution",
    (_image_index("sha256:" + "0" * 64), None),
    ids=("digest-drift", "inspection-failure"),
)
def test_image_reverification_failure_immediately_before_create_never_allocates(
    tmp_path: Path,
    second_resolution: object,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "image_indexes": [_image_index(IMAGE_DIGEST), second_resolution],
        },
    )

    assert result.returncode != 0
    assert "Refusing to allocate: the pinned container image could not be reverified." in (
        result.stderr
    )
    _assert_no_paid_create(commands)
    inspections = [command for command in commands if command[:1] == ["docker"]]
    assert len(inspections) == 2


def test_digest_pinned_create_needs_no_mutable_tag_check_after_allocation(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "pod_image": IMAGE_TAG,
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
        },
    )

    assert result.returncode != 0
    paid_creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(paid_creates) == 1
    assert paid_creates[0][paid_creates[0].index("--image") + 1] == (f"{IMAGE_TAG}@{IMAGE_DIGEST}")
    inspections = [command for command in commands if command[:1] == ["docker"]]
    assert len(inspections) == 2


def test_provider_image_attestation_mismatch_deletes_the_exact_pod(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "pod_image": IMAGE_TAG,
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
        },
    )

    assert result.returncode != 0
    assert "did not report the exact requested image and volume attachment" in result.stderr
    assert ["pod", "delete", "fake-paid-pod"] in commands


def test_larger_model_launch_allows_only_the_pinned_storage_baseline(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(tmp_path, scenario={"spend": 0.005})

    assert result.returncode != 0
    assert "Creating one bounded RunPod worker" in result.stderr
    paid_creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(paid_creates) == 1
    assert "--network-volume-id" in paid_creates[0]
    assert VOLUME_ID in paid_creates[0]
    assert paid_creates[0][paid_creates[0].index("--image") + 1] == (f"{IMAGE_TAG}@{IMAGE_DIGEST}")


def test_paid_create_carries_only_digest_verified_volume_bundle_identity(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
        },
    )

    assert result.returncode != 0
    assert "Container is live; streaming structured progress" in result.stderr
    paid_create_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(paid_create_indexes) == 1
    create = commands[paid_create_indexes[0]]
    assert create[create.index("--image") + 1] == f"{IMAGE_TAG}@{IMAGE_DIGEST}"
    assert create[create.index("--gpu-id") + 1] == "NVIDIA H100 80GB HBM3"
    assert create[create.index("--gpu-count") + 1] == "1"
    assert create[create.index("--cloud-type") + 1] == "SECURE"
    assert create[create.index("--ports") + 1] == "8000/http"
    assert create[create.index("--ssh=false")] == "--ssh=false"
    serialized_environment = create[create.index("--env") + 1]
    assert len(serialized_environment.encode()) <= 4 * 1024
    create_environment = json.loads(create[create.index("--env") + 1])
    assert set(create_environment) == {
        "EQUINOX_RESULT_TOKEN",
        "EQUINOX_BUNDLE_LIVE_STAGE",
        "EQUINOX_BUNDLE_VOLUME_PATH",
        "EQUINOX_BUNDLE_SHA256",
        "EQUINOX_BUNDLE_SIZE_BYTES",
        "EQUINOX_BOOTSTRAP_SOURCE_SHA256",
        "EQUINOX_LARGER_MODEL_PROFILE_ID",
        "EQUINOX_LARGER_MODEL_MANIFEST_SHA256",
        "EQUINOX_SOURCE_HEAD_COMMIT",
        "EQUINOX_SOURCE_CONTRACT_SHA256",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_ID",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID",
        "EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB",
        "EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH",
    }
    assert len(create_environment["EQUINOX_RESULT_TOKEN"]) == 64
    assert create_environment["EQUINOX_BUNDLE_VOLUME_PATH"].endswith(
        "/" + create_environment["EQUINOX_BUNDLE_SHA256"].removeprefix("sha256:") + ".tar.xz"
    )
    assert int(create_environment["EQUINOX_BUNDLE_SIZE_BYTES"]) > 0
    assert create_environment["EQUINOX_BUNDLE_LIVE_STAGE"] == "1"
    assert (
        0
        < int(create_environment["EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH"])
        - int(datetime.now(UTC).timestamp())
        <= 600
    )
    terminate_after = datetime.fromisoformat(
        create[create.index("--terminate-after") + 1].replace("Z", "+00:00")
    )
    assert (
        int(terminate_after.timestamp())
        - int(create_environment["EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH"])
        == 38 * 60
    )
    assert create_environment["EQUINOX_SOURCE_HEAD_COMMIT"] == HEAD_COMMIT
    assert create_environment["EQUINOX_BOOTSTRAP_SOURCE_SHA256"].startswith("sha256:")
    assert "EQUINOX_BUNDLE_B64" not in create_environment
    assert "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256" not in create_environment
    docker_arguments = create[create.index("--docker-args") + 1]
    assert "python3 -I -S -c" in docker_arguments
    assert 'getattr(os,"O_NOFOLLOW",0)' in docker_arguments
    assert "/proc/self/fd/{descriptor}" in docker_arguments
    assert "/tmp/equinox-bootstrap-" in docker_arguments
    assert f"{create_environment['EQUINOX_BUNDLE_VOLUME_PATH']}" not in docker_arguments
    assert (
        create_environment["EQUINOX_BOOTSTRAP_SOURCE_SHA256"].removeprefix("sha256:")
        in docker_arguments
    )
    observer_payloads = [
        json.loads(line)
        for line in (tmp_path / "observer-payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    first_provisioning = next(
        payload for payload in observer_payloads if payload["status"] == "PROVISIONING"
    )
    assert first_provisioning["artifact_publication_required"] is True
    assert "artifact_publication_required" not in first_provisioning["resource_profile"]
    assert first_provisioning["resource_profile"]["profile_id"] == PROFILE_ID
    assert first_provisioning["artifact_publication"] is None

    image_precheck_index = max(
        index for index, command in enumerate(commands) if command[:1] == ["docker"]
    )
    post_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"]
        and "-X" in command
        and command[command.index("-X") + 1] == "POST"
    ]
    progress_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"] and command[-1].endswith("/progress.json")
    ]
    bootstrap_health_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"] and command[-1].endswith("/bootstrap-health")
    ]
    assert len(post_indexes) == 2
    bundle_post = commands[post_indexes[0]]
    activation_post = commands[post_indexes[1]]
    assert "Content-Type: application/x-xz" in bundle_post
    assert f"Content-Length: {create_environment['EQUINOX_BUNDLE_SIZE_BYTES']}" in bundle_post
    assert "Content-Type: application/json" in activation_post
    assert sum(command[-1].endswith("/bundle") for command in (bundle_post, activation_post)) == 1
    assert bootstrap_health_indexes
    assert (
        image_precheck_index
        < paid_create_indexes[0]
        < bootstrap_health_indexes[0]
        < post_indexes[0]
        < post_indexes[1]
        < progress_indexes[0]
    )
    assert create_environment["EQUINOX_RESULT_TOKEN"] not in result.stdout
    assert create_environment["EQUINOX_RESULT_TOKEN"] not in result.stderr
    assert create_environment["EQUINOX_BUNDLE_VOLUME_PATH"] not in result.stdout
    assert create_environment["EQUINOX_BUNDLE_VOLUME_PATH"] not in result.stderr
    observer_payloads = (tmp_path / "observer-payloads.jsonl").read_text(encoding="utf-8")
    assert create_environment["EQUINOX_RESULT_TOKEN"] not in observer_payloads
    observed_resource_profiles = [
        json.loads(line)["resource_profile"] for line in observer_payloads.splitlines() if line
    ]
    assert any(
        profile.get("workload_bundle_digest") == create_environment["EQUINOX_BUNDLE_SHA256"]
        and profile.get("workload_bundle_size_bytes")
        == int(create_environment["EQUINOX_BUNDLE_SIZE_BYTES"])
        and profile.get("workload_bundle_compression") == "xz"
        and profile.get("workload_bundle_path") == create_environment["EQUINOX_BUNDLE_VOLUME_PATH"]
        and str(profile.get("bundle_stage_receipt_digest", "")).startswith("sha256:")
        and profile.get("bootstrap_source_digest")
        == create_environment["EQUINOX_BOOTSTRAP_SOURCE_SHA256"]
        and profile.get("bundle_handoff_revision") == "runpod-volume-bundle-handoff@1"
        and profile.get("source_head_commit") == create_environment["EQUINOX_SOURCE_HEAD_COMMIT"]
        and str(profile.get("torch_retention_evidence_digest", "")).startswith("sha256:")
        for profile in observed_resource_profiles
    )

    delete_indexes = [
        index for index, command in enumerate(commands) if command[:2] == ["pod", "delete"]
    ]
    assert delete_indexes
    assert all(commands[index] == ["pod", "delete", "fake-paid-pod"] for index in delete_indexes)
    assert progress_indexes[0] < min(delete_indexes)
    first_delete = min(delete_indexes)
    idle_polls_after_delete = [
        command for command in commands[first_delete + 1 :] if command == ["pod", "list", "--all"]
    ]
    assert len(idle_polls_after_delete) >= 6


def test_first_remote_progress_must_attest_the_exact_staged_handoff(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
            "progress_responses": [
                {
                    "phase": "container_starting",
                    "workload_bundle_digest": "sha256:" + "0" * 64,
                }
            ],
        },
    )

    assert result.returncode != 0
    assert "did not match the activated workload evidence" in result.stderr
    assert ["pod", "delete", "fake-paid-pod"] in commands


def test_exit_visibility_cannot_drop_the_final_remote_progress(tmp_path: Path) -> None:
    branch_snapshots = [
        {
            "snapshot_id": f"screen-branch-{index}",
            "update": index + 1,
            "level": 0,
            "siblings": [{"index": sibling, "steps": []} for sibling in range(4)],
        }
        for index in range(8)
    ]
    progress_at_readiness = {
        "phase": "screen_branching",
        "branch_groups_completed": 7,
        "branch_groups_total": 8,
        "branch_snapshots": branch_snapshots[:7],
    }
    final_progress = {
        "phase": "screen_complete",
        "branch_groups_completed": 8,
        "branch_groups_total": 8,
        "branch_snapshots": branch_snapshots,
        "latest_branch_snapshot": branch_snapshots[-1],
    }
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
            "progress_responses": [
                progress_at_readiness,
                None,
                None,
                None,
                final_progress,
            ],
            "exit_code_responses": ["1"],
        },
    )

    assert result.returncode != 0
    progress_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"]
        and any(argument.endswith("/progress.json") for argument in command[1:])
    ]
    exit_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"]
        and any(argument.endswith("/exit_code") for argument in command[1:])
    ]
    assert len(progress_indexes) == 5
    assert len(exit_indexes) == 1
    assert progress_indexes[1] < exit_indexes[0] < progress_indexes[2]

    payload_log = tmp_path / "observer-payloads.jsonl"
    payloads = [
        json.loads(line) for line in payload_log.read_text(encoding="utf-8").splitlines() if line
    ]
    terminal_tree_payloads = [
        payload for payload in payloads if payload["progress"].get("branch_groups_completed") == 8
    ]
    assert terminal_tree_payloads
    assert terminal_tree_payloads[0]["progress"]["branch_snapshots"] == branch_snapshots
    assert terminal_tree_payloads[0]["progress"]["latest_branch_snapshot"] == branch_snapshots[-1]


def test_failed_remote_workload_persists_only_operational_failure_evidence(
    tmp_path: Path,
) -> None:
    validation_history = [
        {
            "update": 5,
            "level": 0,
            "fixed_guard_paired_change": {"improved": 0, "regressed": 1},
        }
    ]
    branch_snapshots = [
        {
            "snapshot_id": "update-5-snapshot-a",
            "siblings": [
                {
                    "index": sibling,
                    "steps": [{"step_id": f"sibling-{sibling}-1", "tool": "edit"}],
                }
                for sibling in range(4)
            ],
        }
    ]
    exact_remote_code = "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE"
    exact_remote_message = "No policy-bearing checkpoint passed retention."
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
            "progress_responses": [
                {
                    "phase": "training",
                    "message": "Collecting K=4 continuations.",
                    "validation_history": validation_history,
                    "branch_snapshots": branch_snapshots,
                    "latest_branch_snapshot": branch_snapshots[-1],
                },
                {
                    "phase": "failed",
                    "message": exact_remote_message,
                    "error": exact_remote_code,
                    "validation_history": validation_history,
                    "branch_snapshots": branch_snapshots,
                    "latest_branch_snapshot": branch_snapshots[-1],
                },
            ],
            "exit_code_responses": ["1"],
        },
    )

    assert result.returncode != 0
    receipt_paths = list((tmp_path / "research-proofs").glob("*.failure.json"))
    assert len(receipt_paths) == 1
    receipt_path = receipt_paths[0]
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    digest = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()

    assert receipt["receipt_kind"] == "operational_failure@1"
    assert receipt["scientific_proof"] is False
    assert receipt["proof_id"] is None
    assert receipt["provider"]["handle"] == "runpod://pods/fake-paid-pod"
    assert receipt["teardown_confirmed"] is True
    assert receipt["remote_error"] == {
        "code": exact_remote_code,
        "message": exact_remote_message,
    }
    assert receipt["operator_error"]["message"] == ("The remote workload failed with exit code 1.")
    assert receipt["last_observed_progress"]["validation_history"] == validation_history
    assert receipt["last_observed_progress"]["branch_snapshots"] == branch_snapshots
    assert receipt["handoff"]["workload_bundle_digest"].startswith("sha256:")
    assert receipt["cost"]["estimated"] is True

    payloads = [
        json.loads(line)
        for line in (tmp_path / "observer-payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    failed_payload = next(
        payload for payload in reversed(payloads) if payload["status"] == "FAILED"
    )
    assert failed_payload["failure_receipt_digest"] == digest
    assert failed_payload["progress"]["error"] == exact_remote_code
    assert failed_payload["progress"]["remote_error"]["code"] == exact_remote_code
    assert failed_payload["progress"]["remote_error"]["message"] == exact_remote_message
    assert failed_payload["progress"]["operator_error"]["message"] == (
        "The remote workload failed with exit code 1."
    )
    assert failed_payload["progress"]["branch_snapshots"] == branch_snapshots
    assert "proof_id" not in failed_payload
    transport_posts = [
        command
        for command in commands
        if command[:1] == ["curl"]
        and "-X" in command
        and command[command.index("-X") + 1] == "POST"
    ]
    assert len(transport_posts) == 2
    assert any(command[-1].endswith("/bundle") for command in transport_posts)
    assert any(command[-1].endswith("/activate-staged-bundle") for command in transport_posts)


def test_operational_failure_receipt_creation_is_exact_and_conflict_safe(
    tmp_path: Path,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "persist_operational_failure_receipt")
    execution_id = "runpod-proof-receipt-test"
    receipt = {
        "receipt_kind": "operational_failure@1",
        "scientific_proof": False,
        "proof_id": None,
        "execution_id": execution_id,
        "execution_name": "Atomic receipt test",
        "provider": {
            "name": "RunPod",
            "handle": "runpod://pods/receipt-test",
            "cli_version": "2.7.2",
        },
        "resource_profile": {"gpu_id": "NVIDIA H100 80GB HBM3"},
        "handoff": {"revision": "mounted-volume-bundle@1"},
        "workload": {
            "id": "repository-repair",
            "model_id": MODEL_ID,
            "static_branch_width": 4,
            "complexity_strategy": "adaptive",
        },
        "last_observed_progress": {
            "phase": "failed",
            "message": "é",
            "branch_snapshots": [{"snapshot_id": "snapshot-a"}],
        },
        "remote_error": {
            "code": "REMOTE_WORKLOAD_FAILURE",
            "message": "RuntimeError: exact",
        },
        "operator_error": {
            "code": "RUNPOD_OPERATOR_FAILURE",
            "message": "Observed remote exit.",
        },
        "cost": {
            "total_usd": 0.1,
            "estimated": True,
            "hourly_rate_usd": 2.99,
            "provider_runtime_seconds": 120,
        },
        "started_at": "2026-07-29T15:00:00Z",
        "completed_at": "2026-07-29T15:02:00Z",
        "teardown_confirmed": False,
    }
    conflict = {
        **receipt,
        "operator_error": {
            "code": "RUNPOD_OPERATOR_FAILURE",
            "message": "Different bytes must conflict.",
        },
    }
    harness = f"""set -euo pipefail
repository_root='{REPOSITORY_ROOT}'
receipt_directory='{tmp_path}'
{helper}
first="$(persist_operational_failure_receipt \
  '{execution_id}' \
  '{json.dumps(receipt, separators=(",", ":"))}')"
second="$(persist_operational_failure_receipt \
  '{execution_id}' \
  '{json.dumps(receipt, separators=(",", ":"))}')"
test "$first" = "$second"
if persist_operational_failure_receipt \
  '{execution_id}' \
  '{json.dumps(conflict, separators=(",", ":"))}'; then
  exit 91
fi
printf '%s\\n' "$first"
"""

    completed = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{REPOSITORY_ROOT / '.venv/bin'}:{os.environ['PATH']}",
        },
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt_path = tmp_path / f"{execution_id}.failure.json"
    assert completed.stdout.strip() == (
        "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    assert json.loads(receipt_path.read_bytes()) == receipt
    assert not list(tmp_path.glob(".*.tmp"))
    assert "tempfile.mkstemp" in helper
    assert "os.link(" in helper

    partial_id = "runpod-proof-partial-receipt"
    partial_receipt = {**receipt, "execution_id": partial_id}
    partial_path = tmp_path / f"{partial_id}.failure.json"
    partial_bytes = b'{"receipt_kind":"operational_failure@1"'
    partial_path.write_bytes(partial_bytes)
    partial_harness = f"""set -euo pipefail
repository_root='{REPOSITORY_ROOT}'
receipt_directory='{tmp_path}'
{helper}
persist_operational_failure_receipt \
  '{partial_id}' \
  '{json.dumps(partial_receipt, separators=(",", ":"))}'
"""
    partial = subprocess.run(
        ["bash", "-c", partial_harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{REPOSITORY_ROOT / '.venv/bin'}:{os.environ['PATH']}",
        },
    )
    assert partial.returncode != 0
    assert partial_path.read_bytes() == partial_bytes

    symlink_id = "runpod-proof-symlink-receipt"
    symlink_receipt = {**receipt, "execution_id": symlink_id}
    symlink_target = tmp_path / "outside-receipt.json"
    symlink_target.write_bytes(canonical_json(symlink_receipt))
    symlink_path = tmp_path / f"{symlink_id}.failure.json"
    symlink_path.symlink_to(symlink_target)
    symlink_harness = f"""set -euo pipefail
repository_root='{REPOSITORY_ROOT}'
receipt_directory='{tmp_path}'
{helper}
persist_operational_failure_receipt \
  '{symlink_id}' \
  '{json.dumps(symlink_receipt, separators=(",", ":"))}'
"""
    symlink = subprocess.run(
        ["bash", "-c", symlink_harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{REPOSITORY_ROOT / '.venv/bin'}:{os.environ['PATH']}",
        },
    )
    assert symlink.returncode != 0
    assert symlink_path.is_symlink()
    assert symlink_target.read_bytes() == canonical_json(symlink_receipt)

    fifo_id = "runpod-proof-fifo-receipt"
    fifo_receipt = {**receipt, "execution_id": fifo_id}
    fifo_path = tmp_path / f"{fifo_id}.failure.json"
    os.mkfifo(fifo_path)
    fifo_harness = f"""set -euo pipefail
repository_root='{REPOSITORY_ROOT}'
receipt_directory='{tmp_path}'
{helper}
persist_operational_failure_receipt \
  '{fifo_id}' \
  '{json.dumps(fifo_receipt, separators=(",", ":"))}'
"""
    fifo = subprocess.run(
        ["bash", "-c", fifo_harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{REPOSITORY_ROOT / '.venv/bin'}:{os.environ['PATH']}",
        },
    )
    assert fifo.returncode != 0
    assert fifo_path.is_fifo()


def test_stale_execution_reconciliation_persists_the_same_failure_contract(
    tmp_path: Path,
) -> None:
    stale_id = "runpod-proof-stale-observer"
    exact_remote_code = "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE"
    exact_remote_message = "No policy-bearing checkpoint passed retention."
    stale_execution = {
        "execution_id": stale_id,
        "name": "Interrupted H100 pilot",
        "workload_id": "repository-repair-restored-continuation-post-training",
        "model_id": MODEL_ID,
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "status": "RUNNING",
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/stale-pod",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "hourly_cost_usd": 2.99,
            "maximum_lifetime_minutes": 43,
            "network_volume_id": VOLUME_ID,
        },
        "progress": {
            "phase": "failed",
            "message": exact_remote_message,
            "error": exact_remote_code,
            "validation_history": [{"update": 5, "retention_guard_passed": False}],
            "branch_snapshots": [{"snapshot_id": "update-5-snapshot-a"}],
            "elapsed_seconds": 1200,
        },
        "artifact_publication_required": True,
        "artifact_publication": None,
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
        "started_at": "2026-07-29T15:00:00Z",
        "updated_at": "2026-07-29T15:20:00Z",
        "completed_at": None,
        "teardown_confirmed": False,
    }
    result, _commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "research_executions": {"items": [stale_execution]},
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
        },
    )

    assert result.returncode != 0
    receipt_path = tmp_path / "research-proofs" / f"{stale_id}.failure.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    digest = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    assert receipt["receipt_kind"] == "operational_failure@1"
    assert receipt["scientific_proof"] is False
    assert receipt["proof_id"] is None
    assert receipt["last_observed_progress"] == stale_execution["progress"]
    assert receipt["remote_error"]["code"] == exact_remote_code
    assert receipt["remote_error"]["message"] == exact_remote_message
    assert receipt["operator_error"] == {
        "code": "STALE_OPERATOR_RECONCILIATION",
        "message": "No RunPod pod or active hourly spend remained.",
        "teardown_error": None,
    }
    assert receipt["cost"]["provider_runtime_seconds"] == 1200
    assert receipt["cost"]["total_usd"] == pytest.approx(2.99 / 3)
    assert receipt["teardown_confirmed"] is True

    payloads = [
        json.loads(line)
        for line in (tmp_path / "observer-payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    stale_payload = next(
        payload
        for payload in payloads
        if payload.get("name") == stale_execution["name"] and payload.get("status") == "FAILED"
    )
    assert stale_payload["failure_receipt_digest"] == digest
    assert stale_payload["progress"]["error"] == exact_remote_code
    assert stale_payload["progress"]["remote_error"]["code"] == exact_remote_code
    assert stale_payload["progress"]["remote_error"]["message"] == exact_remote_message
    assert stale_payload["progress"]["operator_error"]["code"] == ("STALE_OPERATOR_RECONCILIATION")
    assert stale_payload["artifact_publication_required"] is True
    assert stale_payload["artifact_publication"] is None
    assert (
        stale_payload["progress"]["branch_snapshots"]
        == (stale_execution["progress"]["branch_snapshots"])
    )


def test_stale_reconciliation_reuses_an_orphan_failure_receipt_exactly(
    tmp_path: Path,
) -> None:
    stale_id = "runpod-proof-orphaned-receipt"
    progress = {
        "phase": "failed",
        "message": "Exact prior progress.",
        "validation_history": [{"update": 11, "retention_guard_passed": False}],
        "branch_snapshots": [{"snapshot_id": "update-11-snapshot-a"}],
    }
    orphan_receipt = {
        "receipt_kind": "operational_failure@1",
        "scientific_proof": False,
        "proof_id": None,
        "execution_id": stale_id,
        "execution_name": "Orphaned failure evidence",
        "provider": {
            "name": "RunPod",
            "handle": "runpod://pods/orphaned-pod",
            "cli_version": "2.7.2",
        },
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "hourly_cost_usd": 2.99,
        },
        "handoff": {"revision": "mounted-volume-bundle@1"},
        "workload": {
            "id": "repository-repair-restored-continuation-post-training",
            "model_id": MODEL_ID,
            "static_branch_width": 4,
            "complexity_strategy": "adaptive",
        },
        "last_observed_progress": progress,
        "remote_error": {
            "code": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
            "message": "No policy-bearing checkpoint passed retention.",
        },
        "operator_error": {
            "code": "RUNPOD_OPERATOR_FAILURE",
            "message": "The observer API did not accept the terminal update.",
        },
        "cost": {
            "total_usd": 1.495,
            "estimated": True,
            "hourly_rate_usd": 2.99,
            "provider_runtime_seconds": 1800,
        },
        "started_at": "2026-07-29T15:00:00Z",
        "completed_at": "2026-07-29T15:30:00Z",
        "teardown_confirmed": False,
    }
    receipt_directory = tmp_path / "research-proofs"
    receipt_directory.mkdir()
    receipt_path = receipt_directory / f"{stale_id}.failure.json"
    orphan_bytes = canonical_json(orphan_receipt)
    receipt_path.write_bytes(orphan_bytes)
    stale_execution = {
        "execution_id": stale_id,
        "name": "Orphaned failure evidence",
        "workload_id": "repository-repair-restored-continuation-post-training",
        "model_id": MODEL_ID,
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "status": "RUNNING",
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/orphaned-pod",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "hourly_cost_usd": 2.99,
        },
        "progress": {"phase": "running", "update": 12},
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
        "started_at": "2026-07-29T15:00:00Z",
        "updated_at": "2026-07-29T15:20:00Z",
        "completed_at": None,
        "teardown_confirmed": False,
    }

    result, _commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "research_executions": {"items": [stale_execution]},
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
        },
    )

    assert result.returncode != 0
    assert receipt_path.read_bytes() == orphan_bytes
    digest = "sha256:" + hashlib.sha256(orphan_bytes).hexdigest()
    payloads = [
        json.loads(line)
        for line in (tmp_path / "observer-payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    stale_payload = next(
        payload
        for payload in payloads
        if payload.get("name") == stale_execution["name"] and payload.get("status") == "FAILED"
    )
    assert stale_payload["failure_receipt_digest"] == digest
    assert stale_payload["teardown_confirmed"] is False
    assert stale_payload["progress"]["teardown_confirmed"] is False
    assert stale_payload["progress"]["validation_history"] == (progress["validation_history"])
    assert stale_payload["progress"]["operator_error"] == orphan_receipt["operator_error"]


def test_success_publication_sets_the_cleanup_fence_immediately() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert (
        'publish_execution "SUCCEEDED" "$last_progress" "$completed_at" "true"\n'
        "  terminal_published=true"
    ) in source
    proof_acceptance = source.index(
        'if proof_response="$(\n      curl --fail --silent --show-error --max-time 60'
    )
    proof_fence = source.index("terminal_published=true", proof_acceptance)
    proof_output = source.index('jq . <<<"$proof_response"', proof_acceptance)
    assert proof_acceptance < proof_fence < proof_output


def test_screen_success_is_fenced_on_one_post_teardown_artifact_commit() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    teardown = source.index(
        "if ! confirm_provider_teardown; then", source.index("Remote loop passed")
    )
    publish_command = "python3 -m research.runpod.artifact_publication \\\n      publish"
    publish = source.index(publish_command)
    success = source.index(
        'publish_execution "SUCCEEDED" "$last_progress" "$completed_at" "true"',
        publish,
    )

    assert source.count(publish_command) == 1
    assert teardown < publish < success
    assert 'bundle_stage_receipt_path="$artifact_staging_directory/bundle-receipt.json"' in source
    assert 'result_path="$artifact_staging_directory/screen-result.json"' in source
    assert 'receipt_path="$artifact_staging_directory/provider-receipt.json"' in source
    assert "schema_version: 2" in source[teardown:publish]
    assert "run_result: $run_result" in source[teardown:publish]
    assert 'artifact_publication="$(' in source[publish:success]
    assert "artifact_set_committed: true" not in source[publish:success]
    assert "artifact_set_manifest_digest: $manifest_digest" in source[publish:success]
    assert "artifact_publication: $artifact_publication" in source
    assert 'artifact_publication_required: ($larger_model_mode != "")' in source[:teardown]


def test_pilot_publication_is_one_private_typed_transaction() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    artifact_block_start = source.index(
        'if [[ -n "$larger_model_mode" ]]; then\n'
        '  attempt_record_path="$artifact_staging_directory/attempt-record.json"'
    )
    artifact_block_end = source.index(
        '\nfi\n\nif [[ "$larger_model_mode" == "screen" ]]; then\n  last_progress="$(',
        artifact_block_start,
    )
    artifact_block = source[artifact_block_start:artifact_block_end]

    assert 'adapter_path="$artifact_staging_directory/model-artifact.tgz"' in source
    assert 'result_path="$artifact_staging_directory/pilot-result.json"' in source
    assert "schema_version: 2" in artifact_block
    assert "publication_type: $publication_type" in artifact_block
    assert "source_screen: $source_screen" in artifact_block
    assert "model_artifact: $model_artifact" in artifact_block
    assert "source_screen_publication_id: $source_publication_id" in artifact_block
    assert "source_screen_manifest_digest: $source_manifest_digest" in artifact_block
    assert "model_artifact_sha256: $model_artifact_digest" in artifact_block
    assert "model_artifact_size_bytes: $model_artifact_size" in artifact_block
    assert "model_artifact_sha256: $model_artifact_sha256" in artifact_block
    assert "model_artifact_size_bytes: $model_artifact_size_bytes" in artifact_block
    assert "artifact_set_committed: true" not in artifact_block
    assert "artifact_set_manifest_digest: $manifest_digest" in artifact_block


def test_pilot_model_artifact_identity_is_read_through_a_stable_nofollow_fd(
    tmp_path: Path,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    marker = 'model_artifact_identity="$(\n      python3 - "$adapter_path" <<\'PY\'\n'
    start = source.index(marker) + len(marker)
    end = source.index('\nPY\n    )"', start)
    identity_code = source[start:end]
    artifact = tmp_path / "model-artifact.tgz"
    payload = b"verified adapter bytes"
    artifact.write_bytes(payload)

    verified = subprocess.run(
        ["python3", "-c", identity_code, str(artifact)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert verified.returncode == 0, verified.stdout + verified.stderr
    identity = json.loads(verified.stdout)
    assert identity == {
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

    target = tmp_path / "attacker.tgz"
    target.write_bytes(payload)
    artifact.unlink()
    artifact.symlink_to(target)
    rejected = subprocess.run(
        ["python3", "-c", identity_code, str(artifact)],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert rejected.returncode != 0


def test_committed_provider_receipt_is_exactly_the_envelope_free_proof_body() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index('if ! jq -e \\\n  --argjson result "$metrics"')
    end = source.index('\n\nif [[ -n "$larger_model_mode" ]]; then', start)
    receipt_guard = source[start:end]

    for key in (
        "completed_at",
        "provider_cli_version",
        "provider_handle",
        "provider_name",
        "resource_profile",
        "result",
        "started_at",
        "teardown_confirmed",
        "workload",
    ):
        assert f'"{key}"' in receipt_guard
    assert ".result == $result" in receipt_guard
    assert ".resource_profile.profile_id == $profile_id" in receipt_guard
    assert '"artifact_publication"' not in receipt_guard
    assert '"result_digest"' not in receipt_guard


def test_pilot_keeps_negative_scientific_outcomes_durable_and_distinct() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    pilot_assertion = source[
        source.index(
            'if [[ "$larger_model_mode" == "pilot" ]]; then', source.index("proof_assertion=")
        ) :
    ]

    assert ".meaningful_post_training == .hypothesis_passed" in pilot_assertion
    assert ".post_training_outcome == (" in pilot_assertion
    assert '"MEANINGFUL_POST_TRAINING"' in pilot_assertion
    assert '"NEGATIVE_EXPERIMENT_COMPLETED"' in pilot_assertion
    assert '"INCONCLUSIVE_EXPERIMENT_COMPLETED"' not in pilot_assertion
    assert 'else "NEGATIVE_RESULT"' in pilot_assertion
    assert ".dynamic_complexity_progressed == (.promotion_count >= 1)" in pilot_assertion
    assert "and .hypothesis_passed == true" not in pilot_assertion


def test_successful_screen_exit_fetches_result_before_terminal_tree_decision(
    tmp_path: Path,
) -> None:
    branch_snapshots = [
        {
            "snapshot_id": f"screen-branch-{index}",
            "update": index + 1,
            "level": 0,
            "siblings": [{"index": sibling, "steps": []} for sibling in range(4)],
        }
        for index in range(7)
    ]
    progress = {
        "phase": "eligibility_branch_collection",
        "branch_groups_completed": 7,
        "branch_groups_total": 8,
        "branch_snapshots": branch_snapshots,
        "latest_branch_snapshot": branch_snapshots[-1],
    }
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
            "progress_responses": [progress],
            "exit_code_responses": ["0"],
            "result_responses": [
                {
                    "eligible": False,
                    "branch_groups": 6,
                    "early_stop_reason": ("INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE"),
                    "gate_results": {"informative_group_rate": False},
                }
            ],
        },
    )

    assert result.returncode != 0
    assert (
        "eligibility result and authenticated terminal branch progress did not match"
        in result.stderr
    )
    exit_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"]
        and any(argument.endswith("/exit_code") for argument in command[1:])
    ]
    result_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"]
        and any(argument.endswith("/result.json") for argument in command[1:])
    ]
    delete_indexes = [
        index for index, command in enumerate(commands) if command[:2] == ["pod", "delete"]
    ]
    assert len(exit_indexes) == 1
    assert len(result_indexes) == 1
    assert exit_indexes[0] < result_indexes[0] < min(delete_indexes)
    result_request = commands[result_indexes[0]]
    assert any(argument.startswith("Authorization: Bearer ") for argument in result_request)


@pytest.mark.parametrize(
    ("branch_groups_completed", "branch_snapshot_count", "expected_complete"),
    [
        (8, 8, True),
        (7, 7, False),
        (8, 7, False),
    ],
)
def test_larger_model_screen_terminal_progress_requires_the_exact_cumulative_tree(
    branch_groups_completed: int,
    branch_snapshot_count: int,
    expected_complete: bool,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "larger_model_screen_progress_complete")
    snapshots = [
        {"snapshot_id": f"screen-branch-{index}"} for index in range(branch_snapshot_count)
    ]
    progress = {
        "branch_groups_completed": branch_groups_completed,
        "branch_groups_total": 8,
        "branch_snapshots": snapshots,
        "latest_branch_snapshot": snapshots[-1] if snapshots else None,
    }
    harness = f"""set -euo pipefail
larger_model_expected_branch_groups=8
{helper}
larger_model_screen_progress_complete <<'JSON'
{json.dumps(progress, separators=(",", ":"))}
JSON
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert (result.returncode == 0) is expected_complete
    assert 'larger_model_screen_progress_matches_result "$metrics" <<<"$last_progress"' in source, (
        "screen results must fail closed when terminal progress disagrees"
    )


@pytest.mark.parametrize(
    ("eligible", "result_groups", "progress_groups", "reason", "expected_match"),
    [
        (True, 8, 8, None, True),
        (
            False,
            7,
            7,
            "INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE",
            True,
        ),
        (
            True,
            7,
            7,
            "INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
        ),
        (False, 7, 7, "screen_collection_deadline", False),
        (
            False,
            7,
            6,
            "INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
        ),
    ],
    ids=(
        "eligible-exact-budget",
        "ineligible-whitelisted-fail-fast",
        "eligible-cannot-stop-short",
        "ineligible-non-mathematical-stop",
        "result-progress-count-mismatch",
    ),
)
def test_terminal_screen_progress_matches_only_exact_or_whitelisted_fail_fast_results(
    eligible: bool,
    result_groups: int,
    progress_groups: int,
    reason: str | None,
    expected_match: bool,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "larger_model_screen_progress_matches_result")
    snapshots = [{"snapshot_id": f"screen-branch-{index}"} for index in range(progress_groups)]
    progress = {
        "branch_groups_completed": progress_groups,
        "branch_groups_total": 8,
        "branch_snapshots": snapshots,
        "latest_branch_snapshot": snapshots[-1] if snapshots else None,
    }
    screen_result = {
        "eligible": eligible,
        "branch_groups": result_groups,
        "early_stop_reason": reason,
        "gate_results": {"informative_group_rate": False},
    }
    harness = f"""set -euo pipefail
larger_model_expected_branch_groups=8
larger_model_expected_baseline_examples=8
larger_model_screen_fail_fast_reasons='{json.dumps(SCREEN_FAIL_FAST_REASONS)}'
larger_model_screen_baseline_fail_fast_reasons='{json.dumps(SCREEN_BASELINE_FAIL_FAST_REASONS)}'
{helper}
screen_result='{json.dumps(screen_result, separators=(",", ":"))}'
larger_model_screen_progress_matches_result "$screen_result" <<'JSON'
{json.dumps(progress, separators=(",", ":"))}
JSON
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert (result.returncode == 0) is expected_match


@pytest.mark.parametrize(
    (
        "reason",
        "eligible",
        "completed_baseline",
        "failed_gate",
        "failed_gate_value",
        "branch_state",
        "expected_match",
    ),
    [
        (
            "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            8,
            "baseline_checkpoint_rate",
            False,
            "absent",
            True,
        ),
        (
            "OVERALL_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            8,
            "baseline_checkpoint_rate",
            False,
            "empty",
            True,
        ),
        (
            "BASELINE_EXACT_HEADROOM_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            8,
            "baseline_exact_rate",
            False,
            "absent",
            True,
        ),
        (
            "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            True,
            8,
            "baseline_checkpoint_rate",
            False,
            "absent",
            False,
        ),
        (
            "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            7,
            "baseline_checkpoint_rate",
            False,
            "absent",
            False,
        ),
        (
            "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            8,
            "baseline_checkpoint_rate",
            True,
            "absent",
            False,
        ),
        (
            "screen_collection_incomplete",
            False,
            8,
            "baseline_checkpoint_rate",
            False,
            "absent",
            False,
        ),
        (
            "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE",
            False,
            8,
            "baseline_checkpoint_rate",
            False,
            "nonempty",
            False,
        ),
    ],
    ids=(
        "level-checkpoint-impossible",
        "overall-checkpoint-impossible-explicit-empty-tree",
        "baseline-exact-headroom-impossible",
        "eligible-cannot-have-zero-branch-tree",
        "baseline-count-must-be-exact",
        "reason-must-match-failed-gate",
        "non-mathematical-stop",
        "zero-branch-result-cannot-hide-snapshot",
    ),
)
def test_terminal_screen_progress_accepts_only_exact_empty_baseline_fail_fast(
    reason: str,
    eligible: bool,
    completed_baseline: int,
    failed_gate: str,
    failed_gate_value: bool,
    branch_state: str,
    expected_match: bool,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "larger_model_screen_progress_matches_result")
    progress: dict[str, object] = {
        "phase": "baseline_evaluation",
        "evaluation_completed": 7,
        "evaluation_total": 8,
    }
    if branch_state == "empty":
        progress.update(
            {
                "branch_groups_completed": 0,
                "branch_groups_total": 8,
                "branch_snapshots": [],
                "latest_branch_snapshot": None,
            }
        )
    elif branch_state == "nonempty":
        snapshot = {"snapshot_id": "unexpected-branch"}
        progress.update(
            {
                "branch_groups_completed": 1,
                "branch_groups_total": 8,
                "branch_snapshots": [snapshot],
                "latest_branch_snapshot": snapshot,
            }
        )
    screen_result = {
        "eligible": eligible,
        "branch_groups": 0,
        "completed_baseline_examples": completed_baseline,
        "early_stop_reason": reason,
        "gate_results": {failed_gate: failed_gate_value},
    }
    harness = f"""set -euo pipefail
larger_model_expected_branch_groups=8
larger_model_expected_baseline_examples=8
larger_model_screen_fail_fast_reasons='{json.dumps(SCREEN_FAIL_FAST_REASONS)}'
larger_model_screen_baseline_fail_fast_reasons='{json.dumps(SCREEN_BASELINE_FAIL_FAST_REASONS)}'
{helper}
screen_result='{json.dumps(screen_result, separators=(",", ":"))}'
larger_model_screen_progress_matches_result "$screen_result" <<'JSON'
{json.dumps(progress, separators=(",", ":"))}
JSON
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert (result.returncode == 0) is expected_match


@pytest.mark.parametrize("transport", ("reject", "drop"))
def test_one_shot_bundle_post_rejects_or_reconciles_without_retry(
    tmp_path: Path,
    transport: str,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": transport,
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
        },
    )

    assert result.returncode != 0
    posts = [
        command
        for command in commands
        if command[:1] == ["curl"]
        and "-X" in command
        and command[command.index("-X") + 1] == "POST"
    ]
    progress = [
        command
        for command in commands
        if command[:1] == ["curl"] and command[-1].endswith("/progress.json")
    ]
    creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(creates) == 1
    bundle_posts = [command for command in posts if command[-1].endswith("/bundle")]
    assert len(bundle_posts) == 1
    if transport == "reject":
        assert "rejected the one-shot bundle upload with HTTP 400" in result.stderr
        assert len(posts) == 1
        assert not progress
    else:
        assert "Container is live; streaming structured progress" in result.stderr
        assert len(posts) == 2
        assert any(command[-1].endswith("/activate-staged-bundle") for command in posts)
        assert progress


@pytest.mark.parametrize("activation_transport", ("reject", "drop"))
def test_one_shot_activation_rejects_or_reconciles_without_retry(
    tmp_path: Path,
    activation_transport: str,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "accept",
            "activation_transport": activation_transport,
            "image_indexes": [_image_index(IMAGE_DIGEST), _image_index(IMAGE_DIGEST)],
        },
    )

    assert result.returncode != 0
    activation_posts = [
        command
        for command in commands
        if command[:1] == ["curl"]
        and "-X" in command
        and command[command.index("-X") + 1] == "POST"
        and command[-1].endswith("/activate-staged-bundle")
    ]
    assert len(activation_posts) == 1
    if activation_transport == "reject":
        assert "rejected stage activation with HTTP 409" in result.stderr
    else:
        assert "Container is live; streaming structured progress" in result.stderr


def test_prebootstrap_shell_never_writes_the_shared_volume() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index('bootstrap_private_directory="/tmp/equinox-bootstrap-')
    end = source.index("\n\nprovider_pods() {", start)
    bootstrap_launch = source[start:end]

    assert "mkdir -p $persistent_work_directory" not in bootstrap_launch
    assert "mkdir -p $model_cache_directory" not in bootstrap_launch
    assert "ln -s" not in bootstrap_launch
    assert "rm -f $persistent_work_directory" not in bootstrap_launch
    assert "mv " not in bootstrap_launch
    assert "mkdir -m 0700 -- $bootstrap_private_directory" in bootstrap_launch
    assert 'getattr(os,"O_NOFOLLOW",0)' in bootstrap_launch
    assert "stat.S_ISREG(metadata.st_mode)" in bootstrap_launch
    assert 'f"/proc/self/fd/{descriptor}"' in bootstrap_launch


def test_private_bootstrap_staging_rejects_preseed_and_holds_open_inode(
    tmp_path: Path,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    code_start = source.index("bootstrap_exec_code='") + len("bootstrap_exec_code='")
    code_end = source.index("'\ndocker_args=", code_start)
    bootstrap_code = source[code_start:code_end]
    private_directory = tmp_path / "private-bootstrap"
    attacker_target = tmp_path / "attacker-target"
    attacker_target.mkdir()
    private_directory.symlink_to(attacker_target, target_is_directory=True)

    preseed = subprocess.run(
        ["mkdir", "-m", "0700", "--", str(private_directory)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert preseed.returncode != 0

    private_directory.unlink()
    private_directory.mkdir(mode=0o700)
    bootstrap_path = private_directory / "bootstrap_server.py"
    original = b"trusted bootstrap bytes\n"
    replacement = b"attacker replacement\n"
    bootstrap_path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    instrumented = bootstrap_code.replace(
        "metadata=os.fstat(descriptor)",
        (
            'os.rename(path,path+".opened")\n'
            f'open(path,"wb").write({replacement!r})\n'
            "metadata=os.fstat(descriptor)"
        ),
        1,
    )
    instrumented = instrumented[: instrumented.index("os.unlink(path)")]
    instrumented += "print(digest.hexdigest())\n"

    held_inode = subprocess.run(
        ["python3", "-I", "-S", "-c", instrumented, str(bootstrap_path), expected],
        text=True,
        capture_output=True,
        check=False,
    )

    assert held_inode.returncode == 0, held_inode.stdout + held_inode.stderr
    assert held_inode.stdout.strip() == expected
    assert bootstrap_path.read_bytes() == replacement


@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    [
        (
            {
                "spend": 0.005,
                "network_volumes": [
                    {
                        "id": VOLUME_ID,
                        "name": "configured",
                        "dataCenterId": DATA_CENTER_ID,
                        "size": 50,
                    },
                    {
                        "id": "network-volume-extra",
                        "name": "unexpected",
                        "dataCenterId": DATA_CENTER_ID,
                        "size": 50,
                    },
                ],
            },
            "Refusing to start: the RunPod network-volume inventory is not exactly the verified configured volume.",
        ),
        (
            {"spend": 0.005, "network_volumes": {"items": []}},
            "Refusing to start: the RunPod network-volume inventory is not exactly the verified configured volume.",
        ),
        (
            {"spend": 0.011},
            "Refusing to start: hourly spend exceeds the permitted idle baseline or could not be verified.",
        ),
        (
            {"spend": 0.005, "pods": [{"id": "pod-active", "name": "another-run"}]},
            "Refusing to start: the RunPod account pod inventory is not empty and valid.",
        ),
        (
            {"spend": 0.005, "pods": {"items": []}},
            "Refusing to start: the RunPod account pod inventory is not empty and valid.",
        ),
    ],
    ids=[
        "extra-network-volume",
        "malformed-network-volume-inventory",
        "excess-storage-spend",
        "active-pod",
        "malformed-pod-inventory",
    ],
)
def test_larger_model_launch_refuses_non_storage_idle_provider_states(
    tmp_path: Path,
    scenario: dict[str, object],
    expected_error: str,
) -> None:
    result, commands = _run_launch(tmp_path, scenario=scenario)

    assert result.returncode != 0
    assert expected_error in result.stderr
    _assert_no_paid_create(commands)


def test_readiness_polling_obeys_one_wall_clock_deadline() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index('provider_allocation_started_epoch="$(date +%s)"')
    end = source.index('rm -f -- "$bundle_source_path"', start)
    readiness_block = source[start:end]

    assert (
        "readiness_deadline_epoch=$((provider_allocation_started_epoch + boot_timeout_seconds))"
    ) in readiness_block
    assert readiness_block.count("readiness_deadline_epoch=$(") == 1
    assert (
        "EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH:\n          $readiness_deadline_epoch"
    ) in readiness_block
    assert "and .readiness_deadline_epoch == $readiness_deadline_epoch" in readiness_block
    assert readiness_block.count('seconds_until_deadline "$readiness_deadline_epoch"') >= 8
    assert "readiness_attempts" not in readiness_block


def test_empty_remote_progress_is_invalid_even_when_jq_reports_success(
    tmp_path: Path,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "fetch_and_publish_remote_progress")
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    _write_executable(
        binary_directory / "curl",
        """#!/usr/bin/env python3
""",
    )
    _write_executable(
        binary_directory / "jq",
        """#!/usr/bin/env python3
import sys

sys.stdin.read()
raise SystemExit(0)
""",
    )
    harness = f"""set -euo pipefail
last_remote_progress_fetch_valid=false
last_remote_progress=''
last_progress='{{}}'
last_progress_change_epoch=0
preparation_started_epoch=''
remote_authorization='Authorization: Bearer test'
progress_url=https://worker.invalid/progress.json
publish_execution() {{ return 0; }}
touch_volume_expiry_guard() {{ return 0; }}
{helper}
fetch_and_publish_remote_progress 1000
printf '%s\\n' "$last_remote_progress_fetch_valid"
"""
    environment = os.environ.copy()
    environment["PATH"] = f"{binary_directory}{os.pathsep}{environment['PATH']}"

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "false"


def test_remote_json_acceptance_guards_require_nonempty_bodies() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'if [[ -n "$remote_progress" ]] &&' in source
    assert 'if [[ -z "$metrics" ]] ||' in source
    assert "if [[ \"$bundle_upload_response\" == *$'\\n'* ]]; then" in source
    assert '"$bundle_upload_status" =~ ^2[0-9][0-9]$' in source
    assert '<<<"$bundle_upload_body"' in source


@pytest.mark.parametrize(
    ("payload", "expected_allowed"),
    [
        ({"phase": "training", "adapter_persisted": True}, True),
        ({"artifact_publication": {"set_digest": "forged"}}, False),
        ({"nested": {"artifact_publication_required": True}}, False),
        ({"nested": [{"artifact_set_committed": True}]}, False),
        ({"artifact_set_manifest_digest": "sha256:" + "a" * 64}, False),
    ],
)
def test_remote_payload_cannot_supply_publication_evidence(
    payload: dict[str, object],
    expected_allowed: bool,
) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    helper = _shell_function(source, "remote_payload_has_no_publication_keys")
    harness = f"""set -euo pipefail
{helper}
remote_payload_has_no_publication_keys <<'JSON'
{json.dumps(payload, separators=(",", ":"))}
JSON
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert (result.returncode == 0) is expected_allowed
