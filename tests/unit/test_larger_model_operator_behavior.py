from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research.runpod.larger_model_gate import (
    canonical_json,
    expected_snapshot_digest,
    load_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/runpod-rl-proof"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
VOLUME_ID = "network-volume-123"
DATA_CENTER_ID = "EU-RO-1"
IMAGE_TAG = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
IMAGE_DIGEST = "sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"


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

if arguments == ["pod", "create", "--help"]:
    print("--env string")
    print("environment variables as json object")
elif arguments[:2] == ["template", "get"]:
    print(json.dumps({
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    }))
elif arguments[:2] == ["network-volume", "get"]:
    print(json.dumps(scenario.get("network_volume", {
        "id": "network-volume-123",
        "dataCenterId": "EU-RO-1",
        "size": 50
    })))
elif arguments == ["network-volume", "list"]:
    print(json.dumps(scenario.get("network_volumes", [{
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
    print(json.dumps({"currentSpendPerHr": scenario.get("spend", 0)}))
elif arguments == ["pod", "list", "--all"]:
    print(json.dumps(scenario.get("pods", [])))
elif arguments[:2] == ["pod", "create"]:
    print(json.dumps({"id": "fake-paid-pod"}))
    if not scenario.get("pod_create_succeeds"):
        raise SystemExit(97)
elif arguments[:2] == ["pod", "get"]:
    print(json.dumps({
        "id": "fake-paid-pod",
        "adjustedCostPerHr": scenario.get("pod_hourly_cost", 2.99)
    }))
elif arguments[:2] == ["pod", "delete"]:
    pass
else:
    print(f"unexpected fake runpodctl arguments: {arguments!r}", file=sys.stderr)
    raise SystemExit(98)
""",
    )
    _write_executable(
        binary_directory / "curl",
        """#!/usr/bin/env python3
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

if transport and url.endswith("/bootstrap-health"):
    print(json.dumps({"status": "awaiting_bundle"}, separators=(",", ":")))
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
        print(json.dumps({"status": "bundle_installed"}, separators=(",", ":")))
        print("202")
elif transport and url.endswith("/progress.json"):
    response = sequenced_response(
        "progress_responses",
        "/progress.json",
        {"phase": "running"},
    )
    if response is not None:
        print(json.dumps(response, separators=(",", ":")))
elif transport and url.endswith("/exit_code"):
    response = sequenced_response("exit_code_responses", "/exit_code", "1")
    if response is not None:
        print(response)
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
    return binary_directory, command_log


def _write_volume_receipt(path: Path) -> None:
    manifest = load_manifest()
    receipt = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "model_id": manifest["model"]["id"],
        "model_revision": manifest["model"]["revision"],
        "manifest_digest": "sha256:" + hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "network_volume_id": VOLUME_ID,
        "network_volume_data_center_id": DATA_CENTER_ID,
        "network_volume_size_gb": 50,
        "snapshot_digest": expected_snapshot_digest(manifest),
        "dependencies": manifest["runtime"]["dependencies"],
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    path.write_bytes(canonical_json(receipt) + b"\n")


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
    environment.update(
        {
            "PATH": f"{binary_directory}{os.pathsep}{environment['PATH']}",
            "FAKE_RUNPOD_LOG": str(command_log),
            "FAKE_RUNPOD_PAYLOAD_LOG": str(tmp_path / "observer-payloads.jsonl"),
            "FAKE_RUNPOD_SCENARIO": json.dumps(scenario),
            "EQUINOX_INTERNAL_TOKEN": "test-internal-token-" + "x" * 40,
            "EQUINOX_API_ROOT": "http://operator-test.invalid",
            "EQUINOX_DASHBOARD_ROOT": "http://dashboard-test.invalid",
            "EQUINOX_LARGER_MODEL_MODE": "screen",
            "EQUINOX_LARGER_MODEL_VOLUME_RECEIPT": str(receipt),
            "EQUINOX_RUNPOD_OPERATOR_STATE_DIR": str(tmp_path / "operator-state"),
            "EQUINOX_RUNPOD_PREFLIGHT_ONLY": "1",
            "EQUINOX_RUNPOD_NETWORK_VOLUME_ID": VOLUME_ID,
            "EQUINOX_RUNPOD_DATA_CENTER_IDS": DATA_CENTER_ID,
            "EQUINOX_RUNPOD_GPU": "NVIDIA H100 80GB HBM3",
            "EQUINOX_RUNPOD_CLOUD_TYPE": "SECURE",
            "EQUINOX_RUNPOD_CONTAINER_DISK_GB": "50",
            "EQUINOX_RUNPOD_VOLUME_GB": "50",
            "EQUINOX_RUNPOD_MIN_GPU_MEMORY_GB": "80",
            "EQUINOX_RUNPOD_MAX_HOURLY_COST": "4.00",
            "EQUINOX_RUNPOD_MAX_TOTAL_COST": "3.00",
            "EQUINOX_RUNPOD_MAX_LIFETIME_MINUTES": "43",
            "EQUINOX_RUNPOD_BOOT_TIMEOUT_SECONDS": "360",
            "EQUINOX_RUNPOD_MODEL_LOAD_TIMEOUT_SECONDS": "1200",
            "EQUINOX_RUNPOD_STALE_PROGRESS_SECONDS": "600",
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1",
            "EQUINOX_RUNPOD_RETRY_RESERVE_SECONDS": "300",
            "EQUINOX_RL_MODEL_ID": MODEL_ID,
            "EQUINOX_RL_SEED": "137",
            "EQUINOX_RL_TARGET_SECONDS": "1500",
            "EQUINOX_RL_MAX_UPDATES": "1",
            "EQUINOX_RL_VALIDATION_EXAMPLES": "8",
            "EQUINOX_RL_TEST_EXAMPLES": "4",
            "EQUINOX_RL_MASTERY_WINDOWS": "1",
            "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": "8",
            "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL": "1",
            "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS": "300",
        }
    )
    return environment


def _run_preflight(
    tmp_path: Path,
    *,
    scenario: dict[str, object] | None = None,
    missing_volume: bool = False,
    missing_receipt: bool = False,
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
        commands = [
            json.loads(line)
            for line in command_log.read_text(encoding="utf-8").splitlines()
            if line
        ]
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
    return result, commands


def _assert_no_paid_create(commands: list[list[str]]) -> None:
    assert not [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]


def _model_cache_setup_from(commands: list[list[str]]) -> str:
    paid_creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(paid_creates) == 1
    create = paid_creates[0]
    docker_args = create[create.index("--docker-args") + 1]
    start = docker_args.index("test -d /workspace/equinox-state/python")
    end = docker_args.index(" echo ", start)
    return docker_args[start:end]


def _shell_function(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


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
        "profile_id": "qwen2.5-coder-7b-runpod-h100@5",
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


@pytest.mark.parametrize(
    "third_resolution",
    (_image_index("sha256:" + "0" * 64), None),
    ids=("digest-drift", "inspection-failure"),
)
def test_image_reverification_failure_after_create_deletes_exact_pod_before_readiness(
    tmp_path: Path,
    third_resolution: object,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
                third_resolution,
            ],
        },
    )

    assert result.returncode != 0
    assert "terminating before readiness" in result.stderr
    paid_creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    assert len(paid_creates) == 1
    assert ["pod", "get", "fake-paid-pod"] in commands
    assert ["pod", "delete", "fake-paid-pod"] in commands
    inspections = [command for command in commands if command[:1] == ["docker"]]
    assert len(inspections) == 3


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
    assert paid_creates[0][paid_creates[0].index("--image") + 1] == IMAGE_TAG
    assert "@" not in paid_creates[0][paid_creates[0].index("--image") + 1]


def test_paid_create_carries_only_token_then_uploads_after_image_postcheck(
    tmp_path: Path,
) -> None:
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
    create_environment = json.loads(create[create.index("--env") + 1])
    assert set(create_environment) == {"EQUINOX_RESULT_TOKEN"}
    assert len(create_environment["EQUINOX_RESULT_TOKEN"]) == 64
    assert "EQUINOX_BUNDLE_B64" not in create[create.index("--env") + 1]

    image_postcheck_index = max(
        index for index, command in enumerate(commands) if command[:1] == ["docker"]
    )
    health_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"] and command[-1].endswith("/bootstrap-health")
    ]
    upload_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"] and command[-1].endswith("/bundle")
    ]
    progress_indexes = [
        index
        for index, command in enumerate(commands)
        if command[:1] == ["curl"] and command[-1].endswith("/progress.json")
    ]
    assert len(upload_indexes) == 1
    assert image_postcheck_index < health_indexes[0] < upload_indexes[0] < progress_indexes[0]
    upload = commands[upload_indexes[0]]
    token = create_environment["EQUINOX_RESULT_TOKEN"]
    assert upload[upload.index("-X") + 1] == "POST"
    assert f"Authorization: Bearer {token}" in upload
    assert "Content-Type: application/gzip" in upload
    assert "Expect:" in upload
    assert upload[upload.index("--data-binary") + 1].startswith("@")

    delete_indexes = [
        index for index, command in enumerate(commands) if command[:2] == ["pod", "delete"]
    ]
    assert delete_indexes
    assert all(commands[index] == ["pod", "delete", "fake-paid-pod"] for index in delete_indexes)
    assert upload_indexes[0] < min(delete_indexes)


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
    assert '[[ "$terminal_progress_complete" != true ]]' in source, (
        "successful screens must fail closed when terminal progress is incomplete"
    )


def test_rejected_bundle_upload_deletes_the_exact_pod_without_retry(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "reject",
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
        },
    )

    assert result.returncode != 0
    assert "rejected the workload bundle with HTTP 400" in result.stderr
    creates = [
        command
        for command in commands
        if command[:2] == ["pod", "create"] and command != ["pod", "create", "--help"]
    ]
    uploads = [
        command
        for command in commands
        if command[:1] == ["curl"] and command[-1].endswith("/bundle")
    ]
    deletes = [command for command in commands if command[:2] == ["pod", "delete"]]
    progress = [
        command
        for command in commands
        if command[:1] == ["curl"] and command[-1].endswith("/progress.json")
    ]
    assert len(creates) == 1
    assert len(uploads) == 1
    assert not progress
    assert deletes
    assert all(command == ["pod", "delete", "fake-paid-pod"] for command in deletes)


def test_dropped_bundle_response_is_resolved_by_progress_without_reupload(
    tmp_path: Path,
) -> None:
    result, commands = _run_launch(
        tmp_path,
        scenario={
            "spend": 0.005,
            "pod_create_succeeds": True,
            "transport": "drop",
            "image_indexes": [
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
                _image_index(IMAGE_DIGEST),
            ],
        },
    )

    assert result.returncode != 0
    assert "Container is live; streaming structured progress" in result.stderr
    assert "no authenticated progress handoff appeared" not in result.stderr
    uploads = [
        command
        for command in commands
        if command[:1] == ["curl"] and command[-1].endswith("/bundle")
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
    assert len(uploads) == 1
    assert progress
    assert len(creates) == 1


@pytest.mark.parametrize("prepared_layout", ("root", "hub"))
def test_larger_model_launch_normalizes_both_verified_cache_layouts(
    tmp_path: Path,
    prepared_layout: str,
) -> None:
    result, commands = _run_launch(tmp_path, scenario={"spend": 0.005})

    assert result.returncode != 0
    setup = _model_cache_setup_from(commands)
    state_root = tmp_path / "remote-state"
    cache_root = state_root / "huggingface"
    model_cache_name = "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    revision = load_manifest()["model"]["revision"]
    root_model = cache_root / model_cache_name
    hub_model = cache_root / "hub" / model_cache_name
    root_snapshot = root_model / "snapshots" / revision
    hub_snapshot = hub_model / "snapshots" / revision
    (state_root / "python").mkdir(parents=True)
    prepared_snapshot = root_snapshot if prepared_layout == "root" else hub_snapshot
    prepared_snapshot.mkdir(parents=True)
    local_setup = setup.replace("/workspace/equinox-state", str(state_root))

    subprocess.run(["bash", "-c", f"set -e; {local_setup}"], check=True)

    assert root_snapshot.samefile(hub_snapshot)
    if prepared_layout == "root":
        assert hub_model.is_symlink()
        assert os.readlink(hub_model) == f"../{model_cache_name}"
    else:
        assert root_model.is_symlink()
        assert os.readlink(root_model) == f"hub/{model_cache_name}"


def test_larger_model_launch_rejects_divergent_cache_layouts(tmp_path: Path) -> None:
    result, commands = _run_launch(tmp_path, scenario={"spend": 0.005})

    assert result.returncode != 0
    setup = _model_cache_setup_from(commands)
    state_root = tmp_path / "remote-state"
    cache_root = state_root / "huggingface"
    model_cache_name = "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    revision = load_manifest()["model"]["revision"]
    (state_root / "python").mkdir(parents=True)
    (cache_root / model_cache_name / "snapshots" / revision).mkdir(parents=True)
    (cache_root / "hub" / model_cache_name / "snapshots" / revision).mkdir(parents=True)
    local_setup = setup.replace("/workspace/equinox-state", str(state_root))

    completed = subprocess.run(
        ["bash", "-c", f"set -e; {local_setup}"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0


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


def test_readiness_polling_obeys_one_wall_clock_deadline(tmp_path: Path) -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index("bootstrap_transport_ready=false\n")
    end = source.index('if [[ "$bootstrap_transport_ready" != true ]]', start)
    readiness_block = source[start:end]

    assert "readiness_deadline_epoch" in readiness_block
    assert "readiness_attempts" not in readiness_block

    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    clock_path = tmp_path / "clock"
    trace_path = tmp_path / "trace.jsonl"
    clock_path.write_text("1000\n", encoding="utf-8")
    _write_executable(
        binary_directory / "date",
        """#!/usr/bin/env python3
import os
import pathlib
import sys

if sys.argv[1:] != ["+%s"]:
    raise SystemExit(f"unsupported fake date arguments: {sys.argv[1:]!r}")
print(pathlib.Path(os.environ["FAKE_CLOCK"]).read_text(encoding="utf-8").strip())
""",
    )
    advancing_command = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

kind = pathlib.Path(sys.argv[0]).name
if kind == "curl":
    arguments = sys.argv[1:]
    seconds = int(arguments[arguments.index("--max-time") + 1])
else:
    seconds = int(float(sys.argv[1]))
clock_path = pathlib.Path(os.environ["FAKE_CLOCK"])
current = int(clock_path.read_text(encoding="utf-8").strip())
clock_path.write_text(f"{current + seconds}\\n", encoding="utf-8")
with open(os.environ["FAKE_TRACE"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"kind": kind, "seconds": seconds}) + "\\n")
if kind == "curl":
    raise SystemExit(28)
"""
    _write_executable(binary_directory / "curl", advancing_command)
    _write_executable(binary_directory / "sleep", advancing_command)

    helper = _shell_function(source, "seconds_until_deadline")
    harness = f"""set -euo pipefail
{helper}
boot_started_epoch=1000
readiness_deadline_epoch=1012
boot_timeout_seconds=12
last_progress='{{}}'
maximum_updates=1
bootstrap_health_url=https://worker.invalid/bootstrap-health
remote_authorization='Authorization: Bearer test'
publish_execution() {{ :; }}
{readiness_block}
date +%s
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binary_directory}{os.pathsep}{environment['PATH']}",
            "FAKE_CLOCK": str(clock_path),
            "FAKE_TRACE": str(trace_path),
        }
    )
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
    assert int(result.stdout.strip()) == 1012
    trace = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert sum(event["seconds"] for event in trace) == 12
    assert all(event["seconds"] > 0 for event in trace)
    assert any(event["kind"] == "curl" for event in trace)
