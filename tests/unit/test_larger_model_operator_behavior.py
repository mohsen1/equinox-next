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


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


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
    raise SystemExit(97)
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
import sys

if "--fail" not in sys.argv:
    raise SystemExit(96)
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
    assert payload["optimization_seed"] == 137
    _assert_no_paid_create(commands)


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
    start = source.index('exit_code=""\n') + len('exit_code=""\n')
    end = source.index('rm -f -- "$bundle_source_path"', start)
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
progress_url=https://worker.invalid/progress.json
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
