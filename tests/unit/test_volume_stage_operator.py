from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import research.runpod.volume_stage_operator as volume_stage_module
from research.runpod.volume_stage_operator import (
    CAPACITY_REJECTION_TEXT,
    EXIT_AMBIGUOUS_CREATE,
    EXIT_CAPACITY_UNAVAILABLE,
    MAXIMUM_LIFETIME_SECONDS,
    RECONCILIATION_SECONDS,
    BundlePlan,
    OperatorStop,
    RemoteArtifacts,
    VolumeStageOperator,
    _tagged_sha256,
    _verify_torch_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
VOLUME_ID = "volume-123"
DATA_CENTER_ID = "EU-RO-1"
POD_ID = "pod-123"


def completed(
    arguments: tuple[str, ...],
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        list(arguments),
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


class FakeClock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class ProviderRunner:
    def __init__(
        self,
        *,
        create_returncode: int = 0,
        create_stdout: bytes | None = None,
        create_stderr: bytes = b"",
        create_exception: BaseException | None = None,
        active_pods: list[dict[str, Any]] | None = None,
        spend: str = "0.01",
        volume_id: str = VOLUME_ID,
        volume_count: int = 1,
        pod_list_failures: set[int] | None = None,
        late_visible_at: int | None = None,
        pod_overrides: dict[str, Any] | None = None,
        persistent_after_create: dict[str, Any] | None = None,
        delete_effective: bool = True,
        head_mismatch_path: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.create_returncode = create_returncode
        self.create_stdout = (
            json.dumps({"id": POD_ID}).encode() if create_stdout is None else create_stdout
        )
        self.create_stderr = create_stderr
        self.create_exception = create_exception
        self.active_pods = active_pods or []
        self.spend = spend
        self.volume_id = volume_id
        self.volume_count = volume_count
        self.pod_list_failures = pod_list_failures or set()
        self.late_visible_at = late_visible_at
        self.pod_overrides = pod_overrides or {}
        self.persistent_after_create = persistent_after_create
        self.delete_effective = delete_effective
        self.head_mismatch_path = head_mismatch_path
        self.pod_list_calls = 0
        self.created_name = ""
        self.terminate_after = ""
        self.deleted_ids: list[str] = []
        self.pod_list_count_at_delete: int | None = None
        manifest = json.loads((ROOT / "research/studies/larger-model-eligibility.json").read_text())
        self.image = f"{manifest['runtime']['image']}@{manifest['runtime']['image_digest']}"
        self.volume_size = manifest["hardware"]["volume_disk_gb"]

    def _pod_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": POD_ID,
            "name": self.created_name,
            "computeType": "CPU",
            "cpuFlavorId": "cpu3c-2-8",
            "vcpuCount": 8,
            "adjustedCostPerHr": "0.20",
            "machine": {"secureCloud": True},
            "imageName": self.image,
            "containerDiskInGb": 5,
            "networkVolume": {
                "id": VOLUME_ID,
                "dataCenterId": DATA_CENTER_ID,
                "size": self.volume_size,
            },
            "volumeMountPath": "/workspace",
            "terminateAfter": self.terminate_after,
        }
        payload.update(self.pod_overrides)
        return payload

    def run(
        self,
        arguments: tuple[str, ...] | list[str],
        *,
        timeout: int,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        del timeout, input_bytes
        command = tuple(arguments)
        self.calls.append(command)
        if command == ("runpodctl", "version"):
            return completed(command, stdout=b"runpodctl 2.7.2-309512b\n")
        if command == ("runpodctl", "pod", "create", "--help"):
            return completed(
                command,
                stdout=(
                    b"create a new pod\n"
                    b"create a cpu pod\n"
                    b"--compute-type string\n"
                    b"--cloud-type string\n"
                    b"--network-volume-id string\n"
                    b"--terminate-after string auto-terminate datetime\n"
                ),
            )
        if command == ("runpodctl", "user"):
            return completed(
                command,
                stdout=json.dumps(
                    {
                        "clientBalance": "10.00",
                        "currentSpendPerHr": self.spend,
                    }
                ).encode(),
            )
        if command == ("runpodctl", "pod", "list", "--all"):
            self.pod_list_calls += 1
            if self.pod_list_calls in self.pod_list_failures:
                return completed(command, returncode=1, stderr=b"transient provider error")
            if self.late_visible_at == self.pod_list_calls and POD_ID not in self.deleted_ids:
                return completed(
                    command,
                    stdout=json.dumps([{"id": POD_ID, "name": self.created_name}]).encode(),
                )
            if self.created_name and self.persistent_after_create is not None:
                persistent = {
                    "id": POD_ID,
                    "name": self.created_name,
                    **self.persistent_after_create,
                }
                if not self.delete_effective or persistent["id"] not in self.deleted_ids:
                    return completed(
                        command,
                        stdout=json.dumps([persistent]).encode(),
                    )
            return completed(command, stdout=json.dumps(self.active_pods).encode())
        if command == ("runpodctl", "network-volume", "list"):
            volumes = [
                {
                    "id": self.volume_id,
                    "dataCenterId": DATA_CENTER_ID,
                    "size": self.volume_size,
                }
                for _ in range(self.volume_count)
            ]
            return completed(command, stdout=json.dumps(volumes).encode())
        if command[:3] == ("runpodctl", "pod", "create"):
            self.created_name = command[command.index("--name") + 1]
            self.terminate_after = command[command.index("--terminate-after") + 1]
            if self.create_exception is not None:
                raise self.create_exception
            return completed(
                command,
                returncode=self.create_returncode,
                stdout=self.create_stdout,
                stderr=self.create_stderr,
            )
        if command[:3] == ("runpodctl", "pod", "get"):
            return completed(command, stdout=json.dumps(self._pod_payload()).encode())
        if command[:3] == ("runpodctl", "pod", "delete"):
            self.deleted_ids.append(command[3])
            self.pod_list_count_at_delete = self.pod_list_calls
            return completed(command, stdout=b"{}")
        if command == ("git", "rev-parse", "HEAD"):
            return completed(command, stdout=("0" * 40 + "\n").encode())
        if command[:2] == ("git", "show"):
            relative = command[2].split(":", 1)[1]
            payload = (ROOT / relative).read_bytes()
            if relative == self.head_mismatch_path:
                payload += b"\n# dirty\n"
            return completed(command, stdout=payload)
        raise AssertionError(f"unexpected command: {command}")


class TransportRunner:
    def __init__(self, tmp_path: Path, inbound: dict[str, bytes]) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.key = tmp_path / "runpod-key"
        self.key.write_text("test-key")
        self.inbound = inbound

    def run(
        self,
        arguments: tuple[str, ...] | list[str],
        *,
        timeout: int,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        del timeout, input_bytes
        command = tuple(arguments)
        self.calls.append(command)
        if command[:3] == ("runpodctl", "ssh", "info"):
            return completed(
                command,
                stdout=json.dumps(
                    {"sshCommand": (f"ssh -i {self.key} -p 2222 root@203.0.113.10")}
                ).encode(),
            )
        if command[0] == "ssh":
            return completed(command)
        if command[0] == "scp":
            if command[-1].startswith("root@"):
                return completed(command)
            remote_source = next(
                argument for argument in command if argument.startswith("root@") and ":" in argument
            )
            remote_name = remote_source.rsplit("/", 1)[-1]
            Path(command[-1]).write_bytes(self.inbound[remote_name])
            return completed(command)
        if command == ("runpodctl", "network-volume", "get", VOLUME_ID):
            return completed(
                command,
                stdout=json.dumps(
                    {
                        "id": VOLUME_ID,
                        "dataCenterId": DATA_CENTER_ID,
                        "size": 50,
                    }
                ).encode(),
            )
        raise AssertionError(f"unexpected transport command: {command}")


def fake_bundle_plan(profile_id: str) -> BundlePlan:
    payload = b"deterministic workload bundle"
    digest = _tagged_sha256(payload)
    source_archive = b"deterministic operator source archive"
    source_sha256 = {
        relative: _tagged_sha256((ROOT / relative).read_bytes())
        for relative in (
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_large_model_trainer.py",
            "research/runpod/retention_checkpoint_probe.py",
        )
    }
    return BundlePlan(
        payload=payload,
        metadata={
            "profile_id": profile_id,
            "manifest_digest": "sha256:" + "1" * 64,
            "source_contract_digest": "sha256:" + "2" * 64,
            "bundle_digest": digest,
            "bundle_size_bytes": len(payload),
            "bundle_path": (
                f"/workspace/equinox-state/workload-bundles/{profile_id}/"
                f"{digest.removeprefix('sha256:')}.tar.xz"
            ),
            "source_archive_digest": _tagged_sha256(source_archive),
            "head_commit": "0" * 40,
        },
        source_archive=source_archive,
        source_archive_digest=_tagged_sha256(source_archive),
        source_sha256=source_sha256,
        head_commit="0" * 40,
    )


def valid_torch_evidence(
    manifest: dict[str, Any],
    plan: BundlePlan,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "test_id": (
            "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        ),
        "profile_id": manifest["profile_id"],
        "head_commit": plan.head_commit,
        "source_archive_digest": plan.source_archive_digest,
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": "12.8",
        "probe_sha256": plan.source_sha256["research/runpod/retention_checkpoint_probe.py"],
        "trainer_sha256": plan.source_sha256[
            "research/runpod/repository_repair_large_model_trainer.py"
        ],
        "environment_sha256": plan.source_sha256["research/runpod/repository_repair_env.py"],
        "source_sha256": {
            name: f"sha256:{digest}"
            for name, digest in manifest["source_contract"]["files"].items()
        },
        "checkpoint_sha256": "sha256:" + "8" * 64,
        "checkpoint_size_bytes": 1024,
        "restored_weight_before_resume_step": [0.99],
        "advanced_weight_after_resume_step": [0.98],
        "effective_policy_update_count_before_resume_step": 1,
        "effective_policy_update_count_after_resume_step": 2,
        "retained_observation_after_resume_step": {"exact_rate": 0.75},
        "optimizer_state_entries_after_resume_step": 1,
        "optimizer_state_digest_before_persist": "sha256:" + "6" * 64,
        "optimizer_state_digest_after_restore": "sha256:" + "6" * 64,
        "optimizer_state_digest_after_resume_step": "sha256:" + "7" * 64,
        "optimizer_parameter_device": "cpu",
        "optimizer_state_devices_before_persist": {
            "exp_avg": ["cpu"],
            "exp_avg_sq": ["cpu"],
            "step": ["cpu"],
        },
        "optimizer_state_devices_after_restore": {
            "exp_avg": ["cpu"],
            "exp_avg_sq": ["cpu"],
            "step": ["cpu"],
        },
    }


def make_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: Any,
    *,
    reconciliation_seconds: int = 10,
) -> VolumeStageOperator:
    monkeypatch.setenv("EQUINOX_RUNPOD_NETWORK_VOLUME_ID", VOLUME_ID)
    monkeypatch.setenv("EQUINOX_RUNPOD_DATA_CENTER_IDS", DATA_CENTER_ID)
    monkeypatch.setenv(
        "EQUINOX_RUNPOD_OPERATOR_STATE_DIR",
        str(tmp_path / "operator-state"),
    )
    operator = VolumeStageOperator(
        ROOT,
        runner=runner,
        clock=FakeClock(),
        proof_directory=tmp_path,
        reconciliation_seconds=reconciliation_seconds,
        poll_seconds=5,
    )
    plan = fake_bundle_plan(operator.manifest["profile_id"])
    monkeypatch.setattr(operator, "_bundle_plan", lambda: plan)
    return operator


def paid_creates(runner: ProviderRunner) -> list[tuple[str, ...]]:
    return [
        call
        for call in runner.calls
        if call[:3] == ("runpodctl", "pod", "create") and "--name" in call and "--help" not in call
    ]


def prepare_publication_lease(
    operator: VolumeStageOperator,
    *,
    attempt_id: str,
) -> None:
    operator.pod_name = attempt_id
    operator.started_at = "2026-07-29T12:00:00Z"
    operator.terminate_after = "2026-07-29T12:10:00Z"
    operator.create_attempted = True
    operator.pod_id = POD_ID
    operator.baseline_balance_usd = "10.00"
    operator.attested_hourly_rate_usd = "0.20"
    operator.teardown_confirmed = True
    operator.idle_polls = [
        {
            "observed_at": "2026-07-29T12:01:00Z",
            "pod_count": "0",
            "network_volume_id": VOLUME_ID,
            "balance_usd": "10.00",
            "spend_usd_per_hour": "0.01",
        }
    ] * 3
    operator._acquire_shared_lease()


def test_preflight_is_read_only_and_reports_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)

    result = operator.preflight()

    assert result["outcome"] == "preflight_passed"
    assert result["allocation_attempted"] is False
    assert result["profile_id"] == operator.manifest["profile_id"]
    assert (
        result["bundle_digest"]
        == fake_bundle_plan(operator.manifest["profile_id"]).metadata["bundle_digest"]
    )
    assert (
        result["source_archive_digest"]
        == fake_bundle_plan(operator.manifest["profile_id"]).source_archive_digest
    )
    assert result["head_commit"] == "0" * 40
    assert result["network_volume_id"] == VOLUME_ID
    assert result["network_volume_data_center_id"] == DATA_CENTER_ID
    assert result["maximum_hourly_cost_usd"] == "0.25"
    assert result["maximum_lifetime_seconds"] == 600
    assert result["ambiguity_reconciliation_seconds"] == RECONCILIATION_SECONDS
    assert result["runpodctl_version"] == "2.7.2-309512b"
    assert result["termination_deadline_contract"] == "absolute_datetime_flag"
    assert result["gpu_fallback_allowed"] is False
    assert not paid_creates(runner)
    assert not any(call[0] in {"ssh", "scp"} for call in runner.calls)
    assert not any(call[:3] == ("runpodctl", "pod", "delete") for call in runner.calls)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (
            ProviderRunner(active_pods=[{"id": "other-pod", "name": "other"}]),
            "pod inventory is not empty",
        ),
        (ProviderRunner(spend="0.02"), "storage-only ceiling"),
        (ProviderRunner(volume_id="wrong-volume"), "identity changed"),
        (ProviderRunner(volume_count=2), "exactly one network volume"),
    ],
)
def test_preflight_rejects_unsafe_provider_state_before_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: ProviderRunner,
    message: str,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(RuntimeError, match=message):
        operator.preflight()

    assert not paid_creates(runner)


def test_preflight_refuses_existing_shared_operator_lease_without_provider_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    operator.lease_directory.mkdir(parents=True)
    operator.lease_path.write_text('{"state":"ambiguous_create_unresolved"}')

    with pytest.raises(RuntimeError, match="unresolved shared"):
        operator.preflight()

    assert runner.calls == []


def test_shared_operator_lease_excludes_a_second_worktree_operator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = make_operator(monkeypatch, tmp_path, ProviderRunner())
    second = make_operator(monkeypatch, tmp_path, ProviderRunner())
    for operator, suffix in ((first, "first"), (second, "second")):
        operator.pod_name = f"equinox-volume-stage-{suffix}"
        operator.started_at = "2026-07-29T12:00:00Z"
        operator.terminate_after = "2026-07-29T12:10:00Z"
        operator.bundle_metadata = dict(fake_bundle_plan(operator.manifest["profile_id"]).metadata)

    first._acquire_shared_lease()

    with pytest.raises(RuntimeError, match="unresolved shared"):
        second._acquire_shared_lease()
    assert first.lease_path.is_file()


def test_failed_live_process_recovery_does_not_block_original_lease_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = make_operator(monkeypatch, tmp_path, ProviderRunner())
    prepare_publication_lease(
        original,
        attempt_id="equinox-volume-stage-live-original",
    )
    recovery = make_operator(monkeypatch, tmp_path, ProviderRunner())

    with pytest.raises(RuntimeError, match="still alive"):
        recovery.recover_artifact_publication()

    assert not original.publication_recovery_lock_path.exists()
    original._persist_lease(state="completed", evidence={"outcome": "test"})
    original._release_shared_lease()
    assert not original.lease_directory.exists()


def test_operator_source_archive_is_deterministic_allowlisted_and_head_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())

    first, first_hashes, first_head = operator._deterministic_head_source_archive()
    second, second_hashes, second_head = operator._deterministic_head_source_archive()

    assert first == second
    assert first_hashes == second_hashes
    assert first_head == second_head == "0" * 40
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(first_hashes)
    assert all(
        member.isfile()
        and member.mtime == 0
        and member.uid == 0
        and member.gid == 0
        and member.uname == ""
        and member.gname == ""
        for member in members
    )
    assert not any("uv.lock" in member.name for member in members)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        archive.extractall(extracted, filter="data")
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import research.runpod.workload_bundle",
        ],
        cwd=extracted,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert imported.returncode == 0, imported.stderr.decode()

    dirty = make_operator(
        monkeypatch,
        tmp_path / "dirty",
        ProviderRunner(head_mismatch_path="research/runpod/retention_checkpoint_probe.py"),
    )
    with pytest.raises(RuntimeError, match="differs from Git HEAD"):
        dirty._deterministic_head_source_archive()


def test_bundle_is_built_from_immutable_head_snapshot_not_live_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    built_from: list[Path] = []
    observed_trainer: list[bytes] = []
    payload = b"bundle-from-head-snapshot"

    def build_from_snapshot(
        repository_root: Path,
        snapshot_manifest: dict[str, Any],
    ) -> tuple[bytes, dict[str, Any]]:
        built_from.append(repository_root)
        observed_trainer.append(
            (
                repository_root / "research/runpod/repository_repair_large_model_trainer.py"
            ).read_bytes()
        )
        return payload, {
            "profile_id": snapshot_manifest["profile_id"],
            "manifest_digest": "sha256:" + "1" * 64,
            "source_contract_digest": "sha256:" + "2" * 64,
            "bundle_digest": _tagged_sha256(payload),
            "bundle_size_bytes": len(payload),
            "bundle_path": "/workspace/immutable.tar.xz",
        }

    monkeypatch.setattr(volume_stage_module, "verify_source_contract", lambda *args: None)
    monkeypatch.setattr(
        volume_stage_module,
        "build_larger_model_bundle",
        build_from_snapshot,
    )

    plan = VolumeStageOperator._bundle_plan(operator)

    assert plan.payload == payload
    assert len(built_from) == 1
    assert built_from[0] != ROOT
    assert not built_from[0].exists()
    assert observed_trainer == [
        (ROOT / "research/runpod/repository_repair_large_model_trainer.py").read_bytes()
    ]


def test_create_contract_is_one_secure_cpu_with_no_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    deadline = "2026-07-29T12:10:00Z"

    command = operator._create_arguments("unique-stage-name", deadline)

    assert command[:3] == ["runpodctl", "pod", "create"]
    assert command[command.index("--compute-type") + 1] == "cpu"
    assert command[command.index("--cloud-type") + 1] == "SECURE"
    assert command[command.index("--network-volume-id") + 1] == VOLUME_ID
    assert command[command.index("--data-center-ids") + 1] == DATA_CENTER_ID
    assert command[command.index("--volume-mount-path") + 1] == "/workspace"
    assert command[command.index("--terminate-after") + 1] == deadline
    assert command[command.index("--container-disk-in-gb") + 1] == "5"
    assert command[command.index("--ports") + 1] == "22/tcp"
    assert command[command.index("--image") + 1].endswith(
        "@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
    )
    assert "--gpu-id" not in command
    assert "H100" not in " ".join(command)


def test_capacity_rejection_requires_clean_reconciliation_and_never_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=CAPACITY_REJECTION_TEXT.encode(),
    )
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_CAPACITY_UNAVAILABLE
    assert raised.value.evidence["outcome"] == "capacity_unavailable_confirmed_zero_allocation"
    assert raised.value.evidence["reconciliation_query_failures"] == 0
    assert raised.value.evidence["teardown_confirmed"] is True
    assert len(raised.value.evidence["idle_polls"]) == 3
    assert len(paid_creates(runner)) == 1
    create_index = runner.calls.index(paid_creates(runner)[0])
    assert runner.calls[create_index - 3 : create_index] == [
        ("runpodctl", "user"),
        ("runpodctl", "pod", "list", "--all"),
        ("runpodctl", "network-volume", "list"),
    ]
    assert not runner.deleted_ids
    assert not any(call[0] in {"ssh", "scp"} for call in runner.calls)
    assert "gpu" not in " ".join(paid_creates(runner)[0]).lower()


def test_production_five_minute_reconciliation_runs_sixty_visibility_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=CAPACITY_REJECTION_TEXT.encode(),
    )
    operator = make_operator(
        monkeypatch,
        tmp_path,
        runner,
        reconciliation_seconds=RECONCILIATION_SECONDS,
    )

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_CAPACITY_UNAVAILABLE
    assert raised.value.evidence["reconciliation_seconds"] == 300
    assert raised.value.evidence["reconciliation_queries"] == 60
    assert operator.clock.monotonic() == 310
    assert len(paid_creates(runner)) == 1


def test_capacity_text_with_any_reconciliation_gap_remains_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=CAPACITY_REJECTION_TEXT.encode(),
        pod_list_failures={3},
    )
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_AMBIGUOUS_CREATE
    assert raised.value.evidence["outcome"] == "ambiguous_create_no_identity"
    assert raised.value.evidence["reconciliation_query_failures"] == 1
    assert len(paid_creates(runner)) == 1


def test_unknown_failed_create_remains_ambiguous_without_hidden_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=b"upstream response was empty",
    )
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_AMBIGUOUS_CREATE
    assert raised.value.evidence["outcome"] == "ambiguous_create_no_identity"
    assert len(paid_creates(runner)) == 1
    assert not runner.deleted_ids


def test_exit_70_persists_shared_lease_and_blocks_sequential_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=b"unknown provider error",
    )
    operator = make_operator(monkeypatch, tmp_path, runner)
    monkeypatch.setattr(
        volume_stage_module,
        "VolumeStageOperator",
        lambda repository_root: operator,
    )

    assert volume_stage_module.main([]) == EXIT_AMBIGUOUS_CREATE
    lease = json.loads(operator.lease_path.read_text())
    assert lease["state"] == "ambiguous_create_unresolved"
    assert lease["evidence"]["outcome"] == "ambiguous_create_no_identity"

    second_runner = ProviderRunner()
    second = make_operator(monkeypatch, tmp_path, second_runner)
    with pytest.raises(RuntimeError, match="unresolved shared"):
        second.preflight()
    assert second_runner.calls == []


def test_confirmed_zero_allocation_capacity_exit_releases_shared_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=CAPACITY_REJECTION_TEXT.encode(),
    )
    operator = make_operator(monkeypatch, tmp_path, runner)
    monkeypatch.setattr(
        volume_stage_module,
        "VolumeStageOperator",
        lambda repository_root: operator,
    )

    assert volume_stage_module.main([]) == EXIT_CAPACITY_UNAVAILABLE
    assert not operator.lease_directory.exists()


@pytest.mark.parametrize(
    "create_exception",
    [
        subprocess.TimeoutExpired(["runpodctl", "pod", "create"], 90),
        KeyboardInterrupt(),
    ],
)
def test_create_timeout_or_interrupt_still_runs_full_ambiguity_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    create_exception: BaseException,
) -> None:
    runner = ProviderRunner(create_exception=create_exception)
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_AMBIGUOUS_CREATE
    assert raised.value.evidence["outcome"] == "ambiguous_create_no_identity"
    assert raised.value.evidence["reconciliation_seconds"] == 10
    assert raised.value.evidence["reconciliation_queries"] == 2
    assert raised.value.evidence["teardown_confirmed"] is True
    assert len(paid_creates(runner)) == 1
    assert not runner.deleted_ids


def test_nonzero_create_with_valid_id_is_deleted_without_remote_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=json.dumps({"id": POD_ID}).encode(),
        create_stderr=b"provider timed out after allocation",
    )
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_AMBIGUOUS_CREATE
    assert raised.value.evidence["allocation_confirmed"] is True
    assert raised.value.evidence["teardown_confirmed"] is True
    assert runner.deleted_ids
    assert set(runner.deleted_ids) == {POD_ID}
    assert len(paid_creates(runner)) == 1
    assert not any(call[0] in {"ssh", "scp"} for call in runner.calls)


def test_late_visible_ambiguous_pod_is_deleted_and_not_staged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(
        create_returncode=1,
        create_stdout=b"",
        create_stderr=b"unknown provider error",
        late_visible_at=4,
    )
    operator = make_operator(
        monkeypatch,
        tmp_path,
        runner,
        reconciliation_seconds=20,
    )

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.evidence["outcome"] == "ambiguous_create_allocated_then_deleted"
    assert runner.deleted_ids
    assert set(runner.deleted_ids) == {POD_ID}
    assert len(paid_creates(runner)) == 1
    assert not any(call[0] in {"ssh", "scp"} for call in runner.calls)


@pytest.mark.parametrize(
    "runner",
    [
        ProviderRunner(
            create_returncode=1,
            create_stdout=b"",
            create_stderr=b"unknown provider error",
            persistent_after_create={"id": "unsafe pod id"},
        ),
        ProviderRunner(
            create_returncode=1,
            create_stdout=json.dumps({"id": POD_ID}).encode(),
            create_stderr=b"failed after allocation",
            persistent_after_create={"id": POD_ID},
            delete_effective=False,
        ),
    ],
)
def test_unusable_identity_or_unconfirmed_delete_is_persistable_exit_70_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: ProviderRunner,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(OperatorStop) as raised:
        operator.execute()

    assert raised.value.exit_code == EXIT_AMBIGUOUS_CREATE
    assert raised.value.evidence["outcome"] == "ambiguous_create_teardown_unconfirmed"
    assert raised.value.evidence["teardown_confirmed"] is False
    assert raised.value.evidence["cleanup_error"]
    assert len(paid_creates(runner)) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "wrong-name"},
        {"computeType": "GPU", "cpuFlavorId": None},
        {"computeType": "GPU", "cpuFlavorId": "cpu3c-2-8"},
        {"adjustedCostPerHr": "0.2501"},
        {"adjustedCostPerHr": "NaN"},
        {"machine": {"secureCloud": False}},
        {"imageName": "runpod/pytorch:mutable"},
        {"networkVolume": {"id": "wrong"}},
        {"volumeMountPath": "/wrong"},
        {"terminateAfter": None},
        {"terminateAfter": "2026-07-29T12:11:00Z"},
    ],
)
def test_created_pod_attestation_rejects_any_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    operator.pod_name = "expected-name"
    runner = ProviderRunner(pod_overrides=overrides)
    runner.created_name = "expected-name"
    runner.terminate_after = "2026-07-29T12:10:00Z"
    pod = runner._pod_payload()
    image = (
        f"{operator.manifest['runtime']['image']}@{operator.manifest['runtime']['image_digest']}"
    )

    with pytest.raises(RuntimeError):
        operator._validate_created_pod(
            pod,
            expected_image=image,
            terminate_after="2026-07-29T12:10:00Z",
        )


def test_created_pod_attestation_uses_documented_cpu_fields_when_optional_fields_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    operator.pod_name = "expected-name"
    runner = ProviderRunner()
    runner.created_name = "expected-name"
    runner.terminate_after = "2026-07-29T12:10:00Z"
    pod = runner._pod_payload()
    del pod["computeType"]
    del pod["terminateAfter"]
    image = (
        f"{operator.manifest['runtime']['image']}@{operator.manifest['runtime']['image_digest']}"
    )

    assert operator._validate_created_pod(
        pod,
        expected_image=image,
        terminate_after="2026-07-29T12:10:00Z",
    ) == Decimal("0.20")


def test_created_pod_attestation_polls_missing_fields_but_not_contradictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    operator.pod_id = POD_ID
    operator.pod_name = "expected-name"
    operator.provider_deadline_monotonic = MAXIMUM_LIFETIME_SECONDS
    runner.created_name = "expected-name"
    runner.terminate_after = "2026-07-29T12:10:00Z"
    real_payload = runner._pod_payload
    reads = 0

    def initializing_payload() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        payload = real_payload()
        if reads == 1:
            payload["networkVolume"] = {"id": VOLUME_ID}
            del payload["adjustedCostPerHr"]
        return payload

    monkeypatch.setattr(runner, "_pod_payload", initializing_payload)
    image = (
        f"{operator.manifest['runtime']['image']}@{operator.manifest['runtime']['image_digest']}"
    )

    assert operator._attest_created_pod(
        expected_image=image,
        terminate_after="2026-07-29T12:10:00Z",
    ) == Decimal("0.20")
    assert reads == 2
    assert operator.clock.monotonic() == 5

    reads = 0

    def contradictory_payload() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return {**real_payload(), "volumeMountPath": "/wrong"}

    monkeypatch.setattr(runner, "_pod_payload", contradictory_payload)
    with pytest.raises(RuntimeError, match="volumeMountPath"):
        operator._attest_created_pod(
            expected_image=image,
            terminate_after="2026-07-29T12:10:00Z",
        )
    assert reads == 1


def test_torch_checkpoint_evidence_binds_exact_runtime_and_checkout() -> None:
    manifest = json.loads((ROOT / "research/studies/larger-model-eligibility.json").read_text())
    source_sha256 = {
        relative: _tagged_sha256((ROOT / relative).read_bytes())
        for relative in (
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_large_model_trainer.py",
            "research/runpod/retention_checkpoint_probe.py",
        )
    }
    source_archive_digest = "sha256:" + "9" * 64
    head_commit = "0" * 40
    evidence = {
        "status": "passed",
        "test_id": (
            "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        ),
        "profile_id": manifest["profile_id"],
        "head_commit": head_commit,
        "source_archive_digest": source_archive_digest,
        "torch_version": manifest["runtime"]["torch_version"],
        "torch_cuda_version": "12.8",
        "probe_sha256": source_sha256["research/runpod/retention_checkpoint_probe.py"],
        "trainer_sha256": source_sha256["research/runpod/repository_repair_large_model_trainer.py"],
        "environment_sha256": source_sha256["research/runpod/repository_repair_env.py"],
        "source_sha256": {
            name: f"sha256:{digest}"
            for name, digest in manifest["source_contract"]["files"].items()
        },
        "checkpoint_sha256": "sha256:" + "8" * 64,
        "checkpoint_size_bytes": 1024,
        "restored_weight_before_resume_step": [0.99],
        "advanced_weight_after_resume_step": [0.98],
        "effective_policy_update_count_before_resume_step": 1,
        "effective_policy_update_count_after_resume_step": 2,
        "retained_observation_after_resume_step": {"exact_rate": 0.75},
        "optimizer_state_entries_after_resume_step": 1,
        "optimizer_state_digest_before_persist": "sha256:" + "6" * 64,
        "optimizer_state_digest_after_restore": "sha256:" + "6" * 64,
        "optimizer_state_digest_after_resume_step": "sha256:" + "7" * 64,
        "optimizer_parameter_device": "cpu",
        "optimizer_state_devices_before_persist": {
            "exp_avg": ["cpu"],
            "exp_avg_sq": ["cpu"],
            "step": ["cpu"],
        },
        "optimizer_state_devices_after_restore": {
            "exp_avg": ["cpu"],
            "exp_avg_sq": ["cpu"],
            "step": ["cpu"],
        },
    }

    assert (
        _verify_torch_evidence(
            json.dumps(evidence).encode(),
            manifest,
            head_commit=head_commit,
            source_archive_digest=source_archive_digest,
            source_sha256=source_sha256,
        )
        == evidence
    )
    for field, value in (
        ("status", "failed"),
        ("head_commit", "wrong"),
        ("torch_version", "2.8.0"),
        ("torch_cuda_version", "12.1"),
        ("probe_sha256", "sha256:" + "0" * 64),
        ("checkpoint_sha256", "sha256:bad"),
        ("checkpoint_size_bytes", 0),
        ("effective_policy_update_count_after_resume_step", 1),
        ("optimizer_state_entries_after_resume_step", 0),
        ("optimizer_state_digest_before_persist", "sha256:bad"),
        ("optimizer_state_digest_after_restore", "sha256:" + "5" * 64),
        ("optimizer_state_digest_after_resume_step", "sha256:" + "6" * 64),
    ):
        invalid = {**evidence, field: value}
        with pytest.raises(RuntimeError, match="evidence is invalid"):
            _verify_torch_evidence(
                json.dumps(invalid).encode(),
                manifest,
                head_commit=head_commit,
                source_archive_digest=source_archive_digest,
                source_sha256=source_sha256,
            )


def test_remote_stage_uses_exact_transport_and_verifies_retrieved_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = json.loads((ROOT / "research/studies/larger-model-eligibility.json").read_text())
    plan = fake_bundle_plan(manifest["profile_id"])
    torch_payload = json.dumps(
        valid_torch_evidence(manifest, plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    inbound = {
        "model-receipt.json": b'{"receipt_digest":"sha256:model"}',
        "stage-receipt.json": b'{"receipt_digest":"sha256:stage"}',
        "torch-retention.log": torch_payload,
        "torch-retention.sha256": (
            f"{_tagged_sha256(torch_payload).removeprefix('sha256:')}  torch-retention.log\n"
        ).encode(),
    }
    transport_root = tmp_path / "transport"
    transport_root.mkdir()
    runner = TransportRunner(transport_root, inbound)
    operator = make_operator(monkeypatch, tmp_path, runner)
    operator.pod_id = POD_ID
    operator.provider_deadline_monotonic = 10_000
    verified: list[str] = []
    monkeypatch.setattr(
        volume_stage_module,
        "verify_volume_readiness_receipt",
        lambda *args, **kwargs: verified.append("model"),
    )
    monkeypatch.setattr(
        volume_stage_module,
        "verify_bundle_stage_receipt",
        lambda *args, **kwargs: verified.append("stage"),
    )
    local_root = tmp_path / "retrieved"
    local_root.mkdir()

    artifacts = operator._perform_remote_stage(
        plan,
        local_root,
        head_commit=plan.head_commit,
    )

    assert artifacts.model_receipt == inbound["model-receipt.json"]
    assert artifacts.stage_receipt == inbound["stage-receipt.json"]
    assert artifacts.torch_log == torch_payload
    assert verified == ["model", "stage"]
    outbound = [call for call in runner.calls if call[0] == "scp" and call[-1].startswith("root@")]
    assert len(outbound) == 1
    assert "-P" in outbound[0]
    assert "-p" not in outbound[0]
    remote_commands = "\n".join(call[-1] for call in runner.calls if call[0] == "ssh")
    assert plan.source_archive_digest in remote_commands
    assert "findmnt -rn -M /workspace -o TARGET" in remote_commands
    assert "test ! -L /workspace" in remote_commands
    assert "stage-mounted" in remote_commands
    assert f"--head-commit {plan.head_commit}" in remote_commands
    assert "--source-archive" in remote_commands
    assert "create-volume-receipt" in remote_commands
    assert "H100" not in remote_commands


def test_verified_artifacts_are_installed_only_after_three_teardown_polls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    model = json.dumps({"receipt_digest": "sha256:" + "a" * 64}).encode()
    stage = json.dumps({"receipt_digest": "sha256:" + "b" * 64}).encode()
    torch_log = b'{"status":"passed"}'
    artifacts = RemoteArtifacts(
        model_receipt=model,
        stage_receipt=stage,
        torch_log=torch_log,
        torch_digest=_tagged_sha256(torch_log),
    )
    model_path = tmp_path / f"larger-model-volume-{VOLUME_ID}.json"
    stage_path = tmp_path / f"larger-model-bundle-stage-{VOLUME_ID}.json"

    def remote_stage(
        plan: BundlePlan,
        temporary_root: Path,
        *,
        head_commit: str,
    ) -> RemoteArtifacts:
        del plan, temporary_root, head_commit
        assert not model_path.exists()
        assert not stage_path.exists()
        return artifacts

    monkeypatch.setattr(operator, "_perform_remote_stage", remote_stage)
    original_install = operator._install_artifacts
    calls_at_publication: list[int] = []

    def install_after_teardown(
        received: RemoteArtifacts,
        *,
        attempt_id: str,
    ) -> dict[str, str]:
        assert runner.pod_list_count_at_delete is not None
        assert runner.pod_list_calls - runner.pod_list_count_at_delete == 3
        assert not model_path.exists()
        assert not stage_path.exists()
        calls_at_publication.append(len(runner.calls))
        return original_install(received, attempt_id=attempt_id)

    monkeypatch.setattr(operator, "_install_artifacts", install_after_teardown)

    result = operator.execute()

    assert result["outcome"] == "staged"
    assert result["teardown_confirmed"] is True
    assert result["gpu_fallback_used"] is False
    assert result["hourly_rate_usd"] == "0.20"
    assert runner.deleted_ids == [POD_ID]
    assert len(paid_creates(runner)) == 1
    assert calls_at_publication == [len(runner.calls)]
    assert model_path.read_bytes() == model
    assert stage_path.read_bytes() == stage
    assert Path(result["torch_evidence_path"]).read_bytes() == torch_log


def test_successful_main_persists_attempt_then_releases_shared_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    torch_log = b'{"status":"passed"}'
    artifacts = RemoteArtifacts(
        model_receipt=json.dumps({"receipt_digest": "sha256:" + "a" * 64}).encode(),
        stage_receipt=json.dumps({"receipt_digest": "sha256:" + "b" * 64}).encode(),
        torch_log=torch_log,
        torch_digest=_tagged_sha256(torch_log),
    )
    monkeypatch.setattr(
        operator,
        "_perform_remote_stage",
        lambda plan, temporary_root, *, head_commit: artifacts,
    )
    monkeypatch.setattr(
        volume_stage_module,
        "VolumeStageOperator",
        lambda repository_root: operator,
    )

    assert volume_stage_module.main([]) == 0

    assert not operator.lease_directory.exists()
    attempts = list(tmp_path.glob("*.volume-stage-attempt.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["outcome"] == "staged"


def test_remote_failure_attempt_retains_three_poll_teardown_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)

    def remote_failure(
        plan: BundlePlan,
        temporary_root: Path,
        *,
        head_commit: str,
    ) -> RemoteArtifacts:
        del plan, temporary_root, head_commit
        raise RuntimeError("injected remote stage failure")

    monkeypatch.setattr(operator, "_perform_remote_stage", remote_failure)
    monkeypatch.setattr(
        volume_stage_module,
        "VolumeStageOperator",
        lambda repository_root: operator,
    )

    with pytest.raises(RuntimeError, match="injected remote stage failure"):
        volume_stage_module.main([])

    attempts = list(tmp_path.glob("*.volume-stage-attempt.json"))
    assert len(attempts) == 1
    evidence = json.loads(attempts[0].read_text())
    assert evidence["teardown_confirmed"] is True
    assert len(evidence["idle_polls"]) == 3
    assert all(poll["pod_count"] == "0" for poll in evidence["idle_polls"])
    assert evidence["baseline_balance_usd"] == "10.00"
    assert evidence["attested_hourly_rate_usd"] == "0.20"
    assert not operator.lease_directory.exists()


@pytest.mark.parametrize("failed_replacement", [1, 2, 3, 4])
def test_artifact_set_publication_rolls_back_every_partial_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_replacement: int,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    attempt_id = "equinox-volume-stage-fault-injection"
    torch_log = b'{"status":"passed"}'
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=torch_log,
        torch_digest=_tagged_sha256(torch_log),
    )
    destinations = [
        tmp_path / f"larger-model-volume-{VOLUME_ID}.json",
        tmp_path / f"larger-model-bundle-stage-{VOLUME_ID}.json",
        tmp_path / f"{attempt_id}.torch-retention.json",
        tmp_path / f"{attempt_id}.torch-retention.sha256",
    ]
    originals = {
        destination: f"old-{index}".encode() for index, destination in enumerate(destinations)
    }
    for destination, payload in originals.items():
        destination.write_bytes(payload)
    prepare_publication_lease(operator, attempt_id=attempt_id)
    replacements = 0
    real_replace = operator._replace_artifact

    def fail_one_replace(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == failed_replacement:
            raise OSError("injected replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(operator, "_replace_artifact", fail_one_replace)

    with pytest.raises(OSError, match="injected replacement failure"):
        operator._install_artifacts(artifacts, attempt_id=attempt_id)

    assert {destination: destination.read_bytes() for destination in destinations} == originals
    assert operator.publication_recovery_required is False
    assert not list(tmp_path.glob(".*.artifact-transaction"))


def test_unsafe_destination_during_preparation_removes_unpublished_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    attempt_id = "equinox-volume-stage-unsafe-preparation"
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=b'{"status":"passed"}',
        torch_digest=_tagged_sha256(b'{"status":"passed"}'),
    )
    (tmp_path / f"larger-model-volume-{VOLUME_ID}.json").mkdir()
    prepare_publication_lease(operator, attempt_id=attempt_id)

    with pytest.raises(RuntimeError, match="existing artifact path is unsafe"):
        operator._install_artifacts(artifacts, attempt_id=attempt_id)

    assert operator.publication_recovery_required is False
    assert not list(tmp_path.glob(".*.artifact-transaction"))
    assert json.loads(operator.lease_path.read_text())["state"] == "preallocation"


def test_conflicting_prior_receipt_archive_fails_closed_without_losing_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    attempt_id = "equinox-volume-stage-prior-archive-conflict"
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=b'{"status":"passed"}',
        torch_digest=_tagged_sha256(b'{"status":"passed"}'),
    )
    model_path = tmp_path / f"larger-model-volume-{VOLUME_ID}.json"
    current = b'{"profile_id":"older-profile@7","receipt":"current"}'
    archived = b'{"profile_id":"older-profile@7","receipt":"already-archived"}'
    archive_path = tmp_path / f"larger-model-volume-{VOLUME_ID}.profile-v7.json"
    model_path.write_bytes(current)
    archive_path.write_bytes(archived)
    prepare_publication_lease(operator, attempt_id=attempt_id)

    with pytest.raises(RuntimeError, match="archive already contains different bytes"):
        operator._install_artifacts(artifacts, attempt_id=attempt_id)

    assert model_path.read_bytes() == current
    assert archive_path.read_bytes() == archived
    assert operator.publication_recovery_required is False
    assert not list(tmp_path.glob(".*.artifact-transaction"))


def test_first_publication_lease_write_failure_retains_recoverable_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    attempt_id = "equinox-volume-stage-first-lease-write-fault"
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=b'{"status":"passed"}',
        torch_digest=_tagged_sha256(b'{"status":"passed"}'),
    )
    prepare_publication_lease(operator, attempt_id=attempt_id)
    real_persist = operator._persist_lease
    failed_once = False

    def fail_first_publication_write(
        *,
        state: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        nonlocal failed_once
        if state == "publishing_artifacts" and not failed_once:
            failed_once = True
            raise OSError("injected first publication lease write failure")
        real_persist(state=state, evidence=evidence)

    monkeypatch.setattr(operator, "_persist_lease", fail_first_publication_write)

    with pytest.raises(OSError, match="first publication lease write failure"):
        operator._install_artifacts(artifacts, attempt_id=attempt_id)

    assert operator.publication_recovery_required is True
    assert operator.publication_state == "publishing_artifacts"
    assert operator.publication_transaction_evidence["artifacts"]
    transaction = tmp_path / f".{attempt_id}.artifact-transaction"
    assert (transaction / "transaction.json").is_file()
    real_persist(
        state=operator.publication_state,
        evidence=operator.publication_transaction_evidence,
    )
    recovery = make_operator(monkeypatch, tmp_path, ProviderRunner())
    monkeypatch.setattr(recovery, "_operator_process_is_alive", lambda process_id: False)

    result = recovery.recover_artifact_publication()

    assert result["outcome"] == "artifact_publication_rolled_back"
    assert not transaction.exists()
    assert not recovery.lease_directory.exists()


def test_main_publication_failure_and_recovery_preserve_both_evidence_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=b'{"status":"passed"}',
        torch_digest=_tagged_sha256(b'{"status":"passed"}'),
    )
    monkeypatch.setattr(
        operator,
        "_perform_remote_stage",
        lambda plan, temporary_root, *, head_commit: artifacts,
    )
    real_persist = operator._persist_lease
    failed_once = False

    def fail_first_publication_write(
        *,
        state: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        nonlocal failed_once
        if state == "publishing_artifacts" and not failed_once:
            failed_once = True
            raise OSError("injected main publication lease write failure")
        real_persist(state=state, evidence=evidence)

    monkeypatch.setattr(operator, "_persist_lease", fail_first_publication_write)
    monkeypatch.setattr(
        volume_stage_module,
        "VolumeStageOperator",
        lambda repository_root: operator,
    )

    with pytest.raises(OSError, match="main publication lease write failure"):
        volume_stage_module.main([])

    failed_attempt_path = tmp_path / f"{operator.pod_name}.volume-stage-attempt.json"
    failed_attempt_bytes = failed_attempt_path.read_bytes()
    assert json.loads(failed_attempt_bytes)["outcome"] == "failed"
    assert json.loads(operator.lease_path.read_text())["state"] == "publishing_artifacts"
    recovery = make_operator(monkeypatch, tmp_path, ProviderRunner())
    monkeypatch.setattr(recovery, "_operator_process_is_alive", lambda process_id: False)

    result = recovery.recover_artifact_publication()

    assert result["outcome"] == "artifact_publication_rolled_back"
    assert failed_attempt_path.read_bytes() == failed_attempt_bytes
    recovery_path = tmp_path / f"{operator.pod_name}.volume-stage-recovery.json"
    assert json.loads(recovery_path.read_text()) == result
    assert not recovery.lease_directory.exists()


def test_hard_kill_partial_publication_is_deterministically_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, first_runner)
    attempt_id = "equinox-volume-stage-simulated-hard-kill"
    artifacts = RemoteArtifacts(
        model_receipt=b'{"receipt_digest":"sha256:new-model"}',
        stage_receipt=b'{"receipt_digest":"sha256:new-stage"}',
        torch_log=b'{"status":"passed"}',
        torch_digest=_tagged_sha256(b'{"status":"passed"}'),
    )
    artifact_payloads = operator._artifact_payloads(artifacts, attempt_id=attempt_id)
    originals = {
        destination: f"original-{index}".encode()
        for index, (_, destination, _) in enumerate(artifact_payloads)
    }
    for destination, payload in originals.items():
        destination.write_bytes(payload)
    prepare_publication_lease(operator, attempt_id=attempt_id)
    transaction = tmp_path / f".{attempt_id}.artifact-transaction"
    transaction.mkdir(mode=0o700)
    descriptors: list[dict[str, Any]] = []
    for index, (role, destination, payload) in enumerate(artifact_payloads):
        new_file = transaction / f"new-{index}"
        old_file = transaction / f"old-{index}"
        new_file.write_bytes(payload)
        old_file.write_bytes(originals[destination])
        descriptors.append(
            {
                "role": role,
                "destination": destination.name,
                "new_file": new_file.name,
                "new_digest": _tagged_sha256(payload),
                "old_file": old_file.name,
                "old_digest": _tagged_sha256(originals[destination]),
            }
        )
    journal = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "state": "publishing",
        "published": [],
        "artifacts": descriptors,
    }
    (transaction / "transaction.json").write_bytes(
        volume_stage_module.canonical_json(journal) + b"\n"
    )
    operator._persist_publication_state(
        state="publishing_artifacts",
        transaction=transaction,
        descriptors=descriptors,
    )
    os.replace(transaction / "new-0", artifact_payloads[0][1])

    recovery_runner = ProviderRunner()
    recovery = make_operator(monkeypatch, tmp_path, recovery_runner)
    monkeypatch.setattr(recovery, "_operator_process_is_alive", lambda process_id: False)
    result = recovery.recover_artifact_publication()

    assert result["outcome"] == "artifact_publication_rolled_back"
    assert result["allocation_attempted"] is True
    assert result["recovery_allocation_attempted"] is False
    assert result["provider_queries_issued"] is False
    assert recovery_runner.calls == []
    assert {destination: destination.read_bytes() for destination in originals} == originals
    assert not transaction.exists()
    assert not recovery.lease_directory.exists()
    recovery_receipt = tmp_path / f"{attempt_id}.volume-stage-recovery.json"
    assert json.loads(recovery_receipt.read_text()) == result


def test_hard_kill_after_published_state_completes_verified_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = make_operator(monkeypatch, tmp_path, ProviderRunner())
    attempt_id = "equinox-volume-stage-published-hard-kill"
    plan = fake_bundle_plan(operator.manifest["profile_id"])
    operator.bundle_metadata = dict(plan.metadata)
    manifest_digest = _tagged_sha256(volume_stage_module.canonical_json(operator.manifest))
    model = {
        "schema_version": 1,
        "profile_id": operator.manifest["profile_id"],
        "model_id": operator.manifest["model"]["id"],
        "model_revision": operator.manifest["model"]["revision"],
        "manifest_digest": manifest_digest,
        "network_volume_id": VOLUME_ID,
        "network_volume_data_center_id": DATA_CENTER_ID,
        "network_volume_size_gb": operator.manifest["hardware"]["volume_disk_gb"],
        "snapshot_digest": "sha256:" + "7" * 64,
        "dependencies": operator.manifest["runtime"]["dependencies"],
        "prepared_at": "2026-07-29T12:00:00Z",
        "ready": True,
    }
    model["receipt_digest"] = _tagged_sha256(volume_stage_module.canonical_json(model))
    stage = {
        "schema_version": 1,
        "profile_id": operator.manifest["profile_id"],
        "manifest_digest": manifest_digest,
        "source_contract_digest": plan.metadata["source_contract_digest"],
        "network_volume_id": VOLUME_ID,
        "network_volume_data_center_id": DATA_CENTER_ID,
        "network_volume_size_gb": operator.manifest["hardware"]["volume_disk_gb"],
        "bundle_digest": plan.metadata["bundle_digest"],
        "bundle_size_bytes": plan.metadata["bundle_size_bytes"],
        "bundle_path": plan.metadata["bundle_path"],
        "staged_at": "2026-07-29T12:00:00Z",
        "ready": True,
    }
    stage["receipt_digest"] = _tagged_sha256(volume_stage_module.canonical_json(stage))
    torch_evidence = {
        "status": "passed",
        "profile_id": operator.manifest["profile_id"],
        "head_commit": plan.head_commit,
        "source_archive_digest": plan.source_archive_digest,
        "torch_version": operator.manifest["runtime"]["torch_version"],
        "torch_cuda_version": "12.8",
    }
    torch_payload = volume_stage_module.canonical_json(torch_evidence)
    artifacts = RemoteArtifacts(
        model_receipt=volume_stage_module.canonical_json(model),
        stage_receipt=volume_stage_module.canonical_json(stage),
        torch_log=torch_payload,
        torch_digest=_tagged_sha256(torch_payload),
    )
    prepare_publication_lease(operator, attempt_id=attempt_id)

    paths = operator._install_artifacts(artifacts, attempt_id=attempt_id)

    assert json.loads(operator.lease_path.read_text())["state"] == "artifacts_published"
    assert not list(tmp_path.glob(".*.artifact-transaction"))
    live_recovery_runner = ProviderRunner()
    live_recovery = make_operator(monkeypatch, tmp_path, live_recovery_runner)
    with pytest.raises(RuntimeError, match="still alive"):
        live_recovery.recover_artifact_publication()
    assert live_recovery_runner.calls == []
    assert operator.lease_path.is_file()
    recovery_runner = ProviderRunner()
    recovery = make_operator(monkeypatch, tmp_path, recovery_runner)
    monkeypatch.setattr(recovery, "_operator_process_is_alive", lambda process_id: False)
    result = recovery.recover_artifact_publication()

    assert result["outcome"] == "artifact_publication_completed"
    assert result["allocation_attempted"] is True
    assert result["recovery_allocation_attempted"] is False
    assert result["provider_queries_issued"] is False
    assert recovery_runner.calls == []
    assert Path(paths["model_receipt_path"]).read_bytes() == artifacts.model_receipt
    assert Path(paths["stage_receipt_path"]).read_bytes() == artifacts.stage_receipt
    assert not recovery.lease_directory.exists()
    recovery_receipt = tmp_path / f"{attempt_id}.volume-stage-recovery.json"
    assert json.loads(recovery_receipt.read_text()) == result


def test_interrupt_after_allocation_tears_down_and_preserves_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    model_path = tmp_path / f"larger-model-volume-{VOLUME_ID}.json"
    stage_path = tmp_path / f"larger-model-bundle-stage-{VOLUME_ID}.json"
    model_path.write_bytes(b"existing-model-receipt")
    stage_path.write_bytes(b"existing-stage-receipt")

    def interrupted(
        plan: BundlePlan,
        temporary_root: Path,
        *,
        head_commit: str,
    ) -> RemoteArtifacts:
        del plan, temporary_root, head_commit
        raise KeyboardInterrupt

    monkeypatch.setattr(operator, "_perform_remote_stage", interrupted)

    with pytest.raises(KeyboardInterrupt):
        operator.execute()

    assert runner.deleted_ids == [POD_ID]
    assert runner.pod_list_count_at_delete is not None
    assert runner.pod_list_calls - runner.pod_list_count_at_delete == 3
    assert model_path.read_bytes() == b"existing-model-receipt"
    assert stage_path.read_bytes() == b"existing-stage-receipt"


def test_bad_pod_attestation_deletes_before_any_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner(pod_overrides={"adjustedCostPerHr": "0.30"})
    operator = make_operator(monkeypatch, tmp_path, runner)

    with pytest.raises(RuntimeError, match="exceeds"):
        operator.execute()

    assert runner.deleted_ids == [POD_ID]
    assert not any(call[0] in {"ssh", "scp"} for call in runner.calls)
    assert len(paid_creates(runner)) == 1


def test_production_guards_pin_ten_minutes_and_five_minute_reconciliation() -> None:
    assert MAXIMUM_LIFETIME_SECONDS == 10 * 60
    assert RECONCILIATION_SECONDS == 5 * 60


def test_paid_step_budget_preserves_full_teardown_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = ProviderRunner()
    operator = make_operator(monkeypatch, tmp_path, runner)
    operator.provider_deadline_monotonic = 199

    with pytest.raises(RuntimeError, match="including teardown reserve"):
        operator._paid_command(
            "ssh",
            "example",
            timeout=20,
            step="late work",
        )

    assert runner.calls == []
    operator.provider_deadline_monotonic = 200
    operator._ensure_paid_step_budget(20, "bounded work")
