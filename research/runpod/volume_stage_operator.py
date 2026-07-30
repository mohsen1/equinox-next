"""Fail-closed RunPod CPU operator for larger-model volume preparation."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from research.runpod.larger_model_gate import (
    canonical_json,
    load_manifest,
    verify_source_contract,
    verify_volume_readiness_receipt,
)
from research.runpod.workload_bundle import (
    PROFILE_MEMBER,
    build_larger_model_bundle,
    canonical_bundle_files,
    verify_bundle_stage_receipt,
)

MAXIMUM_HOURLY_COST = Decimal("0.25")
MAXIMUM_LIFETIME_SECONDS = 10 * 60
RECONCILIATION_SECONDS = 5 * 60
POLL_SECONDS = 5
TEARDOWN_POLLS = 3
TEARDOWN_TIMEOUT_SECONDS = 2 * 60
ATTESTATION_READINESS_SECONDS = 30
DELETE_TIMEOUT_SECONDS = 30
CLEANUP_RESERVE_SECONDS = DELETE_TIMEOUT_SECONDS + TEARDOWN_TIMEOUT_SECONDS + 30
CAPACITY_REJECTION_TEXT = "There are no longer any instances available"
PINNED_MOUNT_PATH = "/workspace"
PROBE_MODULE = "research.runpod.retention_checkpoint_probe"
EXIT_CAPACITY_UNAVAILABLE = 75
EXIT_AMBIGUOUS_CREATE = 70
_SAFE_ID = re.compile(r"^[A-Za-z0-9._@-]+$")
_SAFE_POD_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_HEAD = re.compile(r"^[0-9a-f]{40,64}$")
_SAFE_SOURCE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_TAGGED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_TRANSACTION_NAME = re.compile(r"^\.[A-Za-z0-9._@-]+\.artifact-transaction$")
_FIXED_OPERATOR_SOURCES = (
    "research/__init__.py",
    "research/runpod/__init__.py",
    "research/runpod/bootstrap_server.py",
    "research/runpod/larger_model_gate.py",
    "research/runpod/repository_repair_env.py",
    "research/runpod/retention_checkpoint_probe.py",
    "research/runpod/volume_stage_operator.py",
    "research/runpod/workload_bundle.py",
    "research/studies/larger-model-eligibility.json",
    "scripts/stage-larger-model-runpod-volume",
    "scripts/stage-runpod-workload-bundle",
)


class Runner(Protocol):
    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout: int,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SubprocessRunner:
    """Run one bounded local provider or transport command."""

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout: int,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            list(arguments),
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout,
        )


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class ProviderState:
    balance: Decimal
    spend_per_hour: Decimal
    pods: tuple[dict[str, Any], ...]
    volume: dict[str, Any]


@dataclass(frozen=True)
class BundlePlan:
    payload: bytes
    metadata: dict[str, Any]
    source_archive: bytes
    source_archive_digest: str
    source_sha256: dict[str, str]
    head_commit: str


@dataclass(frozen=True)
class RemoteArtifacts:
    model_receipt: bytes
    stage_receipt: bytes
    torch_log: bytes
    torch_digest: str


class OperatorStop(RuntimeError):
    """Expected fail-closed terminal result with a stable process exit code."""

    def __init__(self, message: str, *, exit_code: int, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.exit_code = exit_code
        self.evidence = dict(evidence)


def _json_object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} did not return valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _json_array(payload: bytes, name: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} did not return valid JSON") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{name} must be an array of objects")
    return value


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} is not numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RuntimeError(f"{name} is not numeric") from error
    if not result.is_finite() or result < 0:
        raise RuntimeError(f"{name} is not a finite non-negative number")
    return result


def _tagged_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    content = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    return _tagged_sha256(canonical_json(content))


def _expected_cuda_version(torch_version: str) -> str:
    match = re.search(r"\+cu(\d{3})$", torch_version)
    if match is None:
        raise RuntimeError("manifest Torch version does not pin a CUDA build")
    digits = match.group(1)
    return f"{digits[:2]}.{digits[2]}"


def _verify_torch_evidence(
    payload: bytes,
    manifest: Mapping[str, Any],
    *,
    head_commit: str,
    source_archive_digest: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    result = _json_object(payload, "Torch retention checkpoint evidence")
    expected_keys = {
        "status",
        "test_id",
        "profile_id",
        "head_commit",
        "source_archive_digest",
        "torch_version",
        "torch_cuda_version",
        "probe_sha256",
        "trainer_sha256",
        "environment_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "restored_weight_before_resume_step",
        "advanced_weight_after_resume_step",
        "effective_policy_update_count_before_resume_step",
        "effective_policy_update_count_after_resume_step",
        "retained_observation_after_resume_step",
        "optimizer_state_entries_after_resume_step",
        "optimizer_state_digest_before_persist",
        "optimizer_state_digest_after_restore",
        "optimizer_state_digest_after_resume_step",
        "optimizer_parameter_device",
        "optimizer_state_devices_before_persist",
        "optimizer_state_devices_after_restore",
    }
    expected_contract = {
        name: f"sha256:{digest}" for name, digest in manifest["source_contract"]["files"].items()
    }
    before = result.get("restored_weight_before_resume_step")
    after = result.get("advanced_weight_after_resume_step")
    checkpoint_size = result.get("checkpoint_size_bytes")
    optimizer_entries = result.get("optimizer_state_entries_after_resume_step")
    optimizer_before = result.get("optimizer_state_digest_before_persist")
    optimizer_restored = result.get("optimizer_state_digest_after_restore")
    optimizer_advanced = result.get("optimizer_state_digest_after_resume_step")
    if (
        set(result) != expected_keys
        or result.get("status") != "passed"
        or result.get("test_id")
        != "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
        or result.get("profile_id") != manifest["profile_id"]
        or result.get("head_commit") != head_commit
        or result.get("source_archive_digest") != source_archive_digest
        or result.get("torch_version") != manifest["runtime"]["torch_version"]
        or result.get("torch_cuda_version")
        != _expected_cuda_version(manifest["runtime"]["torch_version"])
        or result.get("probe_sha256")
        != source_sha256["research/runpod/retention_checkpoint_probe.py"]
        or result.get("trainer_sha256")
        != source_sha256["research/runpod/repository_repair_large_model_trainer.py"]
        or result.get("environment_sha256")
        != source_sha256["research/runpod/repository_repair_env.py"]
        or result.get("source_sha256") != expected_contract
        or not isinstance(result.get("checkpoint_sha256"), str)
        or not _TAGGED_SHA256.fullmatch(result["checkpoint_sha256"])
        or not isinstance(optimizer_before, str)
        or not _TAGGED_SHA256.fullmatch(optimizer_before)
        or not isinstance(optimizer_restored, str)
        or not _TAGGED_SHA256.fullmatch(optimizer_restored)
        or not isinstance(optimizer_advanced, str)
        or not _TAGGED_SHA256.fullmatch(optimizer_advanced)
        or optimizer_before != optimizer_restored
        or optimizer_restored == optimizer_advanced
        or result.get("optimizer_parameter_device") != "cpu"
        or result.get("optimizer_state_devices_before_persist")
        != {"exp_avg": ["cpu"], "exp_avg_sq": ["cpu"], "step": ["cpu"]}
        or result.get("optimizer_state_devices_after_restore")
        != {"exp_avg": ["cpu"], "exp_avg_sq": ["cpu"], "step": ["cpu"]}
        or type(checkpoint_size) is not int
        or checkpoint_size <= 0
        or not isinstance(before, list)
        or len(before) != 1
        or not isinstance(before[0], int | float)
        or isinstance(before[0], bool)
        or not math.isfinite(before[0])
        or not isinstance(after, list)
        or len(after) != 1
        or not isinstance(after[0], int | float)
        or isinstance(after[0], bool)
        or not math.isfinite(after[0])
        or before == after
        or result.get("effective_policy_update_count_before_resume_step") != 1
        or result.get("effective_policy_update_count_after_resume_step") != 2
        or result.get("retained_observation_after_resume_step") != {"exact_rate": 0.75}
        or type(optimizer_entries) is not int
        or optimizer_entries <= 0
    ):
        raise RuntimeError("Torch retention checkpoint evidence is invalid")
    return result


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("operator timestamps must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    pending = Path(pending_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


class VolumeStageOperator:
    """One-allocation state machine for model receipt and bundle staging."""

    def __init__(
        self,
        repository_root: Path,
        *,
        runner: Runner | None = None,
        clock: Clock | None = None,
        proof_directory: Path | None = None,
        reconciliation_seconds: int = RECONCILIATION_SECONDS,
        poll_seconds: int = POLL_SECONDS,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.runner = runner or SubprocessRunner()
        self.clock = clock or SystemClock()
        self.proof_directory = proof_directory or (self.repository_root / "var/research-proofs")
        self.reconciliation_seconds = reconciliation_seconds
        self.poll_seconds = poll_seconds
        self.manifest = load_manifest(
            self.repository_root / "research/studies/larger-model-eligibility.json"
        )
        self.volume_id = os.environ.get("EQUINOX_RUNPOD_NETWORK_VOLUME_ID", "")
        self.data_center_id = os.environ.get("EQUINOX_RUNPOD_DATA_CENTER_IDS", "")
        if (
            not _SAFE_ID.fullmatch(self.volume_id)
            or not _SAFE_ID.fullmatch(self.data_center_id)
            or "," in self.data_center_id
        ):
            raise RuntimeError(
                "one safe EQUINOX_RUNPOD_NETWORK_VOLUME_ID and "
                "EQUINOX_RUNPOD_DATA_CENTER_IDS value are required"
            )
        configured_state = os.environ.get("EQUINOX_RUNPOD_OPERATOR_STATE_DIR")
        self.operator_state_root = (
            Path(configured_state).expanduser()
            if configured_state
            else Path("~/.local/state/equinox/runpod").expanduser()
        )
        if (
            not self.operator_state_root.is_absolute()
            or "\n" in str(self.operator_state_root)
            or "\r" in str(self.operator_state_root)
        ):
            raise RuntimeError(
                "EQUINOX_RUNPOD_OPERATOR_STATE_DIR must be an absolute single-line path"
            )
        self.lease_directory = self.operator_state_root / "operator.lock"
        self.lease_path = self.lease_directory / "lease.json"
        self.publication_recovery_lock_path = self.operator_state_root / "publication-recovery.lock"
        self.pod_id = ""
        self.pod_name = ""
        self.create_attempted = False
        self.teardown_confirmed = False
        self.lease_acquired = False
        self.bundle_metadata: dict[str, Any] = {}
        self.started_at = ""
        self.terminate_after = ""
        self.publication_recovery_required = False
        self.publication_state = ""
        self.publication_transaction_evidence: dict[str, Any] = {}
        self.idle_polls: list[dict[str, str]] = []
        self.baseline_balance_usd: str | None = None
        self.attested_hourly_rate_usd: str | None = None
        self.recovery_lock_descriptor: int | None = None
        self.provider_deadline_monotonic: float | None = None
        self.cli_contract: dict[str, str] = {}

    def _command(
        self,
        *arguments: str,
        timeout: int,
        input_bytes: bytes | None = None,
        checked: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        completed = self.runner.run(
            arguments,
            timeout=timeout,
            input_bytes=input_bytes,
        )
        if checked and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(f"{' '.join(arguments[:3])} failed: {detail[:500]}")
        return completed

    def _ensure_paid_step_budget(self, timeout: int, step: str) -> None:
        if self.provider_deadline_monotonic is None:
            raise RuntimeError("provider lifetime budget was not initialized")
        remaining = self.provider_deadline_monotonic - self.clock.monotonic()
        required = timeout + CLEANUP_RESERVE_SECONDS
        if remaining < required:
            raise RuntimeError(
                f"refusing to start {step}: {remaining:.1f}s remain, "
                f"{required}s are required including teardown reserve"
            )

    def _paid_command(
        self,
        *arguments: str,
        timeout: int,
        step: str,
        input_bytes: bytes | None = None,
        checked: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        self._ensure_paid_step_budget(timeout, step)
        return self._command(
            *arguments,
            timeout=timeout,
            input_bytes=input_bytes,
            checked=checked,
        )

    def _identity_evidence(self) -> dict[str, Any]:
        image = f"{self.manifest['runtime']['image']}@{self.manifest['runtime']['image_digest']}"
        evidence: dict[str, Any] = {
            "profile_id": self.manifest["profile_id"],
            "network_volume_id": self.volume_id,
            "network_volume_data_center_id": self.data_center_id,
            "network_volume_size_gb": self.manifest["hardware"]["volume_disk_gb"],
            "image": image,
            "maximum_hourly_cost_usd": str(MAXIMUM_HOURLY_COST),
            "maximum_lifetime_seconds": MAXIMUM_LIFETIME_SECONDS,
            "gpu_fallback_used": False,
        }
        for key in (
            "manifest_digest",
            "source_contract_digest",
            "bundle_digest",
            "bundle_size_bytes",
            "bundle_path",
            "source_archive_digest",
            "head_commit",
        ):
            if key in self.bundle_metadata:
                evidence[key] = self.bundle_metadata[key]
        evidence.update(self.cli_contract)
        return evidence

    def _lease_payload(
        self,
        *,
        state: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "proof_id": self.pod_name,
            "pod_name": self.pod_name,
            "mode": "larger-model-volume-stage",
            "operator_pid": os.getpid(),
            "started_at": self.started_at,
            "terminate_after": self.terminate_after,
            "state": state,
            **self._identity_evidence(),
            **({"evidence": dict(evidence)} if evidence is not None else {}),
        }

    def _persist_lease(
        self,
        *,
        state: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.lease_acquired:
            raise RuntimeError("shared RunPod operator lease is not held")
        if self.lease_path.exists() or self.lease_path.is_symlink():
            if self.lease_path.is_symlink() or not self.lease_path.is_file():
                raise RuntimeError("shared RunPod operator lease path is unsafe")
            current = _json_object(
                self.lease_path.read_bytes(),
                "shared RunPod operator lease",
            )
            if (
                current.get("proof_id") != self.pod_name
                or current.get("mode") != "larger-model-volume-stage"
            ):
                raise RuntimeError("shared RunPod operator lease ownership changed")
        elif state != "preallocation":
            raise RuntimeError("shared RunPod operator lease disappeared")
        _atomic_write(
            self.lease_path,
            canonical_json(self._lease_payload(state=state, evidence=evidence)) + b"\n",
        )

    def _acquire_shared_lease(self) -> None:
        self.operator_state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.operator_state_root.is_symlink() or not self.operator_state_root.is_dir():
            raise RuntimeError("shared RunPod operator state root is unsafe")
        os.chmod(self.operator_state_root, 0o700)
        try:
            self.lease_directory.mkdir(mode=0o700)
        except FileExistsError as error:
            raise RuntimeError("an unresolved shared RunPod operator lease exists") from error
        self.lease_acquired = True
        _fsync_directory(self.operator_state_root)
        try:
            self._persist_lease(state="preallocation")
        except BaseException:
            with contextlib.suppress(OSError):
                self.lease_path.unlink()
            with contextlib.suppress(OSError):
                self.lease_directory.rmdir()
            self.lease_acquired = False
            raise

    def _release_shared_lease(self) -> None:
        if not self.lease_acquired:
            return
        lease = _json_object(self.lease_path.read_bytes(), "shared RunPod operator lease")
        if lease.get("proof_id") != self.pod_name:
            raise RuntimeError("refusing to release a different RunPod operator lease")
        self.lease_path.unlink()
        _fsync_directory(self.lease_directory)
        if self.recovery_lock_descriptor is not None:
            if (
                self.publication_recovery_lock_path.is_symlink()
                or not self.publication_recovery_lock_path.is_file()
            ):
                raise RuntimeError("publication recovery lock path is unsafe")
            self.publication_recovery_lock_path.unlink()
            _fsync_directory(self.operator_state_root)
        self.lease_directory.rmdir()
        _fsync_directory(self.operator_state_root)
        if self.recovery_lock_descriptor is not None:
            os.close(self.recovery_lock_descriptor)
            self.recovery_lock_descriptor = None
        self.lease_acquired = False

    def _acquire_publication_recovery_lock(self) -> None:
        path = self.publication_recovery_lock_path
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            information = os.fstat(descriptor)
            if not stat.S_ISREG(information.st_mode):
                raise RuntimeError("publication recovery lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another publication recovery process is active") from error
        except BaseException:
            os.close(descriptor)
            raise
        self.recovery_lock_descriptor = descriptor

    def _release_publication_recovery_lock_after_error(self) -> None:
        if self.recovery_lock_descriptor is None:
            return
        descriptor = self.recovery_lock_descriptor
        try:
            descriptor_info = os.fstat(descriptor)
            with contextlib.suppress(FileNotFoundError):
                path_info = self.publication_recovery_lock_path.lstat()
                if (
                    stat.S_ISREG(path_info.st_mode)
                    and descriptor_info.st_dev == path_info.st_dev
                    and descriptor_info.st_ino == path_info.st_ino
                ):
                    self.publication_recovery_lock_path.unlink()
                    _fsync_directory(self.operator_state_root)
        finally:
            os.close(descriptor)
        self.recovery_lock_descriptor = None

    def _operator_process_is_alive(self, process_id: int) -> bool:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _provider_state(self, *, require_idle: bool) -> ProviderState:
        user = _json_object(
            self._command("runpodctl", "user", timeout=20).stdout,
            "RunPod account",
        )
        pods = _json_array(
            self._command("runpodctl", "pod", "list", "--all", timeout=20).stdout,
            "RunPod pod inventory",
        )
        volumes = _json_array(
            self._command("runpodctl", "network-volume", "list", timeout=20).stdout,
            "RunPod network-volume inventory",
        )
        if len(volumes) != 1:
            raise RuntimeError("RunPod must contain exactly one network volume")
        volume = volumes[0]
        expected_size = self.manifest["hardware"]["volume_disk_gb"]
        if (
            volume.get("id") != self.volume_id
            or volume.get("dataCenterId") != self.data_center_id
            or volume.get("size") != expected_size
        ):
            raise RuntimeError("RunPod network-volume identity changed")
        balance = _decimal(user.get("clientBalance"), "RunPod balance")
        spend = _decimal(user.get("currentSpendPerHr"), "RunPod current spend")
        maximum_idle = _decimal(
            self.manifest["provider_safety"]["maximum_storage_only_hourly_spend_usd"],
            "manifest storage-only spend",
        )
        if require_idle and pods:
            raise RuntimeError("RunPod pod inventory is not empty")
        if require_idle and spend > maximum_idle:
            raise RuntimeError("RunPod spend exceeds the storage-only ceiling")
        return ProviderState(
            balance=balance,
            spend_per_hour=spend,
            pods=tuple(pods),
            volume=volume,
        )

    def _validate_cli_contract(self) -> None:
        version = (
            self._command("runpodctl", "version", timeout=10)
            .stdout.decode("utf-8", errors="strict")
            .strip()
        )
        help_text = self._command(
            "runpodctl",
            "pod",
            "create",
            "--help",
            timeout=10,
        ).stdout.decode("utf-8", errors="strict")
        if (
            not re.fullmatch(r"runpodctl [A-Za-z0-9._-]+", version)
            or "--compute-type string" not in help_text
            or "create a cpu pod" not in help_text.lower()
            or "--cloud-type string" not in help_text
            or "--network-volume-id string" not in help_text
            or "--terminate-after string" not in help_text
            or "auto-terminate datetime" not in help_text
        ):
            raise RuntimeError(
                "installed runpodctl does not expose the reviewed CPU/deadline contract"
            )
        self.cli_contract = {
            "runpodctl_version": version.removeprefix("runpodctl "),
            "termination_deadline_contract": "absolute_datetime_flag",
        }

    def _bundle_plan(self) -> BundlePlan:
        source_archive, source_sha256, head_commit = self._deterministic_head_source_archive()
        with tempfile.TemporaryDirectory(prefix="equinox-volume-stage-head-") as directory:
            snapshot_root = Path(directory)
            with tarfile.open(
                fileobj=io.BytesIO(source_archive),
                mode="r:gz",
            ) as archive:
                members = archive.getmembers()
                if [member.name for member in members] != sorted(source_sha256) or any(
                    not member.isfile()
                    or member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    for member in members
                ):
                    raise RuntimeError("deterministic source archive cannot be materialized safely")
                for member in members:
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("deterministic source archive member is unavailable")
                    destination = snapshot_root / member.name
                    _atomic_write(
                        destination,
                        source.read(),
                        mode=member.mode,
                    )
            snapshot_manifest = load_manifest(
                snapshot_root / "research/studies/larger-model-eligibility.json"
            )
            if canonical_json(snapshot_manifest) != canonical_json(self.manifest):
                raise RuntimeError("in-memory manifest differs from the immutable HEAD snapshot")
            verify_source_contract(
                snapshot_manifest,
                snapshot_root / "research/runpod",
            )
            payload, metadata = build_larger_model_bundle(
                snapshot_root,
                snapshot_manifest,
            )
        if (
            metadata["profile_id"] != self.manifest["profile_id"]
            or metadata["bundle_digest"] != _tagged_sha256(payload)
            or metadata["bundle_size_bytes"] != len(payload)
        ):
            raise RuntimeError("canonical workload bundle identity is inconsistent")
        metadata = {
            **metadata,
            "source_archive_digest": _tagged_sha256(source_archive),
            "head_commit": head_commit,
        }
        return BundlePlan(
            payload=payload,
            metadata=metadata,
            source_archive=source_archive,
            source_archive_digest=metadata["source_archive_digest"],
            source_sha256=source_sha256,
            head_commit=head_commit,
        )

    def _deterministic_head_source_archive(
        self,
    ) -> tuple[bytes, dict[str, str], str]:
        head_commit = (
            self._command("git", "rev-parse", "HEAD", timeout=20)
            .stdout.decode("ascii", errors="strict")
            .strip()
        )
        if not _SAFE_HEAD.fullmatch(head_commit):
            raise RuntimeError("Git HEAD is not a full immutable commit identity")
        source_names = self.manifest["source_contract"]["files"]
        if not isinstance(source_names, Mapping) or not all(
            isinstance(name, str) and _SAFE_SOURCE_NAME.fullmatch(name) for name in source_names
        ):
            raise RuntimeError("manifest source-contract paths are unsafe")
        relative_paths = sorted(
            {
                *_FIXED_OPERATOR_SOURCES,
                *(f"research/runpod/{name}" for name in source_names),
                *(
                    (
                        f"research/studies/{member}"
                        if member == PROFILE_MEMBER
                        else f"research/runpod/{member}"
                    )
                    for member in canonical_bundle_files()
                ),
            }
        )
        source_payloads: dict[str, bytes] = {}
        source_sha256: dict[str, str] = {}
        for relative in relative_paths:
            local_path = self.repository_root / relative
            if not local_path.is_file() or local_path.is_symlink():
                raise RuntimeError(f"operator source is unavailable or unsafe: {relative}")
            local_payload = local_path.read_bytes()
            committed_payload = self._command(
                "git",
                "show",
                f"{head_commit}:{relative}",
                timeout=30,
            ).stdout
            if local_payload != committed_payload:
                raise RuntimeError(f"operator source differs from Git HEAD: {relative}")
            source_payloads[relative] = local_payload
            source_sha256[relative] = _tagged_sha256(local_payload)
        tar_stream = io.BytesIO()
        with tarfile.open(
            fileobj=tar_stream,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for relative in relative_paths:
                payload = source_payloads[relative]
                information = tarfile.TarInfo(relative)
                information.size = len(payload)
                information.mode = 0o755 if relative.startswith("scripts/") else 0o644
                information.mtime = 0
                information.uid = 0
                information.gid = 0
                information.uname = ""
                information.gname = ""
                archive.addfile(information, io.BytesIO(payload))
        compressed = io.BytesIO()
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=compressed,
            mtime=0,
        ) as gzip_stream:
            gzip_stream.write(tar_stream.getvalue())
        return compressed.getvalue(), source_sha256, head_commit

    def preflight(self) -> dict[str, Any]:
        if self.lease_directory.exists() or self.lease_directory.is_symlink():
            raise RuntimeError("an unresolved shared RunPod operator lease exists")
        self._validate_cli_contract()
        state = self._provider_state(require_idle=True)
        bundle = self._bundle_plan()
        self.bundle_metadata = dict(bundle.metadata)
        return {
            "schema_version": 1,
            "outcome": "preflight_passed",
            "allocation_attempted": False,
            **self._identity_evidence(),
            "ambiguity_reconciliation_seconds": RECONCILIATION_SECONDS,
            "pod_count": 0,
            "storage_only_spend_usd_per_hour": str(state.spend_per_hour),
            "balance_usd": str(state.balance),
            "gpu_fallback_allowed": False,
        }

    def _unique_name(self) -> str:
        stamp = self.clock.now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"equinox-volume-stage-{stamp}-{secrets.token_hex(4)}"

    def _create_arguments(self, name: str, terminate_after: str) -> list[str]:
        image = f"{self.manifest['runtime']['image']}@{self.manifest['runtime']['image_digest']}"
        return [
            "runpodctl",
            "pod",
            "create",
            "--name",
            name,
            "--compute-type",
            "cpu",
            "--cloud-type",
            "SECURE",
            "--image",
            image,
            "--container-disk-in-gb",
            "5",
            "--network-volume-id",
            self.volume_id,
            "--data-center-ids",
            self.data_center_id,
            "--volume-mount-path",
            PINNED_MOUNT_PATH,
            "--ports",
            "22/tcp",
            "--ssh=true",
            "--terminate-after",
            terminate_after,
        ]

    def _exact_name_pods(self, name: str) -> list[dict[str, Any]]:
        pods = _json_array(
            self._command("runpodctl", "pod", "list", "--all", timeout=20).stdout,
            "RunPod pod inventory",
        )
        return [pod for pod in pods if pod.get("name") == name]

    def _delete_exact(self, pod_id: str) -> None:
        if not _SAFE_POD_ID.fullmatch(pod_id):
            raise RuntimeError("refusing to delete an invalid pod identity")
        # Inventory polling below decides whether teardown succeeded.
        with contextlib.suppress(Exception, KeyboardInterrupt):
            self._command(
                "runpodctl",
                "pod",
                "delete",
                pod_id,
                timeout=DELETE_TIMEOUT_SECONDS,
                checked=False,
            )

    def _safety_sleep(self, seconds: float) -> None:
        while True:
            try:
                self.clock.sleep(seconds)
            except KeyboardInterrupt:
                continue
            return

    def _confirm_idle_polls(self, count: int) -> list[dict[str, str]]:
        evidence: list[dict[str, str]] = []
        consecutive = 0
        deadline = self.clock.monotonic() + TEARDOWN_TIMEOUT_SECONDS
        while self.clock.monotonic() <= deadline:
            try:
                state = self._provider_state(require_idle=True)
            except (Exception, KeyboardInterrupt):
                consecutive = 0
            else:
                consecutive += 1
                evidence.append(
                    {
                        "observed_at": _utc_text(self.clock.now()),
                        "pod_count": "0",
                        "network_volume_id": self.volume_id,
                        "balance_usd": str(state.balance),
                        "spend_usd_per_hour": str(state.spend_per_hour),
                    }
                )
                if consecutive == count:
                    self.teardown_confirmed = True
                    return evidence[-count:]
            self._safety_sleep(self.poll_seconds)
        raise RuntimeError("RunPod idle state was not confirmed")

    def _delete_and_confirm(
        self,
        pod_ids: Sequence[str],
        *,
        pod_name: str,
    ) -> list[dict[str, str]]:
        safe_ids = tuple(dict.fromkeys(pod_ids))
        if not safe_ids or any(not _SAFE_POD_ID.fullmatch(item) for item in safe_ids):
            raise RuntimeError("refusing teardown without an exact pod identity")
        for pod_id in safe_ids:
            self._delete_exact(pod_id)
        evidence: list[dict[str, str]] = []
        consecutive = 0
        deadline = self.clock.monotonic() + TEARDOWN_TIMEOUT_SECONDS
        while self.clock.monotonic() <= deadline:
            try:
                state = self._provider_state(require_idle=False)
            except (Exception, KeyboardInterrupt):
                consecutive = 0
            else:
                matching = [
                    pod
                    for pod in state.pods
                    if pod.get("id") in safe_ids or pod.get("name") == pod_name
                ]
                if matching:
                    consecutive = 0
                    for pod in matching:
                        observed_id = pod.get("id")
                        if isinstance(observed_id, str) and _SAFE_POD_ID.fullmatch(observed_id):
                            self._delete_exact(observed_id)
                elif state.pods:
                    raise RuntimeError("an unrelated RunPod pod appeared during teardown")
                else:
                    maximum_idle = _decimal(
                        self.manifest["provider_safety"]["maximum_storage_only_hourly_spend_usd"],
                        "manifest storage-only spend",
                    )
                    if state.spend_per_hour > maximum_idle:
                        consecutive = 0
                    else:
                        consecutive += 1
                        evidence.append(
                            {
                                "observed_at": _utc_text(self.clock.now()),
                                "pod_count": "0",
                                "network_volume_id": self.volume_id,
                                "balance_usd": str(state.balance),
                                "spend_usd_per_hour": str(state.spend_per_hour),
                            }
                        )
                        if consecutive == TEARDOWN_POLLS:
                            self.teardown_confirmed = True
                            return evidence[-TEARDOWN_POLLS:]
            self._safety_sleep(self.poll_seconds)
        raise RuntimeError("exact RunPod pod teardown was not confirmed")

    def _reconcile_failed_create(
        self,
        *,
        create: subprocess.CompletedProcess[bytes],
        known_capacity_rejection: bool,
        started_at: str,
        terminate_after: str,
        initial_pod_ids: Sequence[str] = (),
    ) -> None:
        observed_pod_ids = set(initial_pod_ids)
        if any(not _SAFE_POD_ID.fullmatch(item) for item in observed_pod_ids):
            raise RuntimeError("create returned an invalid pod identity")
        unsafe_pod_identity_seen = False
        for pod_id in observed_pod_ids:
            self._delete_exact(pod_id)
        deadline = self.clock.monotonic() + self.reconciliation_seconds
        queries = 0
        query_failures = 0
        while self.clock.monotonic() < deadline:
            try:
                matches = self._exact_name_pods(self.pod_name)
            except (Exception, KeyboardInterrupt):
                matches = []
                query_failures += 1
            else:
                queries += 1
            if matches:
                pod_ids = tuple(
                    pod_id
                    for pod in matches
                    if isinstance((pod_id := pod.get("id")), str) and _SAFE_POD_ID.fullmatch(pod_id)
                )
                if len(pod_ids) != len(matches):
                    unsafe_pod_identity_seen = True
                observed_pod_ids.update(pod_ids)
                for pod_id in pod_ids:
                    self._delete_exact(pod_id)
            self._safety_sleep(self.poll_seconds)
        common: dict[str, Any] = {
            "schema_version": 1,
            "allocation_attempted": True,
            "allocation_confirmed": bool(observed_pod_ids or unsafe_pod_identity_seen),
            "pod_ids": sorted(observed_pod_ids),
            "pod_name": self.pod_name,
            "started_at": started_at,
            "terminate_after": terminate_after,
            "create_exit_code": create.returncode,
            "reconciliation_seconds": self.reconciliation_seconds,
            "reconciliation_queries": queries,
            "reconciliation_query_failures": query_failures,
            "unsafe_pod_identity_seen": unsafe_pod_identity_seen,
            "provider_error": (create.stderr or create.stdout).decode("utf-8", errors="replace")[
                :2_000
            ],
            **self._identity_evidence(),
        }
        try:
            if observed_pod_ids:
                idle = self._delete_and_confirm(
                    tuple(sorted(observed_pod_ids)),
                    pod_name=self.pod_name,
                )
            else:
                idle = self._confirm_idle_polls(TEARDOWN_POLLS)
        except (Exception, KeyboardInterrupt) as error:
            raise OperatorStop(
                "RunPod create ambiguity could not be reconciled safely",
                exit_code=EXIT_AMBIGUOUS_CREATE,
                evidence={
                    **common,
                    "outcome": "ambiguous_create_teardown_unconfirmed",
                    "teardown_confirmed": False,
                    "idle_polls": [],
                    "cleanup_error": f"{type(error).__name__}: {error}",
                },
            ) from error
        common.update(
            {
                "teardown_confirmed": True,
                "idle_polls": idle,
            }
        )
        if unsafe_pod_identity_seen:
            raise OperatorStop(
                "ambiguous create exposed an unsafe provider identity",
                exit_code=EXIT_AMBIGUOUS_CREATE,
                evidence={
                    **common,
                    "outcome": "ambiguous_create_unsafe_identity",
                },
            )
        if observed_pod_ids:
            raise OperatorStop(
                "ambiguous create exposed an allocation that was deleted",
                exit_code=EXIT_AMBIGUOUS_CREATE,
                evidence={
                    **common,
                    "outcome": "ambiguous_create_allocated_then_deleted",
                },
            )
        if known_capacity_rejection and query_failures == 0 and queries > 0:
            raise OperatorStop(
                "RunPod confirmed zero-allocation CPU capacity rejection",
                exit_code=EXIT_CAPACITY_UNAVAILABLE,
                evidence={
                    **common,
                    "outcome": "capacity_unavailable_confirmed_zero_allocation",
                },
            )
        raise OperatorStop(
            "RunPod pod creation remained ambiguous after reconciliation",
            exit_code=EXIT_AMBIGUOUS_CREATE,
            evidence={**common, "outcome": "ambiguous_create_no_identity"},
        )

    def _validate_created_pod(
        self,
        pod: Mapping[str, Any],
        *,
        expected_image: str,
        terminate_after: str,
    ) -> Decimal:
        self._reject_created_pod_contradictions(
            pod,
            expected_image=expected_image,
            terminate_after=terminate_after,
        )
        machine = pod.get("machine")
        volume = pod.get("networkVolume")
        cpu_flavor = pod.get("cpuFlavorId")
        vcpu_count = pod.get("vcpuCount")
        machine_cpu_count = machine.get("cpuCount") if isinstance(machine, Mapping) else None
        cost = pod.get(
            "adjustedCostPerHr",
            pod.get("costPerHr", pod.get("costPerHour")),
        )
        if (
            pod.get("name") != self.pod_name
            or not isinstance(cpu_flavor, str)
            or not _SAFE_ID.fullmatch(cpu_flavor)
            or not (
                (type(vcpu_count) is int and vcpu_count > 0)
                or (type(machine_cpu_count) is int and machine_cpu_count > 0)
            )
            or not isinstance(machine, Mapping)
            or machine.get("secureCloud") is not True
            or pod.get("imageName", pod.get("image")) != expected_image
            or pod.get("containerDiskInGb") != 5
            or not isinstance(volume, Mapping)
            or volume.get("id") != self.volume_id
            or volume.get("dataCenterId") != self.data_center_id
            or volume.get("size") != self.manifest["hardware"]["volume_disk_gb"]
            or pod.get("volumeMountPath") != PINNED_MOUNT_PATH
            or cost is None
        ):
            raise RuntimeError("RunPod CPU pod attestation is incomplete")
        rate = _decimal(
            cost,
            "RunPod pod hourly cost",
        )
        if rate > MAXIMUM_HOURLY_COST:
            raise RuntimeError("RunPod CPU hourly cost exceeds $0.25")
        return rate

    def _reject_created_pod_contradictions(
        self,
        pod: Mapping[str, Any],
        *,
        expected_image: str,
        terminate_after: str,
    ) -> None:
        exact_if_present = {
            "name": self.pod_name,
            "computeType": "CPU",
            "containerDiskInGb": 5,
            "volumeMountPath": PINNED_MOUNT_PATH,
            "terminateAfter": terminate_after,
        }
        for field, expected in exact_if_present.items():
            if field in pod and pod[field] != expected:
                raise RuntimeError(f"RunPod pod contradicted requested {field}")
        image = pod.get("imageName", pod.get("image"))
        if image is not None and image != expected_image:
            raise RuntimeError("RunPod image does not match its pinned digest")
        cost = pod.get("adjustedCostPerHr", pod.get("costPerHr", pod.get("costPerHour")))
        if cost is not None and _decimal(cost, "RunPod pod hourly cost") > MAXIMUM_HOURLY_COST:
            raise RuntimeError("RunPod CPU hourly cost exceeds $0.25")
        cpu_flavor = pod.get("cpuFlavorId")
        if cpu_flavor is not None and (
            not isinstance(cpu_flavor, str) or not _SAFE_ID.fullmatch(cpu_flavor)
        ):
            raise RuntimeError("RunPod CPU flavor identity is invalid")
        gpu = pod.get("gpu")
        if isinstance(gpu, Mapping) and (
            gpu.get("id") not in {None, ""}
            or (type(gpu.get("count")) is int and gpu.get("count") > 0)
        ):
            raise RuntimeError("RunPod returned GPU allocation evidence for a CPU request")
        if pod.get("gpuCount") not in {None, 0}:
            raise RuntimeError("RunPod returned a GPU count for a CPU request")
        machine = pod.get("machine")
        if machine is not None:
            if not isinstance(machine, Mapping):
                raise RuntimeError("RunPod machine attestation is malformed")
            if machine.get("secureCloud") is False:
                raise RuntimeError("RunPod did not attest Secure Cloud")
            if machine.get("gpuTypeId") not in {None, ""}:
                raise RuntimeError("RunPod machine attests GPU compute")
            if machine.get("gpuType") not in (None, {}):
                raise RuntimeError("RunPod machine includes a GPU type")
        volume = pod.get("networkVolume")
        if volume is not None:
            if not isinstance(volume, Mapping):
                raise RuntimeError("RunPod network volume attestation is malformed")
            for field, expected in (
                ("id", self.volume_id),
                ("dataCenterId", self.data_center_id),
                ("size", self.manifest["hardware"]["volume_disk_gb"]),
            ):
                if field in volume and volume[field] != expected:
                    raise RuntimeError("RunPod pod has the wrong network volume")

    def _attest_created_pod(
        self,
        *,
        expected_image: str,
        terminate_after: str,
    ) -> Decimal:
        deadline = self.clock.monotonic() + ATTESTATION_READINESS_SECONDS
        last_error = "pod attestation fields were not ready"
        while self.clock.monotonic() <= deadline:
            completed = self._paid_command(
                "runpodctl",
                "pod",
                "get",
                self.pod_id,
                "--include-machine",
                "--include-network-volume",
                timeout=20,
                step="pod attestation query",
                checked=False,
            )
            if completed.returncode == 0:
                try:
                    pod = _json_object(completed.stdout, "RunPod pod")
                    self._reject_created_pod_contradictions(
                        pod,
                        expected_image=expected_image,
                        terminate_after=terminate_after,
                    )
                    return self._validate_created_pod(
                        pod,
                        expected_image=expected_image,
                        terminate_after=terminate_after,
                    )
                except RuntimeError as error:
                    if "incomplete" not in str(error):
                        raise
                    last_error = str(error)
            else:
                last_error = "RunPod pod attestation query was not ready"
            if self.clock.monotonic() == deadline:
                break
            self._safety_sleep(min(self.poll_seconds, deadline - self.clock.monotonic()))
        raise RuntimeError(f"RunPod CPU pod attestation did not become ready: {last_error}")

    def _ssh_parts(self, pod_id: str) -> tuple[str, int, Path]:
        self._ensure_paid_step_budget(4 * 60, "SSH readiness")
        deadline = self.clock.monotonic() + 4 * 60
        while self.clock.monotonic() < deadline:
            completed = self._paid_command(
                "runpodctl",
                "ssh",
                "info",
                pod_id,
                timeout=20,
                step="SSH readiness query",
                checked=False,
            )
            if completed.returncode == 0:
                try:
                    payload = _json_object(completed.stdout, "RunPod SSH info")
                    parts = shlex.split(str(payload["sshCommand"]))
                    target = next(
                        part for part in parts if "@" in part and not part.startswith("-")
                    )
                    port = int(parts[parts.index("-p") + 1])
                    key = (
                        Path(parts[parts.index("-i") + 1]).expanduser()
                        if "-i" in parts
                        else Path("~/.runpod/ssh/runpodctl-ssh-key").expanduser()
                    )
                except (KeyError, ValueError, StopIteration):
                    pass
                else:
                    if key.is_file() and 1 <= port <= 65_535:
                        return target, port, key
            self._safety_sleep(self.poll_seconds)
        raise RuntimeError("RunPod CPU pod did not expose SSH")

    def _ssh_options(self, port: int, key: Path, known_hosts: Path) -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            "-i",
            str(key),
            "-p",
            str(port),
        ]

    def _remote_command(
        self,
        target: str,
        options: Sequence[str],
        command: str,
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        return self._paid_command(
            "ssh",
            *options,
            target,
            command,
            timeout=timeout,
            step="remote staging command",
        )

    def _perform_remote_stage(
        self,
        plan: BundlePlan,
        temporary_root: Path,
        *,
        head_commit: str,
    ) -> RemoteArtifacts:
        target, port, key = self._ssh_parts(self.pod_id)
        known_hosts = temporary_root / "known_hosts"
        options = self._ssh_options(port, key, known_hosts)
        self._remote_command(target, options, "true", timeout=25)
        remote_root = f"/tmp/equinox-volume-stage-{secrets.token_hex(6)}"
        quoted_root = shlex.quote(remote_root)
        self._remote_command(
            target,
            options,
            f"umask 077; mkdir -p {quoted_root}",
            timeout=30,
        )
        archive = temporary_root / "operator-sources.tgz"
        bundle = temporary_root / "workload-bundle.tar.xz"
        metadata = temporary_root / "bundle-metadata.json"
        archive.write_bytes(plan.source_archive)
        bundle.write_bytes(plan.payload)
        metadata.write_bytes(canonical_json(plan.metadata) + b"\n")
        scp_options = [*options[:-2], "-P", str(port)]
        self._paid_command(
            "scp",
            *scp_options,
            str(archive),
            str(bundle),
            str(metadata),
            f"{target}:{remote_root}/",
            timeout=90,
            step="source and bundle upload",
        )
        self._remote_command(
            target,
            options,
            (
                "set -euo pipefail; "
                f'test "$(sha256sum {quoted_root}/operator-sources.tgz '
                f'| awk \'{{print "sha256:"$1}}\')" = '
                f"{shlex.quote(plan.source_archive_digest)}; "
                f"tar -xzf {quoted_root}/operator-sources.tgz -C {quoted_root}; "
                f'test "$(sha256sum {quoted_root}/workload-bundle.tar.xz '
                f'| awk \'{{print "sha256:"$1}}\')" = '
                f"{shlex.quote(plan.metadata['bundle_digest'])}; "
                f'test "$(stat -c %s {quoted_root}/workload-bundle.tar.xz)" = '
                f"{plan.metadata['bundle_size_bytes']}; "
                'test "$(findmnt -rn -M /workspace -o TARGET)" = /workspace; '
                "test ! -L /workspace; "
                "for candidate in "
                "/workspace/equinox-state "
                "/workspace/equinox-state/workload-bundles "
                f"{shlex.quote('/workspace/equinox-state/workload-bundles/' + self.manifest['profile_id'])}; "
                'do test ! -L "$candidate"; done'
            ),
            timeout=90,
        )
        stage_command = (
            "set -euo pipefail; "
            f"export PYTHONPATH={quoted_root}:/workspace/equinox-state/python; "
            f"bash {quoted_root}/scripts/stage-runpod-workload-bundle "
            f"--manifest {quoted_root}/research/studies/larger-model-eligibility.json "
            "stage-mounted "
            f"--bundle {quoted_root}/workload-bundle.tar.xz "
            f"--bundle-digest {shlex.quote(plan.metadata['bundle_digest'])} "
            f"--bundle-size-bytes {plan.metadata['bundle_size_bytes']} "
            "--mount-root /workspace "
            f"--volume-id {shlex.quote(self.volume_id)} "
            f"--data-center-id {shlex.quote(self.data_center_id)} "
            f"--volume-size-gb {self.manifest['hardware']['volume_disk_gb']} "
            f"--receipt-output {quoted_root}/stage-receipt.json "
            f">{quoted_root}/stage-stdout.json"
        )
        self._remote_command(target, options, stage_command, timeout=90)
        probe_command = (
            "set -euo pipefail; "
            f"export PYTHONPATH={quoted_root}:/workspace/equinox-state/python; "
            f"python3 -m {PROBE_MODULE} --source-root {quoted_root} "
            f"--head-commit {shlex.quote(head_commit)} "
            f"--source-archive {quoted_root}/operator-sources.tgz "
            f">{quoted_root}/torch-retention.log; "
            f"sha256sum {quoted_root}/torch-retention.log "
            f">{quoted_root}/torch-retention.sha256"
        )
        self._remote_command(target, options, probe_command, timeout=120)
        model_id = self.manifest["model"]["id"].replace("/", "--")
        snapshot = (
            "/workspace/equinox-state/huggingface/hub/models--"
            f"{model_id}/snapshots/{self.manifest['model']['revision']}"
        )
        receipt_command = (
            "set -euo pipefail; "
            f"export PYTHONPATH=/workspace/equinox-state/python:{quoted_root}; "
            f"python3 {quoted_root}/research/runpod/larger_model_gate.py "
            f"--manifest {quoted_root}/research/studies/larger-model-eligibility.json "
            "create-volume-receipt "
            f"--volume-id {shlex.quote(self.volume_id)} "
            f"--data-center-id {shlex.quote(self.data_center_id)} "
            f"--volume-size-gb {self.manifest['hardware']['volume_disk_gb']} "
            f"--snapshot {shlex.quote(snapshot)} "
            f">{quoted_root}/model-receipt.json"
        )
        self._remote_command(target, options, receipt_command, timeout=240)
        local_files = {
            "model_receipt": temporary_root / "model-receipt.pending.json",
            "stage_receipt": temporary_root / "stage-receipt.pending.json",
            "torch_log": temporary_root / "torch-retention.pending.log",
            "torch_digest": temporary_root / "torch-retention.pending.sha256",
        }
        remote_names = {
            "model_receipt": "model-receipt.json",
            "stage_receipt": "stage-receipt.json",
            "torch_log": "torch-retention.log",
            "torch_digest": "torch-retention.sha256",
        }
        for key_name, destination in local_files.items():
            self._paid_command(
                "scp",
                *scp_options,
                f"{target}:{remote_root}/{remote_names[key_name]}",
                str(destination),
                timeout=60,
                step=f"{key_name} retrieval",
            )
        model_bytes = local_files["model_receipt"].read_bytes()
        stage_bytes = local_files["stage_receipt"].read_bytes()
        torch_bytes = local_files["torch_log"].read_bytes()
        declared_torch = local_files["torch_digest"].read_text(encoding="utf-8").split()[0]
        torch_digest = _tagged_sha256(torch_bytes)
        if declared_torch != torch_digest.removeprefix("sha256:"):
            raise RuntimeError("Torch checkpoint evidence changed during retrieval")
        _verify_torch_evidence(
            torch_bytes,
            self.manifest,
            head_commit=head_commit,
            source_archive_digest=plan.source_archive_digest,
            source_sha256=plan.source_sha256,
        )
        provider_volume = self._paid_command(
            "runpodctl",
            "network-volume",
            "get",
            self.volume_id,
            timeout=20,
            step="fresh volume receipt verification",
        ).stdout
        model_receipt = _json_object(model_bytes, "model-readiness receipt")
        stage_receipt = _json_object(stage_bytes, "bundle-stage receipt")
        verify_volume_readiness_receipt(
            self.manifest,
            model_receipt,
            provider_volume,
            now=self.clock.now(),
        )
        verify_bundle_stage_receipt(
            self.manifest,
            stage_receipt,
            provider_volume,
            expected_bundle_digest=plan.metadata["bundle_digest"],
            expected_bundle_size_bytes=plan.metadata["bundle_size_bytes"],
            expected_bundle_path=plan.metadata["bundle_path"],
            now=self.clock.now(),
        )
        return RemoteArtifacts(
            model_receipt=model_bytes,
            stage_receipt=stage_bytes,
            torch_log=torch_bytes,
            torch_digest=torch_digest,
        )

    def _archive_prior_receipt(self, path: Path) -> None:
        if not path.is_file():
            return
        try:
            old = _json_object(path.read_bytes(), "prior receipt")
        except RuntimeError:
            return
        profile = old.get("profile_id")
        if profile == self.manifest["profile_id"] or not isinstance(profile, str):
            return
        version = profile.rsplit("@", 1)[-1]
        suffix = (
            f"profile-v{version}"
            if version.isdigit()
            else hashlib.sha256(profile.encode()).hexdigest()[:12]
        )
        archive = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
        current = path.read_bytes()
        if archive.is_symlink():
            raise RuntimeError("prior receipt archive path is unsafe")
        if archive.exists():
            if not archive.is_file() or archive.read_bytes() != current:
                raise RuntimeError("prior receipt archive already contains different bytes")
            return
        _atomic_write(archive, current)

    def _replace_artifact(self, source: Path, destination: Path) -> None:
        os.replace(source, destination)

    def _artifact_payloads(
        self,
        artifacts: RemoteArtifacts,
        *,
        attempt_id: str,
    ) -> list[tuple[str, Path, bytes]]:
        torch_path = self.proof_directory / f"{attempt_id}.torch-retention.json"
        return [
            (
                "model_receipt",
                self.proof_directory / f"larger-model-volume-{self.volume_id}.json",
                artifacts.model_receipt,
            ),
            (
                "stage_receipt",
                self.proof_directory / f"larger-model-bundle-stage-{self.volume_id}.json",
                artifacts.stage_receipt,
            ),
            ("torch_evidence", torch_path, artifacts.torch_log),
            (
                "torch_digest",
                self.proof_directory / f"{attempt_id}.torch-retention.sha256",
                (f"{artifacts.torch_digest.removeprefix('sha256:')}  {torch_path.name}\n").encode(),
            ),
        ]

    def _publication_evidence(
        self,
        transaction: Path,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "allocation_attempted": self.create_attempted,
            "allocation_confirmed": bool(self.pod_id),
            "pod_id": self.pod_id or None,
            "pod_name": self.pod_name,
            "started_at": self.started_at,
            "terminate_after": self.terminate_after,
            "baseline_balance_usd": self.baseline_balance_usd,
            "attested_hourly_rate_usd": self.attested_hourly_rate_usd,
            "teardown_confirmed": True,
            "idle_polls": list(self.idle_polls),
            "proof_directory": str(self.proof_directory.resolve()),
            "artifact_transaction": transaction.name,
            "artifact_journal": f"{transaction.name}/transaction.json",
            "artifacts": [dict(descriptor) for descriptor in descriptors],
        }

    def _persist_publication_state(
        self,
        *,
        state: str,
        transaction: Path,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        evidence = self._publication_evidence(transaction, descriptors)
        self.publication_state = state
        self.publication_transaction_evidence = evidence
        self._persist_lease(state=state, evidence=evidence)

    def _remove_artifact_transaction(
        self,
        transaction: Path,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        if not transaction.exists() and not transaction.is_symlink():
            return
        if transaction.is_symlink() or not transaction.is_dir():
            raise RuntimeError("artifact transaction directory is unsafe")
        expected = {"transaction.json"}
        for descriptor in descriptors:
            expected.add(str(descriptor["new_file"]))
            old_file = descriptor["old_file"]
            if old_file is not None:
                expected.add(str(old_file))
        for child in transaction.iterdir():
            if child.name not in expected or child.is_symlink() or not child.is_file():
                raise RuntimeError("artifact transaction contains an unsafe entry")
        for name in sorted(expected):
            candidate = transaction / name
            if candidate.exists() or candidate.is_symlink():
                if candidate.is_symlink() or not candidate.is_file():
                    raise RuntimeError("artifact transaction entry is unsafe")
                candidate.unlink()
        transaction.rmdir()
        _fsync_directory(self.proof_directory)

    def _validate_artifact_descriptors(
        self,
        value: Any,
        *,
        attempt_id: str,
    ) -> list[dict[str, Any]]:
        expected_destinations = {
            "model_receipt": f"larger-model-volume-{self.volume_id}.json",
            "stage_receipt": f"larger-model-bundle-stage-{self.volume_id}.json",
            "torch_evidence": f"{attempt_id}.torch-retention.json",
            "torch_digest": f"{attempt_id}.torch-retention.sha256",
        }
        if not isinstance(value, list) or len(value) != len(expected_destinations):
            raise RuntimeError("artifact transaction descriptor set is invalid")
        descriptors: list[dict[str, Any]] = []
        observed_roles: set[str] = set()
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping) or set(raw) != {
                "role",
                "destination",
                "new_file",
                "new_digest",
                "old_file",
                "old_digest",
            }:
                raise RuntimeError("artifact transaction descriptor is invalid")
            descriptor = dict(raw)
            role = descriptor["role"]
            destination = descriptor["destination"]
            new_file = descriptor["new_file"]
            new_digest = descriptor["new_digest"]
            old_file = descriptor["old_file"]
            old_digest = descriptor["old_digest"]
            if (
                not isinstance(role, str)
                or role not in expected_destinations
                or role in observed_roles
                or destination != expected_destinations[role]
                or new_file != f"new-{index}"
                or not isinstance(new_digest, str)
                or not _TAGGED_SHA256.fullmatch(new_digest)
                or (old_file is None) != (old_digest is None)
                or (
                    old_file is not None
                    and (
                        old_file != f"old-{index}"
                        or not isinstance(old_digest, str)
                        or not _TAGGED_SHA256.fullmatch(old_digest)
                    )
                )
            ):
                raise RuntimeError("artifact transaction descriptor identity is invalid")
            observed_roles.add(role)
            descriptors.append(descriptor)
        if observed_roles != set(expected_destinations):
            raise RuntimeError("artifact transaction roles are incomplete")
        return descriptors

    def _artifact_bytes(
        self,
        descriptor: Mapping[str, Any],
    ) -> bytes | None:
        destination = self.proof_directory / str(descriptor["destination"])
        if destination.is_symlink():
            raise RuntimeError("artifact destination is unsafe")
        if not destination.exists():
            return None
        if not destination.is_file():
            raise RuntimeError("artifact destination is unsafe")
        return destination.read_bytes()

    def _verify_previous_artifact_set(
        self,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        for descriptor in descriptors:
            payload = self._artifact_bytes(descriptor)
            old_digest = descriptor["old_digest"]
            if old_digest is None:
                if payload is not None:
                    raise RuntimeError("artifact rollback did not remove a new destination")
            elif payload is None or _tagged_sha256(payload) != old_digest:
                raise RuntimeError("artifact rollback did not restore the previous bytes")

    def _restore_previous_artifact_set(
        self,
        transaction: Path,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        restore_payloads: list[tuple[Path, bytes | None]] = []
        for descriptor in descriptors:
            current = self._artifact_bytes(descriptor)
            current_digest = _tagged_sha256(current) if current is not None else None
            new_digest = descriptor["new_digest"]
            old_digest = descriptor["old_digest"]
            if current_digest not in {new_digest, old_digest}:
                raise RuntimeError("artifact destination has unrecognized crash-state bytes")
            destination = self.proof_directory / str(descriptor["destination"])
            if old_digest is None:
                restore_payloads.append((destination, None))
                continue
            backup = transaction / str(descriptor["old_file"])
            if backup.is_symlink() or not backup.is_file():
                raise RuntimeError("artifact rollback backup is unavailable or unsafe")
            backup_payload = backup.read_bytes()
            if _tagged_sha256(backup_payload) != old_digest:
                raise RuntimeError("artifact rollback backup digest changed")
            restore_payloads.append((destination, backup_payload))
        for destination, payload in restore_payloads:
            if payload is None:
                destination.unlink(missing_ok=True)
                _fsync_directory(self.proof_directory)
            else:
                _atomic_write(destination, payload)
        self._verify_previous_artifact_set(descriptors)

    def _verify_new_artifact_set(
        self,
        descriptors: Sequence[Mapping[str, Any]],
    ) -> dict[str, bytes]:
        payloads: dict[str, bytes] = {}
        for descriptor in descriptors:
            payload = self._artifact_bytes(descriptor)
            if payload is None or _tagged_sha256(payload) != descriptor["new_digest"]:
                raise RuntimeError("published artifact digest does not match its journal")
            payloads[str(descriptor["role"])] = payload
        return payloads

    def _verify_recovered_receipts(
        self,
        payloads: Mapping[str, bytes],
        lease: Mapping[str, Any],
    ) -> None:
        model = _json_object(payloads["model_receipt"], "recovered model receipt")
        stage = _json_object(payloads["stage_receipt"], "recovered stage receipt")
        torch_evidence = _json_object(
            payloads["torch_evidence"],
            "recovered Torch evidence",
        )
        manifest_digest = _tagged_sha256(canonical_json(self.manifest))
        for receipt, name in ((model, "model"), (stage, "stage")):
            digest = receipt.get("receipt_digest")
            if (
                not isinstance(digest, str)
                or not _TAGGED_SHA256.fullmatch(digest)
                or digest != _receipt_digest(receipt)
            ):
                raise RuntimeError(f"recovered {name} receipt digest is invalid")
            if (
                receipt.get("profile_id") != self.manifest["profile_id"]
                or receipt.get("manifest_digest") != manifest_digest
                or receipt.get("network_volume_id") != self.volume_id
                or receipt.get("network_volume_data_center_id") != self.data_center_id
                or receipt.get("network_volume_size_gb")
                != self.manifest["hardware"]["volume_disk_gb"]
                or receipt.get("ready") is not True
            ):
                raise RuntimeError(f"recovered {name} receipt identity is invalid")
        if (
            model.get("model_id") != self.manifest["model"]["id"]
            or model.get("model_revision") != self.manifest["model"]["revision"]
            or model.get("dependencies") != self.manifest["runtime"]["dependencies"]
            or stage.get("source_contract_digest") != lease.get("source_contract_digest")
            or stage.get("bundle_digest") != lease.get("bundle_digest")
            or stage.get("bundle_size_bytes") != lease.get("bundle_size_bytes")
            or stage.get("bundle_path") != lease.get("bundle_path")
        ):
            raise RuntimeError("recovered receipt workload identity is invalid")
        if (
            torch_evidence.get("status") != "passed"
            or torch_evidence.get("profile_id") != self.manifest["profile_id"]
            or torch_evidence.get("head_commit") != lease.get("head_commit")
            or torch_evidence.get("source_archive_digest") != lease.get("source_archive_digest")
            or torch_evidence.get("torch_version") != self.manifest["runtime"]["torch_version"]
            or torch_evidence.get("torch_cuda_version")
            != _expected_cuda_version(self.manifest["runtime"]["torch_version"])
        ):
            raise RuntimeError("recovered Torch evidence identity is invalid")
        torch_name = next(
            str(descriptor["destination"])
            for descriptor in lease["evidence"]["artifacts"]
            if descriptor["role"] == "torch_evidence"
        )
        expected_hash_file = (
            f"{_tagged_sha256(payloads['torch_evidence']).removeprefix('sha256:')}  {torch_name}\n"
        ).encode()
        if payloads["torch_digest"] != expected_hash_file:
            raise RuntimeError("recovered Torch digest file is invalid")

    def _load_publication_recovery_lease(self) -> dict[str, Any]:
        if (
            self.operator_state_root.is_symlink()
            or not self.operator_state_root.is_dir()
            or self.lease_directory.is_symlink()
            or not self.lease_directory.is_dir()
            or self.lease_path.is_symlink()
            or not self.lease_path.is_file()
        ):
            raise RuntimeError("shared RunPod publication recovery lease is unavailable or unsafe")
        self._acquire_publication_recovery_lock()
        try:
            if self.lease_path.is_symlink() or not self.lease_path.is_file():
                raise RuntimeError("shared RunPod publication recovery lease changed")
            lease = _json_object(
                self.lease_path.read_bytes(),
                "shared RunPod publication recovery lease",
            )
            proof_id = lease.get("proof_id")
            process_id = lease.get("operator_pid")
            expected_image = (
                f"{self.manifest['runtime']['image']}@{self.manifest['runtime']['image_digest']}"
            )
            if (
                lease.get("mode") != "larger-model-volume-stage"
                or not isinstance(proof_id, str)
                or not _SAFE_ID.fullmatch(proof_id)
                or lease.get("pod_name") != proof_id
                or type(process_id) is not int
                or process_id <= 0
                or lease.get("profile_id") != self.manifest["profile_id"]
                or lease.get("network_volume_id") != self.volume_id
                or lease.get("network_volume_data_center_id") != self.data_center_id
                or lease.get("image") != expected_image
            ):
                raise RuntimeError("shared RunPod publication recovery lease identity changed")
            if self._operator_process_is_alive(process_id):
                raise RuntimeError(
                    "recorded RunPod operator process is still alive; recovery refused"
                )
            self.pod_name = proof_id
            self.started_at = str(lease.get("started_at", ""))
            self.terminate_after = str(lease.get("terminate_after", ""))
            for key in (
                "manifest_digest",
                "source_contract_digest",
                "bundle_digest",
                "bundle_size_bytes",
                "bundle_path",
                "source_archive_digest",
                "head_commit",
            ):
                if key in lease:
                    self.bundle_metadata[key] = lease[key]
            self.lease_acquired = True
            self.teardown_confirmed = True
            return lease
        except BaseException:
            self._release_publication_recovery_lock_after_error()
            raise

    def _write_recovery_receipt(self, result: Mapping[str, Any]) -> None:
        path = _recovery_path(self.proof_directory, self.pod_name)
        payload = canonical_json(result) + b"\n"
        if path.is_symlink():
            raise RuntimeError("publication recovery attempt path is unsafe")
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(
                    "publication recovery receipt already contains different evidence"
                )
            return
        _atomic_write(path, payload)

    def recover_artifact_publication(self) -> dict[str, Any]:
        lease = self._load_publication_recovery_lease()
        try:
            return self._recover_loaded_artifact_publication(lease)
        except BaseException:
            self.lease_acquired = False
            self._release_publication_recovery_lock_after_error()
            raise

    def _recover_loaded_artifact_publication(
        self,
        lease: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = lease.get("state")
        if state not in {
            "publishing_artifacts",
            "artifacts_published",
            "artifact_publication_rolled_back",
            "artifact_publication_recovery_complete",
        }:
            self.lease_acquired = False
            raise RuntimeError("shared lease is not an artifact-publication recovery state")
        evidence = lease.get("evidence")
        if not isinstance(evidence, Mapping):
            self.lease_acquired = False
            raise RuntimeError("publication recovery evidence is missing")
        expected_transaction = f".{self.pod_name}.artifact-transaction"
        transaction_name = evidence.get("artifact_transaction")
        if (
            transaction_name != expected_transaction
            or not _SAFE_TRANSACTION_NAME.fullmatch(str(transaction_name))
            or evidence.get("artifact_journal") != f"{expected_transaction}/transaction.json"
            or evidence.get("proof_directory") != str(self.proof_directory.resolve())
            or evidence.get("teardown_confirmed") is not True
        ):
            self.lease_acquired = False
            raise RuntimeError("publication recovery path or teardown evidence is invalid")
        descriptors = self._validate_artifact_descriptors(
            evidence.get("artifacts"),
            attempt_id=self.pod_name,
        )
        transaction = self.proof_directory / expected_transaction
        journal = transaction / "transaction.json"
        if transaction.exists() or transaction.is_symlink():
            if transaction.is_symlink() or not transaction.is_dir():
                raise RuntimeError("artifact transaction directory is unsafe")
            if journal.is_symlink() or not journal.is_file():
                raise RuntimeError("artifact publication journal is unavailable or unsafe")
            journal_payload = _json_object(
                journal.read_bytes(),
                "artifact publication journal",
            )
            if (
                journal_payload.get("schema_version") != 1
                or journal_payload.get("attempt_id") != self.pod_name
                or journal_payload.get("artifacts") != descriptors
            ):
                raise RuntimeError("artifact publication journal identity changed")
        elif state == "publishing_artifacts":
            raise RuntimeError("artifact publication journal is missing")
        self.idle_polls = [
            dict(item) for item in evidence.get("idle_polls", []) if isinstance(item, Mapping)
        ]
        if state in {
            "artifacts_published",
            "artifact_publication_recovery_complete",
        }:
            payloads = self._verify_new_artifact_set(descriptors)
            self._verify_recovered_receipts(payloads, lease)
            terminal_state = "artifact_publication_recovery_complete"
            outcome = "artifact_publication_completed"
        else:
            if not transaction.is_dir():
                self._verify_previous_artifact_set(descriptors)
            else:
                self._restore_previous_artifact_set(transaction, descriptors)
            terminal_state = "artifact_publication_rolled_back"
            outcome = "artifact_publication_rolled_back"
        self._persist_publication_state(
            state=terminal_state,
            transaction=transaction,
            descriptors=descriptors,
        )
        self._remove_artifact_transaction(transaction, descriptors)
        result = {
            "schema_version": 1,
            "outcome": outcome,
            "allocation_attempted": evidence.get("allocation_attempted") is True,
            "allocation_confirmed": evidence.get("allocation_confirmed") is True,
            "pod_id": evidence.get("pod_id"),
            "pod_name": self.pod_name,
            "started_at": self.started_at,
            "terminate_after": self.terminate_after,
            "baseline_balance_usd": evidence.get("baseline_balance_usd"),
            "attested_hourly_rate_usd": evidence.get("attested_hourly_rate_usd"),
            "idle_polls": self.idle_polls,
            "recovery_allocation_attempted": False,
            "provider_queries_issued": False,
            "teardown_confirmed": True,
            "recovery_receipt_path": str(_recovery_path(self.proof_directory, self.pod_name)),
            **self._identity_evidence(),
        }
        self._write_recovery_receipt(result)
        self._release_shared_lease()
        return result

    def _install_artifacts(
        self,
        artifacts: RemoteArtifacts,
        *,
        attempt_id: str,
    ) -> dict[str, str]:
        if not self.lease_acquired or not self.teardown_confirmed:
            raise RuntimeError("artifact publication requires a held lease and confirmed teardown")
        artifact_payloads = self._artifact_payloads(artifacts, attempt_id=attempt_id)
        self.proof_directory.mkdir(parents=True, exist_ok=True)
        if self.proof_directory.is_symlink() or not self.proof_directory.is_dir():
            raise RuntimeError("research proof directory is unsafe")
        transaction = self.proof_directory / f".{attempt_id}.artifact-transaction"
        transaction.mkdir(mode=0o700)
        descriptors: list[dict[str, Any]] = []
        try:
            for index, (role, destination, payload) in enumerate(artifact_payloads):
                if destination.is_symlink():
                    raise RuntimeError("existing artifact path is unsafe")
                if destination.exists():
                    if not destination.is_file():
                        raise RuntimeError("existing artifact path is unsafe")
                    previous = destination.read_bytes()
                    old_file: str | None = f"old-{index}"
                    old_digest: str | None = _tagged_sha256(previous)
                else:
                    previous = None
                    old_file = None
                    old_digest = None
                pending = transaction / f"new-{index}"
                descriptors.append(
                    {
                        "role": role,
                        "destination": destination.name,
                        "new_file": pending.name,
                        "new_digest": _tagged_sha256(payload),
                        "old_file": old_file,
                        "old_digest": old_digest,
                    }
                )
                _atomic_write(pending, payload)
                if pending.read_bytes() != payload:
                    raise RuntimeError("staged artifact bytes changed before publication")
                if previous is not None:
                    _atomic_write(transaction / str(old_file), previous)
            transaction_manifest = {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "state": "prepared",
                "published": [],
                "artifacts": descriptors,
            }
            _atomic_write(
                transaction / "transaction.json",
                canonical_json(transaction_manifest) + b"\n",
            )
        except BaseException:
            self._remove_artifact_transaction(transaction, descriptors)
            raise
        self.publication_recovery_required = True
        self._persist_publication_state(
            state="publishing_artifacts",
            transaction=transaction,
            descriptors=descriptors,
        )
        try:
            self._archive_prior_receipt(artifact_payloads[0][1])
            self._archive_prior_receipt(artifact_payloads[1][1])
            published: list[str] = []
            for descriptor in descriptors:
                transaction_manifest["state"] = "publishing"
                transaction_manifest["published"] = list(published)
                _atomic_write(
                    transaction / "transaction.json",
                    canonical_json(transaction_manifest) + b"\n",
                )
                destination = self.proof_directory / descriptor["destination"]
                self._replace_artifact(
                    transaction / descriptor["new_file"],
                    destination,
                )
                _fsync_directory(self.proof_directory)
                published.append(destination.name)
                transaction_manifest["published"] = list(published)
                _atomic_write(
                    transaction / "transaction.json",
                    canonical_json(transaction_manifest) + b"\n",
                )
            transaction_manifest["state"] = "published"
            _atomic_write(
                transaction / "transaction.json",
                canonical_json(transaction_manifest) + b"\n",
            )
            self._verify_new_artifact_set(descriptors)
            self._persist_publication_state(
                state="artifacts_published",
                transaction=transaction,
                descriptors=descriptors,
            )
        except BaseException as publication_error:
            try:
                self._restore_previous_artifact_set(transaction, descriptors)
                self._persist_publication_state(
                    state="artifact_publication_rolled_back",
                    transaction=transaction,
                    descriptors=descriptors,
                )
                self._remove_artifact_transaction(transaction, descriptors)
            except BaseException:
                raise RuntimeError(
                    "artifact-set publication failed and rollback is incomplete"
                ) from publication_error
            self.publication_recovery_required = False
            raise
        self._remove_artifact_transaction(transaction, descriptors)
        self.publication_recovery_required = False
        paths = {role: path for role, path, _ in artifact_payloads}
        return {
            "model_receipt_path": str(paths["model_receipt"]),
            "stage_receipt_path": str(paths["stage_receipt"]),
            "torch_evidence_path": str(paths["torch_evidence"]),
            "torch_evidence_digest": artifacts.torch_digest,
        }

    def execute(self) -> dict[str, Any]:
        self._validate_cli_contract()
        plan = self._bundle_plan()
        self.bundle_metadata = dict(plan.metadata)
        self._provider_state(require_idle=True)
        self.pod_name = self._unique_name()
        attempt_id = self.pod_name
        start = self.clock.now()
        self.provider_deadline_monotonic = self.clock.monotonic() + MAXIMUM_LIFETIME_SECONDS
        started_at = _utc_text(start)
        terminate_after = _utc_text(start + timedelta(seconds=MAXIMUM_LIFETIME_SECONDS))
        self.started_at = started_at
        self.terminate_after = terminate_after
        create_arguments = self._create_arguments(self.pod_name, terminate_after)
        if "--gpu-id" in create_arguments or create_arguments.count("pod") != 1:
            raise RuntimeError("operator create contract is unsafe")
        self._acquire_shared_lease()
        self._persist_lease(state="requesting_capacity")
        baseline = self._provider_state(require_idle=True)
        self.baseline_balance_usd = str(baseline.balance)
        self.create_attempted = True
        try:
            create = self._command(*create_arguments, timeout=90, checked=False)
        except (Exception, KeyboardInterrupt) as error:
            return_code = (
                130
                if isinstance(error, KeyboardInterrupt)
                else 124
                if isinstance(error, subprocess.TimeoutExpired)
                else 70
            )
            create = subprocess.CompletedProcess(
                create_arguments,
                return_code,
                stdout=b"",
                stderr=(
                    f"{type(error).__name__}: {error}".encode(
                        "utf-8",
                        errors="replace",
                    )
                ),
            )
        try:
            create_payload = _json_object(create.stdout, "RunPod create response")
        except RuntimeError:
            create_payload = {}
        pod_id = create_payload.get("id")
        if not isinstance(pod_id, str):
            nested = create_payload.get("pod")
            pod_id = nested.get("id") if isinstance(nested, Mapping) else None
        valid_id = isinstance(pod_id, str) and bool(_SAFE_POD_ID.fullmatch(pod_id))
        if create.returncode != 0 and valid_id:
            self.pod_id = pod_id
            self._reconcile_failed_create(
                create=create,
                known_capacity_rejection=False,
                started_at=started_at,
                terminate_after=terminate_after,
                initial_pod_ids=(self.pod_id,),
            )
            raise AssertionError("failed create reconciliation must terminate")
        if not valid_id:
            error_text = (create.stderr + b"\n" + create.stdout).decode("utf-8", errors="replace")
            self._reconcile_failed_create(
                create=create,
                known_capacity_rejection=(
                    create.returncode != 0
                    and not valid_id
                    and CAPACITY_REJECTION_TEXT in error_text
                ),
                started_at=started_at,
                terminate_after=terminate_after,
            )
            raise AssertionError("failed create reconciliation must terminate")
        try:
            self.pod_id = str(pod_id)
            self._persist_lease(
                state="allocation_returned",
                evidence={"pod_id": self.pod_id},
            )
            expected_image = (
                f"{self.manifest['runtime']['image']}@{self.manifest['runtime']['image_digest']}"
            )
            hourly_rate = self._attest_created_pod(
                expected_image=expected_image,
                terminate_after=terminate_after,
            )
            self.attested_hourly_rate_usd = str(hourly_rate)
            with tempfile.TemporaryDirectory(prefix="equinox-volume-stage-") as directory:
                artifacts = self._perform_remote_stage(
                    plan,
                    Path(directory),
                    head_commit=plan.head_commit,
                )
        except BaseException:
            self.idle_polls = self._delete_and_confirm(
                (self.pod_id,),
                pod_name=self.pod_name,
            )
            raise
        self.idle_polls = self._delete_and_confirm(
            (self.pod_id,),
            pod_name=self.pod_name,
        )
        self._persist_lease(
            state="teardown_confirmed",
            evidence={"pod_id": self.pod_id, "idle_polls": self.idle_polls},
        )
        paths = self._install_artifacts(artifacts, attempt_id=attempt_id)
        final_balance = Decimal(self.idle_polls[-1]["balance_usd"])
        return {
            "schema_version": 1,
            "outcome": "staged",
            "allocation_attempted": True,
            "allocation_confirmed": True,
            "pod_id": self.pod_id,
            "pod_name": self.pod_name,
            "started_at": started_at,
            "terminate_after": terminate_after,
            "hourly_rate_usd": str(hourly_rate),
            "baseline_balance_usd": str(baseline.balance),
            "final_balance_usd": str(final_balance),
            "balance_delta_usd": str(baseline.balance - final_balance),
            **self._identity_evidence(),
            "model_readiness_receipt_digest": _json_object(
                artifacts.model_receipt, "model receipt"
            )["receipt_digest"],
            "bundle_stage_receipt_digest": _json_object(artifacts.stage_receipt, "stage receipt")[
                "receipt_digest"
            ],
            **paths,
            "teardown_confirmed": True,
            "idle_polls": self.idle_polls,
            "gpu_fallback_used": False,
        }


def _attempt_path(proof_directory: Path, name: str) -> Path:
    safe = name if _SAFE_ID.fullmatch(name) else f"unknown-{secrets.token_hex(4)}"
    return proof_directory / f"{safe}.volume-stage-attempt.json"


def _recovery_path(proof_directory: Path, name: str) -> Path:
    safe = name if _SAFE_ID.fullmatch(name) else f"unknown-{secrets.token_hex(4)}"
    return proof_directory / f"{safe}.volume-stage-recovery.json"


def _write_attempt(operator: VolumeStageOperator, evidence: Mapping[str, Any]) -> None:
    if not operator.create_attempted:
        return
    name = operator.pod_name or "unknown"
    _atomic_write(
        _attempt_path(operator.proof_directory, name),
        canonical_json(evidence) + b"\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--recover-publication", action="store_true")
    arguments = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    operator = VolumeStageOperator(repository_root)
    if arguments.recover_publication:
        print(
            json.dumps(
                operator.recover_artifact_publication(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if arguments.preflight_only:
        print(
            json.dumps(
                operator.preflight(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    try:
        result = operator.execute()
    except OperatorStop as stop:
        evidence = {
            **stop.evidence,
            **operator._identity_evidence(),
        }
        _write_attempt(operator, evidence)
        if stop.exit_code == EXIT_CAPACITY_UNAVAILABLE:
            operator._persist_lease(
                state="capacity_unavailable_confirmed_zero_allocation",
                evidence=evidence,
            )
            operator._release_shared_lease()
        else:
            operator._persist_lease(
                state="ambiguous_create_unresolved",
                evidence=evidence,
            )
        print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
        return stop.exit_code
    except BaseException as error:
        evidence = {
            "schema_version": 1,
            "outcome": "failed",
            "allocation_attempted": operator.create_attempted,
            "pod_id": operator.pod_id or None,
            "pod_name": operator.pod_name or None,
            "teardown_confirmed": operator.teardown_confirmed,
            "idle_polls": operator.idle_polls,
            "baseline_balance_usd": operator.baseline_balance_usd,
            "attested_hourly_rate_usd": operator.attested_hourly_rate_usd,
            "publication_recovery_required": operator.publication_recovery_required,
            "error": f"{type(error).__name__}: {error}",
            **operator._identity_evidence(),
        }
        _write_attempt(operator, evidence)
        if operator.lease_acquired:
            if not operator.create_attempted or (
                operator.teardown_confirmed and not operator.publication_recovery_required
            ):
                operator._persist_lease(state="failed_safe", evidence=evidence)
                operator._release_shared_lease()
            elif operator.publication_recovery_required:
                operator._persist_lease(
                    state=operator.publication_state or "publication_recovery_required",
                    evidence={
                        **evidence,
                        **operator.publication_transaction_evidence,
                    },
                )
            else:
                operator._persist_lease(state="recovery_required", evidence=evidence)
        raise
    operator._persist_lease(state="completed", evidence=result)
    _write_attempt(operator, result)
    operator._release_shared_lease()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
