from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import hmac
import io
import json
import lzma
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from threading import Lock, Timer
from typing import Any

MAXIMUM_BUNDLE_BYTES = 2 * 1024 * 1024
MAXIMUM_EXPANDED_BUNDLE_BYTES = 16 * 1024 * 1024
MAXIMUM_RECEIPT_BYTES = 256 * 1024
MAXIMUM_ACTIVATION_BYTES = 8 * 1024
MAXIMUM_DEPENDENCY_FILES = 200_000
MAXIMUM_DEPENDENCY_BYTES = 12 * 1024 * 1024 * 1024
MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
BUNDLE_READ_TIMEOUT_SECONDS = 30.0
PREPARATION_DEADLINE_SECONDS = 570.0
MAXIMUM_READINESS_WINDOW_SECONDS = 600
BUNDLE_CONTENT_TYPE = "application/x-xz"
BUNDLE_HANDOFF_REVISION = "runpod-volume-bundle-handoff@1"
BOOTSTRAP_TRANSPORT_REVISION = "authenticated-proxy-bundle-bootstrap@1"
BUNDLE_ACTIVATION_REVISION = "authenticated-proxy-stage-activation@2"
BUNDLE_VOLUME_ROOT = Path("/workspace/equinox-state/workload-bundles")
LIVE_WORK_DIRECTORY_ROOT = Path("/workspace/equinox-runs")
BUNDLE_STAGE_RECEIPT_FILENAME = "bundle-stage-receipt.json"
VOLUME_READINESS_RECEIPT_FILENAME = "volume-readiness-receipt.json"
TORCH_RETENTION_EVIDENCE_FILENAME = "torch-retention-evidence.json"
DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME = "dependency-quarantine-evidence.json"
CODE_MATERIALIZATION_EVIDENCE_FILENAME = "code-materialization-evidence.json"
LIVE_STAGE_PLAN_FILENAME = "live-stage-plan.json"
LIVE_STAGE_STATE_FILENAME = "live-stage-state.json"
LIVE_STAGE_ACTIVATION_FILENAME = "live-stage-activation.json"
DEPENDENCY_QUARANTINE_REVISION = "hash-locked-private-dependencies@1"
CODE_MATERIALIZATION_REVISION = "private-code-materialization@1"
DEPENDENCY_LOCK_FILENAME = "larger-model-dependencies.lock"
DEPENDENCY_LOCK_DIGEST = "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
DEPENDENCY_LOCK_SIZE_BYTES = 30_866
DEPENDENCY_INSTALLER_REVISION = "isolated-pip-binary-hash-lock@1"
DEPENDENCY_INDEX_URL = "https://pypi.org/simple"
DEPENDENCY_PYTHON_VERSION = "3.12"
DEPENDENCY_PLATFORM_TAG = "manylinux_2_28_x86_64"
PRIVATE_QUARANTINE_ROOT = Path("/tmp/equinox-quarantine")
PRIVATE_DEPENDENCY_ROOT = PRIVATE_QUARANTINE_ROOT / "dependencies"
PRIVATE_CODE_ROOT = PRIVATE_QUARANTINE_ROOT / "code"
PRIVATE_RUNTIME_ROOT = PRIVATE_QUARANTINE_ROOT / "runtime"
PRIVATE_RUNTIME_REVISION = "bootstrap-private-runtime@1"
PRIVATE_ANCHOR_ROOT = PRIVATE_QUARANTINE_ROOT / "anchors"
PRIVATE_ANCHOR_REVISION = "bootstrap-private-anchor@1"
REQUIRED_DEPENDENCY_VERSIONS = {
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "transformers": "5.14.1",
}
COMMON_BUNDLE_FILES = frozenset({"remote_runner.sh", "result_server.py"})
LARGER_MODEL_BUNDLE_SUPPORT_FILES = frozenset(
    {
        "larger-model-eligibility.json",
        DEPENDENCY_LOCK_FILENAME,
        "larger_model_gate.py",
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_env_v32.py",
        "repository_repair_env_v33.py",
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_pilot.py",
        "repository_repair_large_model_study.py",
        "repository_repair_large_model_trainer.py",
        "retention_checkpoint_probe.py",
    }
)
WORKLOAD_SUPPORT_FILES = {
    "branching_sequence_ladder.py": frozenset(),
    "repository_repair_rl.py": frozenset({"repository_repair_env.py"}),
    "repository_repair_study.py": frozenset(
        {"repository_repair_env.py", "repository_repair_rl.py"}
    ),
    "repository_repair_study_v31.py": frozenset(
        {
            "repository_repair_env.py",
            "repository_repair_env_v31.py",
            "repository_repair_rl.py",
            "repository_repair_study.py",
        }
    ),
    "repository_repair_eligibility.py": frozenset(
        {
            "repository_repair_env.py",
            "repository_repair_rl.py",
            "repository_repair_study.py",
        }
    ),
    "repository_repair_large_model_eligibility.py": LARGER_MODEL_BUNDLE_SUPPORT_FILES,
    "repository_repair_large_model_pilot.py": LARGER_MODEL_BUNDLE_SUPPORT_FILES,
    "research/runpod/revision30_external_eval.py": frozenset(
        {
            "external_eval_remote_runner.sh",
            "research/__init__.py",
            "research/external/revision30_task_pack.py",
            "research/frozen/revision30-external-pack.json",
            "research/runpod/__init__.py",
            "research/runpod/external_eval_transport.py",
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_rl.py",
        }
    ),
    "research/runpod/revision31_external_eval.py": frozenset(
        {
            "external_eval_remote_runner.sh",
            "research/__init__.py",
            "research/external/revision30_task_pack.py",
            "research/external/revision31_task_pack.py",
            "research/frozen/revision31-external-pack.json",
            "research/runpod/__init__.py",
            "research/runpod/external_eval_transport.py",
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_env_v31.py",
            "research/runpod/repository_repair_rl.py",
            "research/runpod/revision30_external_eval.py",
        }
    ),
}
EXTERNAL_EVALUATION_WORKLOAD = "research/runpod/revision30_external_eval.py"
EXTERNAL_EVALUATION_WORKLOADS = frozenset(
    {
        EXTERNAL_EVALUATION_WORKLOAD,
        "research/runpod/revision31_external_eval.py",
    }
)


def expected_bundle_files(workload_file: str) -> frozenset[str]:
    try:
        support_files = WORKLOAD_SUPPORT_FILES[workload_file]
    except KeyError as error:
        raise ValueError("Unsupported workload file.") from error
    common_files = (
        frozenset({"external_eval_remote_runner.sh"})
        if workload_file in EXTERNAL_EVALUATION_WORKLOADS
        else COMMON_BUNDLE_FILES
    )
    return common_files | support_files | {workload_file}


def runner_file(workload_file: str) -> str:
    expected_bundle_files(workload_file)
    return (
        "external_eval_remote_runner.sh"
        if workload_file in EXTERNAL_EVALUATION_WORKLOADS
        else "remote_runner.sh"
    )


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Value could not be represented as canonical JSON.") from error


def tagged_sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    material = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    return tagged_sha256(canonical_json(material))


def _evidence_digest(evidence: Mapping[str, Any]) -> str:
    material = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    return tagged_sha256(canonical_json(material))


def _launch_hmac(material: Mapping[str, Any]) -> str:
    token = os.environ.get("EQUINOX_RESULT_TOKEN", "")
    if not token:
        raise ValueError("Launch token was unavailable for durable state binding.")
    return (
        "hmac-sha256:"
        + hmac.new(
            token.encode(),
            canonical_json(material),
            hashlib.sha256,
        ).hexdigest()
    )


def _expanded_xz_payload(payload: bytes) -> bytes:
    """Decompress one XZ stream without allowing trailing or oversized output."""

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        expanded = decompressor.decompress(
            payload,
            max_length=MAXIMUM_EXPANDED_BUNDLE_BYTES + 1,
        )
    except lzma.LZMAError as error:
        raise ValueError("Bundle was not a valid XZ stream.") from error
    if len(expanded) > MAXIMUM_EXPANDED_BUNDLE_BYTES:
        raise ValueError("Bundle expanded beyond the configured limit.")
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Bundle XZ stream was truncated or had trailing data.")
    return expanded


def _validate_archive_members(
    members: list[tarfile.TarInfo],
    *,
    expected_files: frozenset[str],
) -> None:
    names = [member.name for member in members]
    if len(names) != len(set(names)) or set(names) != expected_files:
        raise ValueError("Bundle file set did not match the workload allowlist.")
    expanded_size = 0
    for member in members:
        relative_path = Path(member.name)
        if (
            not member.isfile()
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or any(part in {"", "."} for part in relative_path.parts)
            or member.name not in expected_files
            or member.size < 0
        ):
            raise ValueError("Bundle members must be allowlisted regular files.")
        expanded_size += member.size
        if expanded_size > MAXIMUM_EXPANDED_BUNDLE_BYTES:
            raise ValueError("Bundle expanded beyond the configured limit.")


def install_bundle(
    payload: bytes,
    *,
    work_directory: Path,
    workload_file: str,
) -> None:
    expected_files = expected_bundle_files(workload_file)
    staging_directory = Path(tempfile.mkdtemp(prefix=".bundle-", dir=work_directory))
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = archive.getmembers()
            _validate_archive_members(members, expected_files=expected_files)
            for member in members:
                relative_path = Path(member.name)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Bundle member could not be read.")
                destination = staging_directory / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Bundle member was truncated.")
                        handle.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError("Bundle member exceeded its declared size.")
                    handle.flush()
                    os.fsync(handle.fileno())
        for filename in sorted(expected_files):
            destination = work_directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_directory / filename, destination)
        directory_descriptor = os.open(work_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)


def install_environment_bundle(
    encoded_payload: str,
    *,
    work_directory: Path,
    workload_file: str,
    expected_digest: str | None = None,
) -> None:
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Environment bundle was not valid base64.") from error
    if not payload or len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Environment bundle size was invalid.")
    if expected_digest is not None:
        observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if not hmac.compare_digest(observed_digest, expected_digest):
            raise ValueError("Environment bundle digest did not match.")
    install_bundle(
        payload,
        work_directory=work_directory,
        workload_file=workload_file,
    )


def install_volume_bundle(
    volume_path: Path,
    *,
    work_directory: Path,
    workload_file: str,
    expected_digest: str,
    expected_size_bytes: int,
    work_directory_descriptor: int | None = None,
) -> None:
    if (
        not volume_path.is_absolute()
        or volume_path.parent.parent != BUNDLE_VOLUME_ROOT
        or not re.fullmatch(r"[A-Za-z0-9._@-]+", volume_path.parent.name)
        or volume_path.name != f"{expected_digest.removeprefix('sha256:')}.tar.xz"
    ):
        raise ValueError("Volume bundle path was not content-addressed.")
    if (
        not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest)
        or type(expected_size_bytes) is not int
        or not 0 < expected_size_bytes <= MAXIMUM_BUNDLE_BYTES
    ):
        raise ValueError("Volume bundle identity was invalid.")
    directory_descriptor = _open_safe_directory(volume_path.parent)
    try:
        payload = _read_regular_at(
            directory_descriptor,
            volume_path.name,
            maximum_bytes=expected_size_bytes,
        )
    finally:
        os.close(directory_descriptor)
    if len(payload) != expected_size_bytes:
        raise ValueError("Volume bundle size did not match.")
    observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise ValueError("Volume bundle digest did not match.")
    if work_directory_descriptor is None:
        install_bundle(
            payload,
            work_directory=work_directory,
            workload_file=workload_file,
        )
    else:
        if not _directory_identity_matches(work_directory_descriptor, work_directory):
            raise ValueError("Larger-model work directory identity changed.")
        _install_flat_live_bundle_at(
            _expanded_xz_payload(payload),
            directory_descriptor=work_directory_descriptor,
            workload_file=workload_file,
        )


@dataclass(frozen=True)
class LiveStagePlan:
    profile_id: str
    manifest_digest: str
    source_contract_digest: str
    bootstrap_source_digest: str
    head_commit: str
    network_volume_id: str
    network_volume_data_center_id: str
    network_volume_size_gb: int
    workload_bundle_digest: str
    workload_bundle_size_bytes: int
    workload_bundle_path: str
    readiness_deadline_epoch: int

    @classmethod
    def from_environment(cls) -> LiveStagePlan:
        def required(name: str) -> str:
            value = os.environ.get(name, "")
            if not value:
                raise ValueError(f"{name} is required for live bundle staging.")
            return value

        try:
            volume_size = int(required("EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB"))
            bundle_size = int(required("EQUINOX_BUNDLE_SIZE_BYTES"))
            readiness_deadline_epoch = int(required("EQUINOX_LIVE_STAGE_READINESS_DEADLINE_EPOCH"))
        except ValueError as error:
            raise ValueError("Live bundle numeric identity is invalid.") from error
        source_contract_digest = os.environ.get(
            "EQUINOX_SOURCE_CONTRACT_SHA256",
            os.environ.get("EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256", ""),
        )
        if not source_contract_digest:
            raise ValueError("EQUINOX_SOURCE_CONTRACT_SHA256 is required for live bundle staging.")
        plan = cls(
            profile_id=required("EQUINOX_LARGER_MODEL_PROFILE_ID"),
            manifest_digest=required("EQUINOX_LARGER_MODEL_MANIFEST_SHA256"),
            source_contract_digest=source_contract_digest,
            bootstrap_source_digest=required("EQUINOX_BOOTSTRAP_SOURCE_SHA256"),
            head_commit=required("EQUINOX_SOURCE_HEAD_COMMIT"),
            network_volume_id=required("EQUINOX_RUNPOD_NETWORK_VOLUME_ID"),
            network_volume_data_center_id=required("EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID"),
            network_volume_size_gb=volume_size,
            workload_bundle_digest=required("EQUINOX_BUNDLE_SHA256"),
            workload_bundle_size_bytes=bundle_size,
            workload_bundle_path=required("EQUINOX_BUNDLE_VOLUME_PATH"),
            readiness_deadline_epoch=readiness_deadline_epoch,
        )
        plan.validate(require_open_deadline=False)
        return plan

    def validate(self, *, require_open_deadline: bool = True) -> None:
        digests = (
            self.manifest_digest,
            self.source_contract_digest,
            self.bootstrap_source_digest,
            self.workload_bundle_digest,
        )
        if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in digests):
            raise ValueError("Live bundle digest identity is invalid.")
        if not re.fullmatch(r"[A-Za-z0-9._@-]+", self.profile_id):
            raise ValueError("Live bundle profile identity is invalid.")
        if not re.fullmatch(r"[0-9a-f]{40}", self.head_commit):
            raise ValueError("Live bundle HEAD identity is invalid.")
        if (
            not re.fullmatch(r"[A-Za-z0-9._-]+", self.network_volume_id)
            or not re.fullmatch(
                r"[A-Za-z0-9._-]+",
                self.network_volume_data_center_id,
            )
            or type(self.network_volume_size_gb) is not int
            or self.network_volume_size_gb < 1
            or type(self.workload_bundle_size_bytes) is not int
            or not 0 < self.workload_bundle_size_bytes <= MAXIMUM_BUNDLE_BYTES
            or type(self.readiness_deadline_epoch) is not int
        ):
            raise ValueError("Live bundle volume or size identity is invalid.")
        current_epoch = int(time.time())
        if self.readiness_deadline_epoch > current_epoch + MAXIMUM_READINESS_WINDOW_SECONDS:
            raise ValueError("Live bundle readiness deadline is outside the safe window.")
        if require_open_deadline and current_epoch >= self.readiness_deadline_epoch:
            raise ValueError("Live bundle readiness deadline has expired.")
        expected_path = (
            BUNDLE_VOLUME_ROOT
            / self.profile_id
            / f"{self.workload_bundle_digest.removeprefix('sha256:')}.tar.xz"
        )
        if Path(self.workload_bundle_path) != expected_path:
            raise ValueError("Live bundle path is not the planned content address.")

    def public_identity(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "head_commit": self.head_commit,
            "manifest_digest": self.manifest_digest,
            "source_contract_digest": self.source_contract_digest,
            "bootstrap_source_digest": self.bootstrap_source_digest,
            "network_volume_id": self.network_volume_id,
            "network_volume_data_center_id": self.network_volume_data_center_id,
            "network_volume_size_gb": self.network_volume_size_gb,
            "workload_bundle_digest": self.workload_bundle_digest,
            "workload_bundle_size_bytes": self.workload_bundle_size_bytes,
            "workload_bundle_path": self.workload_bundle_path,
            "readiness_deadline_epoch": self.readiness_deadline_epoch,
        }


def _open_safe_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("Staging directory must be absolute.")
    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("Staging directory was not canonical.")
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_existing_safe_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("Quarantine source directory must be absolute.")
    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("Quarantine source directory was not canonical.")
            next_descriptor = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise ValueError("Quarantine source component was not a directory.")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_relative_parts(path: str) -> tuple[str, ...]:
    if not path or "\\" in path or "\x00" in path:
        raise ValueError("Quarantine tree path was unsafe.")
    pure_path = PurePosixPath(path)
    parts = pure_path.parts
    if (
        pure_path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or pure_path.as_posix() != path
    ):
        raise ValueError("Quarantine tree path was unsafe.")
    return parts


def _open_relative_directory(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    create: bool = False,
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
            next_descriptor = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise ValueError("Quarantine tree component was not a directory.")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _deadline_check(deadline_epoch: float | None) -> None:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("Quarantine materialization deadline expired.")


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _digest_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    deadline_epoch: float | None,
) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError("Quarantine tree entry was not a bounded unlinked regular file.")
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            _deadline_check(deadline_epoch)
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed_size))
            if not chunk:
                break
            digest.update(chunk)
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise ValueError("Quarantine tree exceeded its byte limit.")
        after = os.fstat(descriptor)
        if (
            observed_size != before.st_size
            or _regular_file_identity(before) != _regular_file_identity(after)
        ):
            raise ValueError("Quarantine tree entry changed while hashing.")
        return {
            "size_bytes": observed_size,
            "sha256": f"sha256:{digest.hexdigest()}",
        }
    finally:
        os.close(descriptor)


def _scan_regular_tree(
    root_descriptor: int,
    *,
    maximum_files: int,
    maximum_bytes: int,
    deadline_epoch: float | None = None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    total_bytes = 0

    def walk(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        nonlocal total_bytes
        _deadline_check(deadline_epoch)
        with os.scandir(directory_descriptor) as iterator:
            directory_entries = sorted(iterator, key=lambda entry: entry.name)
        for directory_entry in directory_entries:
            name = directory_entry.name
            if name in {"", ".", ".."} or "/" in name or "\\" in name or "\x00" in name:
                raise ValueError("Quarantine tree contained an unsafe entry name.")
            metadata = directory_entry.stat(follow_symlinks=False)
            relative_parts = (*prefix, name)
            relative_path = PurePosixPath(*relative_parts).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("Quarantine tree contained a symbolic link.")
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    child_metadata = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISDIR(child_metadata.st_mode)
                        or child_metadata.st_dev != metadata.st_dev
                        or child_metadata.st_ino != metadata.st_ino
                    ):
                        raise ValueError("Quarantine tree directory identity changed.")
                    walk(child_descriptor, relative_parts)
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Quarantine tree contained a non-regular entry.")
            remaining_bytes = maximum_bytes - total_bytes
            observed = _digest_regular_file_at(
                directory_descriptor,
                name,
                maximum_bytes=remaining_bytes,
                deadline_epoch=deadline_epoch,
            )
            entry_metadata = directory_entry.stat(follow_symlinks=False)
            if (
                entry_metadata.st_dev != metadata.st_dev
                or entry_metadata.st_ino != metadata.st_ino
                or entry_metadata.st_size != metadata.st_size
            ):
                raise ValueError("Quarantine tree entry identity changed.")
            total_bytes += int(observed["size_bytes"])
            entries.append({"path": relative_path, **observed})
            if len(entries) > maximum_files or total_bytes > maximum_bytes:
                raise ValueError("Quarantine tree exceeded its configured limits.")

    walk(root_descriptor, ())
    if not entries:
        raise ValueError("Quarantine tree was empty.")
    return entries


def _verify_hardened_tree(
    root_descriptor: int,
    *,
    executable_files: frozenset[str] = frozenset(),
) -> None:
    root_metadata = os.fstat(root_descriptor)
    if stat.S_IMODE(root_metadata.st_mode) != 0o555:
        raise ValueError("Private quarantine root permissions were not immutable.")

    def walk(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        with os.scandir(directory_descriptor) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            relative_path = PurePosixPath(*prefix, entry.name).as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                if stat.S_IMODE(metadata.st_mode) != 0o555:
                    raise ValueError("Private quarantine directory was writable.")
                child = os.open(
                    entry.name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    walk(child, (*prefix, entry.name))
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                expected_mode = 0o555 if relative_path in executable_files else 0o444
                if stat.S_IMODE(metadata.st_mode) != expected_mode:
                    raise ValueError("Private quarantine file permissions were mutable.")
            else:
                raise ValueError("Private quarantine contained an unsafe entry.")

    walk(root_descriptor, ())


def _scan_allowlisted_files(
    root_descriptor: int,
    expected_files: frozenset[str],
    *,
    maximum_bytes: int,
    deadline_epoch: float | None = None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for relative_path in sorted(expected_files):
        parts = _safe_relative_parts(relative_path)
        parent_descriptor = _open_relative_directory(root_descriptor, parts[:-1])
        try:
            observed = _digest_regular_file_at(
                parent_descriptor,
                parts[-1],
                maximum_bytes=maximum_bytes - total_bytes,
                deadline_epoch=deadline_epoch,
            )
        finally:
            os.close(parent_descriptor)
        total_bytes += int(observed["size_bytes"])
        if total_bytes > maximum_bytes:
            raise ValueError("Code materialization exceeded its byte limit.")
        entries.append({"path": relative_path, **observed})
    return entries


def _tree_digest(entries: list[dict[str, object]]) -> str:
    return tagged_sha256(canonical_json(sorted(entries, key=lambda entry: str(entry["path"]))))


def _read_relative_regular(
    root_descriptor: int,
    relative_path: str,
    *,
    expected_entry: Mapping[str, object],
    maximum_bytes: int,
    deadline_epoch: float | None = None,
) -> bytes:
    parts = _safe_relative_parts(relative_path)
    parent_descriptor = _open_relative_directory(root_descriptor, parts[:-1])
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(parts[-1], flags, dir_fd=parent_descriptor)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_entry["size_bytes"]
                or before.st_size > maximum_bytes
            ):
                raise ValueError("Quarantine file metadata did not match its scan.")
            payload = bytearray()
            digest = hashlib.sha256()
            while True:
                _deadline_check(deadline_epoch)
                chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
                if len(payload) > maximum_bytes:
                    raise ValueError("Quarantine file exceeded its byte limit.")
            after = os.fstat(descriptor)
            observed_digest = f"sha256:{digest.hexdigest()}"
            if (
                _regular_file_identity(before) != _regular_file_identity(after)
                or len(payload) != before.st_size
                or not hmac.compare_digest(observed_digest, str(expected_entry["sha256"]))
            ):
                raise ValueError("Quarantine file changed after its scan.")
            return bytes(payload)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_derived_bytecode_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return path.endswith(".pyc") and "__pycache__" in parts


def _is_installer_data_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and parts[0] in {"bin", "share"}


def _is_external_record_data_path(path: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:\.\./)+(?:bin|share)/(?:[A-Za-z0-9._+@-]+/)*[A-Za-z0-9._+@-]+",
            path,
        )
    )


def _dependency_record_closure(
    root_descriptor: int,
    entries: list[dict[str, object]],
    *,
    expected_versions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, object]], str]:
    entries_by_path = {str(entry["path"]): entry for entry in entries}
    record_paths = sorted(
        path for path in entries_by_path if path.endswith(".dist-info/RECORD")
    )
    if not record_paths:
        raise ValueError("Dependency quarantine found no installed distributions.")
    claimed_paths: set[str] = set()
    distributions: list[dict[str, object]] = []
    observed_distributions: dict[str, str] = {}
    for record_path in record_paths:
        record_entry = entries_by_path[record_path]
        record_payload = _read_relative_regular(
            root_descriptor,
            record_path,
            expected_entry=record_entry,
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        try:
            record_text = record_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Dependency RECORD was not UTF-8.") from error
        distribution_directory = record_path.rsplit("/", 1)[0]
        metadata_path = f"{distribution_directory}/METADATA"
        metadata_entry = entries_by_path.get(metadata_path)
        if metadata_entry is None:
            raise ValueError("Dependency distribution had no METADATA.")
        metadata_payload = _read_relative_regular(
            root_descriptor,
            metadata_path,
            expected_entry=metadata_entry,
            maximum_bytes=MAXIMUM_RECORD_BYTES,
        )
        metadata = BytesParser().parsebytes(metadata_payload)
        name = metadata.get("Name", "")
        version = metadata.get("Version", "")
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not name
            or not version
            or any(character in name + version for character in "\r\n\x00")
        ):
            raise ValueError("Dependency METADATA identity was invalid.")
        normalized_name = _normalize_distribution_name(name)
        if normalized_name in observed_distributions:
            raise ValueError("Dependency quarantine found duplicate distributions.")
        observed_distributions[normalized_name] = version
        distribution_files: list[dict[str, object]] = []
        try:
            rows = list(csv.reader(io.StringIO(record_text, newline="")))
        except csv.Error as error:
            raise ValueError("Dependency RECORD was invalid CSV.") from error
        if not rows:
            raise ValueError("Dependency RECORD was empty.")
        local_claims: set[str] = set()
        external_claims: set[str] = set()
        for row in rows:
            if len(row) != 3:
                raise ValueError("Dependency RECORD row shape was invalid.")
            installed_path, hash_field, size_field = row
            if _is_external_record_data_path(installed_path):
                if installed_path in external_claims:
                    raise ValueError("Dependency RECORD contained duplicate external claims.")
                external_claims.add(installed_path)
                continue
            if _is_derived_bytecode_path(installed_path) or _is_installer_data_path(
                installed_path
            ):
                continue
            _safe_relative_parts(installed_path)
            installed_entry = entries_by_path.get(installed_path)
            if installed_entry is None:
                raise ValueError("Dependency RECORD referenced a missing file.")
            if installed_path in local_claims or installed_path in claimed_paths:
                raise ValueError("Dependency RECORD closure contained duplicate claims.")
            local_claims.add(installed_path)
            claimed_paths.add(installed_path)
            if hash_field:
                if not re.fullmatch(r"sha256=[A-Za-z0-9_-]{43}", hash_field):
                    raise ValueError("Dependency RECORD hash was not canonical SHA-256.")
                expected_hash = base64.urlsafe_b64decode(
                    hash_field.removeprefix("sha256=") + "="
                ).hex()
                if not hmac.compare_digest(
                    f"sha256:{expected_hash}",
                    str(installed_entry["sha256"]),
                ):
                    raise ValueError("Dependency RECORD hash did not match installed bytes.")
            if size_field:
                if not re.fullmatch(r"0|[1-9][0-9]*", size_field):
                    raise ValueError("Dependency RECORD size was invalid.")
                if int(size_field) != installed_entry["size_bytes"]:
                    raise ValueError("Dependency RECORD size did not match installed bytes.")
            distribution_files.append(dict(installed_entry))
        if record_path not in local_claims or metadata_path not in local_claims:
            raise ValueError("Dependency distribution metadata was outside its RECORD closure.")
        distributions.append(
            {
                "name": normalized_name,
                "version": version,
                "record_path": record_path,
                "record_digest": tagged_sha256(record_payload),
                "file_count": len(distribution_files),
                "files_digest": tagged_sha256(
                    canonical_json(
                        sorted(distribution_files, key=lambda entry: str(entry["path"]))
                    )
                ),
            }
        )
    if claimed_paths != set(entries_by_path):
        raise ValueError("Dependency tree contained files outside all RECORD closures.")
    required_versions = {
        _normalize_distribution_name(name): version
        for name, version in (expected_versions or REQUIRED_DEPENDENCY_VERSIONS).items()
    }
    if expected_versions is not None and observed_distributions != required_versions:
        raise ValueError("Installed distribution closure did not match the trusted lock.")
    for required_name, required_version in required_versions.items():
        if observed_distributions.get(required_name) != required_version:
            raise ValueError("Dependency version did not match the required runtime manifest.")
    distributions.sort(key=lambda distribution: str(distribution["name"]))
    return distributions, tagged_sha256(canonical_json(distributions))


def _copy_tree_to_private_root(
    source_descriptor: int,
    entries: list[dict[str, object]],
    *,
    private_base: Path,
    tree_digest: str,
    executable_files: frozenset[str] = frozenset(),
    deadline_epoch: float | None = None,
) -> tuple[Path, int]:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", tree_digest):
        raise ValueError("Private tree digest was invalid.")
    base_descriptor = _open_safe_directory(private_base)
    target_name = tree_digest.removeprefix("sha256:")
    try:
        os.fchmod(base_descriptor, 0o700)
        try:
            target_descriptor = os.open(
                target_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=base_descriptor,
            )
        except FileNotFoundError:
            pending_name = f".pending-{secrets.token_hex(16)}"
            os.mkdir(pending_name, mode=0o700, dir_fd=base_descriptor)
            pending_descriptor = os.open(
                pending_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=base_descriptor,
            )
            directory_paths: set[tuple[str, ...]] = {()}
            try:
                entries_by_path = {str(entry["path"]): entry for entry in entries}
                for relative_path in sorted(entries_by_path):
                    _deadline_check(deadline_epoch)
                    entry = entries_by_path[relative_path]
                    parts = _safe_relative_parts(relative_path)
                    for length in range(1, len(parts)):
                        directory_paths.add(parts[:length])
                    source_parent = _open_relative_directory(source_descriptor, parts[:-1])
                    destination_parent = _open_relative_directory(
                        pending_descriptor,
                        parts[:-1],
                        create=True,
                    )
                    try:
                        source_flags = (
                            os.O_RDONLY
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_NONBLOCK", 0)
                        )
                        source_file = os.open(parts[-1], source_flags, dir_fd=source_parent)
                        destination_file = os.open(
                            parts[-1],
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=destination_parent,
                        )
                        try:
                            source_before = os.fstat(source_file)
                            if (
                                not stat.S_ISREG(source_before.st_mode)
                                or source_before.st_nlink != 1
                                or source_before.st_size != entry["size_bytes"]
                            ):
                                raise ValueError("Quarantine copy source metadata changed.")
                            digest = hashlib.sha256()
                            observed_size = 0
                            while True:
                                _deadline_check(deadline_epoch)
                                chunk = os.read(source_file, 1024 * 1024)
                                if not chunk:
                                    break
                                digest.update(chunk)
                                observed_size += len(chunk)
                                offset = 0
                                while offset < len(chunk):
                                    offset += os.write(destination_file, chunk[offset:])
                            source_after = os.fstat(source_file)
                            if (
                                _regular_file_identity(source_before)
                                != _regular_file_identity(source_after)
                                or observed_size != entry["size_bytes"]
                                or not hmac.compare_digest(
                                    f"sha256:{digest.hexdigest()}",
                                    str(entry["sha256"]),
                                )
                            ):
                                raise ValueError("Quarantine copy source changed.")
                            os.fchmod(
                                destination_file,
                                0o555 if relative_path in executable_files else 0o444,
                            )
                            os.fsync(destination_file)
                        finally:
                            os.close(destination_file)
                            os.close(source_file)
                        os.fsync(destination_parent)
                    finally:
                        os.close(destination_parent)
                        os.close(source_parent)
                for directory_parts in sorted(
                    directory_paths,
                    key=lambda parts: (len(parts), parts),
                    reverse=True,
                ):
                    directory_descriptor = _open_relative_directory(
                        pending_descriptor,
                        directory_parts,
                    )
                    try:
                        os.fchmod(directory_descriptor, 0o555)
                        os.fsync(directory_descriptor)
                    finally:
                        os.close(directory_descriptor)
                os.rename(
                    pending_name,
                    target_name,
                    src_dir_fd=base_descriptor,
                    dst_dir_fd=base_descriptor,
                )
                os.fsync(base_descriptor)
            finally:
                os.close(pending_descriptor)
            target_descriptor = os.open(
                target_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=base_descriptor,
            )
        metadata = os.fstat(target_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(target_descriptor)
            raise ValueError("Private quarantine target was not a directory.")
        return private_base / target_name, target_descriptor
    finally:
        os.close(base_descriptor)


def open_live_work_directory(path: Path) -> int:
    if (
        not path.is_absolute()
        or path.parent != LIVE_WORK_DIRECTORY_ROOT
        or not re.fullmatch(r"[A-Za-z0-9._-]+", path.name)
    ):
        raise ValueError("Live work directory is outside the exact safe root.")
    return _open_safe_directory(path)


def _directory_identity_matches(directory_descriptor: int, path: Path) -> bool:
    try:
        path_metadata = os.lstat(path)
        descriptor_metadata = os.fstat(directory_descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(path_metadata.st_mode)
        and not stat.S_ISLNK(path_metadata.st_mode)
        and path_metadata.st_dev == descriptor_metadata.st_dev
        and path_metadata.st_ino == descriptor_metadata.st_ino
    )


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
            raise ValueError("Durable staging artifact was not a bounded regular file.")
        payload = b""
        while len(payload) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        if len(payload) != metadata.st_size:
            raise ValueError("Durable staging artifact size changed while reading.")
        return payload
    finally:
        os.close(descriptor)


def _install_immutable_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> bytes:
    pending_name = f".pending-{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(pending_name, flags, 0o444, dir_fd=directory_descriptor)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        try:
            os.link(
                pending_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_regular_at(
                directory_descriptor,
                name,
                maximum_bytes=len(payload),
            )
            if not hmac.compare_digest(observed, payload):
                raise ValueError(
                    "Content-addressed staging path contains different bytes."
                ) from None
        os.fsync(directory_descriptor)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(pending_name, dir_fd=directory_descriptor)
    observed = _read_regular_at(
        directory_descriptor,
        name,
        maximum_bytes=len(payload),
    )
    if not hmac.compare_digest(observed, payload):
        raise ValueError("Durable staging artifact failed post-write verification.")
    return observed


def _write_atomic_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o444,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("Atomic artifact name was unsafe.")
    pending_name = f".pending-{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(pending_name, flags, mode, dir_fd=directory_descriptor)
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.replace(
            pending_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(pending_name, dir_fd=directory_descriptor)


def _read_json_at(
    directory_descriptor: int,
    name: str,
    description: str,
) -> dict[str, Any] | None:
    try:
        payload = _read_regular_at(
            directory_descriptor,
            name,
            maximum_bytes=MAXIMUM_RECEIPT_BYTES,
        )
    except FileNotFoundError:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} was not valid JSON.") from error
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ValueError(f"{description} was not canonical JSON.")
    return value


def _directory_entry_names(directory_descriptor: int) -> set[str]:
    with os.scandir(directory_descriptor) as iterator:
        entries = list(iterator)
    names: set[str] = set()
    for entry in entries:
        if (
            entry.name in {"", ".", ".."}
            or "/" in entry.name
            or "\\" in entry.name
            or entry.name in names
            or entry.is_symlink()
        ):
            raise ValueError("Live work directory contained an unsafe entry.")
        names.add(entry.name)
    return names


def _require_exact_stage_entries(
    directory_descriptor: int,
    *,
    expected: set[str],
) -> None:
    if _directory_entry_names(directory_descriptor) != expected:
        raise ValueError("Live work directory contained unjournaled stage artifacts.")


def _install_flat_live_bundle_at(
    payload: bytes,
    *,
    directory_descriptor: int,
    workload_file: str,
) -> None:
    expected_files = expected_bundle_files(workload_file)
    if any("/" in name for name in expected_files):
        raise ValueError("Live workload bundle members must be flat.")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
        _validate_archive_members(members, expected_files=expected_files)
        contents: dict[str, bytes] = {}
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Bundle member could not be read.")
            content = source.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError("Bundle member size did not match.")
            contents[member.name] = content
    for name in sorted(contents):
        _write_atomic_at(
            directory_descriptor,
            name,
            contents[name],
            mode=0o755 if name.endswith(".sh") else 0o444,
        )


def _install_or_reuse_stage_receipt(
    directory_descriptor: int,
    name: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    payload = canonical_json(receipt) + b"\n"
    try:
        _install_immutable_at(directory_descriptor, name, payload)
        return receipt
    except ValueError as error:
        try:
            observed_payload = _read_regular_at(
                directory_descriptor,
                name,
                maximum_bytes=MAXIMUM_RECEIPT_BYTES,
            )
            observed = json.loads(observed_payload)
        except (OSError, ValueError, json.JSONDecodeError):
            raise error from None
        comparable_keys = set(receipt) - {"staged_at", "receipt_digest"}
        if (
            not isinstance(observed, dict)
            or set(observed) != set(receipt)
            or any(observed.get(key) != receipt[key] for key in comparable_keys)
            or observed.get("receipt_digest") != _receipt_digest(observed)
        ):
            raise ValueError("Existing stage receipt conflicted with the upload.") from None
        return observed


def _write_atomic(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    pending = Path(pending_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        pending.unlink(missing_ok=True)


def _inspect_live_bundle(
    payload: bytes,
    *,
    workload_file: str,
    plan: LiveStagePlan,
) -> tuple[bytes, dict[str, Any]]:
    expanded = _expanded_xz_payload(payload)
    inspection_directory = Path(tempfile.mkdtemp(prefix=".bundle-inspect-"))
    try:
        install_bundle(
            expanded,
            work_directory=inspection_directory,
            workload_file=workload_file,
        )
        try:
            manifest = json.loads(
                (inspection_directory / "larger-model-eligibility.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Bundle manifest was invalid.") from error
        if not isinstance(manifest, dict) or manifest.get("profile_id") != plan.profile_id:
            raise ValueError("Bundle manifest profile did not match the plan.")
        if (
            not isinstance(manifest.get("screen"), dict)
            or manifest["screen"].get("live_stage_activation_revision")
            != BUNDLE_ACTIVATION_REVISION
        ):
            raise ValueError("Bundle activation revision did not match the bootstrap.")
        if manifest.get("materialization") != {
            "code": {
                "revision": CODE_MATERIALIZATION_REVISION,
                "evidence_schema_version": 1,
            },
            "dependency_lock": {
                "path": DEPENDENCY_LOCK_FILENAME,
                "digest": DEPENDENCY_LOCK_DIGEST,
                "size_bytes": DEPENDENCY_LOCK_SIZE_BYTES,
                "revision": DEPENDENCY_QUARANTINE_REVISION,
                "evidence_schema_version": 2,
                "installer_revision": DEPENDENCY_INSTALLER_REVISION,
                "python_version": DEPENDENCY_PYTHON_VERSION,
                "platform_tag": DEPENDENCY_PLATFORM_TAG,
                "index_url": DEPENDENCY_INDEX_URL,
                "network_policy": (
                    "hash-locked-binary-wheels-during-authenticated-preparation-only"
                ),
            },
        }:
            raise ValueError("Bundle materialization revisions did not match the bootstrap.")
        if not hmac.compare_digest(tagged_sha256(canonical_json(manifest)), plan.manifest_digest):
            raise ValueError("Bundle manifest digest did not match the plan.")
        source_contract = manifest.get("source_contract")
        if not isinstance(source_contract, dict) or source_contract.get("algorithm") != "sha256":
            raise ValueError("Bundle source contract was invalid.")
        expected_sources = source_contract.get("files")
        if not isinstance(expected_sources, dict) or not expected_sources:
            raise ValueError("Bundle source contract file set was invalid.")
        if expected_sources.get(DEPENDENCY_LOCK_FILENAME) != DEPENDENCY_LOCK_DIGEST.removeprefix(
            "sha256:"
        ):
            raise ValueError("Bundle source contract did not bind the dependency lock.")
        for name, expected_digest in expected_sources.items():
            if (
                not isinstance(name, str)
                or name not in expected_bundle_files(workload_file)
                or not isinstance(expected_digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
            ):
                raise ValueError("Bundle source contract file identity was invalid.")
            source_path = inspection_directory / name
            try:
                observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError("Bundle source contract file was unavailable.") from error
            if not hmac.compare_digest(observed, expected_digest):
                raise ValueError("Bundle source contract file digest did not match.")
        source_material = {
            "profile_id": plan.profile_id,
            "source_contract": source_contract,
        }
        if not hmac.compare_digest(
            tagged_sha256(canonical_json(source_material)),
            plan.source_contract_digest,
        ):
            raise ValueError("Bundle source contract digest did not match the plan.")
        return expanded, manifest
    finally:
        shutil.rmtree(inspection_directory, ignore_errors=True)


def build_bundle_stage_receipt(
    plan: LiveStagePlan,
    *,
    workload_file: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    staged_at = now or datetime.now(UTC)
    if staged_at.tzinfo is None or staged_at.utcoffset() is None:
        raise ValueError("Bundle staging time must be timezone-aware.")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": plan.profile_id,
        "manifest_digest": plan.manifest_digest,
        "source_contract_digest": plan.source_contract_digest,
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "bundle_digest": plan.workload_bundle_digest,
        "bundle_size_bytes": plan.workload_bundle_size_bytes,
        "bundle_compression": "xz",
        "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
        "bundle_path": plan.workload_bundle_path,
        "bundle_files": sorted(expected_bundle_files(workload_file)),
        "workload_files": [
            "repository_repair_large_model_eligibility.py",
            "repository_repair_large_model_pilot.py",
        ],
        "staged_at": staged_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def stage_uploaded_bundle(
    payload: bytes,
    *,
    work_directory: Path,
    workload_file: str,
    plan: LiveStagePlan,
    now: datetime | None = None,
    work_directory_descriptor: int | None = None,
) -> dict[str, Any]:
    if workload_file != "repository_repair_large_model_eligibility.py":
        raise ValueError("Live staging is restricted to the eligibility screen.")
    if len(payload) != plan.workload_bundle_size_bytes:
        raise ValueError("Uploaded bundle size did not match the plan.")
    if not hmac.compare_digest(tagged_sha256(payload), plan.workload_bundle_digest):
        raise ValueError("Uploaded bundle digest did not match the plan.")
    expanded, _manifest = _inspect_live_bundle(
        payload,
        workload_file=workload_file,
        plan=plan,
    )
    destination = Path(plan.workload_bundle_path)
    profile_descriptor = _open_safe_directory(destination.parent)
    try:
        _install_immutable_at(profile_descriptor, destination.name, payload)
        receipt = build_bundle_stage_receipt(
            plan,
            workload_file=workload_file,
            now=now,
        )
        immutable_receipt_name = (
            f"{plan.workload_bundle_digest.removeprefix('sha256:')}.receipt.json"
        )
        receipt = _install_or_reuse_stage_receipt(
            profile_descriptor,
            immutable_receipt_name,
            receipt,
        )
    finally:
        os.close(profile_descriptor)
    owned_descriptor = work_directory_descriptor is None
    directory_descriptor = (
        open_live_work_directory(work_directory)
        if work_directory_descriptor is None
        else work_directory_descriptor
    )
    try:
        if not _directory_identity_matches(directory_descriptor, work_directory):
            raise ValueError("Live work directory identity changed.")
        _install_flat_live_bundle_at(
            expanded,
            directory_descriptor=directory_descriptor,
            workload_file=workload_file,
        )
        _write_atomic_at(
            directory_descriptor,
            BUNDLE_STAGE_RECEIPT_FILENAME,
            canonical_json(receipt) + b"\n",
        )
    finally:
        if owned_descriptor:
            os.close(directory_descriptor)
    return receipt


@dataclass(frozen=True)
class QuarantinedTree:
    evidence: dict[str, Any]
    private_root: Path
    private_root_descriptor: int


@dataclass(frozen=True)
class PrivateRuntimeRoot:
    path: Path
    descriptor: int


@dataclass(frozen=True)
class PrivateAnchorRoot:
    path: Path
    descriptor: int


def open_private_runtime_root(
    *,
    profile_id: str,
    bundle_digest: str,
    work_directory_identity: str,
) -> PrivateRuntimeRoot:
    if (
        not re.fullmatch(r"[A-Za-z0-9._@-]+", profile_id)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle_digest)
        or not Path(work_directory_identity).is_absolute()
    ):
        raise ValueError("Private runtime identity was invalid.")
    runtime_binding = {
        "revision": PRIVATE_RUNTIME_REVISION,
        "profile_id": profile_id,
        "bundle_digest": bundle_digest,
        "work_directory_identity": work_directory_identity,
    }
    runtime_name = _launch_hmac(runtime_binding).removeprefix("hmac-sha256:")
    base_descriptor = _open_safe_directory(PRIVATE_RUNTIME_ROOT)
    try:
        os.fchmod(base_descriptor, 0o700)
        with suppress(FileExistsError):
            os.mkdir(runtime_name, mode=0o700, dir_fd=base_descriptor)
            os.fsync(base_descriptor)
        descriptor = os.open(
            runtime_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=base_descriptor,
        )
    finally:
        os.close(base_descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        os.close(descriptor)
        raise ValueError("Private runtime root permissions were unsafe.")
    runtime_path = PRIVATE_RUNTIME_ROOT / runtime_name
    if not _directory_identity_matches(descriptor, runtime_path):
        os.close(descriptor)
        raise ValueError("Private runtime root identity changed.")
    os.set_inheritable(descriptor, True)
    return PrivateRuntimeRoot(runtime_path, descriptor)


def open_private_anchor_root(
    *,
    profile_id: str,
    bundle_digest: str,
    work_directory_identity: str,
) -> PrivateAnchorRoot:
    if (
        not re.fullmatch(r"[A-Za-z0-9._@-]+", profile_id)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle_digest)
        or not Path(work_directory_identity).is_absolute()
    ):
        raise ValueError("Private anchor identity was invalid.")
    anchor_binding = {
        "revision": PRIVATE_ANCHOR_REVISION,
        "profile_id": profile_id,
        "bundle_digest": bundle_digest,
        "work_directory_identity": work_directory_identity,
    }
    anchor_name = _launch_hmac(anchor_binding).removeprefix("hmac-sha256:")
    base_descriptor = _open_safe_directory(PRIVATE_ANCHOR_ROOT)
    try:
        os.fchmod(base_descriptor, 0o700)
        with suppress(FileExistsError):
            os.mkdir(anchor_name, mode=0o700, dir_fd=base_descriptor)
            os.fsync(base_descriptor)
        descriptor = os.open(
            anchor_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=base_descriptor,
        )
    finally:
        os.close(base_descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        os.close(descriptor)
        raise ValueError("Private anchor root permissions were unsafe.")
    anchor_path = PRIVATE_ANCHOR_ROOT / anchor_name
    if not _directory_identity_matches(descriptor, anchor_path):
        os.close(descriptor)
        raise ValueError("Private anchor root identity changed.")
    os.set_inheritable(descriptor, True)
    return PrivateAnchorRoot(anchor_path, descriptor)


def _locked_requirement_versions(lock_payload: bytes) -> dict[str, str]:
    try:
        lines = lock_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Dependency lock was not UTF-8.") from error
    versions: dict[str, str] = {}
    hashes: set[str] = set()
    active_name: str | None = None
    active_hash_count = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            if active_name is not None and active_hash_count < 1:
                raise ValueError("Dependency lock requirement had no hash.")
            match = re.fullmatch(
                r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*) \\",
                line,
            )
            if match is None:
                raise ValueError("Dependency lock requirement was not exact.")
            active_name = _normalize_distribution_name(match.group(1))
            if active_name in versions:
                raise ValueError("Dependency lock contained a duplicate requirement.")
            versions[active_name] = match.group(2)
            active_hash_count = 0
            continue
        if active_name is None:
            raise ValueError("Dependency lock hash had no requirement.")
        match = re.fullmatch(r"    --hash=sha256:([0-9a-f]{64})(?: \\)?", line)
        if match is None or match.group(1) in hashes:
            raise ValueError("Dependency lock hash line was invalid or duplicated.")
        hashes.add(match.group(1))
        active_hash_count += 1
    if active_name is None or active_hash_count < 1:
        raise ValueError("Dependency lock was incomplete.")
    if len(versions) != 30 or "torch" in versions:
        raise ValueError("Dependency lock package closure was invalid.")
    for name, version in REQUIRED_DEPENDENCY_VERSIONS.items():
        if versions.get(name) != version:
            raise ValueError("Dependency lock direct version was invalid.")
    return versions


def _run_locked_dependency_installer(
    *,
    code_root_descriptor: int,
    target_descriptor: int,
    deadline_epoch: float | None,
) -> None:
    if (
        f"{sys.version_info.major}.{sys.version_info.minor}" != DEPENDENCY_PYTHON_VERSION
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
    ):
        raise ValueError("Dependency installer runtime did not match the locked platform.")
    remaining = (
        PREPARATION_DEADLINE_SECONDS
        if deadline_epoch is None
        else min(PREPARATION_DEADLINE_SECONDS, deadline_epoch - time.time())
    )
    if remaining <= 0:
        raise TimeoutError("Dependency installation deadline expired.")
    stable_code_root = Path(f"/proc/self/fd/{code_root_descriptor}")
    stable_target = Path(f"/proc/self/fd/{target_descriptor}")
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
    }
    for name in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--require-hashes",
                "--only-binary=:all:",
                "--no-deps",
                "--no-compile",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--progress-bar",
                "off",
                "--index-url",
                DEPENDENCY_INDEX_URL,
                "--target",
                str(stable_target),
                "--requirement",
                str(stable_code_root / DEPENDENCY_LOCK_FILENAME),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            pass_fds=(code_root_descriptor, target_descriptor),
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Hash-locked dependency installation exceeded its deadline.") from error
    if completed.returncode != 0:
        raise ValueError("Hash-locked dependency installation failed.")


def _new_private_build_directory(private_base: Path) -> tuple[Path, int]:
    build_base = private_base.parent / "dependency-builds"
    base_descriptor = _open_safe_directory(build_base)
    try:
        os.fchmod(base_descriptor, 0o700)
        name = f".pending-{secrets.token_hex(16)}"
        os.mkdir(name, mode=0o700, dir_fd=base_descriptor)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=base_descriptor,
        )
        os.set_inheritable(descriptor, True)
        return build_base / name, descriptor
    finally:
        os.close(base_descriptor)


def _dependency_import_entries(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        entry
        for entry in entries
        if not _is_derived_bytecode_path(str(entry["path"]))
        and not _is_installer_data_path(str(entry["path"]))
    ]


def quarantine_dependencies(
    *,
    profile_id: str,
    code_root_descriptor: int,
    lock_path_identity: str,
    private_base: Path = PRIVATE_DEPENDENCY_ROOT,
    deadline_epoch: float | None = None,
) -> QuarantinedTree:
    lock_entry = _digest_regular_file_at(
        code_root_descriptor,
        DEPENDENCY_LOCK_FILENAME,
        maximum_bytes=DEPENDENCY_LOCK_SIZE_BYTES,
        deadline_epoch=deadline_epoch,
    )
    if (
        lock_entry["size_bytes"] != DEPENDENCY_LOCK_SIZE_BYTES
        or not hmac.compare_digest(str(lock_entry["sha256"]), DEPENDENCY_LOCK_DIGEST)
    ):
        raise ValueError("Bundled dependency lock identity did not match the profile.")
    lock_payload = _read_relative_regular(
        code_root_descriptor,
        DEPENDENCY_LOCK_FILENAME,
        expected_entry=lock_entry,
        maximum_bytes=DEPENDENCY_LOCK_SIZE_BYTES,
        deadline_epoch=deadline_epoch,
    )
    locked_versions = _locked_requirement_versions(lock_payload)
    build_root, build_descriptor = _new_private_build_directory(private_base)
    private_descriptor: int | None = None
    try:
        _run_locked_dependency_installer(
            code_root_descriptor=code_root_descriptor,
            target_descriptor=build_descriptor,
            deadline_epoch=deadline_epoch,
        )
        installed_tree = _scan_regular_tree(
            build_descriptor,
            maximum_files=MAXIMUM_DEPENDENCY_FILES,
            maximum_bytes=MAXIMUM_DEPENDENCY_BYTES,
            deadline_epoch=deadline_epoch,
        )
        install_entries = _dependency_import_entries(installed_tree)
        if not install_entries:
            raise ValueError("Hash-locked dependency install was empty.")
        distributions, record_closure_digest = _dependency_record_closure(
            build_descriptor,
            install_entries,
            expected_versions=locked_versions,
        )
        install_tree_digest = _tree_digest(install_entries)
        private_root, private_descriptor = _copy_tree_to_private_root(
            build_descriptor,
            install_entries,
            private_base=private_base,
            tree_digest=install_tree_digest,
            deadline_epoch=deadline_epoch,
        )
        final_install_tree = _dependency_import_entries(
            _scan_regular_tree(
                build_descriptor,
                maximum_files=MAXIMUM_DEPENDENCY_FILES,
                maximum_bytes=MAXIMUM_DEPENDENCY_BYTES,
                deadline_epoch=deadline_epoch,
            )
        )
        final_distributions, final_record_closure_digest = _dependency_record_closure(
            build_descriptor,
            final_install_tree,
            expected_versions=locked_versions,
        )
        private_entries = _scan_regular_tree(
            private_descriptor,
            maximum_files=MAXIMUM_DEPENDENCY_FILES,
            maximum_bytes=MAXIMUM_DEPENDENCY_BYTES,
            deadline_epoch=deadline_epoch,
        )
        _verify_hardened_tree(private_descriptor)
        if (
            final_install_tree != install_entries
            or final_distributions != distributions
            or final_record_closure_digest != record_closure_digest
            or private_entries != install_entries
        ):
            raise ValueError("Dependency tree changed during private materialization.")
        private_tree_digest = _tree_digest(private_entries)
        if not hmac.compare_digest(private_tree_digest, install_tree_digest):
            raise ValueError("Private dependency tree digest did not match its install.")
        evidence: dict[str, Any] = {
            "schema_version": 2,
            "revision": DEPENDENCY_QUARANTINE_REVISION,
            "profile_id": profile_id,
            "lock_path": lock_path_identity,
            "lock_digest": DEPENDENCY_LOCK_DIGEST,
            "python_version": DEPENDENCY_PYTHON_VERSION,
            "platform_tag": DEPENDENCY_PLATFORM_TAG,
            "installer_revision": DEPENDENCY_INSTALLER_REVISION,
            "index_url": DEPENDENCY_INDEX_URL,
            "private_root": str(private_root),
            "install_tree_digest": install_tree_digest,
            "private_tree_digest": private_tree_digest,
            "record_closure_digest": record_closure_digest,
            "distributions": distributions,
            "installed_file_count": len(install_entries),
            "installed_bytes": sum(int(entry["size_bytes"]) for entry in install_entries),
            "ready": True,
        }
        evidence["evidence_digest"] = _evidence_digest(evidence)
        os.set_inheritable(private_descriptor, True)
        return QuarantinedTree(evidence, private_root, private_descriptor)
    except BaseException:
        if private_descriptor is not None:
            os.close(private_descriptor)
        raise
    finally:
        os.close(build_descriptor)
        with suppress(OSError):
            shutil.rmtree(build_root)


def materialize_code(
    source_descriptor: int,
    *,
    workload_file: str,
    profile_id: str,
    bundle_digest: str,
    bundle_size_bytes: int,
    source_contract_digest: str,
    source_root_identity: str,
    private_base: Path = PRIVATE_CODE_ROOT,
    deadline_epoch: float | None = None,
) -> QuarantinedTree:
    expected_files = expected_bundle_files(workload_file)
    initial_entries = _scan_allowlisted_files(
        source_descriptor,
        expected_files,
        maximum_bytes=MAXIMUM_EXPANDED_BUNDLE_BYTES,
        deadline_epoch=deadline_epoch,
    )
    source_tree_digest = _tree_digest(initial_entries)
    private_descriptor: int | None = None
    try:
        private_root, private_descriptor = _copy_tree_to_private_root(
            source_descriptor,
            initial_entries,
            private_base=private_base,
            tree_digest=source_tree_digest,
            executable_files=frozenset(
                path for path in expected_files if path.endswith(".sh")
            ),
            deadline_epoch=deadline_epoch,
        )
        final_source_entries = _scan_allowlisted_files(
            source_descriptor,
            expected_files,
            maximum_bytes=MAXIMUM_EXPANDED_BUNDLE_BYTES,
            deadline_epoch=deadline_epoch,
        )
        private_entries = _scan_regular_tree(
            private_descriptor,
            maximum_files=len(expected_files),
            maximum_bytes=MAXIMUM_EXPANDED_BUNDLE_BYTES,
            deadline_epoch=deadline_epoch,
        )
        _verify_hardened_tree(
            private_descriptor,
            executable_files=frozenset(
                path for path in expected_files if path.endswith(".sh")
            ),
        )
        if final_source_entries != initial_entries or private_entries != initial_entries:
            raise ValueError("Code tree changed during private materialization.")
        private_tree_digest = _tree_digest(private_entries)
        if not hmac.compare_digest(private_tree_digest, source_tree_digest):
            raise ValueError("Private code tree digest did not match its source.")
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "revision": CODE_MATERIALIZATION_REVISION,
            "profile_id": profile_id,
            "bundle_digest": bundle_digest,
            "bundle_size_bytes": bundle_size_bytes,
            "source_contract_digest": source_contract_digest,
            "source_root": source_root_identity,
            "private_root": str(private_root),
            "source_tree_digest": source_tree_digest,
            "private_tree_digest": private_tree_digest,
            "files": initial_entries,
            "installed_file_count": len(initial_entries),
            "installed_bytes": sum(int(entry["size_bytes"]) for entry in initial_entries),
            "ready": True,
        }
        evidence["evidence_digest"] = _evidence_digest(evidence)
        os.set_inheritable(private_descriptor, True)
        return QuarantinedTree(evidence, private_root, private_descriptor)
    except BaseException:
        if private_descriptor is not None:
            os.close(private_descriptor)
        raise


class BootstrapServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        plan: LiveStagePlan | None = None,
        work_directory: Path | None = None,
        work_directory_descriptor: int | None = None,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.bundle_ready = False
        self.bundle_activated = False
        self.state = "awaiting_bundle"
        self.plan = plan
        self.state_lock = Lock()
        self.bundle_stage_receipt: dict[str, Any] | None = None
        self.volume_readiness_receipt: dict[str, Any] | None = None
        self.torch_retention_evidence: dict[str, Any] | None = None
        self.dependency_quarantine_evidence: dict[str, Any] | None = None
        self.code_materialization_evidence: dict[str, Any] | None = None
        self.dependency_root_descriptor: int | None = None
        self.code_root_descriptor: int | None = None
        self.activation: dict[str, Any] | None = None
        self.work_directory = work_directory
        self.work_directory_descriptor = work_directory_descriptor
        self.deadline_timer: Timer | None = None


def _canonical_json_line(payload: bytes, name: str) -> dict[str, Any]:
    if not payload or len(payload) > MAXIMUM_RECEIPT_BYTES:
        raise ValueError(f"{name} output size was invalid.")
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise ValueError(f"{name} was not one canonical JSON line.")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} was not valid JSON.") from error
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ValueError(f"{name} was not canonical JSON.")
    return value


def _remaining_preparation_seconds(server: BootstrapServer) -> float:
    if server.plan is None:
        raise TimeoutError("Bootstrap preparation has no readiness deadline.")
    remaining = min(
        PREPARATION_DEADLINE_SECONDS,
        server.plan.readiness_deadline_epoch - time.time(),
    )
    if remaining <= 0:
        raise TimeoutError("Bootstrap preparation deadline expired.")
    return remaining


def _run_preparation_command(
    command: list[str],
    *,
    environment: Mapping[str, str],
    server: BootstrapServer,
    name: str,
    pass_fds: tuple[int, ...] = (),
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=dict(environment),
            pass_fds=pass_fds,
            timeout=_remaining_preparation_seconds(server),
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"{name} exceeded the preparation deadline.") from error
    if completed.returncode != 0:
        raise ValueError(f"{name} failed with a nonzero exit status.")
    return _canonical_json_line(completed.stdout, name)


def _close_server_quarantine_descriptors(server: BootstrapServer) -> None:
    for attribute in ("dependency_root_descriptor", "code_root_descriptor"):
        descriptor = getattr(server, attribute)
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
            setattr(server, attribute, None)


def _recover_dependency_quarantine(
    evidence: dict[str, Any],
    *,
    profile_id: str,
    lock_path_identity: str,
) -> QuarantinedTree:
    expected_keys = {
        "schema_version",
        "revision",
        "profile_id",
        "lock_path",
        "lock_digest",
        "python_version",
        "platform_tag",
        "installer_revision",
        "index_url",
        "private_root",
        "install_tree_digest",
        "private_tree_digest",
        "record_closure_digest",
        "distributions",
        "installed_file_count",
        "installed_bytes",
        "ready",
        "evidence_digest",
    }
    private_root = Path(str(evidence.get("private_root", "")))
    if (
        set(evidence) != expected_keys
        or evidence.get("schema_version") != 2
        or evidence.get("revision") != DEPENDENCY_QUARANTINE_REVISION
        or evidence.get("profile_id") != profile_id
        or evidence.get("lock_path") != lock_path_identity
        or evidence.get("lock_digest") != DEPENDENCY_LOCK_DIGEST
        or evidence.get("python_version") != DEPENDENCY_PYTHON_VERSION
        or evidence.get("platform_tag") != DEPENDENCY_PLATFORM_TAG
        or evidence.get("installer_revision") != DEPENDENCY_INSTALLER_REVISION
        or evidence.get("index_url") != DEPENDENCY_INDEX_URL
        or evidence.get("ready") is not True
        or evidence.get("evidence_digest") != _evidence_digest(evidence)
        or private_root.parent != PRIVATE_DEPENDENCY_ROOT
        or private_root.name
        != str(evidence.get("private_tree_digest", "")).removeprefix("sha256:")
    ):
        raise ValueError("Recovered dependency quarantine evidence was invalid.")
    descriptor = _open_existing_safe_directory(private_root)
    os.set_inheritable(descriptor, True)
    return QuarantinedTree(evidence, private_root, descriptor)


def materialize_private_execution(
    *,
    work_directory_descriptor: int,
    workload_file: str,
    profile_id: str,
    bundle_digest: str,
    bundle_size_bytes: int,
    bundle_path: str,
    source_contract_digest: str,
    deadline_epoch: float | None,
    recovered_dependency_evidence: dict[str, Any] | None = None,
    recovered_code_evidence: dict[str, Any] | None = None,
) -> tuple[QuarantinedTree, QuarantinedTree]:
    code = materialize_code(
        work_directory_descriptor,
        workload_file=workload_file,
        profile_id=profile_id,
        bundle_digest=bundle_digest,
        bundle_size_bytes=bundle_size_bytes,
        source_contract_digest=source_contract_digest,
        source_root_identity=bundle_path,
        deadline_epoch=deadline_epoch,
    )
    if recovered_code_evidence is not None and code.evidence != recovered_code_evidence:
        os.close(code.private_root_descriptor)
        raise ValueError("Rematerialized code evidence changed.")
    try:
        lock_path_identity = f"{bundle_path}::{DEPENDENCY_LOCK_FILENAME}"
        if recovered_dependency_evidence is None:
            dependencies = quarantine_dependencies(
                profile_id=profile_id,
                code_root_descriptor=code.private_root_descriptor,
                lock_path_identity=lock_path_identity,
                deadline_epoch=deadline_epoch,
            )
        else:
            try:
                dependencies = _recover_dependency_quarantine(
                    recovered_dependency_evidence,
                    profile_id=profile_id,
                    lock_path_identity=lock_path_identity,
                )
            except FileNotFoundError:
                dependencies = quarantine_dependencies(
                    profile_id=profile_id,
                    code_root_descriptor=code.private_root_descriptor,
                    lock_path_identity=lock_path_identity,
                    deadline_epoch=deadline_epoch,
                )
                if dependencies.evidence != recovered_dependency_evidence:
                    os.close(dependencies.private_root_descriptor)
                    raise ValueError(
                        "Rematerialized dependency evidence changed."
                    ) from None
            _verify_private_execution_trees(
                dependency_descriptor=dependencies.private_root_descriptor,
                code_descriptor=code.private_root_descriptor,
                dependency_evidence=dependencies.evidence,
                code_evidence=code.evidence,
            )
    except BaseException:
        os.close(code.private_root_descriptor)
        raise
    try:
        _write_atomic_at(
            work_directory_descriptor,
            DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
            canonical_json(dependencies.evidence) + b"\n",
        )
        _write_atomic_at(
            work_directory_descriptor,
            CODE_MATERIALIZATION_EVIDENCE_FILENAME,
            canonical_json(code.evidence) + b"\n",
        )
    except BaseException:
        os.close(dependencies.private_root_descriptor)
        os.close(code.private_root_descriptor)
        raise
    return dependencies, code


def prepare_private_materialization(
    server: BootstrapServer,
    *,
    work_directory: Path,
    recovered_dependency_evidence: dict[str, Any] | None = None,
    recovered_code_evidence: dict[str, Any] | None = None,
    enforce_deadline: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = server.plan
    directory_descriptor = server.work_directory_descriptor
    if plan is None or directory_descriptor is None:
        raise ValueError("Private materialization was not bound to a live stage.")
    if not _directory_identity_matches(directory_descriptor, work_directory):
        raise ValueError("Live work directory identity changed before materialization.")
    deadline_epoch = plan.readiness_deadline_epoch if enforce_deadline else None
    dependencies, code = materialize_private_execution(
        work_directory_descriptor=directory_descriptor,
        workload_file="repository_repair_large_model_eligibility.py",
        profile_id=plan.profile_id,
        bundle_digest=plan.workload_bundle_digest,
        bundle_size_bytes=plan.workload_bundle_size_bytes,
        bundle_path=plan.workload_bundle_path,
        source_contract_digest=plan.source_contract_digest,
        deadline_epoch=deadline_epoch,
        recovered_dependency_evidence=recovered_dependency_evidence,
        recovered_code_evidence=recovered_code_evidence,
    )
    _close_server_quarantine_descriptors(server)
    server.dependency_quarantine_evidence = dependencies.evidence
    server.code_materialization_evidence = code.evidence
    server.dependency_root_descriptor = dependencies.private_root_descriptor
    server.code_root_descriptor = code.private_root_descriptor
    return dependencies.evidence, code.evidence


def prepare_pilot_private_materialization(
    *,
    work_directory_descriptor: int,
    workload_file: str,
    deadline_epoch: float,
) -> tuple[QuarantinedTree, QuarantinedTree]:
    profile_id = os.environ.get("EQUINOX_LARGER_MODEL_PROFILE_ID", "")
    bundle_digest = os.environ.get("EQUINOX_BUNDLE_SHA256", "")
    bundle_size_raw = os.environ.get("EQUINOX_BUNDLE_SIZE_BYTES", "")
    bundle_path = os.environ.get("EQUINOX_BUNDLE_VOLUME_PATH", "")
    source_contract_digest = os.environ.get(
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
        "",
    )
    try:
        bundle_size_bytes = int(bundle_size_raw)
    except ValueError as error:
        raise ValueError("Pilot private materialization bundle size was invalid.") from error
    if (
        not re.fullmatch(r"[A-Za-z0-9._@-]+", profile_id)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle_digest)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_contract_digest)
        or not bundle_path
    ):
        raise ValueError("Pilot private materialization identity was invalid.")
    dependencies, code = materialize_private_execution(
        work_directory_descriptor=work_directory_descriptor,
        workload_file=workload_file,
        profile_id=profile_id,
        bundle_digest=bundle_digest,
        bundle_size_bytes=bundle_size_bytes,
        bundle_path=bundle_path,
        source_contract_digest=source_contract_digest,
        deadline_epoch=deadline_epoch,
    )
    expected_bindings = {
        "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": dependencies.evidence[
            "evidence_digest"
        ],
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": dependencies.evidence[
            "private_tree_digest"
        ],
        "EQUINOX_DEPENDENCY_LOCK_SHA256": dependencies.evidence["lock_digest"],
        "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": code.evidence[
            "evidence_digest"
        ],
        "EQUINOX_CODE_PRIVATE_TREE_SHA256": code.evidence["private_tree_digest"],
    }
    if any(
        not hmac.compare_digest(os.environ.get(name, ""), str(value))
        for name, value in expected_bindings.items()
    ):
        os.close(dependencies.private_root_descriptor)
        os.close(code.private_root_descriptor)
        raise ValueError("Pilot private materialization did not match its authorization.")
    return dependencies, code


def pilot_private_materialization_deadline() -> float:
    current = time.time()
    raw = os.environ.get("EQUINOX_PRIVATE_MATERIALIZATION_DEADLINE_EPOCH", "")
    if not raw:
        return current + PREPARATION_DEADLINE_SECONDS
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise ValueError("Pilot private materialization deadline was invalid.")
    deadline = float(int(raw))
    if deadline <= current or deadline > current + MAXIMUM_READINESS_WINDOW_SECONDS:
        raise ValueError("Pilot private materialization deadline was outside the safe window.")
    return deadline


def _prepare_stage_evidence_at(
    server: BootstrapServer,
    *,
    work_directory: Path,
    directory_descriptor: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = server.plan
    stage_receipt = server.bundle_stage_receipt
    dependency_evidence = server.dependency_quarantine_evidence
    code_evidence = server.code_materialization_evidence
    dependency_descriptor = server.dependency_root_descriptor
    code_descriptor = server.code_root_descriptor
    if (
        plan is None
        or stage_receipt is None
        or dependency_evidence is None
        or code_evidence is None
        or dependency_descriptor is None
        or code_descriptor is None
    ):
        raise ValueError("Live stage evidence preparation was not initialized.")
    if not _directory_identity_matches(directory_descriptor, work_directory):
        raise ValueError("Live work directory identity changed before preparation.")
    environment = dict(os.environ)
    stable_work_directory = Path(f"/proc/self/fd/{directory_descriptor}")
    stable_dependency_root = Path(f"/proc/self/fd/{dependency_descriptor}")
    stable_code_root = Path(f"/proc/self/fd/{code_descriptor}")
    dependency_evidence_path = (
        stable_work_directory / DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME
    )
    code_evidence_path = stable_work_directory / CODE_MATERIALIZATION_EVIDENCE_FILENAME
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(stable_code_root), str(stable_dependency_root))
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256": str(stage_receipt["receipt_digest"]),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH": str(
                dependency_evidence_path
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": str(
                dependency_evidence["evidence_digest"]
            ),
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": str(
                dependency_evidence["private_tree_digest"]
            ),
            "EQUINOX_DEPENDENCY_LOCK_SHA256": str(
                dependency_evidence["lock_digest"]
            ),
            "EQUINOX_DEPENDENCY_ROOT": str(stable_dependency_root),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH": str(code_evidence_path),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": str(
                code_evidence["evidence_digest"]
            ),
            "EQUINOX_CODE_PRIVATE_TREE_SHA256": str(
                code_evidence["private_tree_digest"]
            ),
            "EQUINOX_CODE_ROOT": str(stable_code_root),
        }
    )
    manifest_path = stable_code_root / "larger-model-eligibility.json"
    preparation_fds = (
        directory_descriptor,
        dependency_descriptor,
        code_descriptor,
    )
    volume_receipt = _run_preparation_command(
        [
            sys.executable,
            str(stable_code_root / "larger_model_gate.py"),
            "--manifest",
            str(manifest_path),
            "create-attested-volume-receipt",
            "--volume-id",
            plan.network_volume_id,
            "--data-center-id",
            plan.network_volume_data_center_id,
            "--volume-size-gb",
            str(plan.network_volume_size_gb),
            "--dependency-quarantine-evidence",
            str(dependency_evidence_path),
            "--dependency-quarantine-evidence-digest",
            str(dependency_evidence["evidence_digest"]),
            "--dependency-private-tree-digest",
            str(dependency_evidence["private_tree_digest"]),
            "--dependency-lock-digest",
            str(dependency_evidence["lock_digest"]),
            "--code-materialization-evidence",
            str(code_evidence_path),
            "--code-materialization-evidence-digest",
            str(code_evidence["evidence_digest"]),
            "--code-private-tree-digest",
            str(code_evidence["private_tree_digest"]),
        ],
        environment=environment,
        server=server,
        name="volume readiness attestation",
        pass_fds=preparation_fds,
    )
    if (
        volume_receipt.get("profile_id") != plan.profile_id
        or volume_receipt.get("manifest_digest") != plan.manifest_digest
        or volume_receipt.get("network_volume_id") != plan.network_volume_id
        or volume_receipt.get("network_volume_data_center_id") != plan.network_volume_data_center_id
        or volume_receipt.get("network_volume_size_gb") != plan.network_volume_size_gb
        or volume_receipt.get("dependency_quarantine_evidence")
        != dependency_evidence
        or volume_receipt.get("dependency_lock_digest")
        != dependency_evidence["lock_digest"]
        or volume_receipt.get("dependency_quarantine_evidence_digest")
        != dependency_evidence["evidence_digest"]
        or volume_receipt.get("dependency_private_tree_digest")
        != dependency_evidence["private_tree_digest"]
        or volume_receipt.get("code_materialization_evidence") != code_evidence
        or volume_receipt.get("code_materialization_evidence_digest")
        != code_evidence["evidence_digest"]
        or volume_receipt.get("code_private_tree_digest")
        != code_evidence["private_tree_digest"]
        or volume_receipt.get("ready") is not True
        or volume_receipt.get("receipt_digest") != _receipt_digest(volume_receipt)
    ):
        raise ValueError("Volume readiness attestation did not match the stage plan.")
    volume_path = stable_work_directory / VOLUME_READINESS_RECEIPT_FILENAME
    _write_atomic_at(
        directory_descriptor,
        VOLUME_READINESS_RECEIPT_FILENAME,
        canonical_json(volume_receipt) + b"\n",
    )
    retention_evidence = _run_preparation_command(
        [
            sys.executable,
            str(stable_code_root / "retention_checkpoint_probe.py"),
            "--flat-bundle-root",
            str(stable_code_root),
            "--head-commit",
            plan.head_commit,
            "--workload-bundle-digest",
            plan.workload_bundle_digest,
            "--workload-bundle-size-bytes",
            str(plan.workload_bundle_size_bytes),
            "--workload-bundle-path",
            plan.workload_bundle_path,
            "--bundle-stage-receipt-digest",
            str(stage_receipt["receipt_digest"]),
            "--bootstrap-source-digest",
            plan.bootstrap_source_digest,
            "--volume-readiness-receipt",
            str(volume_path),
            "--network-volume-id",
            plan.network_volume_id,
            "--data-center-id",
            plan.network_volume_data_center_id,
            "--volume-size-gb",
            str(plan.network_volume_size_gb),
            "--dependency-quarantine-evidence",
            str(dependency_evidence_path),
            "--dependency-quarantine-evidence-digest",
            str(dependency_evidence["evidence_digest"]),
            "--dependency-private-tree-digest",
            str(dependency_evidence["private_tree_digest"]),
            "--dependency-lock-digest",
            str(dependency_evidence["lock_digest"]),
            "--code-materialization-evidence",
            str(code_evidence_path),
            "--code-materialization-evidence-digest",
            str(code_evidence["evidence_digest"]),
            "--code-private-tree-digest",
            str(code_evidence["private_tree_digest"]),
        ],
        environment=environment,
        server=server,
        name="Torch retention checkpoint probe",
        pass_fds=preparation_fds,
    )
    retention_identity = {
        "profile_id": plan.profile_id,
        "head_commit": plan.head_commit,
        "source_contract_digest": plan.source_contract_digest,
        "workload_bundle_digest": plan.workload_bundle_digest,
        "workload_bundle_size_bytes": plan.workload_bundle_size_bytes,
        "workload_bundle_path": plan.workload_bundle_path,
        "bundle_stage_receipt_digest": stage_receipt["receipt_digest"],
        "bootstrap_source_digest": plan.bootstrap_source_digest,
        "volume_readiness_receipt_digest": volume_receipt["receipt_digest"],
        "dependency_quarantine_evidence_digest": dependency_evidence["evidence_digest"],
        "dependency_private_tree_digest": dependency_evidence["private_tree_digest"],
        "dependency_lock_digest": dependency_evidence["lock_digest"],
        "code_materialization_evidence_digest": code_evidence["evidence_digest"],
        "code_private_tree_digest": code_evidence["private_tree_digest"],
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "status": "passed",
    }
    if any(
        retention_evidence.get(key) != value for key, value in retention_identity.items()
    ) or retention_evidence.get("evidence_digest") != _evidence_digest(retention_evidence):
        raise ValueError("Torch retention evidence did not match the stage plan.")
    _write_atomic_at(
        directory_descriptor,
        TORCH_RETENTION_EVIDENCE_FILENAME,
        canonical_json(retention_evidence) + b"\n",
    )
    return volume_receipt, retention_evidence


def prepare_stage_evidence(
    server: BootstrapServer,
    *,
    work_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    owned_descriptor = server.work_directory_descriptor is None
    descriptor = (
        open_live_work_directory(work_directory)
        if owned_descriptor
        else server.work_directory_descriptor
    )
    if descriptor is None:
        raise ValueError("Live work directory descriptor was unavailable.")
    try:
        return _prepare_stage_evidence_at(
            server,
            work_directory=work_directory,
            directory_descriptor=descriptor,
        )
    finally:
        if owned_descriptor:
            os.close(descriptor)


def expected_activation(server: BootstrapServer) -> dict[str, Any]:
    plan = server.plan
    stage = server.bundle_stage_receipt
    volume = server.volume_readiness_receipt
    retention = server.torch_retention_evidence
    dependencies = server.dependency_quarantine_evidence
    code = server.code_materialization_evidence
    if (
        plan is None
        or stage is None
        or volume is None
        or retention is None
        or dependencies is None
        or code is None
    ):
        raise ValueError("Stage evidence was incomplete.")
    activation: dict[str, Any] = {
        "revision": BUNDLE_ACTIVATION_REVISION,
        "profile_id": plan.profile_id,
        "head_commit": plan.head_commit,
        "source_contract_digest": plan.source_contract_digest,
        "bootstrap_source_digest": plan.bootstrap_source_digest,
        "workload_bundle_digest": plan.workload_bundle_digest,
        "workload_bundle_size_bytes": plan.workload_bundle_size_bytes,
        "workload_bundle_path": plan.workload_bundle_path,
        "bundle_stage_receipt_digest": stage["receipt_digest"],
        "volume_readiness_receipt_digest": volume["receipt_digest"],
        "torch_retention_evidence_digest": retention["evidence_digest"],
        "dependency_quarantine_evidence_digest": dependencies["evidence_digest"],
        "dependency_private_tree_digest": dependencies["private_tree_digest"],
        "dependency_lock_digest": dependencies["lock_digest"],
        "code_materialization_evidence_digest": code["evidence_digest"],
        "code_private_tree_digest": code["private_tree_digest"],
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
    }
    activation["activation_digest"] = tagged_sha256(canonical_json(activation))
    return activation


def _plan_record(plan: LiveStagePlan) -> dict[str, Any]:
    record: dict[str, Any] = {
        "revision": BOOTSTRAP_TRANSPORT_REVISION,
        **plan.public_identity(),
    }
    record["plan_digest"] = tagged_sha256(canonical_json(record))
    record["launch_hmac"] = _launch_hmac(record)
    return record


def _state_record(server: BootstrapServer, status: str) -> dict[str, Any]:
    if status not in {"awaiting_stage_activation", "activated"}:
        raise ValueError("Live stage state was not durable.")
    record: dict[str, Any] = {
        "revision": BOOTSTRAP_TRANSPORT_REVISION,
        "status": status,
        "plan_digest": _plan_record(server.plan)["plan_digest"]
        if server.plan is not None
        else None,
        "bundle_stage_receipt_digest": (server.bundle_stage_receipt or {}).get("receipt_digest"),
        "volume_readiness_receipt_digest": (server.volume_readiness_receipt or {}).get(
            "receipt_digest"
        ),
        "torch_retention_evidence_digest": (server.torch_retention_evidence or {}).get(
            "evidence_digest"
        ),
        "dependency_quarantine_evidence_digest": (
            server.dependency_quarantine_evidence or {}
        ).get("evidence_digest"),
        "dependency_private_tree_digest": (
            server.dependency_quarantine_evidence or {}
        ).get("private_tree_digest"),
        "dependency_lock_digest": (server.dependency_quarantine_evidence or {}).get(
            "lock_digest"
        ),
        "code_materialization_evidence_digest": (
            server.code_materialization_evidence or {}
        ).get("evidence_digest"),
        "code_private_tree_digest": (server.code_materialization_evidence or {}).get(
            "private_tree_digest"
        ),
        "activation_digest": (
            (server.activation or {}).get("activation_digest") if status == "activated" else None
        ),
    }
    if any(
        record[key] is None
        for key in (
            "plan_digest",
            "bundle_stage_receipt_digest",
            "volume_readiness_receipt_digest",
            "torch_retention_evidence_digest",
            "dependency_quarantine_evidence_digest",
            "dependency_private_tree_digest",
            "dependency_lock_digest",
            "code_materialization_evidence_digest",
            "code_private_tree_digest",
        )
    ) or (status == "activated" and record["activation_digest"] is None):
        raise ValueError("Live stage state evidence was incomplete.")
    record["state_digest"] = tagged_sha256(canonical_json(record))
    record["launch_hmac"] = _launch_hmac(record)
    return record


def _persist_state(server: BootstrapServer, status: str) -> None:
    descriptor = server.work_directory_descriptor
    if descriptor is None:
        raise ValueError("Live work directory descriptor was unavailable.")
    _write_atomic_at(
        descriptor,
        LIVE_STAGE_STATE_FILENAME,
        canonical_json(_state_record(server, status)) + b"\n",
    )


def _activation_journal(activation: Mapping[str, Any]) -> dict[str, Any]:
    journal: dict[str, Any] = {
        "revision": BUNDLE_ACTIVATION_REVISION,
        "activation": dict(activation),
    }
    journal["launch_hmac"] = _launch_hmac(journal)
    return journal


def _validate_recovered_evidence(
    server: BootstrapServer,
    stage_receipt: dict[str, Any],
    volume_receipt: dict[str, Any],
    retention_evidence: dict[str, Any],
) -> None:
    plan = server.plan
    if plan is None:
        raise ValueError("Recovered stage plan was unavailable.")
    expected_stage = build_bundle_stage_receipt(
        plan,
        workload_file="repository_repair_large_model_eligibility.py",
    )
    stage_keys = set(expected_stage) - {"staged_at", "receipt_digest"}
    if (
        set(stage_receipt) != set(expected_stage)
        or any(stage_receipt.get(key) != expected_stage[key] for key in stage_keys)
        or stage_receipt.get("receipt_digest") != _receipt_digest(stage_receipt)
    ):
        raise ValueError("Recovered bundle stage receipt did not match the plan.")
    if (
        volume_receipt.get("profile_id") != plan.profile_id
        or volume_receipt.get("manifest_digest") != plan.manifest_digest
        or volume_receipt.get("network_volume_id") != plan.network_volume_id
        or volume_receipt.get("network_volume_data_center_id") != plan.network_volume_data_center_id
        or volume_receipt.get("network_volume_size_gb") != plan.network_volume_size_gb
        or volume_receipt.get("dependency_quarantine_evidence")
        != server.dependency_quarantine_evidence
        or volume_receipt.get("dependency_lock_digest")
        != (server.dependency_quarantine_evidence or {}).get("lock_digest")
        or volume_receipt.get("dependency_quarantine_evidence_digest")
        != (server.dependency_quarantine_evidence or {}).get("evidence_digest")
        or volume_receipt.get("dependency_private_tree_digest")
        != (server.dependency_quarantine_evidence or {}).get("private_tree_digest")
        or volume_receipt.get("code_materialization_evidence")
        != server.code_materialization_evidence
        or volume_receipt.get("code_materialization_evidence_digest")
        != (server.code_materialization_evidence or {}).get("evidence_digest")
        or volume_receipt.get("code_private_tree_digest")
        != (server.code_materialization_evidence or {}).get("private_tree_digest")
        or volume_receipt.get("ready") is not True
        or volume_receipt.get("receipt_digest") != _receipt_digest(volume_receipt)
    ):
        raise ValueError("Recovered volume readiness receipt did not match the plan.")
    retention_identity = {
        "profile_id": plan.profile_id,
        "head_commit": plan.head_commit,
        "source_contract_digest": plan.source_contract_digest,
        "workload_bundle_digest": plan.workload_bundle_digest,
        "workload_bundle_size_bytes": plan.workload_bundle_size_bytes,
        "workload_bundle_path": plan.workload_bundle_path,
        "bundle_stage_receipt_digest": stage_receipt["receipt_digest"],
        "bootstrap_source_digest": plan.bootstrap_source_digest,
        "volume_readiness_receipt_digest": volume_receipt["receipt_digest"],
        "dependency_quarantine_evidence_digest": (
            server.dependency_quarantine_evidence or {}
        ).get("evidence_digest"),
        "dependency_private_tree_digest": (
            server.dependency_quarantine_evidence or {}
        ).get("private_tree_digest"),
        "dependency_lock_digest": (server.dependency_quarantine_evidence or {}).get(
            "lock_digest"
        ),
        "code_materialization_evidence_digest": (
            server.code_materialization_evidence or {}
        ).get("evidence_digest"),
        "code_private_tree_digest": (server.code_materialization_evidence or {}).get(
            "private_tree_digest"
        ),
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "status": "passed",
    }
    if any(
        retention_evidence.get(key) != value for key, value in retention_identity.items()
    ) or retention_evidence.get("evidence_digest") != _evidence_digest(retention_evidence):
        raise ValueError("Recovered Torch retention evidence did not match the plan.")


def _apply_activation_environment(
    activation: Mapping[str, Any],
    work_directory: Path,
) -> None:
    os.environ.update(
        {
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256": str(
                activation["source_contract_digest"]
            ),
            "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256": str(activation["bundle_stage_receipt_digest"]),
            "EQUINOX_VOLUME_READINESS_RECEIPT_PATH": str(
                work_directory / VOLUME_READINESS_RECEIPT_FILENAME
            ),
            "EQUINOX_VOLUME_READINESS_RECEIPT_SHA256": str(
                activation["volume_readiness_receipt_digest"]
            ),
            "EQUINOX_TORCH_RETENTION_EVIDENCE_PATH": str(
                work_directory / TORCH_RETENTION_EVIDENCE_FILENAME
            ),
            "EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256": str(
                activation["torch_retention_evidence_digest"]
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH": str(
                work_directory / DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": str(
                activation["dependency_quarantine_evidence_digest"]
            ),
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": str(
                activation["dependency_private_tree_digest"]
            ),
            "EQUINOX_DEPENDENCY_LOCK_SHA256": str(
                activation["dependency_lock_digest"]
            ),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH": str(
                work_directory / CODE_MATERIALIZATION_EVIDENCE_FILENAME
            ),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": str(
                activation["code_materialization_evidence_digest"]
            ),
            "EQUINOX_CODE_PRIVATE_TREE_SHA256": str(
                activation["code_private_tree_digest"]
            ),
            "EQUINOX_LIVE_STAGE_ACTIVATION_SHA256": str(activation["activation_digest"]),
            "EQUINOX_BUNDLE_ACTIVATION_DIGEST": str(activation["activation_digest"]),
            "EQUINOX_BUNDLE_HANDOFF_REVISION": BUNDLE_HANDOFF_REVISION,
        }
    )


def initialize_or_recover_live_stage(server: BootstrapServer) -> None:
    plan = server.plan
    descriptor = server.work_directory_descriptor
    work_directory = server.work_directory
    if plan is None or descriptor is None or work_directory is None:
        raise ValueError("Live stage server was not bound to a safe work directory.")
    if not _directory_identity_matches(descriptor, work_directory):
        raise ValueError("Live work directory identity changed during startup.")
    plan_payload = canonical_json(_plan_record(plan)) + b"\n"
    _install_immutable_at(descriptor, LIVE_STAGE_PLAN_FILENAME, plan_payload)
    state = _read_json_at(
        descriptor,
        LIVE_STAGE_STATE_FILENAME,
        "live stage state",
    )
    stage = _read_json_at(
        descriptor,
        BUNDLE_STAGE_RECEIPT_FILENAME,
        "bundle stage receipt",
    )
    volume = _read_json_at(
        descriptor,
        VOLUME_READINESS_RECEIPT_FILENAME,
        "volume readiness receipt",
    )
    retention = _read_json_at(
        descriptor,
        TORCH_RETENTION_EVIDENCE_FILENAME,
        "Torch retention evidence",
    )
    dependency_evidence = _read_json_at(
        descriptor,
        DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
        "dependency quarantine evidence",
    )
    code_evidence = _read_json_at(
        descriptor,
        CODE_MATERIALIZATION_EVIDENCE_FILENAME,
        "code materialization evidence",
    )
    activation_journal = _read_json_at(
        descriptor,
        LIVE_STAGE_ACTIVATION_FILENAME,
        "live stage activation",
    )
    if state is None:
        if any(
            value is not None
            for value in (
                stage,
                volume,
                retention,
                dependency_evidence,
                code_evidence,
                activation_journal,
            )
        ):
            raise ValueError("Unjournaled live stage evidence was present.")
        _require_exact_stage_entries(
            descriptor,
            expected={LIVE_STAGE_PLAN_FILENAME},
        )
        plan.validate(require_open_deadline=True)
        return
    if (
        stage is None
        or volume is None
        or retention is None
        or dependency_evidence is None
        or code_evidence is None
    ):
        raise ValueError("Journaled live stage evidence was incomplete.")
    install_volume_bundle(
        Path(plan.workload_bundle_path),
        work_directory=work_directory,
        workload_file="repository_repair_large_model_eligibility.py",
        expected_digest=plan.workload_bundle_digest,
        expected_size_bytes=plan.workload_bundle_size_bytes,
        work_directory_descriptor=descriptor,
    )
    server.bundle_stage_receipt = stage
    prepare_private_materialization(
        server,
        work_directory=work_directory,
        recovered_dependency_evidence=dependency_evidence,
        recovered_code_evidence=code_evidence,
        enforce_deadline=False,
    )
    if (
        server.dependency_quarantine_evidence != dependency_evidence
        or server.code_materialization_evidence != code_evidence
    ):
        raise ValueError("Recovered private materialization evidence changed.")
    server.volume_readiness_receipt = volume
    server.torch_retention_evidence = retention
    server.bundle_ready = True
    _validate_recovered_evidence(server, stage, volume, retention)
    status = state.get("status")
    if status == "awaiting_stage_activation":
        if activation_journal is not None:
            raise ValueError("Unjournaled live stage activation was present.")
        _require_exact_stage_entries(
            descriptor,
            expected={
                *expected_bundle_files(
                    "repository_repair_large_model_eligibility.py"
                ),
                LIVE_STAGE_PLAN_FILENAME,
                LIVE_STAGE_STATE_FILENAME,
                BUNDLE_STAGE_RECEIPT_FILENAME,
                VOLUME_READINESS_RECEIPT_FILENAME,
                TORCH_RETENTION_EVIDENCE_FILENAME,
                DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
                CODE_MATERIALIZATION_EVIDENCE_FILENAME,
            },
        )
        server.state = status
        if state != _state_record(server, status):
            raise ValueError("Recovered live stage state did not match its evidence.")
        plan.validate(require_open_deadline=True)
        return
    if status != "activated" or activation_journal is None:
        raise ValueError("Recovered live stage status was invalid.")
    raw_activation = activation_journal.get("activation")
    if not isinstance(raw_activation, dict) or activation_journal != _activation_journal(
        raw_activation
    ):
        raise ValueError("Recovered live stage activation was not launch-bound.")
    activation = raw_activation
    server.activation = activation
    if activation != expected_activation(server):
        raise ValueError("Recovered live stage activation did not match its evidence.")
    server.bundle_activated = True
    server.state = "activated"
    if state != _state_record(server, "activated"):
        raise ValueError("Recovered activated state did not match its evidence.")
    _apply_activation_environment(activation, work_directory)


def _deadline_is_open(server: BootstrapServer) -> bool:
    return server.plan is not None and time.time() < server.plan.readiness_deadline_epoch


def _expire_live_stage(server: BootstrapServer) -> None:
    with server.state_lock:
        if server.bundle_activated:
            return
        server.state = "readiness_deadline_expired"
    server.shutdown()


def arm_live_stage_deadline(server: BootstrapServer) -> None:
    if server.plan is None:
        raise ValueError("Live stage deadline had no plan.")
    remaining = server.plan.readiness_deadline_epoch - time.time()
    if remaining <= 0:
        raise ValueError("Live stage readiness deadline has expired.")
    timer = Timer(remaining, _expire_live_stage, args=(server,))
    timer.daemon = True
    server.deadline_timer = timer
    timer.start()


class BootstrapHandler(BaseHTTPRequestHandler):
    server: BootstrapServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        expected = f"Bearer {os.environ['EQUINOX_RESULT_TOKEN']}"
        observed = self.headers.get("Authorization", "")
        return hmac.compare_digest(observed, expected)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = canonical_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _health(self) -> dict[str, object]:
        plan = self.server.plan
        payload: dict[str, object] = {
            "revision": BOOTSTRAP_TRANSPORT_REVISION,
            "status": self.server.state,
        }
        if plan is not None:
            payload.update(plan.public_identity())
        payload.update(
            {
                "bundle_stage_receipt_digest": (self.server.bundle_stage_receipt or {}).get(
                    "receipt_digest"
                ),
                "volume_readiness_receipt_digest": (self.server.volume_readiness_receipt or {}).get(
                    "receipt_digest"
                ),
                "torch_retention_evidence_digest": (self.server.torch_retention_evidence or {}).get(
                    "evidence_digest"
                ),
                "dependency_quarantine_evidence_digest": (
                    self.server.dependency_quarantine_evidence or {}
                ).get("evidence_digest"),
                "dependency_private_tree_digest": (
                    self.server.dependency_quarantine_evidence or {}
                ).get("private_tree_digest"),
                "dependency_lock_digest": (
                    self.server.dependency_quarantine_evidence or {}
                ).get("lock_digest"),
                "code_materialization_evidence_digest": (
                    self.server.code_materialization_evidence or {}
                ).get("evidence_digest"),
                "code_private_tree_digest": (
                    self.server.code_materialization_evidence or {}
                ).get("private_tree_digest"),
                "activation_digest": (self.server.activation or {}).get("activation_digest"),
            }
        )
        return payload

    def _content_length(self, *, maximum: int, exact: int | None = None) -> int | None:
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1 or self.headers.get("Transfer-Encoding") is not None:
            return None
        raw = values[0]
        if not re.fullmatch(r"[1-9][0-9]*", raw):
            return None
        value = int(raw)
        if value > maximum or (exact is not None and value != exact):
            return None
        return value

    def _read_body(self, content_length: int) -> bytes | None:
        self.connection.settimeout(BUNDLE_READ_TIMEOUT_SECONDS)
        try:
            body = self.rfile.read(content_length)
        except (OSError, TimeoutError):
            return None
        if len(body) != content_length:
            return None
        return body

    def do_GET(self) -> None:
        endpoints = {
            f"/{BUNDLE_STAGE_RECEIPT_FILENAME}": self.server.bundle_stage_receipt,
            f"/{VOLUME_READINESS_RECEIPT_FILENAME}": self.server.volume_readiness_receipt,
            f"/{TORCH_RETENTION_EVIDENCE_FILENAME}": self.server.torch_retention_evidence,
            f"/{DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME}": (
                self.server.dependency_quarantine_evidence
            ),
            f"/{CODE_MATERIALIZATION_EVIDENCE_FILENAME}": (
                self.server.code_materialization_evidence
            ),
        }
        if self.path != "/bootstrap-health" and self.path not in endpoints:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
            return
        if self.path == "/bootstrap-health":
            self._write_json(HTTPStatus.OK, self._health())
            return
        evidence = endpoints[self.path]
        if evidence is None:
            self._write_json(
                HTTPStatus.CONFLICT,
                {"error": "EVIDENCE_NOT_READY", **self._health()},
            )
            return
        self._write_json(HTTPStatus.OK, evidence)

    def do_POST(self) -> None:
        if self.path not in {"/bundle", "/activate-staged-bundle"}:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
            return
        if not _deadline_is_open(self.server):
            self._write_json(
                HTTPStatus.GONE,
                {"error": "READINESS_DEADLINE_EXPIRED", **self._health()},
            )
            return
        if self.path == "/activate-staged-bundle":
            self._activate()
            return
        self._upload_bundle()

    def _upload_bundle(self) -> None:
        plan = self.server.plan
        if plan is None:
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "LIVE_STAGE_NOT_CONFIGURED"},
            )
            return
        if self.headers.get("Content-Type") != BUNDLE_CONTENT_TYPE:
            self._write_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "INVALID_BUNDLE_CONTENT_TYPE"},
            )
            return
        content_length = self._content_length(
            maximum=MAXIMUM_BUNDLE_BYTES,
            exact=plan.workload_bundle_size_bytes,
        )
        if content_length is None:
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "INVALID_BUNDLE_SIZE"},
            )
            return
        payload = self._read_body(content_length)
        if payload is None:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "INCOMPLETE_BUNDLE"},
            )
            return
        if not hmac.compare_digest(tagged_sha256(payload), plan.workload_bundle_digest):
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "BUNDLE_DIGEST_MISMATCH"},
            )
            return
        with self.server.state_lock:
            if self.server.state in {"awaiting_stage_activation", "activated"}:
                self._write_json(HTTPStatus.ACCEPTED, self._health())
                return
            if self.server.state != "awaiting_bundle":
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "BUNDLE_UPLOAD_NOT_AVAILABLE", **self._health()},
                )
                return
            self.server.state = "preparing_evidence"
        try:
            stage_receipt = stage_uploaded_bundle(
                payload,
                work_directory=Path(os.environ["EQUINOX_REMOTE_WORKDIR"]),
                workload_file=os.environ["EQUINOX_WORKLOAD_FILE"],
                plan=plan,
                work_directory_descriptor=self.server.work_directory_descriptor,
            )
            with self.server.state_lock:
                self.server.bundle_stage_receipt = stage_receipt
                self.server.bundle_ready = True
            prepare_private_materialization(
                self.server,
                work_directory=Path(os.environ["EQUINOX_REMOTE_WORKDIR"]),
            )
            volume_receipt, retention_evidence = prepare_stage_evidence(
                self.server,
                work_directory=Path(os.environ["EQUINOX_REMOTE_WORKDIR"]),
            )
        except (OSError, subprocess.SubprocessError, tarfile.TarError, ValueError):
            with self.server.state_lock:
                self.server.state = "stage_failed"
            self._write_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "STAGE_PREPARATION_FAILED", **self._health()},
            )
            return
        with self.server.state_lock:
            self.server.volume_readiness_receipt = volume_receipt
            self.server.torch_retention_evidence = retention_evidence
            if not _deadline_is_open(self.server):
                self.server.state = "readiness_deadline_expired"
                self._write_json(
                    HTTPStatus.GONE,
                    {"error": "READINESS_DEADLINE_EXPIRED", **self._health()},
                )
                return
            try:
                _persist_state(self.server, "awaiting_stage_activation")
            except (OSError, ValueError):
                self.server.state = "stage_failed"
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "STAGE_STATE_PERSISTENCE_FAILED", **self._health()},
                )
                return
            self.server.state = "awaiting_stage_activation"
        self._write_json(HTTPStatus.ACCEPTED, self._health())

    def _activate(self) -> None:
        if self.headers.get("Content-Type") != "application/json":
            self._write_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "INVALID_ACTIVATION_CONTENT_TYPE"},
            )
            return
        content_length = self._content_length(maximum=MAXIMUM_ACTIVATION_BYTES)
        if content_length is None:
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "INVALID_ACTIVATION_SIZE"},
            )
            return
        body = self._read_body(content_length)
        if body is None:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "INCOMPLETE_ACTIVATION"},
            )
            return
        try:
            activation = json.loads(body)
        except json.JSONDecodeError:
            activation = None
        if (
            not isinstance(activation, dict)
            or body != canonical_json(activation)
            or set(activation)
            != {
                "revision",
                "profile_id",
                "head_commit",
                "source_contract_digest",
                "bootstrap_source_digest",
                "workload_bundle_digest",
                "workload_bundle_size_bytes",
                "workload_bundle_path",
                "bundle_stage_receipt_digest",
                "volume_readiness_receipt_digest",
                "torch_retention_evidence_digest",
                "dependency_quarantine_evidence_digest",
                "dependency_private_tree_digest",
                "dependency_lock_digest",
                "code_materialization_evidence_digest",
                "code_private_tree_digest",
                "network_volume_id",
                "network_volume_data_center_id",
                "network_volume_size_gb",
                "activation_digest",
            }
        ):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "INVALID_ACTIVATION"},
            )
            return
        with self.server.state_lock:
            if self.server.activation is not None:
                if activation == self.server.activation:
                    self._write_json(HTTPStatus.OK, self._health())
                else:
                    self._write_json(
                        HTTPStatus.CONFLICT,
                        {"error": "ACTIVATION_CONFLICT"},
                    )
                return
            if self.server.state != "awaiting_stage_activation":
                self._write_json(
                    HTTPStatus.CONFLICT,
                    {"error": "STAGE_NOT_READY", **self._health()},
                )
                return
            if not _deadline_is_open(self.server):
                self.server.state = "readiness_deadline_expired"
                self._write_json(
                    HTTPStatus.GONE,
                    {"error": "READINESS_DEADLINE_EXPIRED", **self._health()},
                )
                return
            try:
                expected = expected_activation(self.server)
            except ValueError:
                expected = {}
            if activation != expected:
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "ACTIVATION_IDENTITY_MISMATCH"},
                )
                return
            work_directory = Path(os.environ["EQUINOX_REMOTE_WORKDIR"])
            descriptor = self.server.work_directory_descriptor
            if descriptor is None:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "SAFE_WORK_DIRECTORY_UNAVAILABLE"},
                )
                return
            try:
                _install_immutable_at(
                    descriptor,
                    LIVE_STAGE_ACTIVATION_FILENAME,
                    canonical_json(_activation_journal(activation)) + b"\n",
                )
                self.server.activation = activation
                _persist_state(self.server, "activated")
            except (OSError, ValueError):
                self.server.activation = None
                self.server.state = "activation_failed"
                self._write_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "ACTIVATION_PERSISTENCE_FAILED", **self._health()},
                )
                return
            self.server.bundle_activated = True
            self.server.state = "activated"
            _apply_activation_environment(activation, work_directory)
            if self.server.deadline_timer is not None:
                self.server.deadline_timer.cancel()
        self._write_json(HTTPStatus.ACCEPTED, self._health())
        self.server.shutdown()


def _verify_private_execution_trees(
    *,
    dependency_descriptor: int,
    code_descriptor: int,
    dependency_evidence: Mapping[str, Any],
    code_evidence: Mapping[str, Any],
) -> None:
    if (
        dependency_evidence.get("evidence_digest")
        != _evidence_digest(dependency_evidence)
        or code_evidence.get("evidence_digest") != _evidence_digest(code_evidence)
    ):
        raise ValueError("Private execution evidence digest was invalid.")
    if not _directory_identity_matches(
        dependency_descriptor,
        Path(str(dependency_evidence["private_root"])),
    ) or not _directory_identity_matches(
        code_descriptor,
        Path(str(code_evidence["private_root"])),
    ):
        raise ValueError("Private execution root identity changed.")
    code_entries = _scan_regular_tree(
        code_descriptor,
        maximum_files=len(code_evidence["files"]),
        maximum_bytes=MAXIMUM_EXPANDED_BUNDLE_BYTES,
    )
    _verify_hardened_tree(
        code_descriptor,
        executable_files=frozenset(
            str(entry["path"])
            for entry in code_evidence["files"]
            if str(entry["path"]).endswith(".sh")
        ),
    )
    if (
        code_entries != code_evidence["files"]
        or _tree_digest(code_entries) != code_evidence["private_tree_digest"]
    ):
        raise ValueError("Private code tree changed before execution.")
    lock_entry = next(
        (
            entry
            for entry in code_entries
            if entry["path"] == DEPENDENCY_LOCK_FILENAME
        ),
        None,
    )
    if lock_entry is None:
        raise ValueError("Private code tree did not contain the dependency lock.")
    lock_payload = _read_relative_regular(
        code_descriptor,
        DEPENDENCY_LOCK_FILENAME,
        expected_entry=lock_entry,
        maximum_bytes=DEPENDENCY_LOCK_SIZE_BYTES,
    )
    locked_versions = _locked_requirement_versions(lock_payload)
    dependency_entries = _scan_regular_tree(
        dependency_descriptor,
        maximum_files=MAXIMUM_DEPENDENCY_FILES,
        maximum_bytes=MAXIMUM_DEPENDENCY_BYTES,
    )
    _verify_hardened_tree(dependency_descriptor)
    distributions, record_closure_digest = _dependency_record_closure(
        dependency_descriptor,
        dependency_entries,
        expected_versions=locked_versions,
    )
    if (
        _tree_digest(dependency_entries) != dependency_evidence["private_tree_digest"]
        or record_closure_digest != dependency_evidence["record_closure_digest"]
        or distributions != dependency_evidence["distributions"]
        or len(dependency_entries) != dependency_evidence["installed_file_count"]
        or sum(int(entry["size_bytes"]) for entry in dependency_entries)
        != dependency_evidence["installed_bytes"]
    ):
        raise ValueError("Private dependency tree changed before execution.")


def _apply_private_execution_environment(
    directory_descriptor: int,
    *,
    dependency_descriptor: int,
    code_descriptor: int,
    runtime_root: PrivateRuntimeRoot,
    anchor_root: PrivateAnchorRoot,
    dependency_evidence: Mapping[str, Any],
    code_evidence: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    runtime_metadata = os.fstat(runtime_root.descriptor)
    if (
        not stat.S_ISDIR(runtime_metadata.st_mode)
        or stat.S_IMODE(runtime_metadata.st_mode) != 0o700
        or runtime_metadata.st_uid != os.geteuid()
        or not _directory_identity_matches(runtime_root.descriptor, runtime_root.path)
    ):
        raise ValueError("Private runtime root changed before execution.")
    runtime_identity = (runtime_metadata.st_dev, runtime_metadata.st_ino)
    for descriptor in (
        directory_descriptor,
        dependency_descriptor,
        code_descriptor,
    ):
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == runtime_identity:
            raise ValueError("Private runtime root aliased another execution root.")
    anchor_metadata = os.fstat(anchor_root.descriptor)
    if (
        not stat.S_ISDIR(anchor_metadata.st_mode)
        or stat.S_IMODE(anchor_metadata.st_mode) != 0o700
        or anchor_metadata.st_uid != os.geteuid()
        or not _directory_identity_matches(anchor_root.descriptor, anchor_root.path)
    ):
        raise ValueError("Private anchor root changed before execution.")
    anchor_identity = (anchor_metadata.st_dev, anchor_metadata.st_ino)
    for descriptor in (
        directory_descriptor,
        dependency_descriptor,
        code_descriptor,
        runtime_root.descriptor,
    ):
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == anchor_identity:
            raise ValueError("Private anchor root aliased another execution root.")
    for descriptor in (
        directory_descriptor,
        dependency_descriptor,
        code_descriptor,
        runtime_root.descriptor,
        anchor_root.descriptor,
    ):
        os.set_inheritable(descriptor, True)
    stable_work_directory = Path(f"/proc/self/fd/{directory_descriptor}")
    stable_dependency_root = Path(f"/proc/self/fd/{dependency_descriptor}")
    stable_code_root = Path(f"/proc/self/fd/{code_descriptor}")
    stable_runtime_root = Path(f"/proc/self/fd/{runtime_root.descriptor}")
    stable_anchor_root = Path(f"/proc/self/fd/{anchor_root.descriptor}")
    os.environ.update(
        {
            "EQUINOX_REMOTE_WORKDIR": str(stable_work_directory),
            "EQUINOX_RUNTIME_ROOT": str(stable_runtime_root),
            "EQUINOX_PRIVATE_RUNTIME_REVISION": PRIVATE_RUNTIME_REVISION,
            "EQUINOX_RUNTIME_ANCHOR_ROOT": str(stable_anchor_root),
            "EQUINOX_PRIVATE_ANCHOR_REVISION": PRIVATE_ANCHOR_REVISION,
            "EQUINOX_LARGER_MODEL_PROFILE_PATH": str(
                stable_code_root / "larger-model-eligibility.json"
            ),
            "EQUINOX_CODE_ROOT": str(stable_code_root),
            "EQUINOX_DEPENDENCY_ROOT": str(stable_dependency_root),
            "PYTHONPATH": os.pathsep.join(
                (str(stable_code_root), str(stable_dependency_root))
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH": str(
                stable_work_directory / DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": str(
                dependency_evidence["evidence_digest"]
            ),
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": str(
                dependency_evidence["private_tree_digest"]
            ),
            "EQUINOX_DEPENDENCY_LOCK_SHA256": str(
                dependency_evidence["lock_digest"]
            ),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH": str(
                stable_work_directory / CODE_MATERIALIZATION_EVIDENCE_FILENAME
            ),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": str(
                code_evidence["evidence_digest"]
            ),
            "EQUINOX_CODE_PRIVATE_TREE_SHA256": str(
                code_evidence["private_tree_digest"]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return (
        stable_work_directory,
        stable_dependency_root,
        stable_code_root,
        stable_runtime_root,
        stable_anchor_root,
    )


def _exec_runner_from_safe_directory(
    directory_descriptor: int,
    *,
    workload_file: str,
    dependency_descriptor: int,
    code_descriptor: int,
    runtime_root: PrivateRuntimeRoot,
    anchor_root: PrivateAnchorRoot,
    dependency_evidence: Mapping[str, Any],
    code_evidence: Mapping[str, Any],
    activation: Mapping[str, Any] | None = None,
) -> None:
    try:
        _verify_private_execution_trees(
            dependency_descriptor=dependency_descriptor,
            code_descriptor=code_descriptor,
            dependency_evidence=dependency_evidence,
            code_evidence=code_evidence,
        )
        stable_work_directory, _stable_dependency_root, stable_code_root, _, _ = (
            _apply_private_execution_environment(
                directory_descriptor,
                dependency_descriptor=dependency_descriptor,
                code_descriptor=code_descriptor,
                runtime_root=runtime_root,
                anchor_root=anchor_root,
                dependency_evidence=dependency_evidence,
                code_evidence=code_evidence,
            )
        )
        if activation is not None:
            _apply_activation_environment(activation, stable_work_directory)
        os.execvpe(
            "bash",
            ["bash", str(stable_code_root / runner_file(workload_file))],
            os.environ,
        )
    finally:
        for descriptor in (
            anchor_root.descriptor,
            runtime_root.descriptor,
            code_descriptor,
            dependency_descriptor,
            directory_descriptor,
        ):
            with suppress(OSError):
                os.close(descriptor)


def main() -> None:
    work_directory = Path(os.environ["EQUINOX_REMOTE_WORKDIR"])
    workload_file = os.environ["EQUINOX_WORKLOAD_FILE"]
    expected_bundle_files(workload_file)
    if not os.environ.get("EQUINOX_RESULT_TOKEN"):
        raise SystemExit("EQUINOX_RESULT_TOKEN is required.")
    live_stage = os.environ.pop("EQUINOX_BUNDLE_LIVE_STAGE", "") == "1"
    if live_stage:
        if workload_file != "repository_repair_large_model_eligibility.py":
            raise SystemExit("Live bundle staging is restricted to the eligibility screen.")
        if os.environ.get("EQUINOX_BUNDLE_B64"):
            raise SystemExit("Live bundle staging cannot use an inline bundle.")
        try:
            plan = LiveStagePlan.from_environment()
        except ValueError as error:
            raise SystemExit(f"Live bundle stage plan is invalid: {error}") from error
        try:
            observed_bootstrap_digest = tagged_sha256(Path(__file__).read_bytes())
        except OSError as error:
            raise SystemExit("Bootstrap source could not be verified.") from error
        if not hmac.compare_digest(
            observed_bootstrap_digest,
            plan.bootstrap_source_digest,
        ):
            raise SystemExit("Bootstrap source digest did not match the stage plan.")
        try:
            work_directory_descriptor = open_live_work_directory(work_directory)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Live work directory was unsafe: {error}") from error
        port = int(os.environ.get("EQUINOX_BOOTSTRAP_PORT", "8000"))
        server = BootstrapServer(
            ("0.0.0.0", port),
            BootstrapHandler,
            plan=plan,
            work_directory=work_directory,
            work_directory_descriptor=work_directory_descriptor,
        )
        try:
            initialize_or_recover_live_stage(server)
            if not server.bundle_activated:
                arm_live_stage_deadline(server)
                server.serve_forever(poll_interval=0.1)
        finally:
            if server.deadline_timer is not None:
                server.deadline_timer.cancel()
            server.server_close()
        if not server.bundle_activated:
            os.close(work_directory_descriptor)
            raise SystemExit("Live workload bundle was not activated.")
        if (
            server.dependency_root_descriptor is None
            or server.code_root_descriptor is None
            or server.dependency_quarantine_evidence is None
            or server.code_materialization_evidence is None
        ):
            os.close(work_directory_descriptor)
            raise SystemExit("Private execution materialization was unavailable.")
        try:
            runtime_root = open_private_runtime_root(
                profile_id=plan.profile_id,
                bundle_digest=plan.workload_bundle_digest,
                work_directory_identity=str(work_directory),
            )
        except (OSError, ValueError) as error:
            os.close(server.code_root_descriptor)
            os.close(server.dependency_root_descriptor)
            os.close(work_directory_descriptor)
            raise SystemExit(
                f"Private runtime root could not be prepared: {error}"
            ) from error
        try:
            anchor_root = open_private_anchor_root(
                profile_id=plan.profile_id,
                bundle_digest=plan.workload_bundle_digest,
                work_directory_identity=str(work_directory),
            )
        except (OSError, ValueError) as error:
            os.close(runtime_root.descriptor)
            os.close(server.code_root_descriptor)
            os.close(server.dependency_root_descriptor)
            os.close(work_directory_descriptor)
            raise SystemExit(
                f"Private anchor root could not be prepared: {error}"
            ) from error
        time.sleep(0.5)
        _exec_runner_from_safe_directory(
            work_directory_descriptor,
            workload_file=workload_file,
            dependency_descriptor=server.dependency_root_descriptor,
            code_descriptor=server.code_root_descriptor,
            runtime_root=runtime_root,
            anchor_root=anchor_root,
            dependency_evidence=server.dependency_quarantine_evidence,
            code_evidence=server.code_materialization_evidence,
            activation=server.activation,
        )
        return
    if workload_file == "repository_repair_large_model_eligibility.py":
        raise SystemExit("The larger-model eligibility screen requires live bundle staging.")
    pilot_work_directory_descriptor: int | None = None
    pilot_dependencies: QuarantinedTree | None = None
    pilot_code: QuarantinedTree | None = None
    if workload_file == "repository_repair_large_model_pilot.py":
        try:
            pilot_work_directory_descriptor = open_live_work_directory(work_directory)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Larger-model pilot work directory was unsafe: {error}") from error
    else:
        work_directory.mkdir(parents=True, exist_ok=True)
    encoded_environment_bundle = os.environ.pop("EQUINOX_BUNDLE_B64", "")
    volume_bundle_path = os.environ.pop("EQUINOX_BUNDLE_VOLUME_PATH", "")
    expected_environment_bundle_digest = os.environ.pop(
        "EQUINOX_BUNDLE_SHA256",
        "",
    )
    expected_environment_bundle_size = os.environ.pop(
        "EQUINOX_BUNDLE_SIZE_BYTES",
        "",
    )
    if encoded_environment_bundle and volume_bundle_path:
        raise SystemExit("Only one workload bundle transport may be configured.")
    if encoded_environment_bundle:
        if pilot_work_directory_descriptor is not None:
            raise SystemExit("The larger-model pilot requires its exact volume bundle.")
        if expected_environment_bundle_size:
            raise SystemExit("Inline bundles must not declare EQUINOX_BUNDLE_SIZE_BYTES.")
        if (
            len(expected_environment_bundle_digest) != len("sha256:") + 64
            or not expected_environment_bundle_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in expected_environment_bundle_digest.removeprefix("sha256:")
            )
        ):
            raise SystemExit("EQUINOX_BUNDLE_SHA256 is invalid.")
        install_environment_bundle(
            encoded_environment_bundle,
            work_directory=work_directory,
            workload_file=workload_file,
            expected_digest=expected_environment_bundle_digest,
        )
    elif volume_bundle_path:
        try:
            expected_size_bytes = int(expected_environment_bundle_size)
        except ValueError as error:
            raise SystemExit("EQUINOX_BUNDLE_SIZE_BYTES is invalid.") from error
        try:
            install_volume_bundle(
                Path(volume_bundle_path),
                work_directory=work_directory,
                workload_file=workload_file,
                expected_digest=expected_environment_bundle_digest,
                expected_size_bytes=expected_size_bytes,
                work_directory_descriptor=pilot_work_directory_descriptor,
            )
        except (OSError, tarfile.TarError, ValueError) as error:
            raise SystemExit(f"Volume bundle could not be installed: {error}") from error
        os.environ.update(
            {
                "EQUINOX_BUNDLE_VOLUME_PATH": volume_bundle_path,
                "EQUINOX_BUNDLE_SHA256": expected_environment_bundle_digest,
                "EQUINOX_BUNDLE_SIZE_BYTES": expected_environment_bundle_size,
            }
        )
    else:
        if expected_environment_bundle_digest or expected_environment_bundle_size:
            raise SystemExit(
                "Bundle identity requires EQUINOX_BUNDLE_B64 or EQUINOX_BUNDLE_VOLUME_PATH."
            )
        raise SystemExit("No workload bundle transport was configured.")
    time.sleep(0.5)
    if pilot_work_directory_descriptor is not None:
        try:
            pilot_dependencies, pilot_code = prepare_pilot_private_materialization(
                work_directory_descriptor=pilot_work_directory_descriptor,
                workload_file=workload_file,
                deadline_epoch=pilot_private_materialization_deadline(),
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            os.close(pilot_work_directory_descriptor)
            raise SystemExit(
                f"Pilot private materialization failed: {error}"
            ) from error
        try:
            pilot_runtime_root = open_private_runtime_root(
                profile_id=str(pilot_code.evidence["profile_id"]),
                bundle_digest=str(pilot_code.evidence["bundle_digest"]),
                work_directory_identity=str(work_directory),
            )
        except (OSError, ValueError) as error:
            os.close(pilot_code.private_root_descriptor)
            os.close(pilot_dependencies.private_root_descriptor)
            os.close(pilot_work_directory_descriptor)
            raise SystemExit(
                f"Pilot private runtime root could not be prepared: {error}"
            ) from error
        try:
            pilot_anchor_root = open_private_anchor_root(
                profile_id=str(pilot_code.evidence["profile_id"]),
                bundle_digest=str(pilot_code.evidence["bundle_digest"]),
                work_directory_identity=str(work_directory),
            )
        except (OSError, ValueError) as error:
            os.close(pilot_runtime_root.descriptor)
            os.close(pilot_code.private_root_descriptor)
            os.close(pilot_dependencies.private_root_descriptor)
            os.close(pilot_work_directory_descriptor)
            raise SystemExit(
                f"Pilot private anchor root could not be prepared: {error}"
            ) from error
        _exec_runner_from_safe_directory(
            pilot_work_directory_descriptor,
            workload_file=workload_file,
            dependency_descriptor=pilot_dependencies.private_root_descriptor,
            code_descriptor=pilot_code.private_root_descriptor,
            runtime_root=pilot_runtime_root,
            anchor_root=pilot_anchor_root,
            dependency_evidence=pilot_dependencies.evidence,
            code_evidence=pilot_code.evidence,
        )
        return
    os.execvpe(
        "bash",
        ["bash", str(work_directory / runner_file(workload_file))],
        os.environ,
    )


if __name__ == "__main__":
    main()
