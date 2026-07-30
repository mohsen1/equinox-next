"""Crash-safe publication of one complete typed RunPod artifact set.

The commit manifest is the only supported visibility boundary.  Callers must
resolve artifacts through :func:`load_committed_artifact_set`; files in the
hidden generation directory are not committed unless that manifest exists and
verifies.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from packages.equinox_core.canonical import CanonicalizationError, canonical_bytes
from research.runpod.larger_model_gate import GateError, canonical_json, result_digest

SCHEMA_VERSION = 2
ATOMIC_PUBLICATION_PROFILE_ID = "qwen2.5-coder-7b-runpod-h100@10"
MANIFEST_KIND = "equinox.runpod-artifact-set"
JOURNAL_KIND = "equinox.runpod-artifact-publication-journal"
PUBLICATION_TYPES = ("screen", "pilot")
SCREEN_ARTIFACT_ROLES = (
    "readiness_receipt",
    "bundle_receipt",
    "torch_evidence",
    "run_result",
    "provider_receipt",
    "attempt_record",
)
PILOT_ARTIFACT_ROLES = (
    "run_result",
    "provider_receipt",
    "attempt_record",
    "model_artifact",
)
ARTIFACT_ROLES_BY_PUBLICATION_TYPE = {
    "screen": SCREEN_ARTIFACT_ROLES,
    "pilot": PILOT_ARTIFACT_ROLES,
}
JSON_ARTIFACT_ROLES = frozenset(
    {
        "readiness_receipt",
        "bundle_receipt",
        "torch_evidence",
        "run_result",
        "provider_receipt",
        "attempt_record",
    }
)
ARCHIVABLE_PRIOR_ROLES = ("readiness_receipt", "bundle_receipt")
ARTIFACT_FILENAMES = {
    "readiness_receipt": "readiness-receipt.json",
    "bundle_receipt": "bundle-receipt.json",
    "torch_evidence": "torch-evidence.json",
    "run_result": "run-result.json",
    "provider_receipt": "provider-receipt.json",
    "attempt_record": "attempt-record.json",
    "model_artifact": "model-artifact.tgz",
}
ARTIFACT_MEDIA_TYPES = {
    **{role: "application/json" for role in JSON_ARTIFACT_ROLES},
    "model_artifact": "application/vnd.equinox.lora-adapter+gzip",
}
MAXIMUM_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024
MAXIMUM_MODEL_ARTIFACT_BYTES = 512 * 1024 * 1024
MAXIMUM_MODEL_MANIFEST_BYTES = 1024 * 1024
MAXIMUM_MODEL_ARCHIVE_MEMBERS = 4096
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_TAGGED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNTAGGED_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_SUFFIX = ".artifact-set.json"
_GENERATION_DIRECTORY = ".artifact-generations"
_STATE_DIRECTORY = "artifact-publications"
_PROVIDER_RECEIPT_KEYS = frozenset(
    {
        "provider_name",
        "provider_handle",
        "provider_cli_version",
        "resource_profile",
        "workload",
        "result",
        "started_at",
        "completed_at",
        "teardown_confirmed",
    }
)
_WORKLOAD_RESULT_BINDINGS = (
    ("id", "workload"),
    ("revision", "workload_revision"),
    ("algorithm", "algorithm"),
    ("static_branch_width", "branch_width"),
    ("complexity_strategy", "complexity_strategy"),
    ("scale_case_count", "test_case_count"),
    ("maximum_horizon", "maximum_horizon"),
    ("total_action_decisions", "total_action_decisions"),
    ("model_id", "model_id"),
    ("model_revision", "model_revision"),
    ("task_domains", "task_domains"),
    ("maximum_complexity_level", "maximum_complexity_level"),
    ("reached_complexity_level", "reached_complexity_level"),
    ("total_sampled_completions", "total_sampled_completions"),
    ("total_sampled_actions", "total_sampled_actions"),
    ("multi_step", "multi_step"),
    ("restored_continuations", "restored_continuations"),
    ("snapshot_fidelity", "snapshot_fidelity"),
    ("replay_enabled", "replay_enabled"),
)
_ATTEMPT_RESOURCE_BINDINGS = (
    ("profile_id", "profile_id"),
    ("source_head_commit", "source_head_commit"),
    ("source_contract_digest", "source_contract_digest"),
    ("bootstrap_source_digest", "bootstrap_source_digest"),
    ("workload_bundle_digest", "workload_bundle_digest"),
    ("bundle_stage_receipt_digest", "bundle_stage_receipt_digest"),
    ("volume_readiness_receipt_digest", "volume_readiness_receipt_digest"),
    ("torch_retention_evidence_digest", "torch_retention_evidence_digest"),
    ("dependency_quarantine_revision", "dependency_quarantine_revision"),
    ("dependency_lock_digest", "dependency_lock_digest"),
    (
        "dependency_quarantine_evidence_digest",
        "dependency_quarantine_evidence_digest",
    ),
    ("dependency_private_tree_digest", "dependency_private_tree_digest"),
    ("code_materialization_revision", "code_materialization_revision"),
    (
        "code_materialization_evidence_digest",
        "code_materialization_evidence_digest",
    ),
    ("code_private_tree_digest", "code_private_tree_digest"),
    ("activation_digest", "live_stage_activation_digest"),
    ("network_volume_deletion_required", "network_volume_deletion_required"),
    ("network_volume_deletion_deadline", "network_volume_deletion_deadline"),
    ("network_volume_deletion_attempted", "network_volume_deletion_attempted"),
    ("network_volume_deletion_confirmed", "network_volume_deletion_confirmed"),
    ("network_volume_deletion_deadline_met", "network_volume_deletion_deadline_met"),
    ("network_volume_deleted_at", "network_volume_deleted_at"),
)
_PILOT_RESULT_RESOURCE_BINDINGS = (
    ("source_contract_digest", "source_contract_digest"),
    ("dependency_quarantine_revision", "dependency_quarantine_revision"),
    ("dependency_lock_digest", "dependency_lock_digest"),
    (
        "dependency_quarantine_evidence_digest",
        "dependency_quarantine_evidence_digest",
    ),
    ("dependency_private_tree_digest", "dependency_private_tree_digest"),
    ("code_materialization_revision", "code_materialization_revision"),
    (
        "code_materialization_evidence_digest",
        "code_materialization_evidence_digest",
    ),
    ("code_private_tree_digest", "code_private_tree_digest"),
    ("network_volume_id", "network_volume_id"),
)


class ArtifactPublicationError(RuntimeError):
    """A publication contract or durable-state check failed closed."""


class ArtifactPublicationBusy(ArtifactPublicationError):
    """Another live process owns the publication lock."""


@dataclass(frozen=True)
class PublicationRequest:
    """All inputs required to publish one complete post-teardown result."""

    publication_id: str
    publication_type: str
    profile_id: str
    proof_directory: Path
    state_directory: Path
    artifacts: Mapping[str, Path]
    teardown_confirmed: bool
    source_screen_publication_id: str | None = None
    source_screen_manifest_digest: str | None = None
    prior_profile_id: str | None = None
    prior_receipts: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicationResult:
    """A verified committed artifact set."""

    publication_id: str
    publication_type: str
    profile_id: str
    manifest_path: Path
    generation_directory: Path
    artifact_paths: Mapping[str, Path]
    artifact_descriptors: Mapping[str, Mapping[str, Any]]
    archive_paths: Mapping[str, Path]
    set_digest: str
    artifact_set_manifest_digest: str
    source_screen_publication_id: str | None = None
    source_screen_manifest_digest: str | None = None

    @property
    def manifest_digest(self) -> str:
        """Compatibility spelling for callers that do not emit observer evidence."""

        return self.artifact_set_manifest_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "proof_id": self.publication_id,
            "publication_type": self.publication_type,
            "profile_id": self.profile_id,
            "committed": True,
            "artifact_set_committed": True,
            "manifest_path": str(self.manifest_path),
            "generation_directory": str(self.generation_directory),
            "set_digest": self.set_digest,
            "artifact_set_manifest_digest": self.artifact_set_manifest_digest,
            "artifacts": {role: str(path) for role, path in self.artifact_paths.items()},
            "artifact_descriptors": {
                role: dict(descriptor) for role, descriptor in self.artifact_descriptors.items()
            },
            "source_screen": (
                {
                    "publication_id": self.source_screen_publication_id,
                    "artifact_set_manifest_digest": self.source_screen_manifest_digest,
                }
                if self.source_screen_publication_id is not None
                else None
            ),
            "prior_profile_archives": {
                role: str(path) for role, path in sorted(self.archive_paths.items())
            },
        }


FaultInjector = Callable[[str], None]


@dataclass(frozen=True)
class _StagedArtifact:
    role: str
    path: Path
    descriptor: Mapping[str, Any]
    json_payload: bytes | None


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _publication_canonical_bytes(value: Any, name: str) -> bytes:
    try:
        return canonical_bytes(value)
    except CanonicalizationError as error:
        raise ArtifactPublicationError(f"{name} cannot be represented as canonical JSON") from error


def _safe_component(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ArtifactPublicationError(f"{name} is not a safe identifier")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_trusted_mode(info: os.stat_result, name: str) -> None:
    if info.st_uid != os.geteuid():
        raise ArtifactPublicationError(f"{name} is not owned by the current account")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ArtifactPublicationError(f"{name} is group- or world-writable")


def _validate_trusted_directory(path: Path, name: str) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactPublicationError(f"{name} is unavailable or unsafe") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ArtifactPublicationError(f"{name} is not a directory")
        _validate_trusted_mode(info, name)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    path = _absolute(path)
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise ArtifactPublicationError(f"directory path is unsafe: {cursor}")
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise ArtifactPublicationError(f"directory path has no existing parent: {path}")
        cursor = parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ArtifactPublicationError(f"directory path is unsafe: {cursor}")
    _validate_trusted_directory(cursor, f"directory path {cursor}")
    for directory in reversed(missing):
        with contextlib.suppress(FileExistsError):
            directory.mkdir(mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactPublicationError(f"directory path is unsafe: {directory}")
        _validate_trusted_directory(directory, f"directory path {directory}")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactPublicationError(f"directory path is unsafe: {path}")
    _validate_trusted_directory(path, f"directory path {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ArtifactPublicationError(f"artifact parent is unsafe: {path.parent}")
    descriptor, pending_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    pending = Path(pending_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise ArtifactPublicationError(f"artifact path is unsafe: {path}")
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _atomic_install_once(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Atomically create an immutable file without ever replacing old evidence."""

    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ArtifactPublicationError(f"artifact parent is unsafe: {path.parent}")
    descriptor, pending_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    pending = Path(pending_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise ArtifactPublicationError(f"artifact path is unsafe: {path}")
        try:
            os.link(pending, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ArtifactPublicationError(
                f"immutable artifact already exists: {path.name}"
            ) from error
        pending.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _atomic_copy_verified(
    path: Path,
    source: Path,
    descriptor: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ArtifactPublicationError(f"artifact parent is unsafe: {path.parent}")
    destination_descriptor, pending_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    pending = Path(pending_name)
    try:
        os.fchmod(destination_descriptor, mode)
        size_bytes, sha256 = _stream_regular_file(
            source,
            f"staged {descriptor['role']}",
            maximum_bytes=_maximum_artifact_bytes(str(descriptor["role"])),
            destination_descriptor=destination_descriptor,
        )
        if size_bytes != descriptor["size_bytes"] or sha256 != descriptor["sha256"]:
            raise ArtifactPublicationError(
                f"staged {descriptor['role']} changed before it was copied"
            )
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = -1
        if path.is_symlink():
            raise ArtifactPublicationError(f"artifact path is unsafe: {path}")
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except BaseException:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        pending.unlink(missing_ok=True)
        raise


def _read_regular_file(path: Path, name: str, *, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ArtifactPublicationError(f"{name} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactPublicationError(f"{name} is unavailable or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactPublicationError(f"{name} is not a regular file")
        _validate_trusted_mode(before, name)
        if before.st_size > maximum_bytes:
            raise ArtifactPublicationError(f"{name} exceeds the publication size limit")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes or os.read(descriptor, 1):
            raise ArtifactPublicationError(f"{name} exceeds the publication size limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ArtifactPublicationError(f"{name} changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def _json_object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactPublicationError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ArtifactPublicationError(f"{name} must contain one JSON object")
    return value


def _artifact_roles(publication_type: str) -> tuple[str, ...]:
    try:
        return ARTIFACT_ROLES_BY_PUBLICATION_TYPE[publication_type]
    except KeyError as error:
        raise ArtifactPublicationError("publication_type must be screen or pilot") from error


def _maximum_artifact_bytes(role: str) -> int:
    return MAXIMUM_MODEL_ARTIFACT_BYTES if role == "model_artifact" else MAXIMUM_JSON_ARTIFACT_BYTES


def _validate_payload_identity(
    payload: bytes,
    *,
    role: str,
    publication_id: str,
    profile_id: str,
) -> None:
    value = _json_object(payload, role)
    observed_profile_id = value.get("profile_id")
    if observed_profile_id is None and role == "provider_receipt":
        resource_profile = value.get("resource_profile")
        if isinstance(resource_profile, dict):
            observed_profile_id = resource_profile.get("profile_id")
    if observed_profile_id != profile_id:
        raise ArtifactPublicationError(f"{role} has a different or missing profile_id")
    if role == "attempt_record" and value.get("proof_id") != publication_id:
        raise ArtifactPublicationError("attempt_record has a different or missing proof_id")
    for key in ("proof_id", "publication_id"):
        observed = value.get(key)
        if role != "attempt_record" and observed is not None and observed != publication_id:
            raise ArtifactPublicationError(f"{role} has a different {key}")
    if (
        role in {"provider_receipt", "attempt_record"}
        and value.get("teardown_confirmed") is not True
    ):
        raise ArtifactPublicationError(f"{role} does not confirm provider teardown")


def _aware_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactPublicationError(f"{name} is not a timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactPublicationError(f"{name} is not a timestamp") from error
    if timestamp.utcoffset() is None:
        raise ArtifactPublicationError(f"{name} must include a timezone")
    return timestamp.astimezone(UTC)


def _validate_provider_receipt_contract(
    provider_receipt: Mapping[str, Any],
    run_result: Mapping[str, Any],
) -> None:
    workload = provider_receipt.get("workload")
    resource_profile = provider_receipt.get("resource_profile")
    if (
        set(provider_receipt) != _PROVIDER_RECEIPT_KEYS
        or provider_receipt.get("provider_name") != "RunPod"
        or not isinstance(provider_receipt.get("provider_handle"), str)
        or re.fullmatch(
            r"runpod://pods/[a-zA-Z0-9_-]+",
            provider_receipt["provider_handle"],
        )
        is None
        or not isinstance(provider_receipt.get("provider_cli_version"), str)
        or not provider_receipt["provider_cli_version"]
        or not isinstance(resource_profile, Mapping)
        or not isinstance(workload, Mapping)
        or set(workload) != {workload_field for workload_field, _ in _WORKLOAD_RESULT_BINDINGS}
        or provider_receipt.get("result") != run_result
        or provider_receipt.get("teardown_confirmed") is not True
    ):
        raise ArtifactPublicationError("provider_receipt is not the exact proof API request")
    for workload_field, result_field in _WORKLOAD_RESULT_BINDINGS:
        if workload.get(workload_field) != run_result.get(result_field):
            raise ArtifactPublicationError(
                f"provider_receipt workload {workload_field} does not match run_result "
                f"{result_field}"
            )
    if (
        not isinstance(workload.get("id"), str)
        or not workload["id"]
        or not isinstance(workload.get("model_id"), str)
        or not workload["model_id"]
        or type(workload.get("static_branch_width")) is not int
        or workload["static_branch_width"] < 1
        or workload.get("complexity_strategy") != "adaptive"
    ):
        raise ArtifactPublicationError(
            "provider_receipt workload has invalid model, branching, or complexity identity"
        )
    started_at = _aware_timestamp(
        provider_receipt.get("started_at"),
        "provider_receipt started_at",
    )
    completed_at = _aware_timestamp(
        provider_receipt.get("completed_at"),
        "provider_receipt completed_at",
    )
    if completed_at < started_at:
        raise ArtifactPublicationError("provider_receipt completed_at precedes started_at")


def _validate_attempt_resource_bindings(
    attempt_record: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
) -> None:
    for attempt_field, resource_field in _ATTEMPT_RESOURCE_BINDINGS:
        if (
            attempt_field not in attempt_record
            or resource_field not in resource_profile
            or attempt_record[attempt_field] != resource_profile[resource_field]
        ):
            raise ArtifactPublicationError(
                f"attempt_record {attempt_field} does not match provider resource_profile "
                f"{resource_field}"
            )


def _validate_pilot_result_resource_bindings(
    run_result: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
) -> None:
    for result_field, resource_field in _PILOT_RESULT_RESOURCE_BINDINGS:
        if (
            result_field not in run_result
            or resource_field not in resource_profile
            or run_result[result_field] != resource_profile[resource_field]
        ):
            raise ArtifactPublicationError(
                f"pilot run_result {result_field} does not match provider resource_profile "
                f"{resource_field}"
            )


def _json_artifact_descriptor(role: str, payload: bytes) -> dict[str, Any]:
    value = _json_object(payload, role)
    canonical_digest = _sha256(_publication_canonical_bytes(value, role))
    return {
        "role": role,
        "media_type": ARTIFACT_MEDIA_TYPES[role],
        "filename": ARTIFACT_FILENAMES[role],
        "sha256": _sha256(payload),
        "canonical_json_sha256": canonical_digest,
        "size_bytes": len(payload),
    }


def _stream_regular_file(
    path: Path,
    name: str,
    *,
    maximum_bytes: int,
    destination_descriptor: int | None = None,
) -> tuple[int, str]:
    if path.is_symlink():
        raise ArtifactPublicationError(f"{name} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactPublicationError(f"{name} is unavailable or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactPublicationError(f"{name} is not a regular file")
        _validate_trusted_mode(before, name)
        if before.st_size > maximum_bytes:
            raise ArtifactPublicationError(f"{name} exceeds the publication size limit")
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > maximum_bytes:
                raise ArtifactPublicationError(f"{name} exceeds the publication size limit")
            digest.update(chunk)
            if destination_descriptor is not None:
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise ArtifactPublicationError(f"{name} could not be copied")
                    view = view[written:]
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ArtifactPublicationError(f"{name} changed while it was read")
        if observed_size != before.st_size:
            raise ArtifactPublicationError(f"{name} size changed while it was read")
        return observed_size, f"sha256:{digest.hexdigest()}"
    finally:
        os.close(descriptor)


def _binary_artifact_descriptor(role: str, path: Path) -> dict[str, Any]:
    size_bytes, sha256 = _stream_regular_file(
        path,
        f"staged {role}",
        maximum_bytes=_maximum_artifact_bytes(role),
    )
    if size_bytes == 0:
        raise ArtifactPublicationError(f"staged {role} must not be empty")
    return {
        "role": role,
        "media_type": ARTIFACT_MEDIA_TYPES[role],
        "filename": ARTIFACT_FILENAMES[role],
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _safe_archive_name(name: str, *, directory: bool) -> str:
    normalized = name[:-1] if directory and name.endswith("/") else name
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0] != "adapter"
        or str(PurePosixPath(normalized)) != normalized
    ):
        raise ArtifactPublicationError("model_artifact archive contains an unsafe member path")
    return normalized


def _read_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    name: str,
    maximum_bytes: int,
) -> bytes:
    if member.size < 0 or member.size > maximum_bytes:
        raise ArtifactPublicationError(f"{name} exceeds the safe extracted size")
    source = archive.extractfile(member)
    if source is None:
        raise ArtifactPublicationError(f"{name} could not be read")
    payload = source.read(maximum_bytes + 1)
    if len(payload) != member.size or len(payload) > maximum_bytes:
        raise ArtifactPublicationError(f"{name} extracted size does not match")
    return payload


def _hash_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    name: str,
    expected_size: int,
) -> str:
    if member.size != expected_size:
        raise ArtifactPublicationError(f"{name} extracted size does not match")
    source = archive.extractfile(member)
    if source is None:
        raise ArtifactPublicationError(f"{name} could not be read")
    digest = hashlib.sha256()
    observed_size = 0
    while chunk := source.read(1024 * 1024):
        observed_size += len(chunk)
        if observed_size > expected_size:
            raise ArtifactPublicationError(f"{name} extracted size does not match")
        digest.update(chunk)
    if observed_size != expected_size:
        raise ArtifactPublicationError(f"{name} extracted size does not match")
    return digest.hexdigest()


def _verify_model_artifact_archive(
    path: Path,
    descriptor: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> None:
    """Verify an opaque tgz without extracting it or following archive links."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactPublicationError("model_artifact is unavailable or unsafe") from error
    try:
        before = os.fstat(file_descriptor)
        _validate_trusted_mode(before, "model_artifact")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != descriptor["size_bytes"]
            or before.st_size < 1
            or before.st_size > MAXIMUM_MODEL_ARTIFACT_BYTES
        ):
            raise ArtifactPublicationError("model_artifact file identity is invalid")
        try:
            with (
                os.fdopen(os.dup(file_descriptor), "rb") as source,
                tarfile.open(fileobj=source, mode="r:gz") as archive,
            ):
                members: dict[str, tarfile.TarInfo] = {}
                regular_bytes = 0
                for member_count, member in enumerate(archive, start=1):
                    if member_count > MAXIMUM_MODEL_ARCHIVE_MEMBERS:
                        raise ArtifactPublicationError(
                            "model_artifact archive member count is invalid"
                        )
                    if not (member.isdir() or member.isfile()) or member.sparse:
                        raise ArtifactPublicationError(
                            "model_artifact archive contains a link or special member"
                        )
                    normalized = _safe_archive_name(
                        member.name,
                        directory=member.isdir(),
                    )
                    if normalized in members:
                        raise ArtifactPublicationError(
                            "model_artifact archive contains duplicate members"
                        )
                    if member.isdir() and member.size != 0:
                        raise ArtifactPublicationError(
                            "model_artifact archive directory is invalid"
                        )
                    if member.isfile():
                        regular_bytes += member.size
                        if regular_bytes > MAXIMUM_MODEL_ARTIFACT_BYTES:
                            raise ArtifactPublicationError(
                                "model_artifact extracted contents exceed the byte ceiling"
                            )
                    members[normalized] = member
                if not members:
                    raise ArtifactPublicationError("model_artifact archive member count is invalid")

                manifest_member = members.get("adapter/adapter-manifest.json")
                if manifest_member is None or not manifest_member.isfile():
                    raise ArtifactPublicationError(
                        "model_artifact archive omits its adapter manifest"
                    )
                manifest_payload = _read_archive_member(
                    archive,
                    manifest_member,
                    name="model_artifact adapter manifest",
                    maximum_bytes=MAXIMUM_MODEL_MANIFEST_BYTES,
                )
                manifest = _json_object(
                    manifest_payload,
                    "model_artifact adapter manifest",
                )
                if manifest != dict(expected_manifest):
                    raise ArtifactPublicationError(
                        "model_artifact adapter manifest differs from run_result"
                    )
                expected_manifest_keys = {
                    "schema_version",
                    "model_id",
                    "model_revision",
                    "workload_revision",
                    "objective_id",
                    "training_configuration",
                    "files",
                    "digest",
                }
                content = {key: value for key, value in manifest.items() if key != "digest"}
                try:
                    observed_manifest_digest = _sha256(canonical_json(content))
                except GateError as error:
                    raise ArtifactPublicationError(
                        "model_artifact adapter manifest is not canonical JSON"
                    ) from error
                if (
                    set(manifest) != expected_manifest_keys
                    or manifest["schema_version"] != 1
                    or not isinstance(manifest["model_id"], str)
                    or not isinstance(manifest["model_revision"], str)
                    or not isinstance(manifest["workload_revision"], str)
                    or not isinstance(manifest["objective_id"], str)
                    or not isinstance(manifest["training_configuration"], dict)
                    or not isinstance(manifest["digest"], str)
                    or not _TAGGED_SHA256.fullmatch(manifest["digest"])
                    or manifest["digest"] != observed_manifest_digest
                ):
                    raise ArtifactPublicationError(
                        "model_artifact adapter manifest identity is invalid"
                    )

                files = manifest["files"]
                if not isinstance(files, list) or not files:
                    raise ArtifactPublicationError("model_artifact adapter manifest has no files")
                declared_paths: set[str] = set()
                expected_directories = {"adapter"}
                declared_bytes = 0
                for item in files:
                    if not isinstance(item, dict) or set(item) != {
                        "path",
                        "size_bytes",
                        "sha256",
                    }:
                        raise ArtifactPublicationError("model_artifact file evidence is invalid")
                    relative_path = item["path"]
                    relative_parts = (
                        relative_path.split("/") if isinstance(relative_path, str) else []
                    )
                    if (
                        not isinstance(relative_path, str)
                        or not relative_path
                        or relative_path.startswith("/")
                        or "\\" in relative_path
                        or any(part in {"", ".", ".."} for part in relative_parts)
                        or str(PurePosixPath(relative_path)) != relative_path
                        or relative_path == "adapter-manifest.json"
                        or relative_path in declared_paths
                        or type(item["size_bytes"]) is not int
                        or item["size_bytes"] < 0
                        or not isinstance(item["sha256"], str)
                        or not _UNTAGGED_SHA256.fullmatch(item["sha256"])
                    ):
                        raise ArtifactPublicationError("model_artifact file evidence is invalid")
                    declared_paths.add(relative_path)
                    declared_bytes += item["size_bytes"]
                    if declared_bytes > MAXIMUM_MODEL_ARTIFACT_BYTES:
                        raise ArtifactPublicationError(
                            "model_artifact declared contents exceed the byte ceiling"
                        )
                    for parent in PurePosixPath(relative_path).parents:
                        if str(parent) != ".":
                            expected_directories.add(f"adapter/{parent}")
                    member = members.get(f"adapter/{relative_path}")
                    if member is None or not member.isfile() or member.size != item["size_bytes"]:
                        raise ArtifactPublicationError(
                            "model_artifact file size evidence does not match"
                        )
                    observed_digest = _hash_archive_member(
                        archive,
                        member,
                        name=f"model_artifact {relative_path}",
                        expected_size=item["size_bytes"],
                    )
                    if observed_digest != item["sha256"]:
                        raise ArtifactPublicationError("model_artifact file digest does not match")

                observed_files = {
                    name.removeprefix("adapter/")
                    for name, member in members.items()
                    if member.isfile()
                }
                observed_directories = {name for name, member in members.items() if member.isdir()}
                if (
                    observed_files != declared_paths | {"adapter-manifest.json"}
                    or observed_directories != expected_directories
                ):
                    raise ArtifactPublicationError(
                        "model_artifact archive and manifest member sets differ"
                    )
                if not {
                    "adapter_config.json",
                    "adapter_model.safetensors",
                }.issubset(declared_paths):
                    raise ArtifactPublicationError(
                        "model_artifact omits required LoRA adapter files"
                    )
        except (tarfile.TarError, EOFError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactPublicationError(
                "model_artifact is not a valid safe gzip tar archive"
            ) from error
        after = os.fstat(file_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ArtifactPublicationError("model_artifact changed while its archive was verified")
    finally:
        os.close(file_descriptor)


def _validate_cross_artifact_bindings(
    *,
    publication_type: str,
    payloads: Mapping[str, bytes],
    model_artifact_path: Path | None = None,
    model_artifact_descriptor: Mapping[str, Any] | None = None,
) -> None:
    run_result = _json_object(payloads["run_result"], "run_result")
    provider_receipt = _json_object(
        payloads["provider_receipt"],
        "provider_receipt",
    )
    attempt_record = _json_object(payloads["attempt_record"], "attempt_record")
    if attempt_record.get("publication_type") != publication_type:
        raise ArtifactPublicationError("attempt_record has a different or missing publication_type")
    if provider_receipt.get("result") != run_result:
        raise ArtifactPublicationError(
            "provider_receipt does not contain the exact committed run_result"
        )
    _validate_provider_receipt_contract(provider_receipt, run_result)
    resource_profile = provider_receipt.get("resource_profile")
    if not isinstance(resource_profile, Mapping):
        raise ArtifactPublicationError("provider_receipt has no resource_profile")
    for provider_field in ("provider_handle", "started_at", "completed_at"):
        provider_value = provider_receipt.get(provider_field)
        if (
            not isinstance(provider_value, str)
            or not provider_value
            or attempt_record.get(provider_field) != provider_value
        ):
            raise ArtifactPublicationError(
                f"attempt_record does not match provider_receipt {provider_field}"
            )
    try:
        expected_result_digest = result_digest(run_result)
    except GateError as error:
        raise ArtifactPublicationError(
            "run_result cannot be represented as canonical JSON"
        ) from error
    provider_result_digest = provider_receipt.get("result_digest")
    if provider_result_digest is not None and provider_result_digest != expected_result_digest:
        raise ArtifactPublicationError("provider_receipt result_digest does not cover run_result")
    if attempt_record.get("result_digest") != expected_result_digest:
        raise ArtifactPublicationError("attempt_record result_digest does not cover run_result")
    if publication_type == "screen":
        _validate_screen_evidence_bindings(
            payloads=payloads,
            run_result=run_result,
            provider_receipt=provider_receipt,
            attempt_record=attempt_record,
        )
        resource_profile = provider_receipt.get("resource_profile")
        if not isinstance(resource_profile, dict):
            raise ArtifactPublicationError(
                "screen provider_receipt has no resource_profile evidence bindings"
            )
        _validate_screen_handoff_identity(
            readiness_receipt=_json_object(
                payloads["readiness_receipt"],
                "readiness_receipt",
            ),
            bundle_receipt=_json_object(
                payloads["bundle_receipt"],
                "bundle_receipt",
            ),
            torch_evidence=_json_object(
                payloads["torch_evidence"],
                "torch_evidence",
            ),
            run_result=run_result,
            resource_profile=resource_profile,
            attempt_record=attempt_record,
        )
    _validate_attempt_resource_bindings(attempt_record, resource_profile)
    if publication_type == "pilot":
        _validate_pilot_result_resource_bindings(run_result, resource_profile)
        adapter_manifest = run_result.get("adapter_manifest")
        if (
            run_result.get("adapter_persisted") is not True
            or not isinstance(adapter_manifest, dict)
            or not isinstance(adapter_manifest.get("digest"), str)
            or not _TAGGED_SHA256.fullmatch(adapter_manifest["digest"])
        ):
            raise ArtifactPublicationError(
                "pilot run_result does not identify a persisted model artifact"
            )
        if model_artifact_path is None or model_artifact_descriptor is None:
            raise ArtifactPublicationError("pilot publication omits its model_artifact binding")
        if (
            attempt_record.get("model_artifact_sha256") != model_artifact_descriptor["sha256"]
            or attempt_record.get("model_artifact_size_bytes")
            != model_artifact_descriptor["size_bytes"]
        ):
            raise ArtifactPublicationError("attempt_record does not bind the exact model_artifact")
        _verify_model_artifact_archive(
            model_artifact_path,
            model_artifact_descriptor,
            adapter_manifest,
        )


def _embedded_evidence_digest(
    evidence: Mapping[str, Any],
    *,
    digest_field: str,
    name: str,
) -> str:
    observed = evidence.get(digest_field)
    if not isinstance(observed, str) or not _TAGGED_SHA256.fullmatch(observed):
        raise ArtifactPublicationError(f"{name} has no valid {digest_field}")
    material = {key: value for key, value in evidence.items() if key != digest_field}
    try:
        expected = _sha256(canonical_json(material))
    except GateError as error:
        raise ArtifactPublicationError(f"{name} cannot be represented as canonical JSON") from error
    if observed != expected:
        raise ArtifactPublicationError(f"{name} has an invalid self digest")
    return observed


def _validate_screen_handoff_identity(
    *,
    readiness_receipt: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
    torch_evidence: Mapping[str, Any],
    run_result: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
    attempt_record: Mapping[str, Any],
) -> None:
    """Require every screen document to describe one exact staged handoff."""

    document_bindings: tuple[
        tuple[str, Mapping[str, Any], tuple[tuple[str, str], ...]],
        ...,
    ] = (
        (
            "readiness_receipt",
            readiness_receipt,
            (
                ("manifest_digest", "manifest_digest"),
                ("dependency_lock_digest", "dependency_lock_digest"),
                ("dependency_quarantine_revision", "dependency_quarantine_revision"),
                (
                    "dependency_quarantine_evidence_digest",
                    "dependency_quarantine_evidence_digest",
                ),
                ("dependency_private_tree_digest", "dependency_private_tree_digest"),
                ("code_materialization_revision", "code_materialization_revision"),
                (
                    "code_materialization_evidence_digest",
                    "code_materialization_evidence_digest",
                ),
                ("code_private_tree_digest", "code_private_tree_digest"),
            ),
        ),
        (
            "bundle_receipt",
            bundle_receipt,
            (
                ("manifest_digest", "manifest_digest"),
                ("source_contract_digest", "source_contract_digest"),
                ("bundle_digest", "workload_bundle_digest"),
                ("bundle_size_bytes", "workload_bundle_size_bytes"),
                ("bundle_path", "workload_bundle_path"),
                ("bundle_handoff_revision", "bundle_handoff_revision"),
            ),
        ),
        (
            "torch_evidence",
            torch_evidence,
            (
                ("head_commit", "source_head_commit"),
                ("source_contract_digest", "source_contract_digest"),
                ("workload_bundle_digest", "workload_bundle_digest"),
                ("workload_bundle_size_bytes", "workload_bundle_size_bytes"),
                ("workload_bundle_path", "workload_bundle_path"),
                ("bootstrap_source_digest", "bootstrap_source_digest"),
                ("dependency_lock_digest", "dependency_lock_digest"),
                ("dependency_quarantine_revision", "dependency_quarantine_revision"),
                (
                    "dependency_quarantine_evidence_digest",
                    "dependency_quarantine_evidence_digest",
                ),
                ("dependency_private_tree_digest", "dependency_private_tree_digest"),
                ("code_materialization_revision", "code_materialization_revision"),
                (
                    "code_materialization_evidence_digest",
                    "code_materialization_evidence_digest",
                ),
                ("code_private_tree_digest", "code_private_tree_digest"),
            ),
        ),
        (
            "run_result",
            run_result,
            (
                ("source_head_commit", "source_head_commit"),
                ("source_contract_digest", "source_contract_digest"),
                (
                    "live_stage_activation_revision",
                    "live_stage_activation_revision",
                ),
                ("live_stage_activation_digest", "live_stage_activation_digest"),
                ("dependency_lock_digest", "dependency_lock_digest"),
                ("dependency_quarantine_revision", "dependency_quarantine_revision"),
                (
                    "dependency_quarantine_evidence_digest",
                    "dependency_quarantine_evidence_digest",
                ),
                ("dependency_private_tree_digest", "dependency_private_tree_digest"),
                ("code_materialization_revision", "code_materialization_revision"),
                (
                    "code_materialization_evidence_digest",
                    "code_materialization_evidence_digest",
                ),
                ("code_private_tree_digest", "code_private_tree_digest"),
            ),
        ),
        (
            "attempt_record",
            attempt_record,
            (
                ("source_head_commit", "source_head_commit"),
                ("source_contract_digest", "source_contract_digest"),
                ("bootstrap_source_digest", "bootstrap_source_digest"),
                ("workload_bundle_digest", "workload_bundle_digest"),
                ("dependency_lock_digest", "dependency_lock_digest"),
                ("dependency_quarantine_revision", "dependency_quarantine_revision"),
                (
                    "dependency_quarantine_evidence_digest",
                    "dependency_quarantine_evidence_digest",
                ),
                ("dependency_private_tree_digest", "dependency_private_tree_digest"),
                ("code_materialization_revision", "code_materialization_revision"),
                (
                    "code_materialization_evidence_digest",
                    "code_materialization_evidence_digest",
                ),
                ("code_private_tree_digest", "code_private_tree_digest"),
                ("activation_digest", "live_stage_activation_digest"),
            ),
        ),
    )
    for document_name, document, bindings in document_bindings:
        for document_field, resource_field in bindings:
            expected = resource_profile.get(resource_field)
            if expected is None or expected == "":
                raise ArtifactPublicationError(
                    f"provider resource_profile has no valid {resource_field}"
                )
            if document.get(document_field) != expected:
                raise ArtifactPublicationError(
                    f"{document_name} {document_field} does not match "
                    f"provider resource_profile {resource_field}"
                )
    if resource_profile.get("bundle_activation_digest") != resource_profile.get(
        "live_stage_activation_digest"
    ):
        raise ArtifactPublicationError("provider resource_profile activation digest aliases differ")


def _validate_screen_evidence_bindings(
    *,
    payloads: Mapping[str, bytes],
    run_result: Mapping[str, Any],
    provider_receipt: Mapping[str, Any],
    attempt_record: Mapping[str, Any],
) -> None:
    resource_profile = provider_receipt.get("resource_profile")
    if not isinstance(resource_profile, dict):
        raise ArtifactPublicationError(
            "screen provider_receipt has no resource_profile evidence bindings"
        )
    evidence_bindings = (
        (
            "readiness_receipt",
            "receipt_digest",
            "volume_readiness_receipt_digest",
        ),
        (
            "bundle_receipt",
            "receipt_digest",
            "bundle_stage_receipt_digest",
        ),
        (
            "torch_evidence",
            "evidence_digest",
            "torch_retention_evidence_digest",
        ),
    )
    evidence_values: dict[str, dict[str, Any]] = {}
    evidence_digests: dict[str, str] = {}
    for role, digest_field, binding_field in evidence_bindings:
        evidence = _json_object(payloads[role], role)
        evidence_values[role] = evidence
        evidence_digest = _embedded_evidence_digest(
            evidence,
            digest_field=digest_field,
            name=role,
        )
        evidence_digests[role] = evidence_digest
        if attempt_record.get(binding_field) != evidence_digest:
            raise ArtifactPublicationError(f"{role} does not match attempt_record {binding_field}")
        if resource_profile.get(binding_field) != evidence_digest:
            raise ArtifactPublicationError(
                f"{role} does not match provider resource_profile {binding_field}"
            )

    if run_result.get("volume_readiness_receipt") != evidence_values["readiness_receipt"]:
        raise ArtifactPublicationError(
            "readiness_receipt does not match the exact committed run_result"
        )
    if run_result.get("retention_checkpoint_evidence") != evidence_values["torch_evidence"]:
        raise ArtifactPublicationError(
            "torch_evidence does not match the exact committed run_result"
        )
    if run_result.get("volume_readiness_receipt_digest") != evidence_digests["readiness_receipt"]:
        raise ArtifactPublicationError(
            "run_result volume_readiness_receipt_digest does not match readiness_receipt"
        )
    if run_result.get("torch_retention_evidence_digest") != evidence_digests["torch_evidence"]:
        raise ArtifactPublicationError(
            "run_result torch_retention_evidence_digest does not match torch_evidence"
        )
    torch_evidence = evidence_values["torch_evidence"]
    if (
        torch_evidence.get("volume_readiness_receipt_digest")
        != evidence_digests["readiness_receipt"]
    ):
        raise ArtifactPublicationError("torch_evidence does not bind the exact readiness_receipt")
    if torch_evidence.get("bundle_stage_receipt_digest") != evidence_digests["bundle_receipt"]:
        raise ArtifactPublicationError("torch_evidence does not bind the exact bundle_receipt")
    for identity_field in (
        "network_volume_id",
        "network_volume_data_center_id",
        "network_volume_size_gb",
    ):
        expected = resource_profile.get(identity_field)
        if (
            (identity_field != "network_volume_size_gb" and not isinstance(expected, str))
            or (identity_field != "network_volume_size_gb" and not expected)
            or (
                identity_field == "network_volume_size_gb"
                and (type(expected) is not int or expected < 1)
            )
        ):
            raise ArtifactPublicationError(
                f"provider resource_profile has no valid {identity_field}"
            )
        for role, evidence in evidence_values.items():
            if evidence.get(identity_field) != expected:
                raise ArtifactPublicationError(
                    f"{role} does not match provider resource_profile {identity_field}"
                )


def _set_digest(
    publication_id: str,
    publication_type: str,
    profile_id: str,
    descriptors: Sequence[Mapping[str, Any]],
    source_screen: Mapping[str, str] | None,
    archive_descriptors: Sequence[Mapping[str, Any]] = (),
) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "publication_type": publication_type,
        "publication_id": publication_id,
        "proof_id": publication_id,
        "profile_id": profile_id,
        "source_screen": dict(source_screen) if source_screen is not None else None,
        "artifacts": [dict(descriptor) for descriptor in descriptors],
        "prior_profile_archives": [dict(descriptor) for descriptor in archive_descriptors],
    }
    return _sha256(_publication_canonical_bytes(identity, "artifact-set identity"))


def _profile_archive_suffix(profile_id: str) -> str:
    version = profile_id.rsplit("@", 1)[-1]
    if version.isdigit():
        return f"profile-v{version}"
    return "profile-" + hashlib.sha256(profile_id.encode()).hexdigest()[:12]


def _manifest_without_digest(
    *,
    publication_id: str,
    publication_type: str,
    profile_id: str,
    generation_name: str,
    set_digest: str,
    descriptors: Sequence[Mapping[str, Any]],
    archive_descriptors: Sequence[Mapping[str, Any]],
    source_screen: Mapping[str, str] | None,
) -> dict[str, Any]:
    artifact_entries = []
    generation_path = f"{_GENERATION_DIRECTORY}/{generation_name}"
    for descriptor in descriptors:
        entry = dict(descriptor)
        entry["path"] = f"{generation_path}/{descriptor['filename']}"
        del entry["filename"]
        artifact_entries.append(entry)
    archive_entries = []
    for descriptor in archive_descriptors:
        entry = dict(descriptor)
        entry["path"] = f"{generation_path}/{descriptor['path']}"
        archive_entries.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "publication_type": publication_type,
        "publication_id": publication_id,
        "proof_id": publication_id,
        "profile_id": profile_id,
        "visibility": "committed",
        "teardown_confirmed": True,
        "artifact_set_committed": True,
        "set_digest": set_digest,
        "generation": generation_path,
        "artifacts": artifact_entries,
        "source_screen": dict(source_screen) if source_screen is not None else None,
        "prior_profile_archives": archive_entries,
    }


def _manifest_payload(journal: Mapping[str, Any]) -> bytes:
    manifest = _manifest_without_digest(
        publication_id=str(journal["publication_id"]),
        publication_type=str(journal["publication_type"]),
        profile_id=str(journal["profile_id"]),
        generation_name=str(journal["generation_name"]),
        set_digest=str(journal["set_digest"]),
        descriptors=journal["artifacts"],
        archive_descriptors=journal["prior_profile_archives"],
        source_screen=journal["source_screen"],
    )
    manifest["artifact_set_manifest_digest"] = _sha256(
        _publication_canonical_bytes(manifest, "artifact-set manifest")
    )
    return _publication_canonical_bytes(manifest, "artifact-set manifest") + b"\n"


def _manifest_path(proof_directory: Path, publication_id: str) -> Path:
    return proof_directory / f"{publication_id}{_MANIFEST_SUFFIX}"


def _state_paths(
    state_directory: Path,
    publication_id: str,
) -> tuple[Path, Path]:
    root = state_directory / _STATE_DIRECTORY
    return root, root / f"{publication_id}.transaction"


@contextlib.contextmanager
def _publication_lock(proof_directory: Path, publication_id: str):
    lock_path = proof_directory / f".{publication_id}.artifact-publication.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ArtifactPublicationError("publication lock is unavailable or unsafe") from error
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactPublicationError("publication lock is not a regular file")
        _validate_trusted_mode(info, "publication lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ArtifactPublicationBusy(
                f"artifact publication {publication_id!r} is active"
            ) from error
        os.fsync(descriptor)
        _fsync_directory(proof_directory)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _request_directories(request: PublicationRequest) -> tuple[Path, Path]:
    proof_directory = _absolute(request.proof_directory)
    state_directory = _absolute(request.state_directory)
    _ensure_directory(proof_directory)
    _ensure_directory(state_directory)
    _validate_trusted_directory(proof_directory, "proof directory")
    _validate_trusted_directory(state_directory, "publication state directory")
    return proof_directory, state_directory


def _load_staged_artifacts(
    request: PublicationRequest,
    *,
    publication_id: str,
    publication_type: str,
    profile_id: str,
) -> tuple[dict[str, _StagedArtifact], list[dict[str, Any]]]:
    roles = _artifact_roles(publication_type)
    if set(request.artifacts) != set(roles):
        raise ArtifactPublicationError(
            f"{publication_type} publication requires exactly its declared artifact roles"
        )
    artifacts: dict[str, _StagedArtifact] = {}
    descriptors: list[dict[str, Any]] = []
    for role in roles:
        raw_path = _absolute(Path(request.artifacts[role]))
        if role in JSON_ARTIFACT_ROLES:
            payload = _read_regular_file(
                raw_path,
                f"staged {role}",
                maximum_bytes=_maximum_artifact_bytes(role),
            )
            _validate_payload_identity(
                payload,
                role=role,
                publication_id=publication_id,
                profile_id=profile_id,
            )
            descriptor = _json_artifact_descriptor(role, payload)
            artifacts[role] = _StagedArtifact(
                role=role,
                path=raw_path,
                descriptor=descriptor,
                json_payload=payload,
            )
        else:
            descriptor = _binary_artifact_descriptor(role, raw_path)
            artifacts[role] = _StagedArtifact(
                role=role,
                path=raw_path,
                descriptor=descriptor,
                json_payload=None,
            )
        descriptors.append(descriptor)

    _validate_cross_artifact_bindings(
        publication_type=publication_type,
        payloads={
            role: artifact.json_payload
            for role, artifact in artifacts.items()
            if artifact.json_payload is not None
        },
        model_artifact_path=(
            artifacts["model_artifact"].path if publication_type == "pilot" else None
        ),
        model_artifact_descriptor=(
            artifacts["model_artifact"].descriptor if publication_type == "pilot" else None
        ),
    )
    return artifacts, descriptors


def _load_prior_archives(
    request: PublicationRequest,
    *,
    proof_directory: Path,
    profile_id: str,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    prior_profile_id = request.prior_profile_id
    if not request.prior_receipts and prior_profile_id is None:
        return {}, []
    if prior_profile_id is None or set(request.prior_receipts) != set(ARCHIVABLE_PRIOR_ROLES):
        raise ArtifactPublicationError(
            "prior receipt preservation requires both receipt roles and prior_profile_id"
        )
    prior_profile_id = _safe_component(prior_profile_id, "prior_profile_id")
    if prior_profile_id == profile_id:
        raise ArtifactPublicationError("prior_profile_id must differ from profile_id")
    suffix = _profile_archive_suffix(prior_profile_id)
    payloads: dict[str, bytes] = {}
    descriptors: list[dict[str, Any]] = []
    for role in ARCHIVABLE_PRIOR_ROLES:
        source = _absolute(Path(request.prior_receipts[role]))
        if source.parent != proof_directory:
            raise ArtifactPublicationError("prior receipts must be direct proof-directory files")
        payload = _read_regular_file(
            source,
            f"prior {role}",
            maximum_bytes=MAXIMUM_JSON_ARTIFACT_BYTES,
        )
        value = _json_object(payload, f"prior {role}")
        if value.get("profile_id") != prior_profile_id:
            raise ArtifactPublicationError(f"prior {role} has the wrong profile_id")
        archive_name = f"{source.stem}.{suffix}{source.suffix}"
        _safe_component(archive_name, f"{role} archive filename")
        payloads[role] = payload
        descriptors.append(
            {
                "role": role,
                "profile_id": prior_profile_id,
                "path": archive_name,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    return payloads, descriptors


def _resolve_source_screen(
    request: PublicationRequest,
    *,
    proof_directory: Path,
    publication_id: str,
    publication_type: str,
    profile_id: str,
) -> dict[str, str] | None:
    source_publication_id = request.source_screen_publication_id
    source_manifest_digest = request.source_screen_manifest_digest
    if publication_type == "screen":
        if source_publication_id is not None or source_manifest_digest is not None:
            raise ArtifactPublicationError("screen publication must not declare source_screen")
        return None
    if source_publication_id is None or source_manifest_digest is None:
        raise ArtifactPublicationError("pilot publication requires a committed source_screen")
    source_publication_id = _safe_component(
        source_publication_id,
        "source_screen_publication_id",
    )
    if source_publication_id == publication_id:
        raise ArtifactPublicationError("pilot source_screen cannot reference itself")
    if not isinstance(source_manifest_digest, str) or not _TAGGED_SHA256.fullmatch(
        source_manifest_digest
    ):
        raise ArtifactPublicationError("source_screen manifest digest is invalid")
    source = load_committed_artifact_set(
        proof_directory,
        source_publication_id,
        expected_profile_id=profile_id,
        expected_publication_type="screen",
    )
    if source.artifact_set_manifest_digest != source_manifest_digest:
        raise ArtifactPublicationError(
            "source_screen manifest digest does not match the committed screen"
        )
    return {
        "publication_id": source_publication_id,
        "artifact_set_manifest_digest": source_manifest_digest,
    }


def _validate_request(
    request: PublicationRequest,
) -> tuple[
    str,
    str,
    str,
    Path,
    Path,
    dict[str, _StagedArtifact],
    list[dict[str, Any]],
    dict[str, bytes],
    list[dict[str, Any]],
    dict[str, str] | None,
]:
    publication_id = _safe_component(request.publication_id, "publication_id")
    publication_type = _safe_component(request.publication_type, "publication_type")
    _artifact_roles(publication_type)
    profile_id = _safe_component(request.profile_id, "profile_id")
    if request.teardown_confirmed is not True:
        raise ArtifactPublicationError("artifact publication requires confirmed provider teardown")
    proof_directory, state_directory = _request_directories(request)
    artifacts, descriptors = _load_staged_artifacts(
        request,
        publication_id=publication_id,
        publication_type=publication_type,
        profile_id=profile_id,
    )
    source_screen = _resolve_source_screen(
        request,
        proof_directory=proof_directory,
        publication_id=publication_id,
        publication_type=publication_type,
        profile_id=profile_id,
    )
    if publication_type == "pilot" and (
        request.prior_profile_id is not None or request.prior_receipts
    ):
        raise ArtifactPublicationError(
            "pilot publication must not archive predecessor profile receipts"
        )
    archive_payloads, archive_descriptors = _load_prior_archives(
        request,
        proof_directory=proof_directory,
        profile_id=profile_id,
    )
    return (
        publication_id,
        publication_type,
        profile_id,
        proof_directory,
        state_directory,
        artifacts,
        descriptors,
        archive_payloads,
        archive_descriptors,
        source_screen,
    )


def _journal(
    *,
    publication_id: str,
    publication_type: str,
    profile_id: str,
    proof_directory: Path,
    state_directory: Path,
    pending_name: str,
    generation_name: str,
    set_digest: str,
    descriptors: Sequence[Mapping[str, Any]],
    archive_descriptors: Sequence[Mapping[str, Any]],
    source_screen: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": JOURNAL_KIND,
        "publication_type": publication_type,
        "publication_id": publication_id,
        "proof_id": publication_id,
        "profile_id": profile_id,
        "proof_directory": str(proof_directory),
        "state_directory": str(state_directory),
        "owner_pid": os.getpid(),
        "teardown_confirmed": True,
        "manifest": f"{publication_id}{_MANIFEST_SUFFIX}",
        "pending_name": pending_name,
        "generation_name": generation_name,
        "set_digest": set_digest,
        "artifacts": [dict(item) for item in descriptors],
        "source_screen": dict(source_screen) if source_screen is not None else None,
        "prior_profile_archives": [dict(item) for item in archive_descriptors],
    }


def _validate_source_screen_descriptor(
    value: Any,
    *,
    publication_type: str,
    publication_id: str,
) -> dict[str, str] | None:
    if publication_type == "screen":
        if value is not None:
            raise ArtifactPublicationError("screen source_screen must be null")
        return None
    if not isinstance(value, dict) or set(value) != {
        "publication_id",
        "artifact_set_manifest_digest",
    }:
        raise ArtifactPublicationError("pilot source_screen descriptor is invalid")
    source_publication_id = _safe_component(
        value["publication_id"],
        "source_screen publication_id",
    )
    source_manifest_digest = value["artifact_set_manifest_digest"]
    if (
        source_publication_id == publication_id
        or not isinstance(source_manifest_digest, str)
        or not _TAGGED_SHA256.fullmatch(source_manifest_digest)
    ):
        raise ArtifactPublicationError("pilot source_screen identity is invalid")
    return {
        "publication_id": source_publication_id,
        "artifact_set_manifest_digest": source_manifest_digest,
    }


def _validate_descriptors(
    value: Any,
    *,
    publication_type: str,
) -> list[dict[str, Any]]:
    roles = _artifact_roles(publication_type)
    if not isinstance(value, list) or len(value) != len(roles):
        raise ArtifactPublicationError("publication journal artifact set is incomplete")
    descriptors: list[dict[str, Any]] = []
    for role, raw in zip(roles, value, strict=True):
        expected_keys = {
            "role",
            "media_type",
            "filename",
            "sha256",
            "size_bytes",
        }
        if role in JSON_ARTIFACT_ROLES:
            expected_keys.add("canonical_json_sha256")
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ArtifactPublicationError("publication journal artifact descriptor is invalid")
        if (
            raw["role"] != role
            or raw["media_type"] != ARTIFACT_MEDIA_TYPES[role]
            or raw["filename"] != ARTIFACT_FILENAMES[role]
            or not isinstance(raw["sha256"], str)
            or not _TAGGED_SHA256.fullmatch(raw["sha256"])
            or type(raw["size_bytes"]) is not int
            or raw["size_bytes"] < 0
            or (role == "model_artifact" and raw["size_bytes"] == 0)
            or raw["size_bytes"] > _maximum_artifact_bytes(role)
            or (
                role in JSON_ARTIFACT_ROLES
                and (
                    not isinstance(raw["canonical_json_sha256"], str)
                    or not _TAGGED_SHA256.fullmatch(raw["canonical_json_sha256"])
                )
            )
        ):
            raise ArtifactPublicationError("publication journal artifact identity is invalid")
        descriptors.append(dict(raw))
    return descriptors


def _validate_archive_descriptors(
    value: Any,
    *,
    generation_name: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) not in {0, len(ARCHIVABLE_PRIOR_ROLES)}:
        raise ArtifactPublicationError("publication journal archive set is invalid")
    descriptors: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != {
            "role",
            "profile_id",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ArtifactPublicationError("publication journal archive descriptor is invalid")
        role = ARCHIVABLE_PRIOR_ROLES[index]
        raw_path = raw["path"]
        if generation_name is not None:
            prefix = f"{_GENERATION_DIRECTORY}/{generation_name}/"
            if not isinstance(raw_path, str) or not raw_path.startswith(prefix):
                raise ArtifactPublicationError(
                    "artifact-set archive path is outside its immutable generation"
                )
            archive_name = raw_path.removeprefix(prefix)
        else:
            archive_name = raw_path
        if (
            raw["role"] != role
            or _safe_component(raw["profile_id"], "archived profile_id") == ""
            or _safe_component(archive_name, "archive path") != archive_name
            or not isinstance(raw["sha256"], str)
            or not _TAGGED_SHA256.fullmatch(raw["sha256"])
            or type(raw["size_bytes"]) is not int
            or raw["size_bytes"] < 0
            or raw["size_bytes"] > MAXIMUM_JSON_ARTIFACT_BYTES
        ):
            raise ArtifactPublicationError("publication journal archive identity is invalid")
        descriptors.append({**raw, "path": archive_name})
    if descriptors and len({item["profile_id"] for item in descriptors}) != 1:
        raise ArtifactPublicationError("publication journal archive profiles differ")
    return descriptors


def _load_journal(
    transaction_directory: Path,
    *,
    publication_id: str,
    proof_directory: Path,
    state_directory: Path,
) -> dict[str, Any]:
    journal_path = transaction_directory / "journal.json"
    payload = _read_regular_file(
        journal_path,
        "artifact publication journal",
        maximum_bytes=1024 * 1024,
    )
    value = _json_object(payload, "artifact publication journal")
    expected_keys = {
        "schema_version",
        "kind",
        "publication_type",
        "publication_id",
        "proof_id",
        "profile_id",
        "proof_directory",
        "state_directory",
        "owner_pid",
        "teardown_confirmed",
        "manifest",
        "pending_name",
        "generation_name",
        "set_digest",
        "artifacts",
        "source_screen",
        "prior_profile_archives",
    }
    if set(value) != expected_keys:
        raise ArtifactPublicationError("artifact publication journal shape is invalid")
    profile_id = _safe_component(value["profile_id"], "journal profile_id")
    publication_type = _safe_component(
        value["publication_type"],
        "journal publication_type",
    )
    _artifact_roles(publication_type)
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != JOURNAL_KIND
        or value["publication_id"] != publication_id
        or value["proof_id"] != publication_id
        or value["proof_directory"] != str(proof_directory)
        or value["state_directory"] != str(state_directory)
        or type(value["owner_pid"]) is not int
        or value["owner_pid"] <= 0
        or value["teardown_confirmed"] is not True
        or value["manifest"] != f"{publication_id}{_MANIFEST_SUFFIX}"
        or not isinstance(value["set_digest"], str)
        or not _TAGGED_SHA256.fullmatch(value["set_digest"])
    ):
        raise ArtifactPublicationError("artifact publication journal identity is invalid")
    descriptors = _validate_descriptors(
        value["artifacts"],
        publication_type=publication_type,
    )
    source_screen = _validate_source_screen_descriptor(
        value["source_screen"],
        publication_type=publication_type,
        publication_id=publication_id,
    )
    archives = _validate_archive_descriptors(value["prior_profile_archives"])
    if publication_type == "pilot" and archives:
        raise ArtifactPublicationError("pilot publication journal cannot contain archives")
    expected_set_digest = _set_digest(
        publication_id,
        publication_type,
        profile_id,
        descriptors,
        source_screen,
        archives,
    )
    generation_name = f"{publication_id}.{expected_set_digest.removeprefix('sha256:')}"
    pending_name = value["pending_name"]
    if (
        value["set_digest"] != expected_set_digest
        or value["generation_name"] != generation_name
        or not isinstance(pending_name, str)
        or not re.fullmatch(
            rf"\.pending-{re.escape(publication_id)}-[0-9a-f]{{16}}",
            pending_name,
        )
    ):
        raise ArtifactPublicationError("artifact publication journal generation is invalid")
    value["profile_id"] = profile_id
    value["publication_type"] = publication_type
    value["artifacts"] = descriptors
    value["source_screen"] = source_screen
    value["prior_profile_archives"] = archives
    return value


def _verify_file(
    path: Path,
    descriptor: Mapping[str, Any],
    name: str,
) -> bytes | None:
    role = str(descriptor["role"])
    if role == "model_artifact":
        size_bytes, sha256 = _stream_regular_file(
            path,
            name,
            maximum_bytes=_maximum_artifact_bytes(role),
        )
        if size_bytes != descriptor["size_bytes"] or sha256 != descriptor["sha256"]:
            raise ArtifactPublicationError(f"{name} digest or size does not match")
        return None
    payload = _read_regular_file(
        path,
        name,
        maximum_bytes=_maximum_artifact_bytes(role),
    )
    if len(payload) != descriptor["size_bytes"] or _sha256(payload) != descriptor["sha256"]:
        raise ArtifactPublicationError(f"{name} digest or size does not match")
    canonical_digest = descriptor.get("canonical_json_sha256")
    if canonical_digest is not None:
        value = _json_object(payload, name)
        observed_canonical_digest = _sha256(_publication_canonical_bytes(value, name))
        if observed_canonical_digest != canonical_digest:
            raise ArtifactPublicationError(f"{name} canonical JSON digest does not match")
    return payload


def _verify_generation(
    generation_directory: Path,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    archive_descriptors: Sequence[Mapping[str, Any]] = (),
    require_complete: bool = True,
) -> None:
    if generation_directory.is_symlink() or not generation_directory.is_dir():
        raise ArtifactPublicationError("artifact generation is unavailable or unsafe")
    _validate_trusted_directory(generation_directory, "artifact generation")
    expected = {str(item["filename"]): item for item in descriptors}
    for descriptor in archive_descriptors:
        archive_name = str(descriptor["path"])
        if archive_name in expected:
            raise ArtifactPublicationError("prior receipt archive collides with a current artifact")
        expected[archive_name] = descriptor
    observed = {entry.name: entry for entry in generation_directory.iterdir()}
    unexpected = set(observed) - set(expected)
    if unexpected:
        raise ArtifactPublicationError("artifact generation contains unexpected entries")
    if require_complete and set(observed) != set(expected):
        raise ArtifactPublicationError("artifact generation is incomplete")
    for name, path in observed.items():
        descriptor = expected[name]
        qualifier = "prior " if "profile_id" in descriptor else ""
        _verify_file(
            path,
            descriptor,
            f"generated {qualifier}{descriptor['role']}",
        )


def _remove_generation(
    generation_directory: Path,
    descriptors: Sequence[Mapping[str, Any]],
    archive_descriptors: Sequence[Mapping[str, Any]] = (),
) -> None:
    if not generation_directory.exists() and not generation_directory.is_symlink():
        return
    if generation_directory.is_symlink() or not generation_directory.is_dir():
        raise ArtifactPublicationError("artifact generation is unavailable or unsafe")
    for descriptor in descriptors:
        _remove_atomic_temps(
            generation_directory,
            str(descriptor["filename"]),
        )
    for descriptor in archive_descriptors:
        _remove_atomic_temps(
            generation_directory,
            str(descriptor["path"]),
        )
    _verify_generation(
        generation_directory,
        descriptors,
        archive_descriptors=archive_descriptors,
        require_complete=False,
    )
    for descriptor in descriptors:
        path = generation_directory / str(descriptor["filename"])
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ArtifactPublicationError("artifact generation entry is unsafe")
            path.unlink()
            _fsync_directory(generation_directory)
    for descriptor in archive_descriptors:
        path = generation_directory / str(descriptor["path"])
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ArtifactPublicationError("prior receipt archive entry is unsafe")
            path.unlink()
            _fsync_directory(generation_directory)
    generation_directory.rmdir()
    _fsync_directory(generation_directory.parent)


def _remove_atomic_temps(parent: Path, final_name: str) -> None:
    prefix = f".{final_name}."
    for candidate in parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ArtifactPublicationError("publication temporary path is unsafe")
        candidate.unlink()
        _fsync_directory(parent)


def _remove_transaction_directory(transaction_directory: Path) -> None:
    if not transaction_directory.exists() and not transaction_directory.is_symlink():
        return
    if transaction_directory.is_symlink() or not transaction_directory.is_dir():
        raise ArtifactPublicationError("publication transaction directory is unsafe")
    _remove_atomic_temps(transaction_directory, "journal.json")
    observed = {entry.name: entry for entry in transaction_directory.iterdir()}
    if set(observed) - {"journal.json"}:
        raise ArtifactPublicationError("publication transaction contains unexpected entries")
    journal = transaction_directory / "journal.json"
    if journal.exists() or journal.is_symlink():
        if journal.is_symlink() or not journal.is_file():
            raise ArtifactPublicationError("publication journal path is unsafe")
        journal.unlink()
        _fsync_directory(transaction_directory)
    transaction_directory.rmdir()
    _fsync_directory(transaction_directory.parent)


def _write_generation(
    pending_directory: Path,
    artifacts: Mapping[str, _StagedArtifact],
    descriptors: Sequence[Mapping[str, Any]],
    archive_payloads: Mapping[str, bytes],
    archive_descriptors: Sequence[Mapping[str, Any]],
    fault_injector: FaultInjector | None,
) -> None:
    try:
        pending_directory.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ArtifactPublicationError("pending artifact generation already exists") from error
    _fsync_directory(pending_directory)
    _fsync_directory(pending_directory.parent)
    for descriptor in descriptors:
        role = str(descriptor["role"])
        destination = pending_directory / str(descriptor["filename"])
        artifact = artifacts[role]
        if artifact.json_payload is not None:
            _atomic_write(destination, artifact.json_payload)
        else:
            _atomic_copy_verified(
                destination,
                artifact.path,
                descriptor,
            )
        _verify_file(destination, descriptor, f"pending {role}")
        if fault_injector is not None:
            fault_injector(f"after_artifact_fsync:{role}")
    for descriptor in archive_descriptors:
        role = str(descriptor["role"])
        destination = pending_directory / str(descriptor["path"])
        _atomic_write(destination, archive_payloads[role])
        _verify_file(destination, descriptor, f"pending prior {role}")
        if fault_injector is not None:
            fault_injector(f"after_archive:{role}")
    _verify_generation(
        pending_directory,
        descriptors,
        archive_descriptors=archive_descriptors,
    )
    _fsync_directory(pending_directory)


def _install_generation(
    pending_directory: Path,
    generation_directory: Path,
    descriptors: Sequence[Mapping[str, Any]],
    archive_descriptors: Sequence[Mapping[str, Any]],
) -> None:
    if generation_directory.is_symlink():
        raise ArtifactPublicationError("artifact generation destination is unsafe")
    if generation_directory.exists():
        _verify_generation(
            generation_directory,
            descriptors,
            archive_descriptors=archive_descriptors,
        )
        _remove_generation(
            pending_directory,
            descriptors,
            archive_descriptors,
        )
        return
    os.replace(pending_directory, generation_directory)
    _fsync_directory(generation_directory.parent)
    _verify_generation(
        generation_directory,
        descriptors,
        archive_descriptors=archive_descriptors,
    )


def _validate_manifest(
    value: Any,
    *,
    publication_id: str,
    expected_profile_id: str | None,
    expected_publication_type: str | None,
) -> tuple[
    str,
    str,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str] | None,
    str,
    str,
]:
    if not isinstance(value, dict):
        raise ArtifactPublicationError("artifact-set manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "kind",
        "publication_type",
        "publication_id",
        "proof_id",
        "profile_id",
        "visibility",
        "teardown_confirmed",
        "artifact_set_committed",
        "set_digest",
        "generation",
        "artifacts",
        "source_screen",
        "prior_profile_archives",
        "artifact_set_manifest_digest",
    }
    if set(value) != expected_keys:
        raise ArtifactPublicationError("artifact-set manifest shape is invalid")
    profile_id = _safe_component(value["profile_id"], "manifest profile_id")
    publication_type = _safe_component(
        value["publication_type"],
        "manifest publication_type",
    )
    roles = _artifact_roles(publication_type)
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != MANIFEST_KIND
        or value["publication_id"] != publication_id
        or value["proof_id"] != publication_id
        or value["visibility"] != "committed"
        or value["teardown_confirmed"] is not True
        or value["artifact_set_committed"] is not True
        or (expected_profile_id is not None and profile_id != expected_profile_id)
        or (expected_publication_type is not None and publication_type != expected_publication_type)
        or not isinstance(value["set_digest"], str)
        or not _TAGGED_SHA256.fullmatch(value["set_digest"])
        or not isinstance(value["artifact_set_manifest_digest"], str)
        or not _TAGGED_SHA256.fullmatch(value["artifact_set_manifest_digest"])
    ):
        raise ArtifactPublicationError("artifact-set manifest identity is invalid")
    manifest_without_digest = {
        key: item for key, item in value.items() if key != "artifact_set_manifest_digest"
    }
    if (
        _sha256(
            _publication_canonical_bytes(
                manifest_without_digest,
                "artifact-set manifest",
            )
        )
        != value["artifact_set_manifest_digest"]
    ):
        raise ArtifactPublicationError("artifact-set manifest digest is invalid")
    source_screen = _validate_source_screen_descriptor(
        value["source_screen"],
        publication_type=publication_type,
        publication_id=publication_id,
    )
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(roles):
        raise ArtifactPublicationError("artifact-set manifest is incomplete")
    descriptors: list[dict[str, Any]] = []
    generation_name = f"{publication_id}.{value['set_digest'].removeprefix('sha256:')}"
    expected_generation = f"{_GENERATION_DIRECTORY}/{generation_name}"
    if value["generation"] != expected_generation:
        raise ArtifactPublicationError("artifact-set generation identity is invalid")
    for role, raw in zip(roles, raw_artifacts, strict=True):
        expected_keys = {
            "role",
            "media_type",
            "path",
            "sha256",
            "size_bytes",
        }
        if role in JSON_ARTIFACT_ROLES:
            expected_keys.add("canonical_json_sha256")
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ArtifactPublicationError("artifact-set manifest descriptor is invalid")
        expected_path = f"{expected_generation}/{ARTIFACT_FILENAMES[role]}"
        if (
            raw["role"] != role
            or raw["media_type"] != ARTIFACT_MEDIA_TYPES[role]
            or raw["path"] != expected_path
            or not isinstance(raw["sha256"], str)
            or not _TAGGED_SHA256.fullmatch(raw["sha256"])
            or type(raw["size_bytes"]) is not int
            or raw["size_bytes"] < 0
            or (role == "model_artifact" and raw["size_bytes"] == 0)
            or raw["size_bytes"] > _maximum_artifact_bytes(role)
            or (
                role in JSON_ARTIFACT_ROLES
                and (
                    not isinstance(raw["canonical_json_sha256"], str)
                    or not _TAGGED_SHA256.fullmatch(raw["canonical_json_sha256"])
                )
            )
        ):
            raise ArtifactPublicationError("artifact-set manifest descriptor identity is invalid")
        descriptors.append(
            {
                "role": role,
                "media_type": ARTIFACT_MEDIA_TYPES[role],
                "filename": ARTIFACT_FILENAMES[role],
                "sha256": raw["sha256"],
                "size_bytes": raw["size_bytes"],
                **(
                    {"canonical_json_sha256": raw["canonical_json_sha256"]}
                    if role in JSON_ARTIFACT_ROLES
                    else {}
                ),
            }
        )
    archives = _validate_archive_descriptors(
        value["prior_profile_archives"],
        generation_name=generation_name,
    )
    if publication_type == "pilot" and archives:
        raise ArtifactPublicationError("pilot artifact-set cannot contain prior archives")
    if (
        _set_digest(
            publication_id,
            publication_type,
            profile_id,
            descriptors,
            source_screen,
            archives,
        )
        != value["set_digest"]
    ):
        raise ArtifactPublicationError("artifact-set digest is invalid")
    return (
        publication_type,
        profile_id,
        generation_name,
        descriptors,
        archives,
        source_screen,
        value["set_digest"],
        value["artifact_set_manifest_digest"],
    )


def load_committed_artifact_set(
    proof_directory: Path,
    publication_id: str,
    *,
    expected_profile_id: str | None = None,
    expected_publication_type: str | None = None,
) -> PublicationResult:
    """Load and fully verify one committed set through its visibility manifest."""

    publication_id = _safe_component(publication_id, "publication_id")
    if expected_profile_id is not None:
        expected_profile_id = _safe_component(expected_profile_id, "expected_profile_id")
    if expected_publication_type is not None:
        expected_publication_type = _safe_component(
            expected_publication_type,
            "expected_publication_type",
        )
        _artifact_roles(expected_publication_type)
    proof_directory = _absolute(proof_directory)
    if proof_directory.is_symlink() or not proof_directory.is_dir():
        raise ArtifactPublicationError("proof directory is unavailable or unsafe")
    _validate_trusted_directory(proof_directory, "proof directory")
    manifest_path = _manifest_path(proof_directory, publication_id)
    payload = _read_regular_file(
        manifest_path,
        "artifact-set commit manifest",
        maximum_bytes=1024 * 1024,
    )
    value = _json_object(payload, "artifact-set commit manifest")
    (
        publication_type,
        profile_id,
        generation_name,
        descriptors,
        archive_descriptors,
        source_screen,
        set_digest,
        artifact_set_manifest_digest,
    ) = _validate_manifest(
        value,
        publication_id=publication_id,
        expected_profile_id=expected_profile_id,
        expected_publication_type=expected_publication_type,
    )
    generation_directory = proof_directory / _GENERATION_DIRECTORY / generation_name
    _validate_trusted_directory(
        generation_directory.parent,
        "artifact generation root",
    )
    _verify_generation(
        generation_directory,
        descriptors,
        archive_descriptors=archive_descriptors,
    )
    artifact_paths: dict[str, Path] = {}
    json_payloads: dict[str, bytes] = {}
    for descriptor in descriptors:
        role = str(descriptor["role"])
        path = generation_directory / str(descriptor["filename"])
        payload = _verify_file(path, descriptor, f"committed {role}")
        if payload is not None:
            _validate_payload_identity(
                payload,
                role=role,
                publication_id=publication_id,
                profile_id=profile_id,
            )
            json_payloads[role] = payload
        artifact_paths[role] = path
    _validate_cross_artifact_bindings(
        publication_type=publication_type,
        payloads=json_payloads,
        model_artifact_path=artifact_paths.get("model_artifact"),
        model_artifact_descriptor=(
            next(
                (
                    descriptor
                    for descriptor in descriptors
                    if descriptor["role"] == "model_artifact"
                ),
                None,
            )
        ),
    )
    archive_paths: dict[str, Path] = {}
    for descriptor in archive_descriptors:
        role = str(descriptor["role"])
        path = generation_directory / str(descriptor["path"])
        payload = _verify_file(path, descriptor, f"committed prior {role} archive")
        if payload is None:
            raise ArtifactPublicationError("prior receipt archive is not JSON")
        value = _json_object(payload, f"committed prior {role} archive")
        if value.get("profile_id") != descriptor["profile_id"]:
            raise ArtifactPublicationError("archived receipt profile identity is invalid")
        archive_paths[role] = path
    if source_screen is not None:
        source = load_committed_artifact_set(
            proof_directory,
            source_screen["publication_id"],
            expected_profile_id=profile_id,
            expected_publication_type="screen",
        )
        if source.artifact_set_manifest_digest != source_screen["artifact_set_manifest_digest"]:
            raise ArtifactPublicationError(
                "pilot source_screen no longer matches its committed manifest"
            )
    return PublicationResult(
        publication_id=publication_id,
        publication_type=publication_type,
        profile_id=profile_id,
        manifest_path=manifest_path,
        generation_directory=generation_directory,
        artifact_paths=artifact_paths,
        artifact_descriptors={
            str(descriptor["role"]): dict(descriptor) for descriptor in descriptors
        },
        archive_paths=archive_paths,
        set_digest=set_digest,
        artifact_set_manifest_digest=artifact_set_manifest_digest,
        source_screen_publication_id=(
            source_screen["publication_id"] if source_screen is not None else None
        ),
        source_screen_manifest_digest=(
            source_screen["artifact_set_manifest_digest"] if source_screen is not None else None
        ),
    )


def build_committed_proof_replay_payload(
    proof_directory: Path,
    publication_id: str,
) -> tuple[PublicationResult, dict[str, Any]]:
    """Build the exact pilot proof POST from one verified committed set."""

    publication = load_committed_artifact_set(
        proof_directory,
        publication_id,
        expected_profile_id=ATOMIC_PUBLICATION_PROFILE_ID,
        expected_publication_type="pilot",
    )
    provider_receipt = load_committed_provider_receipt(publication)
    envelope = build_artifact_publication_envelope(publication)
    return publication, {
        **provider_receipt,
        "artifact_publication": envelope,
    }


def load_committed_provider_receipt(
    publication: PublicationResult,
) -> dict[str, Any]:
    """Re-read and verify the exact envelope-free provider receipt."""

    provider_receipt_payload = _verify_file(
        publication.artifact_paths["provider_receipt"],
        publication.artifact_descriptors["provider_receipt"],
        "committed provider_receipt",
    )
    if provider_receipt_payload is None:
        raise ArtifactPublicationError("committed provider_receipt is not JSON")
    provider_receipt = _json_object(
        provider_receipt_payload,
        "committed provider_receipt",
    )
    run_result_payload = _verify_file(
        publication.artifact_paths["run_result"],
        publication.artifact_descriptors["run_result"],
        "committed run_result",
    )
    if run_result_payload is None:
        raise ArtifactPublicationError("committed run_result is not JSON")
    run_result = _json_object(run_result_payload, "committed run_result")
    _validate_provider_receipt_contract(provider_receipt, run_result)
    resource_profile = provider_receipt.get("resource_profile")
    if (
        not isinstance(resource_profile, dict)
        or resource_profile.get("profile_id") != publication.profile_id
    ):
        raise ArtifactPublicationError("committed provider_receipt cannot be replayed as a proof")
    return provider_receipt


def build_artifact_publication_envelope(
    publication: PublicationResult,
) -> dict[str, Any]:
    """Derive the compact API envelope only from verified set descriptors."""

    run_result_descriptor = publication.artifact_descriptors["run_result"]
    provider_receipt_descriptor = publication.artifact_descriptors["provider_receipt"]
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "publication_id": publication.publication_id,
        "publication_type": publication.publication_type,
        "profile_id": publication.profile_id,
        "artifact_set_manifest_digest": publication.artifact_set_manifest_digest,
        "set_digest": publication.set_digest,
        "run_result_canonical_json_sha256": run_result_descriptor["canonical_json_sha256"],
        "provider_receipt_canonical_json_sha256": provider_receipt_descriptor[
            "canonical_json_sha256"
        ],
    }
    if publication.publication_type == "pilot":
        source_publication_id = publication.source_screen_publication_id
        source_manifest_digest = publication.source_screen_manifest_digest
        model_artifact_descriptor = publication.artifact_descriptors.get("model_artifact")
        if (
            source_publication_id is None
            or source_manifest_digest is None
            or model_artifact_descriptor is None
        ):
            raise ArtifactPublicationError(
                "committed pilot has no exact source_screen or model_artifact"
            )
        envelope.update(
            {
                "source_screen_publication_id": source_publication_id,
                "source_screen_manifest_digest": source_manifest_digest,
                "model_artifact_sha256": model_artifact_descriptor["sha256"],
                "model_artifact_size_bytes": model_artifact_descriptor["size_bytes"],
            }
        )
    return envelope


def _journal_matches_request(
    journal: Mapping[str, Any],
    *,
    publication_type: str,
    profile_id: str,
    descriptors: Sequence[Mapping[str, Any]],
    archive_descriptors: Sequence[Mapping[str, Any]],
    source_screen: Mapping[str, str] | None,
) -> None:
    if (
        journal["publication_type"] != publication_type
        or journal["profile_id"] != profile_id
        or journal["artifacts"] != list(descriptors)
        or journal["prior_profile_archives"] != list(archive_descriptors)
        or journal["source_screen"] != (dict(source_screen) if source_screen is not None else None)
    ):
        raise ArtifactPublicationError(
            "existing publication transaction belongs to different artifacts"
        )


def _cleanup_publication_temps(
    proof_directory: Path,
    publication_id: str,
) -> None:
    _remove_atomic_temps(
        proof_directory,
        f"{publication_id}{_MANIFEST_SUFFIX}",
    )


def _recover_locked(
    *,
    proof_directory: Path,
    state_directory: Path,
    publication_id: str,
) -> PublicationResult | None:
    state_root, transaction_directory = _state_paths(state_directory, publication_id)
    _ensure_directory(state_root)
    manifest_path = _manifest_path(proof_directory, publication_id)
    if not transaction_directory.exists() and not transaction_directory.is_symlink():
        if manifest_path.exists() or manifest_path.is_symlink():
            return load_committed_artifact_set(proof_directory, publication_id)
        return None
    if transaction_directory.is_symlink() or not transaction_directory.is_dir():
        raise ArtifactPublicationError("publication transaction directory is unsafe")
    journal_path = transaction_directory / "journal.json"
    if not journal_path.exists() and not journal_path.is_symlink():
        _remove_transaction_directory(transaction_directory)
        if manifest_path.exists() or manifest_path.is_symlink():
            return load_committed_artifact_set(proof_directory, publication_id)
        return None
    journal = _load_journal(
        transaction_directory,
        publication_id=publication_id,
        proof_directory=proof_directory,
        state_directory=state_directory,
    )
    generation_root = proof_directory / _GENERATION_DIRECTORY
    pending_directory = generation_root / str(journal["pending_name"])
    generation_directory = generation_root / str(journal["generation_name"])
    descriptors = journal["artifacts"]
    archive_descriptors = journal["prior_profile_archives"]
    if manifest_path.exists() or manifest_path.is_symlink():
        result = load_committed_artifact_set(
            proof_directory,
            publication_id,
            expected_profile_id=str(journal["profile_id"]),
            expected_publication_type=str(journal["publication_type"]),
        )
        if _read_regular_file(
            manifest_path,
            "artifact-set commit manifest",
            maximum_bytes=1024 * 1024,
        ) != _manifest_payload(journal):
            raise ArtifactPublicationError(
                "committed manifest differs from its durable transaction"
            )
        if pending_directory.exists() or pending_directory.is_symlink():
            _remove_generation(
                pending_directory,
                descriptors,
                archive_descriptors,
            )
        _cleanup_publication_temps(
            proof_directory,
            publication_id,
        )
        _remove_transaction_directory(transaction_directory)
        return result
    if pending_directory.exists() or pending_directory.is_symlink():
        _remove_generation(
            pending_directory,
            descriptors,
            archive_descriptors,
        )
    if generation_directory.exists() or generation_directory.is_symlink():
        _remove_generation(
            generation_directory,
            descriptors,
            archive_descriptors,
        )
    _cleanup_publication_temps(
        proof_directory,
        publication_id,
    )
    _remove_transaction_directory(transaction_directory)
    return None


def recover_artifact_publication(
    proof_directory: Path,
    state_directory: Path,
    publication_id: str,
) -> PublicationResult | None:
    """Recover a dead publisher: roll back pre-commit or finish post-commit."""

    publication_id = _safe_component(publication_id, "publication_id")
    proof_directory = _absolute(proof_directory)
    state_directory = _absolute(state_directory)
    _ensure_directory(proof_directory)
    _ensure_directory(state_directory)
    with _publication_lock(proof_directory, publication_id):
        return _recover_locked(
            proof_directory=proof_directory,
            state_directory=state_directory,
            publication_id=publication_id,
        )


def publish_artifact_set(
    request: PublicationRequest,
    *,
    fault_injector: FaultInjector | None = None,
) -> PublicationResult:
    """Durably publish one exact typed artifact set after provider teardown."""

    (
        publication_id,
        publication_type,
        profile_id,
        proof_directory,
        state_directory,
        artifacts,
        descriptors,
        archive_payloads,
        archive_descriptors,
        source_screen,
    ) = _validate_request(request)
    set_digest = _set_digest(
        publication_id,
        publication_type,
        profile_id,
        descriptors,
        source_screen,
        archive_descriptors,
    )
    generation_name = f"{publication_id}.{set_digest.removeprefix('sha256:')}"
    pending_name = f".pending-{publication_id}-{secrets.token_hex(8)}"
    journal = _journal(
        publication_id=publication_id,
        publication_type=publication_type,
        profile_id=profile_id,
        proof_directory=proof_directory,
        state_directory=state_directory,
        pending_name=pending_name,
        generation_name=generation_name,
        set_digest=set_digest,
        descriptors=descriptors,
        archive_descriptors=archive_descriptors,
        source_screen=source_screen,
    )
    manifest_path = _manifest_path(proof_directory, publication_id)
    generation_root = proof_directory / _GENERATION_DIRECTORY
    pending_directory = generation_root / pending_name
    generation_directory = generation_root / generation_name
    _, transaction_directory = _state_paths(state_directory, publication_id)
    with _publication_lock(proof_directory, publication_id):
        recovered = _recover_locked(
            proof_directory=proof_directory,
            state_directory=state_directory,
            publication_id=publication_id,
        )
        if recovered is not None:
            expected_manifest = _manifest_payload(journal)
            if (
                _read_regular_file(
                    recovered.manifest_path,
                    "artifact-set commit manifest",
                    maximum_bytes=1024 * 1024,
                )
                != expected_manifest
            ):
                raise ArtifactPublicationError(
                    "publication id is already committed to different artifacts"
                )
            return recovered
        _ensure_directory(generation_root)
        try:
            transaction_directory.mkdir(mode=0o700)
        except FileExistsError as error:
            raise ArtifactPublicationError(
                "publication transaction could not be initialized"
            ) from error
        _fsync_directory(transaction_directory)
        _fsync_directory(transaction_directory.parent)
        try:
            _atomic_write(
                transaction_directory / "journal.json",
                _publication_canonical_bytes(
                    journal,
                    "artifact publication journal",
                )
                + b"\n",
            )
            if fault_injector is not None:
                fault_injector("after_journal_fsync")
            _write_generation(
                pending_directory,
                artifacts,
                descriptors,
                archive_payloads,
                archive_descriptors,
                fault_injector,
            )
            _install_generation(
                pending_directory,
                generation_directory,
                descriptors,
                archive_descriptors,
            )
            if fault_injector is not None:
                fault_injector("after_generation_rename")
                fault_injector("before_manifest_commit")
            if source_screen is not None:
                verified_source = load_committed_artifact_set(
                    proof_directory,
                    source_screen["publication_id"],
                    expected_profile_id=profile_id,
                    expected_publication_type="screen",
                )
                if (
                    verified_source.artifact_set_manifest_digest
                    != source_screen["artifact_set_manifest_digest"]
                ):
                    raise ArtifactPublicationError(
                        "pilot source_screen changed before manifest commit"
                    )
            _atomic_install_once(manifest_path, _manifest_payload(journal))
            if fault_injector is not None:
                fault_injector("after_manifest_commit")
            result = load_committed_artifact_set(
                proof_directory,
                publication_id,
                expected_profile_id=profile_id,
                expected_publication_type=publication_type,
            )
            _remove_transaction_directory(transaction_directory)
            if fault_injector is not None:
                fault_injector("after_transaction_cleanup")
            return result
        except BaseException as error:
            if manifest_path.exists() or manifest_path.is_symlink():
                try:
                    result = load_committed_artifact_set(
                        proof_directory,
                        publication_id,
                        expected_profile_id=profile_id,
                        expected_publication_type=publication_type,
                    )
                    if _read_regular_file(
                        result.manifest_path,
                        "artifact-set commit manifest",
                        maximum_bytes=1024 * 1024,
                    ) != _manifest_payload(journal):
                        raise ArtifactPublicationError(
                            "commit manifest changed during publication recovery"
                        )
                    if transaction_directory.exists() and not transaction_directory.is_symlink():
                        _remove_transaction_directory(transaction_directory)
                    return result
                except BaseException as recovery_error:
                    raise ArtifactPublicationError(
                        "artifact set committed but post-commit recovery failed"
                    ) from recovery_error
            try:
                if pending_directory.exists() or pending_directory.is_symlink():
                    _remove_generation(
                        pending_directory,
                        descriptors,
                        archive_descriptors,
                    )
                if generation_directory.exists() or generation_directory.is_symlink():
                    _remove_generation(
                        generation_directory,
                        descriptors,
                        archive_descriptors,
                    )
                _cleanup_publication_temps(
                    proof_directory,
                    publication_id,
                )
                _remove_transaction_directory(transaction_directory)
            except BaseException as rollback_error:
                raise ArtifactPublicationError(
                    "artifact-set publication failed and rollback is incomplete"
                ) from rollback_error
            raise error


def _request_from_plan(path: Path) -> PublicationRequest:
    payload = _read_regular_file(path, "publication plan", maximum_bytes=1024 * 1024)
    value = _json_object(payload, "publication plan")
    expected_keys = {
        "schema_version",
        "publication_id",
        "publication_type",
        "profile_id",
        "proof_directory",
        "state_directory",
        "teardown_confirmed",
        "artifacts",
        "source_screen",
        "prior_profile_id",
        "prior_receipts",
    }
    if set(value) != expected_keys or value["schema_version"] != SCHEMA_VERSION:
        raise ArtifactPublicationError("publication plan shape is invalid")
    if not isinstance(value["artifacts"], dict) or not isinstance(value["prior_receipts"], dict):
        raise ArtifactPublicationError("publication plan artifact paths are invalid")
    if not all(isinstance(item, str) for item in value["artifacts"].values()):
        raise ArtifactPublicationError("publication plan artifact paths are invalid")
    if not all(isinstance(item, str) for item in value["prior_receipts"].values()):
        raise ArtifactPublicationError("publication plan prior receipt paths are invalid")
    source_screen = value["source_screen"]
    if source_screen is not None and (
        not isinstance(source_screen, dict)
        or set(source_screen) != {"publication_id", "artifact_set_manifest_digest"}
        or not all(isinstance(item, str) for item in source_screen.values())
    ):
        raise ArtifactPublicationError("publication plan source_screen is invalid")
    return PublicationRequest(
        publication_id=value["publication_id"],
        publication_type=value["publication_type"],
        profile_id=value["profile_id"],
        proof_directory=Path(value["proof_directory"]),
        state_directory=Path(value["state_directory"]),
        artifacts={key: Path(item) for key, item in value["artifacts"].items()},
        teardown_confirmed=value["teardown_confirmed"],
        source_screen_publication_id=(
            source_screen["publication_id"] if source_screen is not None else None
        ),
        source_screen_manifest_digest=(
            source_screen["artifact_set_manifest_digest"] if source_screen is not None else None
        ),
        prior_profile_id=value["prior_profile_id"],
        prior_receipts={key: Path(item) for key, item in value["prior_receipts"].items()},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish or verify a crash-safe typed RunPod artifact set."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--plan", type=Path, required=True)
    recover = commands.add_parser("recover")
    recover.add_argument("--proof-directory", type=Path, required=True)
    recover.add_argument("--state-directory", type=Path, required=True)
    recover.add_argument("--publication-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--proof-directory", type=Path, required=True)
    verify.add_argument("--publication-id", required=True)
    verify.add_argument("--expected-profile-id")
    verify.add_argument("--expected-publication-type", choices=PUBLICATION_TYPES)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.command == "publish":
            result = publish_artifact_set(_request_from_plan(args.plan))
            payload: dict[str, Any] = result.as_dict()
        elif args.command == "recover":
            recovered = recover_artifact_publication(
                args.proof_directory,
                args.state_directory,
                args.publication_id,
            )
            payload = (
                {
                    "publication_id": args.publication_id,
                    "committed": False,
                    "recovered": True,
                }
                if recovered is None
                else {**recovered.as_dict(), "recovered": True}
            )
        else:
            result = load_committed_artifact_set(
                args.proof_directory,
                args.publication_id,
                expected_profile_id=args.expected_profile_id,
                expected_publication_type=args.expected_publication_type,
            )
            payload = {**result.as_dict(), "verified": True}
    except (ArtifactPublicationError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(
        _publication_canonical_bytes(payload, "publication CLI response") + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
