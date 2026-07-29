"""Deterministic, volume-staged workload bundles for larger-model RunPod jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import stat
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from research.runpod.bootstrap_server import (
    MAXIMUM_BUNDLE_BYTES,
    expected_bundle_files,
    install_bundle,
)
from research.runpod.larger_model_gate import (
    GateError,
    canonical_json,
    expected_source_contract_digest,
    load_manifest,
    verify_source_contract,
)

SCREEN_WORKLOAD_FILE = "repository_repair_large_model_eligibility.py"
PILOT_WORKLOAD_FILE = "repository_repair_large_model_pilot.py"
CANONICAL_WORKLOAD_FILES = (SCREEN_WORKLOAD_FILE, PILOT_WORKLOAD_FILE)
BUNDLE_COMPRESSION = "xz"
BUNDLE_HANDOFF_REVISION = "runpod-volume-bundle-handoff@1"
BUNDLE_LOGICAL_ROOT = Path("/workspace/equinox-state/workload-bundles")
PROFILE_MEMBER = "larger-model-eligibility.json"
_PROFILE_COMPONENT = re.compile(r"^[A-Za-z0-9._@-]+$")


class BundleError(ValueError):
    """Raised when a workload bundle or staging receipt is unsafe."""


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return _sha256(canonical_json(manifest))


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    content = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    return _sha256(canonical_json(content))


def canonical_bundle_files() -> tuple[str, ...]:
    """Return the common exact allowlist shared by screen and pilot workloads."""

    screen_files = expected_bundle_files(SCREEN_WORKLOAD_FILE)
    pilot_files = expected_bundle_files(PILOT_WORKLOAD_FILE)
    if screen_files != pilot_files:
        raise BundleError("screen and pilot workload bundle allowlists diverged")
    return tuple(sorted(screen_files))


def _bundle_sources(repository_root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for member in canonical_bundle_files():
        if member == PROFILE_MEMBER:
            source = repository_root / "research/studies" / member
        else:
            source = repository_root / "research/runpod" / member
        if not source.is_file():
            raise BundleError(f"bundle source is unavailable: {member}")
        sources[member] = source
    return sources


def _deterministic_tar(sources: Mapping[str, Path]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for member, source in sorted(sources.items()):
            content = source.read_bytes()
            metadata = tarfile.TarInfo(member)
            metadata.size = len(content)
            metadata.mode = 0o755 if source.suffix == ".sh" else 0o644
            metadata.mtime = 0
            metadata.uid = 0
            metadata.gid = 0
            metadata.uname = ""
            metadata.gname = ""
            archive.addfile(metadata, BytesIO(content))
    return output.getvalue()


def build_larger_model_bundle(
    repository_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Build the canonical screen/pilot bundle with byte-stable metadata."""

    repository_root = repository_root.resolve()
    source_root = repository_root / "research/runpod"
    verify_source_contract(manifest, source_root)
    profile_path = repository_root / "research/studies" / PROFILE_MEMBER
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError("larger-model profile could not be read") from error
    if profile != manifest:
        raise BundleError("larger-model profile does not match the immutable manifest")
    raw_tar = _deterministic_tar(_bundle_sources(repository_root))
    payload = lzma.compress(
        raw_tar,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=6,
    )
    if not payload or len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise BundleError("canonical workload bundle size is invalid")
    digest = _sha256(payload)
    metadata = {
        "profile_id": manifest["profile_id"],
        "manifest_digest": _manifest_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
        "bundle_digest": digest,
        "bundle_size_bytes": len(payload),
        "bundle_compression": BUNDLE_COMPRESSION,
        "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
        "bundle_path": logical_bundle_path(manifest["profile_id"], digest),
        "bundle_files": list(canonical_bundle_files()),
        "workload_files": list(CANONICAL_WORKLOAD_FILES),
    }
    return payload, metadata


def logical_bundle_path(profile_id: str, bundle_digest: str) -> str:
    """Return the one content-addressed path accepted by the worker bootstrap."""

    if not isinstance(profile_id, str) or not _PROFILE_COMPONENT.fullmatch(profile_id):
        raise BundleError("profile id is not safe for a volume path")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", bundle_digest):
        raise BundleError("bundle digest is invalid")
    return str(BUNDLE_LOGICAL_ROOT / profile_id / f"{bundle_digest[7:]}.tar.xz")


def inspect_larger_model_bundle(
    payload: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify compression, exact allowlist, profile, and pinned source content."""

    if not payload or len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise BundleError("canonical workload bundle size is invalid")
    with tempfile.TemporaryDirectory(prefix="equinox-bundle-inspect-") as directory:
        extracted = Path(directory)
        try:
            install_bundle(
                payload,
                work_directory=extracted,
                workload_file=SCREEN_WORKLOAD_FILE,
            )
        except (OSError, tarfile.TarError, ValueError) as error:
            raise BundleError("canonical workload bundle failed its allowlist") from error
        try:
            profile = json.loads((extracted / PROFILE_MEMBER).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BundleError("canonical workload bundle profile is invalid") from error
        if profile != manifest:
            raise BundleError("canonical workload bundle profile does not match")
        try:
            verify_source_contract(manifest, extracted)
        except GateError as error:
            raise BundleError("canonical workload bundle source contract does not match") from error
    digest = _sha256(payload)
    return {
        "bundle_digest": digest,
        "bundle_size_bytes": len(payload),
        "bundle_compression": BUNDLE_COMPRESSION,
        "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
        "bundle_path": logical_bundle_path(manifest["profile_id"], digest),
        "bundle_files": list(canonical_bundle_files()),
        "workload_files": list(CANONICAL_WORKLOAD_FILES),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes, *, mode: int) -> None:
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
        _fsync_directory(path.parent)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _read_existing_bundle(destination: Path, expected_size: int) -> bytes:
    try:
        path_metadata = destination.lstat()
    except OSError as error:
        raise BundleError("content-addressed bundle is unavailable") from error
    if not stat.S_ISREG(path_metadata.st_mode):
        raise BundleError("content-addressed bundle is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(destination, flags)
    except OSError as error:
        raise BundleError("content-addressed bundle is not a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != expected_size
            or metadata.st_dev != path_metadata.st_dev
            or metadata.st_ino != path_metadata.st_ino
        ):
            raise BundleError("content-addressed bundle metadata does not match")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            observed = handle.read(expected_size + 1)
    finally:
        os.close(descriptor)
    if len(observed) != expected_size:
        raise BundleError("content-addressed bundle size does not match")
    return observed


def _stage_content_addressed(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, pending_name = tempfile.mkstemp(
        prefix=".pending-bundle-",
        dir=destination.parent,
    )
    pending = Path(pending_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(pending, destination)
        except FileExistsError:
            observed = _read_existing_bundle(destination, len(payload))
            if observed != payload:
                raise BundleError(
                    "content-addressed bundle path contains different bytes"
                ) from None
        _fsync_directory(destination.parent)
    finally:
        pending.unlink(missing_ok=True)
    try:
        observed_mode = stat.S_IMODE(destination.lstat().st_mode)
        if observed_mode & 0o222:
            destination.chmod(observed_mode & ~0o222)
            _fsync_directory(destination.parent)
    except OSError:
        pass
    if _read_existing_bundle(destination, len(payload)) != payload:
        raise BundleError("staged bundle failed its post-write verification")


def stage_bundle_on_mounted_volume(
    payload: bytes,
    manifest: Mapping[str, Any],
    *,
    mount_root: Path,
    volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically stage a verified bundle on a mounted `/workspace` volume."""

    evidence = inspect_larger_model_bundle(payload, manifest)
    if (
        not isinstance(volume_id, str)
        or not volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb < manifest["hardware"]["volume_disk_gb"]
    ):
        raise BundleError("staging volume identity or size is invalid")
    if not mount_root.is_absolute():
        raise BundleError("mounted volume root must be absolute")
    logical_path = Path(evidence["bundle_path"])
    relative_path = logical_path.relative_to("/workspace")
    destination = mount_root / relative_path
    _stage_content_addressed(destination, payload)
    staged_at = now or datetime.now(UTC)
    if staged_at.tzinfo is None or staged_at.utcoffset() is None:
        raise BundleError("bundle staging time must be timezone-aware")
    receipt = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "manifest_digest": _manifest_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
        "network_volume_id": volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
        **evidence,
        "staged_at": staged_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ready": True,
    }
    receipt["receipt_digest"] = _receipt_digest(receipt)
    return receipt


def verify_bundle_stage_receipt(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    provider_volume: Any,
    *,
    expected_bundle_digest: str,
    expected_bundle_size_bytes: int,
    expected_bundle_path: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify local staging evidence against the profile and current provider volume."""

    required_keys = {
        "schema_version",
        "profile_id",
        "manifest_digest",
        "source_contract_digest",
        "network_volume_id",
        "network_volume_data_center_id",
        "network_volume_size_gb",
        "bundle_digest",
        "bundle_size_bytes",
        "bundle_compression",
        "bundle_handoff_revision",
        "bundle_path",
        "bundle_files",
        "workload_files",
        "staged_at",
        "ready",
        "receipt_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required_keys:
        raise BundleError("bundle stage receipt has an invalid field set")
    exact_identity = {
        "schema_version": 1,
        "profile_id": manifest["profile_id"],
        "manifest_digest": _manifest_digest(manifest),
        "source_contract_digest": expected_source_contract_digest(manifest),
        "bundle_digest": expected_bundle_digest,
        "bundle_size_bytes": expected_bundle_size_bytes,
        "bundle_compression": BUNDLE_COMPRESSION,
        "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
        "bundle_path": expected_bundle_path,
        "bundle_files": list(canonical_bundle_files()),
        "workload_files": list(CANONICAL_WORKLOAD_FILES),
        "ready": True,
    }
    for key, expected in exact_identity.items():
        if receipt.get(key) != expected:
            raise BundleError(f"bundle stage receipt {key} does not match")
    if expected_bundle_path != logical_bundle_path(
        manifest["profile_id"],
        expected_bundle_digest,
    ):
        raise BundleError("expected bundle path is not content-addressed")
    if (
        type(expected_bundle_size_bytes) is not int
        or not 0 < expected_bundle_size_bytes <= MAXIMUM_BUNDLE_BYTES
    ):
        raise BundleError("expected bundle size is invalid")
    if receipt.get("receipt_digest") != _receipt_digest(receipt):
        raise BundleError("bundle stage receipt digest is invalid")
    staged_at_value = receipt.get("staged_at")
    if not isinstance(staged_at_value, str) or not staged_at_value.endswith("Z"):
        raise BundleError("bundle stage receipt timestamp is invalid")
    try:
        staged_at = datetime.fromisoformat(staged_at_value[:-1] + "+00:00")
    except ValueError as error:
        raise BundleError("bundle stage receipt timestamp is invalid") from error
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise BundleError("bundle stage receipt verification time must be timezone-aware")
    current_time = current_time.astimezone(UTC)
    age_seconds = (current_time - staged_at).total_seconds()
    if age_seconds < 0:
        raise BundleError("bundle stage receipt timestamp is in the future")
    if age_seconds > 24 * 3_600:
        raise BundleError("bundle stage receipt is not fresh")
    try:
        payload = (
            json.loads(provider_volume)
            if isinstance(provider_volume, str | bytes | bytearray)
            else provider_volume
        )
    except json.JSONDecodeError as error:
        raise BundleError("RunPod network volume is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise BundleError("RunPod network volume must be an object")
    volume = payload.get("networkVolume", payload)
    if not isinstance(volume, Mapping):
        raise BundleError("RunPod network volume payload is invalid")
    provider_id = volume.get("id", volume.get("networkVolumeId"))
    provider_data_center = volume.get("dataCenterId")
    provider_size = volume.get("size")
    if (
        provider_id != receipt.get("network_volume_id")
        or provider_data_center != receipt.get("network_volume_data_center_id")
        or type(provider_size) is not int
        or type(receipt.get("network_volume_size_gb")) is not int
        or provider_size < receipt["network_volume_size_gb"]
        or provider_size < manifest["hardware"]["volume_disk_gb"]
    ):
        raise BundleError("RunPod network volume no longer matches the bundle stage receipt")
    return {
        "profile_id": manifest["profile_id"],
        "network_volume_id": provider_id,
        "network_volume_data_center_id": provider_data_center,
        "network_volume_size_gb": provider_size,
        "bundle_digest": receipt["bundle_digest"],
        "bundle_size_bytes": receipt["bundle_size_bytes"],
        "bundle_compression": receipt["bundle_compression"],
        "bundle_handoff_revision": receipt["bundle_handoff_revision"],
        "bundle_path": receipt["bundle_path"],
        "receipt_digest": receipt["receipt_digest"],
    }


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        raw = sys.stdin.buffer.read() if str(path) == "-" else path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise BundleError(f"{name} must be a JSON object")
    return payload


def _write_bundle(path: Path, payload: bytes) -> None:
    _write_atomic(path, payload, mode=0o600)


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    _write_atomic(path, canonical_json(receipt) + b"\n", mode=0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--repository-root", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    stage_parser = subparsers.add_parser("stage-mounted")
    stage_parser.add_argument("--bundle", required=True, type=Path)
    stage_parser.add_argument("--bundle-digest", required=True)
    stage_parser.add_argument("--bundle-size-bytes", required=True, type=int)
    stage_parser.add_argument("--mount-root", required=True, type=Path)
    stage_parser.add_argument("--volume-id", required=True)
    stage_parser.add_argument("--data-center-id", required=True)
    stage_parser.add_argument("--volume-size-gb", required=True, type=int)
    stage_parser.add_argument("--receipt-output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify-stage-receipt")
    verify_parser.add_argument("receipt", type=Path)
    verify_parser.add_argument("provider_volume", type=Path)
    verify_parser.add_argument("--bundle-digest", required=True)
    verify_parser.add_argument("--bundle-size-bytes", required=True, type=int)
    verify_parser.add_argument("--bundle-path", required=True)
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == "build":
            payload, output = build_larger_model_bundle(
                arguments.repository_root,
                manifest,
            )
            _write_bundle(arguments.output, payload)
        elif arguments.command == "stage-mounted":
            payload = arguments.bundle.read_bytes()
            if _sha256(payload) != arguments.bundle_digest:
                raise BundleError("staging bundle digest does not match the operator input")
            if len(payload) != arguments.bundle_size_bytes:
                raise BundleError("staging bundle size does not match the operator input")
            output = stage_bundle_on_mounted_volume(
                payload,
                manifest,
                mount_root=arguments.mount_root,
                volume_id=arguments.volume_id,
                data_center_id=arguments.data_center_id,
                volume_size_gb=arguments.volume_size_gb,
            )
            _write_receipt(arguments.receipt_output, output)
        else:
            provider_raw = (
                sys.stdin.buffer.read()
                if str(arguments.provider_volume) == "-"
                else arguments.provider_volume.read_bytes()
            )
            output = verify_bundle_stage_receipt(
                manifest,
                _read_json_object(arguments.receipt, "bundle stage receipt"),
                provider_raw,
                expected_bundle_digest=arguments.bundle_digest,
                expected_bundle_size_bytes=arguments.bundle_size_bytes,
                expected_bundle_path=arguments.bundle_path,
            )
    except (BundleError, GateError, OSError) as error:
        parser.error(str(error))
    sys.stdout.buffer.write(canonical_json(output) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
