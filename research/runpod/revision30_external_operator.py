"""Select and launch the sealed external evaluation for the revision-30 study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - macOS operator compatibility.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from research.runpod.external_eval_transport import (
    MAXIMUM_ADAPTER_BYTES,
    load_input_manifest,
    sha256_file,
)
from research.runpod.revision30_external_eval import (
    ADAPTER_ROLES,
    ADAPTER_SET,
    FROZEN_OBJECTIVE_ID,
    FROZEN_WORKLOAD_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PACK_ID,
    STUDY_ID,
    canonical_json,
    validate_evaluation_manifest,
)
from research.runpod.study_operator import (
    append_event,
    artifact_record,
    load_manifest,
    matching_success_results,
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_evaluation_id(now: datetime | None = None) -> str:
    timestamp = now or datetime.now(UTC)
    return "revision30-external-" + timestamp.astimezone(UTC).strftime("%Y%m%dt%H%M%Sz")


def execution_id_from_result_path(path: Path) -> str:
    suffix = ".result.json"
    if not path.name.endswith(suffix):
        raise ValueError("source result has an invalid filename")
    execution_id = path.name[: -len(suffix)]
    if not execution_id.startswith("runpod-proof-"):
        raise ValueError("source result has an invalid execution identity")
    return execution_id


def retained_conditions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = [
        condition
        for condition in manifest["conditions"]
        if isinstance(condition, dict) and condition.get("role") in ADAPTER_ROLES
    ]
    roles = [condition["role"] for condition in conditions]
    if (
        len(conditions) != 5
        or roles.count("frozen_reference") != 1
        or roles.count("fresh_replication") != 2
        or roles.count("matched_frozen_policy_control") != 1
        or roles.count("branch_width_ablation") != 1
    ):
        raise RuntimeError("the retained revision-30 adapter roster is incomplete")
    return conditions


def source_result_path(
    condition: dict[str, Any],
    receipt_directory: Path,
) -> Path:
    declared_execution_id = condition.get("execution_id")
    if isinstance(declared_execution_id, str):
        path = receipt_directory / f"{declared_execution_id}.result.json"
        if not path.is_file():
            raise RuntimeError(f"{condition['condition_id']} has no source result")
        return path
    matches = matching_success_results(receipt_directory, condition)
    if len(matches) != 1:
        raise RuntimeError(
            f"{condition['condition_id']} requires exactly one successful source result"
        )
    return matches[0][0]


def verified_archive_manifest(
    archive_path: Path,
    expected_manifest_digest: str,
) -> dict[str, Any]:
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size < 1
        or archive_path.stat().st_size > MAXIMUM_ADAPTER_BYTES
    ):
        raise RuntimeError("retained adapter archive size is invalid")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        member_by_name = {}
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "adapter"
                or not (member.isdir() or member.isfile())
                or member.name in member_by_name
            ):
                raise RuntimeError("retained adapter archive contains an unsafe member")
            member_by_name[member.name] = member
        manifest_member = member_by_name.get("adapter/adapter-manifest.json")
        if manifest_member is None or not manifest_member.isfile():
            raise RuntimeError("retained adapter archive omits its manifest")
        manifest_source = archive.extractfile(manifest_member)
        if manifest_source is None:
            raise RuntimeError("retained adapter manifest could not be read")
        manifest = json.load(manifest_source)
        if not isinstance(manifest, dict):
            raise RuntimeError("retained adapter manifest is invalid")
        content = {key: value for key, value in manifest.items() if key != "digest"}
        observed_manifest_digest = "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()
        if (
            manifest.get("digest") != expected_manifest_digest
            or observed_manifest_digest != expected_manifest_digest
            or manifest.get("schema_version") != 1
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("workload_revision") != FROZEN_WORKLOAD_REVISION
            or manifest.get("objective_id") != FROZEN_OBJECTIVE_ID
        ):
            raise RuntimeError("retained adapter manifest identity does not match revision 30")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError("retained adapter manifest has no files")
        declared_paths = set()
        declared_bytes = 0
        for item in files:
            relative_path = item.get("path") if isinstance(item, dict) else None
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
                or relative_path in declared_paths
                or isinstance(item.get("size_bytes"), bool)
                or not isinstance(item.get("size_bytes"), int)
                or item["size_bytes"] < 0
                or not isinstance(item.get("sha256"), str)
            ):
                raise RuntimeError("retained adapter file evidence is invalid")
            declared_paths.add(relative_path)
            declared_bytes += item["size_bytes"]
            if declared_bytes > MAXIMUM_ADAPTER_BYTES:
                raise RuntimeError("retained adapter contents exceed the byte ceiling")
            member = member_by_name.get(f"adapter/{relative_path}")
            if member is None or not member.isfile() or member.size != item["size_bytes"]:
                raise RuntimeError("retained adapter file size evidence does not match")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("retained adapter file could not be read")
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError("retained adapter file digest does not match")
        observed_files = {
            str(Path(name).relative_to("adapter"))
            for name, member in member_by_name.items()
            if member.isfile()
        }
        if observed_files != declared_paths | {"adapter-manifest.json"}:
            raise RuntimeError("retained adapter archive and manifest file sets differ")
        if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(declared_paths):
            raise RuntimeError("retained adapter archive omits required model files")
        return manifest


def adapter_record(
    condition: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("retained adapter source result is invalid")
    execution_id = execution_id_from_result_path(result_path)
    expected_seed = condition["optimization_seed"]
    if (
        result.get("experiment_completed") is not True
        or result.get("final_evaluation_complete") is not True
        or result.get("adapter_persisted") is not True
        or result.get("workload_revision") != FROZEN_WORKLOAD_REVISION
        or result.get("objective_id") != FROZEN_OBJECTIVE_ID
        or result.get("model_id") != MODEL_ID
        or result.get("model_revision") != MODEL_REVISION
        or result.get("seed") != expected_seed
        or result.get("branch_width") != condition["branch_width"]
    ):
        raise RuntimeError(f"{condition['condition_id']} source result does not match")
    if condition["role"] != "frozen_reference":
        study = result.get("study")
        expected_condition = str(condition["condition_id"]).rsplit("_seed", 1)[0]
        if (
            not isinstance(study, dict)
            or study.get("study_id") != STUDY_ID
            or study.get("condition") != expected_condition
            or study.get("optimization_seed") != expected_seed
            or study.get("validation_seed_base") != condition["validation_seed_base"]
            or study.get("test_seed_base") != condition["test_seed_base"]
            or study.get("policy_mutation_enabled") != condition["policy_mutation_enabled"]
        ):
            raise RuntimeError(f"{condition['condition_id']} study evidence does not match")
    adapter_manifest = result.get("adapter_manifest")
    expected_manifest_digest = (
        adapter_manifest.get("digest") if isinstance(adapter_manifest, dict) else None
    )
    if not isinstance(expected_manifest_digest, str):
        raise RuntimeError("retained adapter source result omits its manifest digest")
    archive_path = result_path.with_name(f"{execution_id}.adapter.tgz")
    observed_manifest = verified_archive_manifest(
        archive_path,
        expected_manifest_digest,
    )
    if observed_manifest != adapter_manifest:
        raise RuntimeError("retained adapter result and archive manifests differ")
    archive_sha256 = sha256_file(archive_path)
    return {
        "adapter_id": condition["condition_id"],
        "condition_id": condition["condition_id"],
        "role": condition["role"],
        "optimization_seed": expected_seed,
        "source_execution_id": execution_id,
        "source_result_sha256": sha256_file(result_path),
        "archive_filename": f"adapter-{archive_sha256}.tgz",
        "size_bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "adapter_manifest_digest": expected_manifest_digest,
        "_source_archive_path": str(archive_path),
    }


def build_input_manifest(
    repository_root: Path,
    evaluation_id: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    study_manifest = load_manifest(
        repository_root / "research/studies/revision30-confirmatory-study.json"
    )
    receipt_directory = repository_root / "var/research-proofs"
    records = []
    source_archives = {}
    for condition in retained_conditions(study_manifest):
        record = adapter_record(
            condition,
            source_result_path(condition, receipt_directory),
        )
        source_archives[record["archive_filename"]] = Path(record.pop("_source_archive_path"))
        records.append(record)
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "study_id": STUDY_ID,
        "adapter_set": ADAPTER_SET,
        "pack_id": PACK_ID,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "adapters": records,
    }
    validate_evaluation_manifest(load_input_manifest_from_value(manifest))
    if len(source_archives) != len(records):
        raise RuntimeError("retained adapter archives are not content-unique")
    return manifest, source_archives


def load_input_manifest_from_value(manifest: dict[str, Any]) -> dict[str, Any]:
    """Apply the transport schema without weakening its file-backed implementation."""
    # A private short-lived file keeps launch validation identical to remote acceptance.
    with tempfile.TemporaryDirectory(prefix="equinox-external-manifest-") as directory:
        path = Path(directory) / "manifest.json"
        path.write_bytes(canonical_json(manifest))
        return load_input_manifest(path)


def write_manifest_durably(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("wb") as handle:
        handle.write(canonical_json(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@contextmanager
def external_evaluation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another external evaluation is already running") from error
        yield


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    state_directory = repository_root / "var/revision30-study"
    evaluation_id = new_evaluation_id()
    manifest, source_archives = build_input_manifest(repository_root, evaluation_id)
    manifest_path = state_directory / f"{evaluation_id}.input.json"
    write_manifest_durably(manifest_path, manifest)
    environment = {
        **os.environ,
        "EQUINOX_EXTERNAL_EVALUATION_ID": evaluation_id,
        "EQUINOX_EXTERNAL_MANIFEST_PATH": str(manifest_path),
        "EQUINOX_EXTERNAL_ARCHIVE_MAP": json.dumps(
            {key: str(value) for key, value in source_archives.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if arguments.preflight_only:
        environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] = "1"
        completed = subprocess.run(
            [str(repository_root / "scripts/runpod-external-eval")],
            cwd=repository_root,
            env=environment,
            check=False,
        )
        raise SystemExit(completed.returncode)
    ledger_path = state_directory / "attempts.jsonl"
    with external_evaluation_lock(state_directory / "external-evaluation.lock"):
        started_at = utc_now()
        append_event(
            ledger_path,
            {
                "schema_version": 1,
                "event": "external_evaluation_attempt_started",
                "study_id": STUDY_ID,
                "evaluation_id": evaluation_id,
                "started_at": started_at,
                "adapter_count": len(manifest["adapters"]),
                "manifest": artifact_record(manifest_path),
            },
        )
        completed = subprocess.run(
            [str(repository_root / "scripts/runpod-external-eval")],
            cwd=repository_root,
            env=environment,
            check=False,
        )
        append_event(
            ledger_path,
            {
                "schema_version": 1,
                "event": "external_evaluation_attempt_finished",
                "study_id": STUDY_ID,
                "evaluation_id": evaluation_id,
                "started_at": started_at,
                "finished_at": utc_now(),
                "exit_code": completed.returncode,
                "outcome": "passed" if completed.returncode == 0 else "failed",
                "manifest": artifact_record(manifest_path),
            },
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
