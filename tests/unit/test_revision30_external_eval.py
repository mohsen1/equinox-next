from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from research.runpod.external_eval_transport import sha256_file
from research.runpod.revision30_external_eval import (
    ADAPTER_SET,
    FROZEN_OBJECTIVE_ID,
    FROZEN_WORKLOAD_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PACK_ID,
    STUDY_ID,
    canonical_json,
    safe_extract_adapter,
    validate_evaluation_manifest,
    verify_external_pack,
)


def adapter_manifest(files: dict[str, bytes]) -> dict:
    content = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "workload_revision": FROZEN_WORKLOAD_REVISION,
        "objective_id": FROZEN_OBJECTIVE_ID,
        "files": [
            {
                "path": path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(files.items())
        ],
    }
    return {
        **content,
        "digest": "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest(),
    }


def write_archive(
    path: Path,
    files: dict[str, bytes],
    manifest: dict,
    *,
    extra_members: dict[str, bytes] | None = None,
) -> None:
    members = {
        "adapter/adapter-manifest.json": canonical_json(manifest),
        **{f"adapter/{name}": payload for name, payload in files.items()},
        **(extra_members or {}),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def evaluation_manifest() -> dict:
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
                "adapter_id": "reference-seed113",
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


def test_evaluation_manifest_is_bound_to_frozen_study_and_unique_sources() -> None:
    manifest = evaluation_manifest()
    assert validate_evaluation_manifest(manifest) == manifest

    duplicate = {**manifest, "adapters": [*manifest["adapters"], dict(manifest["adapters"][0])]}
    duplicate["adapters"][1]["adapter_id"] = "another-adapter"
    duplicate["adapters"][1]["archive_filename"] = f"adapter-{'d' * 64}.tgz"
    duplicate["adapters"][1]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="evidence is incomplete"):
        validate_evaluation_manifest(duplicate)

    wrong_pack = {**manifest, "pack_id": "replacement-pack"}
    with pytest.raises(ValueError, match="wrong task pack"):
        validate_evaluation_manifest(wrong_pack)


def test_adapter_extraction_verifies_every_byte_and_frozen_revision(tmp_path: Path) -> None:
    files = {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
        "README.md": b"evidence",
    }
    manifest = adapter_manifest(files)
    archive = tmp_path / "adapter.tgz"
    write_archive(archive, files, manifest)

    extracted = safe_extract_adapter(
        archive,
        tmp_path / "extracted",
        expected_archive_sha256=sha256_file(archive),
        expected_archive_size=archive.stat().st_size,
        expected_manifest_digest=manifest["digest"],
    )

    assert extracted == manifest
    assert (tmp_path / "extracted/adapter_model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize(
    "extra_members",
    (
        {"../escape": b"unsafe"},
        {"adapter/undeclared.txt": b"not in the manifest"},
    ),
)
def test_adapter_extraction_rejects_unsafe_or_undeclared_members(
    tmp_path: Path,
    extra_members: dict[str, bytes],
) -> None:
    files = {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"weights",
    }
    manifest = adapter_manifest(files)
    archive = tmp_path / "adapter.tgz"
    write_archive(archive, files, manifest, extra_members=extra_members)

    with pytest.raises(RuntimeError):
        safe_extract_adapter(
            archive,
            tmp_path / "extracted",
            expected_archive_sha256=sha256_file(archive),
            expected_archive_size=archive.stat().st_size,
            expected_manifest_digest=manifest["digest"],
        )
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "extracted.pending").exists()
    assert not (tmp_path / "escape").exists()


def test_external_evaluator_accepts_only_the_frozen_pack() -> None:
    observed = verify_external_pack(Path.cwd())
    expected = json.loads(
        Path("research/frozen/revision30-external-pack.json").read_text(encoding="utf-8")
    )
    assert observed == expected
