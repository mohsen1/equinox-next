from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research.runpod.larger_model_gate import canonical_json, load_manifest
from research.runpod.workload_bundle import (
    BUNDLE_HANDOFF_REVISION,
    BundleError,
    build_larger_model_bundle,
    canonical_bundle_files,
    stage_bundle_on_mounted_volume,
    verify_bundle_stage_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGING_OPERATOR = REPOSITORY_ROOT / "scripts/stage-runpod-workload-bundle"
NOW = datetime(2026, 7, 29, 16, 0, tzinfo=UTC)
VOLUME = {"id": "network-volume-123", "dataCenterId": "EU-RO-1", "size": 50}


def _build() -> tuple[bytes, dict[str, object]]:
    return build_larger_model_bundle(REPOSITORY_ROOT, load_manifest())


def _stage(
    tmp_path: Path,
    *,
    now: datetime = NOW,
) -> tuple[bytes, dict[str, object]]:
    payload, _ = _build()
    receipt = stage_bundle_on_mounted_volume(
        payload,
        load_manifest(),
        mount_root=tmp_path,
        volume_id="network-volume-123",
        data_center_id="EU-RO-1",
        volume_size_gb=50,
        now=now,
    )
    return payload, receipt


def _verify(
    receipt: dict[str, object],
    *,
    now: datetime = NOW,
    provider_volume: object = VOLUME,
) -> dict[str, object]:
    return verify_bundle_stage_receipt(
        load_manifest(),
        receipt,
        provider_volume,
        expected_bundle_digest=str(receipt["bundle_digest"]),
        expected_bundle_size_bytes=int(receipt["bundle_size_bytes"]),
        expected_bundle_path=str(receipt["bundle_path"]),
        now=now,
    )


def _redigest(receipt: dict[str, object]) -> None:
    content = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


def test_screen_and_pilot_share_one_normalized_deterministic_bundle(
    tmp_path: Path,
) -> None:
    first, first_metadata = _build()
    copied_root = tmp_path / "repository"
    (copied_root / "research").mkdir(parents=True)
    shutil.copytree(
        REPOSITORY_ROOT / "research/runpod",
        copied_root / "research/runpod",
    )
    shutil.copytree(
        REPOSITORY_ROOT / "research/studies",
        copied_root / "research/studies",
    )
    for path in (copied_root / "research").rglob("*"):
        if path.is_file():
            os.utime(path, (1_000_000_000, 1_000_000_000))
            path.chmod(0o777)

    second, second_metadata = build_larger_model_bundle(copied_root, load_manifest())

    assert first == second
    assert first_metadata == second_metadata
    assert first_metadata["bundle_handoff_revision"] == BUNDLE_HANDOFF_REVISION
    assert first_metadata["workload_files"] == [
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_pilot.py",
    ]
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:xz") as archive:
        members = archive.getmembers()
    member_names = [member.name for member in members]
    assert member_names == list(canonical_bundle_files())
    assert {
        "repository_repair_large_model_study.py",
        "repository_repair_large_model_trainer.py",
    } <= set(member_names)
    assert {
        "repository_repair_rl.py",
        "repository_repair_study.py",
    }.isdisjoint(member_names)
    assert all(member.uid == 0 and member.gid == 0 and member.mtime == 0 for member in members)
    assert all(member.mode in {0o644, 0o755} for member in members)


def test_stage_is_content_addressed_read_only_and_reusable(tmp_path: Path) -> None:
    payload, receipt = _stage(tmp_path)
    logical_path = Path(str(receipt["bundle_path"]))
    destination = tmp_path / logical_path.relative_to("/workspace")

    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    second = stage_bundle_on_mounted_volume(
        payload,
        load_manifest(),
        mount_root=tmp_path,
        volume_id="network-volume-123",
        data_center_id="EU-RO-1",
        volume_size_gb=50,
        now=NOW,
    )
    assert second == receipt
    verified = _verify(receipt)
    assert verified["bundle_digest"] == receipt["bundle_digest"]
    assert verified["bundle_handoff_revision"] == BUNDLE_HANDOFF_REVISION


def test_stage_refuses_symlink_or_different_existing_content(tmp_path: Path) -> None:
    payload, metadata = _build()
    destination = tmp_path / Path(str(metadata["bundle_path"])).relative_to("/workspace")
    destination.parent.mkdir(parents=True)
    target = tmp_path / "other"
    target.write_bytes(payload)
    destination.symlink_to(target)

    with pytest.raises(BundleError, match="regular file"):
        stage_bundle_on_mounted_volume(
            payload,
            load_manifest(),
            mount_root=tmp_path,
            volume_id="network-volume-123",
            data_center_id="EU-RO-1",
            volume_size_gb=50,
            now=NOW,
        )

    destination.unlink()
    destination.write_bytes(b"x" * len(payload))
    with pytest.raises(BundleError, match="different bytes"):
        stage_bundle_on_mounted_volume(
            payload,
            load_manifest(),
            mount_root=tmp_path,
            volume_id="network-volume-123",
            data_center_id="EU-RO-1",
            volume_size_gb=50,
            now=NOW,
        )


@pytest.mark.parametrize(
    ("timestamp", "now", "message"),
    [
        ("2026-07-29T16:00:00", NOW, "timestamp is invalid"),
        ("2026-07-29T16:00:01Z", NOW, "future"),
        ("2026-07-28T15:59:59Z", NOW, "not fresh"),
    ],
)
def test_stage_receipt_rejects_naive_future_or_older_than_24_hours(
    tmp_path: Path,
    timestamp: str,
    now: datetime,
    message: str,
) -> None:
    _, receipt = _stage(tmp_path)
    receipt["staged_at"] = timestamp
    _redigest(receipt)

    with pytest.raises(BundleError, match=message):
        _verify(receipt, now=now)


def test_stage_receipt_rejects_naive_verification_time(tmp_path: Path) -> None:
    _, receipt = _stage(tmp_path)

    with pytest.raises(BundleError, match="verification time must be timezone-aware"):
        _verify(receipt, now=datetime(2026, 7, 29, 16, 0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile_id", "other-profile@1", "profile_id"),
        ("bundle_digest", "sha256:" + "0" * 64, "bundle_digest"),
        ("bundle_size_bytes", 1, "bundle_size_bytes"),
        ("bundle_path", "/workspace/wrong.tar.xz", "bundle_path"),
        ("bundle_handoff_revision", "old@1", "handoff"),
        ("ready", False, "ready"),
    ],
)
def test_stage_receipt_fails_closed_on_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _, receipt = _stage(tmp_path)
    expected_digest = str(receipt["bundle_digest"])
    expected_size = int(receipt["bundle_size_bytes"])
    expected_path = str(receipt["bundle_path"])
    receipt[field] = value
    _redigest(receipt)

    with pytest.raises(BundleError, match=message):
        verify_bundle_stage_receipt(
            load_manifest(),
            receipt,
            VOLUME,
            expected_bundle_digest=expected_digest,
            expected_bundle_size_bytes=expected_size,
            expected_bundle_path=expected_path,
            now=NOW,
        )


def test_stage_receipt_is_bound_to_current_provider_volume(tmp_path: Path) -> None:
    _, receipt = _stage(tmp_path)

    with pytest.raises(BundleError, match="provider volume|network volume"):
        _verify(
            receipt,
            provider_volume={
                "id": "network-volume-other",
                "dataCenterId": "EU-RO-1",
                "size": 50,
            },
        )


def test_staging_operator_builds_and_installs_without_allocating_provider_resources(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "workload.tar.xz"
    build = subprocess.run(
        [
            str(STAGING_OPERATOR),
            "build",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(bundle),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    metadata = json.loads(build.stdout)
    receipt_path = tmp_path / "stage-receipt.json"
    subprocess.run(
        [
            str(STAGING_OPERATOR),
            "stage-mounted",
            "--bundle",
            str(bundle),
            "--bundle-digest",
            metadata["bundle_digest"],
            "--bundle-size-bytes",
            str(metadata["bundle_size_bytes"]),
            "--mount-root",
            str(tmp_path / "mounted-volume"),
            "--volume-id",
            "network-volume-123",
            "--data-center-id",
            "EU-RO-1",
            "--volume-size-gb",
            "50",
            "--receipt-output",
            str(receipt_path),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["bundle_digest"] == metadata["bundle_digest"]
    assert receipt["bundle_path"] == metadata["bundle_path"]
    assert os.access(STAGING_OPERATOR, os.X_OK)
