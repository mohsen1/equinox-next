from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from research.runpod.bootstrap_server import (
    expected_bundle_files,
    install_bundle,
)


def bundle_payload(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            metadata = tarfile.TarInfo(name)
            metadata.size = len(content)
            archive.addfile(metadata, io.BytesIO(content))
    return output.getvalue()


def test_install_bundle_accepts_only_the_expected_workload_files(
    tmp_path: Path,
) -> None:
    files = {
        name: f"{name}\n".encode() for name in expected_bundle_files("repository_repair_rl.py")
    }

    install_bundle(
        bundle_payload(files),
        work_directory=tmp_path,
        workload_file="repository_repair_rl.py",
    )

    assert {path.name for path in tmp_path.iterdir() if path.is_file()} == set(files)
    for name, content in files.items():
        assert (tmp_path / name).read_bytes() == content


def test_install_bundle_rejects_an_unexpected_file(tmp_path: Path) -> None:
    files = {name: b"expected" for name in expected_bundle_files("branching_sequence_ladder.py")}
    files["unexpected.py"] = b"untrusted"

    with pytest.raises(ValueError, match="allowlist"):
        install_bundle(
            bundle_payload(files),
            work_directory=tmp_path,
            workload_file="branching_sequence_ladder.py",
        )

    assert list(tmp_path.iterdir()) == []


def test_install_bundle_rejects_nested_paths(tmp_path: Path) -> None:
    files = {name: b"expected" for name in expected_bundle_files("repository_repair_rl.py")}
    files["repository_repair_rl.py"] = b"replacement"
    payload = bundle_payload(
        {
            **{
                name: content
                for name, content in files.items()
                if name != "repository_repair_rl.py"
            },
            "./repository_repair_rl.py": b"nested",
        }
    )

    with pytest.raises(ValueError, match="allowlist"):
        install_bundle(
            payload,
            work_directory=tmp_path,
            workload_file="repository_repair_rl.py",
        )

    assert list(tmp_path.iterdir()) == []
