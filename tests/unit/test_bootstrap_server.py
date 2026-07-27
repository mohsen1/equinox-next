from __future__ import annotations

import io
import json
import tarfile
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from research.runpod.bootstrap_server import (
    BootstrapHandler,
    BootstrapServer,
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


def test_bootstrap_accepts_authenticated_bundle_before_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    workload_file = "branching_sequence_ladder.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}
    payload = bundle_payload(files)
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", token)
    monkeypatch.setenv("EQUINOX_REMOTE_WORKDIR", str(tmp_path))
    monkeypatch.setenv("EQUINOX_WORKLOAD_FILE", workload_file)
    server = BootstrapServer(("127.0.0.1", 0), BootstrapHandler)
    thread = Thread(target=server.handle_request)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request(
            "POST",
            "/bundle",
            body=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/gzip",
                "Content-Length": str(len(payload)),
            },
        )
        response = connection.getresponse()

        assert response.status == HTTPStatus.ACCEPTED
        assert json.loads(response.read()) == {"status": "bundle_installed"}
    finally:
        connection.close()
        thread.join(timeout=2)
        server.server_close()

    assert server.bundle_ready is True
    assert {path.name for path in tmp_path.iterdir()} == set(files)
