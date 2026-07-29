from __future__ import annotations

import base64
import io
import json
import tarfile
from http import HTTPStatus
from http.client import HTTPConnection, RemoteDisconnected
from pathlib import Path
from threading import Thread

import pytest

from research.runpod.bootstrap_server import (
    BootstrapHandler,
    BootstrapServer,
    expected_bundle_files,
    install_bundle,
    install_environment_bundle,
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


def test_study_bundle_requires_the_frozen_trainer_and_environment(
    tmp_path: Path,
) -> None:
    workload_file = "repository_repair_study.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}

    install_bundle(
        bundle_payload(files),
        work_directory=tmp_path,
        workload_file=workload_file,
    )

    assert set(files) == {
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "result_server.py",
    }


def test_eligibility_bundle_requires_the_screen_and_frozen_study_sources(
    tmp_path: Path,
) -> None:
    workload_file = "repository_repair_eligibility.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}

    install_bundle(
        bundle_payload(files),
        work_directory=tmp_path,
        workload_file=workload_file,
    )

    assert set(files) == {
        "remote_runner.sh",
        "repository_repair_eligibility.py",
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "result_server.py",
    }


@pytest.mark.parametrize(
    "workload_file",
    (
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_pilot.py",
    ),
)
def test_large_model_bundle_requires_the_gate_and_versioned_environment(
    tmp_path: Path,
    workload_file: str,
) -> None:
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}

    install_bundle(
        bundle_payload(files),
        work_directory=tmp_path,
        workload_file=workload_file,
    )

    assert set(files) == {
        "larger_model_gate.py",
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_env_v32.py",
        "repository_repair_large_model_eligibility.py"
        if workload_file.endswith("eligibility.py")
        else "repository_repair_large_model_pilot.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "result_server.py",
    }


def test_external_evaluation_bundle_preserves_the_frozen_package_layout(
    tmp_path: Path,
) -> None:
    workload_file = "research/runpod/revision30_external_eval.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}

    install_bundle(
        bundle_payload(files),
        work_directory=tmp_path,
        workload_file=workload_file,
    )

    assert set(files) == {
        "external_eval_remote_runner.sh",
        "research/__init__.py",
        "research/external/revision30_task_pack.py",
        "research/frozen/revision30-external-pack.json",
        "research/runpod/__init__.py",
        "research/runpod/external_eval_transport.py",
        "research/runpod/repository_repair_env.py",
        "research/runpod/repository_repair_rl.py",
        "research/runpod/revision30_external_eval.py",
    }
    assert (tmp_path / workload_file).read_bytes() == files[workload_file]


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


def test_install_environment_bundle_decodes_and_validates_the_allowlist(
    tmp_path: Path,
) -> None:
    workload_file = "repository_repair_rl.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}

    install_environment_bundle(
        base64.b64encode(bundle_payload(files)).decode(),
        work_directory=tmp_path,
        workload_file=workload_file,
    )

    assert {path.name for path in tmp_path.iterdir()} == set(files)


def test_install_environment_bundle_rejects_invalid_base64(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not valid base64"):
        install_environment_bundle(
            "not-base64!",
            work_directory=tmp_path,
            workload_file="repository_repair_rl.py",
        )

    assert list(tmp_path.iterdir()) == []


def test_bootstrap_accepts_authenticated_bundle_on_proxy_rewritten_post_path(
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
            "/opaque-proxy-prefix/bundle?transport=1",
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


def test_bootstrap_hands_off_when_the_proxy_drops_the_acceptance_response(
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

    def dropped_response(
        _: BootstrapHandler,
        __: HTTPStatus,
        ___: dict[str, object],
    ) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(BootstrapHandler, "_write_json", dropped_response)
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
        with pytest.raises(RemoteDisconnected):
            connection.getresponse()
    finally:
        connection.close()
        thread.join(timeout=2)
        server.server_close()

    assert server.bundle_ready is True
    assert {path.name for path in tmp_path.iterdir()} == set(files)
