from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import time
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.client import HTTPConnection, RemoteDisconnected
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from research.runpod import bootstrap_server
from research.runpod.bootstrap_server import (
    BOOTSTRAP_TRANSPORT_REVISION,
    BUNDLE_ACTIVATION_REVISION,
    BUNDLE_CONTENT_TYPE,
    BUNDLE_STAGE_RECEIPT_FILENAME,
    CODE_MATERIALIZATION_EVIDENCE_FILENAME,
    CODE_MATERIALIZATION_REVISION,
    DEPENDENCY_INDEX_URL,
    DEPENDENCY_INSTALLER_REVISION,
    DEPENDENCY_LOCK_DIGEST,
    DEPENDENCY_LOCK_FILENAME,
    DEPENDENCY_LOCK_SIZE_BYTES,
    DEPENDENCY_PLATFORM_TAG,
    DEPENDENCY_PYTHON_VERSION,
    DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
    DEPENDENCY_QUARANTINE_REVISION,
    LIVE_STAGE_ACTIVATION_FILENAME,
    LIVE_STAGE_PLAN_FILENAME,
    LIVE_STAGE_STATE_FILENAME,
    PRIVATE_ANCHOR_REVISION,
    PRIVATE_RUNTIME_REVISION,
    TORCH_RETENTION_EVIDENCE_FILENAME,
    VOLUME_READINESS_RECEIPT_FILENAME,
    BootstrapHandler,
    BootstrapServer,
    LiveStagePlan,
    PrivateAnchorRoot,
    PrivateRuntimeRoot,
    _apply_private_execution_environment,
    _exec_runner_from_safe_directory,
    _locked_requirement_versions,
    _write_atomic_at,
    arm_live_stage_deadline,
    build_bundle_stage_receipt,
    canonical_json,
    expected_activation,
    expected_bundle_files,
    initialize_or_recover_live_stage,
    install_bundle,
    install_environment_bundle,
    install_volume_bundle,
    main,
    materialize_code,
    open_live_work_directory,
    open_private_anchor_root,
    open_private_runtime_root,
    quarantine_dependencies,
    stage_uploaded_bundle,
    tagged_sha256,
)
from research.runpod.larger_model_gate import (
    expected_source_contract_digest,
    load_manifest,
)
from research.runpod.workload_bundle import (
    build_larger_model_bundle,
    verify_bundle_stage_receipt,
)


def bundle_payload(files: dict[str, bytes], *, mode: str = "w:gz") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode=mode) as archive:
        for name, content in files.items():
            metadata = tarfile.TarInfo(name)
            metadata.size = len(content)
            archive.addfile(metadata, io.BytesIO(content))
    return output.getvalue()


def write_regular_at(root_descriptor: int, relative_path: str, payload: bytes) -> None:
    parts = relative_path.split("/")
    directory_descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            with suppress(FileExistsError):
                os.mkdir(component, mode=0o755, dir_fd=directory_descriptor)
            child_descriptor = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=directory_descriptor,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_descriptor)


def record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return encoded.rstrip(b"=").decode()


def populate_locked_dependency_tree(
    target_descriptor: int,
    *,
    corrupt_record: bool = False,
    add_symlink: bool = False,
    add_derived_files: bool = False,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    versions = _locked_requirement_versions(
        (repository_root / "research/runpod" / DEPENDENCY_LOCK_FILENAME).read_bytes()
    )
    for index, (name, version) in enumerate(sorted(versions.items())):
        module_path = f"locked_dependency_{index}.py"
        module_payload = f'VERSION = "{version}"\n'.encode()
        distribution_directory = f"{name.replace('-', '_')}-{version}.dist-info"
        metadata_path = f"{distribution_directory}/METADATA"
        metadata_payload = (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
        ).encode()
        record_path = f"{distribution_directory}/RECORD"
        module_digest = (
            "A" * 43 if corrupt_record and index == 0 else record_hash(module_payload)
        )
        record_payload = (
            f"{module_path},sha256={module_digest},{len(module_payload)}\n"
            f"{metadata_path},sha256={record_hash(metadata_payload)},{len(metadata_payload)}\n"
            f"{record_path},,\n"
        ).encode()
        write_regular_at(target_descriptor, module_path, module_payload)
        write_regular_at(target_descriptor, metadata_path, metadata_payload)
        write_regular_at(target_descriptor, record_path, record_payload)
    if add_derived_files:
        write_regular_at(target_descriptor, "bin/accelerate", b"ignored script\n")
        write_regular_at(
            target_descriptor,
            "accelerate/__pycache__/derived.cpython-312.pyc",
            b"ignored bytecode",
        )
    if add_symlink:
        os.symlink("locked_dependency_0.py", "unsafe.py", dir_fd=target_descriptor)


def open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))


def populate_code_tree(root: Path, workload_file: str) -> dict[str, bytes]:
    repository_root = Path(__file__).resolve().parents[2]
    contents: dict[str, bytes] = {}
    for name in expected_bundle_files(workload_file):
        payload = (
            (repository_root / "research/runpod" / name).read_bytes()
            if name == DEPENDENCY_LOCK_FILENAME
            else f"{name}: immutable code\n".encode()
        )
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        contents[name] = payload
    return contents


def live_bundle(
    volume_root: Path,
) -> tuple[bytes, LiveStagePlan, dict[str, bytes]]:
    workload_file = "repository_repair_large_model_eligibility.py"
    files = {}
    for name in expected_bundle_files(workload_file):
        if name == "larger-model-eligibility.json":
            continue
        files[name] = (
            (Path(__file__).resolve().parents[2] / "research/runpod" / name).read_bytes()
            if name == DEPENDENCY_LOCK_FILENAME
            else f"{name}: trusted test source\n".encode()
        )
    source_names = (
        "repository_repair_env.py",
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_trainer.py",
        "retention_checkpoint_probe.py",
        DEPENDENCY_LOCK_FILENAME,
    )
    source_contract = {
        "algorithm": "sha256",
        "files": {name: hashlib.sha256(files[name]).hexdigest() for name in source_names},
    }
    profile_id = "test-profile@10"
    manifest = {
        "profile_id": profile_id,
        "source_contract": source_contract,
        "materialization": {
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
        },
        "screen": {
            "live_stage_activation_revision": BUNDLE_ACTIVATION_REVISION,
        },
    }
    files["larger-model-eligibility.json"] = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
    ).encode()
    payload = bundle_payload(files, mode="w:xz")
    digest = tagged_sha256(payload)
    plan = LiveStagePlan(
        profile_id=profile_id,
        manifest_digest=tagged_sha256(canonical_json(manifest)),
        source_contract_digest=tagged_sha256(
            canonical_json(
                {
                    "profile_id": profile_id,
                    "source_contract": source_contract,
                }
            )
        ),
        bootstrap_source_digest="sha256:" + "b" * 64,
        head_commit="a" * 40,
        network_volume_id="volume-123",
        network_volume_data_center_id="EU-FR-1",
        network_volume_size_gb=50,
        workload_bundle_digest=digest,
        workload_bundle_size_bytes=len(payload),
        workload_bundle_path=str(
            volume_root / profile_id / f"{digest.removeprefix('sha256:')}.tar.xz"
        ),
        readiness_deadline_epoch=int(time.time()) + 300,
    )
    return payload, plan, files


def serve(server: BootstrapServer) -> Thread:
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    return thread


def request_json(
    server: BootstrapServer,
    method: str,
    path: str,
    *,
    token: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(*server.server_address, timeout=3)
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Length"] = str(len(body))
    if content_type is not None:
        headers["Content-Type"] = content_type
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def prepared_materialization(
    server: BootstrapServer,
    plan: LiveStagePlan,
) -> tuple[dict[str, object], dict[str, object]]:
    dependency: dict[str, object] = {
        "schema_version": 2,
        "revision": DEPENDENCY_QUARANTINE_REVISION,
        "profile_id": plan.profile_id,
        "lock_path": f"{plan.workload_bundle_path}::{DEPENDENCY_LOCK_FILENAME}",
        "lock_digest": DEPENDENCY_LOCK_DIGEST,
        "python_version": DEPENDENCY_PYTHON_VERSION,
        "platform_tag": DEPENDENCY_PLATFORM_TAG,
        "installer_revision": DEPENDENCY_INSTALLER_REVISION,
        "index_url": DEPENDENCY_INDEX_URL,
        "private_root": "/tmp/equinox-quarantine/dependencies/" + "d" * 64,
        "install_tree_digest": "sha256:" + "d" * 64,
        "private_tree_digest": "sha256:" + "d" * 64,
        "record_closure_digest": "sha256:" + "e" * 64,
        "distributions": [],
        "installed_file_count": 1,
        "installed_bytes": 1,
        "ready": True,
    }
    dependency["evidence_digest"] = tagged_sha256(canonical_json(dependency))
    code: dict[str, object] = {
        "schema_version": 1,
        "revision": CODE_MATERIALIZATION_REVISION,
        "profile_id": plan.profile_id,
        "bundle_digest": plan.workload_bundle_digest,
        "bundle_size_bytes": plan.workload_bundle_size_bytes,
        "source_contract_digest": plan.source_contract_digest,
        "source_root": plan.workload_bundle_path,
        "private_root": "/tmp/equinox-quarantine/code/" + "c" * 64,
        "source_tree_digest": "sha256:" + "c" * 64,
        "private_tree_digest": "sha256:" + "c" * 64,
        "files": [],
        "installed_file_count": 1,
        "installed_bytes": 1,
        "ready": True,
    }
    code["evidence_digest"] = tagged_sha256(canonical_json(code))
    assert server.work_directory_descriptor is not None
    _write_atomic_at(
        server.work_directory_descriptor,
        DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
        canonical_json(dependency) + b"\n",
    )
    _write_atomic_at(
        server.work_directory_descriptor,
        CODE_MATERIALIZATION_EVIDENCE_FILENAME,
        canonical_json(code) + b"\n",
    )
    server.dependency_quarantine_evidence = dependency
    server.code_materialization_evidence = code
    return dependency, code


def use_prepared_materialization(
    monkeypatch: pytest.MonkeyPatch,
    plan: LiveStagePlan,
) -> None:
    def prepare(
        server: BootstrapServer,
        *,
        work_directory: Path,
        recovered_dependency_evidence: dict[str, object] | None = None,
        recovered_code_evidence: dict[str, object] | None = None,
        enforce_deadline: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del (
            work_directory,
            recovered_dependency_evidence,
            recovered_code_evidence,
            enforce_deadline,
        )
        return prepared_materialization(server, plan)

    monkeypatch.setattr(
        "research.runpod.bootstrap_server.prepare_private_materialization",
        prepare,
    )


def prepared_evidence(
    server: BootstrapServer,
    plan: LiveStagePlan,
) -> tuple[dict[str, object], dict[str, object]]:
    assert server.bundle_stage_receipt is not None
    dependency = server.dependency_quarantine_evidence
    code = server.code_materialization_evidence
    assert dependency is not None
    assert code is not None
    volume: dict[str, object] = {
        "profile_id": plan.profile_id,
        "manifest_digest": plan.manifest_digest,
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "dependency_lock_digest": dependency["lock_digest"],
        "dependency_quarantine_evidence": dependency,
        "dependency_quarantine_evidence_digest": dependency["evidence_digest"],
        "dependency_private_tree_digest": dependency["private_tree_digest"],
        "code_materialization_evidence": code,
        "code_materialization_evidence_digest": code["evidence_digest"],
        "code_private_tree_digest": code["private_tree_digest"],
        "ready": True,
    }
    volume["receipt_digest"] = tagged_sha256(canonical_json(volume))
    retention: dict[str, object] = {
        "profile_id": plan.profile_id,
        "head_commit": plan.head_commit,
        "source_contract_digest": plan.source_contract_digest,
        "workload_bundle_digest": plan.workload_bundle_digest,
        "workload_bundle_size_bytes": plan.workload_bundle_size_bytes,
        "workload_bundle_path": plan.workload_bundle_path,
        "bundle_stage_receipt_digest": server.bundle_stage_receipt["receipt_digest"],
        "bootstrap_source_digest": plan.bootstrap_source_digest,
        "volume_readiness_receipt_digest": volume["receipt_digest"],
        "dependency_lock_digest": dependency["lock_digest"],
        "dependency_quarantine_evidence_digest": dependency["evidence_digest"],
        "dependency_private_tree_digest": dependency["private_tree_digest"],
        "code_materialization_evidence_digest": code["evidence_digest"],
        "code_private_tree_digest": code["private_tree_digest"],
        "network_volume_id": plan.network_volume_id,
        "network_volume_data_center_id": plan.network_volume_data_center_id,
        "network_volume_size_gb": plan.network_volume_size_gb,
        "status": "passed",
    }
    retention["evidence_digest"] = tagged_sha256(canonical_json(retention))
    assert server.work_directory_descriptor is not None
    _write_atomic_at(
        server.work_directory_descriptor,
        VOLUME_READINESS_RECEIPT_FILENAME,
        canonical_json(volume) + b"\n",
    )
    _write_atomic_at(
        server.work_directory_descriptor,
        TORCH_RETENTION_EVIDENCE_FILENAME,
        canonical_json(retention) + b"\n",
    )
    return volume, retention


def live_stage_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: LiveStagePlan,
    token: str,
) -> tuple[BootstrapServer, Path, int]:
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT",
        tmp_path,
    )
    work_directory = tmp_path / "work"
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", token)
    monkeypatch.setenv("EQUINOX_REMOTE_WORKDIR", str(work_directory))
    monkeypatch.setenv(
        "EQUINOX_WORKLOAD_FILE",
        "repository_repair_large_model_eligibility.py",
    )
    descriptor = open_live_work_directory(work_directory)
    server = BootstrapServer(
        ("127.0.0.1", 0),
        BootstrapHandler,
        plan=plan,
        work_directory=work_directory,
        work_directory_descriptor=descriptor,
    )
    initialize_or_recover_live_stage(server)
    return server, work_directory, descriptor


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
        "larger-model-dependencies.lock",
        "larger-model-eligibility.json",
        "larger_model_gate.py",
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_env_v32.py",
        "repository_repair_env_v33.py",
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_pilot.py",
        "repository_repair_large_model_study.py",
        "repository_repair_large_model_trainer.py",
        "retention_checkpoint_probe.py",
        "result_server.py",
    }


def test_large_model_screen_and_pilot_share_the_transactional_support_set() -> None:
    screen_files = expected_bundle_files("repository_repair_large_model_eligibility.py")
    pilot_files = expected_bundle_files("repository_repair_large_model_pilot.py")

    assert screen_files == pilot_files
    assert {
        "repository_repair_large_model_study.py",
        "repository_repair_large_model_trainer.py",
    } <= screen_files
    assert {
        "repository_repair_rl.py",
        "repository_repair_study.py",
    }.isdisjoint(screen_files)


def test_hash_locked_dependencies_are_record_closed_and_privately_hardened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    lock_payload = (
        Path(__file__).resolve().parents[2]
        / "research/runpod"
        / DEPENDENCY_LOCK_FILENAME
    ).read_bytes()
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(lock_payload)
    code_descriptor = open_directory(code_root)

    def install(**arguments: object) -> None:
        populate_locked_dependency_tree(
            int(arguments["target_descriptor"]),
            add_derived_files=True,
        )

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._run_locked_dependency_installer",
        install,
    )
    try:
        quarantined = quarantine_dependencies(
            profile_id="test-profile@10",
            code_root_descriptor=code_descriptor,
            lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
            private_base=tmp_path / "private/dependencies",
            deadline_epoch=time.time() + 60,
        )
    finally:
        os.close(code_descriptor)
    try:
        evidence = quarantined.evidence
        assert evidence["schema_version"] == 2
        assert evidence["revision"] == DEPENDENCY_QUARANTINE_REVISION
        assert evidence["lock_digest"] == DEPENDENCY_LOCK_DIGEST
        assert evidence["install_tree_digest"] == evidence["private_tree_digest"]
        assert len(evidence["distributions"]) == 30
        assert evidence["installed_file_count"] == 90
        private_files = {
            path.relative_to(quarantined.private_root).as_posix()
            for path in quarantined.private_root.rglob("*")
            if path.is_file()
        }
        assert not any(path.startswith("bin/") for path in private_files)
        assert not any("__pycache__" in path for path in private_files)
        assert all(
            path.stat().st_mode & 0o222 == 0
            for path in quarantined.private_root.rglob("*")
        )
    finally:
        os.close(quarantined.private_root_descriptor)


def test_hash_locked_installer_is_isolated_bounded_and_binary_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    target_root = tmp_path / "target"
    code_root.mkdir()
    target_root.mkdir()
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(b"lock bytes are opened by pip")
    code_descriptor = open_directory(code_root)
    target_descriptor = open_directory(target_root)
    observed: dict[str, object] = {}

    def run(command: list[str], **arguments: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["environment"] = arguments["env"]
        observed["pass_fds"] = arguments["pass_fds"]
        observed["timeout"] = arguments["timeout"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        bootstrap_server.sys,
        "version_info",
        SimpleNamespace(major=3, minor=12),
    )
    monkeypatch.setattr(bootstrap_server.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap_server.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap_server.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/workspace/equinox-state/python")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    try:
        bootstrap_server._run_locked_dependency_installer(
            code_root_descriptor=code_descriptor,
            target_descriptor=target_descriptor,
            deadline_epoch=time.time() + 60,
        )
    finally:
        os.close(target_descriptor)
        os.close(code_descriptor)
    command = observed["command"]
    assert isinstance(command, list)
    assert command[1:5] == ["-I", "-m", "pip", "--isolated"]
    assert "--require-hashes" in command
    assert "--only-binary=:all:" in command
    assert "--no-deps" in command
    assert "--no-compile" in command
    assert command[command.index("--index-url") + 1] == DEPENDENCY_INDEX_URL
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert "PYTHONPATH" not in environment
    assert "PIP_NO_INDEX" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert observed["pass_fds"] == (code_descriptor, target_descriptor)
    assert 0 < float(observed["timeout"]) <= 60


def test_dependency_quarantine_rejects_changed_lock_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    lock_payload = bytearray(
        (
            Path(__file__).resolve().parents[2]
            / "research/runpod"
            / DEPENDENCY_LOCK_FILENAME
        ).read_bytes()
    )
    lock_payload[-1] ^= 1
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(lock_payload)
    code_descriptor = open_directory(code_root)
    called = False

    def install(**arguments: object) -> None:
        nonlocal called
        del arguments
        called = True

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._run_locked_dependency_installer",
        install,
    )
    try:
        with pytest.raises(ValueError, match="lock identity"):
            quarantine_dependencies(
                profile_id="test-profile@10",
                code_root_descriptor=code_descriptor,
                lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
                private_base=tmp_path / "private/dependencies",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(code_descriptor)
    assert called is False


@pytest.mark.parametrize(
    ("corrupt_record", "add_symlink", "message"),
    (
        (True, False, "RECORD hash"),
        (False, True, "symbolic link"),
    ),
)
def test_hash_locked_dependency_quarantine_rejects_unsafe_install_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_record: bool,
    add_symlink: bool,
    message: str,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(
        (
            Path(__file__).resolve().parents[2]
            / "research/runpod"
            / DEPENDENCY_LOCK_FILENAME
        ).read_bytes()
    )
    code_descriptor = open_directory(code_root)

    def install(**arguments: object) -> None:
        populate_locked_dependency_tree(
            int(arguments["target_descriptor"]),
            corrupt_record=corrupt_record,
            add_symlink=add_symlink,
        )

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._run_locked_dependency_installer",
        install,
    )
    try:
        with pytest.raises(ValueError, match=message):
            quarantine_dependencies(
                profile_id="test-profile@10",
                code_root_descriptor=code_descriptor,
                lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
                private_base=tmp_path / "private/dependencies",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(code_descriptor)


def test_dependency_quarantine_rejects_private_tree_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(
        (
            Path(__file__).resolve().parents[2]
            / "research/runpod"
            / DEPENDENCY_LOCK_FILENAME
        ).read_bytes()
    )
    code_descriptor = open_directory(code_root)

    def install(**arguments: object) -> None:
        populate_locked_dependency_tree(int(arguments["target_descriptor"]))

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._run_locked_dependency_installer",
        install,
    )
    try:
        first = quarantine_dependencies(
            profile_id="test-profile@10",
            code_root_descriptor=code_descriptor,
            lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
            private_base=tmp_path / "private/dependencies",
            deadline_epoch=time.time() + 60,
        )
        os.close(first.private_root_descriptor)
        substituted = first.private_root / "locked_dependency_0.py"
        substituted.chmod(0o644)
        with pytest.raises(ValueError, match="permissions"):
            quarantine_dependencies(
                profile_id="test-profile@10",
                code_root_descriptor=code_descriptor,
                lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
                private_base=tmp_path / "private/dependencies",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(code_descriptor)


def test_code_materialization_isolated_from_shared_source_mutation(
    tmp_path: Path,
) -> None:
    workload_file = "repository_repair_large_model_eligibility.py"
    source_root = tmp_path / "shared"
    source_root.mkdir()
    contents = populate_code_tree(source_root, workload_file)
    source_descriptor = open_directory(source_root)
    try:
        materialized = materialize_code(
            source_descriptor,
            workload_file=workload_file,
            profile_id="test-profile@10",
            bundle_digest="sha256:" + "a" * 64,
            bundle_size_bytes=123,
            source_contract_digest="sha256:" + "b" * 64,
            source_root_identity="/bundle/exact.tar.xz",
            private_base=tmp_path / "private/code",
            deadline_epoch=time.time() + 60,
        )
    finally:
        os.close(source_descriptor)
    try:
        (source_root / "remote_runner.sh").write_bytes(b"attacker replacement\n")
        assert (
            materialized.private_root / "remote_runner.sh"
        ).read_bytes() == contents["remote_runner.sh"]
        assert materialized.evidence["source_tree_digest"] == materialized.evidence[
            "private_tree_digest"
        ]
        assert materialized.evidence["installed_file_count"] == len(contents)
    finally:
        os.close(materialized.private_root_descriptor)


def test_code_materialization_rejects_symlinked_bundle_member(tmp_path: Path) -> None:
    workload_file = "repository_repair_large_model_eligibility.py"
    source_root = tmp_path / "shared"
    source_root.mkdir()
    populate_code_tree(source_root, workload_file)
    (source_root / "remote_runner.sh").unlink()
    (source_root / "remote_runner.sh").symlink_to("result_server.py")
    source_descriptor = open_directory(source_root)
    try:
        with pytest.raises(OSError):
            materialize_code(
                source_descriptor,
                workload_file=workload_file,
                profile_id="test-profile@10",
                bundle_digest="sha256:" + "a" * 64,
                bundle_size_bytes=123,
                source_contract_digest="sha256:" + "b" * 64,
                source_root_identity="/bundle/exact.tar.xz",
                private_base=tmp_path / "private/code",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(source_descriptor)


def test_code_materialization_rejects_source_mutation_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload_file = "repository_repair_large_model_eligibility.py"
    source_root = tmp_path / "shared"
    source_root.mkdir()
    populate_code_tree(source_root, workload_file)
    source_descriptor = open_directory(source_root)
    real_copy = bootstrap_server._copy_tree_to_private_root

    def mutate_after_copy(
        copied_source_descriptor: int,
        entries: list[dict[str, object]],
        **arguments: object,
    ) -> tuple[Path, int]:
        result = real_copy(copied_source_descriptor, entries, **arguments)
        descriptor = os.open(
            "remote_runner.sh",
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=copied_source_descriptor,
        )
        try:
            os.write(descriptor, b"mutated during copy\n")
        finally:
            os.close(descriptor)
        return result

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._copy_tree_to_private_root",
        mutate_after_copy,
    )
    try:
        with pytest.raises(ValueError, match="changed during private"):
            materialize_code(
                source_descriptor,
                workload_file=workload_file,
                profile_id="test-profile@10",
                bundle_digest="sha256:" + "a" * 64,
                bundle_size_bytes=123,
                source_contract_digest="sha256:" + "b" * 64,
                source_root_identity="/bundle/exact.tar.xz",
                private_base=tmp_path / "private/code",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(source_descriptor)


def test_dependency_quarantine_rejects_install_mutation_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / DEPENDENCY_LOCK_FILENAME).write_bytes(
        (
            Path(__file__).resolve().parents[2]
            / "research/runpod"
            / DEPENDENCY_LOCK_FILENAME
        ).read_bytes()
    )
    code_descriptor = open_directory(code_root)

    def install(**arguments: object) -> None:
        populate_locked_dependency_tree(int(arguments["target_descriptor"]))

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._run_locked_dependency_installer",
        install,
    )
    real_copy = bootstrap_server._copy_tree_to_private_root

    def mutate_after_copy(
        copied_source_descriptor: int,
        entries: list[dict[str, object]],
        **arguments: object,
    ) -> tuple[Path, int]:
        result = real_copy(copied_source_descriptor, entries, **arguments)
        descriptor = os.open(
            "locked_dependency_0.py",
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=copied_source_descriptor,
        )
        try:
            os.write(descriptor, b"mutated during copy\n")
        finally:
            os.close(descriptor)
        return result

    monkeypatch.setattr(
        "research.runpod.bootstrap_server._copy_tree_to_private_root",
        mutate_after_copy,
    )
    try:
        with pytest.raises(ValueError, match="RECORD hash|changed during private"):
            quarantine_dependencies(
                profile_id="test-profile@10",
                code_root_descriptor=code_descriptor,
                lock_path_identity="/bundle/exact.tar.xz::larger-model-dependencies.lock",
                private_base=tmp_path / "private/dependencies",
                deadline_epoch=time.time() + 60,
            )
    finally:
        os.close(code_descriptor)


def test_private_runtime_root_is_stable_and_launch_token_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_base = tmp_path / "quarantine/runtime"
    monkeypatch.setattr(bootstrap_server, "PRIVATE_RUNTIME_ROOT", runtime_base)
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "first-launch-token")
    identity = {
        "profile_id": "test-profile@10",
        "bundle_digest": "sha256:" + "a" * 64,
        "work_directory_identity": "/workspace/equinox-runs/proof-123",
    }

    first = open_private_runtime_root(**identity)
    second = open_private_runtime_root(**identity)
    try:
        assert first.path == second.path
        assert os.fstat(first.descriptor).st_ino == os.fstat(second.descriptor).st_ino
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o700
        assert stat.S_IMODE(runtime_base.stat().st_mode) == 0o700
    finally:
        os.close(second.descriptor)
        os.close(first.descriptor)

    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "second-launch-token")
    isolated = open_private_runtime_root(**identity)
    try:
        assert isolated.path != first.path
        assert isolated.path.parent == runtime_base
    finally:
        os.close(isolated.descriptor)


@pytest.mark.parametrize("substitution", ("permissions", "symlink"))
def test_private_runtime_root_rejects_preexisting_unsafe_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    runtime_base = tmp_path / "quarantine/runtime"
    monkeypatch.setattr(bootstrap_server, "PRIVATE_RUNTIME_ROOT", runtime_base)
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "launch-token")
    identity = {
        "profile_id": "test-profile@10",
        "bundle_digest": "sha256:" + "a" * 64,
        "work_directory_identity": "/workspace/equinox-runs/proof-123",
    }
    prepared = open_private_runtime_root(**identity)
    os.close(prepared.descriptor)
    if substitution == "permissions":
        prepared.path.chmod(0o755)
        expected_error = ValueError
    else:
        prepared.path.rmdir()
        symlink_target = tmp_path / "attacker-runtime"
        symlink_target.mkdir()
        prepared.path.symlink_to(symlink_target, target_is_directory=True)
        expected_error = OSError

    with pytest.raises(expected_error):
        open_private_runtime_root(**identity)


def test_private_anchor_root_is_stable_launch_token_isolated_and_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_base = tmp_path / "quarantine/runtime"
    anchor_base = tmp_path / "quarantine/anchors"
    monkeypatch.setattr(bootstrap_server, "PRIVATE_RUNTIME_ROOT", runtime_base)
    monkeypatch.setattr(bootstrap_server, "PRIVATE_ANCHOR_ROOT", anchor_base)
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "first-launch-token")
    identity = {
        "profile_id": "test-profile@10",
        "bundle_digest": "sha256:" + "a" * 64,
        "work_directory_identity": "/workspace/equinox-runs/proof-123",
    }

    runtime = open_private_runtime_root(**identity)
    first = open_private_anchor_root(**identity)
    second = open_private_anchor_root(**identity)
    try:
        assert first.path == second.path
        assert os.fstat(first.descriptor).st_ino == os.fstat(second.descriptor).st_ino
        assert (
            os.fstat(first.descriptor).st_dev,
            os.fstat(first.descriptor).st_ino,
        ) != (
            os.fstat(runtime.descriptor).st_dev,
            os.fstat(runtime.descriptor).st_ino,
        )
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o700
        assert stat.S_IMODE(anchor_base.stat().st_mode) == 0o700
    finally:
        os.close(second.descriptor)
        os.close(first.descriptor)
        os.close(runtime.descriptor)

    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "second-launch-token")
    isolated = open_private_anchor_root(**identity)
    try:
        assert isolated.path != first.path
        assert isolated.path.parent == anchor_base
    finally:
        os.close(isolated.descriptor)


@pytest.mark.parametrize("substitution", ("permissions", "symlink"))
def test_private_anchor_root_rejects_preexisting_unsafe_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    anchor_base = tmp_path / "quarantine/anchors"
    monkeypatch.setattr(bootstrap_server, "PRIVATE_ANCHOR_ROOT", anchor_base)
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "launch-token")
    identity = {
        "profile_id": "test-profile@10",
        "bundle_digest": "sha256:" + "a" * 64,
        "work_directory_identity": "/workspace/equinox-runs/proof-123",
    }
    prepared = open_private_anchor_root(**identity)
    os.close(prepared.descriptor)
    if substitution == "permissions":
        prepared.path.chmod(0o755)
        expected_error = ValueError
    else:
        prepared.path.rmdir()
        symlink_target = tmp_path / "attacker-anchor"
        symlink_target.mkdir()
        prepared.path.symlink_to(symlink_target, target_is_directory=True)
        expected_error = OSError

    with pytest.raises(expected_error):
        open_private_anchor_root(**identity)


def test_private_execution_environment_exports_pinned_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = {
        name: tmp_path / name
        for name in ("persistent", "dependencies", "code", "runtime", "anchor")
    }
    for path in roots.values():
        path.mkdir()
    roots["runtime"].chmod(0o700)
    roots["anchor"].chmod(0o700)
    descriptors = {name: open_directory(path) for name, path in roots.items()}
    runtime_root = PrivateRuntimeRoot(roots["runtime"], descriptors["runtime"])
    anchor_root = PrivateAnchorRoot(roots["anchor"], descriptors["anchor"])
    captured_environment: dict[str, str] = {}
    monkeypatch.setattr(bootstrap_server.os, "environ", captured_environment)
    try:
        stable_work, stable_dependencies, stable_code, stable_runtime, stable_anchor = (
            _apply_private_execution_environment(
                descriptors["persistent"],
                dependency_descriptor=descriptors["dependencies"],
                code_descriptor=descriptors["code"],
                runtime_root=runtime_root,
                anchor_root=anchor_root,
                dependency_evidence={
                    "evidence_digest": "sha256:" + "d" * 64,
                    "private_tree_digest": "sha256:" + "e" * 64,
                    "lock_digest": DEPENDENCY_LOCK_DIGEST,
                },
                code_evidence={
                    "evidence_digest": "sha256:" + "f" * 64,
                    "private_tree_digest": "sha256:" + "1" * 64,
                },
            )
        )
        assert stable_work == Path(f"/proc/self/fd/{descriptors['persistent']}")
        assert stable_dependencies == Path(
            f"/proc/self/fd/{descriptors['dependencies']}"
        )
        assert stable_code == Path(f"/proc/self/fd/{descriptors['code']}")
        assert stable_runtime == Path(f"/proc/self/fd/{descriptors['runtime']}")
        assert stable_anchor == Path(f"/proc/self/fd/{descriptors['anchor']}")
        assert captured_environment["EQUINOX_RUNTIME_ROOT"] == str(stable_runtime)
        assert (
            captured_environment["EQUINOX_PRIVATE_RUNTIME_REVISION"]
            == PRIVATE_RUNTIME_REVISION
        )
        assert captured_environment["EQUINOX_RUNTIME_ANCHOR_ROOT"] == str(
            stable_anchor
        )
        assert (
            captured_environment["EQUINOX_PRIVATE_ANCHOR_REVISION"]
            == PRIVATE_ANCHOR_REVISION
        )
        assert all(os.get_inheritable(descriptor) for descriptor in descriptors.values())
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


@pytest.mark.parametrize("alias_name", ("persistent", "dependencies", "code", "runtime"))
def test_private_execution_environment_rejects_anchor_aliases(
    tmp_path: Path,
    alias_name: str,
) -> None:
    roots = {
        name: tmp_path / name
        for name in ("persistent", "dependencies", "code", "runtime", "anchor")
    }
    for path in roots.values():
        path.mkdir()
    roots["runtime"].chmod(0o700)
    roots[alias_name].chmod(0o700)
    descriptors = {name: open_directory(path) for name, path in roots.items()}
    anchor_root = PrivateAnchorRoot(roots[alias_name], descriptors[alias_name])
    try:
        with pytest.raises(ValueError, match="anchor root aliased"):
            _apply_private_execution_environment(
                descriptors["persistent"],
                dependency_descriptor=descriptors["dependencies"],
                code_descriptor=descriptors["code"],
                runtime_root=PrivateRuntimeRoot(
                    roots["runtime"],
                    descriptors["runtime"],
                ),
                anchor_root=anchor_root,
                dependency_evidence={
                    "evidence_digest": "sha256:" + "d" * 64,
                    "private_tree_digest": "sha256:" + "e" * 64,
                    "lock_digest": DEPENDENCY_LOCK_DIGEST,
                },
                code_evidence={
                    "evidence_digest": "sha256:" + "f" * 64,
                    "private_tree_digest": "sha256:" + "1" * 64,
                },
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_private_runner_handoff_closes_anchor_and_execution_descriptors_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = {
        name: tmp_path / name
        for name in ("persistent", "dependencies", "code", "runtime", "anchor")
    }
    for path in roots.values():
        path.mkdir()
    roots["runtime"].chmod(0o700)
    roots["anchor"].chmod(0o700)
    descriptors = {name: open_directory(path) for name, path in roots.items()}
    monkeypatch.setattr(
        bootstrap_server,
        "_verify_private_execution_trees",
        lambda **_arguments: None,
    )
    monkeypatch.setattr(bootstrap_server.os, "environ", {})

    def reject_exec(
        _executable: str,
        _arguments: list[str],
        _environment: dict[str, str],
    ) -> None:
        raise RuntimeError("synthetic exec failure")

    monkeypatch.setattr(bootstrap_server.os, "execvpe", reject_exec)

    with pytest.raises(RuntimeError, match="synthetic exec failure"):
        _exec_runner_from_safe_directory(
            descriptors["persistent"],
            workload_file="repository_repair_large_model_pilot.py",
            dependency_descriptor=descriptors["dependencies"],
            code_descriptor=descriptors["code"],
            runtime_root=PrivateRuntimeRoot(
                roots["runtime"],
                descriptors["runtime"],
            ),
            anchor_root=PrivateAnchorRoot(
                roots["anchor"],
                descriptors["anchor"],
            ),
            dependency_evidence={
                "evidence_digest": "sha256:" + "d" * 64,
                "private_tree_digest": "sha256:" + "e" * 64,
                "lock_digest": DEPENDENCY_LOCK_DIGEST,
            },
            code_evidence={
                "evidence_digest": "sha256:" + "f" * 64,
                "private_tree_digest": "sha256:" + "1" * 64,
            },
        )

    for descriptor in descriptors.values():
        with pytest.raises(OSError):
            os.fstat(descriptor)


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
    payload = bundle_payload(files, mode="w:xz")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"

    install_environment_bundle(
        base64.b64encode(payload).decode(),
        work_directory=tmp_path,
        workload_file=workload_file,
        expected_digest=digest,
    )

    assert {path.name for path in tmp_path.iterdir()} == set(files)


def test_install_environment_bundle_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    workload_file = "repository_repair_rl.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}
    payload = bundle_payload(files, mode="w:xz")

    with pytest.raises(ValueError, match="digest did not match"):
        install_environment_bundle(
            base64.b64encode(payload).decode(),
            work_directory=tmp_path,
            workload_file=workload_file,
            expected_digest="sha256:" + "0" * 64,
        )

    assert list(tmp_path.iterdir()) == []


def test_install_environment_bundle_rejects_invalid_base64(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not valid base64"):
        install_environment_bundle(
            "not-base64!",
            work_directory=tmp_path,
            workload_file="repository_repair_rl.py",
        )

    assert list(tmp_path.iterdir()) == []


def test_install_volume_bundle_verifies_content_address_and_exact_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload_file = "repository_repair_rl.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}
    payload = bundle_payload(files, mode="w:xz")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT",
        tmp_path,
    )
    bundle = volume_root / "profile@1" / f"{digest.removeprefix('sha256:')}.tar.xz"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(payload)
    bundle.chmod(0o644)
    work_directory = tmp_path / "work"
    work_directory.mkdir()

    install_volume_bundle(
        bundle,
        work_directory=work_directory,
        workload_file=workload_file,
        expected_digest=digest,
        expected_size_bytes=len(payload),
    )

    assert {path.name for path in work_directory.iterdir()} == set(files)


def test_volume_bootstrap_restores_verified_handoff_identity_for_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    volume_path = (
        "/workspace/equinox-state/workload-bundles/profile@1/"
        f"{digest.removeprefix('sha256:')}.tar.xz"
    )
    monkeypatch.setenv("EQUINOX_REMOTE_WORKDIR", str(tmp_path))
    monkeypatch.setenv("EQUINOX_WORKLOAD_FILE", "repository_repair_rl.py")
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "test-token")
    monkeypatch.setenv("EQUINOX_BUNDLE_VOLUME_PATH", volume_path)
    monkeypatch.setenv("EQUINOX_BUNDLE_SHA256", digest)
    monkeypatch.setenv("EQUINOX_BUNDLE_SIZE_BYTES", "73208")
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.install_volume_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("research.runpod.bootstrap_server.time.sleep", lambda _seconds: None)
    executed: dict[str, object] = {}

    def capture_exec(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        executed.update(
            executable=executable,
            arguments=arguments,
            environment=dict(environment),
        )

    monkeypatch.setattr(os, "execvpe", capture_exec)

    main()

    assert executed["executable"] == "bash"
    environment = executed["environment"]
    assert isinstance(environment, dict)
    assert environment["EQUINOX_BUNDLE_VOLUME_PATH"] == volume_path
    assert environment["EQUINOX_BUNDLE_SHA256"] == digest
    assert environment["EQUINOX_BUNDLE_SIZE_BYTES"] == "73208"


@pytest.mark.parametrize("failure", ("digest", "size", "symlink", "path"))
def test_install_volume_bundle_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    workload_file = "repository_repair_rl.py"
    files = {name: f"{name}\n".encode() for name in expected_bundle_files(workload_file)}
    payload = bundle_payload(files, mode="w:xz")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    filename_digest = digest if failure != "path" else "sha256:" + "1" * 64
    bundle = volume_root / "profile@1" / f"{filename_digest.removeprefix('sha256:')}.tar.xz"
    bundle.parent.mkdir(parents=True)
    target = tmp_path / "target.tar.xz"
    target.write_bytes(payload)
    if failure == "symlink":
        bundle.symlink_to(target)
    else:
        bundle.write_bytes(payload)
    work_directory = tmp_path / "work"
    work_directory.mkdir()

    with pytest.raises((OSError, ValueError)):
        install_volume_bundle(
            bundle,
            work_directory=work_directory,
            workload_file=workload_file,
            expected_digest=("sha256:" + "0" * 64 if failure == "digest" else digest),
            expected_size_bytes=(len(payload) + 1 if failure == "size" else len(payload)),
        )

    assert list(work_directory.iterdir()) == []


def test_live_stage_is_content_addressed_read_only_canonical_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT",
        tmp_path,
    )
    payload, plan, files = live_bundle(volume_root)
    work_directory = tmp_path / "work"
    work_directory.mkdir()
    now = datetime(2026, 7, 29, 20, 0, tzinfo=UTC)

    first = stage_uploaded_bundle(
        payload,
        work_directory=work_directory,
        workload_file="repository_repair_large_model_eligibility.py",
        plan=plan,
        now=now,
    )
    second = stage_uploaded_bundle(
        payload,
        work_directory=work_directory,
        workload_file="repository_repair_large_model_eligibility.py",
        plan=plan,
        now=now.replace(minute=1),
    )

    assert second == first
    assert first["receipt_digest"] == tagged_sha256(
        canonical_json({key: value for key, value in first.items() if key != "receipt_digest"})
    )
    destination = Path(plan.workload_bundle_path)
    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) & 0o222 == 0
    assert {path.name for path in work_directory.iterdir() if path.is_file()} == set(files) | {
        BUNDLE_STAGE_RECEIPT_FILENAME
    }


def test_live_stage_receipt_is_accepted_by_the_shared_host_verifier() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest()
    payload, metadata = build_larger_model_bundle(repository_root, manifest)
    plan = LiveStagePlan(
        profile_id=manifest["profile_id"],
        manifest_digest=tagged_sha256(canonical_json(manifest)),
        source_contract_digest=expected_source_contract_digest(manifest),
        bootstrap_source_digest="sha256:" + "b" * 64,
        head_commit="a" * 40,
        network_volume_id="volume-123",
        network_volume_data_center_id="EU-FR-1",
        network_volume_size_gb=50,
        workload_bundle_digest=str(metadata["bundle_digest"]),
        workload_bundle_size_bytes=len(payload),
        workload_bundle_path=str(metadata["bundle_path"]),
        readiness_deadline_epoch=int(time.time()) + 300,
    )
    now = datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    receipt = build_bundle_stage_receipt(
        plan,
        workload_file="repository_repair_large_model_eligibility.py",
        now=now,
    )

    verified = verify_bundle_stage_receipt(
        manifest,
        receipt,
        {
            "id": plan.network_volume_id,
            "dataCenterId": plan.network_volume_data_center_id,
            "size": plan.network_volume_size_gb,
        },
        expected_bundle_digest=plan.workload_bundle_digest,
        expected_bundle_size_bytes=plan.workload_bundle_size_bytes,
        expected_bundle_path=plan.workload_bundle_path,
        now=now,
    )

    assert verified["receipt_digest"] == receipt["receipt_digest"]


def test_live_stage_rejects_symlinked_profile_directory_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    volume_root.mkdir(parents=True)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (volume_root / plan.profile_id).symlink_to(attacker, target_is_directory=True)
    work_directory = tmp_path / "work"
    work_directory.mkdir()

    with pytest.raises(OSError):
        stage_uploaded_bundle(
            payload,
            work_directory=work_directory,
            workload_file="repository_repair_large_model_eligibility.py",
            plan=plan,
        )

    assert list(attacker.iterdir()) == []
    assert list(work_directory.iterdir()) == []


def test_fresh_live_stage_rejects_preexisting_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    monkeypatch.setattr("research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT", tmp_path)
    _payload, plan, _files = live_bundle(volume_root)
    work_directory = tmp_path / "work"
    work_directory.mkdir()
    (work_directory / "result.json").write_text('{"forged":true}\n', encoding="utf-8")
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", token)
    descriptor = open_live_work_directory(work_directory)
    server = BootstrapServer(
        ("127.0.0.1", 0),
        BootstrapHandler,
        plan=plan,
        work_directory=work_directory,
        work_directory_descriptor=descriptor,
    )
    try:
        with pytest.raises(ValueError, match="unjournaled stage artifacts"):
            initialize_or_recover_live_stage(server)
    finally:
        server.server_close()
        os.close(descriptor)


def test_live_stage_cannot_be_used_to_bypass_pilot_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    work_directory = tmp_path / "work"
    work_directory.mkdir()

    with pytest.raises(ValueError, match="eligibility screen"):
        stage_uploaded_bundle(
            payload,
            work_directory=work_directory,
            workload_file="repository_repair_large_model_pilot.py",
            plan=plan,
        )

    assert list(work_directory.iterdir()) == []
    assert not volume_root.exists()


def test_larger_model_eligibility_rejects_the_legacy_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQUINOX_REMOTE_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv(
        "EQUINOX_WORKLOAD_FILE",
        "repository_repair_large_model_eligibility.py",
    )
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "token")
    monkeypatch.delenv("EQUINOX_BUNDLE_LIVE_STAGE", raising=False)

    with pytest.raises(SystemExit, match="requires live bundle staging"):
        main()

    assert not (tmp_path / "work").exists()


def test_live_work_directory_rejects_a_persistent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT",
        tmp_path,
    )
    target = tmp_path / "target"
    target.mkdir()
    work_directory = tmp_path / "work"
    work_directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError):
        open_live_work_directory(work_directory)

    assert list(target.iterdir()) == []


def test_larger_model_pilot_rejects_a_persistent_workdir_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.LIVE_WORK_DIRECTORY_ROOT",
        tmp_path,
    )
    target = tmp_path / "target"
    target.mkdir()
    work_directory = tmp_path / "pilot"
    work_directory.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("EQUINOX_REMOTE_WORKDIR", str(work_directory))
    monkeypatch.setenv(
        "EQUINOX_WORKLOAD_FILE",
        "repository_repair_large_model_pilot.py",
    )
    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "token")
    monkeypatch.delenv("EQUINOX_BUNDLE_LIVE_STAGE", raising=False)

    with pytest.raises(SystemExit, match="pilot work directory was unsafe"):
        main()

    assert list(target.iterdir()) == []


def test_absolute_readiness_deadline_is_bound_and_refuses_late_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    server, _work_directory, descriptor = live_stage_server(
        tmp_path,
        monkeypatch,
        plan=plan,
        token=token,
    )

    class FakeTimer:
        def __init__(
            self,
            interval: float,
            function: object,
            args: tuple[object, ...],
        ) -> None:
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.started = False

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            return

    monkeypatch.setattr("research.runpod.bootstrap_server.Timer", FakeTimer)
    arm_live_stage_deadline(server)
    assert server.deadline_timer is not None
    assert 0 < server.deadline_timer.interval <= 300
    assert server.deadline_timer.started is True

    monkeypatch.setattr(
        "research.runpod.bootstrap_server.time.time",
        lambda: plan.readiness_deadline_epoch + 1,
    )
    thread = serve(server)
    try:
        status, response = request_json(
            server,
            "POST",
            "/bundle",
            token=token,
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert status == HTTPStatus.GONE
        assert response["error"] == "READINESS_DEADLINE_EXPIRED"
        assert server.state == "awaiting_bundle"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        os.close(descriptor)


def test_bootstrap_requires_exact_authenticated_paths_content_and_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, files = live_bundle(volume_root)

    use_prepared_materialization(monkeypatch, plan)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.prepare_stage_evidence",
        lambda server, *, work_directory: prepared_evidence(server, plan),
    )
    server, work_directory, descriptor = live_stage_server(
        tmp_path,
        monkeypatch,
        plan=plan,
        token=token,
    )
    thread = serve(server)
    try:
        status, health = request_json(
            server,
            "GET",
            "/bootstrap-health",
            token=token,
        )
        assert status == HTTPStatus.OK
        assert health == {
            "revision": BOOTSTRAP_TRANSPORT_REVISION,
            "status": "awaiting_bundle",
            **plan.public_identity(),
            "bundle_stage_receipt_digest": None,
            "volume_readiness_receipt_digest": None,
            "torch_retention_evidence_digest": None,
            "dependency_quarantine_evidence_digest": None,
            "dependency_private_tree_digest": None,
            "dependency_lock_digest": None,
            "code_materialization_evidence_digest": None,
            "code_private_tree_digest": None,
            "activation_digest": None,
        }
        status, _ = request_json(
            server,
            "POST",
            "/opaque-proxy-prefix/bundle",
            token=token,
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert status == HTTPStatus.NOT_FOUND
        status, _ = request_json(
            server,
            "POST",
            "/bundle",
            token="wrong-token",
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert status == HTTPStatus.UNAUTHORIZED
        status, _ = request_json(
            server,
            "POST",
            "/bundle",
            token=token,
            body=payload,
            content_type="application/octet-stream",
        )
        assert status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE

        status, health = request_json(
            server,
            "POST",
            "/bundle",
            token=token,
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert status == HTTPStatus.ACCEPTED
        assert health["status"] == "awaiting_stage_activation"
        assert health["bundle_stage_receipt_digest"] == (server.bundle_stage_receipt or {}).get(
            "receipt_digest"
        )
        assert set(path.name for path in work_directory.iterdir() if path.is_file()) == (
            set(files)
            | {
                BUNDLE_STAGE_RECEIPT_FILENAME,
                LIVE_STAGE_PLAN_FILENAME,
                LIVE_STAGE_STATE_FILENAME,
                VOLUME_READINESS_RECEIPT_FILENAME,
                TORCH_RETENTION_EVIDENCE_FILENAME,
                DEPENDENCY_QUARANTINE_EVIDENCE_FILENAME,
                CODE_MATERIALIZATION_EVIDENCE_FILENAME,
            }
        )
        for endpoint, digest_key in (
            ("/bundle-stage-receipt.json", "receipt_digest"),
            ("/volume-readiness-receipt.json", "receipt_digest"),
            ("/torch-retention-evidence.json", "evidence_digest"),
            ("/dependency-quarantine-evidence.json", "evidence_digest"),
            ("/code-materialization-evidence.json", "evidence_digest"),
        ):
            status, evidence = request_json(server, "GET", endpoint, token=token)
            assert status == HTTPStatus.OK
            assert isinstance(evidence[digest_key], str)

        activation = expected_activation(server)
        noncanonical = json.dumps(activation, indent=2).encode()
        status, _ = request_json(
            server,
            "POST",
            "/activate-staged-bundle",
            token=token,
            body=noncanonical,
            content_type="application/json",
        )
        assert status == HTTPStatus.BAD_REQUEST
        status, activated = request_json(
            server,
            "POST",
            "/activate-staged-bundle",
            token=token,
            body=canonical_json(activation),
            content_type="application/json",
        )
        assert status == HTTPStatus.ACCEPTED
        assert activated["status"] == "activated"
        assert activated["activation_digest"] == activation["activation_digest"]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        os.close(descriptor)

    assert server.bundle_activated is True
    assert os.environ["EQUINOX_BUNDLE_ACTIVATION_DIGEST"] == activation["activation_digest"]
    assert os.environ["EQUINOX_LIVE_STAGE_ACTIVATION_SHA256"] == activation["activation_digest"]


def test_restart_recovers_only_the_exact_durable_activated_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    use_prepared_materialization(monkeypatch, plan)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.prepare_stage_evidence",
        lambda server, *, work_directory: prepared_evidence(server, plan),
    )
    first, work_directory, first_descriptor = live_stage_server(
        tmp_path,
        monkeypatch,
        plan=plan,
        token=token,
    )
    first_thread = serve(first)
    try:
        upload_status, _ = request_json(
            first,
            "POST",
            "/bundle",
            token=token,
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert upload_status == HTTPStatus.ACCEPTED
        activation = expected_activation(first)
        activation_status, _ = request_json(
            first,
            "POST",
            "/activate-staged-bundle",
            token=token,
            body=canonical_json(activation),
            content_type="application/json",
        )
        assert activation_status == HTTPStatus.ACCEPTED
        first_thread.join(timeout=2)
    finally:
        first.server_close()
        os.close(first_descriptor)

    assert (work_directory / LIVE_STAGE_ACTIVATION_FILENAME).is_file()
    assert (work_directory / LIVE_STAGE_STATE_FILENAME).is_file()
    runner_path = work_directory / "remote_runner.sh"
    runner_path.chmod(0o644)
    runner_path.write_bytes(b"tampered runner\n")
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.time.time",
        lambda: plan.readiness_deadline_epoch + 10,
    )
    recovered_descriptor = open_live_work_directory(work_directory)
    recovered = BootstrapServer(
        ("127.0.0.1", 0),
        BootstrapHandler,
        plan=plan,
        work_directory=work_directory,
        work_directory_descriptor=recovered_descriptor,
    )
    try:
        initialize_or_recover_live_stage(recovered)
        assert recovered.bundle_activated is True
        assert recovered.activation == activation
        assert runner_path.read_bytes() != b"tampered runner\n"
        assert os.environ["EQUINOX_BUNDLE_ACTIVATION_DIGEST"] == activation["activation_digest"]
    finally:
        recovered.server_close()
        os.close(recovered_descriptor)

    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", "fresh-allocation-token")
    stale_descriptor = open_live_work_directory(work_directory)
    stale_server = BootstrapServer(
        ("127.0.0.1", 0),
        BootstrapHandler,
        plan=plan,
        work_directory=work_directory,
        work_directory_descriptor=stale_descriptor,
    )
    try:
        with pytest.raises(ValueError, match="different bytes"):
            initialize_or_recover_live_stage(stale_server)
    finally:
        stale_server.server_close()
        os.close(stale_descriptor)

    monkeypatch.setenv("EQUINOX_RESULT_TOKEN", token)
    switched = replace(plan, head_commit="c" * 40)
    switched_descriptor = open_live_work_directory(work_directory)
    switched_server = BootstrapServer(
        ("127.0.0.1", 0),
        BootstrapHandler,
        plan=switched,
        work_directory=work_directory,
        work_directory_descriptor=switched_descriptor,
    )
    try:
        with pytest.raises(ValueError, match="different bytes"):
            initialize_or_recover_live_stage(switched_server)
    finally:
        switched_server.server_close()
        os.close(switched_descriptor)


def test_concurrent_duplicate_upload_is_fenced_to_one_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    use_prepared_materialization(monkeypatch, plan)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.prepare_stage_evidence",
        lambda server, *, work_directory: prepared_evidence(server, plan),
    )
    entered = Event()
    release = Event()
    calls = 0
    real_stage = stage_uploaded_bundle

    def slow_stage(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = real_stage(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=2)
        return result

    monkeypatch.setattr(
        "research.runpod.bootstrap_server.stage_uploaded_bundle",
        slow_stage,
    )
    server, _work_directory, descriptor = live_stage_server(
        tmp_path,
        monkeypatch,
        plan=plan,
        token=token,
    )
    server_thread = serve(server)
    first_result: list[tuple[int, dict[str, object]]] = []
    first_thread = Thread(
        target=lambda: first_result.append(
            request_json(
                server,
                "POST",
                "/bundle",
                token=token,
                body=payload,
                content_type=BUNDLE_CONTENT_TYPE,
            )
        )
    )
    try:
        first_thread.start()
        assert entered.wait(timeout=2)
        duplicate_status, duplicate = request_json(
            server,
            "POST",
            "/bundle",
            token=token,
            body=payload,
            content_type=BUNDLE_CONTENT_TYPE,
        )
        assert duplicate_status == HTTPStatus.CONFLICT
        assert duplicate["error"] == "BUNDLE_UPLOAD_NOT_AVAILABLE"
        release.set()
        first_thread.join(timeout=2)
        assert first_result[0][0] == HTTPStatus.ACCEPTED
        assert calls == 1
    finally:
        release.set()
        server.shutdown()
        first_thread.join(timeout=2)
        server_thread.join(timeout=2)
        server.server_close()
        os.close(descriptor)


def test_dropped_upload_response_is_reconciled_by_health_without_reupload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "test-transport-token"
    volume_root = tmp_path / "volume/workload-bundles"
    monkeypatch.setattr("research.runpod.bootstrap_server.BUNDLE_VOLUME_ROOT", volume_root)
    payload, plan, _files = live_bundle(volume_root)
    use_prepared_materialization(monkeypatch, plan)
    monkeypatch.setattr(
        "research.runpod.bootstrap_server.prepare_stage_evidence",
        lambda server, *, work_directory: prepared_evidence(server, plan),
    )
    original_write = BootstrapHandler._write_json
    dropped = False

    def drop_acceptance_once(
        handler: BootstrapHandler,
        status: HTTPStatus,
        response: dict[str, object],
    ) -> None:
        nonlocal dropped
        if (
            not dropped
            and status == HTTPStatus.ACCEPTED
            and response.get("status") == "awaiting_stage_activation"
        ):
            dropped = True
            handler.close_connection = True
            return
        original_write(handler, status, response)

    monkeypatch.setattr(BootstrapHandler, "_write_json", drop_acceptance_once)
    server, _work_directory, descriptor = live_stage_server(
        tmp_path,
        monkeypatch,
        plan=plan,
        token=token,
    )
    thread = serve(server)
    connection = HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(
            "POST",
            "/bundle",
            body=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": BUNDLE_CONTENT_TYPE,
                "Content-Length": str(len(payload)),
            },
        )
        with pytest.raises(RemoteDisconnected):
            connection.getresponse()
        status, health = request_json(
            server,
            "GET",
            "/bootstrap-health",
            token=token,
        )
        assert status == HTTPStatus.OK
        assert health["status"] == "awaiting_stage_activation"
        assert health["bundle_stage_receipt_digest"] == (server.bundle_stage_receipt or {}).get(
            "receipt_digest"
        )
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        os.close(descriptor)

    assert dropped is True
    assert server.bundle_ready is True


def test_bundle_rejects_duplicate_members_and_expansion_bomb(tmp_path: Path) -> None:
    workload_file = "branching_sequence_ladder.py"
    expected = sorted(expected_bundle_files(workload_file))
    duplicate = io.BytesIO()
    with tarfile.open(fileobj=duplicate, mode="w:xz") as archive:
        for name in [*expected, expected[0]]:
            content = b"x"
            metadata = tarfile.TarInfo(name)
            metadata.size = len(content)
            archive.addfile(metadata, io.BytesIO(content))
    with pytest.raises(ValueError, match="allowlist"):
        install_bundle(
            duplicate.getvalue(),
            work_directory=tmp_path,
            workload_file=workload_file,
        )

    bomb = io.BytesIO()
    with tarfile.open(fileobj=bomb, mode="w:xz") as archive:
        for name in expected:
            content = b"x" * (17 * 1024 * 1024 if name == expected[0] else 1)
            metadata = tarfile.TarInfo(name)
            metadata.size = len(content)
            archive.addfile(metadata, io.BytesIO(content))
    assert len(bomb.getvalue()) < 2 * 1024 * 1024
    with pytest.raises(ValueError, match="expanded"):
        install_bundle(
            bomb.getvalue(),
            work_directory=tmp_path,
            workload_file=workload_file,
        )
