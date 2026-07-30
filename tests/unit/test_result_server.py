from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def progress_payload(**updates: object) -> bytes:
    progress: dict[str, object] = {
        "schema_version": 2,
        "phase": "training",
        "message": "Training update committed.",
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "attempt": 1,
    }
    progress.update(updates)
    return (
        json.dumps(
            progress,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def replace_progress(root: Path, payload: bytes) -> None:
    pending = root / ".progress-test-pending"
    pending.write_bytes(payload)
    pending.replace(root / "progress.json")


def authorized_request(port: int, path: str = "/progress.json") -> urllib.request.Request:
    return urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers={"Authorization": "Bearer test-token"},
    )


def assert_http_status(request: urllib.request.Request, status: int) -> None:
    with pytest.raises(urllib.error.HTTPError) as rejected:
        urllib.request.urlopen(request, timeout=1)
    assert rejected.value.code == status


def write_integrity_manifest(
    root: Path,
    environment: dict[str, str],
    artifact_name: str,
) -> None:
    identity = {
        "proof_id": environment["EQUINOX_PROOF_ID"],
        "run_identity": environment.get("EQUINOX_RUN_IDENTITY", ""),
        "private_runtime_revision": environment.get(
            "EQUINOX_PRIVATE_RUNTIME_REVISION",
            "",
        ),
        "workload_file": environment["EQUINOX_WORKLOAD_FILE"],
        "model_id": environment.get("EQUINOX_RL_MODEL_ID", ""),
        "optimization_seed": environment.get("EQUINOX_RL_SEED", ""),
        "study_condition": environment.get("EQUINOX_STUDY_CONDITION", ""),
        "bundle_digest": environment.get("EQUINOX_BUNDLE_SHA256", ""),
        "bundle_activation_digest": environment.get(
            "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
            "",
        ),
        "source_contract_digest": environment.get(
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
            "",
        ),
        "code_private_tree_digest": environment.get(
            "EQUINOX_CODE_PRIVATE_TREE_SHA256",
            "",
        ),
        "dependency_private_tree_digest": environment.get(
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
            "",
        ),
        "dependency_lock_digest": environment.get(
            "EQUINOX_DEPENDENCY_LOCK_SHA256",
            "",
        ),
    }
    identity_digest = "sha256:" + hashlib.sha256(canonical_json(identity)).hexdigest()
    payload = (root / artifact_name).read_bytes()
    domain = "equinox/checkpoint/v1" if artifact_name == "adapter.tgz" else "equinox/result/v1"
    artifact_material: dict[str, object] = {
        "domain": domain,
        "launch_identity_digest": identity_digest,
        "path": artifact_name,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    artifact = {
        **artifact_material,
        "hmac": (
            "hmac-sha256:"
            + hmac.new(
                environment["EQUINOX_RESULT_TOKEN"].encode(),
                (domain + "\0").encode() + canonical_json(artifact_material),
                hashlib.sha256,
            ).hexdigest()
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "revision": "token-bound-runner-resume@1",
        "workload_file": environment["EQUINOX_WORKLOAD_FILE"],
        "launch_identity_digest": identity_digest,
        "generation": 1,
        "parent_manifest_sha256": "sha256:" + "0" * 64,
        "directories": [],
        "artifacts": [artifact],
    }
    manifest["hmac"] = (
        "hmac-sha256:"
        + hmac.new(
            environment["EQUINOX_RESULT_TOKEN"].encode(),
            b"equinox/runner-resume-manifest/v1\0" + canonical_json(manifest),
            hashlib.sha256,
        ).hexdigest()
    )
    (root / "runner-resume-state.json").write_bytes(canonical_json(manifest) + b"\n")


@pytest.mark.parametrize("result_host", [None, "127.0.0.1"])
def test_result_transport_requires_auth_and_exposes_only_allowlisted_files(
    tmp_path: Path,
    result_host: str | None,
) -> None:
    initial_progress = progress_payload()
    (tmp_path / "progress.json").write_bytes(initial_progress)
    (tmp_path / "private.txt").write_text("must not be served", encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": "runpod-proof-test-transport",
        "EQUINOX_WORKLOAD_FILE": "repository_repair_rl.py",
        "EQUINOX_RESULT_PORT": str(port),
    }
    if result_host is None:
        environment.pop("EQUINOX_RESULT_HOST", None)
    else:
        environment["EQUINOX_RESULT_HOST"] = result_host
    process = subprocess.Popen(
        [sys.executable, "research/runpod/result_server.py"],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        endpoint = f"http://127.0.0.1:{port}/progress.json"
        for _ in range(50):
            if process.poll() is not None:
                raise AssertionError(
                    "result server exited before readiness: "
                    + (process.stderr.read() if process.stderr is not None else "")
                )
            try:
                urllib.request.urlopen(endpoint, timeout=0.1)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    exc.close()
                    break
                exc.close()
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError(
                "result server did not become ready"
                + (
                    f": {process.stderr.read()}"
                    if process.poll() is not None and process.stderr is not None
                    else ""
                )
            )

        authorized = urllib.request.Request(
            endpoint,
            headers={"Authorization": "Bearer test-token"},
        )
        with urllib.request.urlopen(authorized, timeout=1) as response:
            assert response.read() == initial_progress
            assert response.headers["Cache-Control"] == "no-store"

        hidden = urllib.request.Request(
            f"http://127.0.0.1:{port}/private.txt",
            headers={"Authorization": "Bearer test-token"},
        )
        try:
            urllib.request.urlopen(hidden, timeout=1)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("non-allowlisted file was served")
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_progress_transport_rejects_junk_and_regressions_without_advancing_state(
    tmp_path: Path,
) -> None:
    replace_progress(
        tmp_path,
        progress_payload(
            update=4,
            elapsed_seconds=12.0,
            evaluation_completed=2,
        ),
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": "runpod-proof-test-progress-monotonic",
        "EQUINOX_WORKLOAD_FILE": "repository_repair_rl.py",
        "EQUINOX_RESULT_HOST": "127.0.0.1",
        "EQUINOX_RESULT_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "research/runpod/result_server.py"],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = authorized_request(port)
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen(request, timeout=0.1) as response:
                    assert json.loads(response.read())["update"] == 4
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("result server did not become ready")

        replace_progress(tmp_path, b'{"not":"valid progress"}\n')
        assert_http_status(request, 404)

        replace_progress(
            tmp_path,
            progress_payload(
                update=4,
                elapsed_seconds=12.0,
                evaluation_completed=2,
                message="An untrusted message-only toggle.",
            ),
        )
        assert_http_status(request, 404)

        replace_progress(
            tmp_path,
            progress_payload(
                update=3,
                elapsed_seconds=13.0,
                evaluation_completed=3,
            ),
        )
        assert_http_status(request, 404)

        valid_after_update_regression = progress_payload(
            update=5,
            elapsed_seconds=12.5,
            evaluation_completed=3,
        )
        replace_progress(tmp_path, valid_after_update_regression)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == valid_after_update_regression

        replace_progress(
            tmp_path,
            progress_payload(
                update=6,
                elapsed_seconds=11.0,
                evaluation_completed=4,
            ),
        )
        assert_http_status(request, 404)

        valid_after_rejections = progress_payload(
            update=5,
            elapsed_seconds=13.0,
            evaluation_completed=4,
        )
        replace_progress(tmp_path, valid_after_rejections)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == valid_after_rejections

        replace_progress(
            tmp_path,
            progress_payload(
                phase="protocol_evaluation",
                message="Forged phase rollback.",
                update=6,
                elapsed_seconds=14.0,
            ),
        )
        assert_http_status(request, 404)

        valid_after_phase_regression = progress_payload(
            phase="checkpointing",
            message="Authenticated checkpoint committed.",
            update=6,
            elapsed_seconds=14.0,
            checkpoint_generation=1,
        )
        replace_progress(tmp_path, valid_after_phase_regression)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == valid_after_phase_regression
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_progress_terminal_state_is_sticky_within_an_attempt(
    tmp_path: Path,
) -> None:
    terminal = progress_payload(
        phase="failed",
        message="Workload failed.",
        update=7,
        elapsed_seconds=21.0,
        error="WORKLOAD_FAILED",
    )
    replace_progress(tmp_path, terminal)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": "runpod-proof-test-progress-terminal",
        "EQUINOX_WORKLOAD_FILE": "repository_repair_rl.py",
        "EQUINOX_RESULT_HOST": "127.0.0.1",
        "EQUINOX_RESULT_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "research/runpod/result_server.py"],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = authorized_request(port)
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen(request, timeout=0.1) as response:
                    assert response.read() == terminal
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("result server did not become ready")

        replace_progress(
            tmp_path,
            progress_payload(
                phase="training",
                message="Forged terminal rollback.",
                update=8,
                elapsed_seconds=22.0,
            ),
        )
        assert_http_status(request, 404)

        enriched_terminal = progress_payload(
            phase="failed",
            message="Workload failed.",
            update=7,
            elapsed_seconds=21.0,
            error="WORKLOAD_FAILED",
            remote_error={
                "code": "WORKLOAD_FAILED",
                "message": "Workload failed.",
                "exit_code": 7,
            },
        )
        replace_progress(tmp_path, enriched_terminal)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == enriched_terminal

        replace_progress(
            tmp_path,
            progress_payload(
                phase="failed",
                message="Toggled terminal payload.",
                update=7,
                elapsed_seconds=21.0,
                error="WORKLOAD_FAILED",
            ),
        )
        assert_http_status(request, 404)

        next_attempt = progress_payload(
            phase="resuming",
            message="Retrying from the authenticated checkpoint.",
            attempt=2,
            update=7,
            elapsed_seconds=21.0,
        )
        replace_progress(tmp_path, next_attempt)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == next_attempt

        replace_progress(
            tmp_path,
            progress_payload(
                phase="training",
                message="Forged attempt rollback.",
                attempt=1,
                update=8,
                elapsed_seconds=22.0,
            ),
        )
        assert_http_status(request, 404)

        dependency_setup = progress_payload(
            phase="dependency_setup",
            message="Second attempt dependencies ready.",
            attempt=2,
            update=7,
            elapsed_seconds=0.0,
        )
        replace_progress(tmp_path, dependency_setup)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == dependency_setup

        model_loading = progress_payload(
            phase="model_loading",
            message="Second attempt model loaded.",
            attempt=2,
            update=7,
            elapsed_seconds=1.0,
        )
        replace_progress(tmp_path, model_loading)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == model_loading

        checkpoint_restoring = progress_payload(
            phase="resuming",
            message="Restoring authenticated checkpoint.",
            attempt=2,
            update=7,
            elapsed_seconds=2.0,
        )
        replace_progress(tmp_path, checkpoint_restoring)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == checkpoint_restoring

        valid_second_attempt = progress_payload(
            phase="training",
            message="Second attempt training.",
            attempt=2,
            update=8,
            elapsed_seconds=3.0,
            effective_policy_update_count=10,
            retention_rollback_count=0,
        )
        replace_progress(tmp_path, valid_second_attempt)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == valid_second_attempt

        retained_rollback = progress_payload(
            phase="checkpointing",
            message="Unvalidated policy update rolled back.",
            attempt=2,
            update=8,
            elapsed_seconds=4.0,
            effective_policy_update_count=5,
            retention_rollback_count=1,
        )
        replace_progress(tmp_path, retained_rollback)
        with urllib.request.urlopen(request, timeout=1) as response:
            assert response.read() == retained_rollback
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":2,"phase":"training","message":"bad encoding",'
        b'"branch_width":4,"complexity_strategy":"adaptive","attempt":1}\n',
        b'{"attempt":1,"branch_width":4,"complexity_strategy":"adaptive",'
        b'"message":"bad","phase":"unknown","schema_version":2}\n',
        progress_payload(unexpected_field=True),
        progress_payload(update=True),
        progress_payload(elapsed_seconds="not-a-number"),
        progress_payload(branch_snapshots=[None] * 4097),
    ],
)
def test_progress_transport_rejects_noncanonical_or_out_of_schema_payloads(
    tmp_path: Path,
    payload: bytes,
) -> None:
    replace_progress(tmp_path, payload)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": "runpod-proof-test-progress-schema",
        "EQUINOX_WORKLOAD_FILE": "repository_repair_rl.py",
        "EQUINOX_RESULT_HOST": "127.0.0.1",
        "EQUINOX_RESULT_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "research/runpod/result_server.py"],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = authorized_request(port)
    try:
        for _ in range(50):
            if process.poll() is not None:
                raise AssertionError(
                    "result server exited before readiness: "
                    + (process.stderr.read() if process.stderr is not None else "")
                )
            try:
                urllib.request.urlopen(request, timeout=0.1)
            except urllib.error.HTTPError as error:
                assert error.code == 404
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("result server did not reject invalid progress")
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_result_transport_requires_integrity_commit_for_final_result(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text('{"experiment_completed":true}\n', encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": "runpod-proof-test-integrity",
        "EQUINOX_WORKLOAD_FILE": "repository_repair_rl.py",
        "EQUINOX_RESULT_HOST": "127.0.0.1",
        "EQUINOX_RESULT_PORT": str(port),
    }
    write_integrity_manifest(tmp_path, environment, "result.json")
    process = subprocess.Popen(
        [sys.executable, "research/runpod/result_server.py"],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        endpoint = f"http://127.0.0.1:{port}/result.json"
        request = urllib.request.Request(
            endpoint,
            headers={"Authorization": "Bearer test-token"},
        )
        for _ in range(50):
            try:
                with urllib.request.urlopen(request, timeout=0.1) as response:
                    assert response.read() == b'{"experiment_completed":true}\n'
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("result server did not become ready")

        integrity_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/integrity.json",
            headers={"Authorization": "Bearer test-token"},
        )
        with urllib.request.urlopen(integrity_request, timeout=1) as response:
            integrity = json.loads(response.read())
        assert integrity["generation"] == 1
        assert integrity["artifacts"][0]["path"] == "result.json"

        result_path.write_text('{"experiment_completed":false}\n', encoding="utf-8")
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(request, timeout=1)
        assert rejected.value.code == 404
    finally:
        process.terminate()
        process.wait(timeout=5)
