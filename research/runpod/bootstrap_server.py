from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MAXIMUM_BUNDLE_BYTES = 2 * 1024 * 1024
BUNDLE_VOLUME_ROOT = Path("/workspace/equinox-state/workload-bundles")
COMMON_BUNDLE_FILES = frozenset({"remote_runner.sh", "result_server.py"})
LARGER_MODEL_BUNDLE_SUPPORT_FILES = frozenset(
    {
        "larger-model-eligibility.json",
        "larger_model_gate.py",
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_env_v32.py",
        "repository_repair_env_v33.py",
        "repository_repair_large_model_eligibility.py",
        "repository_repair_large_model_pilot.py",
        "repository_repair_large_model_study.py",
        "repository_repair_large_model_trainer.py",
    }
)
WORKLOAD_SUPPORT_FILES = {
    "branching_sequence_ladder.py": frozenset(),
    "repository_repair_rl.py": frozenset({"repository_repair_env.py"}),
    "repository_repair_study.py": frozenset(
        {"repository_repair_env.py", "repository_repair_rl.py"}
    ),
    "repository_repair_study_v31.py": frozenset(
        {
            "repository_repair_env.py",
            "repository_repair_env_v31.py",
            "repository_repair_rl.py",
            "repository_repair_study.py",
        }
    ),
    "repository_repair_eligibility.py": frozenset(
        {
            "repository_repair_env.py",
            "repository_repair_rl.py",
            "repository_repair_study.py",
        }
    ),
    "repository_repair_large_model_eligibility.py": LARGER_MODEL_BUNDLE_SUPPORT_FILES,
    "repository_repair_large_model_pilot.py": LARGER_MODEL_BUNDLE_SUPPORT_FILES,
    "research/runpod/revision30_external_eval.py": frozenset(
        {
            "external_eval_remote_runner.sh",
            "research/__init__.py",
            "research/external/revision30_task_pack.py",
            "research/frozen/revision30-external-pack.json",
            "research/runpod/__init__.py",
            "research/runpod/external_eval_transport.py",
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_rl.py",
        }
    ),
    "research/runpod/revision31_external_eval.py": frozenset(
        {
            "external_eval_remote_runner.sh",
            "research/__init__.py",
            "research/external/revision30_task_pack.py",
            "research/external/revision31_task_pack.py",
            "research/frozen/revision31-external-pack.json",
            "research/runpod/__init__.py",
            "research/runpod/external_eval_transport.py",
            "research/runpod/repository_repair_env.py",
            "research/runpod/repository_repair_env_v31.py",
            "research/runpod/repository_repair_rl.py",
            "research/runpod/revision30_external_eval.py",
        }
    ),
}
EXTERNAL_EVALUATION_WORKLOAD = "research/runpod/revision30_external_eval.py"
EXTERNAL_EVALUATION_WORKLOADS = frozenset(
    {
        EXTERNAL_EVALUATION_WORKLOAD,
        "research/runpod/revision31_external_eval.py",
    }
)


def expected_bundle_files(workload_file: str) -> frozenset[str]:
    try:
        support_files = WORKLOAD_SUPPORT_FILES[workload_file]
    except KeyError as error:
        raise ValueError("Unsupported workload file.") from error
    common_files = (
        frozenset({"external_eval_remote_runner.sh"})
        if workload_file in EXTERNAL_EVALUATION_WORKLOADS
        else COMMON_BUNDLE_FILES
    )
    return common_files | support_files | {workload_file}


def runner_file(workload_file: str) -> str:
    expected_bundle_files(workload_file)
    return (
        "external_eval_remote_runner.sh"
        if workload_file in EXTERNAL_EVALUATION_WORKLOADS
        else "remote_runner.sh"
    )


def install_bundle(
    payload: bytes,
    *,
    work_directory: Path,
    workload_file: str,
) -> None:
    expected_files = expected_bundle_files(workload_file)
    staging_directory = Path(tempfile.mkdtemp(prefix=".bundle-", dir=work_directory))
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = archive.getmembers()
            observed_files = {member.name for member in members}
            if observed_files != expected_files:
                raise ValueError("Bundle file set did not match the workload allowlist.")
            for member in members:
                relative_path = Path(member.name)
                if (
                    not member.isfile()
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or any(part in {"", "."} for part in relative_path.parts)
                    or member.name not in expected_files
                ):
                    raise ValueError("Bundle members must be allowlisted regular files.")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Bundle member could not be read.")
                destination = staging_directory / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
        for filename in sorted(expected_files):
            destination = work_directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_directory / filename, destination)
        directory_descriptor = os.open(work_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)


def install_environment_bundle(
    encoded_payload: str,
    *,
    work_directory: Path,
    workload_file: str,
    expected_digest: str | None = None,
) -> None:
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Environment bundle was not valid base64.") from error
    if not payload or len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Environment bundle size was invalid.")
    if expected_digest is not None:
        observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if not hmac.compare_digest(observed_digest, expected_digest):
            raise ValueError("Environment bundle digest did not match.")
    install_bundle(
        payload,
        work_directory=work_directory,
        workload_file=workload_file,
    )


def install_volume_bundle(
    volume_path: Path,
    *,
    work_directory: Path,
    workload_file: str,
    expected_digest: str,
    expected_size_bytes: int,
) -> None:
    if (
        not volume_path.is_absolute()
        or volume_path.parent.parent != BUNDLE_VOLUME_ROOT
        or not re.fullmatch(r"[A-Za-z0-9._@-]+", volume_path.parent.name)
        or volume_path.name != f"{expected_digest.removeprefix('sha256:')}.tar.xz"
    ):
        raise ValueError("Volume bundle path was not content-addressed.")
    if (
        not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest)
        or type(expected_size_bytes) is not int
        or not 0 < expected_size_bytes <= MAXIMUM_BUNDLE_BYTES
    ):
        raise ValueError("Volume bundle identity was invalid.")
    path_metadata = os.lstat(volume_path)
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError("Volume bundle path was not a regular file.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(volume_path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != expected_size_bytes
            or metadata.st_dev != path_metadata.st_dev
            or metadata.st_ino != path_metadata.st_ino
        ):
            raise ValueError("Volume bundle file metadata was invalid.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(expected_size_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) != expected_size_bytes:
        raise ValueError("Volume bundle size did not match.")
    observed_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise ValueError("Volume bundle digest did not match.")
    install_bundle(
        payload,
        work_directory=work_directory,
        workload_file=workload_file,
    )


class BootstrapServer(HTTPServer):
    bundle_ready = False


class BootstrapHandler(BaseHTTPRequestHandler):
    server: BootstrapServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        expected = f"Bearer {os.environ['EQUINOX_RESULT_TOKEN']}"
        observed = self.headers.get("Authorization", "")
        return hmac.compare_digest(observed, expected)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path != "/bootstrap-health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
            return
        self._write_json(HTTPStatus.OK, {"status": "awaiting_bundle"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            content_length = -1
        if content_length <= 0 or content_length > MAXIMUM_BUNDLE_BYTES:
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "INVALID_BUNDLE_SIZE"},
            )
            return
        payload = self.rfile.read(content_length)
        if len(payload) != content_length:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "INCOMPLETE_BUNDLE"},
            )
            return
        try:
            install_bundle(
                payload,
                work_directory=Path(os.environ["EQUINOX_REMOTE_WORKDIR"]),
                workload_file=os.environ["EQUINOX_WORKLOAD_FILE"],
            )
        except (OSError, tarfile.TarError, ValueError):
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "INVALID_BUNDLE"},
            )
            return
        self.server.bundle_ready = True
        self._write_json(HTTPStatus.ACCEPTED, {"status": "bundle_installed"})


def main() -> None:
    work_directory = Path(os.environ["EQUINOX_REMOTE_WORKDIR"])
    work_directory.mkdir(parents=True, exist_ok=True)
    workload_file = os.environ["EQUINOX_WORKLOAD_FILE"]
    expected_bundle_files(workload_file)
    if not os.environ.get("EQUINOX_RESULT_TOKEN"):
        raise SystemExit("EQUINOX_RESULT_TOKEN is required.")
    encoded_environment_bundle = os.environ.pop("EQUINOX_BUNDLE_B64", "")
    volume_bundle_path = os.environ.pop("EQUINOX_BUNDLE_VOLUME_PATH", "")
    expected_environment_bundle_digest = os.environ.pop(
        "EQUINOX_BUNDLE_SHA256",
        "",
    )
    expected_environment_bundle_size = os.environ.pop(
        "EQUINOX_BUNDLE_SIZE_BYTES",
        "",
    )
    if encoded_environment_bundle and volume_bundle_path:
        raise SystemExit("Only one workload bundle transport may be configured.")
    if encoded_environment_bundle:
        if expected_environment_bundle_size:
            raise SystemExit("Inline bundles must not declare EQUINOX_BUNDLE_SIZE_BYTES.")
        if (
            len(expected_environment_bundle_digest) != len("sha256:") + 64
            or not expected_environment_bundle_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in expected_environment_bundle_digest.removeprefix("sha256:")
            )
        ):
            raise SystemExit("EQUINOX_BUNDLE_SHA256 is invalid.")
        install_environment_bundle(
            encoded_environment_bundle,
            work_directory=work_directory,
            workload_file=workload_file,
            expected_digest=expected_environment_bundle_digest,
        )
    elif volume_bundle_path:
        try:
            expected_size_bytes = int(expected_environment_bundle_size)
        except ValueError as error:
            raise SystemExit("EQUINOX_BUNDLE_SIZE_BYTES is invalid.") from error
        try:
            install_volume_bundle(
                Path(volume_bundle_path),
                work_directory=work_directory,
                workload_file=workload_file,
                expected_digest=expected_environment_bundle_digest,
                expected_size_bytes=expected_size_bytes,
            )
        except (OSError, tarfile.TarError, ValueError) as error:
            raise SystemExit(f"Volume bundle could not be installed: {error}") from error
        os.environ.update(
            {
                "EQUINOX_BUNDLE_VOLUME_PATH": volume_bundle_path,
                "EQUINOX_BUNDLE_SHA256": expected_environment_bundle_digest,
                "EQUINOX_BUNDLE_SIZE_BYTES": expected_environment_bundle_size,
            }
        )
    else:
        if expected_environment_bundle_digest or expected_environment_bundle_size:
            raise SystemExit(
                "Bundle identity requires EQUINOX_BUNDLE_B64 or EQUINOX_BUNDLE_VOLUME_PATH."
            )
        port = int(os.environ.get("EQUINOX_BOOTSTRAP_PORT", "8000"))
        server = BootstrapServer(("0.0.0.0", port), BootstrapHandler)
        while not server.bundle_ready:
            server.handle_request()
        server.server_close()
    time.sleep(0.5)
    os.execvpe(
        "bash",
        ["bash", str(work_directory / runner_file(workload_file))],
        os.environ,
    )


if __name__ == "__main__":
    main()
