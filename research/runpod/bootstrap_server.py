from __future__ import annotations

import base64
import binascii
import hmac
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MAXIMUM_BUNDLE_BYTES = 2 * 1024 * 1024
COMMON_BUNDLE_FILES = frozenset({"remote_runner.sh", "result_server.py"})
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
}
EXTERNAL_EVALUATION_WORKLOAD = "research/runpod/revision30_external_eval.py"


def expected_bundle_files(workload_file: str) -> frozenset[str]:
    try:
        support_files = WORKLOAD_SUPPORT_FILES[workload_file]
    except KeyError as error:
        raise ValueError("Unsupported workload file.") from error
    common_files = (
        frozenset({"external_eval_remote_runner.sh"})
        if workload_file == EXTERNAL_EVALUATION_WORKLOAD
        else COMMON_BUNDLE_FILES
    )
    return common_files | support_files | {workload_file}


def runner_file(workload_file: str) -> str:
    expected_bundle_files(workload_file)
    return (
        "external_eval_remote_runner.sh"
        if workload_file == EXTERNAL_EVALUATION_WORKLOAD
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
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
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
) -> None:
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Environment bundle was not valid base64.") from error
    if not payload or len(payload) > MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Environment bundle size was invalid.")
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
    if encoded_environment_bundle:
        install_environment_bundle(
            encoded_environment_bundle,
            work_directory=work_directory,
            workload_file=workload_file,
        )
    else:
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
