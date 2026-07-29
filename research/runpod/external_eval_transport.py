"""Authenticated artifact upload and result transport for external evaluation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp")).resolve()
TOKEN = os.environ.get("EQUINOX_RESULT_TOKEN", "")
INPUT_ROOT = ROOT / "inputs"
MAXIMUM_MANIFEST_BYTES = 256 * 1024
MAXIMUM_ADAPTER_BYTES = 200 * 1024 * 1024
MAXIMUM_READY_BYTES = 1_024
ADAPTER_ARCHIVE = re.compile(r"^adapter-[a-f0-9]{64}\.tgz$")
READ_ROUTES = {
    "/error.log": ("error.log", "text/plain; charset=utf-8"),
    "/exit_code": ("exit_code", "text/plain; charset=utf-8"),
    "/progress.json": ("progress.json", "application/json"),
    "/result.json": ("result.json", "application/json"),
}


def configured_result_host() -> str:
    """Return the explicit bind host, preserving wildcard exposure by default."""

    return os.environ.get("EQUINOX_RESULT_HOST", "0.0.0.0")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("external-evaluation input manifest is invalid")
    adapters = manifest.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("external-evaluation input manifest has no adapters")
    observed_ids = set()
    observed_files = set()
    for adapter in adapters:
        if not isinstance(adapter, dict):
            raise ValueError("external-evaluation adapter entry is invalid")
        adapter_id = adapter.get("adapter_id")
        filename = adapter.get("archive_filename")
        size_bytes = adapter.get("size_bytes")
        sha256 = adapter.get("sha256")
        if (
            not isinstance(adapter_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", adapter_id)
            or adapter_id in observed_ids
            or not isinstance(filename, str)
            or ADAPTER_ARCHIVE.fullmatch(filename) is None
            or filename in observed_files
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not 1 <= size_bytes <= MAXIMUM_ADAPTER_BYTES
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", sha256)
            or filename != f"adapter-{sha256}.tgz"
        ):
            raise ValueError("external-evaluation adapter identity is invalid")
        observed_ids.add(adapter_id)
        observed_files.add(filename)
    return manifest


def verify_ready_payload(root: Path, payload: bytes) -> dict[str, Any]:
    ready = json.loads(payload)
    if not isinstance(ready, dict) or set(ready) != {"manifest_sha256"}:
        raise ValueError("external-evaluation ready payload is invalid")
    expected_manifest_sha256 = ready["manifest_sha256"]
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[a-f0-9]{64}",
        expected_manifest_sha256,
    ):
        raise ValueError("external-evaluation ready digest is invalid")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("external-evaluation manifest digest does not match")
    manifest = load_input_manifest(manifest_path)
    for adapter in manifest["adapters"]:
        archive = root / adapter["archive_filename"]
        if (
            not archive.is_file()
            or archive.stat().st_size != adapter["size_bytes"]
            or sha256_file(archive) != adapter["sha256"]
        ):
            raise ValueError("external-evaluation adapter upload is incomplete")
    return ready


class ExternalEvalTransportHandler(BaseHTTPRequestHandler):
    server_version = "EquinoxExternalEvalTransport/1"

    def _authorized(self) -> bool:
        provided = self.headers.get("Authorization", "")
        return bool(TOKEN) and hmac.compare_digest(provided, f"Bearer {TOKEN}")

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = canonical_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _reject_unless_authorized(self) -> bool:
        if self._authorized():
            return False
        self._json(HTTPStatus.UNAUTHORIZED, {"error": "UNAUTHORIZED"})
        return True

    def do_GET(self) -> None:
        if self._reject_unless_authorized():
            return
        path = urlsplit(self.path).path
        if path == "/input-status":
            INPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._json(
                HTTPStatus.OK,
                {
                    "manifest_received": (INPUT_ROOT / "manifest.json").is_file(),
                    "ready": (INPUT_ROOT / "ready.json").is_file(),
                    "adapter_archives_received": len(list(INPUT_ROOT.glob("adapter-*.tgz"))),
                },
            )
            return
        route = READ_ROUTES.get(path)
        if route is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        relative_path, media_type = route
        target = (ROOT / relative_path).resolve()
        if target.parent != ROOT or not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        payload = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _store_input(self) -> None:
        if self._reject_unless_authorized():
            return
        path = urlsplit(self.path).path
        name = path.removeprefix("/inputs/")
        if path == "/inputs/manifest.json":
            maximum_bytes = MAXIMUM_MANIFEST_BYTES
            destination = INPUT_ROOT / "manifest.json"
        elif path == "/inputs/ready":
            maximum_bytes = MAXIMUM_READY_BYTES
            destination = INPUT_ROOT / "ready.json"
        elif path.startswith("/inputs/") and ADAPTER_ARCHIVE.fullmatch(name):
            maximum_bytes = MAXIMUM_ADAPTER_BYTES
            destination = INPUT_ROOT / name
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            content_length = -1
        if not 1 <= content_length <= maximum_bytes:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "INVALID_UPLOAD_SIZE"})
            return
        INPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        if destination.exists():
            self._json(HTTPStatus.CONFLICT, {"error": "UPLOAD_ALREADY_EXISTS"})
            return
        pending = destination.with_suffix(destination.suffix + ".pending")
        try:
            remaining = content_length
            with pending.open("xb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("incomplete upload")
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.name == "manifest.json":
                load_input_manifest(pending)
            elif destination.name == "ready.json":
                verify_ready_payload(INPUT_ROOT, pending.read_bytes())
            os.replace(pending, destination)
            directory_descriptor = os.open(INPUT_ROOT, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except (FileExistsError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            pending.unlink(missing_ok=True)
            self._json(HTTPStatus.BAD_REQUEST, {"error": "INVALID_UPLOAD"})
            return
        self._json(
            HTTPStatus.CREATED,
            {
                "stored": destination.name,
                "size_bytes": content_length,
                "sha256": sha256_file(destination),
            },
        )

    def do_POST(self) -> None:
        self._store_input()

    def do_PUT(self) -> None:
        self._store_input()

    def log_message(self, format: str, *args: object) -> None:
        return


class ExternalEvalHTTPServer(ThreadingHTTPServer):
    """Bind without the HTTPServer reverse-DNS lookup that can stall readiness."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


if __name__ == "__main__":
    host = configured_result_host()
    port = int(os.environ.get("EQUINOX_RESULT_PORT", "8000"))
    ExternalEvalHTTPServer((host, port), ExternalEvalTransportHandler).serve_forever()
