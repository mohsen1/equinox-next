"""Authenticated, allowlisted result transport for the bounded RunPod worker."""

from __future__ import annotations

import hmac
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import urlsplit

ROOT = Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp")).resolve()
TOKEN = os.environ["EQUINOX_RESULT_TOKEN"]
ROUTES = {
    "/adapter.tgz": ("adapter.tgz", "application/gzip"),
    "/error.log": ("error.log", "text/plain; charset=utf-8"),
    "/exit_code": ("exit_code", "text/plain; charset=utf-8"),
    "/progress.json": ("progress.json", "application/json"),
    "/result.json": ("result.json", "application/json"),
}


class ResultHandler(BaseHTTPRequestHandler):
    server_version = "EquinoxResultTransport/1"

    def _authorized(self) -> bool:
        provided = self.headers.get("Authorization", "")
        return hmac.compare_digest(provided, f"Bearer {TOKEN}")

    def _serve(self, *, include_body: bool) -> None:
        if not self._authorized():
            self.send_error(401)
            return
        route = ROUTES.get(urlsplit(self.path).path)
        if route is None:
            self.send_error(404)
            return
        relative_path, media_type = route
        path = (ROOT / relative_path).resolve()
        if path.parent != ROOT or not path.is_file():
            self.send_error(404)
            return
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if include_body:
            self.wfile.write(payload)

    def do_GET(self) -> None:
        self._serve(include_body=True)

    def do_HEAD(self) -> None:
        self._serve(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return


class ResultHTTPServer(ThreadingHTTPServer):
    """Bind without the HTTPServer reverse-DNS lookup that can stall readiness."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


if __name__ == "__main__":
    host = os.environ.get("EQUINOX_RESULT_HOST", "0.0.0.0")
    port = int(os.environ.get("EQUINOX_RESULT_PORT", "8000"))
    ResultHTTPServer((host, port), ResultHandler).serve_forever()
