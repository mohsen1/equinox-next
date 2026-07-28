from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

import research.runpod.external_eval_transport as transport


def request(
    server: transport.ThreadingHTTPServer,
    method: str,
    path: str,
    *,
    token: str,
    body: bytes = b"",
) -> tuple[int, dict]:
    connection = HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.fixture
def server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[transport.ThreadingHTTPServer, str]:
    token = "external-eval-test-token"
    monkeypatch.setattr(transport, "ROOT", tmp_path)
    monkeypatch.setattr(transport, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(transport, "TOKEN", token)
    running = transport.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        transport.ExternalEvalTransportHandler,
    )
    thread = Thread(target=running.serve_forever)
    thread.start()
    try:
        yield running, token
    finally:
        running.shutdown()
        thread.join(timeout=2)
        running.server_close()


def test_transport_authenticates_and_atomically_accepts_complete_inputs(
    server: tuple[transport.ThreadingHTTPServer, str],
) -> None:
    running, token = server
    adapter = b"adapter archive bytes"
    adapter_sha = hashlib.sha256(adapter).hexdigest()
    manifest = transport.canonical_json(
        {
            "schema_version": 1,
            "adapters": [
                {
                    "adapter_id": "reference-seed113",
                    "archive_filename": f"adapter-{adapter_sha}.tgz",
                    "size_bytes": len(adapter),
                    "sha256": adapter_sha,
                }
            ],
        }
    )
    manifest_sha = hashlib.sha256(manifest).hexdigest()

    unauthorized, _ = request(
        running,
        "PUT",
        "/inputs/manifest.json",
        token="wrong-token",
        body=manifest,
    )
    assert unauthorized == HTTPStatus.UNAUTHORIZED

    for path, body in (
        ("/inputs/manifest.json", manifest),
        (f"/inputs/adapter-{adapter_sha}.tgz", adapter),
        (
            "/inputs/ready",
            transport.canonical_json({"manifest_sha256": manifest_sha}),
        ),
    ):
        status, payload = request(running, "PUT", path, token=token, body=body)
        assert status == HTTPStatus.CREATED
        assert payload["size_bytes"] == len(body)

    status, payload = request(running, "GET", "/input-status", token=token)
    assert status == HTTPStatus.OK
    assert payload == {
        "manifest_received": True,
        "ready": True,
        "adapter_archives_received": 1,
    }


def test_transport_rejects_ready_until_all_digests_match(
    server: tuple[transport.ThreadingHTTPServer, str],
) -> None:
    running, token = server
    adapter_sha = hashlib.sha256(b"expected").hexdigest()
    manifest = transport.canonical_json(
        {
            "schema_version": 1,
            "adapters": [
                {
                    "adapter_id": "candidate",
                    "archive_filename": f"adapter-{adapter_sha}.tgz",
                    "size_bytes": 8,
                    "sha256": adapter_sha,
                }
            ],
        }
    )
    assert (
        request(
            running,
            "PUT",
            "/inputs/manifest.json",
            token=token,
            body=manifest,
        )[0]
        == HTTPStatus.CREATED
    )

    status, payload = request(
        running,
        "PUT",
        "/inputs/ready",
        token=token,
        body=transport.canonical_json({"manifest_sha256": hashlib.sha256(manifest).hexdigest()}),
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert payload == {"error": "INVALID_UPLOAD"}
