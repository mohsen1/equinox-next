from __future__ import annotations

import hashlib
import json
import socket
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

import research.runpod.external_eval_transport as transport


def request(
    server: transport.ExternalEvalHTTPServer,
    method: str,
    path: str,
    *,
    token: str,
    body: bytes = b"",
    connect_host: str | None = None,
) -> tuple[int, dict]:
    host, port = server.server_address[:2]
    connection = HTTPConnection(connect_host or host, port, timeout=2)
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
) -> tuple[transport.ExternalEvalHTTPServer, str]:
    token = "external-eval-test-token"
    monkeypatch.setattr(transport, "ROOT", tmp_path)
    monkeypatch.setattr(transport, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(transport, "TOKEN", token)
    running = transport.ExternalEvalHTTPServer(
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
    server: tuple[transport.ExternalEvalHTTPServer, str],
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

    for method, path, body in (
        ("POST", "/inputs/manifest.json", manifest),
        ("PUT", f"/inputs/adapter-{adapter_sha}.tgz", adapter),
        (
            "POST",
            "/inputs/ready",
            transport.canonical_json({"manifest_sha256": manifest_sha}),
        ),
    ):
        status, payload = request(running, method, path, token=token, body=body)
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
    server: tuple[transport.ExternalEvalHTTPServer, str],
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


@pytest.mark.parametrize(
    ("configured_host", "expected_bind_host"),
    [(None, "0.0.0.0"), ("127.0.0.1", "127.0.0.1")],
)
def test_transport_readiness_does_not_depend_on_reverse_dns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_host: str | None,
    expected_bind_host: str,
) -> None:
    token = "external-eval-readiness-token"
    monkeypatch.setattr(transport, "ROOT", tmp_path)
    monkeypatch.setattr(transport, "INPUT_ROOT", tmp_path / "inputs")
    monkeypatch.setattr(transport, "TOKEN", token)

    def reject_reverse_dns(_host: str) -> str:
        raise AssertionError("transport startup attempted reverse DNS")

    monkeypatch.setattr(socket, "getfqdn", reject_reverse_dns)
    if configured_host is None:
        monkeypatch.delenv("EQUINOX_RESULT_HOST", raising=False)
    else:
        monkeypatch.setenv("EQUINOX_RESULT_HOST", configured_host)
    bind_host = transport.configured_result_host()
    assert bind_host == expected_bind_host
    running = transport.ExternalEvalHTTPServer(
        (bind_host, 0),
        transport.ExternalEvalTransportHandler,
    )
    thread = Thread(target=running.serve_forever)
    thread.start()
    try:
        status, payload = request(
            running,
            "GET",
            "/input-status",
            token=token,
            connect_host="127.0.0.1",
        )
        assert status == HTTPStatus.OK
        assert payload["ready"] is False
    finally:
        running.shutdown()
        thread.join(timeout=2)
        running.server_close()
