from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


@pytest.mark.parametrize("result_host", [None, "127.0.0.1"])
def test_result_transport_requires_auth_and_exposes_only_allowlisted_files(
    tmp_path: Path,
    result_host: str | None,
) -> None:
    (tmp_path / "progress.json").write_text('{"phase":"training"}\n', encoding="utf-8")
    (tmp_path / "private.txt").write_text("must not be served", encoding="utf-8")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_RESULT_TOKEN": "test-token",
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
            assert response.read() == b'{"phase":"training"}\n'
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
