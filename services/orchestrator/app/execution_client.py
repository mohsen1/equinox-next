from __future__ import annotations

import os
import time
from typing import Any

import httpx
from equinox_core import canonical_digest, make_id


class RetryableExecutionError(RuntimeError):
    pass


class ExecutionClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("EXECUTION_URL", "http://execution:8081")

    def operation(
        self,
        path: str,
        operation_type: str,
        operation_input: dict[str, Any],
        *,
        operation_id: str | None = None,
        idempotency_key: str | None = None,
        expected_version: int = 0,
        correlation_id: str,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        op_id = operation_id or make_id("op")
        payload = {
            "operation_id": op_id,
            "idempotency_key": idempotency_key or op_id,
            "request_digest": canonical_digest(
                {"operation_type": operation_type, "input": operation_input}
            ),
            "expected_version": expected_version,
            "correlation_id": correlation_id,
            **operation_input,
        }
        deadline = time.monotonic() + timeout
        response: httpx.Response | None = None
        last_error: Exception | None = None
        with httpx.Client(timeout=min(timeout, 10)) as client:
            while time.monotonic() < deadline:
                try:
                    response = client.post(f"{self.base_url}{path}", json=payload)
                    if response.status_code not in (502, 503, 504):
                        break
                except (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.TimeoutException,
                ) as exc:
                    last_error = exc
                time.sleep(0.25)
        if response is None:
            raise RetryableExecutionError(
                f"execution operation {operation_type} remained unavailable"
            ) from last_error
        if response.status_code in (502, 503, 504):
            raise RetryableExecutionError(
                f"execution operation {operation_type} remained unavailable "
                f"({response.status_code}): {response.text}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"execution operation {operation_type} failed "
                f"({response.status_code}): {response.text}"
            )
        return response.json()

    def health(self) -> dict[str, Any]:
        return httpx.get(f"{self.base_url}/healthz", timeout=5).json()


execution_client = ExecutionClient()
