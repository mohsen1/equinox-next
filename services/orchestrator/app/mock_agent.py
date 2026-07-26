from __future__ import annotations

import os
import socket
import time

import httpx

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
WORKER_ID = os.getenv("WORKER_ID", f"fixture-worker:{socket.gethostname()}")


def main() -> None:
    with httpx.Client(timeout=300) as client:
        while True:
            try:
                response = client.post(
                    f"{ORCHESTRATOR_URL}/internal/agent/claims",
                    json={"worker_id": WORKER_ID},
                )
                response.raise_for_status()
                claim = response.json()["claim"]
                if claim is None:
                    time.sleep(1)
                    continue
                attempt_id = claim["attempt_id"]
                lease = {
                    "worker_id": WORKER_ID,
                    "claim_id": claim["claim_id"],
                    "fencing_token": claim["fencing_token"],
                }
                client.post(
                    f"{ORCHESTRATOR_URL}/internal/run-attempts/{attempt_id}/heartbeat",
                    json=lease,
                ).raise_for_status()
                execution = client.post(
                    f"{ORCHESTRATOR_URL}/internal/run-attempts/{attempt_id}/execute",
                    json=lease,
                )
                execution.raise_for_status()
            except (httpx.HTTPError, KeyError):
                time.sleep(2)


if __name__ == "__main__":
    main()
