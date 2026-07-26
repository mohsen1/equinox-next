from __future__ import annotations

import os
import time

import httpx

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")


def main() -> None:
    with httpx.Client(timeout=300) as client:
        while True:
            try:
                response = client.post(f"{ORCHESTRATOR_URL}/internal/agent/claims")
                response.raise_for_status()
                claim = response.json()["claim"]
                if claim is None:
                    time.sleep(1)
                    continue
                attempt_id = claim["attempt_id"]
                client.post(
                    f"{ORCHESTRATOR_URL}/internal/run-attempts/{attempt_id}/heartbeat"
                ).raise_for_status()
                execution = client.post(
                    f"{ORCHESTRATOR_URL}/internal/run-attempts/{attempt_id}/execute"
                )
                execution.raise_for_status()
            except (httpx.HTTPError, KeyError):
                time.sleep(2)


if __name__ == "__main__":
    main()
