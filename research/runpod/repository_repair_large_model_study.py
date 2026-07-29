"""Source-bound adapter around the transactional larger-model trainer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

try:
    import repository_repair_large_model_trainer as trainer
except ModuleNotFoundError:
    from . import repository_repair_large_model_trainer as trainer


BASE_WORKLOAD_REVISION = "runpod-repository-repair-transactional-retention@1"
BASE_OBJECTIVE_ID = "verified-repair-chain-transactional-retention-policy-gradient@18"
TRANSACTIONAL_RETENTION_REVISION = "adapter-optimizer-policy-lineage@1"
SOURCE_SHA256 = {
    "repository_repair_large_model_trainer.py": (
        "27ecb447306dae587089aab3eb6e0d9506b5f79468c4bf9d79bde1f942744689"
    ),
    "repository_repair_env.py": (
        "527050c5444a3731119773d4e0932840068fda9664654dce9bf0350edf56e40f"
    ),
}


def verify_transactional_sources(root: Path | None = None) -> None:
    source_root = root or Path(__file__).resolve().parent
    for filename, expected_digest in SOURCE_SHA256.items():
        observed_digest = hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise RuntimeError(f"{filename} does not match the larger-model source digest")
    if trainer.WORKLOAD_REVISION != BASE_WORKLOAD_REVISION:
        raise RuntimeError("the larger-model trainer base workload revision changed")
    if trainer.OBJECTIVE_ID != BASE_OBJECTIVE_ID:
        raise RuntimeError("the larger-model trainer base objective changed")
    if trainer.TRANSACTIONAL_RETENTION_REVISION != TRANSACTIONAL_RETENTION_REVISION:
        raise RuntimeError("the larger-model retention transaction revision changed")


def run_trainer_and_capture_result() -> dict[str, Any]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        trainer.main()
    lines = [line for line in captured.getvalue().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("the larger-model trainer did not emit a result")
    for diagnostic in lines[:-1]:
        print(diagnostic, file=sys.stderr)
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("the larger-model trainer result was not an object")
    return result
