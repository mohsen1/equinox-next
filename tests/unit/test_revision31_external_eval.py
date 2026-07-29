from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.runpod import revision31_external_operator as operator
from research.runpod.bootstrap_server import expected_bundle_files
from research.runpod.revision31_external_eval import verify_external_pack
from research.runpod.revision31_study_operator import load_manifest


def test_revision31_manifest_and_bundle_contracts_load_in_isolation() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = """
import json
from pathlib import Path
from research.runpod.external_eval_launcher import build_bundle, load_frozen_pack
from research.runpod.revision31_external_eval import (
    ADAPTER_SET,
    MODEL_ID,
    MODEL_REVISION,
    PACK_ID,
    STUDY_ID,
    validate_evaluation_manifest,
)
manifest = {
    "schema_version": 1,
    "evaluation_id": "revision31-external-test",
    "study_id": STUDY_ID,
    "adapter_set": ADAPTER_SET,
    "pack_id": PACK_ID,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
    "adapters": [{
        "adapter_id": "k4_adaptive_seed137",
        "condition_id": "k4_adaptive_seed137",
        "role": "factorial_condition",
        "optimization_seed": 137,
        "source_execution_id": "runpod-proof-20260729T010203Z",
        "source_result_sha256": "a" * 64,
        "archive_filename": "adapter-" + "b" * 64 + ".tgz",
        "size_bytes": 123,
        "sha256": "b" * 64,
        "adapter_manifest_digest": "sha256:" + "c" * 64,
    }],
}
validate_evaluation_manifest(manifest)
root = Path.cwd()
print(json.dumps({
    "task_count": load_frozen_pack(root)["task_count"],
    "bundle_bytes": len(build_bundle(root)),
}))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root),
        "EQUINOX_EXTERNAL_EVALUATION_REVISION": "31",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    observed = json.loads(completed.stdout)
    assert observed["task_count"] == 120
    assert 0 < observed["bundle_bytes"] <= 2 * 1024 * 1024


def test_bootstrap_allowlist_contains_revision31_evaluator_dependencies() -> None:
    files = expected_bundle_files("research/runpod/revision31_external_eval.py")

    assert "research/external/revision31_task_pack.py" in files
    assert "research/frozen/revision31-external-pack.json" in files
    assert "research/runpod/repository_repair_env_v31.py" in files
    assert "external_eval_remote_runner.sh" in files


def test_external_pack_is_verified_by_revision31_workload() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    observed = verify_external_pack(repository_root)

    assert observed["task_count"] == 120
    assert observed["audit"]["golden_repairs_confirmed"] == 120


def test_external_outcomes_remain_sealed_until_all_conditions_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    study_manifest = load_manifest(
        repository_root / "research/studies/revision31-causal-study.json"
    )
    monkeypatch.setattr(operator, "load_manifest", lambda path: study_manifest)
    monkeypatch.setattr(operator, "matching_success_results", lambda *args: [])

    with pytest.raises(RuntimeError, match="until all 20 conditions complete"):
        operator.completed_conditions(tmp_path)


def test_remote_runner_accepts_revision31_workload() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = (repository_root / "research/runpod/external_eval_remote_runner.sh").read_text(
        encoding="utf-8"
    )

    assert "research/runpod/revision31_external_eval.py)" in source
    assert 'python3 "$work_directory/$workload_file"' in source
