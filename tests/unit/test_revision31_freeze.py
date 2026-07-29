from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "research/frozen/repository-repair-revision-31.json"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_revision31_trainer_is_byte_for_byte_frozen() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["freeze_id"] == "repository-repair-revision-31"
    assert manifest["study_id"] == "repository-repair-causal-factorial-study@2"
    assert manifest["workload_revision"] == "runpod-repository-repair-causal-credit@31"
    assert manifest["frozen_design"]["branch_widths"] == [1, 4]
    assert manifest["frozen_design"]["optimization_seeds_per_cell"] == 5
    assert manifest["frozen_design"]["completion_budget"] == 320
    assert manifest["operational_amendments"] == [
        "research/studies/revision31-capacity-amendment-1.json",
        "research/studies/revision31-operational-amendment-2.json",
        "research/studies/revision31-capacity-amendment-3.json",
        "research/studies/revision31-runtime-amendment-4.json",
    ]

    for source in manifest["source_files"]:
        path = REPOSITORY_ROOT / source["path"]
        worktree_payload = path.read_bytes()
        assert sha256(worktree_payload) == source["sha256"]
        if shutil.which("git"):
            committed_payload = subprocess.run(
                ["git", "show", f"{manifest['source_commit']}:{source['path']}"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
            ).stdout
            assert committed_payload == worktree_payload
