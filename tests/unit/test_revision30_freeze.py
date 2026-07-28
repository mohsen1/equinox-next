from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "research/frozen/repository-repair-revision-30.json"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_revision30_source_is_byte_for_byte_frozen() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["freeze_id"] == "repository-repair-revision-30"
    assert manifest["workload_revision"] == "runpod-repository-repair-causal-credit@30"
    assert manifest["objective_id"] == ("verified-fix-coverage-retention-policy-gradient@15")
    assert manifest["frozen_condition"]["branch_width"] == 4
    assert manifest["frozen_condition"]["complexity_strategy"] == "adaptive"

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


def test_revision30_reference_artifacts_match_when_present() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    for artifact in manifest["reference_run"]["artifacts"]:
        path = REPOSITORY_ROOT / artifact["path"]
        if path.exists():
            assert sha256(path.read_bytes()) == artifact["sha256"]
