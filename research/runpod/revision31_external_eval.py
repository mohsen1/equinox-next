"""Evaluate revision-31 adapters on the sealed 120-task external pack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.external import revision31_task_pack
from research.runpod import repository_repair_env_v31
from research.runpod import repository_repair_rl as frozen
from research.runpod import revision30_external_eval as evaluator

WORKLOAD = "revision31-post-freeze-external-adapter-evaluation"
WORKLOAD_REVISION = "revision31-external-adapter-evaluation@1"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
STUDY_ID = "repository-repair-causal-factorial-study@2"
ADAPTER_SET = "completed-factorial-adapters@1"
PACK_ID = revision31_task_pack.PACK_ID
FROZEN_WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@31"
FROZEN_OBJECTIVE_ID = "verified-fix-coverage-retention-policy-gradient@15"
ADAPTER_ROLES = {"factorial_condition"}
canonical_json = evaluator.canonical_json


def install_revision31_contract() -> None:
    evaluator.WORKLOAD = WORKLOAD
    evaluator.WORKLOAD_REVISION = WORKLOAD_REVISION
    evaluator.MODEL_ID = MODEL_ID
    evaluator.MODEL_REVISION = MODEL_REVISION
    evaluator.STUDY_ID = STUDY_ID
    evaluator.ADAPTER_SET = ADAPTER_SET
    evaluator.PACK_ID = PACK_ID
    evaluator.FROZEN_WORKLOAD_REVISION = FROZEN_WORKLOAD_REVISION
    evaluator.FROZEN_OBJECTIVE_ID = FROZEN_OBJECTIVE_ID
    evaluator.ADAPTER_ROLES = ADAPTER_ROLES
    evaluator.MAXIMUM_ACTIONS = revision31_task_pack.MAXIMUM_ACTIONS
    evaluator.ExternalTaskEnvironment = revision31_task_pack.ExternalTaskEnvironment
    evaluator.external_tasks = revision31_task_pack.external_tasks
    evaluator.pack_manifest = revision31_task_pack.pack_manifest
    frozen.SYSTEM_PROMPT = repository_repair_env_v31.SYSTEM_PROMPT
    frozen.ACTION_PROTOCOL_REVISION = repository_repair_env_v31.ACTION_PROTOCOL_REVISION


def validate_evaluation_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    install_revision31_contract()
    return evaluator.validate_evaluation_manifest(manifest)


def verify_external_pack(root: Path) -> dict[str, Any]:
    expected = json.loads(
        (root / "research/frozen/revision31-external-pack.json").read_text(encoding="utf-8")
    )
    observed = revision31_task_pack.pack_manifest()
    if observed != expected:
        raise RuntimeError("the revision-31 task pack does not match its sealed manifest")
    return observed


def run_evaluation(root: Path, input_root: Path) -> dict[str, Any]:
    install_revision31_contract()
    evaluator.verify_external_pack = verify_external_pack
    return evaluator.run_evaluation(root, input_root)


def main() -> None:
    install_revision31_contract()
    evaluator.verify_external_pack = verify_external_pack
    evaluator.main()


if __name__ == "__main__":
    main()
