"""Launch sealed revision-31 external evaluation shards after the matrix closes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from research.runpod import revision30_external_operator as base
from research.runpod.external_eval_transport import load_input_manifest
from research.runpod.revision31_external_eval import (
    ADAPTER_ROLES,
    ADAPTER_SET,
    FROZEN_OBJECTIVE_ID,
    FROZEN_WORKLOAD_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PACK_ID,
    STUDY_ID,
    canonical_json,
    validate_evaluation_manifest,
)
from research.runpod.revision31_study_operator import (
    all_condition_ids,
    condition_from_id,
    load_manifest,
    matching_success_results,
)


def install_revision31_operator_contract() -> None:
    base.ADAPTER_ROLES = ADAPTER_ROLES
    base.ADAPTER_SET = ADAPTER_SET
    base.FROZEN_OBJECTIVE_ID = FROZEN_OBJECTIVE_ID
    base.FROZEN_WORKLOAD_REVISION = FROZEN_WORKLOAD_REVISION
    base.MODEL_ID = MODEL_ID
    base.MODEL_REVISION = MODEL_REVISION
    base.PACK_ID = PACK_ID
    base.STUDY_ID = STUDY_ID


def new_evaluation_id(condition_id: str) -> str:
    return base.new_evaluation_id().replace("revision30-", "revision31-") + f"-{condition_id}"


def completed_conditions(
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    manifest = load_manifest(repository_root / "research/studies/revision31-causal-study.json")
    receipt_directory = repository_root / "var/research-proofs"
    completed = {}
    for condition_id in all_condition_ids(manifest):
        condition = condition_from_id(manifest, condition_id)
        matches = matching_success_results(receipt_directory, condition)
        if len(matches) != 1:
            raise RuntimeError(
                "external outcomes remain sealed until all 20 conditions complete; "
                f"{condition_id} has {len(matches)} verified results"
            )
        completed[condition_id] = matches[0]
    return manifest, completed


def adapter_record(
    condition_id: str,
    manifest: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    install_revision31_operator_contract()
    condition = {
        **condition_from_id(manifest, condition_id),
        "role": "factorial_condition",
        "policy_mutation_enabled": True,
    }
    return base.adapter_record(condition, result_path)


def load_input_manifest_from_value(manifest: dict[str, Any]) -> dict[str, Any]:
    with base.tempfile.TemporaryDirectory(
        prefix="equinox-revision31-external-manifest-"
    ) as directory:
        path = Path(directory) / "manifest.json"
        path.write_bytes(canonical_json(manifest))
        return load_input_manifest(path)


def build_input_manifest(
    repository_root: Path,
    condition_ids: list[str],
    evaluation_id: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    study_manifest, completed = completed_conditions(repository_root)
    records = []
    source_archives = {}
    for condition_id in condition_ids:
        record = adapter_record(
            condition_id,
            study_manifest,
            completed[condition_id][0],
        )
        source_archives[record["archive_filename"]] = Path(record.pop("_source_archive_path"))
        records.append(record)
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "study_id": STUDY_ID,
        "adapter_set": ADAPTER_SET,
        "pack_id": PACK_ID,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "adapters": records,
    }
    validate_evaluation_manifest(load_input_manifest_from_value(manifest))
    if len(source_archives) != len(records):
        raise RuntimeError("revision-31 adapter archives are not content-unique")
    return manifest, source_archives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_id", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.all == (arguments.condition_id is not None):
        parser.error("provide one condition ID or --all")

    repository_root = Path(__file__).resolve().parents[2]
    study_manifest = load_manifest(
        repository_root / "research/studies/revision31-causal-study.json"
    )
    condition_ids = (
        all_condition_ids(study_manifest) if arguments.all else [str(arguments.condition_id)]
    )
    for condition_id in condition_ids:
        condition_from_id(study_manifest, condition_id)
    evaluation_label = "all" if arguments.all else condition_ids[0]
    evaluation_id = new_evaluation_id(evaluation_label)
    manifest, source_archives = build_input_manifest(
        repository_root,
        condition_ids,
        evaluation_id,
    )
    state_directory = repository_root / "var/revision31-study"
    manifest_path = state_directory / f"{evaluation_id}.input.json"
    base.write_manifest_durably(manifest_path, manifest)
    environment = {
        **os.environ,
        "EQUINOX_EXTERNAL_EVALUATION_REVISION": "31",
        "EQUINOX_EXTERNAL_EVALUATION_ID": evaluation_id,
        "EQUINOX_EXTERNAL_MANIFEST_PATH": str(manifest_path),
        "EQUINOX_EXTERNAL_ARCHIVE_MAP": json.dumps(
            {key: str(value) for key, value in source_archives.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "EQUINOX_RUNPOD_GPU": os.environ.get(
            "EQUINOX_RUNPOD_GPU",
            "NVIDIA L4",
        ),
        "EQUINOX_RUNPOD_MAX_HOURLY_COST": os.environ.get(
            "EQUINOX_RUNPOD_MAX_HOURLY_COST",
            "0.5",
        ),
    }
    if arguments.preflight_only:
        environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] = "1"
    completed = subprocess.run(
        [str(repository_root / "scripts/runpod-external-eval")],
        cwd=repository_root,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
