from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from research.runpod.revision30_external_eval import (
    FROZEN_OBJECTIVE_ID,
    FROZEN_WORKLOAD_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    STUDY_ID,
    canonical_json,
)
from research.runpod.revision30_external_operator import (
    adapter_record,
    build_input_manifest,
    retained_conditions,
)


def conditions() -> list[dict]:
    return [
        {
            "condition_id": "k4_train_seed113",
            "role": "frozen_reference",
            "optimization_seed": 113,
            "branch_width": 4,
            "policy_mutation_enabled": True,
            "execution_id": "runpod-proof-20260728T142404Z",
        },
        {
            "condition_id": "k4_train_seed307",
            "role": "fresh_replication",
            "optimization_seed": 307,
            "validation_seed_base": 220_000_000,
            "test_seed_base": 260_000_000,
            "branch_width": 4,
            "policy_mutation_enabled": True,
        },
        {
            "condition_id": "k4_train_seed701",
            "role": "fresh_replication",
            "optimization_seed": 701,
            "validation_seed_base": 620_000_000,
            "test_seed_base": 660_000_000,
            "branch_width": 4,
            "policy_mutation_enabled": True,
        },
        {
            "condition_id": "k4_no_update_seed307",
            "role": "matched_frozen_policy_control",
            "optimization_seed": 307,
            "validation_seed_base": 220_000_000,
            "test_seed_base": 260_000_000,
            "branch_width": 4,
            "policy_mutation_enabled": False,
        },
        {
            "condition_id": "k1_train_seed307",
            "role": "branch_width_ablation",
            "optimization_seed": 307,
            "validation_seed_base": 220_000_000,
            "test_seed_base": 260_000_000,
            "branch_width": 1,
            "policy_mutation_enabled": True,
        },
    ]


def write_adapter_source(
    receipt_directory: Path,
    condition: dict,
    execution_id: str,
) -> None:
    files = {
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": condition["condition_id"].encode(),
    }
    manifest_content = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "workload_revision": FROZEN_WORKLOAD_REVISION,
        "objective_id": FROZEN_OBJECTIVE_ID,
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(files.items())
        ],
    }
    adapter_manifest = {
        **manifest_content,
        "digest": "sha256:" + hashlib.sha256(canonical_json(manifest_content)).hexdigest(),
    }
    archive_path = receipt_directory / f"{execution_id}.adapter.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in {
            "adapter/adapter-manifest.json": canonical_json(adapter_manifest),
            **{f"adapter/{name}": payload for name, payload in files.items()},
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    result = {
        "experiment_completed": True,
        "final_evaluation_complete": True,
        "adapter_persisted": True,
        "workload_revision": FROZEN_WORKLOAD_REVISION,
        "objective_id": FROZEN_OBJECTIVE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": condition["optimization_seed"],
        "branch_width": condition["branch_width"],
        "adapter_manifest": adapter_manifest,
    }
    if condition["role"] != "frozen_reference":
        result["study"] = {
            "study_id": STUDY_ID,
            "condition": condition["condition_id"].rsplit("_seed", 1)[0],
            "optimization_seed": condition["optimization_seed"],
            "validation_seed_base": condition["validation_seed_base"],
            "test_seed_base": condition["test_seed_base"],
            "policy_mutation_enabled": condition["policy_mutation_enabled"],
        }
    (receipt_directory / f"{execution_id}.result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )


def test_operator_builds_exactly_the_complete_retained_adapter_set(tmp_path: Path) -> None:
    study_conditions = conditions()
    study_directory = tmp_path / "research/studies"
    receipt_directory = tmp_path / "var/research-proofs"
    study_directory.mkdir(parents=True)
    receipt_directory.mkdir(parents=True)
    (study_directory / "revision30-confirmatory-study.json").write_text(
        json.dumps({"study_id": STUDY_ID, "conditions": study_conditions}),
        encoding="utf-8",
    )
    for index, condition in enumerate(study_conditions):
        execution_id = condition.get(
            "execution_id",
            f"runpod-proof-20260729T01000{index}Z",
        )
        write_adapter_source(receipt_directory, condition, execution_id)

    manifest, sources = build_input_manifest(
        tmp_path,
        "revision30-external-eval-test",
    )

    assert [adapter["condition_id"] for adapter in manifest["adapters"]] == [
        condition["condition_id"] for condition in study_conditions
    ]
    assert len(sources) == 5
    assert set(sources) == {adapter["archive_filename"] for adapter in manifest["adapters"]}


def test_operator_rejects_tampered_adapter_bytes(tmp_path: Path) -> None:
    condition = conditions()[0]
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    execution_id = condition["execution_id"]
    write_adapter_source(receipt_directory, condition, execution_id)
    archive_path = receipt_directory / f"{execution_id}.adapter.tgz"
    with tarfile.open(archive_path, "r:gz") as archive:
        payloads = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    payloads["adapter/adapter_model.safetensors"] = b"tampered"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="size evidence does not match"):
        adapter_record(
            condition,
            receipt_directory / f"{execution_id}.result.json",
        )


def test_retained_roster_cannot_silently_drop_a_condition() -> None:
    with pytest.raises(RuntimeError, match="roster is incomplete"):
        retained_conditions({"conditions": conditions()[:-1]})
