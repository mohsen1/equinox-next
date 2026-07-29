"""Fenced RunPod operator for the preregistered revision-31 factorial study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - macOS Python 3.9.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - Python 3.9 compatibility.


STUDY_ID = "repository-repair-causal-factorial-study@2"
CONDITION_PATTERN = re.compile(r"^(k(?:1|4)_(?:adaptive|scheduled_dynamic))_seed([1-9][0-9]*)$")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("study_id") != STUDY_ID
        or manifest.get("status") != "preregistered"
        or manifest.get("factors")
        != {
            "branch_width": [1, 4],
            "curriculum_policy": ["scheduled_dynamic", "adaptive"],
        }
        or len(manifest.get("optimization_seeds", [])) != 5
    ):
        raise ValueError("the revision-31 study manifest is invalid")
    return manifest


def condition_from_id(manifest: dict[str, Any], condition_id: str) -> dict[str, Any]:
    match = CONDITION_PATTERN.fullmatch(condition_id)
    if match is None:
        raise ValueError("the condition ID is not a preregistered factorial condition")
    condition, raw_seed = match.groups()
    seed = int(raw_seed)
    matching_seeds = [item for item in manifest["optimization_seeds"] if item.get("seed") == seed]
    if len(matching_seeds) != 1:
        raise ValueError("the condition seed is not preregistered")
    branch_width = 1 if condition.startswith("k1_") else 4
    curriculum_policy = condition.removeprefix(f"k{branch_width}_")
    shared = manifest["shared_configuration"]
    return {
        "condition_id": condition_id,
        "condition": condition,
        "optimization_seed": seed,
        "validation_seed_base": matching_seeds[0]["validation_seed_base"],
        "test_seed_base": matching_seeds[0]["test_seed_base"],
        "branch_width": branch_width,
        "curriculum_policy": curriculum_policy,
        "completion_budget": shared["completion_budget"],
        "training_tasks_per_update": shared["training_tasks_per_update_by_k"][str(branch_width)],
    }


def all_condition_ids(manifest: dict[str, Any]) -> list[str]:
    return [
        f"k{branch_width}_{curriculum_policy}_seed{seed['seed']}"
        for seed in manifest["optimization_seeds"]
        for curriculum_policy in manifest["factors"]["curriculum_policy"]
        for branch_width in manifest["factors"]["branch_width"]
    ]


def result_files(receipt_directory: Path) -> list[Path]:
    return sorted(receipt_directory.glob("runpod-proof-*.result.json"))


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def matching_success_results(
    receipt_directory: Path,
    condition: dict[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in result_files(receipt_directory):
        result = read_json(path)
        study = result.get("study") if result else None
        receipt_path = path.with_name(path.name.removesuffix(".result.json") + ".json")
        receipt = read_json(receipt_path)
        if (
            isinstance(study, dict)
            and study.get("study_id") == STUDY_ID
            and study.get("condition") == condition["condition"]
            and study.get("optimization_seed") == condition["optimization_seed"]
            and study.get("validation_seed_base") == condition["validation_seed_base"]
            and study.get("test_seed_base") == condition["test_seed_base"]
            and study.get("completion_budget") == condition["completion_budget"]
            and result.get("experiment_completed") is True
            and result.get("final_evaluation_complete") is True
            and result.get("adapter_persisted") is True
            and isinstance(receipt, dict)
            and receipt.get("teardown_confirmed") is True
        ):
            matches.append((path, result))
    return matches


def launcher_environment(
    manifest: dict[str, Any],
    condition: dict[str, Any],
    *,
    preflight_only: bool,
) -> dict[str, str]:
    shared = manifest["shared_configuration"]
    model = manifest["model"]
    gpu_override = os.environ.get("EQUINOX_RUNPOD_GPU_OVERRIDE", "")
    selected_gpu = gpu_override or str(shared["hardware_preference"])
    cloud_type_override = os.environ.get("EQUINOX_RUNPOD_CLOUD_TYPE_OVERRIDE", "")
    environment = {
        **os.environ,
        "EQUINOX_RUNPOD_EXPERIMENT": "repository-repair",
        "EQUINOX_STUDY_REVISION": "31",
        "EQUINOX_STUDY_CONDITION": str(condition["condition"]),
        "EQUINOX_RL_MODEL_ID": str(model["id"]),
        "EQUINOX_RL_SEED": str(condition["optimization_seed"]),
        "EQUINOX_RL_TARGET_SECONDS": str(shared["target_runtime_seconds"]),
        "EQUINOX_RUNPOD_RETRY_RESERVE_SECONDS": str(shared["maximum_resume_gap_seconds"]),
        "EQUINOX_RL_MAX_UPDATES": str(shared["maximum_updates"]),
        "EQUINOX_RL_VALIDATION_EXAMPLES": str(shared["validation_examples"]),
        "EQUINOX_RL_TEST_EXAMPLES": str(shared["test_examples_per_level"]),
        "EQUINOX_RL_MASTERY_WINDOWS": str(shared["mastery_windows"]),
        "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL": str(shared["replay_tasks_per_level"]),
        "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS": str(
            shared["maximum_final_evaluation_reserve_seconds"]
        ),
        "EQUINOX_STUDY_VALIDATION_SEED_BASE": str(condition["validation_seed_base"]),
        "EQUINOX_STUDY_TEST_SEED_BASE": str(condition["test_seed_base"]),
        "EQUINOX_STUDY_COMPLETION_BUDGET": str(condition["completion_budget"]),
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": str(condition["training_tasks_per_update"]),
        "EQUINOX_RUNPOD_GPU": selected_gpu,
        "EQUINOX_RUNPOD_MAX_HOURLY_COST": str(shared["maximum_hourly_cost_usd"]),
    }
    if cloud_type_override:
        environment["EQUINOX_RUNPOD_CLOUD_TYPE"] = cloud_type_override
    if preflight_only:
        environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] = "1"
    else:
        environment.pop("EQUINOX_RUNPOD_PREFLIGHT_ONLY", None)
    return environment


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@contextmanager
def exclusive_study_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another revision-31 condition is already running") from error
        yield


def artifact_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run_condition(
    repository_root: Path,
    manifest: dict[str, Any],
    condition_id: str,
    *,
    preflight_only: bool,
    allow_repeat: bool,
) -> int:
    condition = condition_from_id(manifest, condition_id)
    receipt_directory = repository_root / "var/research-proofs"
    prior_successes = matching_success_results(receipt_directory, condition)
    if prior_successes and not allow_repeat:
        print(f"{condition_id} already completed; skipping")
        return 0
    environment = launcher_environment(
        manifest,
        condition,
        preflight_only=preflight_only,
    )
    launcher = repository_root / "scripts/runpod-rl-proof"
    state_directory = repository_root / "var/revision31-study"
    ledger_path = state_directory / "attempts.jsonl"
    before = set(result_files(receipt_directory))
    attempt_started_at = utc_now()
    append_event(
        ledger_path,
        {
            "schema_version": 1,
            "event": "attempt_started",
            "study_id": STUDY_ID,
            "condition_id": condition_id,
            "started_at": attempt_started_at,
            "preflight_only": preflight_only,
            "completion_budget": condition["completion_budget"],
        },
    )
    completed = subprocess.run(
        [str(launcher)],
        cwd=repository_root,
        env=environment,
        check=False,
    )
    new_results = sorted(set(result_files(receipt_directory)) - before)
    append_event(
        ledger_path,
        {
            "schema_version": 1,
            "event": "attempt_finished",
            "study_id": STUDY_ID,
            "condition_id": condition_id,
            "started_at": attempt_started_at,
            "finished_at": utc_now(),
            "preflight_only": preflight_only,
            "exit_code": completed.returncode,
            "outcome": "passed" if completed.returncode == 0 else "failed",
            "result_artifacts": [artifact_record(path) for path in new_results],
        },
    )
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_id", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-repeat", action="store_true")
    arguments = parser.parse_args()
    if arguments.all == (arguments.condition_id is not None):
        parser.error("provide one condition ID or --all")

    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(repository_root / "research/studies/revision31-causal-study.json")
    condition_ids = all_condition_ids(manifest) if arguments.all else [arguments.condition_id]
    lock_path = repository_root / "var/revision31-study/operator.lock"
    with exclusive_study_lock(lock_path):
        for condition_id in condition_ids:
            exit_code = run_condition(
                repository_root,
                manifest,
                str(condition_id),
                preflight_only=arguments.preflight_only,
                allow_repeat=arguments.allow_repeat,
            )
            if exit_code:
                raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
