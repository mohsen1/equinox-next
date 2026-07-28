"""Fenced operator for the preregistered revision-30 confirmatory study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - exercised by the macOS Python 3.9 operator.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - Python 3.9 compatibility.

STUDY_ID = "repository-repair-confirmatory-study@1"
DEPENDENT_BUDGET = "k4_train_seed211.total_sampled_completions"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError("the study manifest identity is invalid")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("the study manifest has no conditions")
    return manifest


def condition_by_id(manifest: dict[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [
        condition
        for condition in manifest["conditions"]
        if isinstance(condition, dict) and condition.get("condition_id") == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate study condition: {condition_id}")
    condition = matches[0]
    if condition.get("role") == "frozen_reference":
        raise ValueError("the frozen reference is already complete and cannot be rerun")
    return condition


def result_files(receipt_directory: Path) -> list[Path]:
    return sorted(receipt_directory.glob("runpod-proof-*.result.json"))


def read_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def matching_success_results(
    receipt_directory: Path,
    condition: dict[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in result_files(receipt_directory):
        result = read_result(path)
        study = result.get("study") if result else None
        if (
            isinstance(study, dict)
            and study.get("study_id") == STUDY_ID
            and study.get("condition") == condition.get("condition_id", "").rsplit("_seed", 1)[0]
            and study.get("optimization_seed") == condition.get("optimization_seed")
            and study.get("validation_seed_base") == condition.get("validation_seed_base")
            and study.get("test_seed_base") == condition.get("test_seed_base")
            and result.get("experiment_completed") is True
            and result.get("final_evaluation_complete") is True
            and result.get("adapter_persisted") is True
        ):
            matches.append((path, result))
    return matches


def resolve_completion_budget(
    condition: dict[str, Any],
    receipt_directory: Path,
) -> int | None:
    declared = condition.get("completion_budget")
    if declared is None:
        return None
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared
    if declared != DEPENDENT_BUDGET:
        raise ValueError("the study condition has an unknown completion-budget dependency")
    source_condition = {
        "condition_id": "k4_train_seed211",
        "optimization_seed": 211,
        "validation_seed_base": 20_000_000,
        "test_seed_base": 50_000_000,
    }
    sources = matching_success_results(receipt_directory, source_condition)
    if len(sources) != 1:
        raise RuntimeError(
            "matched controls require exactly one completed seed-211 K=4 training result"
        )
    budget = sources[0][1].get("total_sampled_completions")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise RuntimeError("the seed-211 K=4 result has no valid continuation budget")
    return budget


def launcher_environment(
    manifest: dict[str, Any],
    condition: dict[str, Any],
    *,
    completion_budget: int | None,
    preflight_only: bool,
) -> dict[str, str]:
    condition_name = str(condition["condition_id"]).rsplit("_seed", 1)[0]
    shared = manifest["shared_configuration"]
    model = manifest["model"]
    environment = {
        **os.environ,
        "EQUINOX_RUNPOD_EXPERIMENT": "repository-repair",
        "EQUINOX_STUDY_CONDITION": condition_name,
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
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": str(condition["training_tasks_per_update"]),
    }
    if completion_budget is not None:
        environment["EQUINOX_STUDY_COMPLETION_BUDGET"] = str(completion_budget)
    else:
        environment.pop("EQUINOX_STUDY_COMPLETION_BUDGET", None)
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
            raise RuntimeError("another revision-30 study condition is already running") from error
        yield


def artifact_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-repeat", action="store_true")
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    manifest_path = repository_root / "research/studies/revision30-confirmatory-study.json"
    receipt_directory = repository_root / "var/research-proofs"
    study_state_directory = repository_root / "var/revision30-study"
    manifest = load_manifest(manifest_path)
    condition = condition_by_id(manifest, arguments.condition_id)
    completion_budget = resolve_completion_budget(condition, receipt_directory)
    prior_successes = matching_success_results(receipt_directory, condition)
    if prior_successes and not arguments.allow_repeat:
        raise SystemExit(
            f"{arguments.condition_id} already has a successful result; refusing a duplicate run"
        )
    environment = launcher_environment(
        manifest,
        condition,
        completion_budget=completion_budget,
        preflight_only=arguments.preflight_only,
    )
    launcher = repository_root / "scripts/runpod-rl-proof"
    ledger_path = study_state_directory / "attempts.jsonl"
    lock_path = study_state_directory / "operator.lock"

    with exclusive_study_lock(lock_path):
        before = set(result_files(receipt_directory))
        attempt_started_at = utc_now()
        append_event(
            ledger_path,
            {
                "schema_version": 1,
                "event": "attempt_started",
                "study_id": STUDY_ID,
                "condition_id": arguments.condition_id,
                "started_at": attempt_started_at,
                "preflight_only": arguments.preflight_only,
                "completion_budget": completion_budget,
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
                "condition_id": arguments.condition_id,
                "started_at": attempt_started_at,
                "finished_at": utc_now(),
                "preflight_only": arguments.preflight_only,
                "exit_code": completed.returncode,
                "outcome": "passed" if completed.returncode == 0 else "failed",
                "result_artifacts": [artifact_record(path) for path in new_results],
            },
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
