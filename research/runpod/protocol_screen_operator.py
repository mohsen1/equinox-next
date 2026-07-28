"""Fenced operator for blind revision-30 protocol-eligibility screens."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from study_operator import (
        append_event,
        artifact_record,
        exclusive_study_lock,
        launcher_environment,
        load_manifest,
        result_files,
        utc_now,
    )
except ModuleNotFoundError:
    from .study_operator import (
        append_event,
        artifact_record,
        exclusive_study_lock,
        launcher_environment,
        load_manifest,
        result_files,
        utc_now,
    )

AMENDMENT_ID = "repository-repair-confirmatory-study@1/amendment-1"
SCREEN_REVISION = "revision30-protocol-eligibility@1"
SCREEN_WORKLOAD = "repository-repair-protocol-eligibility-screen"
REQUIRED_ELIGIBLE_CANDIDATES = 2


def load_amendment(path: Path) -> dict[str, Any]:
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("the protocol-screen amendment identity is invalid")
    candidates = amendment.get("candidate_order")
    if not isinstance(candidates, list) or len(candidates) < REQUIRED_ELIGIBLE_CANDIDATES:
        raise ValueError("the protocol-screen amendment has too few candidates")
    observed = set()
    for candidate in candidates:
        identity = (
            candidate.get("optimization_seed"),
            candidate.get("validation_seed_base"),
            candidate.get("test_seed_base"),
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in identity
            )
            or len(set(identity)) != 3
            or identity in observed
        ):
            raise ValueError("the protocol-screen candidate order is invalid")
        observed.add(identity)
    return amendment


def screen_results(receipt_directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    matches = []
    for path in result_files(receipt_directory):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(result, dict)
            and result.get("workload") == SCREEN_WORKLOAD
            and result.get("workload_revision") == SCREEN_REVISION
            and result.get("screen_completed") is True
            and isinstance(result.get("protocol_eligible"), bool)
            and all(
                not isinstance(result.get(key), bool)
                and isinstance(result.get(key), int)
                and result[key] > 0
                for key in (
                    "optimization_seed",
                    "validation_seed_base",
                    "test_seed_base_reserved_but_unread",
                )
            )
        ):
            matches.append((path, result))
    return matches


def candidate_identity(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(candidate["optimization_seed"]),
        int(candidate["validation_seed_base"]),
        int(candidate["test_seed_base"]),
    )


def result_identity(result: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(result["optimization_seed"]),
        int(result["validation_seed_base"]),
        int(result["test_seed_base_reserved_but_unread"]),
    )


def resolve_candidate(
    amendment: dict[str, Any],
    optimization_seed: int,
    receipt_directory: Path,
    *,
    preflight_only: bool,
) -> dict[str, Any]:
    candidates = amendment["candidate_order"]
    matches = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.get("optimization_seed") == optimization_seed
    ]
    if len(matches) != 1:
        raise ValueError("the requested seed is not a unique preregistered screen candidate")
    index, candidate = matches[0]
    completed = screen_results(receipt_directory)
    by_identity: dict[tuple[int, int, int], list[tuple[Path, dict[str, Any]]]] = {}
    for item in completed:
        by_identity.setdefault(result_identity(item[1]), []).append(item)
    duplicates = [identity for identity, items in by_identity.items() if len(items) > 1]
    if duplicates:
        raise RuntimeError("a protocol-screen candidate has duplicate completed results")
    identity = candidate_identity(candidate)
    if identity in by_identity and not preflight_only:
        raise RuntimeError("the requested protocol-screen candidate is already complete")
    missing_predecessors = [
        candidate_identity(predecessor)
        for predecessor in candidates[:index]
        if candidate_identity(predecessor) not in by_identity
    ]
    if missing_predecessors:
        raise RuntimeError("protocol-screen candidates must run in preregistered order")
    eligible_count = sum(item[1].get("protocol_eligible") is True for item in completed)
    if eligible_count >= REQUIRED_ELIGIBLE_CANDIDATES and identity not in by_identity:
        raise RuntimeError("two eligible candidates are already fixed; refusing extra screening")
    return candidate


def screen_environment(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    *,
    preflight_only: bool,
) -> dict[str, str]:
    condition = {
        "condition_id": f"k4_train_seed{candidate['optimization_seed']}",
        "optimization_seed": candidate["optimization_seed"],
        "validation_seed_base": candidate["validation_seed_base"],
        "test_seed_base": candidate["test_seed_base"],
        "training_tasks_per_update": 4,
    }
    environment = launcher_environment(
        manifest,
        condition,
        completion_budget=None,
        preflight_only=preflight_only,
    )
    environment["EQUINOX_STUDY_ELIGIBILITY_ONLY"] = "1"
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization_seed", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(
        repository_root / "research/studies/revision30-confirmatory-study.json"
    )
    amendment = load_amendment(
        repository_root / "research/studies/revision30-confirmatory-amendment-1.json"
    )
    receipt_directory = repository_root / "var/research-proofs"
    study_state_directory = repository_root / "var/revision30-study"
    candidate = resolve_candidate(
        amendment,
        arguments.optimization_seed,
        receipt_directory,
        preflight_only=arguments.preflight_only,
    )
    environment = screen_environment(
        manifest,
        candidate,
        preflight_only=arguments.preflight_only,
    )
    launcher = repository_root / "scripts/runpod-rl-proof"
    ledger_path = study_state_directory / "attempts.jsonl"
    lock_path = study_state_directory / "operator.lock"
    condition_id = f"protocol_screen_seed{arguments.optimization_seed}"

    with exclusive_study_lock(lock_path):
        before = set(result_files(receipt_directory))
        started_at = utc_now()
        append_event(
            ledger_path,
            {
                "schema_version": 1,
                "event": "eligibility_screen_started",
                "study_id": manifest["study_id"],
                "amendment_id": AMENDMENT_ID,
                "condition_id": condition_id,
                "started_at": started_at,
                "preflight_only": arguments.preflight_only,
                **candidate,
            },
        )
        completed = subprocess.run(
            [str(launcher)],
            cwd=repository_root,
            env=environment,
            check=False,
        )
        new_results = sorted(set(result_files(receipt_directory)) - before)
        new_screens = [
            (path, result)
            for path, result in screen_results(receipt_directory)
            if path in new_results
        ]
        if arguments.preflight_only:
            outcome = "passed" if completed.returncode == 0 else "failed"
        elif completed.returncode != 0:
            outcome = "failed"
        elif len(new_screens) != 1:
            outcome = "invalid_result"
        else:
            outcome = "eligible" if new_screens[0][1]["protocol_eligible"] else "ineligible"
        append_event(
            ledger_path,
            {
                "schema_version": 1,
                "event": "eligibility_screen_finished",
                "study_id": manifest["study_id"],
                "amendment_id": AMENDMENT_ID,
                "condition_id": condition_id,
                "started_at": started_at,
                "finished_at": utc_now(),
                "preflight_only": arguments.preflight_only,
                "exit_code": completed.returncode,
                "outcome": outcome,
                "result_artifacts": [artifact_record(path) for path in new_results],
            },
        )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not arguments.preflight_only and len(new_screens) != 1:
        raise SystemExit("the eligibility screen did not produce exactly one result")


if __name__ == "__main__":
    main()
