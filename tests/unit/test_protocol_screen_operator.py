from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.runpod.protocol_screen_operator import (
    load_amendment,
    resolve_candidate,
    screen_environment,
)


def amendment() -> dict:
    return {
        "amendment_id": "repository-repair-confirmatory-study@1/amendment-1",
        "candidate_order": [
            {
                "optimization_seed": 307,
                "validation_seed_base": 220_000_000,
                "test_seed_base": 260_000_000,
            },
            {
                "optimization_seed": 401,
                "validation_seed_base": 320_000_000,
                "test_seed_base": 360_000_000,
            },
        ],
    }


def write_screen(
    root: Path,
    *,
    seed: int,
    validation_seed: int,
    test_seed: int,
    eligible: bool,
) -> None:
    (root / f"runpod-proof-screen-{seed}.result.json").write_text(
        json.dumps(
            {
                "workload": "repository-repair-protocol-eligibility-screen",
                "workload_revision": "revision30-protocol-eligibility@1",
                "screen_completed": True,
                "protocol_eligible": eligible,
                "optimization_seed": seed,
                "validation_seed_base": validation_seed,
                "test_seed_base_reserved_but_unread": test_seed,
            }
        ),
        encoding="utf-8",
    )


def test_operator_requires_preregistered_candidate_order(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="preregistered order"):
        resolve_candidate(amendment(), 401, tmp_path, preflight_only=False)

    write_screen(
        tmp_path,
        seed=307,
        validation_seed=220_000_000,
        test_seed=260_000_000,
        eligible=False,
    )

    candidate = resolve_candidate(amendment(), 401, tmp_path, preflight_only=False)
    assert candidate["optimization_seed"] == 401


def test_operator_stops_after_two_eligible_candidates(tmp_path: Path) -> None:
    amended = amendment()
    amended["candidate_order"].append(
        {
            "optimization_seed": 503,
            "validation_seed_base": 420_000_000,
            "test_seed_base": 460_000_000,
        }
    )
    for seed, validation_seed, test_seed in (
        (307, 220_000_000, 260_000_000),
        (401, 320_000_000, 360_000_000),
    ):
        write_screen(
            tmp_path,
            seed=seed,
            validation_seed=validation_seed,
            test_seed=test_seed,
            eligible=True,
        )

    with pytest.raises(RuntimeError, match="two eligible"):
        resolve_candidate(amended, 503, tmp_path, preflight_only=False)


def test_screen_environment_is_baseline_only() -> None:
    manifest = {
        "study_id": "repository-repair-confirmatory-study@1",
        "model": {"id": "Qwen/Qwen2.5-Coder-3B-Instruct"},
        "shared_configuration": {
            "target_runtime_seconds": 10800,
            "maximum_resume_gap_seconds": 2700,
            "maximum_updates": 120,
            "validation_examples": 8,
            "test_examples_per_level": 12,
            "mastery_windows": 2,
            "replay_tasks_per_level": 1,
            "maximum_final_evaluation_reserve_seconds": 2700,
        },
    }

    environment = screen_environment(
        manifest,
        amendment()["candidate_order"][0],
        preflight_only=True,
    )

    assert environment["EQUINOX_STUDY_ELIGIBILITY_ONLY"] == "1"
    assert environment["EQUINOX_RUNPOD_PREFLIGHT_ONLY"] == "1"
    assert environment["EQUINOX_STUDY_CONDITION"] == "k4_train"
    assert environment["EQUINOX_RL_SEED"] == "307"


def test_repository_amendment_is_current() -> None:
    root = Path(__file__).resolve().parents[2]
    observed = load_amendment(root / "research/studies/revision30-confirmatory-amendment-2.json")
    original = json.loads(
        (root / "research/studies/revision30-confirmatory-amendment-1.json").read_text(
            encoding="utf-8"
        )
    )

    assert observed["candidate_order"][:4] == original["candidate_order"]
    assert observed["algorithm_change"] is False
    assert observed["gate_change"] is False
    assert [item["optimization_seed"] for item in observed["candidate_order"]] == [
        307,
        401,
        503,
        601,
        701,
        809,
        907,
        1009,
        1103,
        1201,
        1303,
        1409,
    ]
