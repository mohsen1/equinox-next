from __future__ import annotations

from copy import deepcopy
from typing import Any

ENVIRONMENT_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "environment_id": "cad.reconstruction",
        "name": "CAD reconstruction",
        "short_name": "CAD",
        "summary": "Reconstruct a bounded solid from canonical visual evidence.",
        "status": "LOCAL_FIXTURE",
        "launch_enabled": True,
        "action_space": "Typed CAD operations",
        "verifier": "Geometry, topology, dimensions, and canonical renders",
        "snapshot_strategy": "Logical state restore",
        "task_revision": "mounting-plate@sha256:fixture-v1",
        "complexity": {
            "dimensions": ["operation_count", "feature_count", "topology_depth"],
            "default_initial_level": 1,
            "default_max_level": 8,
        },
    },
    {
        "environment_id": "sqlite.repair",
        "name": "SQLite data repair",
        "short_name": "SQLite",
        "summary": "Inspect and repair a persistent database against hidden invariants.",
        "status": "CONFIGURATION_DRAFT",
        "launch_enabled": False,
        "action_space": "SQL queries and migrations",
        "verifier": "Schema, constraints, query assertions, and hidden invariants",
        "snapshot_strategy": "Database file checkpoint",
        "task_revision": "sqlite-repair@pending-runpool",
        "complexity": {
            "dimensions": ["table_count", "constraint_count", "fault_count", "repair_horizon"],
            "default_initial_level": 0,
            "default_max_level": 16,
        },
    },
    {
        "environment_id": "cli.debug",
        "name": "Filesystem and CLI debugging",
        "short_name": "CLI",
        "summary": "Diagnose and repair a sandboxed filesystem and process state.",
        "status": "CONFIGURATION_DRAFT",
        "launch_enabled": False,
        "action_space": "Shell commands through a bounded tool contract",
        "verifier": "Filesystem, process, permission, and service assertions",
        "snapshot_strategy": "Copy-on-write workspace checkpoint",
        "task_revision": "cli-debug@pending-runpool",
        "complexity": {
            "dimensions": ["component_count", "fault_count", "dependency_depth", "tool_budget"],
            "default_initial_level": 0,
            "default_max_level": 16,
        },
    },
    {
        "environment_id": "repository.repair",
        "name": "Micro-repository code repair",
        "short_name": "Repository",
        "summary": "Debug and patch a small repository against visible and hidden tests.",
        "status": "CONFIGURATION_DRAFT",
        "launch_enabled": False,
        "action_space": "Inspect, edit, and test repository files",
        "verifier": "Build, test, static, and hidden behavioral assertions",
        "snapshot_strategy": "Versioned workspace checkpoint",
        "task_revision": "repository-repair@pending-runpool",
        "complexity": {
            "dimensions": [
                "file_count",
                "failing_test_count",
                "dependency_depth",
                "repair_horizon",
            ],
            "default_initial_level": 0,
            "default_max_level": 16,
        },
    },
)


def environment_catalog() -> list[dict[str, Any]]:
    return deepcopy(list(ENVIRONMENT_CATALOG))


def environment_spec(environment_id: str) -> dict[str, Any] | None:
    return next(
        (
            deepcopy(item)
            for item in ENVIRONMENT_CATALOG
            if item["environment_id"] == environment_id
        ),
        None,
    )


def active_complexity_range(*, minimum: int, current: int, sampling_band: int) -> list[int]:
    return [max(minimum, current - sampling_band + 1), current]


def advance_complexity(
    *,
    current_level: int,
    maximum_level: int,
    window_attempts: int,
    window_successes: int,
    ordered_outcomes: list[bool],
    evaluation_window: int,
    mastery_threshold: float,
    promotion_step: int,
) -> dict[str, int | float | bool]:
    if window_attempts >= evaluation_window:
        raise ValueError("stored complexity window must be smaller than the evaluation window")
    if window_successes > window_attempts:
        raise ValueError("stored complexity successes cannot exceed attempts")
    if not ordered_outcomes:
        raise ValueError("at least one ordered complexity outcome is required")

    attempts = window_attempts
    successes = window_successes
    next_level = current_level
    evaluations = 0
    promotions = 0
    last_accuracy = 0.0
    for outcome in ordered_outcomes:
        attempts += 1
        successes += int(outcome)
        if attempts != evaluation_window:
            continue
        evaluations += 1
        last_accuracy = successes / evaluation_window
        if last_accuracy >= mastery_threshold and next_level < maximum_level:
            next_level = min(maximum_level, next_level + promotion_step)
            promotions += 1
        attempts = 0
        successes = 0

    return {
        "current_level": next_level,
        "window_attempts": attempts,
        "window_successes": successes,
        "last_accuracy": last_accuracy,
        "evaluated": evaluations > 0,
        "evaluation_count": evaluations,
        "promoted": promotions > 0,
        "promotion_count": promotions,
    }
