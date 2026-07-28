import json

from research.runpod import repository_repair_env as frozen_environment
from research.runpod.repository_repair_env_v31 import (
    ACTION_PROTOCOL_REVISION,
    ACTION_REMINDER,
    ENVIRONMENT_REVISION,
    SYSTEM_PROMPT,
    RepositoryRepairEnvironment,
)


def task() -> frozen_environment.RepairTask:
    return frozen_environment.make_task(0, 31_031, split="train")


def test_prompt_has_no_fake_nonempty_path_and_leads_with_contract() -> None:
    assert SYSTEM_PROMPT.startswith("OUTPUT CONTRACT\n")
    assert "relative/path.py" not in SYSTEM_PROMPT
    assert '{"tool":"list","path":""}' in SYSTEM_PROMPT
    assert ACTION_PROTOCOL_REVISION == "repository-repair-json-tools@4"
    assert ENVIRONMENT_REVISION == "repository-repair-simulator@5"


def test_invented_path_recovers_through_repository_listing() -> None:
    environment = RepositoryRepairEnvironment(task())

    rejected = environment.step(
        '{"tool":"read","path":"relative/path.py"}',
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )
    recovery = json.loads(rejected.observation)
    assert rejected.accepted is False
    assert recovery == {
        "error": "PATH_NOT_OBSERVED_OR_NOT_FOUND",
        "next_action": {"path": "", "tool": "list"},
    }

    listing = environment.step(
        '{"tool":"list","path":""}',
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )
    observed_path = json.loads(listing.observation)["files"][0]
    read = environment.step(
        json.dumps({"tool": "read", "path": observed_path}, separators=(",", ":")),
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )
    assert listing.accepted is True
    assert read.accepted is True


def test_stale_edit_returns_current_content_and_reread_edit_succeeds() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]

    rejected = environment.step(
        json.dumps(
            {
                "tool": "edit",
                "path": fault.path,
                "old": "stale text that is not present",
                "new": fault.new,
            },
            separators=(",", ":"),
        )
    )
    recovery = json.loads(rejected.observation)
    assert rejected.accepted is False
    assert recovery["error"] == "EDIT_TARGET_NOT_FOUND"
    assert recovery["path"] == fault.path
    assert recovery["current_file"] == repair_task.files[fault.path]

    read = environment.step(json.dumps({"tool": "read", "path": fault.path}, separators=(",", ":")))
    applied = environment.step(
        json.dumps(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            },
            separators=(",", ":"),
        )
    )
    assert read.accepted is True
    assert applied.accepted is True


def test_snapshot_restore_keeps_revision31_environment_type() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    snapshot = environment.capture_snapshot()

    restored = RepositoryRepairEnvironment.restore(repair_task, snapshot)

    assert isinstance(restored, RepositoryRepairEnvironment)
    assert restored.state_digest == environment.state_digest
    assert restored.policy_prompt("continuation").endswith(ACTION_REMINDER)
