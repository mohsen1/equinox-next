import json

from research.runpod import repository_repair_env as frozen_environment
from research.runpod.repository_repair_env_v32 import (
    ACTION_PROTOCOL_REVISION,
    ACTION_REMINDER,
    ENVIRONMENT_REVISION,
    SYSTEM_PROMPT,
    RepositoryRepairEnvironment,
)


def task() -> frozen_environment.RepairTask:
    return frozen_environment.make_task(0, 32_032, split="train")


def action(value: dict[str, str]) -> str:
    return json.dumps(value, separators=(",", ":"))


def test_prompt_encodes_evidence_and_recovery_decisions_without_example_paths() -> None:
    assert SYSTEM_PROMPT.startswith("OUTPUT CONTRACT\n")
    assert "examples only" in SYSTEM_PROMPT
    assert "expose no reusable non-empty repository path" in SYSTEM_PROMPT
    assert '{"tool":"list","path":""}' in SYSTEM_PROMPT
    assert '"next_action"' in ACTION_REMINDER
    assert "A rejected edit invalidates its prior read" in ACTION_REMINDER
    assert ACTION_PROTOCOL_REVISION == "repository-repair-json-tools@5"
    assert ENVIRONMENT_REVISION == "repository-repair-simulator@6"


def test_existing_but_unobserved_path_recovers_through_exact_root_listing() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    existing_path = sorted(repair_task.files)[0]

    rejected = environment.step(
        action({"tool": "read", "path": existing_path}),
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )
    assert rejected.accepted is False
    assert json.loads(rejected.observation) == {
        "error": "PATH_EVIDENCE_REQUIRED",
        "next_action": {"path": "", "tool": "list"},
    }

    listing = environment.step(
        action({"tool": "list", "path": ""}),
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )
    observed_path = json.loads(listing.observation)["files"][0]
    read = environment.step(
        action({"tool": "read", "path": observed_path}),
        allowed_tools=frozen_environment.DIAGNOSTIC_TOOLS,
    )

    assert listing.accepted is True
    assert read.accepted is True


def test_search_or_test_cannot_replace_the_required_first_root_listing() -> None:
    for attempted in ({"tool": "search", "query": "repair"}, {"tool": "test"}):
        environment = RepositoryRepairEnvironment(task())

        rejected = environment.step(action(attempted))

        assert rejected.accepted is False
        assert json.loads(rejected.observation)["next_action"] == {"path": "", "tool": "list"}


def test_stale_edit_requires_recovery_read_before_another_edit() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]

    assert environment.step(action({"tool": "list", "path": ""})).accepted is True
    assert environment.step(action({"tool": "read", "path": fault.path})).accepted is True

    stale_edit = {
        "tool": "edit",
        "path": fault.path,
        "old": "stale text that is not present",
        "new": fault.new,
    }
    rejected = environment.step(action(stale_edit))
    recovery = json.loads(rejected.observation)

    assert rejected.accepted is False
    assert recovery["error"] == "EDIT_TARGET_NOT_FOUND"
    assert recovery["path"] == fault.path
    assert recovery["current_file"] == repair_task.files[fault.path]
    assert recovery["next_action"] == {"path": fault.path, "tool": "read"}

    repeated = environment.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )
    assert repeated.accepted is False
    assert json.loads(repeated.observation) == {
        "error": "FRESH_READ_REQUIRED",
        "next_action": {"path": fault.path, "tool": "read"},
        "path": fault.path,
    }

    reread = environment.step(action({"tool": "read", "path": fault.path}))
    applied = environment.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )
    assert reread.accepted is True
    assert applied.accepted is True


def test_snapshot_restore_reconstructs_path_ledger_and_read_freshness() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]
    environment.step(action({"tool": "list", "path": ""}))
    environment.step(action({"tool": "read", "path": fault.path}))
    snapshot = environment.capture_snapshot()

    restored = RepositoryRepairEnvironment.restore(repair_task, snapshot)
    applied = restored.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )

    assert isinstance(restored, RepositoryRepairEnvironment)
    assert applied.accepted is True


def test_phase_instruction_and_action_reminder_stay_outside_untrusted_data() -> None:
    prompt = RepositoryRepairEnvironment(task()).policy_prompt("shared_prefix")

    opening_tag = prompt.index("<untrusted-environment-data>")
    closing_tag = prompt.index("</untrusted-environment-data>")
    reminder = prompt.index(ACTION_REMINDER)
    untrusted_data = json.loads(prompt[opening_tag:].splitlines()[1])

    assert prompt.startswith("PHASE INSTRUCTION\n")
    assert "instruction" not in untrusted_data
    assert closing_tag < reminder
    assert prompt.endswith(ACTION_REMINDER)
