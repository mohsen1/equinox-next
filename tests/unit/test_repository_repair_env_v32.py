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
    assert '"next_actions"' in ACTION_REMINDER
    assert "Reread after a rejected edit" in ACTION_REMINDER
    assert "Do not repeat unchanged evidence" in ACTION_REMINDER
    assert ACTION_PROTOCOL_REVISION == "repository-repair-json-tools@6"
    assert ENVIRONMENT_REVISION == "repository-repair-simulator@7"


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
    snapshot_payload = json.loads(snapshot.payload)

    restored = RepositoryRepairEnvironment.restore(repair_task, snapshot)
    restored_payload = json.loads(restored.capture_snapshot().payload)
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
    assert snapshot_payload["environment_revision"] == ENVIRONMENT_REVISION
    assert restored_payload["environment_revision"] == ENVIRONMENT_REVISION
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


def test_prompt_scaffold_exposes_unranked_mechanical_progress_without_fault_answers() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    environment.step(action({"tool": "list", "path": ""}))
    first_observed_path = sorted(repair_task.files)[0]
    environment.step(action({"tool": "read", "path": first_observed_path}))

    prompt = environment.policy_prompt("shared_prefix")
    opening_tag = prompt.index("<untrusted-environment-data>")
    environment_data = json.loads(prompt[opening_tag:].splitlines()[1])
    interface_state = environment_data["interface_state"]
    action_space = interface_state["mechanical_action_space"]

    assert interface_state["observed_paths"] == sorted(repair_task.files)
    assert interface_state["read_paths"] == [first_observed_path]
    assert interface_state["progress"] == {
        "accepted_diagnostic_actions": 2,
        "observed_path_count": len(repair_task.files),
        "read_path_count": 1,
        "steps_remaining": repair_task.complexity.repair_horizon - 2,
    }
    assert first_observed_path not in action_space["read_paths"]
    assert action_space["read_paths"] == sorted(set(repair_task.files) - {first_observed_path})
    assert action_space["list_paths"] == []
    assert action_space["test_allowed"] is True
    assert action_space["edit_paths_with_fresh_read"] == []
    assert action_space["finish_allowed"] is False
    assert "faults" not in json.dumps(interface_state)
    assert all(
        fault.old not in json.dumps(interface_state) and fault.new not in json.dumps(interface_state)
        for fault in repair_task.faults
    )


def test_repeated_accepted_diagnostics_are_rejected_with_unranked_recovery_choices() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    listing_action = {"tool": "list", "path": ""}
    first = environment.step(action(listing_action))
    repeated = environment.step(action(listing_action))
    recovery = json.loads(repeated.observation)

    assert first.accepted is True
    assert repeated.accepted is False
    assert recovery == {
        "error": "NO_PROGRESS_REPEAT",
        "next_actions": [
            {"path": path, "tool": "read"} for path in sorted(repair_task.files)
        ],
    }


def test_read_and_test_repeat_only_after_repository_state_changes() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]
    environment.step(action({"tool": "list", "path": ""}))

    first_read = environment.step(action({"tool": "read", "path": fault.path}))
    repeated_read = environment.step(action({"tool": "read", "path": fault.path}))
    first_test = environment.step(action({"tool": "test"}))
    repeated_test = environment.step(action({"tool": "test"}))
    edit = environment.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )
    changed_test = environment.step(action({"tool": "test"}))

    assert first_read.accepted is True
    assert repeated_read.accepted is False
    assert json.loads(repeated_read.observation)["error"] == "NO_PROGRESS_REPEAT"
    assert first_test.accepted is True
    assert repeated_test.accepted is False
    assert json.loads(repeated_test.observation)["error"] == "NO_PROGRESS_REPEAT"
    assert edit.accepted is True
    assert changed_test.accepted is True


def test_unchanged_list_and_read_stay_rejected_after_an_unrelated_edit() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]
    unrelated_path = next(path for path in sorted(repair_task.files) if path != fault.path)
    listing = {"tool": "list", "path": ""}
    unrelated_read = {"tool": "read", "path": unrelated_path}
    environment.step(action(listing))
    environment.step(action(unrelated_read))
    environment.step(action({"tool": "read", "path": fault.path}))
    environment.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )

    repeated_listing = environment.step(action(listing))
    repeated_unrelated_read = environment.step(action(unrelated_read))
    changed_fault_read = environment.step(action({"tool": "read", "path": fault.path}))

    assert repeated_listing.accepted is False
    assert json.loads(repeated_listing.observation)["error"] == "NO_PROGRESS_REPEAT"
    assert repeated_unrelated_read.accepted is False
    assert json.loads(repeated_unrelated_read.observation)["error"] == "NO_PROGRESS_REPEAT"
    assert changed_fault_read.accepted is True


def test_exhausted_repeat_does_not_supply_an_empty_recovery_choice() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    environment.step(action({"tool": "list", "path": ""}))
    for path in sorted(repair_task.files):
        environment.step(action({"tool": "read", "path": path}))
    environment.step(action({"tool": "test"}))

    repeated = environment.step(action({"tool": "test"}))
    recovery = json.loads(repeated.observation)

    assert repeated.accepted is False
    assert recovery == {
        "error": "NO_PROGRESS_REPEAT",
        "recovery": "Choose a new search query or another action enabled by interface_state.",
    }
    assert "next_actions" not in recovery
