import json
from dataclasses import asdict

import pytest

from research.runpod import repository_repair_env as frozen_environment
from research.runpod import repository_repair_env_v32 as revision32
from research.runpod.repository_repair_env_v33 import (
    ACTION_PROTOCOL_REVISION,
    ACTION_REMINDER,
    ENVIRONMENT_REVISION,
    SYSTEM_PROMPT,
    RepositoryRepairEnvironment,
)


def task() -> frozen_environment.RepairTask:
    return frozen_environment.make_task(0, 33_033, split="train")


def action(value: dict[str, str]) -> str:
    return json.dumps(value, separators=(",", ":"))


def repair(environment: revision32.RepositoryRepairEnvironment) -> None:
    fault = environment.task.faults[0]
    assert environment.step(action({"tool": "list", "path": ""})).accepted is True
    assert environment.step(action({"tool": "read", "path": fault.path})).accepted is True
    assert (
        environment.step(
            action(
                {
                    "tool": "edit",
                    "path": fault.path,
                    "old": fault.old,
                    "new": fault.new,
                }
            )
        ).accepted
        is True
    )


def test_passing_test_before_horizon_terminates_as_solved() -> None:
    environment = RepositoryRepairEnvironment(task())
    repair(environment)

    result = environment.step(action({"tool": "test"}))

    assert result.index + 1 < environment.task.complexity.repair_horizon
    assert result.accepted is True
    assert result.verifier_passed is True
    assert result.terminal is True
    assert result.terminal_reason == "solved"
    assert environment.terminal is True
    assert environment.terminal_reason == "solved"


def test_passing_test_exactly_at_horizon_terminates_as_solved() -> None:
    environment = RepositoryRepairEnvironment(task())
    repair(environment)
    searches_needed = environment.task.complexity.repair_horizon - len(environment.steps) - 1
    for index in range(searches_needed):
        result = environment.step(action({"tool": "search", "query": f"padding-{index}"}))
        assert result.accepted is True
        assert result.terminal is False

    result = environment.step(action({"tool": "test"}))

    assert result.index + 1 == environment.task.complexity.repair_horizon
    assert result.accepted is True
    assert result.verifier_passed is True
    assert result.terminal is True
    assert result.terminal_reason == "solved"


def test_passing_edit_exactly_at_horizon_remains_horizon_exhausted() -> None:
    environment = RepositoryRepairEnvironment(task())
    fault = environment.task.faults[0]
    assert environment.step(action({"tool": "list", "path": ""})).accepted is True
    assert environment.step(action({"tool": "read", "path": fault.path})).accepted is True
    searches_needed = environment.task.complexity.repair_horizon - len(environment.steps) - 1
    for index in range(searches_needed):
        result = environment.step(action({"tool": "search", "query": f"padding-{index}"}))
        assert result.accepted is True
        assert result.terminal is False

    result = environment.step(
        action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        )
    )

    assert result.index + 1 == environment.task.complexity.repair_horizon
    assert result.accepted is True
    assert result.verifier_passed is True
    assert result.terminal is True
    assert result.terminal_reason == "horizon_exhausted"
    assert result.reward == 0.0
    assert environment.terminal_reward == 0.0


def test_failing_test_before_horizon_remains_nonterminal() -> None:
    environment = RepositoryRepairEnvironment(task())
    assert environment.step(action({"tool": "list", "path": ""})).accepted is True

    result = environment.step(action({"tool": "test"}))

    assert result.index + 1 < environment.task.complexity.repair_horizon
    assert result.accepted is True
    assert result.verifier_passed is False
    assert result.terminal is False
    assert result.terminal_reason is None
    assert result.reward == 0.0
    assert environment.terminal is False


def test_failing_test_does_not_terminate_as_solved() -> None:
    environment = RepositoryRepairEnvironment(task())
    assert environment.step(action({"tool": "list", "path": ""})).accepted is True
    searches_needed = environment.task.complexity.repair_horizon - len(environment.steps) - 1
    for index in range(searches_needed):
        result = environment.step(action({"tool": "search", "query": f"padding-{index}"}))
        assert result.accepted is True
        assert result.terminal is False

    result = environment.step(action({"tool": "test"}))

    assert result.index + 1 == environment.task.complexity.repair_horizon
    assert result.accepted is True
    assert result.verifier_passed is False
    assert result.terminal is True
    assert result.terminal_reason == "horizon_exhausted"
    assert result.reward == 0.0


def test_passing_finish_semantics_remain_unchanged() -> None:
    for environment_type in (
        revision32.RepositoryRepairEnvironment,
        RepositoryRepairEnvironment,
    ):
        environment = environment_type(task())
        repair(environment)

        result = environment.step(action({"tool": "finish"}))

        assert result.accepted is True
        assert result.verifier_passed is True
        assert result.terminal is True
        assert result.terminal_reason == "solved"
        assert result.reward > 0.0
        assert result.reward == environment.terminal_reward


def test_failing_finish_semantics_remain_unchanged() -> None:
    for environment_type in (
        revision32.RepositoryRepairEnvironment,
        RepositoryRepairEnvironment,
    ):
        environment = environment_type(task())
        assert environment.step(action({"tool": "list", "path": ""})).accepted is True

        result = environment.step(action({"tool": "finish"}))

        assert result.accepted is True
        assert result.verifier_passed is False
        assert result.terminal is True
        assert result.terminal_reason == "finished_with_failures"
        assert result.reward == 0.0
        assert environment.terminal_reward == 0.0


def test_disallowed_passing_test_cannot_solve() -> None:
    environment = RepositoryRepairEnvironment(task())
    repair(environment)

    result = environment.step(
        action({"tool": "test"}),
        allowed_tools=frozenset({"finish"}),
    )

    assert result.accepted is False
    assert result.verifier_passed is True
    assert result.observation == "Action test is not allowed in this trajectory phase."
    assert result.terminal is False
    assert result.terminal_reason is None
    assert result.reward == 0.0
    assert environment.terminal is False


def test_v32_v33_differ_only_on_passing_test_terminalization() -> None:
    repair_task = task()
    revision32_environment = revision32.RepositoryRepairEnvironment(repair_task)
    revision33_environment = RepositoryRepairEnvironment(repair_task)
    fault = repair_task.faults[0]
    trajectory = (
        {"tool": "list", "path": ""},
        {"tool": "read", "path": fault.path},
        {
            "tool": "edit",
            "path": fault.path,
            "old": fault.old,
            "new": fault.new,
        },
        {"tool": "test"},
    )

    revision32_results = [
        revision32_environment.step(action(trajectory_action)) for trajectory_action in trajectory
    ]
    revision33_results = [
        revision33_environment.step(action(trajectory_action)) for trajectory_action in trajectory
    ]
    revision32_test = revision32_results[-1]
    revision33_test = revision33_results[-1]

    assert [result.action for result in revision32_results] == [
        result.action for result in revision33_results
    ]
    assert [result.accepted for result in revision32_results] == [
        result.accepted for result in revision33_results
    ]
    assert revision32_test.observation == revision33_test.observation
    assert revision32_test.verifier_passed is True
    assert revision33_test.verifier_passed is True
    assert revision32_test.terminal is False
    assert revision32_test.terminal_reason is None
    assert revision32_test.reward == 0.0
    assert revision33_test.terminal is True
    assert revision33_test.terminal_reason == "solved"
    assert revision33_test.reward > 0.0


def test_snapshot_restore_preserves_v33_state_and_terminal_semantics() -> None:
    repair_task = task()
    environment = RepositoryRepairEnvironment(repair_task)
    repair(environment)
    snapshot = environment.capture_snapshot()

    restored = RepositoryRepairEnvironment.restore(repair_task, snapshot)
    restored_snapshot = restored.capture_snapshot()
    result = restored.step(action({"tool": "test"}))
    terminal_snapshot = restored.capture_snapshot()
    terminal_restored = RepositoryRepairEnvironment.restore(
        repair_task,
        terminal_snapshot,
    )

    assert json.loads(snapshot.payload)["environment_revision"] == ENVIRONMENT_REVISION
    assert restored_snapshot.payload_digest == snapshot.payload_digest
    assert restored_snapshot.payload == snapshot.payload
    assert result.terminal_reason == "solved"
    assert terminal_restored.capture_snapshot().payload_digest == (terminal_snapshot.payload_digest)
    assert terminal_restored.steps == restored.steps
    assert terminal_restored.terminal_reward == restored.terminal_reward


def test_v32_and_v33_snapshots_fail_closed_across_revisions() -> None:
    repair_task = task()
    revision32_snapshot = revision32.RepositoryRepairEnvironment(repair_task).capture_snapshot()
    revision33_snapshot = RepositoryRepairEnvironment(repair_task).capture_snapshot()

    with pytest.raises(ValueError, match="snapshot environment revision mismatch"):
        RepositoryRepairEnvironment.restore(repair_task, revision32_snapshot)
    with pytest.raises(ValueError, match="snapshot environment revision mismatch"):
        revision32.RepositoryRepairEnvironment.restore(repair_task, revision33_snapshot)


def test_prompt_and_revision_contract_only_add_passing_test_semantics() -> None:
    passing_test_rule = (
        "- An accepted test whose verifier passes finishes immediately; "
        "do not send a redundant finish action."
    )
    expected_system_prompt = revision32.SYSTEM_PROMPT.replace(
        "- Do not repeat an accepted diagnostic when it would return unchanged evidence; "
        "the interface rejects no-progress repeats.\n",
        "- Do not repeat an accepted diagnostic when it would return unchanged evidence; "
        "the interface rejects no-progress repeats.\n"
        f"{passing_test_rule}\n",
    )
    expected_action_reminder = revision32.ACTION_REMINDER.replace(
        "Do not repeat unchanged evidence. Reread after a rejected edit.\n",
        "Do not repeat unchanged evidence. Reread after a rejected edit.\n"
        "An accepted passing test finishes immediately; "
        "do not send a redundant finish action.\n",
    )

    assert ENVIRONMENT_REVISION == "repository-repair-simulator@9"
    assert ACTION_PROTOCOL_REVISION == "repository-repair-json-tools@8"
    assert expected_system_prompt == SYSTEM_PROMPT
    assert expected_action_reminder == ACTION_REMINDER
    assert SYSTEM_PROMPT.count(passing_test_rule) == 1
    assert (
        RepositoryRepairEnvironment(task()).policy_prompt("continuation").endswith(ACTION_REMINDER)
    )


def test_passing_test_reward_and_trajectory_are_internally_consistent() -> None:
    environment = RepositoryRepairEnvironment(task())
    repair(environment)

    result = environment.step(action({"tool": "test"}))
    components = environment.reward_components()
    payload = json.loads(environment.capture_snapshot().payload)

    assert result == environment.steps[-1]
    assert result.reward == environment.terminal_reward
    assert result.reward == components["terminal_aggregate"]
    assert result.reward > 0.0
    assert components["hidden_correctness"] is True
    assert components["accepted_action_count"] == len(environment.steps)
    assert components["verifier_submission_count"] == 1
    assert result.state_digest_after == environment.state_digest
    assert payload["terminal"] is True
    assert payload["terminal_reason"] == "solved"
    assert payload["terminal_reward"] == result.reward
    assert payload["steps"][-1] == asdict(result)
