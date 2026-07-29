import json
from dataclasses import asdict

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


def repair(environment: RepositoryRepairEnvironment) -> None:
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
