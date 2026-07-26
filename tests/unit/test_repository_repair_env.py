import json

import pytest

from research.runpod.repository_repair_env import (
    BRANCH_WIDTH,
    COMPLEXITY_LEVELS,
    DIAGNOSTIC_TOOLS,
    RepositoryRepairEnvironment,
    diagnostic_actions,
    encode_action,
    make_task,
    parse_action,
    teacher_continuation_actions,
)


def test_complexity_changes_repository_faults_depth_and_horizon() -> None:
    tasks = [make_task(level, seed=91) for level in range(len(COMPLEXITY_LEVELS))]

    assert [len(task.files) for task in tasks] == [4, 6, 8, 10]
    assert [len(task.faults) for task in tasks] == [1, 1, 2, 3]
    assert [task.complexity.dependency_depth for task in tasks] == [1, 2, 3, 4]
    assert [task.complexity.repair_horizon for task in tasks] == [8, 10, 14, 18]


def test_snapshot_restores_identical_isolated_continuations() -> None:
    task = make_task(2, seed=17)
    prefix = RepositoryRepairEnvironment(task)
    for action in diagnostic_actions(task):
        step = prefix.step(encode_action(action))
        assert step.accepted
        assert step.tool in DIAGNOSTIC_TOOLS

    snapshot = prefix.capture_snapshot()
    siblings = [RepositoryRepairEnvironment.restore(task, snapshot) for _ in range(BRANCH_WIDTH)]
    assert len({sibling.trajectory_digest for sibling in siblings}) == 1

    first_edit = next(
        action for action in teacher_continuation_actions(task) if action["tool"] == "edit"
    )
    siblings[0].step(encode_action(first_edit))

    assert siblings[0].files != siblings[1].files
    assert siblings[1].capture_snapshot().payload_digest == snapshot.payload_digest
    assert siblings[2].capture_snapshot().payload_digest == snapshot.payload_digest
    assert siblings[3].capture_snapshot().payload_digest == snapshot.payload_digest


def test_teacher_continuation_solves_without_executing_repository_code() -> None:
    task = make_task(3, seed=103)
    prefix = RepositoryRepairEnvironment(task)
    for action in diagnostic_actions(task):
        prefix.step(encode_action(action))
    environment = RepositoryRepairEnvironment.restore(task, prefix.capture_snapshot())

    for action in teacher_continuation_actions(task):
        last_step = environment.step(encode_action(action))

    assert last_step.terminal
    assert last_step.terminal_reason == "solved"
    assert last_step.verifier_passed
    assert 0 < last_step.reward <= 1


@pytest.mark.parametrize(
    "response",
    (
        "",
        ' {"tool":"test"}',
        '{"tool":"test"}\n',
        '```json\n{"tool":"test"}\n```',
        '{"tool":"read","path":"README.md","extra":"x"}',
        '{"tool":"shell","command":"pytest"}',
    ),
)
def test_action_parser_rejects_noncanonical_or_untrusted_shapes(response: str) -> None:
    assert parse_action(response) is None


def test_paths_and_edits_are_bounded() -> None:
    task = make_task(0, seed=7)
    environment = RepositoryRepairEnvironment(task)

    assert not environment.step('{"tool":"read","path":"../secret"}').accepted
    assert not environment.step(
        json.dumps(
            {
                "tool": "edit",
                "path": next(iter(task.files)),
                "old": "x" * 501,
                "new": "safe",
            },
            separators=(",", ":"),
        )
    ).accepted


def test_horizon_terminates_unsolved_trajectory() -> None:
    task = make_task(0, seed=29)
    environment = RepositoryRepairEnvironment(task)

    for _ in range(task.complexity.repair_horizon):
        last_step = environment.step('{"tool":"test"}')

    assert last_step.terminal
    assert last_step.terminal_reason == "horizon_exhausted"
    assert last_step.reward == 0
    with pytest.raises(RuntimeError):
        environment.step('{"tool":"test"}')
