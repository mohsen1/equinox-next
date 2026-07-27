import json

import pytest

import research.runpod.repository_repair_env as repository_repair_env
from research.runpod.repository_repair_env import (
    BRANCH_WIDTH,
    COMPLEXITY_LEVELS,
    DIAGNOSTIC_TOOLS,
    STRUCTURAL_MIRROR_DISCLOSURES,
    TEMPLATE_SPLITS,
    RepositoryRepairEnvironment,
    _semantically_matches,
    diagnostic_actions,
    encode_action,
    make_task,
    make_tasks,
    parse_action,
    self_test,
    semantic_task_universe_size,
    teacher_continuation_actions,
)


def test_environment_self_test_covers_each_level() -> None:
    result = self_test()

    assert result["self_test_passed"] is True
    assert result["levels"] == len(COMPLEXITY_LEVELS)
    assert result["tasks_checked"] == 26


def test_complexity_changes_repository_faults_depth_and_horizon() -> None:
    tasks = [make_task(level, seed=91) for level in range(len(COMPLEXITY_LEVELS))]

    assert [len(task.files) for task in tasks] == [4, 6, 8, 10]
    assert [len(task.faults) for task in tasks] == [1, 1, 2, 3]
    assert [task.complexity.dependency_depth for task in tasks] == [1, 2, 3, 4]
    assert [task.complexity.repair_horizon for task in tasks] == [8, 10, 14, 18]


def test_task_families_are_disjoint_across_train_validation_and_test() -> None:
    families = {
        split: {
            fault.family_id
            for task in (make_task(3, seed, split=split) for seed in range(20))
            for fault in task.faults
        }
        for split in ("train", "validation", "test")
    }

    assert families["train"].isdisjoint(families["validation"])
    assert families["train"].isdisjoint(families["test"])
    assert families["validation"].isdisjoint(families["test"])


def test_held_out_splits_cover_the_configured_semantic_sample_sizes() -> None:
    assert semantic_task_universe_size(0, "validation") == 16
    assert semantic_task_universe_size(0, "test") == 12

    validation = make_tasks(0, 8, seed=101, split="validation")
    test = make_tasks(0, 12, seed=202, split="test")

    assert len({task.semantic_task_id for task in validation}) == 8
    assert len({task.semantic_task_id for task in test}) == 12
    assert all(str(task.task_id) != task.semantic_task_id for task in validation)


def test_task_generation_can_exclude_a_prior_semantic_window() -> None:
    first = make_tasks(0, 8, seed=303, split="validation")
    first_semantics = frozenset(task.semantic_task_id for task in first)
    second = make_tasks(
        0,
        8,
        seed=404,
        split="validation",
        exclude_semantic_task_ids=first_semantics,
    )

    assert first_semantics.isdisjoint(task.semantic_task_id for task in second)


def test_exclusions_from_another_split_do_not_reduce_semantic_capacity() -> None:
    test_semantics = frozenset(
        task.semantic_task_id for task in make_tasks(0, 12, seed=505, split="test")
    )

    validation = make_tasks(
        0,
        16,
        seed=606,
        split="validation",
        exclude_semantic_task_ids=test_semantics,
    )

    assert len(validation) == 16


def test_foreign_prefixed_exclusion_does_not_reduce_semantic_capacity() -> None:
    validation = make_tasks(
        0,
        16,
        seed=707,
        split="validation",
        exclude_semantic_task_ids=frozenset(("repo-semantic-validation-0-not-a-member",)),
    )

    assert len(validation) == 16


def test_task_generation_enumerates_after_rejection_sampling_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = make_task(0, seed=808)
    monkeypatch.setattr(
        repository_repair_env,
        "make_task",
        lambda level, seed, *, split="train": duplicate,
    )

    tasks = repository_repair_env.make_tasks(0, 2, seed=808)

    assert len(tasks) == 2
    assert len({task.semantic_task_id for task in tasks}) == 2


def test_every_faulty_template_is_a_failed_test_instead_of_a_verifier_error() -> None:
    for templates in TEMPLATE_SPLITS.values():
        for family_id, source, old, new, _ in templates:
            expected = source.replace(old, new, 1)

            assert not _semantically_matches(source, expected, family_id)
            assert _semantically_matches(expected, expected, family_id)


@pytest.mark.parametrize(
    ("family_id", "candidate", "expected"),
    (
        (
            "last",
            "def last(values):\n    return max(values)\n",
            "def last(values):\n    return values[-1]\n",
        ),
        (
            "first",
            "def first(values, fallback):\n    return min(values) if values else fallback\n",
            "def first(values, fallback):\n    return values[0] if values else fallback\n",
        ),
        (
            "safe_head",
            "def safe_head(values, fallback):\n    return min(values) if values else fallback\n",
            "def safe_head(values, fallback):\n    return values[0] if values else fallback\n",
        ),
        (
            "middle",
            "def middle(values):\n    return max(values)\n",
            "def middle(values):\n    return values[1]\n",
        ),
        (
            "different",
            "def different(left, right):\n    return left is not right\n",
            "def different(left, right):\n    return left != right\n",
        ),
        (
            "maximum_three",
            ("def maximum_three(first, second, third):\n    return max(first, third)\n"),
            ("def maximum_three(first, second, third):\n    return max(first, second, third)\n"),
        ),
    ),
)
def test_order_sensitive_verifier_rejects_non_equivalent_decoys(
    family_id: str,
    candidate: str,
    expected: str,
) -> None:
    assert not _semantically_matches(candidate, expected, family_id)


def test_square_verifier_accepts_exponentiation() -> None:
    assert _semantically_matches(
        "def square(value):\n    return value ** 2\n",
        "def square(value):\n    return value * value\n",
        "square",
    )
    assert not _semantically_matches(
        "def square(value):\n    return value ** 999999999\n",
        "def square(value):\n    return value * value\n",
        "square",
    )


@pytest.mark.parametrize(
    "candidate",
    (
        "def total(values):\n    return values * 1000000 * 1000000\n",
        "def total(values):\n    return values * 1001\n",
        "def total(values):\n    return 1000001\n",
    ),
)
def test_hidden_verifier_rejects_unbounded_integer_and_sequence_expressions(
    candidate: str,
) -> None:
    assert not _semantically_matches(
        candidate,
        "def total(values):\n    return sum(values)\n",
        "total",
    )


def test_task_keyed_numeric_probes_reject_fixed_probe_reward_hack() -> None:
    assert _semantically_matches(
        "def is_positive(value):\n    return value == 4\n",
        "def is_positive(value):\n    return value > 0\n",
        "is_positive",
    )
    assert not _semantically_matches(
        "def is_positive(value):\n    return value == 4\n",
        "def is_positive(value):\n    return value > 0\n",
        "is_positive",
        probe_seed="repo-task-held-out-17",
    )


def test_task_keyed_sequence_probes_reject_fixed_probe_reward_hack() -> None:
    candidate = (
        "def last(values):\n"
        '    return 3 if values == (1, 2, 3) else (2 if values == (3, 1, 2) else "b")\n'
    )
    expected = "def last(values):\n    return values[-1]\n"

    assert _semantically_matches(candidate, expected, "last")
    assert not _semantically_matches(
        candidate,
        expected,
        "last",
        probe_seed="repo-task-held-out-17",
    )


def test_verifier_probe_seed_is_hidden_from_policy_observation() -> None:
    task = make_task(0, 17)
    observation = json.loads(RepositoryRepairEnvironment(task).initial_observation())

    assert task.verifier_probe_seed not in json.dumps(observation)
    assert "verifier_probe_seed" not in observation


def test_structural_mirror_disclosure_covers_known_cross_split_shapes() -> None:
    disclosed = {family for item in STRUCTURAL_MIRROR_DISCLOSURES for family in item["families"]}

    assert {"combine", "multiply", "subtract", "square"} <= disclosed
    assert {"minimum", "maximum", "bounded_lower", "maximum_three"} <= disclosed
    assert {"last", "middle"} <= disclosed
    assert {"coalesce", "default_zero"} <= disclosed
    assert {"different", "negate", "both"} <= disclosed


def test_hidden_verifier_accepts_semantically_equivalent_bounded_expression() -> None:
    task = next(
        make_task(0, seed)
        for seed in range(100)
        if make_task(0, seed).faults[0].family_id == "combine"
    )
    environment = RepositoryRepairEnvironment(task)
    fault = task.faults[0]
    equivalent = encode_action(
        {
            "tool": "edit",
            "path": fault.path,
            "old": fault.old,
            "new": "return sum((left, right))",
        }
    )

    edited = environment.step(equivalent)
    tested = environment.step('{"tool":"test"}')

    assert edited.accepted
    assert tested.verifier_passed


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


def test_shared_prefix_rejects_mutation_without_changing_repository() -> None:
    task = make_task(0, seed=7)
    environment = RepositoryRepairEnvironment(task)
    fault = task.faults[0]
    response = encode_action(
        {
            "tool": "edit",
            "path": fault.path,
            "old": fault.old,
            "new": fault.new,
        }
    )

    step = environment.step(response, allowed_tools=DIAGNOSTIC_TOOLS)

    assert not step.accepted
    assert environment.files == task.files


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
