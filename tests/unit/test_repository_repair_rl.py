from research.runpod.repository_repair_env import BRANCH_WIDTH, encode_action, make_task
from research.runpod.repository_repair_rl import (
    BranchCollection,
    GeneratedAction,
    collect_branch_group,
    policy_examples,
    serialize_branch_group,
    sibling_advantages,
)


def test_sibling_advantage_is_leave_one_out_centered_and_zero_for_ties() -> None:
    advantages = sibling_advantages([1.0, 0.0, 0.0, 0.0])

    assert advantages[0] > 0
    assert all(value < 0 for value in advantages[1:])
    assert abs(sum(advantages)) < 1e-7
    assert sibling_advantages([0.25] * BRANCH_WIDTH) == [0.0] * BRANCH_WIDTH


def test_prefix_failure_is_excluded_without_manufacturing_returns() -> None:
    task = make_task(0, seed=12)

    def invalid_policy(_: str, __: bool, ___: int) -> GeneratedAction:
        return GeneratedAction(response='{"tool":"finish"}')

    collection = collect_branch_group(
        task,
        invalid_policy,
        stochastic=True,
        sampling_seed=5,
    )

    assert collection.exclusion_reason == "PREFIX_CHECKPOINT_NOT_REACHED"
    assert collection.snapshot is None
    assert collection.returns == []
    assert collection.advantages == []
    assert policy_examples(collection) == []


def test_serialized_branch_has_one_prefix_checkpoint_and_four_step_lanes() -> None:
    task = make_task(0, seed=44)
    calls = 0
    diagnostic = (
        '{"tool":"list","path":""}',
        '{"tool":"read","path":"tests/failures.txt"}',
    )
    fault = task.faults[0]
    continuation = (
        encode_action({"tool": "read", "path": fault.path}),
        encode_action(
            {
                "tool": "edit",
                "path": fault.path,
                "old": fault.old,
                "new": fault.new,
            }
        ),
        '{"tool":"test"}',
        '{"tool":"finish"}',
    )

    def scripted_policy(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        response = diagnostic[calls] if calls < 2 else continuation[(calls - 2) // BRANCH_WIDTH]
        calls += 1
        return GeneratedAction(response=response)

    collection = collect_branch_group(
        task,
        scripted_policy,
        stochastic=False,
        sampling_seed=100,
    )
    serialized = serialize_branch_group(collection, update=3)

    assert serialized["checkpoint"]["fidelity"] == "logical_restore"
    assert serialized["checkpoint"]["static_branch_width"] == 4
    assert len(serialized["shared_prefix"]["steps"]) == 2
    assert len(serialized["siblings"]) == 4
    assert all(len(sibling["steps"]) == 4 for sibling in serialized["siblings"])
    assert len({sibling["sampling_seed"] for sibling in serialized["siblings"]}) == 4
    assert all(sibling["passed"] for sibling in serialized["siblings"])


def test_policy_examples_never_include_prefix_and_split_weight_by_branch_actions() -> None:
    task = make_task(0, seed=8)
    generated = GeneratedAction(
        response='{"tool":"test"}',
        input_ids=(1, 2),
        attention_mask=(1, 1),
        completion_mask=(0, 1),
    )
    collection = BranchCollection(
        task=task,
        snapshot=None,
        prefix=None,  # type: ignore[arg-type]
        siblings=[],
        generated_by_sibling=[[generated, generated] for _ in range(4)],
        sampling_seeds=[1, 2, 3, 4],
        returns=[1.0, 0.0, 0.0, 0.0],
        advantages=[3.0, -1.0, -1.0, -1.0],
        exclusion_reason=None,
        replay=False,
    )

    examples = policy_examples(collection)

    assert len(examples) == 8
    assert [example.weight for example in examples[:2]] == [1.5, 1.5]
    assert all(example.weight == -0.5 for example in examples[2:])
