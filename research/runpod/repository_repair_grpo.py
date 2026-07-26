"""Restored-continuation post-training for deterministic repository repair.

This workload collects one policy-generated diagnostic prefix, snapshots the
repository and transcript, restores four continuations, and applies
sibling-relative credit only to post-snapshot action tokens. Candidate content
is verified as inert simulator state and is never executed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

try:
    from research.runpod.repository_repair_env import (
        ACTION_PROTOCOL_REVISION,
        BRANCH_WIDTH,
        COMPLEXITY_LEVELS,
        DIAGNOSTIC_TOOLS,
        ENVIRONMENT_REVISION,
        VERIFIER_REVISION,
        EnvironmentSnapshot,
        RepairTask,
        RepositoryRepairEnvironment,
        diagnostic_actions,
        encode_action,
        make_task,
        make_tasks,
        teacher_continuation_actions,
    )
except ModuleNotFoundError:
    from repository_repair_env import (  # type: ignore[no-redef]
        ACTION_PROTOCOL_REVISION,
        BRANCH_WIDTH,
        COMPLEXITY_LEVELS,
        DIAGNOSTIC_TOOLS,
        ENVIRONMENT_REVISION,
        VERIFIER_REVISION,
        EnvironmentSnapshot,
        RepairTask,
        RepositoryRepairEnvironment,
        diagnostic_actions,
        encode_action,
        make_task,
        make_tasks,
        teacher_continuation_actions,
    )

SEED = 73
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
WORKLOAD_REVISION = "runpod-repository-repair-grpo@1"
DEPENDENCIES = (
    "transformers==5.14.1",
    "peft==0.19.1",
    "accelerate==1.14.0",
)
MAXIMUM_COMPLEXITY_LEVEL = len(COMPLEXITY_LEVELS) - 1
PREFIX_ACCEPTED_ACTIONS = 2
PREFIX_MAX_ATTEMPTS = 5
EVALUATION_INTERVAL = 5
EVALUATION_EXAMPLES = 4
EVALUATION_SEED_BASE = 40_000
TRAINING_TASKS_PER_UPDATE = 1
REPLAY_TASKS_PER_LEVEL = 1
MAX_UPDATES = 80
MAX_INPUT_TOKENS = 4_096
MAX_NEW_TOKENS = 192
LEARNING_RATE = 8e-5
TEACHER_LOSS_WEIGHT = 0.35
TRAINING_MICROBATCH_SIZE = 2
MASTERY_THRESHOLD = 0.50
MASTERY_WINDOWS = 1
PROGRESS_PATH = os.environ.get("EQUINOX_PROGRESS_PATH")


@dataclass(frozen=True)
class GeneratedAction:
    response: str
    input_ids: tuple[int, ...] = ()
    attention_mask: tuple[int, ...] = ()
    completion_mask: tuple[int, ...] = ()


@dataclass(frozen=True)
class WeightedAction:
    generated: GeneratedAction
    weight: float


@dataclass(frozen=True)
class TeacherExample:
    prompt: str
    completion: str


@dataclass
class BranchCollection:
    task: RepairTask
    snapshot: EnvironmentSnapshot | None
    prefix: RepositoryRepairEnvironment
    siblings: list[RepositoryRepairEnvironment]
    generated_by_sibling: list[list[GeneratedAction]]
    sampling_seeds: list[int]
    returns: list[float]
    advantages: list[float]
    exclusion_reason: str | None
    replay: bool

    @property
    def solved_siblings(self) -> int:
        return sum(
            sibling.terminal and sibling.terminal_reason == "solved" for sibling in self.siblings
        )

    @property
    def informative(self) -> bool:
        return any(abs(advantage) > 1e-8 for advantage in self.advantages)


SampleOne = Callable[[str, bool, int], GeneratedAction]


def emit_progress(phase: str, message: str, **values: Any) -> None:
    if not PROGRESS_PATH:
        return
    payload = {
        "schema_version": 2,
        "phase": phase,
        "message": message,
        "branch_width": BRANCH_WIDTH,
        "complexity_strategy": "adaptive",
        "maximum_level": MAXIMUM_COMPLEXITY_LEVEL,
        "maximum_updates": MAX_UPDATES,
        "multi_step": True,
        "restored_continuations": True,
        **values,
    }
    temporary_path = f"{PROGRESS_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary_path, PROGRESS_PATH)


def sibling_advantages(returns: list[float]) -> list[float]:
    if len(returns) != BRANCH_WIDTH:
        raise ValueError(f"expected {BRANCH_WIDTH} sibling returns")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("sibling returns must be finite")
    pooled_standard_deviation = statistics.pstdev(returns)
    if pooled_standard_deviation <= 1e-8:
        return [0.0] * BRANCH_WIDTH
    total = sum(returns)
    advantages = [
        (value - (total - value) / (BRANCH_WIDTH - 1)) / pooled_standard_deviation
        for value in returns
    ]
    centered = sum(advantages) / BRANCH_WIDTH
    return [round(value - centered, 8) for value in advantages]


def collect_branch_group(
    task: RepairTask,
    sample_one: SampleOne,
    *,
    stochastic: bool,
    sampling_seed: int,
    replay: bool = False,
) -> BranchCollection:
    prefix = RepositoryRepairEnvironment(task)
    accepted_diagnostics = 0
    for attempt in range(PREFIX_MAX_ATTEMPTS):
        generated = sample_one(
            prefix.policy_prompt("shared_prefix"),
            stochastic,
            sampling_seed + attempt,
        )
        step = prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if step.accepted:
            accepted_diagnostics += 1
        if accepted_diagnostics >= PREFIX_ACCEPTED_ACTIONS:
            break

    if accepted_diagnostics < PREFIX_ACCEPTED_ACTIONS:
        return BranchCollection(
            task=task,
            snapshot=None,
            prefix=prefix,
            siblings=[],
            generated_by_sibling=[],
            sampling_seeds=[],
            returns=[],
            advantages=[],
            exclusion_reason="PREFIX_CHECKPOINT_NOT_REACHED",
            replay=replay,
        )

    snapshot = prefix.capture_snapshot()
    siblings = [RepositoryRepairEnvironment.restore(task, snapshot) for _ in range(BRANCH_WIDTH)]
    generated_by_sibling: list[list[GeneratedAction]] = [[] for _ in range(BRANCH_WIDTH)]
    sampling_seeds = [sampling_seed + 10_000 * (index + 1) for index in range(BRANCH_WIDTH)]
    while not all(sibling.terminal for sibling in siblings):
        for sibling_index, sibling in enumerate(siblings):
            if sibling.terminal:
                continue
            action_index = len(generated_by_sibling[sibling_index])
            generated = sample_one(
                sibling.policy_prompt("continuation"),
                stochastic,
                sampling_seeds[sibling_index] + action_index,
            )
            generated_by_sibling[sibling_index].append(generated)
            sibling.step(generated.response)

    returns = [sibling.terminal_reward for sibling in siblings]
    return BranchCollection(
        task=task,
        snapshot=snapshot,
        prefix=prefix,
        siblings=siblings,
        generated_by_sibling=generated_by_sibling,
        sampling_seeds=sampling_seeds,
        returns=returns,
        advantages=sibling_advantages(returns),
        exclusion_reason=None,
        replay=replay,
    )


def collect_greedy_trajectory(
    task: RepairTask,
    sample_one: SampleOne,
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    prefix = RepositoryRepairEnvironment(task)
    accepted_diagnostics = 0
    for attempt in range(PREFIX_MAX_ATTEMPTS):
        generated = sample_one(
            prefix.policy_prompt("shared_prefix"),
            False,
            sampling_seed + attempt,
        )
        step = prefix.step(generated.response, allowed_tools=DIAGNOSTIC_TOOLS)
        if step.accepted:
            accepted_diagnostics += 1
        if accepted_diagnostics >= PREFIX_ACCEPTED_ACTIONS:
            break
    if accepted_diagnostics < PREFIX_ACCEPTED_ACTIONS:
        return {
            "solved": False,
            "checkpoint_reached": False,
            "actions": len(prefix.steps),
            "reward": 0.0,
        }

    continuation = RepositoryRepairEnvironment.restore(task, prefix.capture_snapshot())
    while not continuation.terminal:
        generated = sample_one(
            continuation.policy_prompt("continuation"),
            False,
            sampling_seed + 1_000 + len(continuation.steps),
        )
        continuation.step(generated.response)
    return {
        "solved": continuation.terminal_reason == "solved",
        "checkpoint_reached": True,
        "actions": len(continuation.steps),
        "reward": continuation.terminal_reward,
    }


def teacher_examples(collection: BranchCollection) -> list[TeacherExample]:
    if collection.snapshot is None:
        environment = RepositoryRepairEnvironment(collection.task)
        examples: list[TeacherExample] = []
        for action in diagnostic_actions(collection.task):
            examples.append(
                TeacherExample(
                    prompt=environment.policy_prompt("shared_prefix"),
                    completion=encode_action(action),
                )
            )
            environment.step(
                encode_action(action),
                allowed_tools=DIAGNOSTIC_TOOLS,
            )
    else:
        environment = RepositoryRepairEnvironment.restore(
            collection.task,
            collection.snapshot,
        )
        examples = []

    for action in teacher_continuation_actions(collection.task):
        examples.append(
            TeacherExample(
                prompt=environment.policy_prompt("continuation"),
                completion=encode_action(action),
            )
        )
        environment.step(encode_action(action))
    return examples


def policy_examples(collection: BranchCollection) -> list[WeightedAction]:
    if collection.exclusion_reason or not collection.informative:
        return []
    examples: list[WeightedAction] = []
    for sibling_index, generated_actions in enumerate(collection.generated_by_sibling):
        if not generated_actions:
            continue
        per_action_weight = collection.advantages[sibling_index] / len(generated_actions)
        examples.extend(
            WeightedAction(generated=generated, weight=per_action_weight)
            for generated in generated_actions
            if generated.input_ids
        )
    return examples


def serialize_step(step: Any) -> dict[str, Any]:
    value = asdict(step)
    value["step_id"] = f"step-{step.index}"
    return value


def serialize_branch_group(
    collection: BranchCollection,
    *,
    update: int,
) -> dict[str, Any]:
    snapshot_id = (
        collection.snapshot.snapshot_id
        if collection.snapshot is not None
        else f"unavailable-{collection.task.task_id}"
    )
    best_sibling_index = (
        max(
            range(len(collection.returns)),
            key=lambda index: (collection.returns[index], -index),
        )
        if collection.returns
        else None
    )
    return {
        "schema_version": 2,
        "snapshot_id": f"update-{update}-{snapshot_id}",
        "update": update,
        "level": collection.task.level,
        "domain": "micro_repository",
        "task_id": collection.task.task_id,
        "task": {
            "description": collection.task.description,
            "known_failing_tests": list(collection.task.failing_tests),
            "complexity": asdict(collection.task.complexity),
        },
        "checkpoint": (
            {
                "checkpoint_id": snapshot_id,
                "payload_digest": collection.snapshot.payload_digest,
                "fidelity": collection.snapshot.fidelity,
                "environment_revision": ENVIRONMENT_REVISION,
                "verifier_revision": VERIFIER_REVISION,
                "action_protocol_revision": ACTION_PROTOCOL_REVISION,
                "static_branch_width": BRANCH_WIDTH,
            }
            if collection.snapshot is not None
            else None
        ),
        "shared_prefix": {
            "policy_generated": True,
            "accepted_diagnostic_actions": sum(
                step.accepted and step.tool in DIAGNOSTIC_TOOLS for step in collection.prefix.steps
            ),
            "steps": [serialize_step(step) for step in collection.prefix.steps],
        },
        "best_sibling_index": best_sibling_index,
        "learning_signal": collection.informative,
        "teacher_fallback": collection.solved_siblings == 0,
        "excluded": collection.exclusion_reason is not None,
        "exclusion_reason": collection.exclusion_reason,
        "replay": collection.replay,
        "siblings": [
            {
                "index": index,
                "sampling_seed": collection.sampling_seeds[index],
                "return": collection.returns[index],
                "advantage": collection.advantages[index],
                "policy_signal": abs(collection.advantages[index]) > 1e-8,
                "passed": sibling.terminal_reason == "solved",
                "terminal_reason": sibling.terminal_reason,
                "trajectory_digest": sibling.trajectory_digest,
                "steps": [
                    serialize_step(step) for step in sibling.steps[len(collection.prefix.steps) :]
                ],
            }
            for index, sibling in enumerate(collection.siblings)
        ],
    }


def observation_mastered(observation: dict[str, Any]) -> bool:
    return (
        observation["exact_rate"] >= MASTERY_THRESHOLD
        and observation["checkpoint_rate"] >= MASTERY_THRESHOLD
    )


def self_test() -> dict[str, Any]:
    unequal = sibling_advantages([1.0, 0.0, 0.0, 0.0])
    if not unequal[0] > 0 or not all(value < 0 for value in unequal[1:]):
        raise AssertionError("leave-one-out sibling advantages have the wrong sign")
    if abs(sum(unequal)) > 1e-7:
        raise AssertionError("sibling advantages are not centered")
    if sibling_advantages([0.5] * BRANCH_WIDTH) != [0.0] * BRANCH_WIDTH:
        raise AssertionError("equal sibling returns must produce exact zero advantage")

    task = make_task(2, seed=SEED)
    continuation = teacher_continuation_actions(task)
    calls = 0

    def scripted_sample(_: str, __: bool, ___: int) -> GeneratedAction:
        nonlocal calls
        if calls == 0:
            action = diagnostic_actions(task)[0]
        elif calls == 1:
            action = diagnostic_actions(task)[1]
        else:
            continuation_index = (calls - 2) // BRANCH_WIDTH
            action = continuation[min(continuation_index, len(continuation) - 1)]
        calls += 1
        return GeneratedAction(response=encode_action(action))

    collection = collect_branch_group(
        task,
        scripted_sample,
        stochastic=False,
        sampling_seed=SEED,
    )
    if collection.snapshot is None or len(collection.siblings) != BRANCH_WIDTH:
        raise AssertionError("collector did not restore four siblings")
    if collection.solved_siblings != BRANCH_WIDTH:
        raise AssertionError("scripted sibling continuations did not solve")
    if collection.advantages != [0.0] * BRANCH_WIDTH:
        raise AssertionError("identical sibling outcomes produced a policy signal")
    serialized = serialize_branch_group(collection, update=1)
    if len(serialized["shared_prefix"]["steps"]) != PREFIX_ACCEPTED_ACTIONS or any(
        not sibling["steps"] for sibling in serialized["siblings"]
    ):
        raise AssertionError("serialized lineage omitted multi-step trajectory evidence")
    examples = teacher_examples(collection)
    if len(examples) != len(teacher_continuation_actions(task)):
        raise AssertionError("teacher continuation does not start at the restored checkpoint")

    result = {
        "self_test_passed": True,
        "workload_revision": WORKLOAD_REVISION,
        "branch_width": BRANCH_WIDTH,
        "shared_prefix_actions": PREFIX_ACCEPTED_ACTIONS,
        "sibling_steps": [len(sibling.steps) for sibling in collection.siblings],
        "environment_revision": ENVIRONMENT_REVISION,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def ensure_dependencies() -> None:
    try:
        importlib.import_module("transformers")
        importlib.import_module("peft")
        return
    except ImportError:
        pass
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *DEPENDENCIES],
        check=True,
        stdout=sys.stderr,
    )


def run_experiment() -> None:
    started = time.monotonic()
    emit_progress("dependency_setup", "Preparing model runtime.", elapsed_seconds=0)
    ensure_dependencies()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for repository repair post-training")

    target_seconds = int(os.environ.get("EQUINOX_RL_TARGET_SECONDS", "2700"))
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    emit_progress(
        "model_loading",
        "Loading model and LoRA adapter.",
        elapsed_seconds=round(time.monotonic() - started, 3),
        gpu_name=torch.cuda.get_device_name(0),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    def render_prompt(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def sample_one(prompt: str, stochastic: bool, sampling_seed: int) -> GeneratedAction:
        rendered = render_prompt(prompt)
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)
        input_width = encoded.input_ids.shape[1]
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        model.eval()
        model.config.use_cache = True
        generation_options: dict[str, Any] = {
            "do_sample": stochastic,
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if stochastic:
            generation_options.update(temperature=0.8, top_p=0.95)
        with torch.no_grad():
            sequence = model.generate(**encoded, **generation_options)
        model.config.use_cache = False
        continuation = sequence[0, input_width:]
        response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        eos_hits = continuation.eq(tokenizer.eos_token_id)
        eos_count = eos_hits.cumsum(dim=0)
        continuation_mask = ((eos_count == 0) | (eos_hits & eos_count.eq(1))).long()
        completion_mask = torch.cat(
            (
                torch.zeros(input_width, device=device, dtype=torch.long),
                continuation_mask,
            )
        )
        attention_mask = torch.cat((encoded.attention_mask[0], continuation_mask))
        generated = GeneratedAction(
            response=response,
            input_ids=tuple(int(value) for value in sequence[0].detach().cpu().tolist()),
            attention_mask=tuple(int(value) for value in attention_mask.detach().cpu().tolist()),
            completion_mask=tuple(int(value) for value in completion_mask.detach().cpu().tolist()),
        )
        del encoded, sequence, continuation
        return generated

    def evaluate(level: int, count: int, seed: int) -> dict[str, Any]:
        outcomes = [
            collect_greedy_trajectory(
                task,
                sample_one,
                sampling_seed=seed + index * 101,
            )
            for index, task in enumerate(make_tasks(level, count, seed))
        ]
        return {
            "level": level,
            "examples": count,
            "exact_rate": round(
                sum(outcome["solved"] for outcome in outcomes) / len(outcomes),
                6,
            ),
            "checkpoint_rate": round(
                sum(outcome["checkpoint_reached"] for outcome in outcomes) / len(outcomes),
                6,
            ),
            "mean_reward": round(
                sum(outcome["reward"] for outcome in outcomes) / len(outcomes),
                6,
            ),
            "mean_actions": round(
                sum(outcome["actions"] for outcome in outcomes) / len(outcomes),
                6,
            ),
        }

    def train_policy(
        examples: list[WeightedAction],
        denominator: int,
    ) -> float:
        if not examples:
            return 0.0
        loss_value = 0.0
        for start in range(0, len(examples), TRAINING_MICROBATCH_SIZE):
            batch = examples[start : start + TRAINING_MICROBATCH_SIZE]
            width = max(len(example.generated.input_ids) for example in batch)
            input_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            completion_rows: list[list[int]] = []
            weights: list[float] = []
            for example in batch:
                padding = width - len(example.generated.input_ids)
                input_rows.append(
                    list(example.generated.input_ids) + [tokenizer.pad_token_id] * padding
                )
                attention_rows.append(list(example.generated.attention_mask) + [0] * padding)
                completion_rows.append(list(example.generated.completion_mask) + [0] * padding)
                weights.append(example.weight)
            input_ids = torch.tensor(input_rows, device=device)
            attention_mask = torch.tensor(attention_rows, device=device)
            completion_mask = torch.tensor(
                completion_rows,
                device=device,
                dtype=torch.float32,
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits[:, :-1].float()
            target_ids = input_ids[:, 1:]
            target_mask = completion_mask[:, 1:]
            token_log_probabilities = (
                torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            )
            sequence_log_probability = (token_log_probabilities * target_mask).sum(
                dim=1
            ) / target_mask.sum(dim=1).clamp_min(1.0)
            weight_tensor = torch.tensor(weights, device=device)
            loss = -(weight_tensor.detach() * sequence_log_probability).sum() / denominator
            loss.backward()
            loss_value += float(loss.detach().item())
            del output, logits, token_log_probabilities
        return loss_value

    def train_teacher(examples: list[TeacherExample]) -> float:
        if not examples:
            return 0.0
        loss_value = 0.0
        for example in examples:
            prompt_ids = tokenizer(
                render_prompt(example.prompt),
                add_special_tokens=False,
            )["input_ids"]
            target_ids = tokenizer(
                example.completion + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            prompt_ids = prompt_ids[-(MAX_INPUT_TOKENS - len(target_ids)) :]
            input_ids = torch.tensor([prompt_ids + target_ids], device=device)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(
                [[-100] * len(prompt_ids) + target_ids],
                device=device,
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = output.loss.float() * TEACHER_LOSS_WEIGHT / len(examples)
            loss.backward()
            loss_value += float(loss.detach().item())
            del output
        return loss_value

    emit_progress(
        "baseline_evaluation",
        "Evaluating held-out multi-step tasks.",
        elapsed_seconds=round(time.monotonic() - started, 3),
        current_level=0,
    )
    initial_by_level = {
        str(level): evaluate(
            level,
            EVALUATION_EXAMPLES,
            EVALUATION_SEED_BASE + level * 1_000,
        )
        for level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    }
    initial_reward = statistics.mean(
        observation["exact_rate"] for observation in initial_by_level.values()
    )
    level = 0
    history: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    branch_snapshots: list[dict[str, Any]] = []
    mastery_streak = 0
    updates_completed = 0
    optimizer_update_count = 0
    policy_update_count = 0
    teacher_update_count = 0
    total_task_groups = 0
    replay_task_groups = 0
    informative_task_groups = 0
    excluded_task_groups = 0
    teacher_fallback_groups = 0
    total_sampled_actions = 0
    total_post_branch_actions = 0
    stop_reason = "maximum_updates"
    last_observation = initial_by_level["0"]

    for update in range(1, MAX_UPDATES + 1):
        if time.monotonic() - started >= target_seconds:
            stop_reason = "target_runtime"
            break
        current_tasks = make_tasks(
            level,
            TRAINING_TASKS_PER_UPDATE,
            SEED * 100_000 + update * 17,
        )
        replay_tasks = [
            task
            for replay_level in range(level)
            for task in make_tasks(
                replay_level,
                REPLAY_TASKS_PER_LEVEL,
                SEED * 1_000_000 + update * 101 + replay_level,
            )
        ]
        task_specs = [(task, False) for task in current_tasks] + [
            (task, True) for task in replay_tasks
        ]
        random.Random(SEED + update).shuffle(task_specs)
        collections = [
            collect_branch_group(
                task,
                sample_one,
                stochastic=True,
                sampling_seed=SEED * 10_000_000 + update * 100_000 + index * 1_000,
                replay=replay,
            )
            for index, (task, replay) in enumerate(task_specs)
        ]
        policy_training_examples = [
            example for collection in collections for example in policy_examples(collection)
        ]
        teacher_collections = [
            collection
            for collection in collections
            if collection.exclusion_reason is not None or collection.solved_siblings == 0
        ]
        supervised_examples = [
            example
            for collection in teacher_collections
            for example in teacher_examples(collection)
        ]
        informative_collections = sum(collection.informative for collection in collections)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        policy_loss = train_policy(
            policy_training_examples,
            max(1, informative_collections * BRANCH_WIDTH),
        )
        teacher_loss = train_teacher(supervised_examples)
        if policy_training_examples or supervised_examples:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()
            optimizer_update_count += 1
        else:
            gradient_norm = 0.0
        if policy_training_examples:
            policy_update_count += 1
        if supervised_examples:
            teacher_update_count += 1

        updates_completed = update
        total_task_groups += len(collections)
        replay_task_groups += len(replay_tasks)
        informative_task_groups += informative_collections
        excluded_task_groups += sum(
            collection.exclusion_reason is not None for collection in collections
        )
        teacher_fallback_groups += len(teacher_collections)
        total_sampled_actions += sum(
            len(collection.prefix.steps)
            + sum(
                len(sibling.steps) - len(collection.prefix.steps) for sibling in collection.siblings
            )
            for collection in collections
        )
        total_post_branch_actions += sum(
            sum(len(actions) for actions in collection.generated_by_sibling)
            for collection in collections
        )
        latest_snapshot = serialize_branch_group(collections[0], update=update)
        branch_snapshots.append(latest_snapshot)
        branch_snapshots = branch_snapshots[-24:]

        emit_progress(
            "training",
            "Collecting restored continuations and updating the adapter.",
            update=update,
            current_level=level,
            promotion_count=len(promotions),
            exact_rate=last_observation["exact_rate"],
            checkpoint_rate=last_observation["checkpoint_rate"],
            informative_group_rate=round(
                informative_task_groups / total_task_groups,
                6,
            ),
            teacher_fallback_rate=round(
                teacher_fallback_groups / total_task_groups,
                6,
            ),
            total_sampled_actions=total_sampled_actions,
            policy_update_count=policy_update_count,
            teacher_update_count=teacher_update_count,
            latest_branch_snapshot=latest_snapshot,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

        if update % EVALUATION_INTERVAL == 0:
            observation = evaluate(
                level,
                EVALUATION_EXAMPLES,
                EVALUATION_SEED_BASE + level * 1_000,
            )
            mastered = observation_mastered(observation)
            mastery_streak = mastery_streak + 1 if mastered else 0
            history.append(
                {
                    "update": update,
                    **observation,
                    "policy_loss": round(policy_loss, 6),
                    "teacher_loss": round(teacher_loss, 6),
                    "gradient_norm": round(float(gradient_norm), 6),
                    "informative_group_rate": round(
                        informative_task_groups / total_task_groups,
                        6,
                    ),
                    "teacher_fallback_rate": round(
                        teacher_fallback_groups / total_task_groups,
                        6,
                    ),
                    "mastery_streak": mastery_streak,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            last_observation = observation
            if mastery_streak >= MASTERY_WINDOWS and level < MAXIMUM_COMPLEXITY_LEVEL:
                promotions.append(
                    {
                        "update": update,
                        "from_level": level,
                        "to_level": level + 1,
                        "exact_rate": observation["exact_rate"],
                        "checkpoint_rate": observation["checkpoint_rate"],
                        "mastery_windows": mastery_streak,
                    }
                )
                level += 1
                mastery_streak = 0
            elif level == MAXIMUM_COMPLEXITY_LEVEL and mastery_streak >= MASTERY_WINDOWS:
                stop_reason = "maximum_level_mastered"
                break

    emit_progress(
        "finalizing",
        "Evaluating retained levels.",
        update=updates_completed,
        current_level=level,
        promotion_count=len(promotions),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    final_by_level = {
        str(candidate_level): evaluate(
            candidate_level,
            EVALUATION_EXAMPLES,
            EVALUATION_SEED_BASE + candidate_level * 1_000,
        )
        for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    }
    final_reward = statistics.mean(
        observation["exact_rate"] for observation in final_by_level.values()
    )
    reward_gain = final_reward - initial_reward
    retention_passed = all(
        observation_mastered(final_by_level[str(candidate_level)])
        for candidate_level in range(level + 1)
    )
    restored_branching_observed = any(
        snapshot["checkpoint"] is not None
        and len(snapshot["siblings"]) == BRANCH_WIDTH
        and all(sibling["steps"] for sibling in snapshot["siblings"])
        for snapshot in branch_snapshots
    )
    hypothesis_passed = (
        restored_branching_observed
        and informative_task_groups > 0
        and policy_update_count > 0
        and retention_passed
        and reward_gain > 0
    )
    adapter_path = os.environ.get("EQUINOX_ADAPTER_PATH")
    if adapter_path:
        model.save_pretrained(adapter_path, safe_serialization=True)
    result = {
        "schema_version": 2,
        "experiment_completed": True,
        "post_training_completed": True,
        "hypothesis_passed": hypothesis_passed,
        "workload": "repository-repair-restored-continuation-post-training",
        "workload_revision": WORKLOAD_REVISION,
        "algorithm": "shared-prefix-sibling-relative-policy-optimization",
        "branch_width": BRANCH_WIDTH,
        "static_branch_width": BRANCH_WIDTH,
        "multi_step": True,
        "restored_continuations": True,
        "prefix_gradient": False,
        "complexity_strategy": "adaptive",
        "replay_enabled": True,
        "task_domains": ["micro_repository"],
        "environment_revision": ENVIRONMENT_REVISION,
        "verifier_revision": VERIFIER_REVISION,
        "action_protocol_revision": ACTION_PROTOCOL_REVISION,
        "snapshot_fidelity": "logical_restore",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "seed": SEED,
        "mastery_threshold": MASTERY_THRESHOLD,
        "mastery_windows": MASTERY_WINDOWS,
        "maximum_complexity_level": MAXIMUM_COMPLEXITY_LEVEL,
        "complexity_levels": [asdict(item) for item in COMPLEXITY_LEVELS],
        "evaluation_examples": EVALUATION_EXAMPLES,
        "training_tasks_per_update": TRAINING_TASKS_PER_UPDATE,
        "replay_tasks_per_level": REPLAY_TASKS_PER_LEVEL,
        "reached_complexity_level": level,
        "promotion_count": len(promotions),
        "promotions": promotions,
        "initial_reward": round(initial_reward, 6),
        "final_reward": round(final_reward, 6),
        "reward_gain": round(reward_gain, 6),
        "initial_by_level": initial_by_level,
        "final_by_level": final_by_level,
        "history": history,
        "branch_snapshots": branch_snapshots,
        "updates_completed": updates_completed,
        "optimizer_update_count": optimizer_update_count,
        "policy_update_count": policy_update_count,
        "teacher_update_count": teacher_update_count,
        "total_task_groups": total_task_groups,
        "replay_task_groups": replay_task_groups,
        "informative_task_groups": informative_task_groups,
        "excluded_task_groups": excluded_task_groups,
        "teacher_fallback_groups": teacher_fallback_groups,
        "informative_group_rate": round(
            informative_task_groups / max(1, total_task_groups),
            6,
        ),
        "teacher_fallback_rate": round(
            teacher_fallback_groups / max(1, total_task_groups),
            6,
        ),
        "teacher_loss_weight": TEACHER_LOSS_WEIGHT,
        "total_sampled_actions": total_sampled_actions,
        "total_post_branch_actions": total_post_branch_actions,
        "stop_reason": stop_reason,
        "target_runtime_seconds": target_seconds,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "adapter_persisted": bool(adapter_path),
        "retention_passed": retention_passed,
        "restored_branching_observed": restored_branching_observed,
    }
    reached = final_by_level[str(level)]
    emit_progress(
        "finalizing",
        "Result ready for persistence and teardown.",
        update=updates_completed,
        current_level=level,
        promotion_count=len(promotions),
        exact_rate=reached["exact_rate"],
        checkpoint_rate=reached["checkpoint_rate"],
        informative_group_rate=result["informative_group_rate"],
        teacher_fallback_rate=result["teacher_fallback_rate"],
        total_sampled_actions=total_sampled_actions,
        policy_update_count=policy_update_count,
        teacher_update_count=teacher_update_count,
        hypothesis_passed=hypothesis_passed,
        elapsed_seconds=result["elapsed_seconds"],
        stop_reason=stop_reason,
        latest_branch_snapshot=branch_snapshots[-1] if branch_snapshots else None,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    try:
        run_experiment()
    except Exception as exc:
        emit_progress(
            "failed",
            "Repository repair workload failed.",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
