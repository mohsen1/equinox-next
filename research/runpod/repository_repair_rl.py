"""Restored-continuation REINFORCE post-training for repository repair.

This workload collects one policy-generated diagnostic prefix, snapshots the
repository and transcript, restores four continuations, and applies
sibling-relative credit only to post-snapshot action tokens. Candidate content
is verified as inert simulator state and is never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import random
import shutil
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

DEFAULT_SEED = 73
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
WORKLOAD_REVISION = "runpod-repository-repair-loo-reinforce@2"
OBJECTIVE_ID = "leave-one-out-group-normalized-reinforce@1"
DEPENDENCIES = (
    "transformers==5.14.1",
    "peft==0.19.1",
    "accelerate==1.14.0",
)
DEPENDENCY_VERSIONS = {
    requirement.split("==", 1)[0]: requirement.split("==", 1)[1] for requirement in DEPENDENCIES
}
MAXIMUM_COMPLEXITY_LEVEL = len(COMPLEXITY_LEVELS) - 1
PREFIX_ACCEPTED_ACTIONS = 2
PREFIX_MAX_ATTEMPTS = 5
EVALUATION_INTERVAL = 5
EVALUATION_EXAMPLES = 4
VALIDATION_SEED_BASE = 40_000
TEST_SEED_BASE = 90_000
TRAINING_TASKS_PER_UPDATE = 1
REPLAY_TASKS_PER_LEVEL = 1
MAX_UPDATES = 80
MAX_INPUT_TOKENS = 4_096
MAX_NEW_TOKENS = 192
LEARNING_RATE = 8e-5
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

    task = make_task(2, seed=DEFAULT_SEED)
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
        sampling_seed=DEFAULT_SEED,
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
    def versions_match() -> bool:
        try:
            return all(
                importlib.metadata.version(package) == version
                for package, version in DEPENDENCY_VERSIONS.items()
            )
        except importlib.metadata.PackageNotFoundError:
            return False

    if versions_match():
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *DEPENDENCIES],
        check=True,
        stdout=sys.stderr,
    )
    importlib.invalidate_caches()
    if not versions_match():
        raise RuntimeError("pinned research dependency versions were not installed exactly")


def run_experiment() -> None:
    started = time.monotonic()
    emit_progress("dependency_setup", "Preparing model runtime.", elapsed_seconds=0)
    ensure_dependencies()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for repository repair post-training")

    target_seconds = int(os.environ.get("EQUINOX_RL_TARGET_SECONDS", "2700"))
    experiment_seed = int(os.environ.get("EQUINOX_RL_SEED", str(DEFAULT_SEED)))
    if not 0 <= experiment_seed <= 2**31 - 1:
        raise ValueError("EQUINOX_RL_SEED must be between 0 and 2^31 - 1")
    random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    adapter_path = os.environ.get("EQUINOX_ADAPTER_PATH")
    checkpoints_root = os.path.join(adapter_path, "checkpoints") if adapter_path else None
    latest_checkpoint_path = (
        os.path.join(checkpoints_root, "latest.json") if checkpoints_root else None
    )
    checkpoint_directory: str | None = None
    if latest_checkpoint_path and os.path.isfile(latest_checkpoint_path):
        with open(latest_checkpoint_path, encoding="utf-8") as handle:
            checkpoint_metadata = json.load(handle)
        checkpoint_name = checkpoint_metadata.get("checkpoint")
        if (
            not isinstance(checkpoint_name, str)
            or not checkpoint_name.startswith("update-")
            or "/" in checkpoint_name
        ):
            raise RuntimeError("training checkpoint pointer is invalid")
        checkpoint_directory = os.path.join(checkpoints_root, checkpoint_name)
    if checkpoint_directory:
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_directory,
            is_trainable=True,
        )
    else:
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
    resume_state: dict[str, Any] | None = None
    if checkpoint_directory:
        resume_state = torch.load(
            os.path.join(checkpoint_directory, "training-state.pt"),
            map_location=device,
            weights_only=False,
        )
        if (
            resume_state.get("seed") != experiment_seed
            or resume_state.get("workload_revision") != WORKLOAD_REVISION
            or resume_state.get("model_revision") != MODEL_REVISION
            or resume_state.get("objective_id") != OBJECTIVE_ID
        ):
            raise RuntimeError("training checkpoint identity does not match this workload")
        optimizer.load_state_dict(resume_state["optimizer"])
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

    def evaluate(
        level: int,
        count: int,
        seed: int,
        *,
        split: str,
    ) -> dict[str, Any]:
        outcomes = [
            collect_greedy_trajectory(
                task,
                sample_one,
                sampling_seed=seed + index * 101,
            )
            for index, task in enumerate(make_tasks(level, count, seed, split=split))
        ]
        successes = sum(outcome["solved"] for outcome in outcomes)
        exact_rate = successes / len(outcomes)
        interval_radius = 1.96 * math.sqrt(max(exact_rate * (1 - exact_rate), 0.0) / len(outcomes))
        return {
            "level": level,
            "split": split,
            "examples": count,
            "exact_successes": successes,
            "exact_rate": round(exact_rate, 6),
            "exact_rate_95ci": [
                round(max(0.0, exact_rate - interval_radius), 6),
                round(min(1.0, exact_rate + interval_radius), 6),
            ],
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
            sequence_log_probability = (token_log_probabilities * target_mask).sum(dim=1)
            weight_tensor = torch.tensor(weights, device=device)
            loss = -(weight_tensor.detach() * sequence_log_probability).sum() / denominator
            loss.backward()
            loss_value += float(loss.detach().item())
            del output, logits, token_log_probabilities
        return loss_value

    emit_progress(
        "baseline_evaluation",
        "Evaluating the validation split.",
        elapsed_seconds=round(time.monotonic() - started, 3),
        current_level=0,
    )
    validation_baseline_by_level = (
        resume_state["validation_baseline_by_level"]
        if resume_state
        else {
            str(level): evaluate(
                level,
                EVALUATION_EXAMPLES,
                VALIDATION_SEED_BASE + level * 1_000,
                split="validation",
            )
            for level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
        }
    )
    level = int(resume_state["level"]) if resume_state else 0
    history = list(resume_state["history"]) if resume_state else []
    promotions = list(resume_state["promotions"]) if resume_state else []
    branch_snapshots = list(resume_state["branch_snapshots"]) if resume_state else []
    mastery_streak = int(resume_state["mastery_streak"]) if resume_state else 0
    updates_completed = int(resume_state["updates_completed"]) if resume_state else 0
    optimizer_update_count = int(resume_state["optimizer_update_count"]) if resume_state else 0
    policy_update_count = int(resume_state["policy_update_count"]) if resume_state else 0
    total_task_groups = int(resume_state["total_task_groups"]) if resume_state else 0
    replay_task_groups = int(resume_state["replay_task_groups"]) if resume_state else 0
    informative_task_groups = int(resume_state["informative_task_groups"]) if resume_state else 0
    excluded_task_groups = int(resume_state["excluded_task_groups"]) if resume_state else 0
    total_sampled_actions = int(resume_state["total_sampled_actions"]) if resume_state else 0
    total_post_branch_actions = (
        int(resume_state["total_post_branch_actions"]) if resume_state else 0
    )
    stop_reason = "maximum_updates"
    last_observation = (
        resume_state["last_observation"] if resume_state else validation_baseline_by_level["0"]
    )
    if resume_state:
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
    first_update = updates_completed + 1

    def persist_training_checkpoint(update: int) -> None:
        if not checkpoints_root or not latest_checkpoint_path:
            return
        os.makedirs(checkpoints_root, exist_ok=True)
        checkpoint_name = f"update-{update:04d}"
        target = os.path.join(checkpoints_root, checkpoint_name)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)
        model.save_pretrained(target, safe_serialization=True)
        state = {
            "schema_version": 1,
            "workload_revision": WORKLOAD_REVISION,
            "model_revision": MODEL_REVISION,
            "objective_id": OBJECTIVE_ID,
            "seed": experiment_seed,
            "updates_completed": updates_completed,
            "level": level,
            "history": history,
            "promotions": promotions,
            "branch_snapshots": branch_snapshots,
            "mastery_streak": mastery_streak,
            "optimizer_update_count": optimizer_update_count,
            "policy_update_count": policy_update_count,
            "total_task_groups": total_task_groups,
            "replay_task_groups": replay_task_groups,
            "informative_task_groups": informative_task_groups,
            "excluded_task_groups": excluded_task_groups,
            "total_sampled_actions": total_sampled_actions,
            "total_post_branch_actions": total_post_branch_actions,
            "last_observation": last_observation,
            "validation_baseline_by_level": validation_baseline_by_level,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        temporary_state = os.path.join(target, "training-state.pt.pending")
        torch.save(state, temporary_state)
        os.replace(temporary_state, os.path.join(target, "training-state.pt"))
        previous = None
        if os.path.isfile(latest_checkpoint_path):
            with open(latest_checkpoint_path, encoding="utf-8") as handle:
                previous = json.load(handle).get("checkpoint")
        temporary_pointer = latest_checkpoint_path + ".pending"
        with open(temporary_pointer, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": 1, "checkpoint": checkpoint_name},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
        os.replace(temporary_pointer, latest_checkpoint_path)
        if (
            isinstance(previous, str)
            and previous != checkpoint_name
            and previous.startswith("update-")
            and "/" not in previous
        ):
            previous_path = os.path.join(checkpoints_root, previous)
            if os.path.isdir(previous_path):
                shutil.rmtree(previous_path)

    for update in range(first_update, MAX_UPDATES + 1):
        if time.monotonic() - started >= target_seconds:
            stop_reason = "target_runtime"
            break
        current_tasks = make_tasks(
            level,
            TRAINING_TASKS_PER_UPDATE,
            experiment_seed * 100_000 + update * 17,
            split="train",
        )
        replay_tasks = [
            task
            for replay_level in range(level)
            for task in make_tasks(
                replay_level,
                REPLAY_TASKS_PER_LEVEL,
                experiment_seed * 1_000_000 + update * 101 + replay_level,
                split="train",
            )
        ]
        task_specs = [(task, False) for task in current_tasks] + [
            (task, True) for task in replay_tasks
        ]
        random.Random(experiment_seed + update).shuffle(task_specs)
        collections = [
            collect_branch_group(
                task,
                sample_one,
                stochastic=True,
                sampling_seed=(experiment_seed * 10_000_000 + update * 100_000 + index * 1_000),
                replay=replay,
            )
            for index, (task, replay) in enumerate(task_specs)
        ]
        policy_training_examples = [
            example for collection in collections for example in policy_examples(collection)
        ]
        informative_collections = sum(collection.informative for collection in collections)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        policy_loss = train_policy(
            policy_training_examples,
            max(1, informative_collections * BRANCH_WIDTH),
        )
        if policy_training_examples:
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

        updates_completed = update
        total_task_groups += len(collections)
        replay_task_groups += len(replay_tasks)
        informative_task_groups += informative_collections
        excluded_task_groups += sum(
            collection.exclusion_reason is not None for collection in collections
        )
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
            total_sampled_actions=total_sampled_actions,
            policy_update_count=policy_update_count,
            latest_branch_snapshot=latest_snapshot,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

        stop_after_checkpoint = False
        if update % EVALUATION_INTERVAL == 0:
            observation = evaluate(
                level,
                EVALUATION_EXAMPLES,
                VALIDATION_SEED_BASE + level * 1_000,
                split="validation",
            )
            mastered = observation_mastered(observation)
            mastery_streak = mastery_streak + 1 if mastered else 0
            history.append(
                {
                    "update": update,
                    **observation,
                    "policy_loss": round(policy_loss, 6),
                    "gradient_norm": round(float(gradient_norm), 6),
                    "informative_group_rate": round(
                        informative_task_groups / total_task_groups,
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
                stop_after_checkpoint = True
        persist_training_checkpoint(update)
        if stop_after_checkpoint:
            break

    emit_progress(
        "finalizing",
        "Evaluating retained levels.",
        update=updates_completed,
        current_level=level,
        promotion_count=len(promotions),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    with model.disable_adapter():
        initial_by_level = {
            str(candidate_level): evaluate(
                candidate_level,
                EVALUATION_EXAMPLES,
                TEST_SEED_BASE + candidate_level * 1_000,
                split="test",
            )
            for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
        }
    final_by_level = {
        str(candidate_level): evaluate(
            candidate_level,
            EVALUATION_EXAMPLES,
            TEST_SEED_BASE + candidate_level * 1_000,
            split="test",
        )
        for candidate_level in range(MAXIMUM_COMPLEXITY_LEVEL + 1)
    }
    initial_reward = statistics.mean(
        observation["exact_rate"] for observation in initial_by_level.values()
    )
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
    adapter_manifest: dict[str, Any] | None = None
    if adapter_path:
        model.save_pretrained(adapter_path, safe_serialization=True)
        files = []
        for directory, _, names in os.walk(adapter_path):
            for name in sorted(names):
                path = os.path.join(directory, name)
                relative_path = os.path.relpath(path, adapter_path)
                if relative_path == "adapter-manifest.json":
                    continue
                with open(path, "rb") as handle:
                    payload = handle.read()
                files.append(
                    {
                        "path": relative_path,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        if not files or not any(item["size_bytes"] > 0 for item in files):
            raise RuntimeError("adapter persistence produced no durable bytes")
        adapter_manifest_content = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "objective_id": OBJECTIVE_ID,
            "files": files,
        }
        manifest_bytes = json.dumps(
            adapter_manifest_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        adapter_manifest = {
            **adapter_manifest_content,
            "digest": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        }
        manifest_path = os.path.join(adapter_path, "adapter-manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(adapter_manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        with open(manifest_path, encoding="utf-8") as handle:
            if json.load(handle) != adapter_manifest:
                raise RuntimeError("adapter manifest failed its read-after-write check")
    result = {
        "schema_version": 2,
        "experiment_completed": True,
        "post_training_completed": True,
        "hypothesis_passed": hypothesis_passed,
        "workload": "repository-repair-restored-continuation-post-training",
        "workload_revision": WORKLOAD_REVISION,
        "algorithm": "leave-one-out-group-normalized-reinforce",
        "objective_id": OBJECTIVE_ID,
        "objective_sequence_reduction": "sum_completion_token_log_probabilities",
        "teacher_data_used": False,
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
        "seed": experiment_seed,
        "determinism": {
            "cuda_seeded_all_devices": True,
            "deterministic_algorithms": "warn_on_unavailable_kernel",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
        },
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
        "validation_baseline_by_level": validation_baseline_by_level,
        "final_by_level": final_by_level,
        "history": history,
        "branch_snapshots": branch_snapshots,
        "updates_completed": updates_completed,
        "optimizer_update_count": optimizer_update_count,
        "policy_update_count": policy_update_count,
        "total_task_groups": total_task_groups,
        "replay_task_groups": replay_task_groups,
        "informative_task_groups": informative_task_groups,
        "excluded_task_groups": excluded_task_groups,
        "informative_group_rate": round(
            informative_task_groups / max(1, total_task_groups),
            6,
        ),
        "total_sampled_actions": total_sampled_actions,
        "total_post_branch_actions": total_post_branch_actions,
        "stop_reason": stop_reason,
        "target_runtime_seconds": target_seconds,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dependency_versions": {
            package: importlib.metadata.version(package) for package in sorted(DEPENDENCY_VERSIONS)
        },
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "adapter_persisted": adapter_manifest is not None,
        "adapter_manifest": adapter_manifest,
        "resumed_from_checkpoint": resume_state is not None,
        "training_state_checkpointed": bool(
            latest_checkpoint_path and os.path.isfile(latest_checkpoint_path)
        ),
        "optimization_seed_count": 1,
        "claim_strength": "EXPLORATORY_SINGLE_SEED",
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
        total_sampled_actions=total_sampled_actions,
        policy_update_count=policy_update_count,
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
