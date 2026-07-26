"""Bounded GPU policy-gradient proof for the opt-in RunPod research bridge."""

from __future__ import annotations

import json
import math
import time

import torch

SEED = 17
BRANCH_WIDTH = 4
MAX_LEVEL = 2
UPDATES = 120
EVALUATION_INTERVAL = 20
MASTERY_THRESHOLD = 0.80
BATCH_SIZE = 256
EVALUATION_EPISODES = 4096
FEATURE_COUNT = 8


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RunPod research proof")

    started = time.monotonic()
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(SEED)
    teacher = torch.tensor(
        [1.8, -1.2, 0.9, -0.7, 0.55, -0.45, 0.35, -0.25],
        device=device,
    )
    policy = torch.nn.Linear(FEATURE_COUNT, 2, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.035)

    def sample_tasks(level: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        active_features = 2 + level * 2
        contexts = torch.randn(
            count,
            FEATURE_COUNT,
            generator=generator,
            device=device,
        )
        contexts[:, active_features:] = 0
        score = contexts @ teacher
        noise = torch.randn(count, generator=generator, device=device) * (0.08 + level * 0.06)
        targets = (score + noise >= 0).long()
        return contexts, targets

    @torch.no_grad()
    def evaluate(level: int) -> float:
        contexts, targets = sample_tasks(level, EVALUATION_EPISODES)
        actions = policy(contexts).argmax(dim=-1)
        return float((actions == targets).float().mean().item())

    initial_reward = evaluate(0)
    level = 0
    curriculum_events: list[dict[str, int | float | bool]] = []
    last_loss = math.nan

    for update in range(1, UPDATES + 1):
        contexts, targets = sample_tasks(level, BATCH_SIZE)
        distribution = torch.distributions.Categorical(logits=policy(contexts))
        actions = distribution.sample((BRANCH_WIDTH,))
        rewards = (actions == targets.unsqueeze(0)).float()
        baseline = rewards.mean(dim=0, keepdim=True)
        advantages = rewards - baseline
        log_probabilities = distribution.log_prob(actions)
        loss = -(log_probabilities * advantages.detach()).mean()
        loss -= 0.01 * distribution.entropy().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

        if update % EVALUATION_INTERVAL == 0:
            accuracy = evaluate(level)
            promoted = accuracy >= MASTERY_THRESHOLD and level < MAX_LEVEL
            curriculum_events.append(
                {
                    "update": update,
                    "level": level,
                    "accuracy": round(accuracy, 6),
                    "promoted": promoted,
                }
            )
            if promoted:
                level += 1

    final_reward_by_level = {
        str(candidate_level): round(evaluate(candidate_level), 6)
        for candidate_level in range(MAX_LEVEL + 1)
    }
    final_reward = final_reward_by_level[str(level)]
    result = {
        "schema_version": 1,
        "workload": "contextual-bandit-policy-gradient",
        "workload_revision": "runpod-contextual-bandit@1",
        "algorithm": "REINFORCE",
        "branch_width": BRANCH_WIDTH,
        "complexity_strategy": "adaptive",
        "seed": SEED,
        "updates": UPDATES,
        "episodes_per_update": BATCH_SIZE,
        "evaluation_episodes": EVALUATION_EPISODES,
        "mastery_threshold": MASTERY_THRESHOLD,
        "initial_reward": round(initial_reward, 6),
        "final_reward": final_reward,
        "reward_gain": round(final_reward - initial_reward, 6),
        "final_reward_by_level": final_reward_by_level,
        "final_level": level,
        "promotion_count": sum(1 for event in curriculum_events if event["promoted"]),
        "curriculum_events": curriculum_events,
        "last_loss": round(last_loss, 6),
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
