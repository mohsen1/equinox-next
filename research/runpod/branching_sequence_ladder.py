"""Bounded long-horizon GPU proof for the opt-in RunPod research bridge."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import torch

SEED = 29
BRANCH_WIDTH = 4
MASTERY_THRESHOLD = 0.82
EVALUATION_INTERVAL = 20


@dataclass(frozen=True)
class ScaleCase:
    name: str
    feature_count: int
    horizon_levels: tuple[int, ...]
    updates: int
    batch_size: int
    evaluation_episodes: int


SCALE_CASES = (
    ScaleCase("small", 16, (4, 8), 100, 256, 4_096),
    ScaleCase("medium", 32, (8, 16, 24), 160, 512, 8_192),
    ScaleCase("large", 64, (16, 32, 64), 240, 1_024, 16_384),
)


class StepPolicy(torch.nn.Module):
    """A position-aware binary policy for a deterministic repair plan."""

    def __init__(
        self,
        feature_count: int,
        maximum_horizon: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        initial_weights = torch.randn(
            maximum_horizon,
            feature_count,
            device=device,
            generator=generator,
        )
        self.weights = torch.nn.Parameter(initial_weights * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(maximum_horizon, device=device))

    def forward(self, contexts: torch.Tensor, horizon: int) -> torch.Tensor:
        scores = torch.einsum(
            "bf,hf->bh",
            contexts,
            self.weights[:horizon],
        )
        scores = scores + self.bias[:horizon]
        return torch.stack((-scores, scores), dim=-1)


def run_case(
    case: ScaleCase,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    maximum_horizon = max(case.horizon_levels)
    teacher = torch.randn(
        maximum_horizon,
        case.feature_count,
        device=device,
        generator=generator,
    )
    teacher = torch.nn.functional.normalize(teacher, dim=-1)
    policy = StepPolicy(
        case.feature_count,
        maximum_horizon,
        device=device,
        generator=generator,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    def sample_tasks(
        count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = torch.randn(
            count,
            case.feature_count,
            device=device,
            generator=generator,
        )
        target_scores = torch.einsum(
            "bf,hf->bh",
            contexts,
            teacher[:horizon],
        )
        return contexts, (target_scores >= 0).long()

    @torch.no_grad()
    def evaluate(horizon: int) -> tuple[float, float]:
        contexts, targets = sample_tasks(case.evaluation_episodes, horizon)
        actions = policy(contexts, horizon).argmax(dim=-1)
        correct = actions == targets
        step_reward = float(correct.float().mean().item())
        episode_success = float(correct.all(dim=-1).float().mean().item())
        return step_reward, episode_success

    initial_reward, initial_episode_success = evaluate(case.horizon_levels[0])
    level = 0
    action_decisions = 0
    trajectory_count = 0
    last_loss = math.nan
    curriculum_events: list[dict[str, object]] = []

    for update in range(1, case.updates + 1):
        horizon = case.horizon_levels[level]
        contexts, targets = sample_tasks(case.batch_size, horizon)
        distribution = torch.distributions.Categorical(logits=policy(contexts, horizon))
        actions = distribution.sample((BRANCH_WIDTH,))
        rewards = (actions == targets.unsqueeze(0)).float()
        baseline = rewards.mean(dim=0, keepdim=True)
        advantages = rewards - baseline
        log_probabilities = distribution.log_prob(actions)
        loss = -(log_probabilities * advantages.detach()).mean()
        loss -= 0.005 * distribution.entropy().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
        action_decisions += case.batch_size * BRANCH_WIDTH * horizon
        trajectory_count += case.batch_size * BRANCH_WIDTH

        if update % EVALUATION_INTERVAL == 0:
            accuracy, episode_success = evaluate(horizon)
            promoted = accuracy >= MASTERY_THRESHOLD and level < len(case.horizon_levels) - 1
            curriculum_events.append(
                {
                    "update": update,
                    "level": level,
                    "horizon": horizon,
                    "step_reward": round(accuracy, 6),
                    "episode_success": round(episode_success, 6),
                    "promoted": promoted,
                    "next_horizon": (case.horizon_levels[level + 1] if promoted else horizon),
                }
            )
            if promoted:
                level += 1

    final_horizon = case.horizon_levels[level]
    final_reward, final_episode_success = evaluate(final_horizon)
    return {
        "name": case.name,
        "feature_count": case.feature_count,
        "horizon_levels": list(case.horizon_levels),
        "initial_horizon": case.horizon_levels[0],
        "maximum_horizon": maximum_horizon,
        "final_horizon": final_horizon,
        "updates": case.updates,
        "episodes_per_update": case.batch_size,
        "evaluation_episodes": case.evaluation_episodes,
        "branch_width": BRANCH_WIDTH,
        "initial_reward": round(initial_reward, 6),
        "final_reward": round(final_reward, 6),
        "reward_gain": round(final_reward - initial_reward, 6),
        "initial_episode_success": round(initial_episode_success, 6),
        "final_episode_success": round(final_episode_success, 6),
        "final_level": level,
        "promotion_count": sum(1 for event in curriculum_events if event["promoted"]),
        "curriculum_events": curriculum_events,
        "trajectory_count": trajectory_count,
        "action_decisions": action_decisions,
        "last_loss": round(last_loss, 6),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RunPod research proof")

    started = time.monotonic()
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    test_cases = [
        run_case(case, device=device, seed=SEED + index * 101)
        for index, case in enumerate(SCALE_CASES)
    ]
    initial_reward = sum(float(case["initial_reward"]) for case in test_cases) / len(test_cases)
    final_reward = sum(float(case["final_reward"]) for case in test_cases) / len(test_cases)
    result = {
        "schema_version": 1,
        "workload": "branching-sequence-policy-gradient-ladder",
        "workload_revision": "runpod-branching-sequence@1",
        "algorithm": "REINFORCE",
        "branch_width": BRANCH_WIDTH,
        "complexity_strategy": "adaptive",
        "seed": SEED,
        "mastery_threshold": MASTERY_THRESHOLD,
        "test_cases": test_cases,
        "test_case_count": len(test_cases),
        "maximum_horizon": max(int(case["maximum_horizon"]) for case in test_cases),
        "initial_reward": round(initial_reward, 6),
        "final_reward": round(final_reward, 6),
        "reward_gain": round(final_reward - initial_reward, 6),
        "promotion_count": sum(int(case["promotion_count"]) for case in test_cases),
        "total_trajectories": sum(int(case["trajectory_count"]) for case in test_cases),
        "total_action_decisions": sum(int(case["action_decisions"]) for case in test_cases),
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
