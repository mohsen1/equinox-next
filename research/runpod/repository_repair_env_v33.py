"""Revision-33 terminal semantics for the repository-repair simulator.

Revision 33 builds on revision 32 without changing task construction, hidden
verification, reward calculation, action validation, or snapshot fidelity. An
accepted passing test now completes the task without a redundant finish action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Literal

try:
    from research.runpod import repository_repair_env as frozen_environment
    from research.runpod import repository_repair_env_v32 as revision32
except ModuleNotFoundError:
    import repository_repair_env as frozen_environment  # type: ignore[no-redef]
    import repository_repair_env_v32 as revision32  # type: ignore[no-redef]


ENVIRONMENT_REVISION = "repository-repair-simulator@9"
ACTION_PROTOCOL_REVISION = "repository-repair-json-tools@8"
_PASSING_TEST_RULE = (
    "- An accepted test whose verifier passes finishes immediately; "
    "do not send a redundant finish action."
)
SYSTEM_PROMPT = revision32.SYSTEM_PROMPT.replace(
    "- Do not repeat an accepted diagnostic when it would return unchanged evidence; "
    "the interface rejects no-progress repeats.\n",
    "- Do not repeat an accepted diagnostic when it would return unchanged evidence; "
    "the interface rejects no-progress repeats.\n"
    f"{_PASSING_TEST_RULE}\n",
)
ACTION_REMINDER = revision32.ACTION_REMINDER.replace(
    "Do not repeat unchanged evidence. Reread after a rejected edit.\n",
    "Do not repeat unchanged evidence. Reread after a rejected edit.\n"
    "An accepted passing test finishes immediately; do not send a redundant finish action.\n",
)


class RepositoryRepairEnvironment(revision32.RepositoryRepairEnvironment):
    """Complete the trajectory as soon as an accepted test passes."""

    def _state_material(self, *, include_steps: bool) -> dict[str, Any]:
        material = super()._state_material(include_steps=include_steps)
        material["environment_revision"] = ENVIRONMENT_REVISION
        return material

    @classmethod
    def restore(
        cls,
        task: Any,
        snapshot: Any,
    ) -> RepositoryRepairEnvironment:
        observed_digest = "sha256:" + hashlib.sha256(snapshot.payload.encode()).hexdigest()
        if observed_digest != snapshot.payload_digest:
            raise ValueError("snapshot payload digest mismatch")
        material = json.loads(snapshot.payload)
        if snapshot.task_id != task.task_id or material.get("task_id") != task.task_id:
            raise ValueError("snapshot task mismatch")
        if material.get("environment_revision") != ENVIRONMENT_REVISION:
            raise ValueError("snapshot environment revision mismatch")
        environment = cls(task)
        environment.files = dict(material["files"])
        environment.terminal = bool(material["terminal"])
        environment.terminal_reason = material["terminal_reason"]
        environment.terminal_reward = float(material["terminal_reward"])
        environment.steps = [frozen_environment.StepResult(**step) for step in material["steps"]]
        if environment.capture_snapshot().payload_digest != snapshot.payload_digest:
            raise ValueError("snapshot did not restore exactly")
        return environment

    def policy_prompt(self, phase: Literal["shared_prefix", "continuation"]) -> str:
        prompt = super().policy_prompt(phase)
        if not prompt.endswith(revision32.ACTION_REMINDER):
            raise RuntimeError("the revision-32 environment prompt contract changed")
        return prompt[: -len(revision32.ACTION_REMINDER)] + ACTION_REMINDER

    def step(
        self,
        response: str,
        *,
        allowed_tools: frozenset[str] | None = None,
    ) -> frozen_environment.StepResult:
        if self.terminal:
            raise RuntimeError("cannot act after terminal state")
        state_before = self.state_digest
        action = frozen_environment.parse_action(response)
        accepted = False
        observation = "Invalid action. Return exactly one allowed JSON object."
        tool: str | None = None
        if action is not None:
            tool = action["tool"]
            if allowed_tools is not None and tool not in allowed_tools:
                observation = f"Action {tool} is not allowed in this trajectory phase."
            else:
                accepted, observation = self._execute(action)

        fixed, failing = self._test_results()
        if accepted and tool == "test" and not failing:
            self.terminal = True
            self.terminal_reason = "solved"
        elif accepted and tool == "finish":
            self.terminal = True
            self.terminal_reason = "solved" if not failing else "finished_with_failures"
        elif len(self.steps) + 1 >= self.task.complexity.repair_horizon:
            self.terminal = True
            self.terminal_reason = "horizon_exhausted"

        provisional = frozen_environment.StepResult(
            index=len(self.steps),
            tool=tool,
            action=action,
            accepted=accepted,
            observation=frozen_environment._bounded(observation),
            state_digest_before=state_before,
            state_digest_after="",
            verifier_passed=not failing,
            fixed_faults=fixed,
            total_faults=len(self.task.faults),
            terminal=self.terminal,
            terminal_reason=self.terminal_reason,
            reward=0.0,
        )
        self.steps.append(provisional)
        if self.terminal:
            self.terminal_reward = self._reward()
        completed = frozen_environment.StepResult(
            **{
                **asdict(provisional),
                "state_digest_after": self.state_digest,
                "reward": self.terminal_reward if self.terminal else 0.0,
            }
        )
        self.steps[-1] = completed
        return completed


__all__ = [
    "ACTION_PROTOCOL_REVISION",
    "ACTION_REMINDER",
    "ENVIRONMENT_REVISION",
    "SYSTEM_PROMPT",
    "RepositoryRepairEnvironment",
]
