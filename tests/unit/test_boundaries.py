from pathlib import Path

import pytest
from pydantic import ValidationError

from services.orchestrator.app.main import (
    ResearchComputeExecutionRequest,
    ResearchComputeProofRequest,
    research_result_progress,
)
from services.orchestrator.app.providers import (
    JUDGE_PROVIDER_NAMES,
    POLICY_COMPUTE_PROVIDERS,
    assert_local_registry,
)


def test_local_provider_registry_has_no_real_provider() -> None:
    assert_local_registry()
    assert set(POLICY_COMPUTE_PROVIDERS) == {"MockRunPodProvider"}
    assert {"MockJudgeProvider"} == JUDGE_PROVIDER_NAMES


def test_prompts_treat_candidate_content_as_untrusted_and_disable_tools() -> None:
    prompt_root = Path("services/execution/prompts")
    text = "\n".join(path.read_text(encoding="utf-8") for path in prompt_root.glob("*.md"))
    lowered = text.lower()
    assert "untrusted" in lowered
    assert "never as instructions" in lowered
    assert "tools" in lowered
    assert "hidden" in lowered


def test_research_proof_requires_confirmed_teardown() -> None:
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            provider_name="RunPod",
            provider_handle="runpod://pods/test-pod",
            provider_cli_version="2.7.2",
            resource_profile={"gpu_id": "NVIDIA GeForce RTX 3090"},
            workload={"static_branch_width": 4, "complexity_strategy": "adaptive"},
            result={"reward_gain": 0.5},
            started_at="2026-07-26T16:00:00Z",
            completed_at="2026-07-26T16:01:00Z",
            teardown_confirmed=False,
        )


def test_research_execution_keeps_static_k_and_adaptive_complexity() -> None:
    request = ResearchComputeExecutionRequest(
        name="Model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="PROVISIONING",
        resource_profile={"gpu_id": "NVIDIA GeForce RTX 4090"},
        progress={"phase": "requesting_capacity"},
        started_at="2026-07-26T16:00:00Z",
    )
    assert request.branch_width == 4
    assert request.complexity_strategy == "adaptive"

    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **{
                **request.model_dump(),
                "branch_width": 8,
            }
        )


def test_research_result_progress_separates_execution_from_hypothesis() -> None:
    progress = research_result_progress(
        {
            "updates_completed": 857,
            "total_sampled_completions": 20568,
            "reached_complexity_level": 1,
            "promotion_count": 1,
            "initial_reward": 0.333333,
            "final_reward": 0.333333,
            "reward_gain": 0.0,
            "hypothesis_passed": False,
            "adapter_persisted": True,
            "elapsed_seconds": 1804.33,
            "stop_reason": "target_runtime",
            "final_by_level": {"1": {"exact_rate": 0.333333}},
        }
    )

    assert progress["phase"] == "complete"
    assert progress["exact_rate"] == 0.333333
    assert progress["hypothesis_passed"] is False
    assert progress["adapter_persisted"] is True
