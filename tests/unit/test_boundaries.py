from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

import services.orchestrator.app.main as orchestrator_main
from services.orchestrator.app.main import (
    ResearchComputeExecutionRequest,
    ResearchComputeProofRequest,
    estimated_compute_cost,
    research_execution_response,
    research_proof_response,
    research_result_progress,
    research_trajectory,
)
from services.orchestrator.app.providers import (
    JUDGE_PROVIDER_NAMES,
    POLICY_COMPUTE_PROVIDERS,
    assert_local_registry,
)


def test_local_provider_registry_has_no_real_provider() -> None:
    assert_local_registry()
    assert set(POLICY_COMPUTE_PROVIDERS) == {"LocalFixtureComputeProvider"}
    assert {"DeterministicJudgeFixture"} == JUDGE_PROVIDER_NAMES


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


def test_estimated_compute_cost_uses_recorded_rate_and_elapsed_runtime() -> None:
    cost = estimated_compute_cost(0.44, 2792.289)

    assert cost == {
        "total_usd": 0.34128,
        "estimated": True,
        "hourly_rate_usd": 0.44,
        "elapsed_seconds": 2792.289,
    }
    assert estimated_compute_cost(None, 120) == {
        "total_usd": None,
        "estimated": False,
        "hourly_rate_usd": None,
        "elapsed_seconds": 120.0,
    }


def test_research_execution_response_exposes_gpu_and_live_estimated_cost() -> None:
    response = research_execution_response(
        {
            "execution_id": "runpod-proof-live",
            "provider_handle": "runpod://pods/live",
            "resource_profile": {
                "gpu_id": "NVIDIA A40",
                "hourly_cost_usd": 0.44,
            },
            "progress": {"elapsed_seconds": 900},
        }
    )

    assert response["allocated_gpu"] == "NVIDIA A40"
    assert response["cost"]["total_usd"] == 0.11
    assert response["cost"]["estimated"] is True


def test_research_execution_response_handles_awaiting_allocation() -> None:
    response = research_execution_response(
        {
            "execution_id": "runpod-proof-pending",
            "provider_handle": None,
            "resource_profile": {"gpu_id": "NVIDIA A40"},
            "progress": {"phase": "requesting_capacity"},
        }
    )

    assert response["allocated_gpu"] is None
    assert response["cost"]["total_usd"] is None
    assert response["cost"]["estimated"] is False


def test_proof_list_and_detail_contracts_keep_summary_focused() -> None:
    item = {
        "proof_id": "research_proof_test",
        "execution_id": "runpod-proof-test",
        "execution_name": "Repository repair post-training",
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/pod-test",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA A40",
            "image": "runpod/pytorch:test",
            "cloud_type": "SECURE",
            "hourly_cost_usd": 0.44,
        },
        "workload": {
            "id": "repository-repair",
            "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "static_branch_width": 4,
            "complexity_strategy": "adaptive",
        },
        "result": {
            "initial_reward": 0.0,
            "final_reward": 1.0,
            "reward_gain": 1.0,
            "elapsed_seconds": 900,
            "promotion_count": 2,
            "reached_complexity_level": 2,
            "maximum_complexity_level": 3,
        },
        "receipt_digest": "sha256:receipt",
        "started_at": "2026-07-26T21:00:00Z",
        "completed_at": "2026-07-26T21:15:00Z",
        "teardown_confirmed": True,
    }

    summary = research_proof_response(item, detail=False)
    detail = research_proof_response(item, detail=True)

    assert summary["execution_id"] == "runpod-proof-test"
    assert summary["learning"]["reward_gain"] == 1.0
    assert summary["cost"]["total_usd"] == 0.11
    assert "provider" not in summary
    assert "evidence" not in summary

    assert detail["provider"]["handle"] == "runpod://pods/pod-test"
    assert detail["workload"]["model_id"].endswith("1.5B-Instruct")
    assert detail["curriculum"]["reached_level"] == 2
    assert detail["evidence"]["receipt_digest"] == "sha256:receipt"


def test_proof_api_list_and_detail_use_dedicated_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = {
        "proof_id": "research_proof_api",
        "execution_id": "runpod-proof-api",
        "execution_name": "Repository repair post-training",
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/api",
        "provider_cli_version": "2.7.2",
        "resource_profile": {"gpu_id": "NVIDIA A40", "hourly_cost_usd": 0.44},
        "workload": {"id": "repository-repair"},
        "result": {"elapsed_seconds": 900, "reward_gain": 1.0},
        "receipt_digest": "sha256:api",
        "started_at": "2026-07-26T21:00:00Z",
        "completed_at": "2026-07-26T21:15:00Z",
        "teardown_confirmed": True,
    }

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            _query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            return Result([item])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    proof_list = orchestrator_main.list_research_compute_proofs()
    proof_detail = orchestrator_main.get_research_compute_proof("research_proof_api")

    assert proof_list["items"][0]["proof_id"] == "research_proof_api"
    assert "provider" not in proof_list["items"][0]
    assert proof_detail["provider"]["handle"] == "runpod://pods/api"
    assert proof_detail["evidence"]["receipt_digest"] == "sha256:api"


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
            "post_training_completed": True,
            "informative_group_rate": 0.2,
            "policy_update_count": 10,
            "claim_strength": "EXPLORATORY_SINGLE_SEED",
            "seed_count": 1,
            "objective": "leave-one-out-group-normalized-reinforce@1",
            "elapsed_seconds": 1804.33,
            "stop_reason": "target_runtime",
            "final_by_level": {"1": {"exact_rate": 0.333333}},
        }
    )

    assert progress["phase"] == "complete"
    assert progress["exact_rate"] == 0.333333
    assert progress["hypothesis_passed"] is False
    assert progress["adapter_persisted"] is True
    assert progress["post_training_completed"] is True
    assert progress["informative_group_rate"] == 0.2
    assert progress["claim_strength"] == "EXPLORATORY_SINGLE_SEED"
    assert progress["seed_count"] == 1


def test_research_trajectory_keeps_only_persisted_training_evidence() -> None:
    trajectory = research_trajectory(
        {
            "branch_width": 4,
            "complexity_strategy": "adaptive",
            "maximum_complexity_level": 3,
            "reached_complexity_level": 2,
            "updates_completed": 80,
            "initial_reward": 0.4,
            "final_reward": 0.9,
            "reward_gain": 0.5,
            "history": [
                {"update": 20, "level": 0, "exact_rate": 0.8},
                "invalid",
                {"update": 40, "level": 1, "exact_rate": 0.9},
            ],
            "promotions": [{"update": 40, "from_level": 0, "to_level": 1}],
            "branch_snapshots": [
                {
                    "snapshot_id": "update-20-sqlite_repair",
                    "siblings": [{"index": index} for index in range(4)],
                },
                "invalid",
            ],
            "initial_by_level": {"0": {"exact_rate": 0.4}},
            "final_by_level": {"0": {"exact_rate": 0.9}},
            "policy_update_count": 7,
        }
    )

    assert trajectory["branch_width"] == 4
    assert trajectory["checkpoints"] == [
        {"update": 20, "level": 0, "exact_rate": 0.8},
        {"update": 40, "level": 1, "exact_rate": 0.9},
    ]
    assert trajectory["promotions"][0]["to_level"] == 1
    assert trajectory["branch_snapshots"] == [
        {
            "snapshot_id": "update-20-sqlite_repair",
            "siblings": [{"index": index} for index in range(4)],
        }
    ]
    assert trajectory["policy_update_count"] == 7


def test_research_trajectory_projects_live_multi_step_branch_lineage() -> None:
    latest_branch = {
        "schema_version": 2,
        "snapshot_id": "update-2-snapshot-a",
        "shared_prefix": {"steps": [{"step_id": "prefix-0"}, {"step_id": "prefix-1"}]},
        "checkpoint": {
            "checkpoint_id": "snapshot-a",
            "fidelity": "logical_restore",
            "static_branch_width": 4,
        },
        "siblings": [
            {
                "index": index,
                "steps": [
                    {"step_id": f"sibling-{index}-2"},
                    {"step_id": f"sibling-{index}-3"},
                ],
            }
            for index in range(4)
        ],
    }

    trajectory = research_trajectory(
        {
            "schema_version": 2,
            "branch_width": 4,
            "complexity_strategy": "adaptive",
            "multi_step": True,
            "restored_continuations": True,
            "latest_branch_snapshot": latest_branch,
            "total_sampled_actions": 34,
        }
    )

    assert trajectory["schema_version"] == 2
    assert trajectory["multi_step"] is True
    assert trajectory["restored_continuations"] is True
    assert trajectory["branch_snapshots"] == [latest_branch]
    assert trajectory["total_sampled_actions"] == 34
