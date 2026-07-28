from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from equinox_core import canonical_digest, make_id, utc_now
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .database import connection, migrate
from .environments import (
    active_complexity_range,
    advance_complexity,
    environment_catalog,
    environment_spec,
)
from .providers import POLICY_COMPUTE_PROVIDERS, assert_local_registry
from .science import artifact_store, emit_event
from .studies import find_study_report, list_study_summaries
from .workflow import execute_run_attempt, rejudge_transition, release_run_resources


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BranchConfig(StrictModel):
    mode: Literal["static"] = "static"
    width: Literal[1, 4]
    decision_after_actions: Literal[3] = 3
    rng_mode: Literal["split_stream"] = "split_stream"


class ComplexityConfig(StrictModel):
    strategy: Literal["adaptive"] = "adaptive"
    minimum_level: int = Field(default=0, ge=0, le=100)
    initial_level: int = Field(default=1, ge=0, le=100)
    maximum_level: int = Field(default=8, ge=0, le=100)
    sampling_band: int = Field(default=4, ge=1, le=20)
    mastery_threshold: float = Field(default=0.9, gt=0, le=1)
    evaluation_window: int = Field(default=32, ge=8, le=1024)
    promotion_step: int = Field(default=1, ge=1, le=10)


class BudgetConfig(StrictModel):
    transitions: int = Field(default=64, ge=8, le=256)
    environment_cpu_seconds: int = Field(default=600, ge=60, le=3600)
    render_cpu_seconds: int = Field(default=300, ge=30, le=1800)
    judge_input_tokens: int = Field(default=100000, ge=1000, le=1000000)


class LaunchRunRequest(StrictModel):
    name: str = Field(min_length=3, max_length=80)
    algorithm: Literal["independent_rollout_baseline", "bpo_local_metric"]
    environment_id: Literal["cad.reconstruction"] = "cad.reconstruction"
    policy_compute_provider: Literal["LocalFixtureComputeProvider"] = "LocalFixtureComputeProvider"
    judge_provider: Literal["DeterministicJudgeFixture"] = "DeterministicJudgeFixture"
    task_revision: Literal["mounting-plate@sha256:fixture-v1"] = "mounting-plate@sha256:fixture-v1"
    branch: BranchConfig
    complexity: ComplexityConfig = ComplexityConfig()
    budgets: BudgetConfig = BudgetConfig()
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    retention_class: Literal["local-research"] = "local-research"

    @model_validator(mode="after")
    def validate_supported_plan(self) -> LaunchRunRequest:
        expected_width = 1 if self.algorithm == "independent_rollout_baseline" else 4
        if self.branch.width != expected_width:
            raise ValueError(f"{self.algorithm} requires static branch width K={expected_width}")
        required_transitions = 24 if expected_width == 1 else 15
        if self.budgets.transitions < required_transitions:
            raise ValueError(
                f"{self.algorithm} requires at least {required_transitions} transition credits"
            )
        return self


class CancelRequest(StrictModel):
    mode: Literal["terminate"] = "terminate"
    operation_id: str


class RejudgeApiRequest(StrictModel):
    verification_run_id: str
    fixture_scenario: Literal[
        "valid", "low", "tie", "abstain", "malformed", "retry", "disagreement", "integrity"
    ] = "valid"


class WorkerClaimRequest(StrictModel):
    worker_id: str = Field(min_length=3, max_length=160)


class ClaimedAttemptRequest(StrictModel):
    worker_id: str = Field(min_length=3, max_length=160)
    claim_id: str = Field(min_length=3, max_length=160)
    fencing_token: int = Field(ge=1)


class ComplexityObservationRequest(StrictModel):
    operation_id: str = Field(min_length=3, max_length=160)
    level: int = Field(ge=0, le=100)
    collection_closure_id: str = Field(min_length=3, max_length=160)
    behavior_policy_version_id: str = Field(min_length=3, max_length=160)
    task_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    success_definition: str = Field(min_length=3, max_length=240)
    ordered_outcomes: list[bool] = Field(min_length=1, max_length=1024)


class ResearchComputeProofRequest(StrictModel):
    provider_name: Literal["RunPod"]
    provider_handle: str = Field(pattern=r"^runpod://pods/[a-zA-Z0-9_-]+$")
    provider_cli_version: str = Field(min_length=1, max_length=80)
    resource_profile: dict[str, Any]
    workload: dict[str, Any]
    result: dict[str, Any]
    started_at: datetime
    completed_at: datetime
    teardown_confirmed: Literal[True]


class ResearchComputeExecutionRequest(StrictModel):
    name: str = Field(min_length=3, max_length=120)
    workload_id: str = Field(min_length=3, max_length=160)
    model_id: str | None = Field(default=None, max_length=200)
    branch_width: Literal[1, 4] = 4
    complexity_strategy: Literal["adaptive"] = "adaptive"
    status: Literal["PROVISIONING", "RUNNING", "FINALIZING", "SUCCEEDED", "FAILED"]
    provider_name: Literal["RunPod"] = "RunPod"
    provider_handle: str | None = Field(
        default=None,
        pattern=r"^runpod://pods/[a-zA-Z0-9_-]+$",
    )
    resource_profile: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime | None = None
    teardown_confirmed: bool = False


RECOVERABLE_PROOF_CONTRACT_ERROR = "The remote result did not satisfy the declared proof contract."


def research_proof_can_recover_execution(
    execution: dict[str, Any],
    request: ResearchComputeProofRequest,
) -> bool:
    progress = execution.get("progress")
    workload_id = request.workload.get("id")
    result_workload_id = request.result.get("workload")
    receipt_model_id = request.workload.get("model_id") or request.result.get("model_id")
    return (
        execution.get("status") == "FAILED"
        and execution.get("teardown_confirmed") is True
        and isinstance(progress, dict)
        and progress.get("error") == RECOVERABLE_PROOF_CONTRACT_ERROR
        and execution.get("provider_handle") == request.provider_handle
        and execution.get("workload_id") == workload_id
        and workload_id == result_workload_id
        and execution.get("model_id") == receipt_model_id
        and execution.get("started_at") == request.started_at
        and request.teardown_confirmed is True
    )


def lightweight_research_validation_history(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    keys = (
        "update",
        "level",
        "exact_successes",
        "exact_rate",
        "exact_rate_95ci",
        "checkpoint_successes",
        "checkpoint_rate",
        "checkpoint_rate_95ci",
        "examples",
        "mastered",
        "mastery_streak",
        "policy_loss",
        "reinforce_loss",
        "reference_kl",
        "gradient_norm",
        "curriculum_baseline_exact_successes",
        "curriculum_baseline_exact_rate",
        "curriculum_exact_successes",
        "curriculum_exact_rate",
        "curriculum_paired_change",
        "fixed_guard_levels",
        "fixed_guard_paired_change",
        "retention_guard_passed",
        "elapsed_seconds",
        "validation_elapsed_seconds",
        "final_evaluation_reserve_seconds",
        "regression_streak",
        "best_checkpoint",
    )
    return [
        {key: item[key] for key in keys if key in item} for item in value if isinstance(item, dict)
    ]


def research_result_progress(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("workload") == "revision30-post-freeze-external-adapter-evaluation":
        adapter_count = result.get("adapter_count")
        task_count = result.get("task_count")
        policy_count = (
            adapter_count + 1
            if isinstance(adapter_count, int) and not isinstance(adapter_count, bool)
            else None
        )
        evaluation_total = (
            policy_count * task_count
            if policy_count is not None
            and isinstance(task_count, int)
            and not isinstance(task_count, bool)
            else None
        )
        return {
            key: value
            for key, value in {
                "phase": "complete",
                "message": (
                    "External adapter evaluation completed; artifacts persisted "
                    "and provider teardown confirmed."
                ),
                "external_evaluation_completed": result.get("external_evaluation_completed"),
                "adapter_count": adapter_count,
                "policy_count": policy_count,
                "task_count": task_count,
                "evaluation_completed": evaluation_total,
                "evaluation_total": evaluation_total,
                "evaluation_split": "post-freeze external pack",
                "elapsed_seconds": result.get("elapsed_seconds"),
                "model_id": result.get("model", {}).get("id")
                if isinstance(result.get("model"), dict)
                else None,
                "pack_id": result.get("pack", {}).get("pack_id")
                if isinstance(result.get("pack"), dict)
                else None,
            }.items()
            if value is not None
        }
    level = result.get("reached_complexity_level")
    final_evaluation_partial = result.get("final_evaluation_partial") is True
    initial_by_level = result.get("initial_by_level")
    final_by_level = result.get("final_by_level")
    initial_level_result: dict[str, Any] = {}
    if isinstance(initial_by_level, dict) and level is not None:
        candidate = initial_by_level.get(str(level))
        if isinstance(candidate, dict):
            initial_level_result = candidate
    level_result: dict[str, Any] = {}
    if isinstance(final_by_level, dict) and level is not None:
        candidate = final_by_level.get(str(level))
        if isinstance(candidate, dict):
            level_result = candidate
    level_exact_rate = level_result.get("exact_rate")
    aggregate_rate_fallback = level_exact_rate is None
    aggregate_has_test_provenance = (
        isinstance(final_by_level, dict) and result.get("test_examples") is not None
    )
    exact_rate = level_exact_rate
    if aggregate_rate_fallback and not final_evaluation_partial:
        exact_rate = result.get("final_reward")
    checkpoint_rate = level_result.get("checkpoint_rate")

    values = {
        "phase": "complete",
        "message": (
            (
                "Training completed; final evaluation reached its workload deadline "
                "and partial evidence was persisted."
            )
            if final_evaluation_partial
            else (
                "Training and final evaluation completed; artifacts persisted "
                "and provider teardown confirmed."
            )
        ),
        "update": result.get("updates_completed"),
        "current_level": level,
        "promotion_count": result.get("promotion_count"),
        "frontier_probe_task_groups": result.get("frontier_probe_task_groups"),
        "maximum_sampled_complexity_level": result.get("maximum_sampled_complexity_level"),
        "sampled_completions": result.get("total_sampled_completions"),
        "exact_rate": exact_rate,
        "exact_rate_95ci": (
            None if aggregate_rate_fallback else level_result.get("exact_rate_95ci")
        ),
        "exact_rate_source": (
            (
                (
                    "aggregate_test_mean"
                    if aggregate_has_test_provenance
                    else "final_evaluation_reward"
                )
                if aggregate_rate_fallback
                else "reached_level"
            )
            if exact_rate is not None
            else None
        ),
        "initial_exact_rate": (None if final_evaluation_partial else result.get("initial_reward")),
        "initial_level_exact_rate": (
            None if final_evaluation_partial else initial_level_result.get("exact_rate")
        ),
        "final_exact_rate": (None if final_evaluation_partial else result.get("final_reward")),
        "reward_gain": (None if final_evaluation_partial else result.get("reward_gain")),
        "paired_test_change": (
            None if final_evaluation_partial else result.get("paired_test_change")
        ),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "stop_reason": result.get("stop_reason"),
        "hypothesis_passed": result.get("hypothesis_passed"),
        "adapter_persisted": result.get("adapter_persisted"),
        "post_training_completed": result.get("post_training_completed"),
        "informative_group_rate": result.get("informative_group_rate"),
        "policy_update_count": result.get("policy_update_count"),
        "optimizer_update_count": result.get("optimizer_update_count"),
        "pending_informative_group_count": result.get("pending_informative_group_count"),
        "pending_informative_group_ids": result.get("pending_informative_group_ids"),
        "pending_policy_example_count": result.get("pending_policy_example_count"),
        "pending_training_example_count": result.get("pending_training_example_count"),
        "action_protocol_validity_rate": result.get("action_protocol_validity_rate"),
        "recent_malformed_action_rate": (
            result.get("recent_action_protocol", {}).get("malformed_rate")
            if isinstance(result.get("recent_action_protocol"), dict)
            else None
        ),
        "best_validation": result.get("best_validation"),
        "rollback_applied": result.get("rollback_applied"),
        "validation_history": lightweight_research_validation_history(result.get("history")),
        "curriculum_history": result.get("promotions"),
        "reward_contract": result.get("reward_contract"),
        "optimizer_contract": result.get("optimizer_contract"),
        "final_evaluation_reserve_seconds": result.get("final_evaluation_reserve_seconds"),
        "target_runtime_seconds": result.get("target_runtime_seconds"),
        "claim_strength": result.get("claim_strength"),
        "seed_count": result.get("seed_count", result.get("optimization_seed_count")),
        "objective": result.get("objective", result.get("objective_id")),
        "checkpoint_rate": checkpoint_rate,
        "checkpoint_rate_95ci": (
            level_result.get("checkpoint_rate_95ci") if checkpoint_rate is not None else None
        ),
        "checkpoint_rate_source": ("reached_level" if checkpoint_rate is not None else None),
        "evaluation_examples": (None if aggregate_rate_fallback else level_result.get("examples")),
        "evaluation_completed": (None if aggregate_rate_fallback else level_result.get("examples")),
        "evaluation_total": result.get("test_examples"),
        "evaluation_split": (
            (
                ("test" if aggregate_has_test_provenance else None)
                if aggregate_rate_fallback
                else level_result.get("split")
            )
            if exact_rate is not None
            else None
        ),
        "validation_examples": result.get("validation_examples"),
        "test_examples": result.get("test_examples"),
        "mastery_windows": result.get("mastery_windows"),
        "teacher_data_used": result.get("teacher_data_used"),
        "resumed_from_checkpoint": result.get("resumed_from_checkpoint"),
        "attempt_count": result.get("attempt_count"),
        "attempt_elapsed_seconds": result.get("attempt_elapsed_seconds"),
        "raw_resume_gap_seconds": result.get("raw_resume_gap_seconds"),
        "applied_resume_gap_seconds": result.get("applied_resume_gap_seconds"),
        "crash_tail_actions_unaccounted": result.get("crash_tail_actions_unaccounted"),
        "crash_tail_cost_accounting": result.get("crash_tail_cost_accounting"),
        "final_evaluation_reserve_exceeded_ceiling": result.get(
            "final_evaluation_reserve_exceeded_ceiling"
        ),
        "final_evaluation_complete": result.get("final_evaluation_complete"),
        "final_evaluation_partial": result.get("final_evaluation_partial"),
        "final_evaluation_deadline_seconds": result.get("final_evaluation_deadline_seconds"),
        "distinct_semantic_examples": level_result.get("distinct_semantic_examples"),
        "semantic_universe_size": level_result.get("semantic_universe_size"),
        "structural_mirror_disclosures": result.get("structural_mirror_disclosures"),
        "maximum_measured_final_evaluation_reserve_seconds": result.get(
            "maximum_measured_final_evaluation_reserve_seconds"
        ),
        "total_sampled_actions": result.get("total_sampled_actions"),
        "discarded_task_groups": result.get("discarded_task_groups"),
        "discarded_sampled_actions": result.get("discarded_sampled_actions"),
        "discarded_post_branch_actions": result.get("discarded_post_branch_actions"),
        "multi_step": result.get("multi_step"),
        "restored_continuations": result.get("restored_continuations"),
        "restored_branching_observed": result.get("restored_branching_observed"),
    }
    return {key: value for key, value in values.items() if value is not None}


def research_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    initial_by_level = result.get("initial_by_level")
    final_by_level = result.get("final_by_level")
    history = result.get("history", result.get("validation_history"))
    promotions = result.get("promotions", result.get("curriculum_history"))
    branch_snapshots = result.get("branch_snapshots")
    latest_branch_snapshot = result.get("latest_branch_snapshot")
    persisted_branch_snapshots = (
        [item for item in branch_snapshots if isinstance(item, dict)]
        if isinstance(branch_snapshots, list)
        else []
    )
    if not persisted_branch_snapshots and isinstance(latest_branch_snapshot, dict):
        persisted_branch_snapshots = [latest_branch_snapshot]
    return {
        "schema_version": 2 if result.get("multi_step") else 1,
        "branch_width": result.get("branch_width"),
        "complexity_strategy": result.get("complexity_strategy"),
        "multi_step": result.get("multi_step", False),
        "restored_continuations": result.get("restored_continuations", False),
        "prefix_gradient": result.get("prefix_gradient"),
        "replay_enabled": result.get("replay_enabled"),
        "environment_revision": result.get("environment_revision"),
        "verifier_revision": result.get("verifier_revision"),
        "action_protocol_revision": result.get("action_protocol_revision"),
        "snapshot_fidelity": result.get("snapshot_fidelity"),
        "maximum_level": result.get("maximum_complexity_level", result.get("maximum_level")),
        "reached_level": result.get("reached_complexity_level", result.get("current_level")),
        "maximum_sampled_level": result.get("maximum_sampled_complexity_level"),
        "updates_completed": result.get("updates_completed", result.get("update")),
        "initial_exact_rate": (
            None if result.get("final_evaluation_partial") is True else result.get("initial_reward")
        ),
        "final_exact_rate": (
            None if result.get("final_evaluation_partial") is True else result.get("final_reward")
        ),
        "exact_gain": (
            None if result.get("final_evaluation_partial") is True else result.get("reward_gain")
        ),
        "stop_reason": result.get("stop_reason"),
        "best_validation": result.get("best_validation"),
        "rollback_applied": result.get("rollback_applied"),
        "action_protocol_validity_rate": result.get("action_protocol_validity_rate"),
        "reward_contract": result.get("reward_contract"),
        "optimizer_contract": result.get("optimizer_contract"),
        "checkpoints": (
            [item for item in history if isinstance(item, dict)]
            if isinstance(history, list)
            else []
        ),
        "promotions": (
            [item for item in promotions if isinstance(item, dict)]
            if isinstance(promotions, list)
            else []
        ),
        "branch_snapshots": persisted_branch_snapshots,
        "initial_by_level": initial_by_level if isinstance(initial_by_level, dict) else {},
        "final_by_level": final_by_level if isinstance(final_by_level, dict) else {},
        "policy_update_count": result.get("policy_update_count"),
        "optimizer_update_count": result.get("optimizer_update_count"),
        "frontier_probe_task_groups": result.get("frontier_probe_task_groups"),
        "pending_informative_group_count": result.get("pending_informative_group_count"),
        "pending_informative_group_ids": result.get("pending_informative_group_ids"),
        "pending_policy_example_count": result.get("pending_policy_example_count"),
        "pending_training_example_count": result.get("pending_training_example_count"),
        "informative_group_rate": result.get("informative_group_rate"),
        "total_sampled_completions": result.get("total_sampled_completions"),
        "total_sampled_actions": result.get("total_sampled_actions"),
        "total_post_branch_actions": result.get("total_post_branch_actions"),
        "discarded_task_groups": result.get("discarded_task_groups"),
        "discarded_sampled_actions": result.get("discarded_sampled_actions"),
        "discarded_post_branch_actions": result.get("discarded_post_branch_actions"),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def estimated_compute_cost(
    hourly_rate_usd: Any,
    elapsed_seconds: Any,
) -> dict[str, Any]:
    rate = _number(hourly_rate_usd)
    elapsed = _number(elapsed_seconds)
    total = round(rate * elapsed / 3600, 6) if rate is not None and elapsed is not None else None
    return {
        "total_usd": total,
        "estimated": total is not None,
        "hourly_rate_usd": rate,
        "elapsed_seconds": elapsed,
    }


def research_execution_response(item: dict[str, Any]) -> dict[str, Any]:
    response = dict(item)
    resource_profile = (
        item["resource_profile"] if isinstance(item.get("resource_profile"), dict) else {}
    )
    progress = item["progress"] if isinstance(item.get("progress"), dict) else {}
    gpu = (
        progress.get("gpu_name") or resource_profile.get("gpu_id")
        if item.get("provider_handle")
        else None
    )
    response["allocated_gpu"] = gpu if isinstance(gpu, str) and gpu else None
    response["cost"] = estimated_compute_cost(
        resource_profile.get("hourly_cost_usd"),
        progress.get("elapsed_seconds"),
    )
    return response


def research_proof_response(
    item: dict[str, Any],
    *,
    detail: bool,
) -> dict[str, Any]:
    resource_profile = (
        item["resource_profile"] if isinstance(item.get("resource_profile"), dict) else {}
    )
    workload = item["workload"] if isinstance(item.get("workload"), dict) else {}
    result = item["result"] if isinstance(item.get("result"), dict) else {}
    final_evaluation_partial = result.get("final_evaluation_partial") is True
    elapsed = result.get("elapsed_seconds")
    gpu = result.get("gpu_name") or resource_profile.get("gpu_id")

    response: dict[str, Any] = {
        "proof_id": item["proof_id"],
        "execution_id": item.get("execution_id"),
        "run_name": item.get("execution_name"),
        "completed_at": item["completed_at"],
        "learning": {
            "initial_reward": (None if final_evaluation_partial else result.get("initial_reward")),
            "final_reward": (None if final_evaluation_partial else result.get("final_reward")),
            "reward_gain": (None if final_evaluation_partial else result.get("reward_gain")),
            "hypothesis_passed": result.get("hypothesis_passed"),
            "claim_strength": result.get("claim_strength"),
            "seed_count": result.get("seed_count", result.get("optimization_seed_count")),
        },
        "hardware": {
            "provider": item["provider_name"],
            "gpu": gpu if isinstance(gpu, str) and gpu else None,
            "image": resource_profile.get("image"),
            "cloud_type": resource_profile.get("cloud_type"),
            "hourly_rate_usd": _number(resource_profile.get("hourly_cost_usd")),
        },
        "cost": estimated_compute_cost(
            resource_profile.get("hourly_cost_usd"),
            elapsed,
        ),
        "teardown_confirmed": item["teardown_confirmed"],
    }
    if not detail:
        return response

    response.update(
        {
            "started_at": item["started_at"],
            "runtime_seconds": _number(elapsed),
            "provider": {
                "name": item["provider_name"],
                "handle": item["provider_handle"],
                "cli_version": item["provider_cli_version"],
            },
            "workload": {
                "id": workload.get("id"),
                "revision": workload.get("revision"),
                "algorithm": workload.get("algorithm"),
                "objective": (
                    result.get("objective")
                    or result.get("objective_id")
                    or workload.get("objective")
                ),
                "model_id": workload.get("model_id") or result.get("model_id"),
                "model_revision": workload.get("model_revision") or result.get("model_revision"),
                "branch_width": workload.get("static_branch_width"),
                "complexity_strategy": workload.get("complexity_strategy"),
                "task_domains": workload.get("task_domains"),
                "snapshot_fidelity": workload.get("snapshot_fidelity"),
                "multi_step": workload.get("multi_step"),
                "restored_continuations": workload.get("restored_continuations"),
            },
            "curriculum": {
                "promotion_count": result.get("promotion_count"),
                "promotions": result.get("promotions"),
                "reached_level": result.get(
                    "reached_complexity_level",
                    workload.get("reached_complexity_level"),
                ),
                "maximum_level": result.get(
                    "maximum_complexity_level",
                    workload.get("maximum_complexity_level"),
                ),
                "updates_completed": result.get("updates_completed"),
                "stop_reason": result.get("stop_reason"),
                "retention_passed": result.get("retention_passed"),
            },
            "evidence": {
                "receipt_digest": item["receipt_digest"],
                "teardown_confirmed": item["teardown_confirmed"],
            },
        }
    )
    return response


def _wait_for_dependencies() -> None:
    last_error: Exception | None = None
    for _ in range(40):
        try:
            migrate()
            artifact_store.ensure_bucket()
            assert_local_registry()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError("orchestrator dependencies did not become ready") from last_error


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("EQUINOX_DEPLOYMENT_MODE") != "local-only":
        raise RuntimeError(
            "This build has no user authentication and only supports local-only deployment"
        )
    _wait_for_dependencies()
    with connection() as conn:
        canceled_run_ids = [
            row["run_id"]
            for row in conn.execute(
                """
                SELECT run_id FROM runs
                WHERE desired_state = 'CANCELED'
                  AND status NOT IN ('SUCCEEDED', 'FAILED', 'CANCELED')
                ORDER BY created_at
                """
            )
        ]
        expired = list(
            conn.execute(
                """
                UPDATE run_attempts
                SET status = 'QUEUED',
                    claim_id = NULL,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = now(),
                    updated_at = now()
                WHERE status IN ('PROVISIONING', 'PREPARING', 'RUNNING')
                  AND lease_expires_at < now()
                RETURNING run_id
                """
            )
        )
        if expired:
            conn.execute(
                """
                UPDATE runs
                SET status = 'QUEUED', updated_at = now()
                WHERE run_id = ANY(%s)
                  AND desired_state = 'RUNNING'
                  AND status NOT IN ('SUCCEEDED', 'FAILED', 'CANCELED')
                """,
                ([row["run_id"] for row in expired],),
            )
    for run_id in canceled_run_ids:
        release_run_resources(run_id)
    yield


app = FastAPI(title="Equinox Next orchestrator", version="0.1.0", lifespan=lifespan)
request_logger = logging.getLogger("equinox.requests")
INTERNAL_TOKEN = os.environ.get("EQUINOX_INTERNAL_TOKEN", "")
if len(INTERNAL_TOKEN) < 32:
    raise RuntimeError("EQUINOX_INTERNAL_TOKEN must contain at least 32 characters")


@app.middleware("http")
async def authenticate_internal_api(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "")
    if not request_id or len(request_id) > 128:
        request_id = uuid.uuid4().hex
    started = time.monotonic()
    if request.url.path.startswith("/internal/"):
        authorization = request.headers.get("authorization", "")
        supplied = (
            authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        )
        if not supplied or not secrets.compare_digest(supplied, INTERNAL_TOKEN):
            response = JSONResponse(
                status_code=401,
                content={"detail": {"code": "INTERNAL_AUTH_REQUIRED"}},
            )
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    request_logger.info(
        json.dumps(
            {
                "event": "http_request",
                "service": "orchestrator",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            },
            sort_keys=True,
        )
    )
    return response


def _launch(request: LaunchRunRequest, *, source_run_id: str | None = None) -> dict[str, Any]:
    environment = environment_spec(request.environment_id)
    if environment is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "UNKNOWN_ENVIRONMENT",
                "message": f"Unknown environment {request.environment_id}.",
            },
        )
    if not environment["launch_enabled"]:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "RUNPOOL_ADAPTER_REQUIRED",
                "message": (
                    f"{environment['name']} has a configuration draft. "
                    "Connect the RunPool execution adapter before launching it."
                ),
            },
        )
    if not (
        request.complexity.minimum_level
        <= request.complexity.initial_level
        <= request.complexity.maximum_level
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_COMPLEXITY_RANGE",
                "message": "Difficulty must satisfy minimum ≤ initial ≤ maximum.",
            },
        )
    if request.algorithm == "bpo_local_metric" and request.branch.width != 4:
        raise HTTPException(
            status_code=422,
            detail={"code": "BRANCH_WIDTH_REQUIRED", "message": "BPO local requires width four."},
        )
    if request.algorithm == "independent_rollout_baseline" and request.branch.width != 1:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INDEPENDENT_WIDTH_REQUIRED",
                "message": "Independent rollout baseline uses width one.",
            },
        )
    normalized = {
        "schema_version": 1,
        "profile": "local-contract-proof",
        "environment": {
            "id": request.environment_id,
            "version": "1.0.0",
            "status": environment["status"],
            "execution_provider": "ComposeExecutionProvider",
            "snapshot_fidelity": "logical_restore",
        },
        "task_revision": request.task_revision,
        "algorithm": {"id": request.algorithm, "version": "local@1"},
        "policy_compute": {
            "provider": request.policy_compute_provider,
            "resource_profile": "fixture-cpu",
        },
        "verification": {
            "plan_id": "cad.transition-composite@1",
            "judge_provider": request.judge_provider,
            "judge_spec_id": "cad.pointwise.fixture@1",
            "group_judge_spec_id": "cad.sibling-group.fixture@1",
            "calibration_status": "FIXTURE_CONTRACT_ONLY",
        },
        "branch": request.branch.model_dump(),
        "complexity": request.complexity.model_dump(),
        "budgets": request.budgets.model_dump(),
        "seed": request.seed,
        "retention_class": request.retention_class,
        "network": "disabled",
        "credentials_required": [],
    }
    run_id = make_id("run")
    attempt_id = make_id("attempt")
    policy_id = make_id("policy")
    provider = POLICY_COMPUTE_PROVIDERS[request.policy_compute_provider]
    allocation = provider.allocate("fixture-cpu")
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO runs(
              run_id, source_run_id, name, algorithm, status, manifest, manifest_digest
            ) VALUES (%s, %s, %s, %s, 'QUEUED', %s, %s)
            """,
            (
                run_id,
                source_run_id,
                request.name,
                request.algorithm,
                Jsonb(normalized),
                canonical_digest(normalized),
            ),
        )
        conn.execute(
            """
            INSERT INTO run_attempts(
              attempt_id, run_id, attempt_number, status, provider_name, allocation_id
            ) VALUES (%s, %s, 1, 'QUEUED', 'LocalFixtureComputeProvider', %s)
            """,
            (attempt_id, run_id, allocation.allocation_id),
        )
        conn.execute(
            """
            INSERT INTO compute_allocations(
              allocation_id, run_attempt_id, provider_name, desired_state, observed_state,
              resource_profile, provider_handle
            ) VALUES (%s, %s, 'LocalFixtureComputeProvider', 'ALLOCATED', 'ALLOCATED', %s, %s)
            """,
            (
                allocation.allocation_id,
                attempt_id,
                allocation.resource_profile,
                allocation.provider_handle,
            ),
        )
        conn.execute(
            """
            INSERT INTO complexity_states(
              run_id, environment_id, strategy, minimum_level, current_level,
              maximum_level, sampling_band, mastery_threshold, evaluation_window,
              promotion_step
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                request.environment_id,
                request.complexity.strategy,
                request.complexity.minimum_level,
                request.complexity.initial_level,
                request.complexity.maximum_level,
                request.complexity.sampling_band,
                request.complexity.mastery_threshold,
                request.complexity.evaluation_window,
                request.complexity.promotion_step,
            ),
        )
        policy_manifest = {
            "model": "deterministic-cad-policy@1",
            "tokenizer": "fixture-tokenizer@1",
            "prompt_template": "cad-policy@1",
            "tool_schema": "cad-actions@1",
            "sampling": {"temperature": 0, "seed": request.seed},
        }
        conn.execute(
            """
            INSERT INTO policy_versions(
              policy_version_id, run_id, ordinal, artifact_digest,
              behavior_manifest, update_kind
            ) VALUES (%s, %s, 0, %s, %s, 'INITIAL_FIXTURE')
            """,
            (policy_id, run_id, canonical_digest(policy_manifest), Jsonb(policy_manifest)),
        )
        emit_event(
            conn,
            event_type="run.created",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=run_id,
            payload={
                "run_id": run_id,
                "attempt_id": attempt_id,
                "manifest_digest": canonical_digest(normalized),
                "providers": ["LocalFixtureComputeProvider", "DeterministicJudgeFixture"],
                "source_run_id": source_run_id,
            },
        )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": "QUEUED",
        "manifest": normalized,
        "manifest_digest": canonical_digest(normalized),
    }


@app.get("/healthz")
def health() -> dict[str, Any]:
    return readiness()


@app.get("/livez")
def liveness() -> dict[str, str]:
    return {"status": "alive", "service": "orchestrator"}


@app.get("/readyz")
def readiness() -> dict[str, Any]:
    with connection() as conn:
        conn.execute("SELECT 1")
    artifact_store.ensure_bucket()
    return {
        "status": "ready",
        "service": "orchestrator",
        "policy_compute_providers": sorted(POLICY_COMPUTE_PROVIDERS),
        "judge_providers": ["DeterministicJudgeFixture"],
        "profile": "local-contract-proof",
    }


@app.get("/v1/run-templates")
def run_templates() -> dict[str, Any]:
    return {
        "templates": [
            {
                "template_id": "cad-independent-local@1",
                "name": "Independent CAD baseline",
                "algorithm": "independent_rollout_baseline",
                "branch_width": 1,
                "policy_compute_provider": "LocalFixtureComputeProvider",
                "judge_provider": "DeterministicJudgeFixture",
                "estimated_cost": {
                    "execution_credits": 0.6,
                    "render_credits": 0.3,
                    "judge_credits": 0,
                    "total_credits": 0.9,
                },
            },
            {
                "template_id": "cad-branch-local@1",
                "name": "Branch-aware CAD proof",
                "algorithm": "bpo_local_metric",
                "branch_width": 4,
                "policy_compute_provider": "LocalFixtureComputeProvider",
                "judge_provider": "DeterministicJudgeFixture",
                "estimated_cost": {
                    "execution_credits": 0.4,
                    "render_credits": 0.2,
                    "judge_credits": 0,
                    "total_credits": 0.6,
                },
            },
        ],
        "environments": environment_catalog(),
        "schema": LaunchRunRequest.model_json_schema(),
    }


@app.get("/v1/environments")
def environments() -> dict[str, Any]:
    return {
        "items": environment_catalog(),
        "complexity_schema": ComplexityConfig.model_json_schema(),
        "branching": {
            "mode": "static",
            "branch_width": 4,
            "note": "Branch width is fixed for each run and recorded in its manifest.",
        },
    }


@app.post("/v1/runs", status_code=202)
def launch_run(request: LaunchRunRequest) -> dict[str, Any]:
    return _launch(request)


@app.get("/v1/runs")
def list_runs() -> dict[str, Any]:
    with connection() as conn:
        data = list(
            conn.execute(
                """
                SELECT r.*,
                  (SELECT count(*) FROM collection_batches cb
                   JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                   WHERE ra.run_id = r.run_id) AS collection_batch_count,
                  (SELECT count(*) FROM training_iterations ti
                   WHERE ti.run_id = r.run_id) AS iteration_count,
                  (SELECT count(*) FROM rollout_trees rt
                   JOIN collection_batches cb ON cb.collection_batch_id = rt.collection_batch_id
                   JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                   WHERE ra.run_id = r.run_id) AS rollout_tree_count,
                  (SELECT count(*) FROM verification_runs vr
                   WHERE vr.run_id = r.run_id) AS verification_run_count,
                  (SELECT count(*) FROM judge_results jr
                   JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
                   JOIN verification_runs vr ON vr.verification_run_id = ji.verification_run_id
                   WHERE vr.run_id = r.run_id AND jr.outcome = 'ABSTAINED') AS abstention_count,
                  (SELECT count(*) FROM verifier_step_runs vsr
                   JOIN verification_runs vr ON vr.verification_run_id = vsr.verification_run_id
                   WHERE vr.run_id = r.run_id AND vsr.attempt_count > 1) AS retry_count
                FROM runs r ORDER BY r.created_at DESC
                """
            )
        )
        research_items = list(
            conn.execute(
                """
                SELECT *
                FROM research_compute_executions
                ORDER BY started_at DESC
                LIMIT 20
                """
            )
        )
    for run in data:
        run["providers"] = {
            "policy_compute": "LocalFixtureComputeProvider",
            "judge": "DeterministicJudgeFixture",
            "execution": "ComposeExecutionProvider",
        }
        run["cost"] = {
            "execution_credits": round(float(run["verification_run_count"]) * 0.0142, 4),
            "judge_credits": 0,
        }
    return {
        "items": data,
        "research_items": [research_execution_response(item) for item in research_items],
        "next_cursor": None,
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        if not run:
            raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
        attempts = list(
            conn.execute(
                "SELECT * FROM run_attempts WHERE run_id = %s ORDER BY attempt_number",
                (run_id,),
            )
        )
        allocations = list(
            conn.execute(
                """
                SELECT ca.* FROM compute_allocations ca
                JOIN run_attempts ra ON ra.attempt_id = ca.run_attempt_id
                WHERE ra.run_id = %s
                """,
                (run_id,),
            )
        )
        policies = list(
            conn.execute(
                "SELECT * FROM policy_versions WHERE run_id = %s ORDER BY ordinal",
                (run_id,),
            )
        )
        failures = [attempt["failure"] for attempt in attempts if attempt["failure"]]
        metrics = list(
            conn.execute(
                """
                SELECT descriptor, value, unit, subject_id, created_at
                FROM metric_observations WHERE run_id = %s ORDER BY created_at
                """,
                (run_id,),
            )
        )
        rewards = list(
            conn.execute(
                """
                SELECT reward_signal_id, subject_type, subject_id, name, value,
                  reward_pipeline_id, metric_observation_ids, created_at
                FROM reward_signals WHERE run_id = %s ORDER BY created_at
                """,
                (run_id,),
            )
        )
    return {
        "run": run,
        "attempts": attempts,
        "allocations": allocations,
        "policy_versions": policies,
        "metrics": metrics,
        "reward_signals": rewards,
        "failures": failures,
        "providers": {
            "policy_compute": "LocalFixtureComputeProvider",
            "judge": "DeterministicJudgeFixture",
            "execution": "ComposeExecutionProvider",
        },
        "calibration": {
            "status": "FIXTURE_CONTRACT_ONLY",
            "message": "Deterministic fixture assessments are contract fixtures, not human-aligned visual judgments.",
        },
    }


@app.post("/v1/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str, request: CancelRequest) -> dict[str, Any]:
    with connection() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = %s FOR UPDATE", (run_id,)).fetchone()
        if not run:
            raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
        if run["status"] in ("SUCCEEDED", "FAILED", "CANCELED"):
            return {"run_id": run_id, "status": run["status"], "already_terminal": True}
        release_without_worker = run["status"] in ("QUEUED", "PROVISIONING", "PREPARING")
        conn.execute(
            """
            UPDATE runs SET desired_state = 'CANCELED', status = 'CANCEL_REQUESTED',
              version = version + 1, updated_at = now() WHERE run_id = %s
            """,
            (run_id,),
        )
        emit_event(
            conn,
            event_type="run.cancel_requested",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=request.operation_id,
            payload={"mode": request.mode},
        )
    if release_without_worker:
        release_run_resources(run_id)
        return {"run_id": run_id, "status": "CANCELED"}
    return {"run_id": run_id, "status": "CANCEL_REQUESTED"}


@app.post("/v1/runs/{run_id}/reproduce", status_code=202)
def reproduce_run(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        source = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    if not source:
        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
    manifest = source["manifest"]
    request = LaunchRunRequest(
        name=_reproduction_name(source["name"]),
        algorithm=manifest["algorithm"]["id"],
        environment_id=manifest["environment"]["id"],
        policy_compute_provider="LocalFixtureComputeProvider",
        judge_provider="DeterministicJudgeFixture",
        task_revision=manifest["task_revision"],
        branch=BranchConfig(**manifest["branch"]),
        complexity=ComplexityConfig(**manifest.get("complexity", {})),
        budgets=BudgetConfig(**manifest["budgets"]),
        seed=manifest["seed"],
        retention_class="local-research",
    )
    return _launch(request, source_run_id=run_id)


def _reproduction_name(source_name: str) -> str:
    prefix = "Reproduction of "
    base_name = source_name
    while base_name.startswith(prefix):
        base_name = base_name.removeprefix(prefix)
    return f"{prefix}{base_name}"[:80]


def _complexity_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        **state,
        "active_range": active_complexity_range(
            minimum=state["minimum_level"],
            current=state["current_level"],
            sampling_band=state["sampling_band"],
        ),
        "window_progress": {
            "attempts": state["window_attempts"],
            "required": state["evaluation_window"],
            "successes": state["window_successes"],
        },
    }


@app.get("/v1/runs/{run_id}/complexity")
def run_complexity(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        state = conn.execute(
            "SELECT * FROM complexity_states WHERE run_id = %s", (run_id,)
        ).fetchone()
        run = conn.execute(
            "SELECT run_id, manifest FROM runs WHERE run_id = %s", (run_id,)
        ).fetchone()
    if not run:
        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
    if not state:
        return {
            "available": False,
            "run_id": run_id,
            "environment_id": run["manifest"]
            .get("environment", {})
            .get("id", "cad.reconstruction"),
            "reason": "legacy_run",
        }
    return _complexity_payload(state)


@app.post("/internal/runs/{run_id}/complexity-observations")
def observe_complexity(run_id: str, request: ComplexityObservationRequest) -> dict[str, Any]:
    request_digest = canonical_digest(request.model_dump())
    with connection() as conn:
        existing = conn.execute(
            "SELECT request_digest FROM complexity_observations WHERE operation_id = %s",
            (request.operation_id,),
        ).fetchone()
        if existing:
            if existing["request_digest"] != request_digest:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "IDEMPOTENCY_CONFLICT"},
                )
            state = conn.execute(
                "SELECT * FROM complexity_states WHERE run_id = %s", (run_id,)
            ).fetchone()
            return _complexity_payload(state)
        observed_closure = conn.execute(
            """
            SELECT
              run_id,
              level,
              behavior_policy_version_id,
              task_set_digest,
              success_definition,
              ordered_outcomes
            FROM complexity_observations
            WHERE collection_closure_id = %s
            """,
            (request.collection_closure_id,),
        ).fetchone()
        if observed_closure:
            same_observation = (
                observed_closure["run_id"] == run_id
                and observed_closure["level"] == request.level
                and observed_closure["behavior_policy_version_id"]
                == request.behavior_policy_version_id
                and observed_closure["task_set_digest"] == request.task_set_digest
                and observed_closure["success_definition"] == request.success_definition
                and observed_closure["ordered_outcomes"] == request.ordered_outcomes
            )
            if not same_observation:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "COMPLEXITY_COLLECTION_REUSE_CONFLICT"},
                )
            state = conn.execute(
                "SELECT * FROM complexity_states WHERE run_id = %s", (run_id,)
            ).fetchone()
            return _complexity_payload(state)

        state = conn.execute(
            "SELECT * FROM complexity_states WHERE run_id = %s FOR UPDATE", (run_id,)
        ).fetchone()
        if not state:
            raise HTTPException(status_code=404, detail={"code": "COMPLEXITY_STATE_NOT_FOUND"})
        closure = conn.execute(
            """
            SELECT
              cc.collection_closure_id,
              cc.closure_digest,
              cc.member_count,
              cb.behavior_policy_version_id,
              ra.run_id
            FROM collection_closures cc
            JOIN collection_batches cb
              ON cb.collection_batch_id = cc.collection_batch_id
            JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
            WHERE cc.collection_closure_id = %s
            """,
            (request.collection_closure_id,),
        ).fetchone()
        members = (
            list(
                conn.execute(
                    """
                    SELECT
                      cm.ordinal,
                      rt.task_revision,
                      cc.candidate_digest,
                      cc.episode_id,
                      mo.value AS terminal_quality
                    FROM collection_closures closure
                    JOIN collection_memberships cm
                      ON cm.collection_batch_id = closure.collection_batch_id
                    JOIN collection_candidates cc
                      ON cc.collection_candidate_id = cm.collection_candidate_id
                    JOIN rollout_trees rt ON rt.rollout_tree_id = cc.rollout_tree_id
                    JOIN metric_observations mo
                      ON mo.subject_type = 'TRANSITION'
                     AND mo.subject_id = cc.terminal_transition_id
                     AND mo.descriptor = 'deterministic.terminal_quality'
                    WHERE closure.collection_closure_id = %s
                    ORDER BY cm.ordinal
                    """,
                    (request.collection_closure_id,),
                )
            )
            if closure
            else []
        )
        expected_task_set_digest = canonical_digest(
            [
                {
                    "ordinal": member["ordinal"],
                    "task_revision": member["task_revision"],
                    "candidate_digest": member["candidate_digest"],
                    "episode_id": member["episode_id"],
                }
                for member in members
            ]
        )
        expected_outcomes = [float(member["terminal_quality"]) == 1.0 for member in members]
        if (
            not closure
            or closure["run_id"] != run_id
            or closure["behavior_policy_version_id"] != request.behavior_policy_version_id
            or closure["member_count"] != len(request.ordered_outcomes)
            or closure["member_count"] != len(members)
            or request.task_set_digest != expected_task_set_digest
            or request.success_definition != "deterministic.terminal_quality == 1.0"
            or request.ordered_outcomes != expected_outcomes
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "COMPLEXITY_COLLECTION_BINDING_CONFLICT"},
            )
        if request.level != state["current_level"]:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "STALE_COMPLEXITY_LEVEL",
                    "message": (
                        f"Observation level {request.level} does not match "
                        f"current level {state['current_level']}."
                    ),
                },
            )
        update = advance_complexity(
            current_level=state["current_level"],
            maximum_level=state["maximum_level"],
            window_attempts=state["window_attempts"],
            window_successes=state["window_successes"],
            ordered_outcomes=request.ordered_outcomes,
            evaluation_window=state["evaluation_window"],
            mastery_threshold=state["mastery_threshold"],
            promotion_step=state["promotion_step"],
        )
        conn.execute(
            """
            INSERT INTO complexity_observations(
              operation_id, run_id, request_digest, level, successes, attempts, promoted,
              collection_closure_id, behavior_policy_version_id, task_set_digest,
              success_definition, ordered_outcomes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request.operation_id,
                run_id,
                request_digest,
                request.level,
                sum(request.ordered_outcomes),
                len(request.ordered_outcomes),
                update["promoted"],
                request.collection_closure_id,
                request.behavior_policy_version_id,
                request.task_set_digest,
                request.success_definition,
                Jsonb(request.ordered_outcomes),
            ),
        )
        state = conn.execute(
            """
            UPDATE complexity_states SET
              current_level = %s,
              window_attempts = %s,
              window_successes = %s,
              promotion_count = promotion_count + %s,
              last_accuracy = CASE WHEN %s THEN %s ELSE last_accuracy END,
              updated_at = now()
            WHERE run_id = %s
            RETURNING *
            """,
            (
                update["current_level"],
                update["window_attempts"],
                update["window_successes"],
                update["promotion_count"],
                update["evaluated"],
                update["last_accuracy"],
                run_id,
            ),
        ).fetchone()
    return _complexity_payload(state)


@app.post("/v1/runs/{run_id}/rejudge", status_code=202)
def rejudge(run_id: str, request: RejudgeApiRequest) -> dict[str, Any]:
    try:
        new_id = rejudge_transition(
            run_id=run_id,
            verification_run_id=request.verification_run_id,
            fixture_scenario=request.fixture_scenario,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"code": "VERIFICATION_NOT_FOUND"}) from exc
    return {
        "run_id": run_id,
        "verification_run_id": new_id,
        "rejudges_verification_run_id": request.verification_run_id,
        "evidence_reused": True,
    }


@app.post("/internal/agent/claims")
def claim_run(request: WorkerClaimRequest) -> dict[str, Any]:
    claim_id = make_id("claim")
    with connection() as conn:
        attempt = conn.execute(
            """
            SELECT ra.* FROM run_attempts ra
            JOIN runs r ON r.run_id = ra.run_id
            WHERE (
                    ra.status = 'QUEUED'
                    OR (
                      ra.status IN ('PROVISIONING', 'PREPARING', 'RUNNING')
                      AND ra.lease_expires_at < now()
                    )
                  )
              AND ra.next_eligible_at <= now()
              AND r.desired_state = 'RUNNING'
            ORDER BY ra.created_at
            FOR UPDATE OF ra SKIP LOCKED
            LIMIT 1
            """
        ).fetchone()
        if not attempt:
            return {"claim": None}
        conn.execute(
            """
            UPDATE run_attempts
            SET status = 'PROVISIONING',
                claim_id = %s,
                lease_owner = %s,
                lease_expires_at = now() + interval '15 minutes',
                fencing_token = fencing_token + 1,
                heartbeat_at = now(),
                updated_at = now()
            WHERE attempt_id = %s
            """,
            (claim_id, request.worker_id, attempt["attempt_id"]),
        )
        claimed = conn.execute(
            """
            SELECT claim_id, lease_owner, lease_expires_at, fencing_token
            FROM run_attempts WHERE attempt_id = %s
            """,
            (attempt["attempt_id"],),
        ).fetchone()
        conn.execute(
            "UPDATE runs SET status = 'PROVISIONING', updated_at = now() WHERE run_id = %s",
            (attempt["run_id"],),
        )
    return {
        "claim": {
            "attempt_id": attempt["attempt_id"],
            "run_id": attempt["run_id"],
            **claimed,
        }
    }


@app.post("/internal/run-attempts/{attempt_id}/heartbeat")
def heartbeat(attempt_id: str, request: ClaimedAttemptRequest) -> dict[str, Any]:
    with connection() as conn:
        updated = conn.execute(
            """
            UPDATE run_attempts
            SET heartbeat_at = now(),
                lease_expires_at = now() + interval '15 minutes',
                updated_at = now()
            WHERE attempt_id = %s
              AND claim_id = %s
              AND lease_owner = %s
              AND fencing_token = %s
              AND lease_expires_at > now()
              AND status IN ('PROVISIONING', 'PREPARING', 'RUNNING')
            RETURNING run_id, lease_expires_at
            """,
            (
                attempt_id,
                request.claim_id,
                request.worker_id,
                request.fencing_token,
            ),
        ).fetchone()
    if not updated:
        raise HTTPException(status_code=409, detail={"code": "STALE_ATTEMPT_CLAIM"})
    return {
        "attempt_id": attempt_id,
        "heartbeat_at": utc_now(),
        "lease_expires_at": updated["lease_expires_at"],
    }


@app.post("/internal/run-attempts/{attempt_id}/execute")
def execute_claim(attempt_id: str, request: ClaimedAttemptRequest) -> dict[str, Any]:
    execute_run_attempt(
        attempt_id,
        worker_id=request.worker_id,
        claim_id=request.claim_id,
        fencing_token=request.fencing_token,
    )
    return {"attempt_id": attempt_id, "status": "TERMINAL"}


@app.get("/v1/runs/{run_id}/collection-batches")
def collection_batches(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT cb.*,
                  (SELECT count(*) FROM rollout_trees rt
                   WHERE rt.collection_batch_id = cb.collection_batch_id) AS tree_count
                FROM collection_batches cb
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s ORDER BY cb.created_at
                """,
                (run_id,),
            )
        )
    return {"items": [research_execution_response(item) for item in items]}


@app.get("/v1/runs/{run_id}/iterations")
def iterations(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT ti.*, ii.digest AS iteration_input_digest,
                  (SELECT count(*) FROM rollout_trees rt
                   WHERE rt.collection_batch_id = ti.collection_batch_id) AS tree_count,
                  (SELECT count(*) FROM eligibility_decisions ed
                   JOIN rollout_trees rt ON rt.rollout_tree_id = ed.rollout_tree_id
                   WHERE rt.collection_batch_id = ti.collection_batch_id
                     AND ed.status = 'EXCLUDED') AS excluded_count
                FROM training_iterations ti
                LEFT JOIN iteration_inputs ii ON ii.manifest_id = ti.iteration_input_id
                WHERE ti.run_id = %s ORDER BY ti.created_at
                """,
                (run_id,),
            )
        )
    return {"items": items}


@app.get("/v1/runs/{run_id}/rollout-trees")
def run_rollout_trees(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT rt.*,
                  cb.status AS collection_status,
                  cb.created_at AS collection_created_at,
                  (SELECT count(*) FROM states s
                   WHERE s.rollout_tree_id = rt.rollout_tree_id) AS state_count,
                  (SELECT count(*) FROM transitions t
                   WHERE t.rollout_tree_id = rt.rollout_tree_id) AS transition_count,
                  (SELECT count(*) FROM branch_members bm
                   JOIN branch_groups bg ON bg.branch_group_id = bm.branch_group_id
                   WHERE bg.rollout_tree_id = rt.rollout_tree_id) AS sibling_count,
                  (SELECT count(*) FROM eligibility_decisions ed
                   WHERE ed.rollout_tree_id = rt.rollout_tree_id
                     AND ed.status = 'EXCLUDED') AS excluded_count,
                  (SELECT count(*) FROM branch_members bm
                   JOIN branch_groups bg ON bg.branch_group_id = bm.branch_group_id
                   WHERE bg.rollout_tree_id = rt.rollout_tree_id
                     AND (bm.failure_mode IS NOT NULL OR bm.status <> 'SUCCEEDED'))
                    AS exception_count
                FROM rollout_trees rt
                JOIN collection_batches cb
                  ON cb.collection_batch_id = rt.collection_batch_id
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s
                ORDER BY rt.created_at
                """,
                (run_id,),
            )
        )
    return {"items": items}


@app.get("/v1/iterations/{iteration_id}")
def iteration_detail(iteration_id: str) -> dict[str, Any]:
    with connection() as conn:
        iteration = conn.execute(
            """
            SELECT ti.*, ii.manifest AS iteration_input_manifest,
              ii.digest AS iteration_input_digest
            FROM training_iterations ti
            LEFT JOIN iteration_inputs ii ON ii.manifest_id = ti.iteration_input_id
            WHERE ti.training_iteration_id = %s
            """,
            (iteration_id,),
        ).fetchone()
        if not iteration:
            raise HTTPException(status_code=404, detail={"code": "ITERATION_NOT_FOUND"})
        trees = list(
            conn.execute(
                """
                SELECT rt.* FROM rollout_trees rt
                WHERE rt.collection_batch_id = %s ORDER BY rt.created_at
                """,
                (iteration["collection_batch_id"],),
            )
        )
        eligibility = list(
            conn.execute(
                """
                SELECT ed.* FROM eligibility_decisions ed
                JOIN rollout_trees rt ON rt.rollout_tree_id = ed.rollout_tree_id
                WHERE rt.collection_batch_id = %s ORDER BY ed.created_at
                """,
                (iteration["collection_batch_id"],),
            )
        )
    return {"iteration": iteration, "rollout_trees": trees, "eligibility_decisions": eligibility}


@app.get("/v1/iterations/{iteration_id}/rollout-trees")
def iteration_trees(iteration_id: str) -> dict[str, Any]:
    detail = iteration_detail(iteration_id)
    decisions_by_tree: dict[str, list[dict[str, Any]]] = {}
    for decision in detail["eligibility_decisions"]:
        decisions_by_tree.setdefault(decision["rollout_tree_id"], []).append(decision)
    return {
        "items": [
            {**tree, "eligibility_decisions": decisions_by_tree.get(tree["rollout_tree_id"], [])}
            for tree in detail["rollout_trees"]
        ],
        "complete_set": True,
    }


@app.get("/v1/rollout-trees/{tree_id}")
def rollout_tree(tree_id: str) -> dict[str, Any]:
    with connection() as conn:
        tree = conn.execute(
            "SELECT * FROM rollout_trees WHERE rollout_tree_id = %s", (tree_id,)
        ).fetchone()
        if not tree:
            raise HTTPException(status_code=404, detail={"code": "TREE_NOT_FOUND"})
        batch = conn.execute(
            "SELECT * FROM collection_batches WHERE collection_batch_id = %s",
            (tree["collection_batch_id"],),
        ).fetchone()
    return {"tree": tree, "collection_batch": batch}


@app.get("/v1/rollout-trees/{tree_id}/graph")
def rollout_tree_graph(tree_id: str) -> dict[str, Any]:
    with connection() as conn:
        tree = conn.execute(
            """
            SELECT rt.*, ra.run_id, r.name AS run_name, r.algorithm
            FROM rollout_trees rt
            JOIN collection_batches cb
              ON cb.collection_batch_id = rt.collection_batch_id
            JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
            JOIN runs r ON r.run_id = ra.run_id
            WHERE rt.rollout_tree_id = %s
            """,
            (tree_id,),
        ).fetchone()
        if not tree:
            raise HTTPException(status_code=404, detail={"code": "TREE_NOT_FOUND"})
        states = list(
            conn.execute(
                "SELECT * FROM states WHERE rollout_tree_id = %s ORDER BY sequence, created_at",
                (tree_id,),
            )
        )
        transitions = list(
            conn.execute(
                """
                SELECT t.*,
                  (SELECT vr.verification_run_id FROM verification_runs vr
                   WHERE vr.subject_id = t.transition_id AND vr.rejudges_verification_run_id IS NULL
                   ORDER BY vr.created_at LIMIT 1) AS verification_run_id,
                  (SELECT vr.proof_bundle_id FROM verification_runs vr
                   WHERE vr.subject_id = t.transition_id AND vr.rejudges_verification_run_id IS NULL
                   ORDER BY vr.created_at LIMIT 1) AS proof_bundle_id
                FROM transitions t WHERE t.rollout_tree_id = %s
                ORDER BY t.created_at
                """,
                (tree_id,),
            )
        )
        branch_groups = list(
            conn.execute(
                "SELECT * FROM branch_groups WHERE rollout_tree_id = %s ORDER BY created_at",
                (tree_id,),
            )
        )
        branch_members = list(
            conn.execute(
                """
                SELECT bm.*,
                  eligibility.status AS eligibility_status,
                  eligibility.reason_code AS eligibility_reason,
                  (SELECT t.outcome FROM transitions t
                   WHERE t.branch_member_id = bm.branch_member_id
                   ORDER BY t.created_at DESC LIMIT 1) AS terminal_outcome,
                  (SELECT vr.status FROM transitions t
                   JOIN verification_runs vr ON vr.subject_id = t.transition_id
                     AND vr.rejudges_verification_run_id IS NULL
                   WHERE t.branch_member_id = bm.branch_member_id
                   ORDER BY t.created_at DESC, vr.created_at LIMIT 1)
                    AS verification_status,
                  (SELECT count(*) FROM transitions t
                   JOIN verification_runs vr ON vr.subject_id = t.transition_id
                     AND vr.rejudges_verification_run_id IS NULL
                   JOIN verifier_step_runs vsr
                     ON vsr.verification_run_id = vr.verification_run_id
                   WHERE t.branch_member_id = bm.branch_member_id
                     AND vsr.attempt_count > 1) AS retry_count
                FROM branch_members bm
                JOIN branch_groups bg ON bg.branch_group_id = bm.branch_group_id
                LEFT JOIN LATERAL (
                  SELECT ed.status, ed.reason_code
                  FROM eligibility_decisions ed
                  WHERE ed.branch_member_id = bm.branch_member_id
                  ORDER BY ed.created_at DESC LIMIT 1
                ) eligibility ON true
                WHERE bg.rollout_tree_id = %s ORDER BY bm.sibling_index
                """,
                (tree_id,),
            )
        )
        checkpoints = list(
            conn.execute(
                "SELECT * FROM decision_checkpoints WHERE rollout_tree_id = %s ORDER BY created_at",
                (tree_id,),
            )
        )
        snapshots = list(
            conn.execute(
                """
                SELECT es.* FROM environment_snapshots es
                JOIN decision_checkpoints dc ON dc.snapshot_id = es.snapshot_id
                WHERE dc.rollout_tree_id = %s ORDER BY es.created_at
                """,
                (tree_id,),
            )
        )
    return {
        "tree": tree,
        "nodes": [
            {
                "id": state["state_id"],
                "type": "state",
                "sequence": state["sequence"],
                "semantic_status": state["semantic_status"],
                "payload": state["payload"],
                "created_at": state["created_at"],
            }
            for state in states
        ],
        "edges": [
            {
                "id": transition["transition_id"],
                "source": transition["source_state_id"],
                "target": transition["destination_state_id"],
                "branch_member_id": transition["branch_member_id"],
                "outcome": transition["outcome"],
                "action": transition["payload"]["action"],
                "verification_run_id": transition["verification_run_id"],
                "proof_bundle_id": transition["proof_bundle_id"],
                "operation_id": transition["operation_id"],
                "action_artifact_id": transition["action_artifact_id"],
                "runtime_cursor_id": transition["runtime_cursor_id"],
                "cursor_version": transition["cursor_version"],
                "created_at": transition["created_at"],
            }
            for transition in transitions
        ],
        "branch_groups": branch_groups,
        "branch_members": branch_members,
        "decision_checkpoints": checkpoints,
        "environment_snapshots": snapshots,
        "accessible_outline": [
            {
                "state_id": state["state_id"],
                "sequence": state["sequence"],
                "parent_transition_id": state["parent_transition_id"],
                "semantic_status": state["semantic_status"],
            }
            for state in states
        ],
    }


@app.get("/v1/states/{state_id}")
def state_detail(state_id: str) -> dict[str, Any]:
    with connection() as conn:
        state = conn.execute("SELECT * FROM states WHERE state_id = %s", (state_id,)).fetchone()
        if not state:
            raise HTTPException(status_code=404, detail={"code": "STATE_NOT_FOUND"})
        artifacts = list(
            conn.execute(
                """
                SELECT a.*, ar.role, ar.viewer_hint, ar.visibility, ar.trust_class
                FROM artifact_refs ar JOIN artifacts a ON a.artifact_id = ar.artifact_id
                WHERE ar.entity_type = 'state' AND ar.entity_id = %s ORDER BY ar.ordinal
                """,
                (state_id,),
            )
        )
    return {"state": state, "artifacts": artifacts}


@app.get("/v1/transitions/{transition_id}")
def transition_detail(transition_id: str) -> dict[str, Any]:
    with connection() as conn:
        transition = conn.execute(
            "SELECT * FROM transitions WHERE transition_id = %s",
            (transition_id,),
        ).fetchone()
        if not transition:
            raise HTTPException(status_code=404, detail={"code": "TRANSITION_NOT_FOUND"})
        verifications = list(
            conn.execute(
                """
                SELECT * FROM verification_runs WHERE subject_id = %s ORDER BY created_at
                """,
                (transition_id,),
            )
        )
        rewards = list(
            conn.execute(
                "SELECT * FROM reward_signals WHERE subject_id = %s ORDER BY created_at",
                (transition_id,),
            )
        )
        metrics = list(
            conn.execute(
                "SELECT * FROM metric_observations WHERE subject_id = %s ORDER BY created_at",
                (transition_id,),
            )
        )
    return {
        "transition": transition,
        "verification_runs": verifications,
        "metric_observations": metrics,
        "reward_signals": rewards,
    }


@app.get("/v1/branch-groups/{branch_group_id}")
def branch_group(branch_group_id: str) -> dict[str, Any]:
    with connection() as conn:
        group = conn.execute(
            "SELECT * FROM branch_groups WHERE branch_group_id = %s",
            (branch_group_id,),
        ).fetchone()
        if not group:
            raise HTTPException(status_code=404, detail={"code": "BRANCH_GROUP_NOT_FOUND"})
        members = list(
            conn.execute(
                "SELECT * FROM branch_members WHERE branch_group_id = %s ORDER BY sibling_index",
                (branch_group_id,),
            )
        )
    return {"branch_group": group, "members": members}


@app.get("/v1/branch-groups/{branch_group_id}/comparison")
def branch_comparison(branch_group_id: str) -> dict[str, Any]:
    with connection() as conn:
        group = conn.execute(
            "SELECT * FROM branch_groups WHERE branch_group_id = %s",
            (branch_group_id,),
        ).fetchone()
        if not group:
            raise HTTPException(status_code=404, detail={"code": "BRANCH_GROUP_NOT_FOUND"})
        members = list(
            conn.execute(
                """
                SELECT bm.*,
                  (SELECT t.transition_id FROM transitions t
                   WHERE t.branch_member_id = bm.branch_member_id
                   ORDER BY t.created_at DESC LIMIT 1) AS terminal_transition_id
                FROM branch_members bm WHERE bm.branch_group_id = %s
                ORDER BY bm.sibling_index
                """,
                (branch_group_id,),
            )
        )
        for member in members:
            terminal = member["terminal_transition_id"]
            if not terminal:
                member["proof_bundle"] = None
                continue
            member["proof_bundle"] = conn.execute(
                """
                SELECT eb.manifest FROM evidence_bundles eb
                JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
                WHERE vr.subject_id = %s ORDER BY vr.created_at DESC LIMIT 1
                """,
                (terminal,),
            ).fetchone()["manifest"]
            member["reward_signals"] = list(
                conn.execute(
                    """
                    SELECT * FROM reward_signals
                    WHERE subject_id IN (%s, %s) ORDER BY created_at
                    """,
                    (terminal, member["branch_member_id"]),
                )
            )
        judgment = conn.execute(
            """
            SELECT jr.result, eb.manifest FROM judge_results jr
            JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
            JOIN verification_runs vr ON vr.verification_run_id = ji.verification_run_id
            JOIN evidence_bundles eb ON eb.proof_bundle_id = vr.proof_bundle_id
            WHERE vr.subject_type = 'BRANCH_GROUP' AND vr.subject_id = %s
            ORDER BY jr.created_at DESC LIMIT 1
            """,
            (branch_group_id,),
        ).fetchone()
    return {
        "branch_group": group,
        "members": members,
        "model_assessment": judgment["result"] if judgment else None,
        "presentation_order": group["presentation_order"],
        "member_bindings": judgment["manifest"].get("member_bindings", []) if judgment else [],
    }


@app.get("/v1/verification-runs/{verification_run_id}")
def verification_detail(verification_run_id: str) -> dict[str, Any]:
    with connection() as conn:
        verification = conn.execute(
            "SELECT * FROM verification_runs WHERE verification_run_id = %s",
            (verification_run_id,),
        ).fetchone()
        if not verification:
            raise HTTPException(status_code=404, detail={"code": "VERIFICATION_NOT_FOUND"})
        proof = conn.execute(
            "SELECT * FROM evidence_bundles WHERE proof_bundle_id = %s",
            (verification["proof_bundle_id"],),
        ).fetchone()
        steps = list(
            conn.execute(
                """
                SELECT * FROM verifier_step_runs
                WHERE verification_run_id = %s ORDER BY started_at, step_id
                """,
                (verification_run_id,),
            )
        )
        judge = conn.execute(
            """
            SELECT jr.result, ji.*, js.spec
            FROM judge_invocations ji
            JOIN judge_results jr ON jr.judge_invocation_id = ji.judge_invocation_id
            JOIN judge_specs js ON js.judge_spec_id = ji.judge_spec_id
            WHERE ji.verification_run_id = %s ORDER BY ji.created_at DESC LIMIT 1
            """,
            (verification_run_id,),
        ).fetchone()
    return {
        "verification_run": verification,
        "proof_bundle": proof,
        "steps": steps,
        "judge": judge,
        "model_assessment_notice": (
            "Deterministic fixture output is contract evidence, not objective visual truth."
        ),
    }


@app.get("/v1/verification-runs/{verification_run_id}/graph")
def verification_graph(verification_run_id: str) -> dict[str, Any]:
    detail = verification_detail(verification_run_id)
    steps = detail["steps"]
    dependency_map = {
        "geometry": [],
        "render": [],
        "proof": ["geometry", "render"],
        "pointwise-judge": ["proof"],
        "aggregate": ["geometry", "pointwise-judge"],
        "group-judge": [],
    }
    return {
        **detail,
        "nodes": [
            {
                "id": step["step_run_id"],
                "step_id": step["step_id"],
                "type": step["step_type"],
                "status": step["status"],
                "attempt_count": step["attempt_count"],
                "cache_status": step["cache_status"],
                "evidence_roles": step["evidence_roles"],
                "cost": step["cost"],
            }
            for step in steps
        ],
        "edges": [
            {
                "id": f"{dependency}->{step['step_id']}",
                "source_step_id": dependency,
                "target_step_id": step["step_id"],
            }
            for step in steps
            for dependency in dependency_map.get(step["step_id"], [])
        ],
        "accessible_outline": [
            {
                "position": index + 1,
                "step": step["step_id"],
                "depends_on": dependency_map.get(step["step_id"], []),
                "status": step["status"],
            }
            for index, step in enumerate(steps)
        ],
    }


@app.get("/v1/verifier-step-runs/{step_run_id}")
def verifier_step(step_run_id: str) -> dict[str, Any]:
    with connection() as conn:
        step = conn.execute(
            "SELECT * FROM verifier_step_runs WHERE step_run_id = %s",
            (step_run_id,),
        ).fetchone()
    if not step:
        raise HTTPException(status_code=404, detail={"code": "VERIFIER_STEP_NOT_FOUND"})
    return {"step": step}


@app.get("/v1/evidence-bundles/{proof_bundle_id}")
def evidence_bundle(proof_bundle_id: str) -> dict[str, Any]:
    with connection() as conn:
        bundle = conn.execute(
            "SELECT * FROM evidence_bundles WHERE proof_bundle_id = %s",
            (proof_bundle_id,),
        ).fetchone()
        if not bundle:
            raise HTTPException(status_code=404, detail={"code": "EVIDENCE_BUNDLE_NOT_FOUND"})
        artifacts = list(
            conn.execute(
                """
                SELECT a.*, ar.role, ar.ordinal, ar.viewer_hint, ar.visibility, ar.trust_class
                FROM artifact_refs ar JOIN artifacts a ON a.artifact_id = ar.artifact_id
                WHERE (ar.entity_type = 'evidence_bundle' AND ar.entity_id = %s)
                   OR (ar.entity_type = 'verification_run' AND ar.entity_id = %s)
                ORDER BY ar.role, ar.ordinal
                """,
                (proof_bundle_id, bundle["verification_run_id"]),
            )
        )
    visible = [artifact for artifact in artifacts if artifact["visibility"] != "HIDDEN"]
    return {
        "evidence_bundle": bundle,
        "artifacts": visible,
        "hidden_evidence": len(artifacts) != len(visible),
    }


@app.get("/v1/judge-results/{judge_result_id}")
def judge_result(judge_result_id: str) -> dict[str, Any]:
    with connection() as conn:
        result = conn.execute(
            """
            SELECT jr.*, ji.judge_spec_id, ji.proof_bundle_digest, ji.sample_index,
              ji.usage, js.spec
            FROM judge_results jr
            JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
            JOIN judge_specs js ON js.judge_spec_id = ji.judge_spec_id
            WHERE jr.judge_result_id = %s
            """,
            (judge_result_id,),
        ).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail={"code": "JUDGE_RESULT_NOT_FOUND"})
    return {
        "judge_result": result,
        "label": "Fixture assessment",
        "calibration_status": "FIXTURE_CONTRACT_ONLY",
        "hidden_chain_of_thought_stored": False,
    }


@app.get("/v1/artifacts/{artifact_id}")
def artifact(artifact_id: str) -> Response:
    with connection() as conn:
        item = conn.execute(
            """
            SELECT a.*, bool_or(ar.visibility = 'HIDDEN') AS hidden,
              bool_or(ar.trust_class = 'CANDIDATE') AS candidate
            FROM artifacts a JOIN artifact_refs ar ON ar.artifact_id = a.artifact_id
            WHERE a.artifact_id = %s GROUP BY a.artifact_id
            """,
            (artifact_id,),
        ).fetchone()
    if not item:
        raise HTTPException(status_code=404, detail={"code": "ARTIFACT_NOT_FOUND"})
    if item["hidden"]:
        raise HTTPException(status_code=403, detail={"code": "HIDDEN_EVIDENCE"})
    data = artifact_store.get_bytes(item["digest"])
    headers = {
        "ETag": f'"{item["digest"]}"',
        "Cache-Control": "public, immutable, max-age=31536000",
        "X-Content-Type-Options": "nosniff",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    if item["candidate"] or item["media_type"] == "image/svg+xml":
        headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"
        )
    return Response(
        content=data,
        media_type=item["media_type"],
        headers=headers,
    )


@app.get("/v1/runs/{run_id}/events")
def events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT o.outbox_id AS cursor, e.*
                FROM outbox o JOIN events e ON e.event_id = o.event_id
                WHERE e.run_id = %s AND o.outbox_id > %s
                ORDER BY o.outbox_id LIMIT %s
                """,
                (run_id, after, limit),
            )
        )
        if items:
            conn.execute(
                """
                UPDATE outbox
                SET published_at = COALESCE(published_at, now())
                WHERE outbox_id = ANY(%s)
                """,
                ([item["cursor"] for item in items],),
            )
    return {
        "items": items,
        "next_cursor": items[-1]["cursor"] if items else after,
    }


@app.get("/v1/runs/{run_id}/stream")
async def event_stream(run_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
    async def generate():
        cursor = after
        idle_cycles = 0
        while idle_cycles < 30:
            page = events(run_id, after=cursor, limit=100)
            if page["items"]:
                idle_cycles = 0
                for item in page["items"]:
                    cursor = item["cursor"]
                    yield f"id: {cursor}\nevent: {item['event_type']}\ndata: {json.dumps(item, default=str)}\n\n"
            else:
                idle_cycles += 1
                yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/v1/research-compute-executions")
def list_research_compute_executions() -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT *
                FROM research_compute_executions
                ORDER BY started_at DESC
                LIMIT 100
                """
            )
        )
    return {"items": items}


@app.get("/v1/studies")
def list_research_studies() -> dict[str, Any]:
    return {"items": list_study_summaries()}


@app.get("/v1/studies/{study_id}")
def get_research_study(study_id: str) -> dict[str, Any]:
    report = find_study_report(study_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "RESEARCH_STUDY_NOT_FOUND"},
        )
    return report


@app.get("/v1/research-compute-executions/{execution_id}")
def get_research_compute_execution(execution_id: str) -> dict[str, Any]:
    with connection() as conn:
        item = conn.execute(
            """
            SELECT *
            FROM research_compute_executions
            WHERE execution_id = %s
            """,
            (execution_id,),
        ).fetchone()
    if not item:
        raise HTTPException(
            status_code=404,
            detail={"code": "RESEARCH_EXECUTION_NOT_FOUND"},
        )
    return research_execution_response(item)


@app.get("/v1/research-compute-executions/{execution_id}/trajectory")
def get_research_compute_trajectory(execution_id: str) -> dict[str, Any]:
    with connection() as conn:
        item = conn.execute(
            """
            SELECT e.*, p.result AS proof_result
            FROM research_compute_executions e
            LEFT JOIN research_compute_proofs p ON p.proof_id = e.proof_id
            WHERE e.execution_id = %s
            """,
            (execution_id,),
        ).fetchone()
    if not item:
        raise HTTPException(
            status_code=404,
            detail={"code": "RESEARCH_EXECUTION_NOT_FOUND"},
        )
    execution = dict(item)
    proof_result = execution.pop("proof_result")
    trajectory_source = (
        proof_result if isinstance(proof_result, dict) else execution.get("progress")
    )
    return {
        "execution": execution,
        "trajectory": (
            research_trajectory(trajectory_source) if isinstance(trajectory_source, dict) else None
        ),
    }


@app.put("/internal/research-compute-executions/{execution_id}")
def update_research_compute_execution(
    execution_id: str,
    request: ResearchComputeExecutionRequest,
) -> dict[str, Any]:
    if not execution_id.startswith("runpod-proof-"):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_RESEARCH_EXECUTION_ID"},
        )
    terminal = request.status in {"SUCCEEDED", "FAILED"}
    if terminal != (request.completed_at is not None):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_RESEARCH_EXECUTION_TERMINAL_STATE"},
        )
    if request.status == "SUCCEEDED" and not request.teardown_confirmed:
        raise HTTPException(
            status_code=422,
            detail={"code": "RESEARCH_EXECUTION_TEARDOWN_UNCONFIRMED"},
        )
    if request.status in {"RUNNING", "FINALIZING", "SUCCEEDED"} and not request.provider_handle:
        raise HTTPException(
            status_code=422,
            detail={"code": "RESEARCH_EXECUTION_PROVIDER_HANDLE_REQUIRED"},
        )

    status_order = {
        "PROVISIONING": 0,
        "RUNNING": 1,
        "FINALIZING": 2,
        "SUCCEEDED": 3,
        "FAILED": 3,
    }
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT *
            FROM research_compute_executions
            WHERE execution_id = %s
            """,
            (execution_id,),
        ).fetchone()
        if existing:
            immutable_conflict = (
                existing["name"] != request.name
                or existing["workload_id"] != request.workload_id
                or existing["model_id"] != request.model_id
                or existing["started_at"] != request.started_at
            )
            if immutable_conflict:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "RESEARCH_EXECUTION_CONFLICT"},
                )
            if existing["status"] in {"SUCCEEDED", "FAILED"} and (
                existing["status"] != request.status
                or existing["teardown_confirmed"] != request.teardown_confirmed
            ):
                raise HTTPException(
                    status_code=409,
                    detail={"code": "RESEARCH_EXECUTION_TERMINAL"},
                )
            if status_order[request.status] < status_order[existing["status"]]:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "RESEARCH_EXECUTION_STATUS_REGRESSION"},
                )

        row = conn.execute(
            """
            INSERT INTO research_compute_executions(
              execution_id, name, workload_id, model_id, branch_width,
              complexity_strategy, status, provider_name, provider_handle,
              resource_profile, progress, started_at, completed_at,
              teardown_confirmed
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (execution_id) DO UPDATE SET
              status = EXCLUDED.status,
              provider_handle = COALESCE(
                EXCLUDED.provider_handle,
                research_compute_executions.provider_handle
              ),
              resource_profile = (
                research_compute_executions.resource_profile
                || EXCLUDED.resource_profile
              ),
              progress = (
                research_compute_executions.progress
                || EXCLUDED.progress
              ),
              completed_at = EXCLUDED.completed_at,
              teardown_confirmed = EXCLUDED.teardown_confirmed,
              updated_at = now()
            RETURNING *
            """,
            (
                execution_id,
                request.name,
                request.workload_id,
                request.model_id,
                request.branch_width,
                request.complexity_strategy,
                request.status,
                request.provider_name,
                request.provider_handle,
                Jsonb(request.resource_profile),
                Jsonb(request.progress),
                request.started_at,
                request.completed_at,
                request.teardown_confirmed,
            ),
        ).fetchone()
    return row


@app.get("/v1/resources")
def resources() -> dict[str, Any]:
    with connection() as conn:
        allocations = list(
            conn.execute("SELECT * FROM compute_allocations ORDER BY created_at DESC LIMIT 100")
        )
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM compute_allocations WHERE observed_state != 'RELEASED') AS allocations,
              (SELECT count(*) FROM environment_snapshots) AS snapshots,
              (SELECT count(*) FROM decision_checkpoints) AS checkpoints,
              (SELECT count(*) FROM operations WHERE status = 'AUTHORIZED') AS pending_operations
            """
        ).fetchone()
        research_compute_proofs = list(
            conn.execute(
                """
                SELECT * FROM research_compute_proofs
                ORDER BY completed_at DESC
                LIMIT 20
                """
            )
        )
        research_compute_executions = list(
            conn.execute(
                """
                SELECT *
                FROM research_compute_executions
                ORDER BY started_at DESC
                LIMIT 20
                """
            )
        )
    return {
        "allocations": allocations,
        "counts": counts,
        "research_compute_proofs": research_compute_proofs,
        "research_compute_executions": research_compute_executions,
        "provider_boundaries": {
            "policy_compute": ["LocalFixtureComputeProvider"],
            "judge": ["DeterministicJudgeFixture"],
            "execution": ["ComposeExecutionProvider"],
        },
        "external_capacity": 0,
    }


@app.get("/v1/proofs")
def list_research_compute_proofs() -> dict[str, Any]:
    with connection() as conn:
        items = list(
            conn.execute(
                """
                SELECT p.*, e.execution_id, e.name AS execution_name
                FROM research_compute_proofs p
                LEFT JOIN research_compute_executions e ON e.proof_id = p.proof_id
                ORDER BY p.completed_at DESC
                LIMIT 100
                """
            )
        )
    return {"items": [research_proof_response(item, detail=False) for item in items]}


@app.get("/v1/proofs/{proof_id}")
def get_research_compute_proof(proof_id: str) -> dict[str, Any]:
    with connection() as conn:
        item = conn.execute(
            """
            SELECT p.*, e.execution_id, e.name AS execution_name
            FROM research_compute_proofs p
            LEFT JOIN research_compute_executions e ON e.proof_id = p.proof_id
            WHERE p.proof_id = %s
            """,
            (proof_id,),
        ).fetchone()
    if not item:
        raise HTTPException(
            status_code=404,
            detail={"code": "RESEARCH_PROOF_NOT_FOUND"},
        )
    return research_proof_response(item, detail=True)


@app.post("/internal/research-compute-proofs", status_code=201)
def ingest_research_compute_proof(
    request: ResearchComputeProofRequest,
) -> dict[str, Any]:
    receipt = request.model_dump(mode="json")
    receipt_digest = canonical_digest(receipt)
    result_progress = research_result_progress(request.result)
    with connection() as conn:
        execution = conn.execute(
            """
            SELECT *
            FROM research_compute_executions
            WHERE provider_handle = %s
            """,
            (request.provider_handle,),
        ).fetchone()
        recovery_execution_id = (
            execution["execution_id"]
            if execution and research_proof_can_recover_execution(dict(execution), request)
            else ""
        )
        existing = conn.execute(
            """
            SELECT proof_id, receipt_digest
            FROM research_compute_proofs
            WHERE provider_handle = %s
            """,
            (request.provider_handle,),
        ).fetchone()
        if existing:
            if existing["receipt_digest"] != receipt_digest:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "RESEARCH_PROOF_CONFLICT"},
                )
            conn.execute(
                """
                UPDATE research_compute_executions SET
                  status = 'SUCCEEDED',
                  progress = (progress - 'error') || %s,
                  proof_id = %s,
                  receipt_digest = %s,
                  completed_at = %s,
                  teardown_confirmed = true,
                  updated_at = now()
                WHERE provider_handle = %s
                  AND (status != 'FAILED' OR execution_id = %s)
                """,
                (
                    Jsonb(result_progress),
                    existing["proof_id"],
                    receipt_digest,
                    request.completed_at,
                    request.provider_handle,
                    recovery_execution_id,
                ),
            )
            return {
                "proof_id": existing["proof_id"],
                "receipt_digest": receipt_digest,
                "already_recorded": True,
            }

        proof_id = make_id("research_proof")
        conn.execute(
            """
            INSERT INTO research_compute_proofs(
              proof_id, provider_name, provider_handle, provider_cli_version,
              resource_profile, workload, result, receipt_digest, started_at,
              completed_at, teardown_confirmed
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                proof_id,
                request.provider_name,
                request.provider_handle,
                request.provider_cli_version,
                Jsonb(request.resource_profile),
                Jsonb(request.workload),
                Jsonb(request.result),
                receipt_digest,
                request.started_at,
                request.completed_at,
                request.teardown_confirmed,
            ),
        )
        conn.execute(
            """
            UPDATE research_compute_executions SET
              status = 'SUCCEEDED',
              progress = (progress - 'error') || %s,
              proof_id = %s,
              receipt_digest = %s,
              completed_at = %s,
              teardown_confirmed = true,
              updated_at = now()
            WHERE provider_handle = %s
              AND (status != 'FAILED' OR execution_id = %s)
            """,
            (
                Jsonb(result_progress),
                proof_id,
                receipt_digest,
                request.completed_at,
                request.provider_handle,
                recovery_execution_id,
            ),
        )
    return {
        "proof_id": proof_id,
        "receipt_digest": receipt_digest,
        "already_recorded": False,
    }
