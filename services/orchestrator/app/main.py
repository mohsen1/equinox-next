from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from equinox_core import canonical_digest, make_id, utc_now
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from .database import connection, migrate
from .environments import (
    active_complexity_range,
    advance_complexity,
    environment_catalog,
    environment_spec,
)
from .providers import POLICY_COMPUTE_PROVIDERS, assert_local_registry
from .science import artifact_store, emit_event
from .workflow import execute_run_attempt, rejudge_transition, release_run_resources


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BranchConfig(StrictModel):
    mode: Literal["static"] = "static"
    width: Literal[1, 4]
    decision_after_actions: int = Field(default=3, ge=1, le=8)
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
    environment_id: str = Field(default="cad.reconstruction", min_length=3, max_length=80)
    policy_compute_provider: Literal["MockRunPodProvider"] = "MockRunPodProvider"
    judge_provider: Literal["MockJudgeProvider"] = "MockJudgeProvider"
    task_revision: str = Field(
        default="mounting-plate@sha256:fixture-v1", min_length=3, max_length=160
    )
    branch: BranchConfig
    complexity: ComplexityConfig = ComplexityConfig()
    budgets: BudgetConfig = BudgetConfig()
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    retention_class: Literal["local-research"] = "local-research"


class CancelRequest(StrictModel):
    mode: Literal["terminate"] = "terminate"
    operation_id: str


class RejudgeApiRequest(StrictModel):
    verification_run_id: str
    fixture_scenario: Literal[
        "valid", "low", "tie", "abstain", "malformed", "retry", "disagreement", "integrity"
    ] = "valid"


class ComplexityObservationRequest(StrictModel):
    operation_id: str = Field(min_length=3, max_length=160)
    level: int = Field(ge=0, le=100)
    successes: int = Field(ge=0)
    attempts: int = Field(gt=0)


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
    branch_width: Literal[4] = 4
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


def research_result_progress(result: dict[str, Any]) -> dict[str, Any]:
    level = result.get("reached_complexity_level")
    final_by_level = result.get("final_by_level")
    level_result: dict[str, Any] = {}
    if isinstance(final_by_level, dict) and level is not None:
        candidate = final_by_level.get(str(level))
        if isinstance(candidate, dict):
            level_result = candidate

    values = {
        "phase": "complete",
        "message": (
            "Training and final evaluation completed; artifacts persisted "
            "and provider teardown confirmed."
        ),
        "update": result.get("updates_completed"),
        "current_level": level,
        "promotion_count": result.get("promotion_count"),
        "sampled_completions": result.get("total_sampled_completions"),
        "exact_rate": level_result.get("exact_rate", result.get("final_reward")),
        "initial_exact_rate": result.get("initial_reward"),
        "final_exact_rate": result.get("final_reward"),
        "reward_gain": result.get("reward_gain"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "stop_reason": result.get("stop_reason"),
        "hypothesis_passed": result.get("hypothesis_passed"),
        "adapter_persisted": result.get("adapter_persisted"),
        "post_training_completed": result.get("post_training_completed"),
        "informative_group_rate": result.get("informative_group_rate"),
        "teacher_fallback_rate": result.get("teacher_fallback_rate"),
        "policy_update_count": result.get("policy_update_count"),
        "teacher_update_count": result.get("teacher_update_count"),
    }
    return {key: value for key, value in values.items() if value is not None}


def research_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    initial_by_level = result.get("initial_by_level")
    final_by_level = result.get("final_by_level")
    history = result.get("history")
    promotions = result.get("promotions")
    branch_snapshots = result.get("branch_snapshots")
    return {
        "schema_version": 1,
        "branch_width": result.get("branch_width"),
        "complexity_strategy": result.get("complexity_strategy"),
        "maximum_level": result.get("maximum_complexity_level"),
        "reached_level": result.get("reached_complexity_level"),
        "updates_completed": result.get("updates_completed"),
        "initial_exact_rate": result.get("initial_reward"),
        "final_exact_rate": result.get("final_reward"),
        "exact_gain": result.get("reward_gain"),
        "stop_reason": result.get("stop_reason"),
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
        "branch_snapshots": (
            [item for item in branch_snapshots if isinstance(item, dict)]
            if isinstance(branch_snapshots, list)
            else []
        ),
        "initial_by_level": initial_by_level if isinstance(initial_by_level, dict) else {},
        "final_by_level": final_by_level if isinstance(final_by_level, dict) else {},
        "policy_update_count": result.get("policy_update_count"),
        "teacher_update_count": result.get("teacher_update_count"),
        "informative_group_rate": result.get("informative_group_rate"),
        "teacher_fallback_rate": result.get("teacher_fallback_rate"),
        "total_sampled_completions": result.get("total_sampled_completions"),
    }


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
        conn.execute(
            """
            UPDATE runs SET status = 'QUEUED', updated_at = now()
            WHERE status IN ('PROVISIONING', 'PREPARING', 'RUNNING')
              AND desired_state = 'RUNNING'
            """
        )
        conn.execute(
            """
            UPDATE run_attempts SET status = 'QUEUED', updated_at = now()
            WHERE status IN ('PROVISIONING', 'PREPARING', 'RUNNING')
              AND EXISTS (
                SELECT 1 FROM runs r
                WHERE r.run_id = run_attempts.run_id AND r.status = 'QUEUED'
              )
            """
        )
    for run_id in canceled_run_ids:
        release_run_resources(run_id)
    yield


app = FastAPI(title="Equinox Next orchestrator", version="0.1.0", lifespan=lifespan)


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
            "resource_profile": "mock-cpu",
        },
        "verification": {
            "plan_id": "cad.transition-composite@1",
            "judge_provider": request.judge_provider,
            "judge_spec_id": "cad.pointwise.mock@1",
            "group_judge_spec_id": "cad.sibling-group.mock@1",
            "calibration_status": "MOCK_CONTRACT_ONLY",
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
    allocation = provider.allocate("mock-cpu")
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
            ) VALUES (%s, %s, 1, 'QUEUED', 'MockRunPodProvider', %s)
            """,
            (attempt_id, run_id, allocation.allocation_id),
        )
        conn.execute(
            """
            INSERT INTO compute_allocations(
              allocation_id, run_attempt_id, provider_name, desired_state, observed_state,
              resource_profile, provider_handle
            ) VALUES (%s, %s, 'MockRunPodProvider', 'ALLOCATED', 'ALLOCATED', %s, %s)
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
            "tokenizer": "mock-tokenizer@1",
            "prompt_template": "cad-policy@1",
            "tool_schema": "cad-actions@1",
            "sampling": {"temperature": 0, "seed": request.seed},
        }
        conn.execute(
            """
            INSERT INTO policy_versions(
              policy_version_id, run_id, ordinal, artifact_digest, behavior_manifest
            ) VALUES (%s, %s, 0, %s, %s)
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
                "providers": ["MockRunPodProvider", "MockJudgeProvider"],
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
    with connection() as conn:
        conn.execute("SELECT 1")
    return {
        "status": "ready",
        "service": "orchestrator",
        "policy_compute_providers": sorted(POLICY_COMPUTE_PROVIDERS),
        "judge_providers": ["MockJudgeProvider"],
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
                "policy_compute_provider": "MockRunPodProvider",
                "judge_provider": "MockJudgeProvider",
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
                "policy_compute_provider": "MockRunPodProvider",
                "judge_provider": "MockJudgeProvider",
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


@app.post("/internal/seed", status_code=202)
def seed_demo() -> dict[str, Any]:
    with connection() as conn:
        existing = list(
            conn.execute(
                """
                SELECT run_id, algorithm, status FROM runs
                WHERE name IN ('Local independent baseline', 'Local branch-aware CAD proof')
                ORDER BY created_at
                """
            )
        )
    if existing:
        return {"runs": existing, "existing": True}
    baseline = _launch(
        LaunchRunRequest(
            name="Local independent baseline",
            algorithm="independent_rollout_baseline",
            branch=BranchConfig(width=1),
        )
    )
    branch = _launch(
        LaunchRunRequest(
            name="Local branch-aware CAD proof",
            algorithm="bpo_local_metric",
            branch=BranchConfig(width=4),
        )
    )
    return {"runs": [baseline, branch], "existing": False}


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
            "policy_compute": "MockRunPodProvider",
            "judge": "MockJudgeProvider",
            "execution": "ComposeExecutionProvider",
        }
        run["cost"] = {
            "execution_credits": round(float(run["verification_run_count"]) * 0.0142, 4),
            "judge_credits": 0,
        }
    return {
        "items": data,
        "research_items": research_items,
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
            "policy_compute": "MockRunPodProvider",
            "judge": "MockJudgeProvider",
            "execution": "ComposeExecutionProvider",
        },
        "calibration": {
            "status": "MOCK_CONTRACT_ONLY",
            "message": "Mock assessments are contract fixtures, not human-aligned visual judgments.",
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
        policy_compute_provider="MockRunPodProvider",
        judge_provider="MockJudgeProvider",
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
    if request.successes > request.attempts:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_COMPLEXITY_OBSERVATION",
                "message": "Successes cannot exceed attempts.",
            },
        )
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

        state = conn.execute(
            "SELECT * FROM complexity_states WHERE run_id = %s FOR UPDATE", (run_id,)
        ).fetchone()
        if not state:
            raise HTTPException(status_code=404, detail={"code": "COMPLEXITY_STATE_NOT_FOUND"})
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
            new_attempts=request.attempts,
            new_successes=request.successes,
            evaluation_window=state["evaluation_window"],
            mastery_threshold=state["mastery_threshold"],
            promotion_step=state["promotion_step"],
        )
        conn.execute(
            """
            INSERT INTO complexity_observations(
              operation_id, run_id, request_digest, level, successes, attempts, promoted
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request.operation_id,
                run_id,
                request_digest,
                request.level,
                request.successes,
                request.attempts,
                update["promoted"],
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
                1 if update["promoted"] else 0,
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
def claim_run() -> dict[str, Any]:
    with connection() as conn:
        attempt = conn.execute(
            """
            SELECT ra.* FROM run_attempts ra
            JOIN runs r ON r.run_id = ra.run_id
            WHERE ra.status = 'QUEUED' AND r.desired_state = 'RUNNING'
            ORDER BY ra.created_at
            FOR UPDATE OF ra SKIP LOCKED
            LIMIT 1
            """
        ).fetchone()
        if not attempt:
            return {"claim": None}
        conn.execute(
            """
            UPDATE run_attempts SET status = 'PROVISIONING', heartbeat_at = now(), updated_at = now()
            WHERE attempt_id = %s
            """,
            (attempt["attempt_id"],),
        )
        conn.execute(
            "UPDATE runs SET status = 'PROVISIONING', updated_at = now() WHERE run_id = %s",
            (attempt["run_id"],),
        )
    return {"claim": {"attempt_id": attempt["attempt_id"], "run_id": attempt["run_id"]}}


@app.post("/internal/run-attempts/{attempt_id}/heartbeat")
def heartbeat(attempt_id: str) -> dict[str, Any]:
    with connection() as conn:
        updated = conn.execute(
            "UPDATE run_attempts SET heartbeat_at = now() WHERE attempt_id = %s RETURNING run_id",
            (attempt_id,),
        ).fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail={"code": "ATTEMPT_NOT_FOUND"})
    return {"attempt_id": attempt_id, "heartbeat_at": utc_now()}


@app.post("/internal/run-attempts/{attempt_id}/execute")
def execute_claim(attempt_id: str) -> dict[str, Any]:
    execute_run_attempt(attempt_id)
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
    return {"items": items}


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
            "Mock judge output is stochastic-evidence contract data, not objective truth."
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
        "label": "Mock model assessment",
        "calibration_status": "MOCK_CONTRACT_ONLY",
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
    if item["candidate"]:
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
    return item


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
    return {
        "execution": execution,
        "trajectory": (
            research_trajectory(proof_result) if isinstance(proof_result, dict) else None
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
            "policy_compute": ["MockRunPodProvider"],
            "judge": ["MockJudgeProvider"],
            "execution": ["ComposeExecutionProvider"],
        },
        "external_capacity": 0,
    }


@app.post("/internal/research-compute-proofs", status_code=201)
def ingest_research_compute_proof(
    request: ResearchComputeProofRequest,
) -> dict[str, Any]:
    receipt = request.model_dump(mode="json")
    receipt_digest = canonical_digest(receipt)
    result_progress = research_result_progress(request.result)
    with connection() as conn:
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
                  progress = progress || %s,
                  proof_id = %s,
                  receipt_digest = %s,
                  completed_at = %s,
                  teardown_confirmed = true,
                  updated_at = now()
                WHERE provider_handle = %s
                  AND status != 'FAILED'
                """,
                (
                    Jsonb(result_progress),
                    existing["proof_id"],
                    receipt_digest,
                    request.completed_at,
                    request.provider_handle,
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
              progress = progress || %s,
              proof_id = %s,
              receipt_digest = %s,
              completed_at = %s,
              teardown_confirmed = true,
              updated_at = now()
            WHERE provider_handle = %s
              AND status != 'FAILED'
            """,
            (
                Jsonb(result_progress),
                proof_id,
                receipt_digest,
                request.completed_at,
                request.provider_handle,
            ),
        )
    return {
        "proof_id": proof_id,
        "receipt_digest": receipt_digest,
        "already_recorded": False,
    }
