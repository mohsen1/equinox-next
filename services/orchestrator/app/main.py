from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from equinox_core import canonical_digest, make_id, utc_now
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from .database import connection, migrate
from .providers import POLICY_COMPUTE_PROVIDERS, assert_local_registry
from .science import artifact_store, emit_event
from .workflow import execute_run_attempt, rejudge_transition, release_run_resources


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BranchConfig(StrictModel):
    width: Literal[1, 4]
    decision_after_actions: int = Field(default=3, ge=1, le=8)
    rng_mode: Literal["split_stream"] = "split_stream"


class BudgetConfig(StrictModel):
    transitions: int = Field(default=64, ge=8, le=256)
    environment_cpu_seconds: int = Field(default=600, ge=60, le=3600)
    render_cpu_seconds: int = Field(default=300, ge=30, le=1800)
    judge_input_tokens: int = Field(default=100000, ge=1000, le=1000000)


class LaunchRunRequest(StrictModel):
    name: str = Field(min_length=3, max_length=80)
    algorithm: Literal["independent_rollout_baseline", "bpo_local_metric"]
    study_id: str = Field(
        default="study_cad_branching_contract_v1",
        min_length=3,
        max_length=96,
        pattern=r"^[a-z0-9][a-z0-9_-]+$",
    )
    study_condition: Literal["BRANCH_AWARE", "INDEPENDENT_CONTROL"] | None = None
    research_question: str = Field(
        default=(
            "Does a shared decision checkpoint produce more useful CAD "
            "continuations than independent rollouts under a matched protocol?"
        ),
        min_length=12,
        max_length=240,
    )
    protocol_revision: str = Field(default="cad-contract-protocol@1", min_length=3, max_length=96)
    policy_compute_provider: Literal["MockRunPodProvider"] = "MockRunPodProvider"
    judge_provider: Literal["MockJudgeProvider"] = "MockJudgeProvider"
    task_revision: Literal["mounting-plate@sha256:fixture-v1"] = "mounting-plate@sha256:fixture-v1"
    branch: BranchConfig
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


def _study_condition(algorithm: str, requested: str | None = None) -> str:
    if requested:
        return requested
    return "BRANCH_AWARE" if algorithm == "bpo_local_metric" else "INDEPENDENT_CONTROL"


def _data_protocol(
    *,
    task_revision: str,
    seed: int,
    verification_plan_id: str = "cad.transition-composite@1",
) -> dict[str, Any]:
    generator = {
        "id": "cad-fixture-generator",
        "revision": "cad-fixture-generator@1",
        "task_revision": task_revision,
        "seed": seed,
        "sampling": "paired deterministic fixture",
    }
    manifest = {
        "schema_version": 1,
        "source": "generated",
        "generator": {**generator, "digest": canonical_digest(generator)},
        "splits": {
            "training": {
                "task_groups": 1,
                "candidate_trajectories": 4,
                "task_families": {"cad_reconstruction": 1},
                "complexity_levels": {"bounded_fixture": 1},
            },
            "validation": {"task_groups": 0, "reason": "not part of the contract-proof milestone"},
            "fixed_guard": {"task_groups": 0, "reason": "not part of the contract-proof milestone"},
            "test": {"task_groups": 0, "reason": "not part of the contract-proof milestone"},
        },
        "sampling_policy": {
            "strategy": "paired_by_seed",
            "seed": seed,
            "temperature": 0,
        },
        "quality_checks": {
            "deduplication": "single canonical task revision",
            "leakage": "not measurable without held-out splits",
            "evaluation_pack": verification_plan_id,
        },
    }
    return {**manifest, "digest": canonical_digest(manifest)}


def _study_manifest(request: LaunchRunRequest) -> dict[str, Any]:
    condition = _study_condition(request.algorithm, request.study_condition)
    match_contract = {
        "protocol_revision": request.protocol_revision,
        "task_revision": request.task_revision,
        "model_revision": "deterministic-cad-policy@1",
        "evaluation_pack": "cad.transition-composite@1",
        "budgets": request.budgets.model_dump(),
        "data_generator_revision": "cad-fixture-generator@1",
        "seed_policy": "paired_by_seed",
    }
    return {
        "study_id": request.study_id,
        "condition": condition,
        "research_question": request.research_question,
        "protocol_revision": request.protocol_revision,
        "match_contract": match_contract,
        "match_contract_digest": canonical_digest(match_contract),
    }


def _study_from_manifest(manifest: dict[str, Any], algorithm: str) -> dict[str, Any]:
    study = manifest.get("study")
    if isinstance(study, dict):
        return study
    task_revision = str(manifest.get("task_revision", "unknown-task"))
    fallback_contract = {
        "protocol_revision": "legacy-contract-protocol@1",
        "task_revision": task_revision,
        "model_revision": "deterministic-cad-policy@1",
        "evaluation_pack": "cad.transition-composite@1",
        "budgets": manifest.get("budgets", {}),
        "data_generator_revision": "cad-fixture-generator@1",
        "seed_policy": "paired_by_seed",
    }
    return {
        "study_id": "study_cad_branching_contract_v1",
        "condition": _study_condition(algorithm),
        "research_question": (
            "Does a shared decision checkpoint produce more useful CAD continuations "
            "than independent rollouts under a matched protocol?"
        ),
        "protocol_revision": "legacy-contract-protocol@1",
        "match_contract": fallback_contract,
        "match_contract_digest": canonical_digest(fallback_contract),
        "derived_for_legacy_run": True,
    }


def _data_protocol_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol = manifest.get("data_protocol")
    if isinstance(protocol, dict):
        return protocol
    return {
        **_data_protocol(
            task_revision=str(manifest.get("task_revision", "mounting-plate@sha256:fixture-v1")),
            seed=int(manifest.get("seed", 17)),
            verification_plan_id=str(
                manifest.get("verification", {}).get("plan_id", "cad.transition-composite@1")
            ),
        ),
        "derived_for_legacy_run": True,
    }


def _learning_semantics(status: str, committed_iterations: int) -> dict[str, str]:
    if status in {"QUEUED", "PROVISIONING", "PREPARING", "RUNNING", "FINALIZING"}:
        return {
            "learning_outcome": "NOT_EVALUATED",
            "evidence_strength": "CONTRACT_ONLY",
            "summary": "Execution is in progress. No learning conclusion is available yet.",
        }
    if status == "SUCCEEDED" and committed_iterations:
        return {
            "learning_outcome": "INCONCLUSIVE",
            "evidence_strength": "CONTRACT_ONLY",
            "summary": (
                "Execution completed and a policy update was committed. "
                "This contract fixture has no held-out evaluation, so improvement is not claimed."
            ),
        }
    if status == "SUCCEEDED":
        return {
            "learning_outcome": "NO_UPDATE",
            "evidence_strength": "CONTRACT_ONLY",
            "summary": "Execution completed without a committed policy update.",
        }
    if status == "CANCELED":
        return {
            "learning_outcome": "NOT_EVALUATED",
            "evidence_strength": "CONTRACT_ONLY",
            "summary": "Execution was canceled before a learning conclusion could be measured.",
        }
    return {
        "learning_outcome": "INCONCLUSIVE",
        "evidence_strength": "CONTRACT_ONLY",
        "summary": "Execution failed. No learning conclusion is available.",
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
    study = _study_manifest(request)
    with connection() as conn:
        existing_study = conn.execute(
            """
            SELECT manifest->'study'->>'match_contract_digest' AS match_contract_digest
            FROM runs
            WHERE manifest->'study'->>'study_id' = %s
            ORDER BY created_at
            LIMIT 1
            """,
            (request.study_id,),
        ).fetchone()
    if (
        existing_study
        and existing_study["match_contract_digest"]
        and existing_study["match_contract_digest"] != study["match_contract_digest"]
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "STUDY_PROTOCOL_MISMATCH",
                "message": (
                    "This study already pins a different task, model, evaluation pack, "
                    "budget, data generator, or seed policy. Start a new study instead."
                ),
            },
        )
    data_protocol = _data_protocol(
        task_revision=request.task_revision,
        seed=request.seed,
    )
    normalized = {
        "schema_version": 1,
        "profile": "local-contract-proof",
        "study": study,
        "data_protocol": data_protocol,
        "environment": {
            "id": "cad.reconstruction",
            "version": "1.0.0",
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
                "study_id": study["study_id"],
                "study_condition": study["condition"],
                "data_protocol_digest": data_protocol["digest"],
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
        "schema": LaunchRunRequest.model_json_schema(),
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


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    study = _study_from_manifest(manifest, run["algorithm"])
    semantics = _learning_semantics(run["status"], int(run.get("committed_iteration_count", 0)))
    return {
        "run_id": run["run_id"],
        "source_run_id": run["source_run_id"],
        "name": run["name"],
        "algorithm": run["algorithm"],
        "status": run["status"],
        "desired_state": run["desired_state"],
        "manifest": {
            "environment": manifest.get("environment", {}),
            "task_revision": manifest.get("task_revision"),
            "seed": manifest.get("seed"),
        },
        "manifest_digest": run["manifest_digest"],
        "cleanup_status": run["cleanup_status"],
        "collection_batch_count": int(run["collection_batch_count"]),
        "iteration_count": int(run["iteration_count"]),
        "committed_iteration_count": int(run.get("committed_iteration_count", 0)),
        "rollout_tree_count": int(run["rollout_tree_count"]),
        "verification_run_count": int(run["verification_run_count"]),
        "proof_count": int(run.get("proof_count", run["verification_run_count"])),
        "abstention_count": int(run["abstention_count"]),
        "retry_count": int(run["retry_count"]),
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "providers": {
            "policy_compute": "MockRunPodProvider",
            "judge": "MockJudgeProvider",
            "execution": "ComposeExecutionProvider",
        },
        "cost": {
            "execution_credits": round(float(run["verification_run_count"]) * 0.0142, 4),
            "judge_credits": 0,
        },
        "study": {
            "study_id": study["study_id"],
            "condition": study["condition"],
            "protocol_revision": study["protocol_revision"],
            "research_question": study["research_question"],
        },
        **semantics,
    }


@app.get("/v1/runs")
def list_runs(
    q: str | None = Query(default=None, max_length=120),
    status: str | None = Query(default=None, max_length=32),
    study_id: str | None = Query(default=None, max_length=96),
    sort: Literal["updated_desc", "created_desc", "name_asc"] = "updated_desc",
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    where = ["1 = 1"]
    params: list[Any] = []
    if q:
        where.append("(r.name ILIKE %s OR r.run_id ILIKE %s)")
        needle = f"%{q.strip()}%"
        params.extend([needle, needle])
    if status:
        where.append("r.status = %s")
        params.append(status)
    if study_id:
        where.append(
            "COALESCE(r.manifest->'study'->>'study_id', 'study_cad_branching_contract_v1') = %s"
        )
        params.append(study_id)
    where_sql = " AND ".join(where)
    order_by = {
        "updated_desc": "r.updated_at DESC, r.run_id",
        "created_desc": "r.created_at DESC, r.run_id",
        "name_asc": "lower(r.name), r.run_id",
    }[sort]
    with connection() as conn:
        total = conn.execute(
            f"SELECT count(*) AS count FROM runs r WHERE {where_sql}",
            params,
        ).fetchone()["count"]
        data = list(
            conn.execute(
                f"""
                SELECT r.*,
                  (SELECT count(*) FROM collection_batches cb
                   JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                   WHERE ra.run_id = r.run_id) AS collection_batch_count,
                  (SELECT count(*) FROM training_iterations ti
                   WHERE ti.run_id = r.run_id) AS iteration_count,
                  (SELECT count(*) FROM training_iterations ti
                   WHERE ti.run_id = r.run_id
                     AND ti.output_policy_version_id IS NOT NULL)
                    AS committed_iteration_count,
                  (SELECT count(*) FROM rollout_trees rt
                   JOIN collection_batches cb ON cb.collection_batch_id = rt.collection_batch_id
                   JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                   WHERE ra.run_id = r.run_id) AS rollout_tree_count,
                  (SELECT count(*) FROM verification_runs vr
                   WHERE vr.run_id = r.run_id) AS verification_run_count,
                  (SELECT count(*) FROM evidence_bundles eb
                   JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
                   WHERE vr.run_id = r.run_id) AS proof_count,
                  (SELECT count(*) FROM judge_results jr
                   JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
                   JOIN verification_runs vr ON vr.verification_run_id = ji.verification_run_id
                   WHERE vr.run_id = r.run_id AND jr.outcome = 'ABSTAINED') AS abstention_count,
                  (SELECT count(*) FROM verifier_step_runs vsr
                   JOIN verification_runs vr ON vr.verification_run_id = vsr.verification_run_id
                   WHERE vr.run_id = r.run_id AND vsr.attempt_count > 1) AS retry_count
                FROM runs r
                WHERE {where_sql}
                ORDER BY {order_by}
                OFFSET %s LIMIT %s
                """,
                [*params, cursor, limit],
            )
        )
    next_cursor = cursor + len(data) if cursor + len(data) < total else None
    return {
        "items": [_run_summary(run) for run in data],
        "next_cursor": next_cursor,
        "total": total,
    }


def _study_payload(study_id: str, *, include_runs: bool = False) -> dict[str, Any]:
    page = list_runs(
        q=None,
        status=None,
        study_id=study_id,
        sort="created_desc",
        cursor=0,
        limit=100,
    )
    runs = page["items"]
    if not runs:
        raise HTTPException(status_code=404, detail={"code": "STUDY_NOT_FOUND"})
    with connection() as conn:
        manifests = list(
            conn.execute(
                """
                SELECT algorithm, manifest
                FROM runs
                WHERE COALESCE(
                  manifest->'study'->>'study_id',
                  'study_cad_branching_contract_v1'
                ) = %s
                ORDER BY created_at
                """,
                (study_id,),
            )
        )
    study = _study_from_manifest(manifests[0]["manifest"], manifests[0]["algorithm"])
    protocol_digests = {
        _study_from_manifest(item["manifest"], item["algorithm"])["match_contract_digest"]
        for item in manifests
    }
    condition_map: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        condition_map.setdefault(run["study"]["condition"], []).append(run)
    seed_sets = [
        {
            int(item["manifest"]["seed"])
            for item in condition_runs
            if item["manifest"].get("seed") is not None
        }
        for condition_runs in condition_map.values()
    ]
    paired_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    conditions = []
    for condition, condition_runs in sorted(condition_map.items()):
        seeds = sorted(
            {
                int(item["manifest"]["seed"])
                for item in condition_runs
                if item["manifest"].get("seed") is not None
            }
        )
        conditions.append(
            {
                "condition": condition,
                "seeds": seeds,
                "run_count": len(condition_runs),
                "completed_count": sum(item["status"] == "SUCCEEDED" for item in condition_runs),
                "learning_outcomes": sorted({item["learning_outcome"] for item in condition_runs}),
                "test_result": None,
                "test_result_label": "Not measured",
                "execution_credits": round(
                    sum(item["cost"]["execution_credits"] for item in condition_runs), 4
                ),
            }
        )
    comparison_valid = len(protocol_digests) == 1 and len(condition_map) >= 2 and bool(paired_seeds)
    payload = {
        "study_id": study_id,
        "research_question": study["research_question"],
        "protocol_revision": study["protocol_revision"],
        "run_count": len(runs),
        "conditions": conditions,
        "comparison": {
            "status": "MATCHED" if comparison_valid else "INCOMPLETE",
            "paired_seeds": paired_seeds,
            "protocol_digest": next(iter(protocol_digests)) if len(protocol_digests) == 1 else None,
            "constraints": [
                "dataset generator and task revision",
                "model revision",
                "evaluation pack",
                "budgets",
                "paired seed policy",
            ],
            "learning_claim": (
                "Comparison inputs are matched, but no held-out test pack exists; "
                "model improvement remains inconclusive."
            ),
        },
        "updated_at": max(run["updated_at"] for run in runs),
    }
    if include_runs:
        payload["runs"] = runs
    return payload


@app.get("/v1/studies")
def list_studies() -> dict[str, Any]:
    with connection() as conn:
        ids = [
            row["study_id"]
            for row in conn.execute(
                """
                SELECT DISTINCT COALESCE(
                  manifest->'study'->>'study_id',
                  'study_cad_branching_contract_v1'
                ) AS study_id
                FROM runs
                ORDER BY study_id
                """
            )
        ]
    return {"items": [_study_payload(study_id) for study_id in ids]}


@app.get("/v1/studies/{study_id}")
def study_detail(study_id: str) -> dict[str, Any]:
    return _study_payload(study_id, include_runs=True)


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
        committed_iterations = list(
            conn.execute(
                """
                SELECT ti.training_iteration_id, ti.status, ti.output_policy_version_id,
                  ii.manifest, ii.digest
                FROM training_iterations ti
                LEFT JOIN iteration_inputs ii ON ii.manifest_id = ti.iteration_input_id
                WHERE ti.run_id = %s
                ORDER BY ti.created_at
                """,
                (run_id,),
            )
        )
        proof_count = conn.execute(
            """
            SELECT count(*) AS count
            FROM evidence_bundles eb
            JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
            WHERE vr.run_id = %s
            """,
            (run_id,),
        ).fetchone()["count"]
    committed = [
        item for item in committed_iterations if item["output_policy_version_id"] is not None
    ]
    latest_input = committed[-1]["manifest"] if committed else None
    gradient_lineage = {
        "status": "MATERIALIZED" if latest_input else "PENDING",
        "contributing_rollout_trees": len(latest_input.get("rollout_tree_ids", []))
        if latest_input
        else 0,
        "contributing_proofs": len(latest_input.get("proof_bundle_ids", [])) if latest_input else 0,
        "contributing_reward_signals": len(latest_input.get("reward_signal_ids", []))
        if latest_input
        else 0,
        "eligibility_decisions_recorded": len(
            [item for item in (latest_input or {}).get("eligibility_decision_ids", []) if item]
        ),
        "consumed_by_policy_version": committed[-1]["output_policy_version_id"]
        if committed
        else None,
        "iteration_input_digest": committed[-1]["digest"] if committed else None,
    }
    study = _study_from_manifest(run["manifest"], run["algorithm"])
    data_protocol = _data_protocol_from_manifest(run["manifest"])
    semantics = _learning_semantics(run["status"], len(committed))
    return {
        "run": run,
        "attempts": attempts,
        "allocations": allocations,
        "policy_versions": policies,
        "metrics": metrics,
        "reward_signals": rewards,
        "failures": failures,
        "study": study,
        "data_protocol": {**data_protocol, "gradient_lineage": gradient_lineage},
        "outcome": semantics,
        "proof_count": proof_count,
        "evaluation": {
            "held_out_examples": 0,
            "test_result": None,
            "claim": "No held-out evaluation is configured for this contract fixture.",
        },
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


@app.get("/v1/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict[str, Any]:
    page = list_runs(
        q=run_id,
        status=None,
        study_id=None,
        sort="updated_desc",
        cursor=0,
        limit=10,
    )
    item = next((run for run in page["items"] if run["run_id"] == run_id), None)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
    return item


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
        name=f"Reproduction of {source['name']}"[:80],
        algorithm=manifest["algorithm"]["id"],
        study_id=_study_from_manifest(manifest, source["algorithm"])["study_id"],
        study_condition=_study_from_manifest(manifest, source["algorithm"])["condition"],
        research_question=_study_from_manifest(manifest, source["algorithm"])["research_question"],
        protocol_revision=_study_from_manifest(manifest, source["algorithm"])["protocol_revision"],
        policy_compute_provider="MockRunPodProvider",
        judge_provider="MockJudgeProvider",
        task_revision=manifest["task_revision"],
        branch=BranchConfig(**manifest["branch"]),
        budgets=BudgetConfig(**manifest["budgets"]),
        seed=manifest["seed"],
        retention_class="local-research",
    )
    return _launch(request, source_run_id=run_id)


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
            "SELECT * FROM rollout_trees WHERE rollout_tree_id = %s", (tree_id,)
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
                SELECT bm.* FROM branch_members bm JOIN branch_groups bg
                  ON bg.branch_group_id = bm.branch_group_id
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
    lane_steps: dict[str, int] = {}
    edge_items = []
    for transition in transitions:
        lane = transition["branch_member_id"] or "shared-prefix"
        lane_steps[lane] = lane_steps.get(lane, 0) + 1
        action = transition["payload"]["action"]
        edge_items.append(
            {
                "id": transition["transition_id"],
                "source": transition["source_state_id"],
                "target": transition["destination_state_id"],
                "branch_member_id": transition["branch_member_id"],
                "lane": lane,
                "local_step": lane_steps[lane],
                "outcome": transition["outcome"],
                "action": action,
                "action_label": str(action["kind"]).replace("_", " "),
                "verification_run_id": transition["verification_run_id"],
                "proof_bundle_id": transition["proof_bundle_id"],
            }
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
            }
            for state in states
        ],
        "edges": edge_items,
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


@app.get("/v1/rollout-trees/{tree_id}/index")
def rollout_tree_index(tree_id: str) -> dict[str, Any]:
    graph = rollout_tree_graph(tree_id)
    return {
        "tree": graph["tree"],
        "states": [
            {
                "id": node["id"],
                "sequence": node["sequence"],
                "semantic_status": node["semantic_status"],
            }
            for node in graph["nodes"]
        ],
        "transitions": [
            {
                key: edge[key]
                for key in (
                    "id",
                    "source",
                    "target",
                    "branch_member_id",
                    "lane",
                    "local_step",
                    "outcome",
                    "action_label",
                    "verification_run_id",
                )
            }
            for edge in graph["edges"]
        ],
        "branch_members": graph["branch_members"],
        "updated_cursor": len(graph["edges"]),
    }


@app.get("/v1/rollout-trees/{tree_id}/branches/{branch_member_id}")
def rollout_branch_snapshot(tree_id: str, branch_member_id: str) -> dict[str, Any]:
    graph = rollout_tree_graph(tree_id)
    member = next(
        (item for item in graph["branch_members"] if item["branch_member_id"] == branch_member_id),
        None,
    )
    if not member:
        raise HTTPException(status_code=404, detail={"code": "BRANCH_MEMBER_NOT_FOUND"})
    edges = [
        edge for edge in graph["edges"] if edge["branch_member_id"] in (None, branch_member_id)
    ]
    state_ids = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
    return {
        "tree": graph["tree"],
        "branch_member": member,
        "nodes": [node for node in graph["nodes"] if node["id"] in state_ids],
        "edges": edges,
        "decision_checkpoint": graph["decision_checkpoints"][0]
        if graph["decision_checkpoints"]
        else None,
        "environment_snapshot": graph["environment_snapshots"][0]
        if graph["environment_snapshots"]
        else None,
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


@app.get("/v1/transitions/{transition_id}/explanation")
def transition_explanation(transition_id: str) -> dict[str, Any]:
    with connection() as conn:
        transition = conn.execute(
            """
            SELECT t.*, rt.task_revision, cb.behavior_policy_version_id,
              ra.run_id, r.manifest AS run_manifest
            FROM transitions t
            JOIN rollout_trees rt ON rt.rollout_tree_id = t.rollout_tree_id
            JOIN collection_batches cb ON cb.collection_batch_id = rt.collection_batch_id
            JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
            JOIN runs r ON r.run_id = ra.run_id
            WHERE t.transition_id = %s
            """,
            (transition_id,),
        ).fetchone()
        if not transition:
            raise HTTPException(status_code=404, detail={"code": "TRANSITION_NOT_FOUND"})
        source = conn.execute(
            "SELECT state_id, sequence, semantic_status FROM states WHERE state_id = %s",
            (transition["source_state_id"],),
        ).fetchone()
        destination = conn.execute(
            "SELECT state_id, sequence, semantic_status FROM states WHERE state_id = %s",
            (transition["destination_state_id"],),
        ).fetchone()
        verification = conn.execute(
            """
            SELECT * FROM verification_runs
            WHERE subject_id = %s AND rejudges_verification_run_id IS NULL
            ORDER BY created_at
            LIMIT 1
            """,
            (transition_id,),
        ).fetchone()
        steps = (
            list(
                conn.execute(
                    """
                    SELECT step_id, status, attempt_count, cache_status, metrics
                    FROM verifier_step_runs
                    WHERE verification_run_id = %s
                    ORDER BY started_at, step_id
                    """,
                    (verification["verification_run_id"],),
                )
            )
            if verification
            else []
        )
        metrics = list(
            conn.execute(
                """
                SELECT descriptor, value, unit
                FROM metric_observations
                WHERE subject_id = %s
                ORDER BY descriptor
                """,
                (transition_id,),
            )
        )
        rewards = list(
            conn.execute(
                """
                SELECT reward_signal_id, name, value, metric_observation_ids
                FROM reward_signals
                WHERE subject_id = %s
                ORDER BY name
                """,
                (transition_id,),
            )
        )
        eligibility = conn.execute(
            """
            SELECT status, reason_code
            FROM eligibility_decisions
            WHERE rollout_tree_id = %s
              AND branch_member_id IS NOT DISTINCT FROM %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (transition["rollout_tree_id"], transition["branch_member_id"]),
        ).fetchone()
        iteration = conn.execute(
            """
            SELECT ti.status, ti.training_iteration_id, ti.output_policy_version_id,
              ii.digest AS iteration_input_digest
            FROM training_iterations ti
            LEFT JOIN iteration_inputs ii ON ii.manifest_id = ti.iteration_input_id
            WHERE ti.collection_batch_id = (
              SELECT collection_batch_id
              FROM rollout_trees
              WHERE rollout_tree_id = %s
            )
            ORDER BY ti.created_at DESC
            LIMIT 1
            """,
            (transition["rollout_tree_id"],),
        ).fetchone()
        judge = (
            conn.execute(
                """
                SELECT jr.outcome, jr.result
                FROM judge_results jr
                JOIN judge_invocations ji
                  ON ji.judge_invocation_id = jr.judge_invocation_id
                WHERE ji.verification_run_id = %s
                ORDER BY jr.created_at DESC
                LIMIT 1
                """,
                (verification["verification_run_id"],),
            ).fetchone()
            if verification
            else None
        )
        local_step = conn.execute(
            """
            SELECT count(*) AS count
            FROM transitions
            WHERE rollout_tree_id = %s
              AND branch_member_id IS NOT DISTINCT FROM %s
              AND created_at <= %s
            """,
            (
                transition["rollout_tree_id"],
                transition["branch_member_id"],
                transition["created_at"],
            ),
        ).fetchone()["count"]
    terminal_quality = next(
        (
            metric["value"]
            for metric in metrics
            if metric["descriptor"] == "deterministic.terminal_quality"
        ),
        None,
    )
    action = transition["payload"]["action"]
    study = _study_from_manifest(
        transition["run_manifest"],
        transition["run_manifest"]["algorithm"]["id"],
    )
    return {
        "transition_id": transition_id,
        "lane": transition["branch_member_id"] or "shared-prefix",
        "local_step": int(local_step),
        "task": {
            "title": "Reconstruct the canonical mounting plate",
            "revision": transition["task_revision"],
            "families": ["cad reconstruction"],
            "known_checks": [
                "geometry validity",
                "constraint compliance",
                "canonical render",
                "reference correspondence",
            ],
            "study_question": study["research_question"],
        },
        "checkpoint": {
            "policy_version_id": transition["behavior_policy_version_id"],
            "source_state_id": source["state_id"],
            "source_sequence": source["sequence"],
            "source_status": source["semantic_status"],
        },
        "action": {
            "kind": action["kind"],
            "label": str(action["kind"]).replace("_", " "),
            "parameters": {key: value for key, value in action.items() if key != "kind"},
        },
        "effect": {
            "destination_state_id": destination["state_id"],
            "destination_sequence": destination["sequence"],
            "destination_status": destination["semantic_status"],
            "transition_outcome": transition["outcome"],
            "terminal_quality": terminal_quality,
        },
        "verifier": {
            "status": verification["status"] if verification else "PENDING",
            "steps": steps,
            "deterministic_facts": [
                metric
                for metric in metrics
                if str(metric["descriptor"]).startswith("deterministic.")
            ],
            "model_assessment": {
                "outcome": judge["outcome"],
                "explanation": judge["result"].get("explanation"),
            }
            if judge
            else None,
        },
        "learning": {
            "eligibility": eligibility["status"] if eligibility else "PENDING",
            "eligibility_reason": eligibility["reason_code"] if eligibility else None,
            "reward_signals": rewards,
            "optimizer_status": iteration["status"] if iteration else "PENDING",
            "contributed_to_policy_version": iteration["output_policy_version_id"]
            if iteration and eligibility and eligibility["status"] == "ADMITTED"
            else None,
            "iteration_input_digest": iteration["iteration_input_digest"] if iteration else None,
        },
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


@app.get("/v1/proofs")
def proofs(
    q: str | None = Query(default=None, max_length=120),
    status: str | None = Query(default=None, max_length=40),
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    where = ["1 = 1"]
    params: list[Any] = []
    if q:
        where.append(
            "(r.name ILIKE %s OR r.run_id ILIKE %s "
            "OR eb.proof_bundle_id ILIKE %s OR eb.digest ILIKE %s)"
        )
        needle = f"%{q.strip()}%"
        params.extend([needle, needle, needle, needle])
    if status:
        where.append("vr.status = %s")
        params.append(status)
    where_sql = " AND ".join(where)
    with connection() as conn:
        total = conn.execute(
            f"""
            SELECT count(*) AS count
            FROM evidence_bundles eb
            JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
            JOIN runs r ON r.run_id = vr.run_id
            WHERE {where_sql}
            """,
            params,
        ).fetchone()["count"]
        items = list(
            conn.execute(
                f"""
                SELECT eb.proof_bundle_id, eb.digest, eb.subject_type, eb.subject_id,
                  eb.created_at, vr.verification_run_id, vr.status,
                  vr.plan_id, vr.rejudges_verification_run_id,
                  r.run_id, r.name AS run_name, r.algorithm, r.manifest
                FROM evidence_bundles eb
                JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
                JOIN runs r ON r.run_id = vr.run_id
                WHERE {where_sql}
                ORDER BY eb.created_at DESC
                OFFSET %s LIMIT %s
                """,
                [*params, cursor, limit],
            )
        )
    for item in items:
        study = _study_from_manifest(item.pop("manifest"), item["algorithm"])
        item["study_id"] = study["study_id"]
        item["study_condition"] = study["condition"]
        item["label"] = (
            "Solve rate evidence" if item["subject_type"] == "TRANSITION" else "Group evidence"
        )
    next_cursor = cursor + len(items) if cursor + len(items) < total else None
    return {"items": items, "total": total, "next_cursor": next_cursor}


def _proof_detail_payload(proof_bundle_id: str) -> dict[str, Any]:
    with connection() as conn:
        bundle = conn.execute(
            """
            SELECT eb.*, vr.run_id, vr.status AS verification_status, vr.plan_id,
              vr.subject_type AS verification_subject_type,
              vr.rejudges_verification_run_id,
              r.name AS run_name, r.algorithm, r.status AS run_status,
              r.manifest AS run_manifest, r.manifest_digest
            FROM evidence_bundles eb
            JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
            JOIN runs r ON r.run_id = vr.run_id
            WHERE eb.proof_bundle_id = %s
            """,
            (proof_bundle_id,),
        ).fetchone()
        if not bundle:
            raise HTTPException(status_code=404, detail={"code": "EVIDENCE_BUNDLE_NOT_FOUND"})
        artifacts = list(
            conn.execute(
                """
                SELECT a.artifact_id, a.digest, a.media_type, a.size_bytes,
                  ar.role, ar.ordinal, ar.viewer_hint, ar.visibility, ar.trust_class
                FROM artifact_refs ar
                JOIN artifacts a ON a.artifact_id = ar.artifact_id
                WHERE (ar.entity_type = 'evidence_bundle' AND ar.entity_id = %s)
                   OR (
                     ar.entity_type = 'verification_run'
                     AND ar.entity_id = %s
                   )
                ORDER BY ar.role, ar.ordinal
                """,
                (proof_bundle_id, bundle["verification_run_id"]),
            )
        )
        policies = list(
            conn.execute(
                """
                SELECT policy_version_id, ordinal, artifact_digest, behavior_manifest, created_at
                FROM policy_versions
                WHERE run_id = %s
                ORDER BY ordinal
                """,
                (bundle["run_id"],),
            )
        )
        allocations = list(
            conn.execute(
                """
                SELECT ca.allocation_id, ca.provider_name, ca.desired_state,
                  ca.observed_state, ca.cleanup_warning, ca.updated_at
                FROM compute_allocations ca
                JOIN run_attempts ra ON ra.attempt_id = ca.run_attempt_id
                WHERE ra.run_id = %s
                ORDER BY ca.created_at
                """,
                (bundle["run_id"],),
            )
        )
    study = _study_from_manifest(bundle["run_manifest"], bundle["algorithm"])
    data_protocol = _data_protocol_from_manifest(bundle["run_manifest"])
    return {
        "proof": {
            key: bundle[key]
            for key in (
                "proof_bundle_id",
                "verification_run_id",
                "subject_type",
                "subject_id",
                "digest",
                "manifest",
                "created_at",
                "verification_status",
                "plan_id",
                "rejudges_verification_run_id",
            )
        },
        "run": {
            "run_id": bundle["run_id"],
            "name": bundle["run_name"],
            "status": bundle["run_status"],
            "manifest_digest": bundle["manifest_digest"],
        },
        "exact_run_manifest": bundle["run_manifest"],
        "study": study,
        "data_protocol": data_protocol,
        "artifacts": [
            {
                **artifact,
                "downloadable": artifact["visibility"] != "HIDDEN",
            }
            for artifact in artifacts
        ],
        "policy_versions": policies,
        "teardown_receipt": {
            "complete": bool(allocations)
            and all(item["observed_state"] == "RELEASED" for item in allocations),
            "allocations": allocations,
        },
        "evaluation": {
            "examples": [],
            "claim": "No held-out evaluation examples exist for this local contract proof.",
        },
        "downloads": [
            {"item": "evidence-manifest", "label": "Evidence manifest"},
            {"item": "data-manifest", "label": "Data and protocol manifest"},
            {"item": "run-manifest", "label": "Exact run configuration"},
            {"item": "policy-manifest", "label": "Policy lineage"},
            {"item": "teardown-receipt", "label": "Teardown receipt"},
        ],
    }


@app.get("/v1/proofs/{proof_bundle_id}")
def proof_detail(proof_bundle_id: str) -> dict[str, Any]:
    return _proof_detail_payload(proof_bundle_id)


@app.get("/v1/proofs/{proof_bundle_id}/download")
def proof_download(
    proof_bundle_id: str,
    item: Literal[
        "evidence-manifest",
        "data-manifest",
        "run-manifest",
        "policy-manifest",
        "teardown-receipt",
    ],
) -> Response:
    detail = _proof_detail_payload(proof_bundle_id)
    payload = {
        "evidence-manifest": detail["proof"]["manifest"],
        "data-manifest": detail["data_protocol"],
        "run-manifest": {
            "manifest": detail["exact_run_manifest"],
            "manifest_digest": detail["run"]["manifest_digest"],
        },
        "policy-manifest": {"policy_versions": detail["policy_versions"]},
        "teardown-receipt": detail["teardown_receipt"],
    }[item]
    filename = f"{proof_bundle_id}-{item}.json"
    return Response(
        content=json.dumps(payload, indent=2, default=str),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/v1/environments")
def environments() -> dict[str, Any]:
    with connection() as conn:
        usage = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM rollout_trees) AS rollout_trees,
              (SELECT count(*) FROM training_iterations
               WHERE output_policy_version_id IS NOT NULL)
                AS committed_iterations,
              (SELECT count(*) FROM verification_runs) AS verification_runs
            """
        ).fetchone()
    used_in_training = int(usage["committed_iterations"]) > 0
    return {
        "items": [
            {
                "environment_id": "cad.reconstruction",
                "name": "CAD reconstruction",
                "version": "1.0.0",
                "description": (
                    "A bounded multi-turn mounting-plate reconstruction fixture "
                    "with canonical renders and deterministic geometry reports."
                ),
                "dataset": {
                    "source": "generated",
                    "revision": "mounting-plate@sha256:fixture-v1",
                    "task_families": ["cad_reconstruction"],
                },
                "harness": {
                    "provider": "ComposeExecutionProvider",
                    "snapshot_fidelity": "logical_restore",
                    "network": "disabled",
                },
                "reward_function": {
                    "pipeline": "cad-local-rewards@1",
                    "deterministic": [
                        "geometry validity",
                        "constraint compliance",
                        "terminal quality",
                        "execution cost",
                    ],
                    "model_assessed": [
                        "reference correspondence",
                        "progress from source state",
                    ],
                },
                "readiness": {
                    "current": "USED_IN_TRAINING" if used_in_training else "SANDBOX_INTEGRATED",
                    "contract_defined": True,
                    "simulator_verified": True,
                    "sandbox_integrated": True,
                    "used_in_training": used_in_training,
                },
                "usage": usage,
            }
        ]
    }


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
    return {
        "allocations": allocations,
        "counts": counts,
        "provider_boundaries": {
            "policy_compute": ["MockRunPodProvider"],
            "judge": ["MockJudgeProvider"],
            "execution": ["ComposeExecutionProvider"],
        },
        "external_capacity": 0,
    }
