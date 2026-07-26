from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from equinox_core import (
    ArtifactStore,
    canonical_digest,
    content_digest,
    make_id,
    validate_contract,
    validate_judge_result,
)
from fastapi import FastAPI, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from . import cad
from .database import connection, migrate
from .judge import (
    DeterministicJudgeFixture,
    RetryableJudgeError,
    blinded_order,
    group_spec,
    pointwise_spec,
    prompt_registry,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OperationEnvelope(StrictModel):
    operation_id: str
    idempotency_key: str
    request_digest: str
    expected_version: int = Field(ge=0)
    correlation_id: str


class CreateSessionRequest(OperationEnvelope):
    task_revision: str
    scientific_state_id: str
    lease_owner: str
    rng_state: str


class ApplyActionRequest(OperationEnvelope):
    cursor_id: str
    lease_owner: str
    expected_state_id: str
    expected_fencing_token: int = Field(ge=1)
    destination_scientific_state_id: str
    policy_decision_id: str
    action: dict[str, Any]


class CaptureSnapshotRequest(OperationEnvelope):
    cursor_id: str
    lease_owner: str
    source_state_id: str
    expected_cursor_version: int = Field(ge=0)
    expected_fencing_token: int = Field(ge=1)
    requested_fidelity: str
    ephemeral: bool = False


class ChildSpec(StrictModel):
    sibling_index: int = Field(ge=0, le=3)
    scientific_state_id: str
    rng_derivation: str
    lease_owner: str


class ForkRequest(OperationEnvelope):
    operational_snapshot_id: str
    children: list[ChildSpec] = Field(min_length=4, max_length=4)


class VerifyTransitionRequest(OperationEnvelope):
    verification_run_id: str
    subject_id: str
    source_state: dict[str, Any]
    candidate_state: dict[str, Any]
    action: dict[str, Any]
    fixture_scenario: str = "valid"
    simulate_infra_retry: bool = False
    judge_spec_version: int = Field(default=1, ge=1, le=2)


class RejudgeRequest(OperationEnvelope):
    verification_run_id: str
    proof_bundle_digest: str
    fixture_scenario: str = "valid"
    judge_spec_version: int = Field(default=2, ge=1, le=2)


class GroupJudgeRequest(OperationEnvelope):
    verification_run_id: str
    subject_id: str
    sibling_proofs: list[dict[str, Any]] = Field(min_length=2, max_length=4)


class CursorActionRequest(OperationEnvelope):
    cursor_id: str


artifact_store = ArtifactStore()
judge_provider = DeterministicJudgeFixture()

WORKLOAD_PROFILES = {
    "cad.session@1": {
        "trust_class": "PUBLIC_CANDIDATE",
        "candidate_network": "NONE",
    },
    "cad.action@1": {
        "trust_class": "PUBLIC_CANDIDATE",
        "candidate_network": "NONE",
    },
    "cad.geometry-verify@1": {
        "trust_class": "TRUSTED_EVIDENCE",
        "candidate_network": "NOT_APPLICABLE",
    },
    "cad.canonical-render@1": {
        "trust_class": "TRUSTED_EVIDENCE",
        "candidate_network": "NOT_APPLICABLE",
    },
    "cad.proof-bundle@1": {
        "trust_class": "TRUSTED_EVIDENCE",
        "candidate_network": "NOT_APPLICABLE",
    },
    "judge.multimodal-rubric@1": {
        "trust_class": "ISOLATED_JUDGE",
        "provider_access": "DETERMINISTIC_FIXTURE_ONLY",
    },
    "judge.branch-group@1": {
        "trust_class": "ISOLATED_JUDGE",
        "provider_access": "DETERMINISTIC_FIXTURE_ONLY",
    },
}
JUDGE_PROVIDERS = {"DeterministicJudgeFixture": judge_provider}
EXECUTION_CONCURRENCY = int(os.getenv("EXECUTION_CONCURRENCY", "1"))


def _wait_for_dependencies() -> None:
    last_error: Exception | None = None
    for _ in range(40):
        try:
            migrate()
            artifact_store.ensure_bucket()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError("execution dependencies did not become ready") from last_error


@asynccontextmanager
async def lifespan(_: FastAPI):
    _wait_for_dependencies()
    yield


app = FastAPI(
    title="Equinox execution and verifier service",
    version="0.1.0",
    lifespan=lifespan,
)


def _operation_input(request: OperationEnvelope) -> dict[str, Any]:
    return request.model_dump(
        exclude={
            "operation_id",
            "idempotency_key",
            "request_digest",
            "expected_version",
            "correlation_id",
        }
    )


def _validate_request_digest(request: OperationEnvelope, operation_type: str) -> dict[str, Any]:
    operation_input = _operation_input(request)
    expected = canonical_digest({"operation_type": operation_type, "input": operation_input})
    if request.request_digest != expected:
        raise HTTPException(
            status_code=422,
            detail={"code": "REQUEST_DIGEST_MISMATCH", "expected": expected},
        )
    return operation_input


def _retryable_worker_exception(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(
            current,
            RetryableJudgeError | psycopg.OperationalError | ConnectionError | TimeoutError,
        ):
            return True
        if type(current).__module__.split(".", 1)[0] in {"botocore", "httpx"}:
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class ActivityClaim:
    job_id: str
    attempt_id: str
    attempt_number: int
    lease_owner: str
    fencing_token: int
    max_attempts: int


ACTIVITY_LEASE_SECONDS = 300
ACTIVITY_MAX_ATTEMPTS = 3


def _operation_failure(exc: Exception, *, retryable: bool) -> dict[str, Any]:
    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        code = str(detail.get("code", "REQUEST_REJECTED"))
        failure_class = "CALLER_OR_PRECONDITION"
    else:
        code = type(exc).__name__
        failure_class = "PLATFORM_TRANSIENT" if retryable else "PLATFORM_TERMINAL"
    return {
        "code": code,
        "failure_class": failure_class,
        "retryable": retryable,
    }


def _prepare_operation(
    request: OperationEnvelope,
    *,
    operation_type: str,
    profile: str,
    operation_input: dict[str, Any],
    simulate_infra_retry: bool,
) -> tuple[dict[str, Any] | None, ActivityClaim | None]:
    with connection() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (request.idempotency_key,),
        )
        existing = conn.execute(
            """
            SELECT operation_id, request_digest, status, result
            FROM operations
            WHERE idempotency_key = %s
            FOR UPDATE
            """,
            (request.idempotency_key,),
        ).fetchone()
        if existing:
            if existing["operation_id"] != request.operation_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "IDEMPOTENCY_OPERATION_CONFLICT",
                        "operation_id": existing["operation_id"],
                    },
                )
            if existing["request_digest"] != request.request_digest:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "IDEMPOTENCY_CONFLICT",
                        "operation_id": request.operation_id,
                    },
                )
            if existing["result"] is not None:
                return existing["result"], None
            if existing["status"] == "FAILED":
                raise HTTPException(
                    status_code=409,
                    detail={"code": "OPERATION_FAILED", "operation_id": request.operation_id},
                )
            job = conn.execute(
                "SELECT *, now() AS database_now FROM jobs WHERE operation_id = %s FOR UPDATE",
                (request.operation_id,),
            ).fetchone()
            if (
                job["status"] in {"LEASED", "RUNNING"}
                and job["lease_expires_at"] is not None
                and job["lease_expires_at"] > job["database_now"]
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "OPERATION_IN_PROGRESS",
                        "operation_id": request.operation_id,
                    },
                )
            if job["next_eligible_at"] > job["database_now"]:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "OPERATION_BACKOFF_ACTIVE",
                        "operation_id": request.operation_id,
                    },
                )
            latest = conn.execute(
                """
                SELECT * FROM attempts
                WHERE job_id = %s
                ORDER BY attempt_number DESC
                LIMIT 1
                FOR UPDATE
                """,
                (job["job_id"],),
            ).fetchone()
            if latest and latest["status"] == "RUNNING":
                conn.execute(
                    """
                    UPDATE attempts
                    SET status = 'STALE_RESULT',
                        outcome = 'STALE_RESULT',
                        failure_class = 'LOST_LEASE',
                        completed_at = now()
                    WHERE attempt_id = %s
                    """,
                    (latest["attempt_id"],),
                )
            attempt_number = (latest["attempt_number"] if latest else 0) + 1
            job_id = job["job_id"]
            max_attempts = job["max_attempts"]
            fencing_token = job["fencing_token"] + 1
        else:
            job_id = make_id("job")
            max_attempts = ACTIVITY_MAX_ATTEMPTS
            conn.execute(
                """
                INSERT INTO operations(
                  operation_id, idempotency_key, request_digest,
                  operation_type, status, request
                ) VALUES (%s, %s, %s, %s, 'AUTHORIZED', %s)
                """,
                (
                    request.operation_id,
                    request.idempotency_key,
                    request.request_digest,
                    operation_type,
                    Jsonb(operation_input),
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs(
                  job_id, operation_id, profile, status, max_attempts,
                  retry_policy_id, deadline_at
                ) VALUES (
                  %s, %s, %s, 'QUEUED', %s,
                  'bounded-exponential@1', now() + interval '10 minutes'
                )
                """,
                (job_id, request.operation_id, profile, max_attempts),
            )
            if simulate_infra_retry:
                conn.execute(
                    """
                    INSERT INTO attempts(
                      attempt_id, job_id, attempt_number, fencing_token,
                      status, outcome, failure, failure_class, cost,
                      usage, completed_at
                    ) VALUES (
                      %s, %s, 1, 1, 'INFRA_FAILED', 'INFRA_FAILED',
                      %s, 'PLATFORM_TRANSIENT', %s, %s, now()
                    )
                    """,
                    (
                        make_id("attempt"),
                        job_id,
                        Jsonb(
                            {
                                "code": "FIXTURE_WORKER_LOST",
                                "failure_class": "PLATFORM_TRANSIENT",
                                "retryable": True,
                            }
                        ),
                        Jsonb({"kind": "MEASURED", "wall_seconds": 0}),
                        Jsonb({}),
                    ),
                )
            attempt_number = 2 if simulate_infra_retry else 1
            fencing_token = attempt_number

        if attempt_number > max_attempts:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'INFRA_FAILED',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = now()
                WHERE job_id = %s
                """,
                (job_id,),
            )
            conn.execute(
                """
                UPDATE operations SET status = 'FAILED', updated_at = now()
                WHERE operation_id = %s
                """,
                (request.operation_id,),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "OPERATION_RETRIES_EXHAUSTED",
                    "operation_id": request.operation_id,
                },
            )

        lease_owner = make_id("activity_claim")
        attempt_id = make_id("attempt")
        conn.execute(
            """
            INSERT INTO attempts(
              attempt_id, job_id, attempt_number, fencing_token,
              status, lease_owner, cost, usage
            ) VALUES (%s, %s, %s, %s, 'RUNNING', %s, %s, %s)
            """,
            (
                attempt_id,
                job_id,
                attempt_number,
                fencing_token,
                lease_owner,
                Jsonb({"kind": "PENDING_MEASUREMENT"}),
                Jsonb({}),
            ),
        )
        conn.execute(
            """
            UPDATE jobs
            SET status = 'RUNNING',
                attempt_count = %s,
                lease_owner = %s,
                lease_expires_at = now() + (%s * interval '1 second'),
                fencing_token = %s,
                updated_at = now()
            WHERE job_id = %s
            """,
            (
                attempt_number,
                lease_owner,
                ACTIVITY_LEASE_SECONDS,
                fencing_token,
                job_id,
            ),
        )
        conn.execute(
            """
            UPDATE operations SET status = 'RUNNING', updated_at = now()
            WHERE operation_id = %s
            """,
            (request.operation_id,),
        )
        return (
            None,
            ActivityClaim(
                job_id=job_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                lease_owner=lease_owner,
                fencing_token=fencing_token,
                max_attempts=max_attempts,
            ),
        )


def _assert_step_result(
    conn: Any,
    *,
    operation_id: str,
    verification_run_id: str,
    step: dict[str, Any],
) -> None:
    result_digest = canonical_digest(step)
    existing = conn.execute(
        """
        SELECT operation_id, result_digest, result
        FROM accepted_step_results
        WHERE verification_run_id = %s AND step_id = %s
        FOR UPDATE
        """,
        (verification_run_id, step["step_id"]),
    ).fetchone()
    if existing:
        if (
            existing["operation_id"] != operation_id
            or existing["result_digest"] != result_digest
            or existing["result"] != step
        ):
            raise RuntimeError("accepted verification step replay conflicts with immutable result")
        return
    conn.execute(
        """
        INSERT INTO accepted_step_results(
          accepted_step_result_id, operation_id, verification_run_id,
          step_id, result_digest, result
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            make_id("accepted_step"),
            operation_id,
            verification_run_id,
            step["step_id"],
            result_digest,
            Jsonb(step),
        ),
    )


def _accept_activity_result(
    conn: Any,
    *,
    request: OperationEnvelope,
    claim: ActivityClaim,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    current = conn.execute(
        """
        SELECT j.status, j.lease_owner, j.lease_expires_at,
          j.fencing_token, o.result, now() AS database_now
        FROM jobs j
        JOIN operations o ON o.operation_id = j.operation_id
        WHERE j.job_id = %s
        FOR UPDATE OF j, o
        """,
        (claim.job_id,),
    ).fetchone()
    if current["result"] is not None:
        if canonical_digest(current["result"]) != canonical_digest(result):
            raise RuntimeError("operation replay produced a divergent accepted result")
        return current["result"]
    if (
        current["status"] != "RUNNING"
        or current["lease_owner"] != claim.lease_owner
        or current["fencing_token"] != claim.fencing_token
        or current["lease_expires_at"] <= current["database_now"]
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "STALE_ACTIVITY_RESULT", "operation_id": request.operation_id},
        )
    if result.get("verification_run_id") and result.get("steps"):
        for step in result["steps"]:
            _assert_step_result(
                conn,
                operation_id=request.operation_id,
                verification_run_id=result["verification_run_id"],
                step=step,
            )
    result["operation"] = {
        "operation_id": request.operation_id,
        "job_id": claim.job_id,
        "attempt_count": claim.attempt_number,
        "fencing_token": claim.fencing_token,
        "duplicate": False,
    }
    result_digest = canonical_digest(result)
    usage = {
        "kind": "MEASURED_PROCESS_WALL",
        "wall_seconds": round(elapsed_seconds, 6),
    }
    updated = conn.execute(
        """
        UPDATE attempts
        SET status = 'SUCCEEDED',
            outcome = 'SUCCEEDED',
            cost = %s,
            usage = %s,
            completed_at = now()
        WHERE attempt_id = %s
          AND lease_owner = %s
          AND fencing_token = %s
          AND status = 'RUNNING'
        """,
        (
            Jsonb({"kind": "MEASURED_PROCESS_WALL", "wall_seconds": usage["wall_seconds"]}),
            Jsonb(usage),
            claim.attempt_id,
            claim.lease_owner,
            claim.fencing_token,
        ),
    )
    if updated.rowcount != 1:
        raise RuntimeError("activity attempt lost its acceptance fence")
    conn.execute(
        """
        UPDATE jobs
        SET status = 'SUCCEEDED',
            result = %s,
            result_digest = %s,
            lease_owner = NULL,
            lease_expires_at = NULL,
            updated_at = now()
        WHERE job_id = %s
        """,
        (Jsonb(result), result_digest, claim.job_id),
    )
    conn.execute(
        """
        UPDATE operations
        SET status = 'SUCCEEDED',
            result = %s,
            result_digest = %s,
            updated_at = now()
        WHERE operation_id = %s
        """,
        (Jsonb(result), result_digest, request.operation_id),
    )
    return result


def _record_activity_failure(
    *,
    request: OperationEnvelope,
    claim: ActivityClaim,
    exc: Exception,
    elapsed_seconds: float,
) -> bool:
    retryable = _retryable_worker_exception(exc) and not isinstance(exc, HTTPException)
    can_retry = retryable and claim.attempt_number < claim.max_attempts
    failure = _operation_failure(exc, retryable=can_retry)
    attempt_status = (
        "INFRA_FAILED"
        if retryable
        else "CALLER_FAILED"
        if isinstance(exc, HTTPException)
        else "PLATFORM_FAILED"
    )
    with connection() as conn:
        current = conn.execute(
            """
            SELECT status, lease_owner, fencing_token
            FROM jobs WHERE job_id = %s FOR UPDATE
            """,
            (claim.job_id,),
        ).fetchone()
        if (
            not current
            or current["status"] != "RUNNING"
            or current["lease_owner"] != claim.lease_owner
            or current["fencing_token"] != claim.fencing_token
        ):
            return False
        conn.execute(
            """
            UPDATE attempts
            SET status = %s,
                outcome = %s,
                failure = %s,
                failure_class = %s,
                cost = %s,
                usage = %s,
                completed_at = now()
            WHERE attempt_id = %s
              AND lease_owner = %s
              AND fencing_token = %s
            """,
            (
                attempt_status,
                attempt_status,
                Jsonb(failure),
                failure["failure_class"],
                Jsonb(
                    {
                        "kind": "MEASURED_PROCESS_WALL",
                        "wall_seconds": round(elapsed_seconds, 6),
                    }
                ),
                Jsonb(
                    {
                        "kind": "MEASURED_PROCESS_WALL",
                        "wall_seconds": round(elapsed_seconds, 6),
                    }
                ),
                claim.attempt_id,
                claim.lease_owner,
                claim.fencing_token,
            ),
        )
        conn.execute(
            """
            UPDATE jobs
            SET status = %s,
                next_eligible_at = CASE
                  WHEN %s THEN now() + (
                    power(2, LEAST(%s - 1, 6)) * interval '250 milliseconds'
                  )
                  ELSE next_eligible_at
                END,
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = now()
            WHERE job_id = %s
            """,
            (
                "QUEUED" if can_retry else attempt_status,
                can_retry,
                claim.attempt_number,
                claim.job_id,
            ),
        )
        conn.execute(
            """
            UPDATE operations SET status = %s, updated_at = now()
            WHERE operation_id = %s
            """,
            ("RETRYABLE" if can_retry else "FAILED", request.operation_id),
        )
    return can_retry


def _run_operation(
    request: OperationEnvelope,
    operation_type: str,
    profile: str,
    worker: Callable[[Any], dict[str, Any]],
    *,
    simulate_infra_retry: bool = False,
    detached: bool = False,
) -> dict[str, Any]:
    operation_input = _validate_request_digest(request, operation_type)
    accepted, claim = _prepare_operation(
        request,
        operation_type=operation_type,
        profile=profile,
        operation_input=operation_input,
        simulate_infra_retry=simulate_infra_retry,
    )
    if accepted is not None:
        return accepted
    if claim is None:
        raise RuntimeError("operation preparation returned neither result nor claim")
    started = time.monotonic()
    try:
        if detached:
            result = worker(None)
            with connection() as conn:
                return _accept_activity_result(
                    conn,
                    request=request,
                    claim=claim,
                    result=result,
                    elapsed_seconds=time.monotonic() - started,
                )
        with connection() as conn:
            result = worker(conn)
            return _accept_activity_result(
                conn,
                request=request,
                claim=claim,
                result=result,
                elapsed_seconds=time.monotonic() - started,
            )
    except Exception as exc:
        can_retry = _record_activity_failure(
            request=request,
            claim=claim,
            exc=exc,
            elapsed_seconds=time.monotonic() - started,
        )
        if can_retry:
            raise HTTPException(
                status_code=503,
                detail={"code": "OPERATION_RETRYABLE", "operation_id": request.operation_id},
            ) from exc
        raise


def _stored_payload(stored: Any) -> dict[str, Any]:
    return {
        **stored.ref(ordinal=0),
        "object_key": stored.object_key,
        "size_bytes": stored.size_bytes,
    }


@app.get("/healthz")
def health() -> dict[str, Any]:
    with connection() as conn:
        conn.execute("SELECT 1")
    return {
        "status": "ready",
        "service": "execution",
        "provider": "ComposeExecutionProvider",
        "snapshot_fidelity": "logical_restore",
        "judge_providers": sorted(JUDGE_PROVIDERS),
    }


@app.get("/v1/profiles")
def profiles() -> dict[str, Any]:
    with connection() as conn:
        active = conn.execute(
            """
            SELECT count(*) AS count FROM jobs
            WHERE status IN ('LEASED', 'RUNNING')
            """
        ).fetchone()["count"]
    return {
        "execution_provider": "ComposeExecutionProvider",
        "profiles": WORKLOAD_PROFILES,
        "judge_providers": sorted(JUDGE_PROVIDERS),
        "capacity": {
            "fixture-worker": {
                "total": EXECUTION_CONCURRENCY,
                "active": active,
                "available": max(0, EXECUTION_CONCURRENCY - active),
            },
            "external-model-api": {"total": 0, "available": 0},
        },
    }


@app.post("/v1/sessions")
def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    def worker(conn: Any) -> dict[str, Any]:
        logical_state = cad.initial_state()
        session_id = make_id("session")
        cursor_id = make_id("cursor")
        runtime_instance_id = make_id("runtime")
        session_root = f"logical-fixture://sessions/{session_id}"
        conn.execute(
            """
            INSERT INTO sessions(
              session_id, runtime_instance_id, task_revision, logical_state,
              current_state_digest, status, session_root, rng_state
            ) VALUES (%s, %s, %s, %s, %s, 'READY', %s, %s)
            """,
            (
                session_id,
                runtime_instance_id,
                request.task_revision,
                Jsonb(logical_state),
                canonical_digest(logical_state),
                session_root,
                request.rng_state,
            ),
        )
        conn.execute(
            """
            INSERT INTO runtime_cursors(
              cursor_id, session_id, current_scientific_state_id, lease_owner,
              lease_expires_at, fencing_token, observed_status
            ) VALUES (%s, %s, %s, %s, now() + interval '5 minutes', 1, 'READY')
            """,
            (cursor_id, session_id, request.scientific_state_id, request.lease_owner),
        )
        return {
            "session_id": session_id,
            "runtime_instance_id": runtime_instance_id,
            "cursor_id": cursor_id,
            "current_state_id": request.scientific_state_id,
            "version": 0,
            "fencing_token": 1,
            "lease_owner": request.lease_owner,
            "obtained_fidelity": "logical_restore",
            "logical_state": logical_state,
        }

    return _run_operation(request, "create_session", "cad.session@1", worker)


@app.post("/v1/sessions/actions")
def apply_session_action(request: ApplyActionRequest) -> dict[str, Any]:
    def worker(conn: Any) -> dict[str, Any]:
        cursor = conn.execute(
            """
            UPDATE runtime_cursors
            SET observed_status = 'BUSY', updated_at = now()
            WHERE cursor_id = %s
              AND current_scientific_state_id = %s
              AND version = %s
              AND fencing_token = %s
              AND lease_owner = %s
              AND lease_expires_at > now()
              AND observed_status = 'READY'
            RETURNING *
            """,
            (
                request.cursor_id,
                request.expected_state_id,
                request.expected_version,
                request.expected_fencing_token,
                request.lease_owner,
            ),
        ).fetchone()
        if not cursor:
            raise HTTPException(
                status_code=409,
                detail={"code": "CURSOR_PRECONDITION_CONFLICT"},
            )

        session = conn.execute(
            "SELECT logical_state FROM sessions WHERE session_id = %s FOR UPDATE",
            (cursor["session_id"],),
        ).fetchone()
        source = session["logical_state"]
        try:
            candidate = cad.apply_action(source, request.action)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_CAD_ACTION", "message": str(exc)},
            ) from exc
        candidate_digest = canonical_digest(candidate)
        next_version = cursor["version"] + 1
        next_fence = cursor["fencing_token"] + 1
        conn.execute(
            """
            UPDATE sessions SET logical_state = %s, current_state_digest = %s, updated_at = now()
            WHERE session_id = %s
            """,
            (Jsonb(candidate), candidate_digest, cursor["session_id"]),
        )
        conn.execute(
            """
            UPDATE runtime_cursors
            SET current_scientific_state_id = %s, version = %s, fencing_token = %s,
              lease_expires_at = now() + interval '5 minutes', observed_status = 'READY',
              updated_at = now()
            WHERE cursor_id = %s AND lease_owner = %s
              AND fencing_token = %s AND observed_status = 'BUSY'
            """,
            (
                request.destination_scientific_state_id,
                next_version,
                next_fence,
                request.cursor_id,
                request.lease_owner,
                request.expected_fencing_token,
            ),
        )
        return {
            "cursor_id": request.cursor_id,
            "source_state": source,
            "candidate_state": candidate,
            "candidate_state_digest": candidate_digest,
            "destination_scientific_state_id": request.destination_scientific_state_id,
            "policy_decision_id": request.policy_decision_id,
            "version": next_version,
            "fencing_token": next_fence,
            "semantic_outcome": "TERMINATED" if candidate["submitted"] else "CONTINUED",
        }

    return _run_operation(request, "apply_action", "cad.action@1", worker)


@app.post("/v1/snapshots")
def capture_snapshot(request: CaptureSnapshotRequest) -> dict[str, Any]:
    def worker(conn: Any) -> dict[str, Any]:
        if request.requested_fidelity != "logical_restore":
            raise HTTPException(
                status_code=422,
                detail={"code": "UNSUPPORTED_FIDELITY", "obtained": "logical_restore"},
            )
        cursor = conn.execute(
            """
            SELECT c.*, s.logical_state, s.rng_state
            FROM runtime_cursors c JOIN sessions s ON s.session_id = c.session_id
            WHERE c.cursor_id = %s FOR UPDATE
            """,
            (request.cursor_id,),
        ).fetchone()
        if not cursor:
            raise HTTPException(status_code=404, detail={"code": "CURSOR_NOT_FOUND"})
        if (
            cursor["current_scientific_state_id"] != request.source_state_id
            or cursor["version"] != request.expected_cursor_version
            or cursor["fencing_token"] != request.expected_fencing_token
            or cursor["lease_owner"] != request.lease_owner
            or cursor["lease_expires_at"] <= conn.execute("SELECT now() AS now").fetchone()["now"]
            or cursor["observed_status"] != "READY"
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "SNAPSHOT_SOURCE_PRECONDITION_CONFLICT"},
            )
        logical_state = cursor["logical_state"]
        digest = canonical_digest(logical_state)
        restored_a = json.loads(json.dumps(logical_state))
        restored_b = json.loads(json.dumps(logical_state))
        probe_action = {"kind": "add_hole_pattern"}
        source_probe = cad.apply_action(logical_state, probe_action)
        restored_a_probe = cad.apply_action(restored_a, probe_action)
        restored_b_probe = cad.apply_action(restored_b, probe_action)
        probe_receipt = {
            "schema_version": 1,
            "recipe": "cad-logical-restore-probe@1",
            "source_digest": digest,
            "restored_digests": [
                canonical_digest(restored_a),
                canonical_digest(restored_b),
            ],
            "probe_action": probe_action,
            "probe_result_digests": [
                canonical_digest(source_probe),
                canonical_digest(restored_a_probe),
                canonical_digest(restored_b_probe),
            ],
            "render_digests": [
                content_digest(cad.render_svg(restored_a, label="probe")),
                content_digest(cad.render_svg(restored_b, label="probe")),
            ],
        }
        fidelity_probe_passed = (
            probe_receipt["restored_digests"] == [digest, digest]
            and len(set(probe_receipt["probe_result_digests"])) == 1
            and len(set(probe_receipt["render_digests"])) == 1
        )
        if not fidelity_probe_passed:
            raise HTTPException(
                status_code=422,
                detail={"code": "SNAPSHOT_FIDELITY_PROBE_FAILED"},
            )
        operational_snapshot_id = make_id("opsnap")
        conn.execute(
            """
            INSERT INTO snapshots(
              operational_snapshot_id, source_session_id, source_scientific_state_id,
              logical_state, logical_state_digest, requested_fidelity, obtained_fidelity,
              fidelity_probe_passed, rng_state, ephemeral, captured_cursor_version,
              captured_fencing_token, fidelity_probe_receipt
            ) VALUES (%s, %s, %s, %s, %s, %s, 'logical_restore', %s, %s, %s, %s, %s, %s)
            """,
            (
                operational_snapshot_id,
                cursor["session_id"],
                request.source_state_id,
                Jsonb(logical_state),
                digest,
                request.requested_fidelity,
                fidelity_probe_passed,
                cursor["rng_state"],
                request.ephemeral,
                cursor["version"],
                cursor["fencing_token"],
                Jsonb(probe_receipt),
            ),
        )
        return {
            "operational_snapshot_id": operational_snapshot_id,
            "source_state_id": request.source_state_id,
            "logical_state": logical_state,
            "logical_state_digest": digest,
            "requested_fidelity": request.requested_fidelity,
            "obtained_fidelity": "logical_restore",
            "fidelity_probe_passed": fidelity_probe_passed,
            "fidelity_probe_receipt": probe_receipt,
            "captured_cursor_version": cursor["version"],
            "captured_fencing_token": cursor["fencing_token"],
            "rng_state": cursor["rng_state"],
            "runtime_snapshot_handle": None,
        }

    return _run_operation(request, "capture_snapshot", "cad.session@1", worker)


@app.post("/v1/snapshots/forks")
def fork_snapshot(request: ForkRequest) -> dict[str, Any]:
    def worker(conn: Any) -> dict[str, Any]:
        sibling_indexes = [child.sibling_index for child in request.children]
        if sorted(sibling_indexes) != [0, 1, 2, 3]:
            raise HTTPException(status_code=422, detail={"code": "INVALID_SIBLING_INDEXES"})
        snapshot = conn.execute(
            "SELECT * FROM snapshots WHERE operational_snapshot_id = %s",
            (request.operational_snapshot_id,),
        ).fetchone()
        if not snapshot:
            raise HTTPException(status_code=404, detail={"code": "SNAPSHOT_NOT_FOUND"})
        children: list[dict[str, Any]] = []
        for child in request.children:
            session_id = make_id("session")
            cursor_id = make_id("cursor")
            runtime_instance_id = make_id("runtime")
            session_root = f"logical-fixture://sessions/{session_id}"
            derived_rng = canonical_digest(
                {"parent": snapshot["rng_state"], "derivation": child.rng_derivation}
            )
            conn.execute(
                """
                INSERT INTO sessions(
                  session_id, runtime_instance_id, task_revision, logical_state,
                  current_state_digest, status, session_root, rng_state
                )
                SELECT %s, %s, s.task_revision, %s, %s, 'READY', %s, %s
                FROM sessions s WHERE s.session_id = %s
                """,
                (
                    session_id,
                    runtime_instance_id,
                    Jsonb(snapshot["logical_state"]),
                    snapshot["logical_state_digest"],
                    session_root,
                    derived_rng,
                    snapshot["source_session_id"],
                ),
            )
            conn.execute(
                """
                INSERT INTO runtime_cursors(
                  cursor_id, session_id, current_scientific_state_id, lease_owner,
                  lease_expires_at, fencing_token, observed_status
                ) VALUES (%s, %s, %s, %s, now() + interval '5 minutes', 1, 'READY')
                """,
                (cursor_id, session_id, child.scientific_state_id, child.lease_owner),
            )
            children.append(
                {
                    "sibling_index": child.sibling_index,
                    "session_id": session_id,
                    "runtime_instance_id": runtime_instance_id,
                    "cursor_id": cursor_id,
                    "current_state_id": child.scientific_state_id,
                    "version": 0,
                    "fencing_token": 1,
                    "rng_derivation": child.rng_derivation,
                    "rng_state": derived_rng,
                    "logical_state_digest": snapshot["logical_state_digest"],
                }
            )
        return {
            "operational_snapshot_id": request.operational_snapshot_id,
            "obtained_fidelity": "logical_restore",
            "children": children,
        }

    return _run_operation(request, "fork_snapshot", "cad.session@1", worker)


def _invoke_judge(
    *,
    proof_bundle: dict[str, Any],
    verification_run_id: str,
    subject_id: str,
    fixture_scenario: str,
    judge_spec_version: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    spec = pointwise_spec(version=judge_spec_version)
    validate_contract("JudgeSpec", spec)
    evidence = [
        proof_bundle[role.replace("-", "_")]
        for role in spec["allowed_evidence_roles"]
        if role.replace("-", "_") in proof_bundle
    ]
    evidence_manifest = {
        "proof_bundle_digest": proof_bundle["digest"],
        "evidence": evidence,
    }
    template = prompt_registry()[spec["prompt_template_id"]]
    rendered_prompt = template.render(evidence_manifest)
    request = {
        "judge_spec": spec,
        "proof_bundle_digest": proof_bundle["digest"],
        "evidence": evidence,
        "rendered_prompt": rendered_prompt,
        "fixture_scenario": fixture_scenario,
        "tools": [],
        "model_tool_access": "disabled",
        "writable_systems": [],
        "subject_blind_label": canonical_digest(subject_id)[:20],
    }
    provider_attempts = 0
    provider_response = None
    for provider_attempts in range(1, 3):
        try:
            provider_response = judge_provider.invoke(request, attempt_number=provider_attempts)
            break
        except RetryableJudgeError:
            if provider_attempts == 2:
                raise
    assert provider_response is not None

    request_artifact = artifact_store.put_bytes(
        rendered_prompt.encode("utf-8"),
        role="judge-rendered-request",
        media_type="text/plain",
        visibility="OPERATOR",
        trust_class="PLATFORM_DERIVED",
    )
    raw_artifact = artifact_store.put_bytes(
        provider_response.raw.encode(),
        role="judge-raw-output",
        media_type="application/json",
        visibility="OPERATOR",
        trust_class="MODEL_OUTPUT_UNVALIDATED",
    )
    try:
        parsed = json.loads(provider_response.raw)
        outcome = parsed["outcome"]
    except (json.JSONDecodeError, KeyError, TypeError):
        parsed = {
            "outcome": "INVALID_RESULT",
            "assessments": [],
            "confidence": 0,
            "abstained": False,
            "abstention_reason": None,
            "disagreement": False,
            "integrity_flags": [],
            "tie": False,
            "explanation": "Fixture provider output did not conform to judge-result.v1.",
        }
        outcome = "INVALID_RESULT"

    parsed_artifact = artifact_store.put_json(
        parsed,
        role="judge-parsed-output",
        visibility="OPERATOR",
        trust_class="MODEL_OUTPUT_UNVALIDATED",
    )
    result = {
        "judge_result_id": make_id("judge_result"),
        "judge_spec_id": spec["judge_spec_id"],
        "proof_bundle_digest": proof_bundle["digest"],
        "outcome": outcome,
        "assessments": parsed.get("assessments", []),
        "preferences": parsed.get("preferences", []),
        "ranking": parsed.get("ranking", []),
        "tie": parsed.get("tie", False),
        "confidence": parsed.get("confidence", 0),
        "abstained": parsed.get("abstained", False),
        "abstention_reason": parsed.get("abstention_reason"),
        "disagreement": parsed.get("disagreement", False),
        "integrity_flags": parsed.get("integrity_flags", []),
        "explanation": parsed.get("explanation", "")[:500],
        "provider_model_identity": provider_response.model_identity,
        "raw_output_artifact": raw_artifact.ref(ordinal=0),
        "parsed_output_artifact": parsed_artifact.ref(ordinal=0),
        "usage": provider_response.usage,
    }
    validate_judge_result(result, spec=spec, proof_digest=proof_bundle["digest"])
    return (
        result,
        {
            "spec": spec,
            "request": _stored_payload(request_artifact),
            "raw_output": _stored_payload(raw_artifact),
            "parsed_output": _stored_payload(parsed_artifact),
            "request_digest": canonical_digest(request),
            "verification_run_id": verification_run_id,
        },
        provider_attempts,
    )


@app.post("/v1/verification-runs/transition")
def verify_transition(request: VerifyTransitionRequest) -> dict[str, Any]:
    def worker(_conn: Any) -> dict[str, Any]:
        geometry = cad.geometry_report(request.candidate_state)
        source_progress = cad.progress_score(request.source_state)
        candidate_progress = cad.progress_score(request.candidate_state)
        reference_metadata = artifact_store.put_json(
            cad.task_reference(),
            role="task-reference-metadata",
            visibility="OPERATOR",
            trust_class="REFERENCE_AUTHORITY",
        )
        reference_artifact = artifact_store.put_bytes(
            cad.render_svg(cad.TARGET_STATE, label="Task reference"),
            role="task-reference",
            media_type="image/svg+xml",
            visibility="OPERATOR",
            trust_class="REFERENCE_AUTHORITY",
            viewer_hint="image-comparison",
        )
        source_render = artifact_store.put_bytes(
            cad.render_svg(request.source_state, label="Source state"),
            role="source-render",
            media_type="image/svg+xml",
            visibility="OPERATOR",
            trust_class="PLATFORM_DERIVED",
            viewer_hint="image-comparison",
        )
        candidate_render = artifact_store.put_bytes(
            cad.render_svg(request.candidate_state, label="Candidate state"),
            role="candidate-render",
            media_type="image/svg+xml",
            visibility="OPERATOR",
            trust_class="PLATFORM_DERIVED",
            viewer_hint="image-comparison",
        )
        action_summary = artifact_store.put_json(
            {"action": request.action, "source_turn": request.source_state["turn"]},
            role="action-summary",
            trust_class="CANDIDATE_CONTROLLED",
        )
        geometry_artifact = artifact_store.put_json(
            geometry,
            role="geometry-report",
            trust_class="HIDDEN_VERIFIER",
            viewer_hint="geometry-report",
        )
        artifacts = [
            reference_metadata,
            reference_artifact,
            source_render,
            candidate_render,
            action_summary,
            geometry_artifact,
        ]
        proof_content = {
            "subject_type": "TRANSITION",
            "subject_id": request.subject_id,
            "task_reference_metadata": reference_metadata.ref(ordinal=0),
            "task_reference": reference_artifact.ref(ordinal=0),
            "source_render": source_render.ref(ordinal=0),
            "candidate_render": candidate_render.ref(ordinal=0),
            "action_summary": action_summary.ref(ordinal=0),
            "geometry_report": geometry_artifact.ref(ordinal=0),
            "renderer": {
                "version": "cad-fixture-renderer@1",
                "camera": "isometric-orthographic@1",
                "material": "drafting-blue@1",
                "lighting": "three-point-neutral@1",
                "resolution": [640, 480],
                "views": ["top", "isometric"],
            },
            "ordered_artifact_digests": [artifact.digest for artifact in artifacts],
        }
        proof_bundle = {
            "proof_bundle_id": make_id("proof"),
            **proof_content,
            "digest": canonical_digest(proof_content),
        }
        validate_contract("ProofBundle", proof_bundle)
        proof_artifact = artifact_store.put_json(
            proof_bundle,
            role="proof-bundle",
            trust_class="PLATFORM_DERIVED",
            viewer_hint="structured-json",
        )
        judge_result, judge_meta, provider_attempts = _invoke_judge(
            proof_bundle=proof_bundle,
            verification_run_id=request.verification_run_id,
            subject_id=request.subject_id,
            fixture_scenario=request.fixture_scenario,
            judge_spec_version=request.judge_spec_version,
        )

        steps = [
            {
                "step_id": "geometry",
                "step_type": "deterministic_check",
                "status": "CANDIDATE_FAILED" if geometry["candidate_failure"] else "SUCCEEDED",
                "attempt_count": 1,
                "cache_status": "MISS",
                "evidence_roles": ["geometry-report"],
                "metrics": geometry,
                "failure": None,
                "cost": {"cpu_seconds": 0.01, "credits": 0.001},
            },
            {
                "step_id": "render",
                "step_type": "evidence_producer",
                "status": "SUCCEEDED",
                "attempt_count": 2 if request.simulate_infra_retry else 1,
                "cache_status": "MISS",
                "evidence_roles": ["task-reference", "source-render", "candidate-render"],
                "metrics": {},
                "failure": (
                    {"code": "FIXTURE_RENDER_WORKER_LOST", "retryable": True, "recovered": True}
                    if request.simulate_infra_retry
                    else None
                ),
                "cost": {"cpu_seconds": 0.03, "credits": 0.003},
            },
            {
                "step_id": "proof",
                "step_type": "deterministic_aggregation",
                "status": "SUCCEEDED",
                "attempt_count": 1,
                "cache_status": "MISS",
                "evidence_roles": ["proof-bundle"],
                "metrics": {},
                "failure": None,
                "cost": {"cpu_seconds": 0.002, "credits": 0.0002},
            },
            {
                "step_id": "pointwise-judge",
                "step_type": "llm_judge",
                "status": judge_result["outcome"],
                "attempt_count": provider_attempts,
                "cache_status": "MISS",
                "evidence_roles": [
                    "task-reference",
                    "source-render",
                    "candidate-render",
                    "geometry-report",
                ],
                "metrics": {},
                "failure": None,
                "cost": {
                    "judge_tokens": sum(
                        judge_result["usage"][key] for key in ("input_tokens", "output_tokens")
                    ),
                    "credits": 0,
                },
            },
            {
                "step_id": "aggregate",
                "step_type": "deterministic_aggregation",
                "status": "SUCCEEDED",
                "attempt_count": 1,
                "cache_status": "MISS",
                "evidence_roles": ["geometry-report", "judge-parsed-output"],
                "metrics": {
                    "source_progress": source_progress,
                    "candidate_progress": candidate_progress,
                    "progress_delta": round(candidate_progress - source_progress, 4),
                },
                "failure": None,
                "cost": {"cpu_seconds": 0.001, "credits": 0.0001},
            },
        ]
        return {
            "verification_run_id": request.verification_run_id,
            "status": "SUCCEEDED",
            "steps": steps,
            "proof_bundle": proof_bundle,
            "proof_artifact": _stored_payload(proof_artifact),
            "artifacts": [_stored_payload(artifact) for artifact in artifacts]
            + [judge_meta["request"], judge_meta["raw_output"], judge_meta["parsed_output"]],
            "judge_result": judge_result,
            "judge": judge_meta,
            "deterministic_metrics": {
                "geometry_valid": geometry["geometry_valid"],
                "constraint_compliance": geometry["constraints_passed"],
                "reference_alignment": geometry["feature_correspondence"],
                "progress_from_source_state": round(candidate_progress - source_progress, 4),
                "regression_penalty": -0.35 if not geometry["geometry_valid"] else 0,
                "terminal_quality": candidate_progress
                if request.candidate_state["submitted"]
                else 0,
                "execution_cost": 0.0142,
            },
        }

    return _run_operation(
        request,
        "verify_transition",
        "cad.proof-bundle@1",
        worker,
        simulate_infra_retry=request.simulate_infra_retry,
        detached=True,
    )


@app.post("/v1/verification-runs/rejudge")
def rejudge(request: RejudgeRequest) -> dict[str, Any]:
    def worker(_conn: Any) -> dict[str, Any]:
        with connection() as conn:
            original = conn.execute(
                """
                SELECT result
                FROM operations
                WHERE operation_type = 'verify_transition'
                  AND status = 'SUCCEEDED'
                  AND result->'proof_bundle'->>'digest' = %s
                ORDER BY created_at
                LIMIT 1
                """,
                (request.proof_bundle_digest,),
            ).fetchone()
        if not original:
            raise HTTPException(
                status_code=404,
                detail={"code": "PROOF_BUNDLE_NOT_FOUND"},
            )
        proof_bundle = original["result"]["proof_bundle"]
        validate_contract("ProofBundle", proof_bundle)
        for field in (
            "task_reference_metadata",
            "task_reference",
            "source_render",
            "candidate_render",
            "action_summary",
            "geometry_report",
        ):
            artifact_store.get_bytes(proof_bundle[field]["digest"])
        judge_result, judge_meta, provider_attempts = _invoke_judge(
            proof_bundle=proof_bundle,
            verification_run_id=request.verification_run_id,
            subject_id=proof_bundle["subject_id"],
            fixture_scenario=request.fixture_scenario,
            judge_spec_version=request.judge_spec_version,
        )
        step = {
            "step_id": "pointwise-judge",
            "step_type": "llm_judge",
            "status": judge_result["outcome"],
            "attempt_count": provider_attempts,
            "cache_status": "MISS",
            "evidence_roles": [
                "task-reference",
                "source-render",
                "candidate-render",
                "geometry-report",
            ],
            "metrics": {},
            "failure": None,
            "cost": {
                "judge_tokens": sum(
                    judge_result["usage"][key] for key in ("input_tokens", "output_tokens")
                ),
                "credits": judge_result["usage"]["cost"],
            },
        }
        return {
            "verification_run_id": request.verification_run_id,
            "status": judge_result["outcome"],
            "steps": [step],
            "proof_bundle": proof_bundle,
            "judge_result": judge_result,
            "judge": judge_meta,
            "artifacts": [
                judge_meta["request"],
                judge_meta["raw_output"],
                judge_meta["parsed_output"],
            ],
            "evidence_reused": True,
            "execution_steps_rerun": 0,
            "render_steps_rerun": 0,
            "geometry_steps_rerun": 0,
        }

    return _run_operation(
        request,
        "rejudge",
        "judge.multimodal-rubric@1",
        worker,
        detached=True,
    )


@app.post("/v1/verification-runs/group")
def judge_group(request: GroupJudgeRequest) -> dict[str, Any]:
    def worker(_conn: Any) -> dict[str, Any]:
        spec = group_spec()
        validate_contract("JudgeSpec", spec)
        order = blinded_order(request.subject_id, len(request.sibling_proofs))
        blinded = [
            {
                "label": f"candidate-{chr(65 + position)}",
                "proof_bundle_digest": request.sibling_proofs[index]["proof_bundle"]["digest"],
                "progress": request.sibling_proofs[index]["progress"],
                "original_index": index,
                "branch_member_id": request.sibling_proofs[index]["branch_member_id"],
                "sibling_index": request.sibling_proofs[index]["sibling_index"],
            }
            for position, index in enumerate(order)
        ]
        ranked = sorted(blinded, key=lambda value: (-value["progress"], value["label"]))
        tie = len(ranked) > 1 and ranked[0]["progress"] == ranked[1]["progress"]
        parsed = {
            "outcome": "SUCCEEDED",
            "assessments": [
                {
                    "criterion": "sibling_preference",
                    "score": round(ranked[0]["progress"], 4),
                    "confidence": 0.9,
                    "evidence_roles": ["task-reference", "source-render", "sibling-render"],
                },
                {
                    "criterion": "progress",
                    "score": round(sum(item["progress"] for item in ranked) / len(ranked), 4),
                    "confidence": 0.9,
                    "evidence_roles": ["sibling-render", "geometry-report"],
                },
                {
                    "criterion": "regressions",
                    "score": round(1 - min(item["progress"] for item in ranked), 4),
                    "confidence": 0.86,
                    "evidence_roles": ["sibling-render", "geometry-report"],
                },
            ],
            "preferences": [] if tie else [ranked[0]["label"]],
            "ranking": [item["label"] for item in ranked],
            "tie": tie,
            "confidence": 0.9,
            "abstained": False,
            "abstention_reason": None,
            "disagreement": False,
            "integrity_flags": [],
            "explanation": (
                "Deterministic fixture ranking compares blinded sibling proof bundles "
                "using the persisted fixture progress signal."
            ),
        }
        raw_artifact = artifact_store.put_json(
            parsed,
            role="judge-raw-output",
            visibility="OPERATOR",
            trust_class="MODEL_OUTPUT_UNVALIDATED",
        )
        parsed_artifact = artifact_store.put_json(
            parsed,
            role="judge-parsed-output",
            visibility="OPERATOR",
            trust_class="MODEL_OUTPUT_UNVALIDATED",
        )
        group_manifest = {
            "subject_id": request.subject_id,
            "presentation_order": [item["label"] for item in blinded],
            "blinded_proof_digests": [item["proof_bundle_digest"] for item in blinded],
            "member_bindings": [
                {
                    "label": item["label"],
                    "branch_member_id": item["branch_member_id"],
                    "sibling_index": item["sibling_index"],
                    "proof_bundle_digest": item["proof_bundle_digest"],
                }
                for item in blinded
            ],
            "judge_spec_id": spec["judge_spec_id"],
        }
        proof_digest = canonical_digest(group_manifest)
        judge_result = {
            "judge_result_id": make_id("judge_result"),
            "judge_spec_id": spec["judge_spec_id"],
            "proof_bundle_digest": proof_digest,
            **parsed,
            "provider_model_identity": "deterministic-group-ranking-fixture@1",
            "raw_output_artifact": raw_artifact.ref(ordinal=0),
            "parsed_output_artifact": parsed_artifact.ref(ordinal=0),
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "images": 0,
                "latency_ms": 0,
                "cost": 0,
            },
        }
        validate_judge_result(
            judge_result,
            spec=spec,
            proof_digest=proof_digest,
            group_labels={item["label"] for item in blinded},
        )
        return {
            "verification_run_id": request.verification_run_id,
            "status": "SUCCEEDED",
            "judge_spec": spec,
            "judge_result": judge_result,
            "group_manifest": group_manifest,
            "presentation_order": order,
            "artifacts": [_stored_payload(raw_artifact), _stored_payload(parsed_artifact)],
            "steps": [
                {
                    "step_id": "group-judge",
                    "step_type": "deterministic_group_ranking_fixture",
                    "status": "SUCCEEDED",
                    "attempt_count": 1,
                    "cache_status": "MISS",
                    "evidence_roles": ["task-reference", "source-render", "sibling-render"],
                    "metrics": {},
                    "failure": None,
                    "cost": {"judge_tokens": 1060, "credits": 0},
                }
            ],
        }

    return _run_operation(
        request,
        "group_judge",
        "judge.branch-group@1",
        worker,
        detached=True,
    )


@app.post("/v1/sessions/cancel")
def cancel_session(request: CursorActionRequest) -> dict[str, Any]:
    def worker(conn: Any) -> dict[str, Any]:
        cursor = conn.execute(
            "SELECT * FROM runtime_cursors WHERE cursor_id = %s FOR UPDATE",
            (request.cursor_id,),
        ).fetchone()
        if not cursor:
            return {"cursor_id": request.cursor_id, "already_released": True}
        conn.execute(
            "UPDATE runtime_cursors SET observed_status = 'CANCELED', updated_at = now() WHERE cursor_id = %s",
            (request.cursor_id,),
        )
        conn.execute(
            "UPDATE sessions SET status = 'CANCELED', updated_at = now() WHERE session_id = %s",
            (cursor["session_id"],),
        )
        return {"cursor_id": request.cursor_id, "released": True}

    return _run_operation(request, "cancel_session", "cad.session@1", worker)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND"})
        attempts = list(
            conn.execute(
                "SELECT * FROM attempts WHERE job_id = %s ORDER BY attempt_number", (job_id,)
            )
        )
    return {"job": job, "attempts": attempts}


@app.get("/v1/capacity")
def capacity() -> dict[str, Any]:
    with connection() as conn:
        active = conn.execute(
            "SELECT count(*) AS count FROM jobs WHERE status IN ('SUBMITTED', 'LEASED', 'RUNNING')"
        ).fetchone()["count"]
        sessions = conn.execute(
            "SELECT count(*) AS count FROM sessions WHERE status = 'READY'"
        ).fetchone()["count"]
    return {
        "resource_classes": {
            "fixture-worker": {
                "capacity": EXECUTION_CONCURRENCY,
                "active": active,
                "available": max(0, EXECUTION_CONCURRENCY - active),
            },
            "external-model-api": {"capacity": 0, "active": 0},
        },
        "live_sessions": sessions,
    }
