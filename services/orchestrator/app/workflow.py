from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from equinox_core import canonical_digest, make_id, utc_now, validate_contract
from psycopg.types.json import Jsonb

from .database import connection
from .execution_client import RetryableExecutionError, execution_client
from .science import (
    accept_operation_result,
    artifact_store,
    emit_event,
    record_artifact,
    record_operation_intent,
    store_json_artifact,
)

VERIFICATION_PLAN_ID = "cad.transition-composite@1"
REWARD_PIPELINE_ID = "cad-local-rewards@1"
TASK_REVISION = "mounting-plate@sha256:fixture-v1"


class RunCanceled(RuntimeError):
    pass


@dataclass
class Cursor:
    cursor_id: str
    state_id: str
    version: int
    fencing_token: int


@dataclass
class TransitionEvidence:
    transition_id: str
    destination_state_id: str
    verification_run_id: str
    proof_bundle_id: str
    proof_bundle: dict[str, Any]
    judge_result_id: str
    reward_signal_ids: list[str]
    progress: float
    cursor: Cursor


def _run_status(run_id: str) -> dict[str, Any]:
    with connection() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    if not run:
        raise ValueError(f"run not found: {run_id}")
    return run


def _ensure_running(run_id: str) -> None:
    run = _run_status(run_id)
    if run["desired_state"] == "CANCELED":
        raise RunCanceled(run_id)


def _fixture_step_delay() -> None:
    delay_ms = int(os.getenv("WORKFLOW_STEP_DELAY_MS", "0"))
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)


def _authorize(
    *,
    run_id: str,
    operation_id: str,
    operation_type: str,
    operation_input: dict[str, Any],
    expected_version: int = 0,
    budget_reservation_id: str | None = None,
) -> None:
    with connection() as conn:
        record_operation_intent(
            conn,
            operation_id=operation_id,
            run_id=run_id,
            operation_type=operation_type,
            operation_input=operation_input,
            expected_version=expected_version,
            budget_reservation_id=budget_reservation_id,
        )


def _create_initial_state(
    *,
    run_id: str,
    rollout_tree_id: str,
    lease_owner: str,
    seed: int,
) -> tuple[Cursor, str]:
    with connection() as conn:
        intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'create_session'
              AND operation_input->>'lease_owner' = %s
            ORDER BY created_at LIMIT 1
            """,
            (run_id, lease_owner),
        ).fetchone()
    if intent:
        operation_id = intent["operation_id"]
        operation_input = intent["operation_input"]
        state_id = operation_input["scientific_state_id"]
    else:
        state_id = make_id("state")
        operation_id = make_id("op")
        operation_input = {
            "task_revision": TASK_REVISION,
            "scientific_state_id": state_id,
            "lease_owner": lease_owner,
            "rng_state": f"seed:{seed}",
        }
        _authorize(
            run_id=run_id,
            operation_id=operation_id,
            operation_type="create_session",
            operation_input=operation_input,
        )
    result = execution_client.operation(
        "/v1/sessions",
        "create_session",
        operation_input,
        operation_id=operation_id,
        idempotency_key=operation_id,
        correlation_id=run_id,
    )
    logical_state = result["logical_state"]
    with connection() as conn:
        accept_operation_result(conn, operation_id=operation_id, result=result)
        logical_artifact_id, _ = store_json_artifact(
            conn,
            logical_state,
            role="logical-state",
            entity_type="state",
            entity_id=state_id,
            trust_class="CANDIDATE",
        )
        observation = {
            "turn": logical_state["turn"],
            "feature_count": len(logical_state["features"]),
            "submitted": logical_state["submitted"],
        }
        observation_artifact_id, _ = store_json_artifact(
            conn,
            observation,
            role="observation",
            entity_type="state",
            entity_id=state_id,
            visibility="POLICY",
            trust_class="CANDIDATE",
        )
        state_contract = {
            "state_id": state_id,
            "rollout_tree_id": rollout_tree_id,
            "parent_transition_id": None,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": TASK_REVISION,
            "logical_state": {
                "artifact_id": logical_artifact_id,
                "digest": canonical_digest(logical_state),
                "role": "logical-state",
                "ordinal": 0,
                "media_type": "application/json",
                "viewer_hint": "structured-json",
                "visibility": "OPERATOR",
                "trust_class": "CANDIDATE",
            },
            "observation": {
                "artifact_id": observation_artifact_id,
                "digest": canonical_digest(observation),
                "role": "observation",
                "ordinal": 0,
                "media_type": "application/json",
                "viewer_hint": "structured-json",
                "visibility": "POLICY",
                "trust_class": "CANDIDATE",
            },
            "semantic_status": "ACTIVE",
            "sequence": 0,
        }
        validate_contract("State", state_contract)
        conn.execute(
            """
            INSERT INTO states(
              state_id, rollout_tree_id, parent_transition_id, environment_version,
              task_revision, logical_state_artifact_id, observation_artifact_id,
              logical_state_digest, semantic_status, sequence, payload
            ) VALUES (%s, %s, NULL, 'cad.reconstruction@1.0.0', %s, %s, %s, %s, 'ACTIVE', 0, %s)
            ON CONFLICT (state_id) DO NOTHING
            """,
            (
                state_id,
                rollout_tree_id,
                TASK_REVISION,
                logical_artifact_id,
                observation_artifact_id,
                canonical_digest(logical_state),
                Jsonb(state_contract),
            ),
        )
        conn.execute(
            "UPDATE rollout_trees SET root_state_id = %s WHERE rollout_tree_id = %s",
            (state_id, rollout_tree_id),
        )
        emit_event(
            conn,
            event_type="state.accepted",
            aggregate_type="rollout_tree",
            aggregate_id=rollout_tree_id,
            run_id=run_id,
            correlation_id=operation_id,
            payload={"state_id": state_id, "sequence": 0},
        )
    return (
        Cursor(
            cursor_id=result["cursor_id"],
            state_id=state_id,
            version=result["version"],
            fencing_token=result["fencing_token"],
        ),
        state_id,
    )


def _state_logical(state_id: str) -> dict[str, Any]:
    with connection() as conn:
        state = conn.execute(
            "SELECT payload FROM states WHERE state_id = %s", (state_id,)
        ).fetchone()
        artifact = conn.execute(
            """
            SELECT a.digest FROM artifacts a
            JOIN artifact_refs r ON r.artifact_id = a.artifact_id
            WHERE r.entity_type = 'state' AND r.entity_id = %s AND r.role = 'logical-state'
            """,
            (state_id,),
        ).fetchone()
    if not state or not artifact:
        raise ValueError(f"state not found: {state_id}")
    return json_load_artifact(artifact["digest"])


def json_load_artifact(digest: str) -> Any:
    import json

    return json.loads(artifact_store.get_bytes(digest))


def _accept_verification(
    *,
    run_id: str,
    transition_id: str,
    result: dict[str, Any],
    operation_id: str,
    rejudges_verification_run_id: str | None = None,
    create_rewards: bool = True,
) -> tuple[str, str, list[str], float]:
    verification_run_id = result["verification_run_id"]
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT proof_bundle_id FROM verification_runs
            WHERE verification_run_id = %s
            """,
            (verification_run_id,),
        ).fetchone()
        if existing:
            accepted_judge = conn.execute(
                """
                SELECT jr.judge_result_id FROM judge_results jr
                JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
                WHERE ji.verification_run_id = %s
                ORDER BY jr.created_at LIMIT 1
                """,
                (verification_run_id,),
            ).fetchone()
            reward_ids = [
                row["reward_signal_id"]
                for row in conn.execute(
                    """
                    SELECT reward_signal_id FROM reward_signals
                    WHERE subject_type = 'TRANSITION' AND subject_id = %s
                    ORDER BY created_at
                    """,
                    (transition_id,),
                )
            ]
            progress = conn.execute(
                """
                SELECT value FROM metric_observations
                WHERE subject_id = %s AND descriptor = 'deterministic.terminal_quality'
                ORDER BY created_at LIMIT 1
                """,
                (transition_id,),
            ).fetchone()
            return (
                existing["proof_bundle_id"],
                accepted_judge["judge_result_id"],
                reward_ids,
                float(progress["value"]) if progress else 0,
            )
    proof_bundle = result["proof_bundle"]
    proof_bundle_id = proof_bundle["proof_bundle_id"]
    judge_result = result["judge_result"]
    judge_result_id = judge_result["judge_result_id"]
    now = utc_now()
    computed_proof_digest = canonical_digest(
        {key: value for key, value in proof_bundle.items() if key != "digest"}
    )
    if computed_proof_digest != proof_bundle["digest"]:
        raise ValueError(
            f"proof bundle digest mismatch: expected {computed_proof_digest}, "
            f"received {proof_bundle['digest']}"
        )
    with connection() as conn:
        accept_operation_result(conn, operation_id=operation_id, result=result)
        conn.execute(
            """
            INSERT INTO verification_runs(
              verification_run_id, run_id, subject_type, subject_id, plan_id,
              plan_version, status, proof_bundle_id, rejudges_verification_run_id,
              completed_at
            ) VALUES (%s, %s, 'TRANSITION', %s, %s, 1, %s, %s, %s, %s)
            ON CONFLICT (verification_run_id) DO NOTHING
            """,
            (
                verification_run_id,
                run_id,
                transition_id,
                VERIFICATION_PLAN_ID,
                result["status"],
                proof_bundle_id,
                rejudges_verification_run_id,
                now,
            ),
        )
        for artifact in result["artifacts"]:
            record_artifact(
                conn,
                artifact,
                entity_type="verification_run",
                entity_id=verification_run_id,
                role=artifact["role"],
            )
        if result.get("proof_artifact"):
            record_artifact(
                conn,
                result["proof_artifact"],
                entity_type="evidence_bundle",
                entity_id=proof_bundle_id,
                role="proof-bundle",
            )
        conn.execute(
            """
            INSERT INTO evidence_bundles(
              proof_bundle_id, verification_run_id, subject_type, subject_id, digest, manifest
            ) VALUES (%s, %s, 'TRANSITION', %s, %s, %s)
            ON CONFLICT (proof_bundle_id) DO NOTHING
            """,
            (
                proof_bundle_id,
                verification_run_id,
                transition_id,
                proof_bundle["digest"],
                Jsonb(proof_bundle),
            ),
        )

        step_ids: dict[str, str] = {}
        for step in result["steps"]:
            step_run_id = make_id("step_run")
            conn.execute(
                """
                INSERT INTO verifier_step_runs(
                  step_run_id, verification_run_id, step_id, step_type, status,
                  attempt_count, cache_status, evidence_roles, metrics, failure, cost,
                  started_at, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (verification_run_id, step_id) DO NOTHING
                """,
                (
                    step_run_id,
                    verification_run_id,
                    step["step_id"],
                    step["step_type"],
                    step["status"],
                    step["attempt_count"],
                    step["cache_status"],
                    Jsonb(step["evidence_roles"]),
                    Jsonb(step["metrics"]),
                    Jsonb(step["failure"]) if step["failure"] else None,
                    Jsonb(step["cost"]),
                    now,
                    now,
                ),
            )
            step_ids[step["step_id"]] = conn.execute(
                """
                SELECT step_run_id FROM verifier_step_runs
                WHERE verification_run_id = %s AND step_id = %s
                """,
                (verification_run_id, step["step_id"]),
            ).fetchone()["step_run_id"]
        actual_judge_step = conn.execute(
            """
            SELECT step_run_id FROM verifier_step_runs
            WHERE verification_run_id = %s AND step_id IN ('pointwise-judge', 'group-judge')
            ORDER BY step_id LIMIT 1
            """,
            (verification_run_id,),
        ).fetchone()
        spec = result["judge"]["spec"]
        spec_digest = canonical_digest(spec)
        conn.execute(
            """
            INSERT INTO judge_specs(judge_spec_id, provider_name, digest, spec)
            VALUES (%s, 'MockJudgeProvider', %s, %s)
            ON CONFLICT (judge_spec_id) DO NOTHING
            """,
            (spec["judge_spec_id"], spec_digest, Jsonb(spec)),
        )
        invocation_id = make_id("judge_invocation")
        conn.execute(
            """
            INSERT INTO judge_invocations(
              judge_invocation_id, verification_run_id, step_run_id, judge_spec_id,
              proof_bundle_digest, sample_index, status, raw_output_artifact_id,
              accepted_result_id, usage
            ) VALUES (%s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            ON CONFLICT (step_run_id, judge_spec_id, proof_bundle_digest, sample_index) DO NOTHING
            """,
            (
                invocation_id,
                verification_run_id,
                actual_judge_step["step_run_id"],
                spec["judge_spec_id"],
                proof_bundle["digest"],
                judge_result["outcome"],
                judge_result["raw_output_artifact"]["artifact_id"],
                judge_result_id,
                Jsonb(judge_result["usage"]),
            ),
        )
        actual_invocation = conn.execute(
            """
            SELECT judge_invocation_id FROM judge_invocations
            WHERE step_run_id = %s AND judge_spec_id = %s
              AND proof_bundle_digest = %s AND sample_index = 0
            """,
            (actual_judge_step["step_run_id"], spec["judge_spec_id"], proof_bundle["digest"]),
        ).fetchone()["judge_invocation_id"]
        conn.execute(
            """
            INSERT INTO judge_results(
              judge_result_id, judge_invocation_id, outcome, result
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (judge_invocation_id) DO NOTHING
            """,
            (judge_result_id, actual_invocation, judge_result["outcome"], Jsonb(judge_result)),
        )

        metric_ids: dict[str, str] = {}
        deterministic = result.get("deterministic_metrics", {})
        metric_step = step_ids.get("aggregate") or next(iter(step_ids.values()))
        for descriptor, value in deterministic.items():
            metric_id = make_id("metric")
            unit = (
                "boolean"
                if isinstance(value, bool)
                else ("credits" if descriptor == "execution_cost" else "ratio")
            )
            conn.execute(
                """
                INSERT INTO metric_observations(
                  metric_observation_id, run_id, subject_type, subject_id, descriptor,
                  value, unit, source_step_run_id, accepted
                ) VALUES (%s, %s, 'TRANSITION', %s, %s, %s, %s, %s, true)
                ON CONFLICT (source_step_run_id, descriptor, subject_id) DO NOTHING
                """,
                (
                    metric_id,
                    run_id,
                    transition_id,
                    f"deterministic.{descriptor}",
                    Jsonb(value),
                    unit,
                    metric_step,
                ),
            )
            actual = conn.execute(
                """
                SELECT metric_observation_id FROM metric_observations
                WHERE source_step_run_id = %s AND descriptor = %s AND subject_id = %s
                """,
                (metric_step, f"deterministic.{descriptor}", transition_id),
            ).fetchone()["metric_observation_id"]
            metric_ids[descriptor] = actual

        if judge_result["outcome"] == "SUCCEEDED":
            for assessment in judge_result["assessments"]:
                descriptor = f"judge.{assessment['criterion']}"
                metric_id = make_id("metric")
                conn.execute(
                    """
                    INSERT INTO metric_observations(
                      metric_observation_id, run_id, subject_type, subject_id, descriptor,
                      value, unit, source_step_run_id, judge_result_id, accepted
                    ) VALUES (%s, %s, 'TRANSITION', %s, %s, %s, 'ratio', %s, %s, true)
                    ON CONFLICT (source_step_run_id, descriptor, subject_id) DO NOTHING
                    """,
                    (
                        metric_id,
                        run_id,
                        transition_id,
                        descriptor,
                        Jsonb(assessment["score"]),
                        actual_judge_step["step_run_id"],
                        judge_result_id,
                    ),
                )
                metric_ids[descriptor] = conn.execute(
                    """
                    SELECT metric_observation_id FROM metric_observations
                    WHERE source_step_run_id = %s AND descriptor = %s AND subject_id = %s
                    """,
                    (actual_judge_step["step_run_id"], descriptor, transition_id),
                ).fetchone()["metric_observation_id"]

        reward_sources: dict[str, tuple[float, list[str]]] = {}
        if create_rewards and deterministic:
            reward_sources.update(
                {
                    "geometry_valid": (
                        1.0 if deterministic.get("geometry_valid") else 0.0,
                        [metric_ids["geometry_valid"]],
                    ),
                    "constraint_compliance": (
                        1.0 if deterministic.get("constraint_compliance") else 0.0,
                        [metric_ids["constraint_compliance"]],
                    ),
                    "regression_penalty": (
                        float(deterministic.get("regression_penalty", 0)),
                        [metric_ids["regression_penalty"]],
                    ),
                    "terminal_quality": (
                        float(deterministic.get("terminal_quality", 0)),
                        [metric_ids["terminal_quality"]],
                    ),
                    "execution_cost": (
                        -float(deterministic.get("execution_cost", 0)),
                        [metric_ids["execution_cost"]],
                    ),
                }
            )
        if create_rewards and judge_result["outcome"] == "SUCCEEDED":
            alignment_key = "judge.reference_correspondence"
            progress_key = "judge.progress_from_source_state"
            assessment_by_name = {
                item["criterion"]: float(item["score"]) for item in judge_result["assessments"]
            }
            reward_sources["reference_alignment"] = (
                assessment_by_name["reference_correspondence"],
                [metric_ids[alignment_key]],
            )
            reward_sources["progress_from_source_state"] = (
                assessment_by_name["progress_from_source_state"],
                [metric_ids[progress_key]],
            )

        reward_signal_ids: list[str] = []
        for name, (value, source_ids) in reward_sources.items():
            reward_id = make_id("reward")
            reward_contract = {
                "reward_signal_id": reward_id,
                "subject_type": "TRANSITION",
                "subject_id": transition_id,
                "name": name,
                "value": value,
                "reward_pipeline_id": REWARD_PIPELINE_ID,
                "metric_observation_ids": source_ids,
            }
            validate_contract("RewardSignal", reward_contract)
            conn.execute(
                """
                INSERT INTO reward_signals(
                  reward_signal_id, run_id, subject_type, subject_id, name, value,
                  reward_pipeline_id, metric_observation_ids
                ) VALUES (%s, %s, 'TRANSITION', %s, %s, %s, %s, %s)
                ON CONFLICT (subject_id, name, reward_pipeline_id) DO NOTHING
                """,
                (
                    reward_id,
                    run_id,
                    transition_id,
                    name,
                    value,
                    REWARD_PIPELINE_ID,
                    Jsonb(source_ids),
                ),
            )
            actual_reward = conn.execute(
                """
                SELECT reward_signal_id FROM reward_signals
                WHERE subject_id = %s AND name = %s AND reward_pipeline_id = %s
                """,
                (transition_id, name, REWARD_PIPELINE_ID),
            ).fetchone()["reward_signal_id"]
            reward_signal_ids.append(actual_reward)
        emit_event(
            conn,
            event_type="verification.accepted",
            aggregate_type="verification_run",
            aggregate_id=verification_run_id,
            run_id=run_id,
            correlation_id=operation_id,
            payload={
                "verification_run_id": verification_run_id,
                "proof_bundle_id": proof_bundle_id,
                "judge_result_id": judge_result_id,
                "judge_outcome": judge_result["outcome"],
                "reward_signal_ids": reward_signal_ids,
            },
        )
    return (
        proof_bundle_id,
        judge_result_id,
        reward_signal_ids,
        float(result.get("deterministic_metrics", {}).get("terminal_quality", 0)),
    )


def _execute_transition(
    *,
    run_id: str,
    rollout_tree_id: str,
    cursor: Cursor,
    action: dict[str, Any],
    branch_member_id: str | None = None,
    fixture_scenario: str = "valid",
    simulate_infra_retry: bool = False,
) -> TransitionEvidence:
    _ensure_running(run_id)
    source_state = _state_logical(cursor.state_id)
    transition_id = make_id("transition")
    with connection() as conn:
        action_intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'apply_action'
              AND operation_input->>'cursor_id' = %s
              AND operation_input->>'expected_state_id' = %s
              AND expected_version = %s
            ORDER BY created_at LIMIT 1
            """,
            (run_id, cursor.cursor_id, cursor.state_id, cursor.version),
        ).fetchone()
    if action_intent:
        action_operation_id = action_intent["operation_id"]
        action_input = action_intent["operation_input"]
        if action_input["action"] != action:
            raise RuntimeError("persisted action intent diverges from the declared action plan")
        destination_state_id = action_input["destination_scientific_state_id"]
    else:
        destination_state_id = make_id("state")
        action_operation_id = make_id("op")
        action_input = {
            "cursor_id": cursor.cursor_id,
            "expected_state_id": cursor.state_id,
            "expected_fencing_token": cursor.fencing_token,
            "destination_scientific_state_id": destination_state_id,
            "action": action,
        }
        _authorize(
            run_id=run_id,
            operation_id=action_operation_id,
            operation_type="apply_action",
            operation_input=action_input,
            expected_version=cursor.version,
        )
    action_result = execution_client.operation(
        "/v1/sessions/actions",
        "apply_action",
        action_input,
        operation_id=action_operation_id,
        idempotency_key=action_operation_id,
        expected_version=cursor.version,
        correlation_id=run_id,
    )
    candidate_state = action_result["candidate_state"]
    semantic_status = (
        "TERMINATED" if action_result["semantic_outcome"] == "TERMINATED" else "ACTIVE"
    )
    with connection() as conn:
        accept_operation_result(conn, operation_id=action_operation_id, result=action_result)
        action_artifact_id, _ = store_json_artifact(
            conn,
            action,
            role="action",
            entity_type="transition",
            entity_id=transition_id,
            visibility="POLICY",
            trust_class="CANDIDATE",
        )
        logical_artifact_id, logical_artifact = store_json_artifact(
            conn,
            candidate_state,
            role="logical-state",
            entity_type="state",
            entity_id=destination_state_id,
            trust_class="CANDIDATE",
        )
        observation = {
            "turn": candidate_state["turn"],
            "feature_count": len(candidate_state["features"]),
            "submitted": candidate_state["submitted"],
            "last_action": action["kind"],
        }
        observation_artifact_id, observation_artifact = store_json_artifact(
            conn,
            observation,
            role="observation",
            entity_type="state",
            entity_id=destination_state_id,
            visibility="POLICY",
            trust_class="CANDIDATE",
        )
        sequence = conn.execute(
            "SELECT COALESCE(max(sequence), -1) + 1 AS next FROM states WHERE rollout_tree_id = %s",
            (rollout_tree_id,),
        ).fetchone()["next"]
        state_contract = {
            "state_id": destination_state_id,
            "rollout_tree_id": rollout_tree_id,
            "parent_transition_id": transition_id,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": TASK_REVISION,
            "logical_state": {
                key: logical_artifact[key]
                for key in (
                    "artifact_id",
                    "digest",
                    "role",
                    "ordinal",
                    "media_type",
                    "viewer_hint",
                    "visibility",
                    "trust_class",
                )
            },
            "observation": {
                key: observation_artifact[key]
                for key in (
                    "artifact_id",
                    "digest",
                    "role",
                    "ordinal",
                    "media_type",
                    "viewer_hint",
                    "visibility",
                    "trust_class",
                )
            },
            "semantic_status": semantic_status,
            "sequence": sequence,
        }
        validate_contract("State", state_contract)
        conn.execute(
            """
            INSERT INTO states(
              state_id, rollout_tree_id, parent_transition_id, environment_version,
              task_revision, logical_state_artifact_id, observation_artifact_id,
              logical_state_digest, semantic_status, sequence, payload
            ) VALUES (%s, %s, %s, 'cad.reconstruction@1.0.0', %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (state_id) DO NOTHING
            """,
            (
                destination_state_id,
                rollout_tree_id,
                transition_id,
                TASK_REVISION,
                logical_artifact_id,
                observation_artifact_id,
                action_result["candidate_state_digest"],
                semantic_status,
                sequence,
                Jsonb(state_contract),
            ),
        )
        conn.execute(
            """
            INSERT INTO transitions(
              transition_id, rollout_tree_id, branch_member_id, source_state_id,
              destination_state_id, runtime_cursor_id, cursor_version, cursor_fencing_token,
              action_artifact_id, operation_id, outcome, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (operation_id) DO NOTHING
            """,
            (
                transition_id,
                rollout_tree_id,
                branch_member_id,
                cursor.state_id,
                destination_state_id,
                cursor.cursor_id,
                cursor.version,
                action_result["fencing_token"],
                action_artifact_id,
                action_operation_id,
                action_result["semantic_outcome"],
                Jsonb(
                    {
                        "transition_id": transition_id,
                        "action": action,
                        "source_state_id": cursor.state_id,
                        "destination_state_id": destination_state_id,
                        "outcome": action_result["semantic_outcome"],
                        "operation_id": action_operation_id,
                    }
                ),
            ),
        )
        emit_event(
            conn,
            event_type="transition.accepted",
            aggregate_type="rollout_tree",
            aggregate_id=rollout_tree_id,
            run_id=run_id,
            correlation_id=action_operation_id,
            payload={
                "transition_id": transition_id,
                "source_state_id": cursor.state_id,
                "destination_state_id": destination_state_id,
                "branch_member_id": branch_member_id,
                "outcome": action_result["semantic_outcome"],
            },
        )

    verification_run_id = make_id("verification")
    verify_operation_id = make_id("op")
    verify_input = {
        "verification_run_id": verification_run_id,
        "subject_id": transition_id,
        "source_state": source_state,
        "candidate_state": candidate_state,
        "action": action,
        "fixture_scenario": fixture_scenario,
        "simulate_infra_retry": simulate_infra_retry,
        "judge_spec_version": 1,
    }
    _authorize(
        run_id=run_id,
        operation_id=verify_operation_id,
        operation_type="verify_transition",
        operation_input=verify_input,
    )
    verify_result = execution_client.operation(
        "/v1/verification-runs/transition",
        "verify_transition",
        verify_input,
        operation_id=verify_operation_id,
        idempotency_key=verify_operation_id,
        correlation_id=run_id,
    )
    proof_bundle_id, judge_result_id, reward_ids, progress = _accept_verification(
        run_id=run_id,
        transition_id=transition_id,
        result=verify_result,
        operation_id=verify_operation_id,
    )
    _fixture_step_delay()
    next_cursor = Cursor(
        cursor_id=cursor.cursor_id,
        state_id=destination_state_id,
        version=action_result["version"],
        fencing_token=action_result["fencing_token"],
    )
    return TransitionEvidence(
        transition_id=transition_id,
        destination_state_id=destination_state_id,
        verification_run_id=verification_run_id,
        proof_bundle_id=proof_bundle_id,
        proof_bundle=verify_result["proof_bundle"],
        judge_result_id=judge_result_id,
        reward_signal_ids=reward_ids,
        progress=progress,
        cursor=next_cursor,
    )


def _evidence_from_science(
    *,
    transition: dict[str, Any],
    cursor: Cursor,
) -> TransitionEvidence | None:
    """Load an already accepted verifier result without reconstructing scientific facts."""
    with connection() as conn:
        verification = conn.execute(
            """
            SELECT vr.verification_run_id, vr.proof_bundle_id, eb.manifest
            FROM verification_runs vr
            JOIN evidence_bundles eb ON eb.proof_bundle_id = vr.proof_bundle_id
            WHERE vr.subject_type = 'TRANSITION' AND vr.subject_id = %s
              AND vr.rejudges_verification_run_id IS NULL
            ORDER BY vr.created_at LIMIT 1
            """,
            (transition["transition_id"],),
        ).fetchone()
        if not verification:
            return None
        judge = conn.execute(
            """
            SELECT jr.judge_result_id
            FROM judge_results jr
            JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
            WHERE ji.verification_run_id = %s
            ORDER BY jr.created_at LIMIT 1
            """,
            (verification["verification_run_id"],),
        ).fetchone()
        rewards = [
            row["reward_signal_id"]
            for row in conn.execute(
                """
                SELECT reward_signal_id FROM reward_signals
                WHERE subject_type = 'TRANSITION' AND subject_id = %s
                ORDER BY created_at
                """,
                (transition["transition_id"],),
            )
        ]
        progress = conn.execute(
            """
            SELECT value FROM metric_observations
            WHERE subject_id = %s AND descriptor = 'deterministic.terminal_quality'
            ORDER BY created_at LIMIT 1
            """,
            (transition["transition_id"],),
        ).fetchone()
    return TransitionEvidence(
        transition_id=transition["transition_id"],
        destination_state_id=transition["destination_state_id"],
        verification_run_id=verification["verification_run_id"],
        proof_bundle_id=verification["proof_bundle_id"],
        proof_bundle=verification["manifest"],
        judge_result_id=judge["judge_result_id"],
        reward_signal_ids=rewards,
        progress=float(progress["value"]) if progress else 0,
        cursor=cursor,
    )


def _ensure_existing_transition_verified(
    *,
    run_id: str,
    transition: dict[str, Any],
    fixture_scenario: str,
    simulate_infra_retry: bool,
) -> TransitionEvidence:
    next_cursor = Cursor(
        cursor_id=transition["runtime_cursor_id"],
        state_id=transition["destination_state_id"],
        version=transition["cursor_version"] + 1,
        fencing_token=transition["cursor_fencing_token"],
    )
    accepted = _evidence_from_science(transition=transition, cursor=next_cursor)
    if accepted:
        return accepted

    source_state = _state_logical(transition["source_state_id"])
    candidate_state = _state_logical(transition["destination_state_id"])
    action = transition["payload"]["action"]
    with connection() as conn:
        intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'verify_transition'
              AND operation_input->>'subject_id' = %s
            ORDER BY created_at LIMIT 1
            """,
            (run_id, transition["transition_id"]),
        ).fetchone()
    if intent:
        operation_id = intent["operation_id"]
        operation_input = intent["operation_input"]
    else:
        operation_id = make_id("op")
        operation_input = {
            "verification_run_id": make_id("verification"),
            "subject_id": transition["transition_id"],
            "source_state": source_state,
            "candidate_state": candidate_state,
            "action": action,
            "fixture_scenario": fixture_scenario,
            "simulate_infra_retry": simulate_infra_retry,
            "judge_spec_version": 1,
        }
        _authorize(
            run_id=run_id,
            operation_id=operation_id,
            operation_type="verify_transition",
            operation_input=operation_input,
        )
    result = execution_client.operation(
        "/v1/verification-runs/transition",
        "verify_transition",
        operation_input,
        operation_id=operation_id,
        idempotency_key=operation_id,
        correlation_id=run_id,
    )
    _accept_verification(
        run_id=run_id,
        transition_id=transition["transition_id"],
        result=result,
        operation_id=operation_id,
    )
    accepted = _evidence_from_science(transition=transition, cursor=next_cursor)
    if not accepted:
        raise RuntimeError("accepted verification could not be reloaded")
    return accepted


def _resume_path(
    *,
    run_id: str,
    rollout_tree_id: str,
    branch_member_id: str | None,
    initial_cursor: Cursor,
    actions: list[dict[str, Any]],
    final_scenario: str = "valid",
    retry_first: bool = False,
) -> tuple[Cursor, list[TransitionEvidence]]:
    with connection() as conn:
        existing = list(
            conn.execute(
                """
                SELECT t.* FROM transitions t
                JOIN states s ON s.state_id = t.destination_state_id
                WHERE t.rollout_tree_id = %s
                  AND t.branch_member_id IS NOT DISTINCT FROM %s
                ORDER BY s.sequence, t.created_at
                """,
                (rollout_tree_id, branch_member_id),
            )
        )
    if len(existing) > len(actions):
        raise RuntimeError("persisted path is longer than its declared action plan")

    cursor = initial_cursor
    evidences: list[TransitionEvidence] = []
    for index, transition in enumerate(existing):
        expected_action = actions[index]
        if transition["payload"]["action"] != expected_action:
            raise RuntimeError("persisted transition diverges from the declared action plan")
        evidence = _ensure_existing_transition_verified(
            run_id=run_id,
            transition=transition,
            fixture_scenario=final_scenario if index == len(actions) - 1 else "valid",
            simulate_infra_retry=retry_first and index == 0,
        )
        cursor = evidence.cursor
        evidences.append(evidence)

    for index, action in enumerate(actions[len(existing) :], start=len(existing)):
        evidence = _execute_transition(
            run_id=run_id,
            rollout_tree_id=rollout_tree_id,
            cursor=cursor,
            action=action,
            branch_member_id=branch_member_id,
            fixture_scenario=final_scenario if index == len(actions) - 1 else "valid",
            simulate_infra_retry=retry_first and index == 0,
        )
        cursor = evidence.cursor
        evidences.append(evidence)
    return cursor, evidences


def _root_cursor(run_id: str, rollout_tree_id: str, lease_owner: str, seed: int) -> Cursor:
    with connection() as conn:
        tree = conn.execute(
            "SELECT root_state_id FROM rollout_trees WHERE rollout_tree_id = %s",
            (rollout_tree_id,),
        ).fetchone()
        if tree and tree["root_state_id"]:
            intent = conn.execute(
                """
                SELECT result FROM operations
                WHERE run_id = %s AND operation_type = 'create_session'
                  AND operation_input->>'scientific_state_id' = %s
                ORDER BY created_at LIMIT 1
                """,
                (run_id, tree["root_state_id"]),
            ).fetchone()
            if not intent or not intent["result"]:
                raise RuntimeError("root state exists without an accepted create-session operation")
            result = intent["result"]
            return Cursor(
                cursor_id=result["cursor_id"],
                state_id=tree["root_state_id"],
                version=result["version"],
                fencing_token=result["fencing_token"],
            )
    cursor, _ = _create_initial_state(
        run_id=run_id,
        rollout_tree_id=rollout_tree_id,
        lease_owner=lease_owner,
        seed=seed,
    )
    return cursor


def _capture_checkpoint(
    *,
    run_id: str,
    rollout_tree_id: str,
    policy_version_id: str,
    cursor: Cursor,
    turns_remaining: int,
) -> tuple[str, str, str]:
    with connection() as conn:
        intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'capture_snapshot'
              AND operation_input->>'cursor_id' = %s
              AND operation_input->>'source_state_id' = %s
              AND expected_version = %s
            ORDER BY created_at LIMIT 1
            """,
            (run_id, cursor.cursor_id, cursor.state_id, cursor.version),
        ).fetchone()
    if intent:
        operation_id = intent["operation_id"]
        snapshot_input = intent["operation_input"]
    else:
        operation_id = make_id("op")
        snapshot_input = {
            "cursor_id": cursor.cursor_id,
            "source_state_id": cursor.state_id,
            "requested_fidelity": "logical_restore",
            "ephemeral": False,
        }
        _authorize(
            run_id=run_id,
            operation_id=operation_id,
            operation_type="capture_snapshot",
            operation_input=snapshot_input,
            expected_version=cursor.version,
        )
    result = execution_client.operation(
        "/v1/snapshots",
        "capture_snapshot",
        snapshot_input,
        operation_id=operation_id,
        idempotency_key=operation_id,
        expected_version=cursor.version,
        correlation_id=run_id,
    )
    snapshot_id = make_id("snapshot")
    checkpoint_id = make_id("checkpoint")
    budget_reservation_id = make_id("budget_reservation")
    with connection() as conn:
        accept_operation_result(conn, operation_id=operation_id, result=result)
        logical_artifact_id, logical_artifact = store_json_artifact(
            conn,
            result["logical_state"],
            role="logical-snapshot",
            entity_type="environment_snapshot",
            entity_id=snapshot_id,
            trust_class="TRUSTED",
        )
        snapshot_contract = {
            "snapshot_id": snapshot_id,
            "source_state_id": cursor.state_id,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": TASK_REVISION,
            "logical_state": {
                key: logical_artifact[key]
                for key in (
                    "artifact_id",
                    "digest",
                    "role",
                    "ordinal",
                    "media_type",
                    "viewer_hint",
                    "visibility",
                    "trust_class",
                )
            },
            "logical_state_digest": result["logical_state_digest"],
            "requested_fidelity": "logical_restore",
            "obtained_fidelity": result["obtained_fidelity"],
            "fidelity_probe_passed": result["fidelity_probe_passed"],
            "rng_state": result["rng_state"],
            "runtime_snapshot_handle": None,
        }
        validate_contract("EnvironmentSnapshot", snapshot_contract)
        conn.execute(
            """
            INSERT INTO environment_snapshots(
              snapshot_id, source_state_id, logical_state_artifact_id, logical_state_digest,
              requested_fidelity, obtained_fidelity, fidelity_probe_passed, rng_state,
              payload, ephemeral
            ) VALUES (%s, %s, %s, %s, 'logical_restore', 'logical_restore', true, %s, %s, false)
            """,
            (
                snapshot_id,
                cursor.state_id,
                logical_artifact_id,
                result["logical_state_digest"],
                result["rng_state"],
                Jsonb(snapshot_contract),
            ),
        )
        policy_context = {
            "behavior_policy_version_id": policy_version_id,
            "conversation": [
                {"role": "system", "content": "Use registered CAD actions only."},
                {
                    "role": "assistant",
                    "content": "Shared prefix reached the fixed decision boundary.",
                },
            ],
            "sampling": {"temperature": 0, "seed": 17},
        }
        policy_context_artifact_id, policy_context_artifact = store_json_artifact(
            conn,
            policy_context,
            role="policy-context",
            entity_type="decision_checkpoint",
            entity_id=checkpoint_id,
            visibility="POLICY",
            trust_class="CANDIDATE",
        )
        checkpoint_contract = {
            "checkpoint_id": checkpoint_id,
            "environment_snapshot_id": snapshot_id,
            "source_state_id": cursor.state_id,
            "rollout_tree_id": rollout_tree_id,
            "policy_context": {
                key: policy_context_artifact[key]
                for key in (
                    "artifact_id",
                    "digest",
                    "role",
                    "ordinal",
                    "media_type",
                    "viewer_hint",
                    "visibility",
                    "trust_class",
                )
            },
            "behavior_policy_version_id": policy_version_id,
            "policy_rng_state": "policy-seed:17",
            "tokenizer_revision": "mock-tokenizer@1",
            "prompt_template_revision": "cad-policy@1",
            "tool_schema_revision": "cad-actions@1",
            "turns_remaining": turns_remaining,
            "branch_strategy": "fixed-boundary@1",
            "budget_reservation_id": budget_reservation_id,
        }
        validate_contract("DecisionCheckpoint", checkpoint_contract)
        conn.execute(
            """
            INSERT INTO decision_checkpoints(
              checkpoint_id, snapshot_id, source_state_id, rollout_tree_id,
              behavior_policy_version_id, policy_context_artifact_id, turns_remaining,
              budget_reservation_id, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                checkpoint_id,
                snapshot_id,
                cursor.state_id,
                rollout_tree_id,
                policy_version_id,
                policy_context_artifact_id,
                turns_remaining,
                budget_reservation_id,
                Jsonb(checkpoint_contract),
            ),
        )
        conn.execute(
            """
            INSERT INTO budget_ledger(
              ledger_entry_id, run_id, operation_id, entry_type, dimensions
            ) VALUES (%s, %s, %s, 'RESERVE', %s)
            """,
            (
                make_id("ledger"),
                run_id,
                operation_id,
                Jsonb(
                    {
                        "transitions": 12,
                        "environment_cpu_seconds": 60,
                        "render_cpu_seconds": 30,
                        "judge_input_tokens": 20000,
                    }
                ),
            ),
        )
        emit_event(
            conn,
            event_type="decision_checkpoint.published",
            aggregate_type="rollout_tree",
            aggregate_id=rollout_tree_id,
            run_id=run_id,
            correlation_id=operation_id,
            payload={
                "snapshot_id": snapshot_id,
                "checkpoint_id": checkpoint_id,
                "obtained_fidelity": "logical_restore",
            },
        )
    return snapshot_id, checkpoint_id, result["operational_snapshot_id"]


def _fork_checkpoint(
    *,
    run_id: str,
    rollout_tree_id: str,
    checkpoint_id: str,
    operational_snapshot_id: str,
    shared_state_id: str,
) -> tuple[str, list[tuple[str, Cursor]]]:
    children = [
        {
            "sibling_index": index,
            "scientific_state_id": shared_state_id,
            "rng_derivation": f"split-stream:{index}",
            "lease_owner": f"run:{run_id}:sibling:{index}",
        }
        for index in range(4)
    ]
    with connection() as conn:
        intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'fork_snapshot'
              AND operation_input->>'operational_snapshot_id' = %s
            ORDER BY created_at LIMIT 1
            """,
            (run_id, operational_snapshot_id),
        ).fetchone()
    if intent:
        operation_id = intent["operation_id"]
        operation_input = intent["operation_input"]
    else:
        operation_id = make_id("op")
        operation_input = {"operational_snapshot_id": operational_snapshot_id, "children": children}
        _authorize(
            run_id=run_id,
            operation_id=operation_id,
            operation_type="fork_snapshot",
            operation_input=operation_input,
        )
    result = execution_client.operation(
        "/v1/snapshots/forks",
        "fork_snapshot",
        operation_input,
        operation_id=operation_id,
        idempotency_key=operation_id,
        correlation_id=run_id,
    )
    branch_group_id = make_id("branch_group")
    members: list[tuple[str, Cursor]] = []
    with connection() as conn:
        accept_operation_result(conn, operation_id=operation_id, result=result)
        conn.execute(
            """
            INSERT INTO branch_groups(
              branch_group_id, checkpoint_id, rollout_tree_id, width, status
            ) VALUES (%s, %s, %s, 4, 'RUNNING')
            """,
            (branch_group_id, checkpoint_id, rollout_tree_id),
        )
        for child in sorted(result["children"], key=lambda item: item["sibling_index"]):
            member_id = make_id("branch_member")
            conn.execute(
                """
                INSERT INTO branch_members(
                  branch_member_id, branch_group_id, sibling_index, runtime_cursor_id,
                  rollout_id, status
                ) VALUES (%s, %s, %s, %s, %s, 'RUNNING')
                """,
                (
                    member_id,
                    branch_group_id,
                    child["sibling_index"],
                    child["cursor_id"],
                    make_id("rollout"),
                ),
            )
            members.append(
                (
                    member_id,
                    Cursor(
                        cursor_id=child["cursor_id"],
                        state_id=child["current_state_id"],
                        version=child["version"],
                        fencing_token=child["fencing_token"],
                    ),
                )
            )
        emit_event(
            conn,
            event_type="branch_group.forked",
            aggregate_type="rollout_tree",
            aggregate_id=rollout_tree_id,
            run_id=run_id,
            correlation_id=operation_id,
            payload={
                "branch_group_id": branch_group_id,
                "checkpoint_id": checkpoint_id,
                "width": 4,
                "child_cursor_ids": [child["cursor_id"] for child in result["children"]],
            },
        )
    return branch_group_id, members


def _accept_group_judgment(
    *,
    run_id: str,
    branch_group_id: str,
    member_ids: list[str],
    result: dict[str, Any],
    operation_id: str,
) -> tuple[str, list[str]]:
    verification_run_id = result["verification_run_id"]
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT verification_run_id FROM verification_runs
            WHERE verification_run_id = %s
            """,
            (verification_run_id,),
        ).fetchone()
        if existing:
            rewards = [
                row["reward_signal_id"]
                for row in conn.execute(
                    """
                    SELECT reward_signal_id FROM reward_signals
                    WHERE subject_type = 'BRANCH_MEMBER'
                      AND subject_id = ANY(%s)
                      AND name = 'sibling_preference'
                    ORDER BY created_at
                    """,
                    (member_ids,),
                )
            ]
            return verification_run_id, rewards
    judge_result = result["judge_result"]
    spec = result["judge_spec"]
    group_manifest = result["group_manifest"]
    group_manifest_digest = canonical_digest(group_manifest)
    group_proof_id = make_id("proof")
    now = utc_now()
    reward_ids: list[str] = []
    with connection() as conn:
        accept_operation_result(conn, operation_id=operation_id, result=result)
        group_artifact_id, group_artifact = store_json_artifact(
            conn,
            group_manifest,
            role="branch-group-proof",
            entity_type="evidence_bundle",
            entity_id=group_proof_id,
            trust_class="TRUSTED",
        )
        conn.execute(
            """
            INSERT INTO verification_runs(
              verification_run_id, run_id, subject_type, subject_id, plan_id,
              plan_version, status, proof_bundle_id, completed_at
            ) VALUES (%s, %s, 'BRANCH_GROUP', %s, 'cad.branch-group@1', 1, %s, %s, %s)
            """,
            (verification_run_id, run_id, branch_group_id, result["status"], group_proof_id, now),
        )
        conn.execute(
            """
            INSERT INTO evidence_bundles(
              proof_bundle_id, verification_run_id, subject_type, subject_id, digest, manifest
            ) VALUES (%s, %s, 'BRANCH_GROUP', %s, %s, %s)
            """,
            (
                group_proof_id,
                verification_run_id,
                branch_group_id,
                group_manifest_digest,
                Jsonb(
                    {**group_manifest, "artifact_id": group_artifact_id, "artifact": group_artifact}
                ),
            ),
        )
        for artifact in result["artifacts"]:
            record_artifact(
                conn,
                artifact,
                entity_type="verification_run",
                entity_id=verification_run_id,
                role=artifact["role"],
            )
        step = result["steps"][0]
        step_run_id = make_id("step_run")
        conn.execute(
            """
            INSERT INTO verifier_step_runs(
              step_run_id, verification_run_id, step_id, step_type, status,
              attempt_count, cache_status, evidence_roles, metrics, failure, cost,
              started_at, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s)
            """,
            (
                step_run_id,
                verification_run_id,
                step["step_id"],
                step["step_type"],
                step["status"],
                step["attempt_count"],
                step["cache_status"],
                Jsonb(step["evidence_roles"]),
                Jsonb(step["metrics"]),
                Jsonb(step["cost"]),
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO judge_specs(judge_spec_id, provider_name, digest, spec)
            VALUES (%s, 'MockJudgeProvider', %s, %s)
            ON CONFLICT (judge_spec_id) DO NOTHING
            """,
            (spec["judge_spec_id"], canonical_digest(spec), Jsonb(spec)),
        )
        invocation_id = make_id("judge_invocation")
        conn.execute(
            """
            INSERT INTO judge_invocations(
              judge_invocation_id, verification_run_id, step_run_id, judge_spec_id,
              proof_bundle_digest, sample_index, status, raw_output_artifact_id,
              accepted_result_id, usage
            ) VALUES (%s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
            """,
            (
                invocation_id,
                verification_run_id,
                step_run_id,
                spec["judge_spec_id"],
                judge_result["proof_bundle_digest"],
                judge_result["outcome"],
                judge_result["raw_output_artifact"]["artifact_id"],
                judge_result["judge_result_id"],
                Jsonb(judge_result["usage"]),
            ),
        )
        conn.execute(
            """
            INSERT INTO judge_results(judge_result_id, judge_invocation_id, outcome, result)
            VALUES (%s, %s, %s, %s)
            """,
            (
                judge_result["judge_result_id"],
                invocation_id,
                judge_result["outcome"],
                Jsonb(judge_result),
            ),
        )

        preference_label = judge_result["preferences"][0] if judge_result["preferences"] else None
        winning_member_id = next(
            (
                binding["branch_member_id"]
                for binding in group_manifest["member_bindings"]
                if binding["label"] == preference_label
            ),
            None,
        )
        admitted_ids = {
            row["branch_member_id"]
            for row in conn.execute(
                """
                SELECT ed.branch_member_id
                FROM eligibility_decisions ed
                JOIN branch_members bm ON bm.branch_member_id = ed.branch_member_id
                WHERE bm.branch_group_id = %s AND ed.status = 'ADMITTED'
                """,
                (branch_group_id,),
            )
        }
        rewardable_members = (
            member_ids if not judge_result["tie"] and preference_label is not None else []
        )
        for member_id in rewardable_members:
            if member_id not in admitted_ids:
                continue
            value = 1.0 if member_id == winning_member_id else 0.0
            metric_id = make_id("metric")
            conn.execute(
                """
                INSERT INTO metric_observations(
                  metric_observation_id, run_id, subject_type, subject_id, descriptor,
                  value, unit, source_step_run_id, judge_result_id, accepted
                ) VALUES (%s, %s, 'BRANCH_MEMBER', %s, 'judge.sibling_preference',
                  %s, 'ratio', %s, %s, true)
                """,
                (
                    metric_id,
                    run_id,
                    member_id,
                    Jsonb(value),
                    step_run_id,
                    judge_result["judge_result_id"],
                ),
            )
            reward_id = make_id("reward")
            reward_contract = {
                "reward_signal_id": reward_id,
                "subject_type": "BRANCH_MEMBER",
                "subject_id": member_id,
                "name": "sibling_preference",
                "value": value,
                "reward_pipeline_id": REWARD_PIPELINE_ID,
                "metric_observation_ids": [metric_id],
            }
            validate_contract("RewardSignal", reward_contract)
            conn.execute(
                """
                INSERT INTO reward_signals(
                  reward_signal_id, run_id, subject_type, subject_id, name, value,
                  reward_pipeline_id, metric_observation_ids
                ) VALUES (%s, %s, 'BRANCH_MEMBER', %s, 'sibling_preference', %s, %s, %s)
                """,
                (reward_id, run_id, member_id, value, REWARD_PIPELINE_ID, Jsonb([metric_id])),
            )
            reward_ids.append(reward_id)
        conn.execute(
            """
            UPDATE branch_groups SET status = 'SUCCEEDED', presentation_order = %s
            WHERE branch_group_id = %s
            """,
            (Jsonb(result["presentation_order"]), branch_group_id),
        )
        emit_event(
            conn,
            event_type="branch_group.judged",
            aggregate_type="branch_group",
            aggregate_id=branch_group_id,
            run_id=run_id,
            correlation_id=operation_id,
            payload={
                "verification_run_id": verification_run_id,
                "judge_result_id": judge_result["judge_result_id"],
                "presentation_order": result["presentation_order"],
                "reward_signal_ids": reward_ids,
            },
        )
    return verification_run_id, reward_ids


def _start_collection(
    *,
    run_id: str,
    attempt_id: str,
    tree_count: int,
) -> tuple[str, str, str, list[str]]:
    with connection() as conn:
        policy = conn.execute(
            "SELECT * FROM policy_versions WHERE run_id = %s AND ordinal = 0",
            (run_id,),
        ).fetchone()
        existing = conn.execute(
            """
            SELECT cb.collection_batch_id, cb.behavior_policy_version_id,
              ti.training_iteration_id
            FROM collection_batches cb
            JOIN training_iterations ti ON ti.collection_batch_id = cb.collection_batch_id
            WHERE cb.run_attempt_id = %s
            ORDER BY cb.created_at LIMIT 1
            """,
            (attempt_id,),
        ).fetchone()
        if existing:
            tree_ids = [
                row["rollout_tree_id"]
                for row in conn.execute(
                    """
                    SELECT rollout_tree_id FROM rollout_trees
                    WHERE collection_batch_id = %s ORDER BY created_at
                    """,
                    (existing["collection_batch_id"],),
                )
            ]
            if len(tree_ids) != tree_count:
                raise RuntimeError("persisted collection has an unexpected rollout-tree count")
            return (
                existing["collection_batch_id"],
                existing["training_iteration_id"],
                existing["behavior_policy_version_id"],
                tree_ids,
            )
        collection_batch_id = make_id("batch")
        batch_contract = {
            "collection_batch_id": collection_batch_id,
            "run_attempt_id": attempt_id,
            "behavior_policy_version_id": policy["policy_version_id"],
            "verification_plan_id": VERIFICATION_PLAN_ID,
            "judge_spec_id": "cad.pointwise.mock@1",
            "reward_pipeline_id": REWARD_PIPELINE_ID,
            "status": "COLLECTING",
        }
        validate_contract("CollectionBatch", batch_contract)
        conn.execute(
            """
            INSERT INTO collection_batches(
              collection_batch_id, run_attempt_id, behavior_policy_version_id,
              verification_plan_id, judge_spec_id, reward_pipeline_id, status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'COLLECTING')
            """,
            (
                collection_batch_id,
                attempt_id,
                policy["policy_version_id"],
                VERIFICATION_PLAN_ID,
                "cad.pointwise.mock@1",
                REWARD_PIPELINE_ID,
            ),
        )
        training_iteration_id = make_id("iteration")
        conn.execute(
            """
            INSERT INTO training_iterations(
              training_iteration_id, run_id, collection_batch_id, status,
              input_policy_version_id, expected_policy_ordinal
            ) VALUES (%s, %s, %s, 'PLANNED', %s, 0)
            """,
            (
                training_iteration_id,
                run_id,
                collection_batch_id,
                policy["policy_version_id"],
            ),
        )
        tree_ids = []
        for _ in range(tree_count):
            tree_id = make_id("tree")
            tree_ids.append(tree_id)
            conn.execute(
                """
                INSERT INTO rollout_trees(
                  rollout_tree_id, collection_batch_id, task_revision, status
                ) VALUES (%s, %s, %s, 'COLLECTING')
                """,
                (tree_id, collection_batch_id, TASK_REVISION),
            )
        emit_event(
            conn,
            event_type="collection_batch.started",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=collection_batch_id,
            payload={
                "collection_batch_id": collection_batch_id,
                "training_iteration_id": training_iteration_id,
                "rollout_tree_ids": tree_ids,
            },
        )
    return (
        collection_batch_id,
        training_iteration_id,
        policy["policy_version_id"],
        tree_ids,
    )


def _record_eligibility(
    *,
    rollout_tree_id: str,
    branch_member_id: str | None,
    status: str,
    reason_code: str,
) -> str:
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT decision_id, status, reason_code FROM eligibility_decisions
            WHERE rollout_tree_id = %s AND branch_member_id IS NOT DISTINCT FROM %s
            ORDER BY created_at LIMIT 1
            """,
            (rollout_tree_id, branch_member_id),
        ).fetchone()
    if existing:
        if existing["status"] != status or existing["reason_code"] != reason_code:
            raise RuntimeError("eligibility replay conflicts with the accepted decision")
        return existing["decision_id"]
    decision_id = make_id("eligibility")
    contract = {
        "decision_id": decision_id,
        "rollout_tree_id": rollout_tree_id,
        "branch_member_id": branch_member_id,
        "status": status,
        "reason_code": reason_code,
        "version": "local-eligibility@1",
    }
    validate_contract("EligibilityDecision", contract)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO eligibility_decisions(
              decision_id, rollout_tree_id, branch_member_id, status, reason_code, version, payload
            ) VALUES (%s, %s, %s, %s, %s, 'local-eligibility@1', %s)
            """,
            (
                decision_id,
                rollout_tree_id,
                branch_member_id,
                status,
                reason_code,
                Jsonb(contract),
            ),
        )
    return decision_id


def _commit_iteration(
    *,
    run_id: str,
    collection_batch_id: str,
    training_iteration_id: str,
    tree_ids: list[str],
    eligibility_ids: list[str],
    group_verification_ids: list[str] | None = None,
) -> None:
    with connection() as conn:
        current = conn.execute(
            "SELECT status FROM training_iterations WHERE training_iteration_id = %s",
            (training_iteration_id,),
        ).fetchone()
        if current and current["status"] == "COMMITTED":
            return
        proof_ids = [
            row["proof_bundle_id"]
            for row in conn.execute(
                """
                SELECT eb.proof_bundle_id
                FROM evidence_bundles eb
                JOIN verification_runs vr ON vr.verification_run_id = eb.verification_run_id
                WHERE vr.run_id = %s
                  AND (
                    vr.subject_type = 'BRANCH_GROUP'
                    OR (
                      vr.subject_type = 'TRANSITION'
                      AND EXISTS (
                        SELECT 1 FROM transitions t
                        WHERE t.transition_id = vr.subject_id
                          AND (
                            t.branch_member_id IS NULL
                            OR EXISTS (
                              SELECT 1 FROM eligibility_decisions ed
                              WHERE ed.branch_member_id = t.branch_member_id
                                AND ed.status = 'ADMITTED'
                            )
                          )
                      )
                    )
                  )
                ORDER BY eb.created_at, eb.proof_bundle_id
                """,
                (run_id,),
            )
        ]
        verification_ids = [
            row["verification_run_id"]
            for row in conn.execute(
                """
                SELECT vr.verification_run_id FROM verification_runs vr
                WHERE vr.run_id = %s
                  AND (
                    vr.subject_type = 'BRANCH_GROUP'
                    OR (
                      vr.subject_type = 'TRANSITION'
                      AND EXISTS (
                        SELECT 1 FROM transitions t
                        WHERE t.transition_id = vr.subject_id
                          AND (
                            t.branch_member_id IS NULL
                            OR EXISTS (
                              SELECT 1 FROM eligibility_decisions ed
                              WHERE ed.branch_member_id = t.branch_member_id
                                AND ed.status = 'ADMITTED'
                            )
                          )
                      )
                    )
                  )
                ORDER BY vr.created_at
                """,
                (run_id,),
            )
        ]
        judge_ids = [
            row["judge_result_id"]
            for row in conn.execute(
                """
                SELECT jr.judge_result_id FROM judge_results jr
                JOIN judge_invocations ji ON ji.judge_invocation_id = jr.judge_invocation_id
                JOIN verification_runs vr ON vr.verification_run_id = ji.verification_run_id
                WHERE vr.run_id = %s
                  AND (
                    vr.subject_type = 'BRANCH_GROUP'
                    OR (
                      vr.subject_type = 'TRANSITION'
                      AND EXISTS (
                        SELECT 1 FROM transitions t
                        WHERE t.transition_id = vr.subject_id
                          AND (
                            t.branch_member_id IS NULL
                            OR EXISTS (
                              SELECT 1 FROM eligibility_decisions ed
                              WHERE ed.branch_member_id = t.branch_member_id
                                AND ed.status = 'ADMITTED'
                            )
                          )
                      )
                    )
                  )
                ORDER BY jr.created_at
                """,
                (run_id,),
            )
        ]
        reward_ids = [
            row["reward_signal_id"]
            for row in conn.execute(
                """
                SELECT rs.reward_signal_id FROM reward_signals rs
                WHERE rs.run_id = %s
                  AND (
                    (
                      rs.subject_type = 'TRANSITION'
                      AND EXISTS (
                        SELECT 1 FROM transitions t
                        WHERE t.transition_id = rs.subject_id
                          AND (
                            t.branch_member_id IS NULL
                            OR EXISTS (
                              SELECT 1 FROM eligibility_decisions ed
                              WHERE ed.branch_member_id = t.branch_member_id
                                AND ed.status = 'ADMITTED'
                            )
                          )
                      )
                    )
                    OR (
                      rs.subject_type = 'BRANCH_MEMBER'
                      AND EXISTS (
                        SELECT 1 FROM eligibility_decisions ed
                        WHERE ed.branch_member_id = rs.subject_id
                          AND ed.status = 'ADMITTED'
                      )
                    )
                  )
                ORDER BY rs.created_at
                """,
                (run_id,),
            )
        ]
        run_manifest = conn.execute(
            "SELECT manifest FROM runs WHERE run_id = %s",
            (run_id,),
        ).fetchone()["manifest"]
        base_manifest = {
            "manifest_id": make_id("iteration_input"),
            "rollout_tree_ids": tree_ids,
            "proof_bundle_ids": proof_ids,
            "eligibility_decision_ids": eligibility_ids,
            "verification_run_ids": verification_ids,
            "judge_result_ids": judge_ids,
            "reward_signal_ids": reward_ids,
            "materializer_version": "branch-jsonl@1",
            "weights": {tree_id: 1.0 for tree_id in tree_ids},
            "data_protocol_digest": run_manifest.get("data_protocol", {}).get(
                "digest",
                canonical_digest(
                    {
                        "task_revision": run_manifest.get("task_revision"),
                        "seed": run_manifest.get("seed"),
                        "source": "generated",
                    }
                ),
            ),
            "gradient_contributors": {
                "rollout_tree_ids": tree_ids,
                "proof_bundle_ids": proof_ids,
                "reward_signal_ids": reward_ids,
            },
        }
        manifest = {**base_manifest, "digest": canonical_digest(base_manifest)}
        validate_contract("IterationInput", manifest)
        artifact_id, _ = store_json_artifact(
            conn,
            manifest,
            role="iteration-input-manifest",
            entity_type="training_iteration",
            entity_id=training_iteration_id,
            trust_class="TRUSTED",
        )
        conn.execute(
            """
            INSERT INTO iteration_inputs(manifest_id, digest, manifest, artifact_id)
            VALUES (%s, %s, %s, %s)
            """,
            (manifest["manifest_id"], manifest["digest"], Jsonb(manifest), artifact_id),
        )
        iteration = conn.execute(
            """
            SELECT * FROM training_iterations
            WHERE training_iteration_id = %s FOR UPDATE
            """,
            (training_iteration_id,),
        ).fetchone()
        current_policy = conn.execute(
            "SELECT * FROM policy_versions WHERE run_id = %s ORDER BY ordinal DESC LIMIT 1 FOR UPDATE",
            (run_id,),
        ).fetchone()
        if current_policy["ordinal"] != iteration["expected_policy_ordinal"]:
            raise ValueError("policy compare-and-swap conflict")
        output_policy_id = make_id("policy")
        policy_manifest = {
            "trainer": "mock-bpo-trainer@1",
            "input_policy_version_id": iteration["input_policy_version_id"],
            "iteration_input_digest": manifest["digest"],
            "estimator": "trainer-owned-local-sibling@1",
            "group_verification_ids": group_verification_ids or [],
        }
        conn.execute(
            """
            INSERT INTO policy_versions(
              policy_version_id, run_id, ordinal, artifact_digest, behavior_manifest
            ) VALUES (%s, %s, 1, %s, %s)
            """,
            (
                output_policy_id,
                run_id,
                canonical_digest(policy_manifest),
                Jsonb(policy_manifest),
            ),
        )
        commit_operation_id = make_id("commit")
        conn.execute(
            """
            UPDATE training_iterations
            SET status = 'COMMITTED', output_policy_version_id = %s,
              iteration_input_id = %s, commit_operation_id = %s, committed_at = now(),
              metrics = %s
            WHERE training_iteration_id = %s AND status = 'PLANNED'
            """,
            (
                output_policy_id,
                manifest["manifest_id"],
                commit_operation_id,
                Jsonb(
                    {
                        "tree_count": len(tree_ids),
                        "proof_count": len(proof_ids),
                        "judge_result_count": len(judge_ids),
                        "reward_signal_count": len(reward_ids),
                    }
                ),
                training_iteration_id,
            ),
        )
        conn.execute(
            """
            UPDATE collection_batches SET status = 'CLOSED', updated_at = now()
            WHERE collection_batch_id = %s
            """,
            (collection_batch_id,),
        )
        emit_event(
            conn,
            event_type="training_iteration.committed",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=commit_operation_id,
            payload={
                "training_iteration_id": training_iteration_id,
                "iteration_input_id": manifest["manifest_id"],
                "input_policy_version_id": iteration["input_policy_version_id"],
                "output_policy_version_id": output_policy_id,
                "rollout_tree_ids": tree_ids,
                "proof_bundle_ids": proof_ids,
                "verification_run_ids": verification_ids,
                "judge_result_ids": judge_ids,
                "reward_signal_ids": reward_ids,
            },
        )


def _finish_tree(rollout_tree_id: str) -> None:
    with connection() as conn:
        transitions = [
            row["transition_id"]
            for row in conn.execute(
                """
                SELECT transition_id FROM transitions
                WHERE rollout_tree_id = %s ORDER BY created_at, transition_id
                """,
                (rollout_tree_id,),
            )
        ]
        conn.execute(
            """
            UPDATE rollout_trees SET status = 'SUCCEEDED', digest = %s
            WHERE rollout_tree_id = %s
            """,
            (canonical_digest({"transitions": transitions}), rollout_tree_id),
        )


def _process_baseline(
    *,
    run_id: str,
    attempt_id: str,
) -> None:
    collection_id, iteration_id, _, tree_ids = _start_collection(
        run_id=run_id,
        attempt_id=attempt_id,
        tree_count=4,
    )
    eligibility_ids: list[str] = []
    actions = [
        {"kind": "create_base"},
        {"kind": "add_boss"},
        {"kind": "add_bore"},
        {"kind": "add_hole_pattern"},
        {"kind": "fillet_edges"},
        {"kind": "submit"},
    ]
    for tree_index, tree_id in enumerate(tree_ids):
        cursor = _root_cursor(
            run_id,
            tree_id,
            f"run:{run_id}:baseline:{tree_index}",
            100 + tree_index,
        )
        _resume_path(
            run_id=run_id,
            rollout_tree_id=tree_id,
            branch_member_id=None,
            initial_cursor=cursor,
            actions=actions,
        )
        _finish_tree(tree_id)
        eligibility_ids.append(
            _record_eligibility(
                rollout_tree_id=tree_id,
                branch_member_id=None,
                status="ADMITTED",
                reason_code="COMPLETE_INDEPENDENT_ROLLOUT",
            )
        )
    _commit_iteration(
        run_id=run_id,
        collection_batch_id=collection_id,
        training_iteration_id=iteration_id,
        tree_ids=tree_ids,
        eligibility_ids=eligibility_ids,
    )


def _process_branch_run(
    *,
    run_id: str,
    attempt_id: str,
) -> None:
    collection_id, iteration_id, policy_version_id, tree_ids = _start_collection(
        run_id=run_id,
        attempt_id=attempt_id,
        tree_count=1,
    )
    tree_id = tree_ids[0]
    cursor = _root_cursor(
        run_id,
        tree_id,
        f"run:{run_id}:backbone",
        17,
    )
    cursor, _ = _resume_path(
        run_id=run_id,
        rollout_tree_id=tree_id,
        branch_member_id=None,
        initial_cursor=cursor,
        actions=[
            {"kind": "create_base"},
            {"kind": "add_boss"},
            {"kind": "add_bore"},
        ],
    )
    with connection() as conn:
        checkpoint = conn.execute(
            """
            SELECT dc.checkpoint_id, o.result->>'operational_snapshot_id' AS operational_snapshot_id
            FROM decision_checkpoints dc
            JOIN operations o ON o.operation_type = 'capture_snapshot'
              AND o.operation_input->>'source_state_id' = dc.source_state_id
            WHERE dc.rollout_tree_id = %s
            ORDER BY dc.created_at LIMIT 1
            """,
            (tree_id,),
        ).fetchone()
    if checkpoint:
        checkpoint_id = checkpoint["checkpoint_id"]
        operational_snapshot_id = checkpoint["operational_snapshot_id"]
    else:
        _, checkpoint_id, operational_snapshot_id = _capture_checkpoint(
            run_id=run_id,
            rollout_tree_id=tree_id,
            policy_version_id=policy_version_id,
            cursor=cursor,
            turns_remaining=3,
        )
    with connection() as conn:
        branch_group = conn.execute(
            """
            SELECT branch_group_id FROM branch_groups
            WHERE checkpoint_id = %s ORDER BY created_at LIMIT 1
            """,
            (checkpoint_id,),
        ).fetchone()
        persisted_members = (
            list(
                conn.execute(
                    """
                    SELECT branch_member_id, runtime_cursor_id FROM branch_members
                    WHERE branch_group_id = %s ORDER BY sibling_index
                    """,
                    (branch_group["branch_group_id"],),
                )
            )
            if branch_group
            else []
        )
    if branch_group:
        branch_group_id = branch_group["branch_group_id"]
        members = [
            (
                member["branch_member_id"],
                Cursor(
                    cursor_id=member["runtime_cursor_id"],
                    state_id=cursor.state_id,
                    version=0,
                    fencing_token=1,
                ),
            )
            for member in persisted_members
        ]
        if len(members) != 4:
            raise RuntimeError("persisted branch group does not contain four members")
    else:
        branch_group_id, members = _fork_checkpoint(
            run_id=run_id,
            rollout_tree_id=tree_id,
            checkpoint_id=checkpoint_id,
            operational_snapshot_id=operational_snapshot_id,
            shared_state_id=cursor.state_id,
        )
    member_plans = [
        (
            [{"kind": "add_hole_pattern"}, {"kind": "fillet_edges"}, {"kind": "submit"}],
            "valid",
            False,
        ),
        (
            [{"kind": "add_hole_pattern"}, {"kind": "add_hole_pattern"}, {"kind": "submit"}],
            "retry",
            True,
        ),
        (
            [{"kind": "oversize_bore"}, {"kind": "add_hole_pattern"}, {"kind": "submit"}],
            "low",
            False,
        ),
        (
            [{"kind": "add_hole_pattern"}, {"kind": "fillet_edges"}, {"kind": "submit"}],
            "abstain",
            False,
        ),
    ]
    admitted_terminal_proofs: list[dict[str, Any]] = []
    admitted_member_ids: list[str] = []
    eligibility_ids: list[str] = []
    for index, ((member_id, member_cursor), (actions, final_scenario, retry_once)) in enumerate(
        zip(members, member_plans, strict=True)
    ):
        _, member_evidence = _resume_path(
            run_id=run_id,
            rollout_tree_id=tree_id,
            branch_member_id=member_id,
            initial_cursor=member_cursor,
            actions=actions,
            final_scenario=final_scenario,
            retry_first=retry_once,
        )
        last_evidence = member_evidence[-1]
        if index == 3:
            status = "EXCLUDED"
            reason = "POINTWISE_JUDGE_ABSTAINED"
            member_status = "EXCLUDED"
            failure_mode = "JUDGE_ABSTENTION"
        else:
            status = "ADMITTED"
            reason = "VALID_NEGATIVE_CANDIDATE_OUTCOME" if index == 2 else "COMPLETE_SIBLING"
            member_status = "SUCCEEDED"
            failure_mode = "VALID_CANDIDATE_FAILURE" if index == 2 else None
            admitted_member_ids.append(member_id)
            admitted_terminal_proofs.append(
                {
                    "branch_member_id": member_id,
                    "sibling_index": index,
                    "proof_bundle": last_evidence.proof_bundle,
                    "progress": last_evidence.progress,
                }
            )
        eligibility_ids.append(
            _record_eligibility(
                rollout_tree_id=tree_id,
                branch_member_id=member_id,
                status=status,
                reason_code=reason,
            )
        )
        with connection() as conn:
            conn.execute(
                """
                UPDATE branch_members SET status = %s, failure_mode = %s
                WHERE branch_member_id = %s
                """,
                (member_status, failure_mode, member_id),
            )

    with connection() as conn:
        accepted_group = conn.execute(
            """
            SELECT verification_run_id FROM verification_runs
            WHERE subject_type = 'BRANCH_GROUP' AND subject_id = %s
            ORDER BY created_at LIMIT 1
            """,
            (branch_group_id,),
        ).fetchone()
    if accepted_group:
        accepted_group_id = accepted_group["verification_run_id"]
    else:
        with connection() as conn:
            group_intent = conn.execute(
                """
                SELECT * FROM operations
                WHERE run_id = %s AND operation_type = 'group_judge'
                  AND operation_input->>'subject_id' = %s
                ORDER BY created_at LIMIT 1
                """,
                (run_id, branch_group_id),
            ).fetchone()
        if group_intent:
            group_operation_id = group_intent["operation_id"]
            group_input = group_intent["operation_input"]
        else:
            group_operation_id = make_id("op")
            group_input = {
                "verification_run_id": make_id("verification"),
                "subject_id": branch_group_id,
                "sibling_proofs": admitted_terminal_proofs,
            }
            _authorize(
                run_id=run_id,
                operation_id=group_operation_id,
                operation_type="group_judge",
                operation_input=group_input,
            )
        group_result = execution_client.operation(
            "/v1/verification-runs/group",
            "group_judge",
            group_input,
            operation_id=group_operation_id,
            idempotency_key=group_operation_id,
            correlation_id=run_id,
        )
        accepted_group_id, _ = _accept_group_judgment(
            run_id=run_id,
            branch_group_id=branch_group_id,
            member_ids=admitted_member_ids,
            result=group_result,
            operation_id=group_operation_id,
        )
    _finish_tree(tree_id)
    _commit_iteration(
        run_id=run_id,
        collection_batch_id=collection_id,
        training_iteration_id=iteration_id,
        tree_ids=tree_ids,
        eligibility_ids=eligibility_ids,
        group_verification_ids=[accepted_group_id],
    )


def _runtime_cursor_ids(run_id: str) -> set[str]:
    with connection() as conn:
        cursor_ids = {
            row["runtime_cursor_id"]
            for row in conn.execute(
                """
                SELECT DISTINCT t.runtime_cursor_id
                FROM transitions t
                JOIN rollout_trees rt ON rt.rollout_tree_id = t.rollout_tree_id
                JOIN collection_batches cb ON cb.collection_batch_id = rt.collection_batch_id
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s
                """,
                (run_id,),
            )
        }
        cursor_ids.update(
            row["runtime_cursor_id"]
            for row in conn.execute(
                """
                SELECT bm.runtime_cursor_id FROM branch_members bm
                JOIN branch_groups bg ON bg.branch_group_id = bm.branch_group_id
                JOIN rollout_trees rt ON rt.rollout_tree_id = bg.rollout_tree_id
                JOIN collection_batches cb ON cb.collection_batch_id = rt.collection_batch_id
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s
                """,
                (run_id,),
            )
        )
        operation_results = list(
            conn.execute(
                """
                SELECT operation_type, result FROM operations
                WHERE run_id = %s AND status = 'ACCEPTED'
                  AND operation_type IN ('create_session', 'fork_snapshot')
                """,
                (run_id,),
            )
        )
    for operation in operation_results:
        result = operation["result"] or {}
        if operation["operation_type"] == "create_session" and result.get("cursor_id"):
            cursor_ids.add(result["cursor_id"])
        if operation["operation_type"] == "fork_snapshot":
            cursor_ids.update(
                child["cursor_id"] for child in result.get("children", []) if child.get("cursor_id")
            )
    return cursor_ids


def _cancel_runtime_cursors(run_id: str) -> list[dict[str, str]]:
    cleanup_warnings: list[dict[str, str]] = []
    cursor_ids = _runtime_cursor_ids(run_id)
    for cursor_id in sorted(cursor_ids):
        operation_id = make_id("op")
        operation_input = {"cursor_id": cursor_id}
        try:
            _authorize(
                run_id=run_id,
                operation_id=operation_id,
                operation_type="cancel_session",
                operation_input=operation_input,
            )
            result = execution_client.operation(
                "/v1/sessions/cancel",
                "cancel_session",
                operation_input,
                operation_id=operation_id,
                idempotency_key=operation_id,
                correlation_id=run_id,
            )
            with connection() as conn:
                accept_operation_result(conn, operation_id=operation_id, result=result)
        except Exception as exc:
            cleanup_warnings.append({"cursor_id": cursor_id, "message": str(exc)})
    return cleanup_warnings


def _release_run_resources(run_id: str) -> None:
    cursor_ids = _runtime_cursor_ids(run_id)
    cleanup_warnings = _cancel_runtime_cursors(run_id)
    with connection() as conn:
        conn.execute(
            """
            UPDATE compute_allocations
            SET desired_state = 'RELEASED', observed_state = %s,
              cleanup_warning = %s, updated_at = now()
            WHERE run_attempt_id IN (SELECT attempt_id FROM run_attempts WHERE run_id = %s)
            """,
            (
                "RELEASE_FAILED" if cleanup_warnings else "RELEASED",
                Jsonb(cleanup_warnings) if cleanup_warnings else None,
                run_id,
            ),
        )
        conn.execute(
            """
            UPDATE run_attempts SET status = 'CANCELED', updated_at = now()
            WHERE run_id = %s AND status NOT IN ('SUCCEEDED', 'FAILED')
            """,
            (run_id,),
        )
        conn.execute(
            """
            UPDATE runs SET status = 'CANCELED', cleanup_status = %s, warning = %s,
              updated_at = now()
            WHERE run_id = %s
            """,
            (
                "WARNING" if cleanup_warnings else "SUCCEEDED",
                Jsonb({"cleanup_warnings": cleanup_warnings}) if cleanup_warnings else None,
                run_id,
            ),
        )
        emit_event(
            conn,
            event_type="run.canceled",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=make_id("cancel"),
            payload={
                "released_cursor_count": len(cursor_ids) - len(cleanup_warnings),
                "cleanup_warnings": cleanup_warnings,
            },
        )


def release_run_resources(run_id: str) -> None:
    """Finish desired-state cancellation for work that has no active worker."""
    _release_run_resources(run_id)


def execute_run_attempt(attempt_id: str) -> None:
    with connection() as conn:
        attempt = conn.execute(
            """
            SELECT ra.*, r.algorithm, r.run_id, r.desired_state
            FROM run_attempts ra JOIN runs r ON r.run_id = ra.run_id
            WHERE ra.attempt_id = %s FOR UPDATE
            """,
            (attempt_id,),
        ).fetchone()
        if not attempt:
            raise ValueError(f"attempt not found: {attempt_id}")
        if attempt["status"] == "SUCCEEDED":
            return
        conn.execute(
            "UPDATE run_attempts SET status = 'RUNNING', updated_at = now() WHERE attempt_id = %s",
            (attempt_id,),
        )
        conn.execute(
            "UPDATE runs SET status = 'RUNNING', updated_at = now() WHERE run_id = %s",
            (attempt["run_id"],),
        )
        emit_event(
            conn,
            event_type="run_attempt.started",
            aggregate_type="run",
            aggregate_id=attempt["run_id"],
            run_id=attempt["run_id"],
            correlation_id=attempt_id,
            payload={"attempt_id": attempt_id, "algorithm": attempt["algorithm"]},
            producer="mock-run-agent",
        )
    try:
        if attempt["algorithm"] == "independent_rollout_baseline":
            _process_baseline(run_id=attempt["run_id"], attempt_id=attempt_id)
        else:
            _process_branch_run(run_id=attempt["run_id"], attempt_id=attempt_id)
        cleanup_warnings = _cancel_runtime_cursors(attempt["run_id"])
        with connection() as conn:
            conn.execute(
                "UPDATE run_attempts SET status = 'SUCCEEDED', updated_at = now() WHERE attempt_id = %s",
                (attempt_id,),
            )
            conn.execute(
                """
                UPDATE compute_allocations
                SET desired_state = 'RELEASED', observed_state = %s,
                  cleanup_warning = %s, updated_at = now()
                WHERE run_attempt_id = %s
                """,
                (
                    "RELEASE_FAILED" if cleanup_warnings else "RELEASED",
                    Jsonb(cleanup_warnings) if cleanup_warnings else None,
                    attempt_id,
                ),
            )
            conn.execute(
                """
                UPDATE runs SET status = 'SUCCEEDED', cleanup_status = %s,
                  warning = %s, updated_at = now() WHERE run_id = %s
                """,
                (
                    "WARNING" if cleanup_warnings else "SUCCEEDED",
                    Jsonb({"cleanup_warnings": cleanup_warnings}) if cleanup_warnings else None,
                    attempt["run_id"],
                ),
            )
            emit_event(
                conn,
                event_type="run.succeeded",
                aggregate_type="run",
                aggregate_id=attempt["run_id"],
                run_id=attempt["run_id"],
                correlation_id=attempt_id,
                payload={"attempt_id": attempt_id},
                producer="mock-run-agent",
            )
    except RunCanceled:
        _release_run_resources(attempt["run_id"])
    except RetryableExecutionError as exc:
        with connection() as conn:
            failure = {
                "code": "EXECUTION_INFRASTRUCTURE_TRANSIENT",
                "category": "INFRASTRUCTURE_TRANSIENT",
                "retryable": True,
                "message": str(exc),
            }
            conn.execute(
                """
                UPDATE run_attempts SET status = 'QUEUED', failure = %s, updated_at = now()
                WHERE attempt_id = %s
                """,
                (Jsonb(failure), attempt_id),
            )
            conn.execute(
                """
                UPDATE runs SET status = 'QUEUED', warning = %s, updated_at = now()
                WHERE run_id = %s AND desired_state = 'RUNNING'
                """,
                (Jsonb(failure), attempt["run_id"]),
            )
            emit_event(
                conn,
                event_type="run.retry_scheduled",
                aggregate_type="run",
                aggregate_id=attempt["run_id"],
                run_id=attempt["run_id"],
                correlation_id=attempt_id,
                payload=failure,
                producer="mock-run-agent",
            )
    except Exception as exc:
        with connection() as conn:
            failure = {
                "code": "LOCAL_WORKFLOW_FAILED",
                "category": "CONTROL_PLANE_BUG",
                "retryable": False,
                "message": str(exc),
            }
            conn.execute(
                """
                UPDATE run_attempts SET status = 'FAILED', failure = %s, updated_at = now()
                WHERE attempt_id = %s
                """,
                (Jsonb(failure), attempt_id),
            )
            conn.execute(
                "UPDATE runs SET status = 'FAILED', warning = %s, updated_at = now() WHERE run_id = %s",
                (Jsonb(failure), attempt["run_id"]),
            )
            emit_event(
                conn,
                event_type="run.failed",
                aggregate_type="run",
                aggregate_id=attempt["run_id"],
                run_id=attempt["run_id"],
                correlation_id=attempt_id,
                payload=failure,
                producer="mock-run-agent",
            )
        raise


def rejudge_transition(
    *,
    run_id: str,
    verification_run_id: str,
    fixture_scenario: str = "valid",
) -> str:
    with connection() as conn:
        original = conn.execute(
            """
            SELECT vr.*, eb.manifest
            FROM verification_runs vr
            JOIN evidence_bundles eb ON eb.proof_bundle_id = vr.proof_bundle_id
            WHERE vr.verification_run_id = %s AND vr.run_id = %s
            """,
            (verification_run_id, run_id),
        ).fetchone()
        if not original:
            raise ValueError("verification run not found")
        progress = conn.execute(
            """
            SELECT value FROM metric_observations
            WHERE subject_id = %s AND descriptor = 'deterministic.terminal_quality'
            LIMIT 1
            """,
            (original["subject_id"],),
        ).fetchone()
    new_verification_id = make_id("verification")
    operation_id = make_id("op")
    operation_input = {
        "verification_run_id": new_verification_id,
        "subject_id": original["subject_id"],
        "proof_bundle": original["manifest"],
        "deterministic_progress": float(progress["value"]) if progress else 0.5,
        "fixture_scenario": fixture_scenario,
        "judge_spec_version": 2,
    }
    _authorize(
        run_id=run_id,
        operation_id=operation_id,
        operation_type="rejudge",
        operation_input=operation_input,
    )
    result = execution_client.operation(
        "/v1/verification-runs/rejudge",
        "rejudge",
        operation_input,
        operation_id=operation_id,
        idempotency_key=operation_id,
        correlation_id=run_id,
    )
    _accept_verification(
        run_id=run_id,
        transition_id=original["subject_id"],
        result=result,
        operation_id=operation_id,
        rejudges_verification_run_id=verification_run_id,
        create_rewards=False,
    )
    return new_verification_id
