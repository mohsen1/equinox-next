from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from equinox_core import (
    canonical_bytes,
    canonical_digest,
    content_digest,
    make_id,
    utc_now,
    validate_contract,
)
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
    lease_owner: str


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


@dataclass(frozen=True)
class RunPlan:
    algorithm: str
    task_revision: str
    seed: int
    branch_width: int
    decision_after_actions: int
    transition_budget: int

    @classmethod
    def from_run(cls, run: dict[str, Any]) -> RunPlan:
        manifest = run["manifest"]
        if canonical_digest(manifest) != run["manifest_digest"]:
            raise RuntimeError("run manifest bytes do not match the accepted digest")
        if manifest["algorithm"]["id"] != run["algorithm"]:
            raise RuntimeError("run algorithm column diverges from the immutable manifest")
        expected_width = 1 if run["algorithm"] == "independent_rollout_baseline" else 4
        if (
            manifest["profile"] != "local-contract-proof"
            or manifest["environment"]["id"] != "cad.reconstruction"
            or manifest["environment"]["version"] != "1.0.0"
            or manifest["task_revision"] != TASK_REVISION
            or manifest["branch"]["mode"] != "static"
            or manifest["branch"]["width"] != expected_width
            or manifest["branch"]["decision_after_actions"] != 3
            or manifest["verification"]["plan_id"] != VERIFICATION_PLAN_ID
            or manifest["verification"]["judge_provider"] != "DeterministicJudgeFixture"
            or manifest["policy_compute"]["provider"] != "LocalFixtureComputeProvider"
        ):
            raise RuntimeError("run manifest requests an unsupported executable configuration")
        required_transitions = 24 if expected_width == 1 else 15
        transition_budget = manifest["budgets"]["transitions"]
        if transition_budget < required_transitions:
            raise RuntimeError("run transition budget cannot execute the declared plan")
        return cls(
            algorithm=run["algorithm"],
            task_revision=manifest["task_revision"],
            seed=manifest["seed"],
            branch_width=expected_width,
            decision_after_actions=manifest["branch"]["decision_after_actions"],
            transition_budget=transition_budget,
        )


def _record_fixture_policy_decision(
    *,
    run_id: str,
    rollout_tree_id: str,
    branch_member_id: str | None,
    behavior_policy_version_id: str,
    cursor: Cursor,
    action: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT pd.*, a.digest AS action_digest, a.object_key, a.media_type, a.size_bytes
            FROM policy_decisions pd
            JOIN artifacts a ON a.artifact_id = pd.action_artifact_id
            WHERE pd.rollout_tree_id = %s
              AND pd.branch_member_id IS NOT DISTINCT FROM %s
              AND pd.source_state_id = %s
            """,
            (rollout_tree_id, branch_member_id, cursor.state_id),
        ).fetchone()
        source = conn.execute(
            """
            SELECT s.logical_state_digest, s.observation_artifact_id, a.digest AS observation_digest
            FROM states s
            JOIN artifacts a ON a.artifact_id = s.observation_artifact_id
            WHERE s.state_id = %s
            """,
            (cursor.state_id,),
        ).fetchone()
    if not source:
        raise RuntimeError("policy decision source state is unavailable")
    action_stored = artifact_store.put_json(
        action,
        role="selected-action",
        visibility="POLICY",
        trust_class="CANDIDATE_CONTROLLED",
        viewer_hint="structured-json",
    )
    action_artifact = {
        **action_stored.ref(ordinal=0),
        "object_key": action_stored.object_key,
        "size_bytes": action_stored.size_bytes,
    }
    decision_content = {
        "behavior_policy_version_id": behavior_policy_version_id,
        "decision_kind": "FIXTURE_SCRIPT",
        "source_state_digest": source["logical_state_digest"],
        "observation_digest": source["observation_digest"],
        "selected_action_digest": action_stored.digest,
        "sampling": {"temperature": 0, "stochastic": False},
        "rng_receipt": {
            "kind": "DETERMINISTIC_NO_DRAW",
            "derivation_digest": canonical_digest(
                {
                    "behavior_policy_version_id": behavior_policy_version_id,
                    "rollout_tree_id": rollout_tree_id,
                    "branch_member_id": branch_member_id,
                    "source_state_digest": source["logical_state_digest"],
                    "cursor_version": cursor.version,
                }
            ),
        },
    }
    decision_digest = canonical_digest(decision_content)
    if existing:
        if (
            existing["behavior_policy_version_id"] != behavior_policy_version_id
            or existing["decision_digest"] != decision_digest
            or existing["action_digest"] != action_stored.digest
        ):
            raise RuntimeError("fixture policy decision replay conflicts with accepted lineage")
        return existing["policy_decision_id"], action_artifact

    decision_id = make_id("policy_decision")
    payload = {
        "policy_decision_id": decision_id,
        "run_id": run_id,
        "rollout_tree_id": rollout_tree_id,
        "branch_member_id": branch_member_id,
        "source_state_id": cursor.state_id,
        **decision_content,
    }
    with connection() as conn:
        action_artifact_id = record_artifact(
            conn,
            action_artifact,
            entity_type="policy_decision",
            entity_id=decision_id,
            role="selected-action",
        )
        conn.execute(
            """
            INSERT INTO policy_decisions(
              policy_decision_id, run_id, rollout_tree_id, branch_member_id,
              behavior_policy_version_id, source_state_id, observation_artifact_id,
              action_artifact_id, decision_kind, decision_digest, rng_receipt, payload
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s,
              'FIXTURE_SCRIPT', %s, %s, %s
            )
            """,
            (
                decision_id,
                run_id,
                rollout_tree_id,
                branch_member_id,
                behavior_policy_version_id,
                cursor.state_id,
                source["observation_artifact_id"],
                action_artifact_id,
                decision_digest,
                Jsonb(decision_content["rng_receipt"]),
                Jsonb(payload),
            ),
        )
    return decision_id, action_artifact


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
    task_revision: str,
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
            "task_revision": task_revision,
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
            trust_class="CANDIDATE_CONTROLLED",
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
            trust_class="CANDIDATE_CONTROLLED",
        )
        state_contract = {
            "state_id": state_id,
            "rollout_tree_id": rollout_tree_id,
            "parent_transition_id": None,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": task_revision,
            "logical_state": {
                "artifact_id": logical_artifact_id,
                "digest": canonical_digest(logical_state),
                "role": "logical-state",
                "ordinal": 0,
                "media_type": "application/json",
                "viewer_hint": "structured-json",
                "visibility": "OPERATOR",
                "trust_class": "CANDIDATE_CONTROLLED",
            },
            "observation": {
                "artifact_id": observation_artifact_id,
                "digest": canonical_digest(observation),
                "role": "observation",
                "ordinal": 0,
                "media_type": "application/json",
                "viewer_hint": "structured-json",
                "visibility": "POLICY",
                "trust_class": "CANDIDATE_CONTROLLED",
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
                task_revision,
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
            lease_owner=lease_owner,
        ),
        state_id,
    )


def _state_logical(state_id: str) -> dict[str, Any]:
    with connection() as conn:
        state = conn.execute(
            """
            SELECT
              s.logical_state_artifact_id,
              s.logical_state_digest,
              s.payload,
              a.digest AS artifact_digest,
              r.artifact_id AS referenced_artifact_id
            FROM states s
            JOIN artifacts a ON a.artifact_id = s.logical_state_artifact_id
            JOIN artifact_refs r
              ON r.artifact_id = s.logical_state_artifact_id
             AND r.entity_type = 'state'
             AND r.entity_id = s.state_id
             AND r.role = 'logical-state'
             AND r.ordinal = 0
            WHERE s.state_id = %s
            """,
            (state_id,),
        ).fetchone()
    if not state:
        raise ValueError(f"state not found: {state_id}")
    payload_ref = state["payload"]["logical_state"]
    if (
        state["logical_state_artifact_id"] != state["referenced_artifact_id"]
        or payload_ref["artifact_id"] != state["logical_state_artifact_id"]
        or payload_ref["digest"] != state["logical_state_digest"]
        or state["artifact_digest"] != state["logical_state_digest"]
    ):
        raise RuntimeError(f"state provenance binding is inconsistent: {state_id}")
    value = json_load_artifact(state["artifact_digest"])
    if canonical_digest(value) != state["logical_state_digest"]:
        raise RuntimeError(f"state artifact content diverges from state digest: {state_id}")
    return value


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
        {
            key: value
            for key, value in proof_bundle.items()
            if key not in {"proof_bundle_id", "digest"}
        }
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
            VALUES (%s, 'DeterministicJudgeFixture', %s, %s)
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
            accepted_sources = list(
                conn.execute(
                    """
                    SELECT metric_observation_id, ordinal
                    FROM reward_metric_links
                    WHERE reward_signal_id = %s
                    ORDER BY ordinal
                    """,
                    (actual_reward,),
                )
            )
            if accepted_sources:
                if [row["metric_observation_id"] for row in accepted_sources] != source_ids:
                    raise RuntimeError("reward replay conflicts with normalized metric lineage")
            else:
                for ordinal, metric_observation_id in enumerate(source_ids):
                    conn.execute(
                        """
                        INSERT INTO reward_metric_links(
                          reward_signal_id, metric_observation_id, ordinal
                        ) VALUES (%s, %s, %s)
                        """,
                        (actual_reward, metric_observation_id, ordinal),
                    )
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
    task_revision: str,
    cursor: Cursor,
    action: dict[str, Any],
    behavior_policy_version_id: str,
    branch_member_id: str | None = None,
    fixture_scenario: str = "valid",
    simulate_infra_retry: bool = False,
) -> TransitionEvidence:
    _ensure_running(run_id)
    source_state = _state_logical(cursor.state_id)
    policy_decision_id, action_artifact = _record_fixture_policy_decision(
        run_id=run_id,
        rollout_tree_id=rollout_tree_id,
        branch_member_id=branch_member_id,
        behavior_policy_version_id=behavior_policy_version_id,
        cursor=cursor,
        action=action,
    )
    with connection() as conn:
        action_intent = conn.execute(
            """
            SELECT o.*, t.transition_id AS accepted_transition_id
            FROM operations o
            LEFT JOIN transitions t ON t.operation_id = o.operation_id
            WHERE run_id = %s AND operation_type = 'apply_action'
              AND operation_input->>'cursor_id' = %s
              AND operation_input->>'expected_state_id' = %s
              AND expected_version = %s
            ORDER BY o.created_at LIMIT 1
            """,
            (run_id, cursor.cursor_id, cursor.state_id, cursor.version),
        ).fetchone()
    if action_intent:
        action_operation_id = action_intent["operation_id"]
        action_input = action_intent["operation_input"]
        if action_input["action"] != action:
            raise RuntimeError("persisted action intent diverges from the declared action plan")
        destination_state_id = action_input["destination_scientific_state_id"]
        transition_id = action_intent["accepted_transition_id"] or make_id("transition")
    else:
        transition_id = make_id("transition")
        destination_state_id = make_id("state")
        action_operation_id = make_id("op")
        action_input = {
            "cursor_id": cursor.cursor_id,
            "lease_owner": cursor.lease_owner,
            "expected_state_id": cursor.state_id,
            "expected_fencing_token": cursor.fencing_token,
            "destination_scientific_state_id": destination_state_id,
            "policy_decision_id": policy_decision_id,
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
        action_artifact_id = record_artifact(
            conn,
            action_artifact,
            role="action",
            entity_type="transition",
            entity_id=transition_id,
        )
        logical_artifact_id, logical_artifact = store_json_artifact(
            conn,
            candidate_state,
            role="logical-state",
            entity_type="state",
            entity_id=destination_state_id,
            trust_class="CANDIDATE_CONTROLLED",
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
            trust_class="CANDIDATE_CONTROLLED",
        )
        sequence = conn.execute(
            """
            UPDATE rollout_trees
            SET next_state_sequence = next_state_sequence + 1
            WHERE rollout_tree_id = %s
            RETURNING next_state_sequence - 1 AS next
            """,
            (rollout_tree_id,),
        ).fetchone()["next"]
        state_contract = {
            "state_id": destination_state_id,
            "rollout_tree_id": rollout_tree_id,
            "parent_transition_id": transition_id,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": task_revision,
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
                task_revision,
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
              action_artifact_id, policy_decision_id, operation_id, outcome, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                policy_decision_id,
                action_operation_id,
                action_result["semantic_outcome"],
                Jsonb(
                    {
                        "transition_id": transition_id,
                        "action": action,
                        "lease_owner": cursor.lease_owner,
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
        lease_owner=cursor.lease_owner,
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
        lease_owner=transition["payload"]["lease_owner"],
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
    task_revision: str,
    branch_member_id: str | None,
    initial_cursor: Cursor,
    actions: list[dict[str, Any]],
    behavior_policy_version_id: str,
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
            task_revision=task_revision,
            cursor=cursor,
            action=action,
            behavior_policy_version_id=behavior_policy_version_id,
            branch_member_id=branch_member_id,
            fixture_scenario=final_scenario if index == len(actions) - 1 else "valid",
            simulate_infra_retry=retry_first and index == 0,
        )
        cursor = evidence.cursor
        evidences.append(evidence)
    return cursor, evidences


def _root_cursor(
    run_id: str,
    rollout_tree_id: str,
    task_revision: str,
    lease_owner: str,
    seed: int,
) -> Cursor:
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
                lease_owner=lease_owner,
            )
    cursor, _ = _create_initial_state(
        run_id=run_id,
        rollout_tree_id=rollout_tree_id,
        task_revision=task_revision,
        lease_owner=lease_owner,
        seed=seed,
    )
    return cursor


def _capture_checkpoint(
    *,
    run_id: str,
    rollout_tree_id: str,
    task_revision: str,
    policy_version_id: str,
    policy_seed: int,
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
            "lease_owner": cursor.lease_owner,
            "source_state_id": cursor.state_id,
            "expected_cursor_version": cursor.version,
            "expected_fencing_token": cursor.fencing_token,
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
            trust_class="PLATFORM_DERIVED",
        )
        snapshot_contract = {
            "snapshot_id": snapshot_id,
            "source_state_id": cursor.state_id,
            "environment_version": "cad.reconstruction@1.0.0",
            "task_revision": task_revision,
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
            "fidelity_probe_receipt": result["fidelity_probe_receipt"],
            "captured_cursor_version": result["captured_cursor_version"],
            "captured_fencing_token": result["captured_fencing_token"],
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
            "sampling": {"temperature": 0, "seed": policy_seed},
        }
        policy_context_artifact_id, policy_context_artifact = store_json_artifact(
            conn,
            policy_context,
            role="policy-context",
            entity_type="decision_checkpoint",
            entity_id=checkpoint_id,
            visibility="POLICY",
            trust_class="CANDIDATE_CONTROLLED",
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
            "tokenizer_revision": "fixture-tokenizer@1",
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
                        lease_owner=f"run:{run_id}:sibling:{child['sibling_index']}",
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
            trust_class="PLATFORM_DERIVED",
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
            VALUES (%s, 'DeterministicJudgeFixture', %s, %s)
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
    task_revision: str,
    tree_count: int,
) -> tuple[str, str, str, list[str]]:
    with connection() as conn:
        policy = conn.execute(
            """
            SELECT * FROM policy_versions
            WHERE run_id = %s
            ORDER BY ordinal DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if not policy:
            raise RuntimeError("collection cannot start without an accepted behavior policy")
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
            "judge_spec_id": "cad.pointwise.fixture@1",
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
                "cad.pointwise.fixture@1",
                REWARD_PIPELINE_ID,
            ),
        )
        training_iteration_id = make_id("iteration")
        conn.execute(
            """
            INSERT INTO training_iterations(
              training_iteration_id, run_id, collection_batch_id, status,
              input_policy_version_id, expected_policy_ordinal
            ) VALUES (%s, %s, %s, 'PLANNED', %s, %s)
            """,
            (
                training_iteration_id,
                run_id,
                collection_batch_id,
                policy["policy_version_id"],
                policy["ordinal"],
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
                (tree_id, collection_batch_id, task_revision),
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
    collection_batch_id: str,
    rollout_tree_id: str,
    branch_member_id: str | None,
    terminal_evidence: TransitionEvidence,
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
        decision_id = existing["decision_id"]
    else:
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
                  decision_id, rollout_tree_id, branch_member_id, status,
                  reason_code, version, payload
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

    with connection() as conn:
        terminal = conn.execute(
            """
            SELECT
              t.transition_id,
              s.logical_state_digest,
              pd.decision_digest,
              eb.digest AS proof_digest
            FROM transitions t
            JOIN states s ON s.state_id = t.destination_state_id
            JOIN policy_decisions pd ON pd.policy_decision_id = t.policy_decision_id
            JOIN evidence_bundles eb ON eb.proof_bundle_id = %s
            WHERE t.transition_id = %s
              AND t.rollout_tree_id = %s
              AND t.branch_member_id IS NOT DISTINCT FROM %s
            """,
            (
                terminal_evidence.proof_bundle_id,
                terminal_evidence.transition_id,
                rollout_tree_id,
                branch_member_id,
            ),
        ).fetchone()
        if not terminal:
            raise RuntimeError("eligibility candidate lacks accepted terminal evidence")
        candidate_content = {
            "terminal_state_digest": terminal["logical_state_digest"],
            "terminal_policy_decision_digest": terminal["decision_digest"],
            "proof_bundle_digest": terminal["proof_digest"],
            "eligibility": {
                "status": status,
                "reason_code": reason_code,
                "version": "local-eligibility@1",
            },
        }
        candidate_digest = canonical_digest(candidate_content)
        episode_row = conn.execute(
            """
            SELECT
              e.*,
              cb.behavior_policy_version_id,
              ra.run_id
            FROM collection_batches cb
            JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
            LEFT JOIN episodes e
              ON e.rollout_tree_id = %s
             AND e.branch_member_id IS NOT DISTINCT FROM %s
            WHERE cb.collection_batch_id = %s
            """,
            (rollout_tree_id, branch_member_id, collection_batch_id),
        ).fetchone()
        step_count = conn.execute(
            """
            SELECT count(*) AS count
            FROM transitions
            WHERE rollout_tree_id = %s
              AND branch_member_id IS NOT DISTINCT FROM %s
            """,
            (rollout_tree_id, branch_member_id),
        ).fetchone()["count"]
        return_value = conn.execute(
            """
            SELECT COALESCE(sum(value), 0) AS value
            FROM reward_signals
            WHERE subject_type = 'TRANSITION' AND subject_id = %s
            """,
            (terminal_evidence.transition_id,),
        ).fetchone()["value"]
        episode_content = {
            "run_id": episode_row["run_id"],
            "collection_batch_id": collection_batch_id,
            "rollout_tree_id": rollout_tree_id,
            "branch_member_id": branch_member_id,
            "behavior_policy_version_id": episode_row["behavior_policy_version_id"],
            "terminal_transition_id": terminal_evidence.transition_id,
            "status": "EXCLUDED" if status == "EXCLUDED" else "TERMINATED",
            "terminal_reason": reason_code,
            "step_count": step_count,
            "return_value": float(return_value),
            "policy_lag": 0,
        }
        episode_digest = canonical_digest(episode_content)
        if episode_row["episode_id"]:
            if (
                episode_row["manifest_digest"] != episode_digest
                or episode_row["manifest"] != episode_content
            ):
                raise RuntimeError("episode replay conflicts with immutable trajectory summary")
            episode_id = episode_row["episode_id"]
        else:
            episode_id = make_id("episode")
            conn.execute(
                """
                INSERT INTO episodes(
                  episode_id, run_id, collection_batch_id, rollout_tree_id,
                  branch_member_id, behavior_policy_version_id, terminal_transition_id,
                  status, terminal_reason, step_count, return_value, policy_lag,
                  manifest_digest, manifest, completed_at
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, 0, %s, %s, now()
                )
                """,
                (
                    episode_id,
                    episode_content["run_id"],
                    collection_batch_id,
                    rollout_tree_id,
                    branch_member_id,
                    episode_content["behavior_policy_version_id"],
                    terminal_evidence.transition_id,
                    episode_content["status"],
                    reason_code,
                    step_count,
                    return_value,
                    episode_digest,
                    Jsonb(episode_content),
                ),
            )
        accepted = conn.execute(
            """
            SELECT
              collection_candidate_id,
              terminal_transition_id,
              eligibility_decision_id,
              proof_bundle_id,
              candidate_digest,
              episode_id
            FROM collection_candidates
            WHERE collection_batch_id = %s
              AND rollout_tree_id = %s
              AND branch_member_id IS NOT DISTINCT FROM %s
            FOR UPDATE
            """,
            (collection_batch_id, rollout_tree_id, branch_member_id),
        ).fetchone()
        expected = {
            "terminal_transition_id": terminal_evidence.transition_id,
            "eligibility_decision_id": decision_id,
            "proof_bundle_id": terminal_evidence.proof_bundle_id,
            "candidate_digest": candidate_digest,
            "episode_id": episode_id,
        }
        if accepted:
            if any(accepted[key] != value for key, value in expected.items()):
                raise RuntimeError("collection candidate replay conflicts with frozen evidence")
        else:
            conn.execute(
                """
                INSERT INTO collection_candidates(
                  collection_candidate_id, collection_batch_id, rollout_tree_id,
                  branch_member_id, terminal_transition_id, eligibility_decision_id,
                  proof_bundle_id, candidate_digest, episode_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    make_id("collection_candidate"),
                    collection_batch_id,
                    rollout_tree_id,
                    branch_member_id,
                    terminal_evidence.transition_id,
                    decision_id,
                    terminal_evidence.proof_bundle_id,
                    candidate_digest,
                    episode_id,
                ),
            )
    return decision_id


def _candidate_rows(
    conn: Any,
    collection_batch_id: str,
) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            """
            SELECT
              cc.*,
              ed.status AS eligibility_status,
              ed.reason_code,
              ed.version AS eligibility_version,
              bm.sibling_index,
              pv.artifact_digest AS behavior_policy_digest
            FROM collection_candidates cc
            JOIN eligibility_decisions ed
              ON ed.decision_id = cc.eligibility_decision_id
            JOIN collection_batches cb
              ON cb.collection_batch_id = cc.collection_batch_id
            JOIN policy_versions pv
              ON pv.policy_version_id = cb.behavior_policy_version_id
            LEFT JOIN branch_members bm
              ON bm.branch_member_id = cc.branch_member_id
            WHERE cc.collection_batch_id = %s
            """,
            (collection_batch_id,),
        )
    )


def _close_collection(
    *,
    collection_batch_id: str,
    tree_ids: list[str],
    eligibility_ids: list[str],
) -> dict[str, Any]:
    tree_order = {tree_id: index for index, tree_id in enumerate(tree_ids)}
    with connection() as conn:
        existing = conn.execute(
            "SELECT * FROM collection_closures WHERE collection_batch_id = %s",
            (collection_batch_id,),
        ).fetchone()
        if existing:
            return existing
        candidates = _candidate_rows(conn, collection_batch_id)
    if {row["rollout_tree_id"] for row in candidates} != set(tree_ids):
        raise RuntimeError("collection candidates do not match the declared rollout trees")
    if {row["eligibility_decision_id"] for row in candidates} != set(eligibility_ids):
        raise RuntimeError("collection closure is missing an eligibility decision")
    admitted = [row for row in candidates if row["eligibility_status"] == "ADMITTED"]
    if not admitted:
        raise RuntimeError("a collection cannot close without admitted candidates")
    admitted.sort(
        key=lambda row: (
            tree_order[row["rollout_tree_id"]],
            row["sibling_index"] if row["sibling_index"] is not None else -1,
            row["candidate_digest"],
        )
    )
    all_decisions = sorted(
        (
            {
                "candidate_digest": row["candidate_digest"],
                "status": row["eligibility_status"],
                "reason_code": row["reason_code"],
                "version": row["eligibility_version"],
            }
            for row in candidates
        ),
        key=lambda item: item["candidate_digest"],
    )
    closure_content = {
        "schema": "equinox.collection-closure.v1",
        "behavior_policy_digest": admitted[0]["behavior_policy_digest"],
        "members": [
            {
                "ordinal": ordinal,
                "candidate_digest": row["candidate_digest"],
                "weight": 1.0,
            }
            for ordinal, row in enumerate(admitted)
        ],
        "candidate_decisions": all_decisions,
    }
    closure_id = make_id("collection_closure")
    closure_digest = canonical_digest(closure_content)
    closure_manifest = {
        "collection_closure_id": closure_id,
        "collection_batch_id": collection_batch_id,
        "closure_digest": closure_digest,
        "content": closure_content,
    }
    stored = artifact_store.put_json(
        closure_manifest,
        role="collection-closure-manifest",
        visibility="OPERATOR",
        trust_class="PLATFORM_DERIVED",
        viewer_hint="structured-json",
    )
    artifact = {
        **stored.ref(ordinal=0),
        "object_key": stored.object_key,
        "size_bytes": stored.size_bytes,
    }
    with connection() as conn:
        batch = conn.execute(
            """
            SELECT status FROM collection_batches
            WHERE collection_batch_id = %s
            FOR UPDATE
            """,
            (collection_batch_id,),
        ).fetchone()
        if not batch:
            raise RuntimeError("collection batch disappeared before closure")
        existing = conn.execute(
            "SELECT * FROM collection_closures WHERE collection_batch_id = %s",
            (collection_batch_id,),
        ).fetchone()
        if existing:
            if existing["closure_digest"] != closure_digest:
                raise RuntimeError("collection closure replay produced different content")
            return existing
        current = _candidate_rows(conn, collection_batch_id)
        if sorted(row["candidate_digest"] for row in current) != sorted(
            row["candidate_digest"] for row in candidates
        ):
            raise RuntimeError("collection membership changed during closure")
        artifact_id = record_artifact(
            conn,
            artifact,
            entity_type="collection_closure",
            entity_id=closure_id,
            role="collection-closure-manifest",
        )
        for ordinal, row in enumerate(admitted):
            membership_content = {
                "candidate_digest": row["candidate_digest"],
                "ordinal": ordinal,
                "weight": 1.0,
            }
            conn.execute(
                """
                INSERT INTO collection_memberships(
                  collection_membership_id, collection_batch_id,
                  collection_candidate_id, ordinal, weight, membership_digest
                ) VALUES (%s, %s, %s, %s, 1.0, %s)
                """,
                (
                    make_id("collection_membership"),
                    collection_batch_id,
                    row["collection_candidate_id"],
                    ordinal,
                    canonical_digest(membership_content),
                ),
            )
        conn.execute(
            """
            INSERT INTO collection_closures(
              collection_closure_id, collection_batch_id, closure_digest,
              member_count, manifest_artifact_id, manifest
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                closure_id,
                collection_batch_id,
                closure_digest,
                len(admitted),
                artifact_id,
                Jsonb(closure_manifest),
            ),
        )
        conn.execute(
            """
            UPDATE collection_batches
            SET status = 'CLOSED', updated_at = now()
            WHERE collection_batch_id = %s
            """,
            (collection_batch_id,),
        )
        return conn.execute(
            "SELECT * FROM collection_closures WHERE collection_closure_id = %s",
            (closure_id,),
        ).fetchone()


def _materialize_iteration_input(
    *,
    training_iteration_id: str,
    closure: dict[str, Any],
    tree_ids: list[str],
    eligibility_ids: list[str],
    group_verification_ids: list[str],
) -> dict[str, Any]:
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT manifest FROM iteration_inputs
            WHERE collection_closure_id = %s
            """,
            (closure["collection_closure_id"],),
        ).fetchone()
        if existing:
            return existing["manifest"]
        members = list(
            conn.execute(
                """
                SELECT
                  cm.ordinal,
                  cm.weight,
                  cc.candidate_digest,
                  cc.terminal_transition_id,
                  cc.branch_member_id,
                  eb.digest AS proof_bundle_digest,
                  ed.status AS eligibility_status,
                  ed.reason_code,
                  ed.version AS eligibility_version,
                  pd.decision_digest AS policy_decision_digest,
                  pv.artifact_digest AS behavior_policy_digest,
                  source.logical_state_digest AS source_state_digest,
                  destination.logical_state_digest AS destination_state_digest,
                  destination.task_revision,
                  action.digest AS action_digest,
                  vr.verification_run_id,
                  jr.judge_result_id
                FROM collection_memberships cm
                JOIN collection_candidates cc
                  ON cc.collection_candidate_id = cm.collection_candidate_id
                JOIN evidence_bundles eb
                  ON eb.proof_bundle_id = cc.proof_bundle_id
                JOIN eligibility_decisions ed
                  ON ed.decision_id = cc.eligibility_decision_id
                JOIN transitions t
                  ON t.transition_id = cc.terminal_transition_id
                JOIN policy_decisions pd
                  ON pd.policy_decision_id = t.policy_decision_id
                JOIN policy_versions pv
                  ON pv.policy_version_id = pd.behavior_policy_version_id
                JOIN states source ON source.state_id = t.source_state_id
                JOIN states destination ON destination.state_id = t.destination_state_id
                JOIN artifacts action ON action.artifact_id = t.action_artifact_id
                JOIN verification_runs vr ON vr.subject_id = t.transition_id
                JOIN judge_invocations ji ON ji.verification_run_id = vr.verification_run_id
                JOIN judge_results jr ON jr.judge_invocation_id = ji.judge_invocation_id
                WHERE cm.collection_batch_id = %s
                ORDER BY cm.ordinal
                """,
                (closure["collection_batch_id"],),
            )
        )
        dataset_rows: list[dict[str, Any]] = []
        reward_ids: list[str] = []
        for member in members:
            rewards = list(
                conn.execute(
                    """
                    SELECT reward_signal_id, name, value, reward_pipeline_id
                    FROM reward_signals
                    WHERE subject_id = %s
                       OR (%s::text IS NOT NULL AND subject_id = %s)
                    ORDER BY name, reward_pipeline_id
                    """,
                    (
                        member["terminal_transition_id"],
                        member["branch_member_id"],
                        member["branch_member_id"],
                    ),
                )
            )
            reward_ids.extend(row["reward_signal_id"] for row in rewards)
            dataset_rows.append(
                {
                    "schema": "equinox.training-sample.v1",
                    "ordinal": member["ordinal"],
                    "weight": member["weight"],
                    "candidate_digest": member["candidate_digest"],
                    "behavior_policy_digest": member["behavior_policy_digest"],
                    "policy_decision_digest": member["policy_decision_digest"],
                    "task_revision": member["task_revision"],
                    "source_state_digest": member["source_state_digest"],
                    "action_digest": member["action_digest"],
                    "destination_state_digest": member["destination_state_digest"],
                    "proof_bundle_digest": member["proof_bundle_digest"],
                    "eligibility": {
                        "status": member["eligibility_status"],
                        "reason_code": member["reason_code"],
                        "version": member["eligibility_version"],
                    },
                    "rewards": [
                        {
                            "name": reward["name"],
                            "value": reward["value"],
                            "reward_pipeline_id": reward["reward_pipeline_id"],
                        }
                        for reward in rewards
                    ],
                }
            )
        proof_ids = [
            row["proof_bundle_id"]
            for row in conn.execute(
                """
                SELECT cc.proof_bundle_id
                FROM collection_memberships cm
                JOIN collection_candidates cc
                  ON cc.collection_candidate_id = cm.collection_candidate_id
                WHERE cm.collection_batch_id = %s
                ORDER BY cm.ordinal
                """,
                (closure["collection_batch_id"],),
            )
        ]
        verification_ids = [member["verification_run_id"] for member in members]
        judge_ids = [member["judge_result_id"] for member in members]
    dataset_bytes = b"".join(canonical_bytes(row) + b"\n" for row in dataset_rows)
    dataset_digest = content_digest(dataset_bytes)
    dataset_stored = artifact_store.put_bytes(
        dataset_bytes,
        role="training-dataset",
        media_type="application/x-ndjson",
        visibility="POLICY",
        trust_class="PLATFORM_DERIVED",
        viewer_hint="json-lines",
    )
    dataset_artifact = {
        **dataset_stored.ref(ordinal=0),
        "object_key": dataset_stored.object_key,
        "size_bytes": dataset_stored.size_bytes,
    }
    manifest_id = make_id("iteration_input")
    content = {
        "collection_closure_digest": closure["closure_digest"],
        "dataset_digest": dataset_digest,
        "dataset_row_count": len(dataset_rows),
        "materializer_version": "branch-jsonl@1",
        "weights": {row["candidate_digest"]: row["weight"] for row in dataset_rows},
    }
    manifest = {
        "manifest_id": manifest_id,
        "digest": canonical_digest(content),
        "collection_closure_id": closure["collection_closure_id"],
        "collection_closure_digest": closure["closure_digest"],
        "dataset_artifact_id": dataset_stored.artifact_id,
        "dataset_digest": dataset_digest,
        "dataset_row_count": len(dataset_rows),
        "rollout_tree_ids": tree_ids,
        "proof_bundle_ids": proof_ids,
        "eligibility_decision_ids": eligibility_ids,
        "verification_run_ids": verification_ids + group_verification_ids,
        "judge_result_ids": judge_ids,
        "reward_signal_ids": reward_ids,
        "materializer_version": "branch-jsonl@1",
        "weights": content["weights"],
    }
    validate_contract("IterationInput", manifest)
    manifest_stored = artifact_store.put_json(
        manifest,
        role="iteration-input-manifest",
        visibility="OPERATOR",
        trust_class="PLATFORM_DERIVED",
        viewer_hint="structured-json",
    )
    manifest_artifact = {
        **manifest_stored.ref(ordinal=0),
        "object_key": manifest_stored.object_key,
        "size_bytes": manifest_stored.size_bytes,
    }
    with connection() as conn:
        existing = conn.execute(
            """
            SELECT manifest FROM iteration_inputs
            WHERE collection_closure_id = %s
            FOR UPDATE
            """,
            (closure["collection_closure_id"],),
        ).fetchone()
        if existing:
            if existing["manifest"]["digest"] != manifest["digest"]:
                raise RuntimeError("iteration materialization replay changed content")
            return existing["manifest"]
        dataset_artifact_id = record_artifact(
            conn,
            dataset_artifact,
            entity_type="training_iteration",
            entity_id=training_iteration_id,
            role="training-dataset",
        )
        manifest_artifact_id = record_artifact(
            conn,
            manifest_artifact,
            entity_type="training_iteration",
            entity_id=training_iteration_id,
            role="iteration-input-manifest",
        )
        conn.execute(
            """
            INSERT INTO iteration_inputs(
              manifest_id, digest, manifest, artifact_id, collection_closure_id,
              dataset_artifact_id, dataset_digest, dataset_row_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                manifest_id,
                manifest["digest"],
                Jsonb(manifest),
                manifest_artifact_id,
                closure["collection_closure_id"],
                dataset_artifact_id,
                dataset_digest,
                len(dataset_rows),
            ),
        )
    return manifest


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
        if current and current["status"] in {"COMMITTED", "SIMULATED_COMMITTED"}:
            return
    closure = _close_collection(
        collection_batch_id=collection_batch_id,
        tree_ids=tree_ids,
        eligibility_ids=eligibility_ids,
    )
    manifest = _materialize_iteration_input(
        training_iteration_id=training_iteration_id,
        closure=closure,
        tree_ids=tree_ids,
        eligibility_ids=eligibility_ids,
        group_verification_ids=group_verification_ids or [],
    )
    with connection() as conn:
        iteration = conn.execute(
            """
            SELECT * FROM training_iterations
            WHERE training_iteration_id = %s FOR UPDATE
            """,
            (training_iteration_id,),
        ).fetchone()
        if iteration["status"] in {"COMMITTED", "SIMULATED_COMMITTED"}:
            return
        current_policy = conn.execute(
            """
            SELECT * FROM policy_versions
            WHERE run_id = %s
            ORDER BY ordinal DESC
            LIMIT 1
            FOR UPDATE
            """,
            (run_id,),
        ).fetchone()
        if current_policy["ordinal"] != iteration["expected_policy_ordinal"]:
            raise ValueError("policy compare-and-swap conflict")
        output_policy_id = make_id("policy")
        next_ordinal = current_policy["ordinal"] + 1
        policy_content = {
            "update_kind": "SIMULATED_POLICY_COMMIT",
            "fixture": "deterministic-lineage-fixture@1",
            "input_policy_artifact_digest": current_policy["artifact_digest"],
            "iteration_input_digest": manifest["digest"],
            "dataset_digest": manifest["dataset_digest"],
            "note": "No optimizer step or model artifact was produced.",
        }
        conn.execute(
            """
            INSERT INTO policy_versions(
              policy_version_id, run_id, ordinal, artifact_digest,
              behavior_manifest, update_kind
            ) VALUES (%s, %s, %s, %s, %s, 'SIMULATED_POLICY_COMMIT')
            """,
            (
                output_policy_id,
                run_id,
                next_ordinal,
                canonical_digest(policy_content),
                Jsonb(policy_content),
            ),
        )
        commit_operation_id = make_id("simulated_commit")
        updated = conn.execute(
            """
            UPDATE training_iterations
            SET status = 'SIMULATED_COMMITTED', output_policy_version_id = %s,
              iteration_input_id = %s, commit_operation_id = %s, committed_at = now(),
              update_kind = 'SIMULATED_POLICY_COMMIT', metrics = %s
            WHERE training_iteration_id = %s AND status = 'PLANNED'
            """,
            (
                output_policy_id,
                manifest["manifest_id"],
                commit_operation_id,
                Jsonb(
                    {
                        "tree_count": len(tree_ids),
                        "dataset_row_count": manifest["dataset_row_count"],
                        "proof_count": len(manifest["proof_bundle_ids"]),
                        "judge_result_count": len(manifest["judge_result_ids"]),
                        "reward_signal_count": len(manifest["reward_signal_ids"]),
                        "optimizer_steps": 0,
                        "model_artifact_count": 0,
                    }
                ),
                training_iteration_id,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("simulated policy commit lost its iteration compare-and-set")
        emit_event(
            conn,
            event_type="training_iteration.simulated_policy_committed",
            aggregate_type="run",
            aggregate_id=run_id,
            run_id=run_id,
            correlation_id=commit_operation_id,
            payload={
                "training_iteration_id": training_iteration_id,
                "update_kind": "SIMULATED_POLICY_COMMIT",
                "iteration_input_id": manifest["manifest_id"],
                "dataset_digest": manifest["dataset_digest"],
                "input_policy_version_id": iteration["input_policy_version_id"],
                "output_policy_version_id": output_policy_id,
            },
        )


def _finish_tree(rollout_tree_id: str) -> None:
    with connection() as conn:
        transitions = [
            {
                "source_state_digest": row["source_state_digest"],
                "policy_decision_digest": row["policy_decision_digest"],
                "action_digest": row["action_digest"],
                "destination_state_digest": row["destination_state_digest"],
                "outcome": row["outcome"],
            }
            for row in conn.execute(
                """
                SELECT
                  source.logical_state_digest AS source_state_digest,
                  pd.decision_digest AS policy_decision_digest,
                  action.digest AS action_digest,
                  destination.logical_state_digest AS destination_state_digest,
                  t.outcome,
                  destination.sequence
                FROM transitions t
                JOIN states source ON source.state_id = t.source_state_id
                JOIN states destination ON destination.state_id = t.destination_state_id
                JOIN policy_decisions pd ON pd.policy_decision_id = t.policy_decision_id
                JOIN artifacts action ON action.artifact_id = t.action_artifact_id
                WHERE t.rollout_tree_id = %s
                ORDER BY destination.sequence, t.created_at
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
    plan: RunPlan,
) -> None:
    collection_id, iteration_id, policy_version_id, tree_ids = _start_collection(
        run_id=run_id,
        attempt_id=attempt_id,
        task_revision=plan.task_revision,
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
            plan.task_revision,
            f"run:{run_id}:baseline:{tree_index}",
            plan.seed + tree_index,
        )
        _, tree_evidence = _resume_path(
            run_id=run_id,
            rollout_tree_id=tree_id,
            task_revision=plan.task_revision,
            branch_member_id=None,
            initial_cursor=cursor,
            actions=actions,
            behavior_policy_version_id=policy_version_id,
        )
        _finish_tree(tree_id)
        eligibility_ids.append(
            _record_eligibility(
                collection_batch_id=collection_id,
                rollout_tree_id=tree_id,
                branch_member_id=None,
                terminal_evidence=tree_evidence[-1],
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
    plan: RunPlan,
) -> None:
    collection_id, iteration_id, policy_version_id, tree_ids = _start_collection(
        run_id=run_id,
        attempt_id=attempt_id,
        task_revision=plan.task_revision,
        tree_count=1,
    )
    tree_id = tree_ids[0]
    cursor = _root_cursor(
        run_id,
        tree_id,
        plan.task_revision,
        f"run:{run_id}:backbone",
        plan.seed,
    )
    cursor, _ = _resume_path(
        run_id=run_id,
        rollout_tree_id=tree_id,
        task_revision=plan.task_revision,
        branch_member_id=None,
        initial_cursor=cursor,
        behavior_policy_version_id=policy_version_id,
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
            task_revision=plan.task_revision,
            policy_version_id=policy_version_id,
            policy_seed=plan.seed,
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
                    SELECT branch_member_id, runtime_cursor_id, sibling_index
                    FROM branch_members
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
                    lease_owner=(f"run:{run_id}:sibling:{member['sibling_index']}"),
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
            task_revision=plan.task_revision,
            branch_member_id=member_id,
            initial_cursor=member_cursor,
            actions=actions,
            behavior_policy_version_id=policy_version_id,
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
                collection_batch_id=collection_id,
                rollout_tree_id=tree_id,
                branch_member_id=member_id,
                terminal_evidence=last_evidence,
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


def execute_run_attempt(
    attempt_id: str,
    *,
    worker_id: str,
    claim_id: str,
    fencing_token: int,
) -> None:
    with connection() as conn:
        attempt = conn.execute(
            """
            SELECT
              ra.*,
              r.algorithm,
              r.run_id,
              r.desired_state,
              r.manifest,
              r.manifest_digest
            FROM run_attempts ra JOIN runs r ON r.run_id = ra.run_id
            WHERE ra.attempt_id = %s
              AND ra.claim_id = %s
              AND ra.lease_owner = %s
              AND ra.fencing_token = %s
              AND ra.lease_expires_at > now()
            FOR UPDATE
            """,
            (attempt_id, claim_id, worker_id, fencing_token),
        ).fetchone()
        if not attempt:
            raise ValueError(f"attempt claim is absent, expired, or fenced: {attempt_id}")
        if attempt["status"] == "SUCCEEDED":
            return
        plan = RunPlan.from_run(attempt)
        claimed = conn.execute(
            """
            UPDATE run_attempts
            SET status = 'RUNNING', updated_at = now()
            WHERE attempt_id = %s
              AND status = 'PROVISIONING'
              AND claim_id = %s
              AND lease_owner = %s
              AND fencing_token = %s
              AND lease_expires_at > now()
            """,
            (attempt_id, claim_id, worker_id, fencing_token),
        )
        if claimed.rowcount != 1:
            raise RuntimeError("attempt claim lost its start compare-and-set")
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
            producer="fixture-run-worker",
        )
    try:
        if attempt["algorithm"] == "independent_rollout_baseline":
            _process_baseline(
                run_id=attempt["run_id"],
                attempt_id=attempt_id,
                plan=plan,
            )
        else:
            _process_branch_run(
                run_id=attempt["run_id"],
                attempt_id=attempt_id,
                plan=plan,
            )
        cleanup_warnings = _cancel_runtime_cursors(attempt["run_id"])
        with connection() as conn:
            accepted = conn.execute(
                """
                UPDATE run_attempts
                SET status = 'SUCCEEDED',
                    claim_id = NULL,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = now()
                WHERE attempt_id = %s
                  AND status = 'RUNNING'
                  AND claim_id = %s
                  AND lease_owner = %s
                  AND fencing_token = %s
                  AND lease_expires_at > now()
                """,
                (attempt_id, claim_id, worker_id, fencing_token),
            )
            if accepted.rowcount != 1:
                raise RuntimeError("attempt result was rejected by the claim fence")
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
                producer="fixture-run-worker",
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
            retry = conn.execute(
                """
                UPDATE run_attempts
                SET status = CASE
                      WHEN retry_count < max_retries THEN 'QUEUED'
                      ELSE 'FAILED'
                    END,
                    retry_count = LEAST(retry_count + 1, max_retries),
                    next_eligible_at = now() + (
                      interval '1 second' * power(2, LEAST(retry_count, 6))
                    ),
                    failure = %s,
                    claim_id = NULL,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = now()
                WHERE attempt_id = %s
                  AND claim_id = %s
                  AND lease_owner = %s
                  AND fencing_token = %s
                RETURNING status, retry_count, max_retries
                """,
                (
                    Jsonb(failure),
                    attempt_id,
                    claim_id,
                    worker_id,
                    fencing_token,
                ),
            ).fetchone()
            if not retry:
                return
            scheduled = retry["status"] == "QUEUED"
            conn.execute(
                """
                UPDATE runs SET status = %s, warning = %s, updated_at = now()
                WHERE run_id = %s AND desired_state = 'RUNNING'
                """,
                (
                    "QUEUED" if scheduled else "FAILED",
                    Jsonb(failure),
                    attempt["run_id"],
                ),
            )
            emit_event(
                conn,
                event_type="run.retry_scheduled" if scheduled else "run.retry_exhausted",
                aggregate_type="run",
                aggregate_id=attempt["run_id"],
                run_id=attempt["run_id"],
                correlation_id=attempt_id,
                payload={
                    **failure,
                    "retry_count": retry["retry_count"],
                    "max_retries": retry["max_retries"],
                },
                producer="fixture-run-worker",
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
                UPDATE run_attempts
                SET status = 'FAILED',
                    failure = %s,
                    claim_id = NULL,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = now()
                WHERE attempt_id = %s
                  AND claim_id = %s
                  AND lease_owner = %s
                  AND fencing_token = %s
                """,
                (Jsonb(failure), attempt_id, claim_id, worker_id, fencing_token),
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
                producer="fixture-run-worker",
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
            SELECT vr.*, eb.manifest, eb.digest
            FROM verification_runs vr
            JOIN evidence_bundles eb ON eb.proof_bundle_id = vr.proof_bundle_id
            WHERE vr.verification_run_id = %s AND vr.run_id = %s
            """,
            (verification_run_id, run_id),
        ).fetchone()
        if not original:
            raise ValueError("verification run not found")
    new_verification_id = make_id("verification")
    operation_id = make_id("op")
    operation_input = {
        "verification_run_id": new_verification_id,
        "proof_bundle_digest": original["digest"],
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
