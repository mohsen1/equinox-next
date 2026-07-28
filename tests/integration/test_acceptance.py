from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import psycopg
import pytest
from equinox_core import canonical_digest
from fastapi import HTTPException
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.execution.app.main import (
    ActivityClaim,
    CreateSessionRequest,
    _accept_activity_result,
    _assert_step_result,
)
from services.orchestrator.app.database import connection
from services.orchestrator.app.science import accept_operation_result
from services.orchestrator.app.workflow import (
    _accept_group_judgment,
    _accept_verification,
    _commit_iteration,
)

BASE = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8080")
EXECUTION = os.getenv("EXECUTION_URL", "http://execution:8081")
SCIENCE_DSN = os.getenv("DATABASE_URL", "")
OPS_DSN = os.getenv("OPERATIONAL_DATABASE_URL", "")
INTERNAL_TOKEN = os.getenv("EQUINOX_INTERNAL_TOKEN", "")


def api(path: str, *, method: str = "GET", json: dict[str, Any] | None = None) -> Any:
    headers = (
        {"Authorization": f"Bearer {INTERNAL_TOKEN}"} if path.startswith("/internal/") else None
    )
    response = httpx.request(
        method,
        f"{BASE}{path}",
        json=json,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.integration
def test_internal_api_requires_service_authentication() -> None:
    response = httpx.post(
        f"{BASE}/internal/agent/claims",
        json={"worker_id": "unauthenticated-worker"},
        timeout=10,
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "INTERNAL_AUTH_REQUIRED"
    assert response.headers["x-request-id"]


@pytest.fixture(scope="module")
def completed_runs() -> dict[str, dict[str, Any]]:
    items = api("/v1/runs")["items"]
    by_algorithm: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["status"] == "SUCCEEDED":
            by_algorithm.setdefault(item["algorithm"], item)
    assert set(by_algorithm) == {"independent_rollout_baseline", "bpo_local_metric"}
    return by_algorithm


@pytest.mark.integration
@pytest.mark.acceptance
def test_seeded_algorithms_commit_exact_inputs(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    baseline = completed_runs["independent_rollout_baseline"]
    branch = completed_runs["bpo_local_metric"]
    assert baseline["rollout_tree_count"] == 4
    assert baseline["verification_run_count"] == 24
    assert branch["rollout_tree_count"] == 1
    assert branch["verification_run_count"] == 16
    for run in completed_runs.values():
        detail = api(f"/v1/runs/{run['run_id']}")
        assert len(detail["policy_versions"]) == 2
        assert detail["policy_versions"][1]["ordinal"] == 1
        iterations = api(f"/v1/runs/{run['run_id']}/iterations")["items"]
        assert len(iterations) == 1
        inspected = api(f"/v1/iterations/{iterations[0]['training_iteration_id']}")
        manifest = inspected["iteration"]["iteration_input_manifest"]
        assert manifest["digest"] == inspected["iteration"]["iteration_input_digest"]
        assert manifest["collection_closure_digest"]
        assert manifest["dataset_digest"]
        assert manifest["dataset_row_count"] > 0
        assert inspected["iteration"]["update_kind"] == "SIMULATED_POLICY_COMMIT"
        assert inspected["iteration"]["metrics"]["optimizer_steps"] == 0
        assert inspected["iteration"]["metrics"]["model_artifact_count"] == 0
        assert manifest["rollout_tree_ids"]
        assert manifest["proof_bundle_ids"]
        assert manifest["verification_run_ids"]
        assert manifest["judge_result_ids"]
        assert manifest["reward_signal_ids"]


@pytest.mark.integration
def test_episodes_sequences_and_complexity_are_bound_to_frozen_collection(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    run_id = completed_runs["bpo_local_metric"]["run_id"]
    with psycopg.connect(SCIENCE_DSN, row_factory=dict_row) as conn:
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM episodes WHERE run_id = %s) AS episodes,
              (
                SELECT count(*)
                FROM collection_candidates cc
                JOIN collection_batches cb USING (collection_batch_id)
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s
              ) AS candidates
            """,
            (run_id, run_id),
        ).fetchone()
        assert counts["episodes"] == counts["candidates"] == 4
        assert (
            conn.execute(
                """
                SELECT count(*) FROM collection_candidates cc
                JOIN collection_batches cb USING (collection_batch_id)
                JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
                WHERE ra.run_id = %s AND cc.episode_id IS NULL
                """,
                (run_id,),
            ).fetchone()["count"]
            == 0
        )
        sequence_gaps = conn.execute(
            """
            SELECT count(*) AS count
            FROM (
              SELECT aggregate_type, aggregate_id
              FROM events
              WHERE run_id = %s
              GROUP BY aggregate_type, aggregate_id
              HAVING min(aggregate_sequence) <> 1
                 OR max(aggregate_sequence) <> count(*)
            ) gaps
            """,
            (run_id,),
        ).fetchone()["count"]
        assert sequence_gaps == 0
        closure = conn.execute(
            """
            SELECT
              closure.collection_closure_id,
              cb.behavior_policy_version_id,
              cs.current_level
            FROM collection_closures closure
            JOIN collection_batches cb USING (collection_batch_id)
            JOIN run_attempts ra ON ra.attempt_id = cb.run_attempt_id
            JOIN complexity_states cs ON cs.run_id = ra.run_id
            WHERE ra.run_id = %s
            """,
            (run_id,),
        ).fetchone()
        members = list(
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
                (closure["collection_closure_id"],),
            )
        )
    task_set = [
        {
            "ordinal": member["ordinal"],
            "task_revision": member["task_revision"],
            "candidate_digest": member["candidate_digest"],
            "episode_id": member["episode_id"],
        }
        for member in members
    ]
    payload = {
        "operation_id": f"complexity_observation_{uuid.uuid4().hex}",
        "level": closure["current_level"],
        "collection_closure_id": closure["collection_closure_id"],
        "behavior_policy_version_id": closure["behavior_policy_version_id"],
        "task_set_digest": canonical_digest(task_set),
        "success_definition": "deterministic.terminal_quality == 1.0",
        "ordered_outcomes": [float(member["terminal_quality"]) == 1.0 for member in members],
    }
    observed = api(
        f"/internal/runs/{run_id}/complexity-observations",
        method="POST",
        json=payload,
    )
    replayed = api(
        f"/internal/runs/{run_id}/complexity-observations",
        method="POST",
        json=payload,
    )
    assert observed == replayed
    assert observed["window_progress"]["attempts"] == len(members)


@pytest.mark.integration
@pytest.mark.acceptance
def test_branch_shape_retry_negative_and_abstention(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    run = completed_runs["bpo_local_metric"]
    iteration = api(f"/v1/runs/{run['run_id']}/iterations")["items"][0]
    tree = api(f"/v1/iterations/{iteration['training_iteration_id']}/rollout-trees")["items"][0]
    graph = api(f"/v1/rollout-trees/{tree['rollout_tree_id']}/graph")
    assert (len(graph["nodes"]), len(graph["edges"])) == (16, 15)
    assert all(
        edge[field]
        for edge in graph["edges"]
        for field in (
            "operation_id",
            "action_artifact_id",
            "runtime_cursor_id",
            "created_at",
        )
    )
    assert all(isinstance(edge["cursor_version"], int) for edge in graph["edges"])
    assert len(graph["decision_checkpoints"]) == 1
    assert len(graph["environment_snapshots"]) == 1
    assert len({member["runtime_cursor_id"] for member in graph["branch_members"]}) == 4
    assert any(
        member["failure_mode"] == "VALID_CANDIDATE_FAILURE" for member in graph["branch_members"]
    )
    assert any(member["failure_mode"] == "JUDGE_ABSTENTION" for member in graph["branch_members"])
    assert {member["eligibility_status"] for member in graph["branch_members"]} == {
        "ADMITTED",
        "EXCLUDED",
    }
    assert any(member["retry_count"] > 0 for member in graph["branch_members"])
    assert run["retry_count"] >= 2
    assert run["abstention_count"] == 1

    abstained_member = next(
        member for member in graph["branch_members"] if member["failure_mode"] == "JUDGE_ABSTENTION"
    )
    comparison = api(f"/v1/branch-groups/{graph['branch_groups'][0]['branch_group_id']}/comparison")
    item = next(
        member
        for member in comparison["members"]
        if member["branch_member_id"] == abstained_member["branch_member_id"]
    )
    terminal_rewards = {reward["name"] for reward in item["reward_signals"]}
    assert "reference_alignment" not in terminal_rewards
    assert "progress_from_source_state" not in terminal_rewards
    assert "sibling_preference" not in terminal_rewards
    assert comparison["model_assessment"]["tie"] is False
    sibling_rewards = [
        reward
        for member in comparison["members"]
        for reward in member["reward_signals"]
        if reward["name"] == "sibling_preference"
    ]
    assert len(sibling_rewards) == 3
    assert sorted(reward["value"] for reward in sibling_rewards) == [0, 0, 1]
    admitted_members = {
        member["branch_member_id"]: member["sibling_index"]
        for member in comparison["members"]
        if member["branch_member_id"] != abstained_member["branch_member_id"]
    }
    bindings = {
        binding["branch_member_id"]: binding["sibling_index"]
        for binding in comparison["member_bindings"]
    }
    assert bindings == admitted_members
    winning_label = comparison["model_assessment"]["preferences"][0]
    winning_member_id = next(
        binding["branch_member_id"]
        for binding in comparison["member_bindings"]
        if binding["label"] == winning_label
    )
    rewarded_member_id = next(
        member["branch_member_id"]
        for member in comparison["members"]
        if any(
            reward["name"] == "sibling_preference" and reward["value"] == 1
            for reward in member["reward_signals"]
        )
    )
    assert rewarded_member_id == winning_member_id
    manifest = api(f"/v1/iterations/{iteration['training_iteration_id']}")["iteration"][
        "iteration_input_manifest"
    ]
    excluded_verifications = {
        edge["verification_run_id"]
        for edge in graph["edges"]
        if edge["branch_member_id"] == abstained_member["branch_member_id"]
    }
    excluded_proofs = {
        api(f"/v1/verification-runs/{verification_id}")["proof_bundle"]["proof_bundle_id"]
        for verification_id in excluded_verifications
    }
    excluded_rewards = {reward["reward_signal_id"] for reward in item["reward_signals"]}
    assert excluded_verifications.isdisjoint(manifest["verification_run_ids"])
    assert excluded_proofs.isdisjoint(manifest["proof_bundle_ids"])
    assert excluded_rewards.isdisjoint(manifest["reward_signal_ids"])


@pytest.mark.integration
@pytest.mark.acceptance
def test_every_transition_has_canonical_proof(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    for run in completed_runs.values():
        iteration = api(f"/v1/runs/{run['run_id']}/iterations")["items"][0]
        trees = api(f"/v1/iterations/{iteration['training_iteration_id']}/rollout-trees")["items"]
        for tree in trees:
            graph = api(f"/v1/rollout-trees/{tree['rollout_tree_id']}/graph")
            for edge in graph["edges"]:
                verification = api(f"/v1/verification-runs/{edge['verification_run_id']}")
                proof = verification["proof_bundle"]["manifest"]
                assert proof["source_render"]["media_type"] == "image/svg+xml"
                assert proof["candidate_render"]["media_type"] == "image/svg+xml"
                assert proof["geometry_report"]["media_type"] == "application/json"
                assert proof["digest"].startswith("sha256:")


@pytest.mark.integration
@pytest.mark.acceptance
def test_rejudge_reuses_proof_and_emits_no_new_reward(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    run = completed_runs["bpo_local_metric"]
    iteration = api(f"/v1/runs/{run['run_id']}/iterations")["items"][0]
    tree = api(f"/v1/iterations/{iteration['training_iteration_id']}/rollout-trees")["items"][0]
    original_id = api(f"/v1/rollout-trees/{tree['rollout_tree_id']}/graph")["edges"][0][
        "verification_run_id"
    ]
    before = api(f"/v1/runs/{run['run_id']}")
    original = api(f"/v1/verification-runs/{original_id}")
    result = api(
        f"/v1/runs/{run['run_id']}/rejudge",
        method="POST",
        json={"verification_run_id": original_id, "fixture_scenario": "integrity"},
    )
    rejudged = api(f"/v1/verification-runs/{result['verification_run_id']}")
    after = api(f"/v1/runs/{run['run_id']}")
    assert rejudged["verification_run"]["rejudges_verification_run_id"] == original_id
    assert (
        rejudged["proof_bundle"]["proof_bundle_id"] == original["proof_bundle"]["proof_bundle_id"]
    )
    assert [step["step_id"] for step in rejudged["steps"]] == ["pointwise-judge"]
    assert rejudged["judge"]["result"]["outcome"] == "INTEGRITY_VIOLATION"
    assert len(after["reward_signals"]) == len(before["reward_signals"])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("scenario", "outcome", "attempt_count"),
    [
        ("valid", "SUCCEEDED", 1),
        ("low", "SUCCEEDED", 1),
        ("tie", "SUCCEEDED", 1),
        ("abstain", "ABSTAINED", 1),
        ("malformed", "INVALID_RESULT", 1),
        ("retry", "SUCCEEDED", 2),
        ("disagreement", "DISAGREEMENT", 1),
        ("integrity", "INTEGRITY_VIOLATION", 1),
    ],
)
def test_rejudge_fixture_matrix_never_creates_training_reward(
    completed_runs: dict[str, dict[str, Any]],
    scenario: str,
    outcome: str,
    attempt_count: int,
) -> None:
    run = completed_runs["bpo_local_metric"]
    iteration = api(f"/v1/runs/{run['run_id']}/iterations")["items"][0]
    tree = api(f"/v1/iterations/{iteration['training_iteration_id']}/rollout-trees")["items"][0]
    original_id = api(f"/v1/rollout-trees/{tree['rollout_tree_id']}/graph")["edges"][1][
        "verification_run_id"
    ]
    before_count = len(api(f"/v1/runs/{run['run_id']}")["reward_signals"])
    result = api(
        f"/v1/runs/{run['run_id']}/rejudge",
        method="POST",
        json={"verification_run_id": original_id, "fixture_scenario": scenario},
    )
    detail = api(f"/v1/verification-runs/{result['verification_run_id']}")
    assert detail["judge"]["result"]["outcome"] == outcome
    assert detail["steps"][0]["attempt_count"] == attempt_count
    assert len(api(f"/v1/runs/{run['run_id']}")["reward_signals"]) == before_count


@pytest.mark.integration
def test_artifact_and_openapi_boundaries(completed_runs: dict[str, dict[str, Any]]) -> None:
    schema = api("/openapi.json")
    assert "/v1/runs" in schema["paths"]
    run = completed_runs["bpo_local_metric"]
    iteration = api(f"/v1/runs/{run['run_id']}/iterations")["items"][0]
    tree = api(f"/v1/iterations/{iteration['training_iteration_id']}/rollout-trees")["items"][0]
    verification_id = api(f"/v1/rollout-trees/{tree['rollout_tree_id']}/graph")["edges"][0][
        "verification_run_id"
    ]
    proof = api(f"/v1/verification-runs/{verification_id}")["proof_bundle"]["manifest"]
    response = httpx.get(
        f"{BASE}/v1/artifacts/{proof['candidate_render']['artifact_id']}", timeout=10
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["etag"].strip('"') == proof["candidate_render"]["digest"]
    assert response.headers["content-security-policy"].startswith("sandbox;")
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.integration
def test_duplicate_operation_is_single_job_and_conflict_is_rejected() -> None:
    suffix = uuid.uuid4().hex
    operation_input = {
        "task_revision": "mounting-plate@sha256:fixture-v1",
        "scientific_state_id": f"state_duplicate_{suffix}",
        "lease_owner": "test:duplicate",
        "rng_state": "seed:911",
    }
    operation_id = f"op_duplicate_{suffix}"
    payload = {
        "operation_id": operation_id,
        "idempotency_key": operation_id,
        "request_digest": canonical_digest(
            {"operation_type": "create_session", "input": operation_input}
        ),
        "expected_version": 0,
        "correlation_id": f"run_test_{suffix}",
        **operation_input,
    }
    first = httpx.post(f"{EXECUTION}/v1/sessions", json=payload, timeout=10)
    second = httpx.post(f"{EXECUTION}/v1/sessions", json=payload, timeout=10)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    conflict = httpx.post(
        f"{EXECUTION}/v1/sessions",
        json={**payload, "request_digest": "sha256:" + "0" * 64},
        timeout=10,
    )
    assert conflict.status_code in (409, 422)
    identity_conflict = httpx.post(
        f"{EXECUTION}/v1/sessions",
        json={**payload, "operation_id": f"op_other_{suffix}"},
        timeout=10,
    )
    assert identity_conflict.status_code == 409
    assert identity_conflict.json()["detail"]["code"] == "IDEMPOTENCY_OPERATION_CONFLICT"
    with psycopg.connect(OPS_DSN) as conn:
        operation_count = conn.execute(
            "SELECT count(*) FROM operations WHERE operation_id = %s", (operation_id,)
        ).fetchone()[0]
        job_count = conn.execute(
            "SELECT count(*) FROM jobs WHERE operation_id = %s", (operation_id,)
        ).fetchone()[0]
    assert (operation_count, job_count) == (1, 1)

    session = first.json()
    action_input = {
        "cursor_id": session["cursor_id"],
        "lease_owner": session["lease_owner"],
        "expected_state_id": operation_input["scientific_state_id"],
        "expected_fencing_token": session["fencing_token"],
        "destination_scientific_state_id": f"state_destination_{suffix}",
        "policy_decision_id": f"policy_decision_{suffix}",
        "action": {"kind": "create_base"},
    }
    action_id = f"op_action_duplicate_{suffix}"
    action_payload = {
        "operation_id": action_id,
        "idempotency_key": action_id,
        "request_digest": canonical_digest(
            {"operation_type": "apply_action", "input": action_input}
        ),
        "expected_version": session["version"],
        "correlation_id": f"run_test_{suffix}",
        **action_input,
    }
    action_first = httpx.post(f"{EXECUTION}/v1/sessions/actions", json=action_payload, timeout=10)
    action_second = httpx.post(f"{EXECUTION}/v1/sessions/actions", json=action_payload, timeout=10)
    assert action_first.status_code == action_second.status_code == 200
    assert action_first.json() == action_second.json()
    with psycopg.connect(OPS_DSN) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM jobs WHERE operation_id = %s", (action_id,)
            ).fetchone()[0]
            == 1
        )
    cancel_input = {"cursor_id": session["cursor_id"]}
    cancel_id = f"op_cancel_duplicate_{suffix}"
    cancel = httpx.post(
        f"{EXECUTION}/v1/sessions/cancel",
        json={
            "operation_id": cancel_id,
            "idempotency_key": cancel_id,
            "request_digest": canonical_digest(
                {"operation_type": "cancel_session", "input": cancel_input}
            ),
            "expected_version": action_first.json()["version"],
            "correlation_id": f"run_test_{suffix}",
            **cancel_input,
        },
        timeout=10,
    )
    assert cancel.status_code == 200


@pytest.mark.integration
def test_concurrent_duplicate_delivery_accepts_one_activity_result() -> None:
    suffix = uuid.uuid4().hex
    operation_input = {
        "task_revision": "mounting-plate@sha256:fixture-v1",
        "scientific_state_id": f"state_concurrent_{suffix}",
        "lease_owner": "test:concurrent",
        "rng_state": "seed:1229",
    }
    operation_id = f"op_concurrent_{suffix}"
    payload = {
        "operation_id": operation_id,
        "idempotency_key": operation_id,
        "request_digest": canonical_digest(
            {"operation_type": "create_session", "input": operation_input}
        ),
        "expected_version": 0,
        "correlation_id": f"run_test_{suffix}",
        **operation_input,
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda _: httpx.post(f"{EXECUTION}/v1/sessions", json=payload, timeout=10),
                range(2),
            )
        )

    assert 200 in {response.status_code for response in responses}
    assert {response.status_code for response in responses} <= {200, 409}
    with psycopg.connect(OPS_DSN) as conn:
        job_id = conn.execute(
            "SELECT job_id FROM jobs WHERE operation_id = %s", (operation_id,)
        ).fetchone()[0]
        assert (
            conn.execute(
                "SELECT count(*) FROM attempts WHERE job_id = %s AND status = 'SUCCEEDED'",
                (job_id,),
            ).fetchone()[0]
            == 1
        )
    session = next(response.json() for response in responses if response.status_code == 200)
    cancel_input = {"cursor_id": session["cursor_id"]}
    cancel_id = f"op_cancel_concurrent_{suffix}"
    canceled = httpx.post(
        f"{EXECUTION}/v1/sessions/cancel",
        json={
            "operation_id": cancel_id,
            "idempotency_key": cancel_id,
            "request_digest": canonical_digest(
                {"operation_type": "cancel_session", "input": cancel_input}
            ),
            "expected_version": session["version"],
            "correlation_id": f"run_test_{suffix}",
            **cancel_input,
        },
        timeout=10,
    )
    assert canceled.status_code == 200


@pytest.mark.integration
def test_expired_activity_lease_and_divergent_step_replay_are_rejected() -> None:
    suffix = uuid.uuid4().hex
    operation_id = f"op_stale_{suffix}"
    job_id = f"job_stale_{suffix}"
    attempt_id = f"attempt_stale_{suffix}"
    lease_owner = f"claim_stale_{suffix}"
    operation_input = {
        "task_revision": "mounting-plate@sha256:fixture-v1",
        "scientific_state_id": f"state_stale_{suffix}",
        "lease_owner": "test:stale",
        "rng_state": "seed:1231",
    }
    request = CreateSessionRequest(
        operation_id=operation_id,
        idempotency_key=operation_id,
        request_digest=canonical_digest(
            {"operation_type": "create_session", "input": operation_input}
        ),
        expected_version=0,
        correlation_id=f"run_test_{suffix}",
        **operation_input,
    )
    claim = ActivityClaim(
        job_id=job_id,
        attempt_id=attempt_id,
        attempt_number=1,
        lease_owner=lease_owner,
        fencing_token=1,
        max_attempts=3,
    )
    with psycopg.connect(OPS_DSN, row_factory=dict_row) as conn:
        conn.execute(
            """
            INSERT INTO operations(
              operation_id, idempotency_key, request_digest,
              operation_type, status, request
            ) VALUES (%s, %s, %s, 'create_session', 'RUNNING', %s)
            """,
            (operation_id, operation_id, request.request_digest, Jsonb(operation_input)),
        )
        conn.execute(
            """
            INSERT INTO jobs(
              job_id, operation_id, profile, status, max_attempts, attempt_count,
              lease_owner, lease_expires_at, fencing_token
            ) VALUES (%s, %s, 'cad.session@1', 'RUNNING', 3, 1, %s, %s, 1)
            """,
            (
                job_id,
                operation_id,
                lease_owner,
                datetime.now(UTC) - timedelta(seconds=1),
            ),
        )
        conn.execute(
            """
            INSERT INTO attempts(
              attempt_id, job_id, attempt_number, fencing_token,
              status, lease_owner, cost, usage
            ) VALUES (%s, %s, 1, 1, 'RUNNING', %s, %s, %s)
            """,
            (
                attempt_id,
                job_id,
                lease_owner,
                Jsonb({"kind": "PENDING_MEASUREMENT"}),
                Jsonb({}),
            ),
        )
        with pytest.raises(HTTPException) as stale:
            _accept_activity_result(
                conn,
                request=request,
                claim=claim,
                result={"cursor_id": "must-not-be-accepted"},
                elapsed_seconds=0.1,
            )
        assert stale.value.detail["code"] == "STALE_ACTIVITY_RESULT"
        conn.rollback()

        accepted = conn.execute(
            """
            SELECT operation_id, verification_run_id, result
            FROM accepted_step_results
            ORDER BY created_at
            LIMIT 1
            """
        ).fetchone()
        divergent = {**accepted["result"], "status": "DIVERGENT_REPLAY"}
        with pytest.raises(RuntimeError, match="immutable result"):
            _assert_step_result(
                conn,
                operation_id=accepted["operation_id"],
                verification_run_id=accepted["verification_run_id"],
                step=divergent,
            )


@pytest.mark.integration
def test_scientific_acceptance_and_commit_replays_are_idempotent(
    completed_runs: dict[str, dict[str, Any]],
) -> None:
    run_id = completed_runs["bpo_local_metric"]["run_id"]
    with connection() as conn:
        iteration = conn.execute(
            "SELECT * FROM training_iterations WHERE run_id = %s", (run_id,)
        ).fetchone()
        tree_ids = [
            row["rollout_tree_id"]
            for row in conn.execute(
                """
                SELECT rt.rollout_tree_id FROM rollout_trees rt
                WHERE rt.collection_batch_id = %s ORDER BY rt.created_at
                """,
                (iteration["collection_batch_id"],),
            )
        ]
        eligibility_ids = [
            row["decision_id"]
            for row in conn.execute(
                "SELECT decision_id FROM eligibility_decisions WHERE rollout_tree_id = ANY(%s)",
                (tree_ids,),
            )
        ]
        verification_intent = conn.execute(
            """
            SELECT * FROM operations
            WHERE run_id = %s AND operation_type = 'verify_transition'
              AND status = 'ACCEPTED'
            ORDER BY created_at LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        group_intent = conn.execute(
            """
            SELECT * FROM operations WHERE run_id = %s
              AND operation_type = 'group_judge' AND status = 'ACCEPTED'
            ORDER BY created_at LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        member_ids = [
            row["branch_member_id"]
            for row in conn.execute(
                """
                SELECT branch_member_id FROM eligibility_decisions
                WHERE rollout_tree_id = ANY(%s) AND status = 'ADMITTED'
                  AND branch_member_id IS NOT NULL ORDER BY created_at
                """,
                (tree_ids,),
            )
        ]
        before = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM policy_versions WHERE run_id = %s) AS policies,
              (SELECT count(*) FROM judge_results jr JOIN judge_invocations ji
                USING (judge_invocation_id) JOIN verification_runs vr
                USING (verification_run_id) WHERE vr.run_id = %s) AS judges,
              (SELECT count(*) FROM reward_signals WHERE run_id = %s) AS rewards
            """,
            (run_id, run_id, run_id),
        ).fetchone()

    _accept_verification(
        run_id=run_id,
        transition_id=verification_intent["operation_input"]["subject_id"],
        result=verification_intent["result"],
        operation_id=verification_intent["operation_id"],
    )
    _accept_group_judgment(
        run_id=run_id,
        branch_group_id=group_intent["operation_input"]["subject_id"],
        member_ids=member_ids,
        result=group_intent["result"],
        operation_id=group_intent["operation_id"],
    )
    _commit_iteration(
        run_id=run_id,
        collection_batch_id=iteration["collection_batch_id"],
        training_iteration_id=iteration["training_iteration_id"],
        tree_ids=tree_ids,
        eligibility_ids=eligibility_ids,
    )
    with connection() as conn:
        after = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM policy_versions WHERE run_id = %s) AS policies,
              (SELECT count(*) FROM judge_results jr JOIN judge_invocations ji
                USING (judge_invocation_id) JOIN verification_runs vr
                USING (verification_run_id) WHERE vr.run_id = %s) AS judges,
              (SELECT count(*) FROM reward_signals WHERE run_id = %s) AS rewards
            """,
            (run_id, run_id, run_id),
        ).fetchone()
    assert after == before


@pytest.mark.integration
def test_migrations_are_recorded_in_both_authority_schemas() -> None:
    with psycopg.connect(SCIENCE_DSN) as conn:
        science = conn.execute(
            """
            SELECT version, filename, checksum, dirty
            FROM schema_migrations ORDER BY version
            """
        ).fetchall()
        assert [row[0] for row in science] == list(range(1, 12))
        assert all(row[1] and row[2].startswith("sha256:") and not row[3] for row in science)
    with psycopg.connect(OPS_DSN) as conn:
        operations = conn.execute(
            """
            SELECT version, filename, checksum, dirty
            FROM schema_migrations ORDER BY version
            """
        ).fetchall()
        assert [row[0] for row in operations] == [1, 2, 3]
        assert all(row[1] and row[2].startswith("sha256:") and not row[3] for row in operations)


@pytest.mark.integration
def test_research_compute_execution_schema_accepts_k1_ablation() -> None:
    execution_id = f"runpod-proof-k1-schema-{uuid.uuid4().hex}"
    with psycopg.connect(SCIENCE_DSN) as conn:
        row = conn.execute(
            """
            INSERT INTO research_compute_executions(
              execution_id, name, workload_id, model_id, branch_width,
              complexity_strategy, status, provider_name, started_at
            ) VALUES (
              %s, 'K=1 schema acceptance', 'repository-repair-k1-ablation',
              'Qwen/Qwen2.5-Coder-3B-Instruct', 1, 'adaptive',
              'PROVISIONING', 'RunPod', now()
            )
            RETURNING branch_width
            """,
            (execution_id,),
        ).fetchone()
        assert row[0] == 1
        conn.rollback()


@pytest.mark.integration
def test_hidden_artifact_is_not_exposed() -> None:
    suffix = uuid.uuid4().hex
    artifact_id = f"art_hidden_{suffix}"
    digest = "sha256:" + suffix.ljust(64, "0")[:64]
    ref_id = f"artifact_ref_hidden_{suffix}"
    with psycopg.connect(SCIENCE_DSN) as conn:
        conn.execute(
            """
            INSERT INTO artifacts(artifact_id, digest, object_key, media_type, size_bytes)
            VALUES (%s, %s, %s, 'application/json', 2)
            """,
            (artifact_id, digest, f"test/{suffix}"),
        )
        conn.execute(
            """
            INSERT INTO artifact_refs(
              artifact_ref_id, artifact_id, entity_type, entity_id, role, ordinal,
              visibility, trust_class
            ) VALUES (%s, %s, 'test', %s, 'hidden-test', 0, 'HIDDEN', 'TRUSTED')
            """,
            (ref_id, artifact_id, suffix),
        )
    try:
        response = httpx.get(f"{BASE}/v1/artifacts/{artifact_id}", timeout=10)
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "HIDDEN_EVIDENCE"
    finally:
        with psycopg.connect(SCIENCE_DSN) as conn:
            conn.execute("DELETE FROM artifact_refs WHERE artifact_ref_id = %s", (ref_id,))
            conn.execute("DELETE FROM artifacts WHERE artifact_id = %s", (artifact_id,))


@pytest.mark.integration
def test_out_of_order_result_without_authorized_intent_is_rejected() -> None:
    with (
        connection() as conn,
        pytest.raises(ValueError, match="no authorized intent"),
    ):
        accept_operation_result(
            conn,
            operation_id=f"op_out_of_order_{uuid.uuid4().hex}",
            result={"unexpected": True},
        )
