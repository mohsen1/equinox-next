import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from equinox_core import canonical_digest
from fastapi import HTTPException
from pydantic import ValidationError

import services.orchestrator.app.main as orchestrator_main
from services.orchestrator.app.main import (
    RECOVERABLE_PROOF_INGESTION_ERROR,
    ResearchComputeExecutionRequest,
    ResearchComputeProofRequest,
    estimated_compute_cost,
    research_execution_response,
    research_proof_can_recover_execution,
    research_proof_response,
    research_result_progress,
    research_trajectory,
    research_trajectory_source,
)
from services.orchestrator.app.providers import (
    JUDGE_PROVIDER_NAMES,
    POLICY_COMPUTE_PROVIDERS,
    assert_local_registry,
)

ROOT = Path(__file__).resolve().parents[2]


def _screen_publication(
    publication_id: str,
    *,
    profile_id: str = "qwen2.5-coder-7b-runpod-h100@7",
    run_result_digest: str | None = None,
    provider_receipt_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "publication_id": publication_id,
        "publication_type": "screen",
        "profile_id": profile_id,
        "artifact_set_manifest_digest": "sha256:" + "a" * 64,
        "set_digest": "sha256:" + "b" * 64,
        "run_result_canonical_json_sha256": (run_result_digest or "sha256:" + "c" * 64),
        "provider_receipt_canonical_json_sha256": (provider_receipt_digest or "sha256:" + "d" * 64),
    }


def _bound_screen_publication(
    proof_fields: dict[str, object],
    publication_id: str,
) -> dict[str, object]:
    resource_profile = proof_fields["resource_profile"]
    assert isinstance(resource_profile, dict)
    return _screen_publication(
        publication_id,
        profile_id=str(resource_profile["profile_id"]),
        run_result_digest=canonical_digest(proof_fields["result"]),
        provider_receipt_digest=canonical_digest(proof_fields),
    )


def _pilot_publication(
    publication_id: str,
    *,
    source_screen_publication_id: str = "runpod-proof-screen-source",
    profile_id: str = "qwen2.5-coder-7b-runpod-h100@7",
    run_result_digest: str | None = None,
    provider_receipt_digest: str | None = None,
) -> dict[str, object]:
    publication = _screen_publication(
        publication_id,
        profile_id=profile_id,
        run_result_digest=run_result_digest,
        provider_receipt_digest=provider_receipt_digest,
    )
    publication.update(
        {
            "publication_type": "pilot",
            "source_screen_publication_id": source_screen_publication_id,
            "source_screen_manifest_digest": "sha256:" + "e" * 64,
            "model_artifact_sha256": "sha256:" + "f" * 64,
            "model_artifact_size_bytes": 1024,
        }
    )
    return publication


def _committed_screen_finalization_request(
    publication_id: str,
) -> ResearchComputeExecutionRequest:
    profile_id = "qwen2.5-coder-7b-runpod-h100@10"
    return ResearchComputeExecutionRequest(
        name="Committed larger-model eligibility screen",
        workload_id="repository-repair-larger-model-eligibility",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="SUCCEEDED",
        provider_name="RunPod",
        provider_handle="runpod://pods/committed-screen",
        resource_profile={
            "profile_id": profile_id,
            "source_contract_digest": "sha256:" + "1" * 64,
        },
        progress={
            "profile_id": profile_id,
            "phase": "complete",
            "message": "Eligibility recorded and teardown confirmed.",
        },
        artifact_publication_required=True,
        artifact_publication=_screen_publication(
            publication_id,
            profile_id=profile_id,
        ),
        started_at="2026-07-30T08:00:00Z",
        completed_at="2026-07-30T08:20:00Z",
        teardown_confirmed=True,
    )


def _stored_execution(
    execution_id: str,
    request: ResearchComputeExecutionRequest,
    **overrides: object,
) -> dict[str, object]:
    return {
        **request.model_dump(mode="python"),
        "execution_id": execution_id,
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
        **overrides,
    }


def _install_execution_update_rows(
    monkeypatch: pytest.MonkeyPatch,
    existing: dict[str, object],
    *returned_rows: dict[str, object],
) -> tuple[
    dict[str, object],
    list[str],
    list[tuple[object, ...]],
]:
    state = dict(existing)
    pending_rows = [dict(row) for row in returned_rows]
    queries: list[str] = []
    parameters: list[tuple[object, ...]] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            parameters.append(params)
            if "SELECT *" in query:
                return Result([state])
            if "INSERT INTO research_compute_executions" in query:
                assert pending_rows
                state.clear()
                state.update(pending_rows.pop(0))
                return Result([state])
            raise AssertionError(f"unexpected query: {query}")

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)
    return state, queries, parameters


@pytest.mark.parametrize("publication_type", ("screen", "pilot"))
def test_launcher_envelope_filter_matches_the_strict_api_contract(
    publication_type: str,
) -> None:
    source = (ROOT / "scripts/runpod-rl-proof").read_text(encoding="utf-8")
    block = source.index('  artifact_publication="$(')
    filter_start = source.index("      '{", block) + len("      '")
    filter_end = source.index("'\n  )\"", filter_start)
    jq_filter = source[filter_start:filter_end]
    digest = "sha256:" + "a" * 64
    command = [
        "jq",
        "-cn",
        "--arg",
        "publication_id",
        "runpod-proof-launcher-contract",
        "--arg",
        "publication_type",
        publication_type,
        "--arg",
        "profile_id",
        "qwen2.5-coder-7b-runpod-h100@10",
        "--arg",
        "manifest_digest",
        digest,
        "--arg",
        "set_digest",
        digest,
        "--arg",
        "run_result_digest",
        digest,
        "--arg",
        "provider_receipt_digest",
        digest,
        "--arg",
        "source_publication_id",
        "runpod-proof-source-screen",
        "--arg",
        "source_manifest_digest",
        digest,
        "--arg",
        "model_artifact_digest",
        digest,
        "--argjson",
        "model_artifact_size",
        "1024",
        jq_filter,
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    publication = orchestrator_main.ARTIFACT_PUBLICATION_ADAPTER.validate_python(payload)

    assert publication.schema_version == 2
    assert publication.publication_type == publication_type


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


def test_research_evidence_requires_aware_ordered_timestamps() -> None:
    proof_fields = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/timestamp-proof",
        "provider_cli_version": "2.7.2",
        "resource_profile": {"gpu_id": "NVIDIA H100 80GB HBM3"},
        "workload": {"id": "repository-repair"},
        "result": {"workload": "repository-repair"},
        "teardown_confirmed": True,
    }
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **proof_fields,
            started_at="2026-07-30T08:00:00",
            completed_at="2026-07-30T08:20:00Z",
        )
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **proof_fields,
            started_at="2026-07-30T08:20:00Z",
            completed_at="2026-07-30T08:00:00Z",
        )
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            name="Invalid execution interval",
            workload_id="repository-repair",
            status="FAILED",
            provider_handle="runpod://pods/timestamp-execution",
            started_at="2026-07-30T08:20:00Z",
            completed_at="2026-07-30T08:00:00Z",
            teardown_confirmed=True,
        )


def test_at10_proof_binds_workload_model_k_and_complexity_to_result() -> None:
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/at10-workload-mismatch",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
        },
        "workload": {
            "id": "repository-repair",
            "model_id": "Different/Model",
            "static_branch_width": 4,
            "complexity_strategy": "adaptive",
        },
        "result": {
            "workload": "repository-repair",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "branch_width": 4,
            "complexity_strategy": "adaptive",
        },
        "started_at": "2026-07-30T08:00:00Z",
        "completed_at": "2026-07-30T08:20:00Z",
        "teardown_confirmed": True,
    }
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **proof_fields,
            artifact_publication=_bound_screen_publication(
                proof_fields,
                "runpod-proof-at10-workload-mismatch",
            ),
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
    assert (
        ResearchComputeExecutionRequest(
            **{
                **request.model_dump(),
                "branch_width": 1,
            }
        ).branch_width
        == 1
    )

    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **{
                **request.model_dump(),
                "branch_width": 8,
            }
        )


def test_artifact_publication_envelope_is_strict_and_reserves_marker_keys() -> None:
    base = {
        "name": "Model repair observer",
        "workload_id": "model-repair-group-policy-optimization",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "branch_width": 4,
        "complexity_strategy": "adaptive",
        "status": "RUNNING",
        "provider_handle": "runpod://pods/strict-envelope",
        "started_at": "2026-07-29T16:00:00Z",
    }

    for field, value in (
        ("progress", {"artifact_set_committed": True}),
        ("progress", {"nested": {"artifact_publication_required": True}}),
        (
            "resource_profile",
            {"nested": [{"artifact_set_manifest_digest": "sha256:" + "a" * 64}]},
        ),
    ):
        with pytest.raises(ValidationError):
            ResearchComputeExecutionRequest(**base, **{field: value})

    publication = _screen_publication("runpod-proof-strict-envelope")
    publication["committed"] = True
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **base,
            artifact_publication=publication,
        )

    pilot = _pilot_publication(
        "runpod-proof-strict-envelope",
        source_screen_publication_id="runpod-proof-strict-envelope",
    )
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **base,
            artifact_publication=pilot,
        )

    obsolete = _screen_publication("runpod-proof-obsolete-envelope")
    obsolete["schema_version"] = 1
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **base,
            artifact_publication_required=True,
            artifact_publication=obsolete,
        )


def test_operational_receipt_digest_is_valid_only_for_a_failed_execution() -> None:
    request = ResearchComputeExecutionRequest(
        name="Failed model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="FAILED",
        provider_handle="runpod://pods/failed-pod",
        progress={"phase": "failed"},
        failure_receipt_digest="sha256:" + "a" * 64,
        started_at="2026-07-29T16:00:00Z",
        completed_at="2026-07-29T16:20:00Z",
        teardown_confirmed=True,
    )

    assert request.failure_receipt_digest == "sha256:" + "a" * 64
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            **{
                **request.model_dump(),
                "status": "SUCCEEDED",
            }
        )


def test_execution_success_requires_the_atomic_artifact_set_commit() -> None:
    request = ResearchComputeExecutionRequest(
        name="Uncommitted model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="SUCCEEDED",
        provider_handle="runpod://pods/uncommitted-success",
        artifact_publication_required=True,
        progress={"phase": "complete"},
        started_at="2026-07-29T16:00:00Z",
        completed_at="2026-07-29T16:20:00Z",
        teardown_confirmed=True,
    )

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-uncommitted-success",
            request,
        )

    assert error.value.status_code == 422
    assert error.value.detail == {"code": "RESEARCH_EXECUTION_ARTIFACT_SET_UNCOMMITTED"}


def test_execution_publication_id_must_match_the_path() -> None:
    request = ResearchComputeExecutionRequest(
        name="Published model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="SUCCEEDED",
        provider_handle="runpod://pods/publication-path",
        artifact_publication_required=True,
        artifact_publication=_screen_publication("runpod-proof-another-run"),
        started_at="2026-07-29T16:00:00Z",
        completed_at="2026-07-29T16:20:00Z",
        teardown_confirmed=True,
    )

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-publication-path",
            request,
        )

    assert error.value.status_code == 422
    assert error.value.detail == {"code": "RESEARCH_EXECUTION_ARTIFACT_PUBLICATION_MISMATCH"}


def test_execution_publication_is_persisted_once_and_replayed_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screen_publication("runpod-proof-publication-once")
    request = ResearchComputeExecutionRequest(
        name="Published model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="RUNNING",
        provider_handle="runpod://pods/publication-once",
        artifact_publication_required=True,
        artifact_publication=publication,
        started_at="2026-07-29T16:00:00Z",
    )
    existing = {
        **request.model_dump(mode="python"),
        "execution_id": "runpod-proof-publication-once",
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
    }
    queries: list[str] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            return Result([existing])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    stored = orchestrator_main.update_research_compute_execution(
        "runpod-proof-publication-once",
        request,
    )

    assert stored["artifact_publication"] == publication
    assert "artifact_publication = COALESCE" in queries[-1]

    conflicting_publication = dict(publication)
    conflicting_publication["artifact_set_manifest_digest"] = "sha256:" + "9" * 64
    conflicting = ResearchComputeExecutionRequest(
        **{
            **request.model_dump(mode="python"),
            "artifact_publication": conflicting_publication,
        }
    )
    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-publication-once",
            conflicting,
        )
    assert error.value.status_code == 409
    assert error.value.detail == {"code": "RESEARCH_EXECUTION_ARTIFACT_PUBLICATION_CONFLICT"}

    omitted = ResearchComputeExecutionRequest(
        **{
            **request.model_dump(mode="python"),
            "artifact_publication": None,
        }
    )
    with pytest.raises(HTTPException) as omission:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-publication-once",
            omitted,
        )
    assert omission.value.detail == {"code": "RESEARCH_EXECUTION_ARTIFACT_PUBLICATION_CONFLICT"}

    downgraded = ResearchComputeExecutionRequest(
        **{
            **request.model_dump(mode="python"),
            "artifact_publication_required": False,
            "artifact_publication": None,
        }
    )
    with pytest.raises(HTTPException) as requirement:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-publication-once",
            downgraded,
        )
    assert requirement.value.detail == {
        "code": "RESEARCH_EXECUTION_ARTIFACT_PUBLICATION_REQUIREMENT_CONFLICT"
    }


def test_committed_screen_recovers_a_dropped_terminal_put_from_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "runpod-proof-dropped-screen-terminal"
    request = _committed_screen_finalization_request(execution_id)
    existing = _stored_execution(
        execution_id,
        request,
        status="FINALIZING",
        artifact_publication=None,
        progress={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            "phase": "finalizing",
        },
        completed_at=None,
        teardown_confirmed=False,
    )
    returned = _stored_execution(execution_id, request)
    _state, queries, _parameters = _install_execution_update_rows(
        monkeypatch,
        existing,
        returned,
    )

    recovered = orchestrator_main.update_research_compute_execution(
        execution_id,
        request,
    )

    assert recovered["status"] == "SUCCEEDED"
    assert recovered["teardown_confirmed"] is True
    assert (
        recovered["artifact_publication"]
        == request.model_dump(mode="python")["artifact_publication"]
    )
    assert len(queries) == 2


def test_committed_screen_cannot_skip_the_finalizing_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "runpod-proof-screen-still-running"
    request = _committed_screen_finalization_request(execution_id)
    existing = _stored_execution(
        execution_id,
        request,
        status="RUNNING",
        artifact_publication=None,
        progress={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            "phase": "running",
        },
        completed_at=None,
        teardown_confirmed=False,
    )
    _state, queries, _parameters = _install_execution_update_rows(
        monkeypatch,
        existing,
    )

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            execution_id,
            request,
        )

    assert error.value.status_code == 409
    assert error.value.detail == {"code": "RESEARCH_EXECUTION_TERMINAL"}
    assert len(queries) == 1


def test_committed_screen_recovers_the_exact_publication_failure_and_retains_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "runpod-proof-recoverable-screen-failure"
    request = _committed_screen_finalization_request(execution_id)
    failure_receipt_digest = "sha256:" + "f" * 64
    existing = _stored_execution(
        execution_id,
        request,
        status="FAILED",
        artifact_publication=None,
        progress={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            "phase": "failed",
            "operator_error": {
                "code": "RUNPOD_OPERATOR_FAILURE",
                "message": (
                    "The completed larger-model eligibility screen could not be published."
                ),
            },
        },
        completed_at=request.completed_at,
        teardown_confirmed=True,
        failure_receipt_digest=failure_receipt_digest,
    )
    returned = _stored_execution(
        execution_id,
        request,
        failure_receipt_digest=failure_receipt_digest,
    )
    _state, queries, parameters = _install_execution_update_rows(
        monkeypatch,
        existing,
        returned,
    )

    recovered = orchestrator_main.update_research_compute_execution(
        execution_id,
        request,
    )

    assert recovered["status"] == "SUCCEEDED"
    assert recovered["failure_receipt_digest"] == failure_receipt_digest
    assert request.failure_receipt_digest is None
    assert parameters[-1][-1] is None
    assert "failure_receipt_digest = COALESCE" in queries[-1]
    assert "research_compute_executions.progress" in queries[-1]
    assert "- 'operator_error'" in queries[-1]


def test_recovered_committed_screen_is_exactly_idempotent_with_retained_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "runpod-proof-idempotent-recovered-screen"
    request = _committed_screen_finalization_request(execution_id)
    failure_receipt_digest = "sha256:" + "f" * 64
    existing = _stored_execution(
        execution_id,
        request,
        failure_receipt_digest=failure_receipt_digest,
    )
    returned = dict(existing)
    _state, queries, _parameters = _install_execution_update_rows(
        monkeypatch,
        existing,
        returned,
    )

    replayed = orchestrator_main.update_research_compute_execution(
        execution_id,
        request,
    )

    assert replayed["status"] == "SUCCEEDED"
    assert replayed["failure_receipt_digest"] == failure_receipt_digest
    assert len(queries) == 2


@pytest.mark.parametrize(
    ("progress", "teardown_confirmed"),
    (
        (
            {
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
                "phase": "failed",
                "operator_error": {
                    "code": "RUNPOD_OPERATOR_FAILURE",
                    "message": "RunPod teardown could not be confirmed.",
                },
            },
            True,
        ),
        (
            {
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
                "phase": "failed",
                "operator_error": {
                    "code": "RUNPOD_OPERATOR_FAILURE",
                    "message": (
                        "The completed larger-model eligibility screen could not be published."
                    ),
                },
            },
            False,
        ),
    ),
)
def test_committed_screen_does_not_recover_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    progress: dict[str, object],
    teardown_confirmed: bool,
) -> None:
    execution_id = "runpod-proof-nonrecoverable-screen-failure"
    request = _committed_screen_finalization_request(execution_id)
    existing = _stored_execution(
        execution_id,
        request,
        status="FAILED",
        artifact_publication=None,
        progress=progress,
        completed_at=request.completed_at,
        teardown_confirmed=teardown_confirmed,
        failure_receipt_digest="sha256:" + "f" * 64,
    )
    _state, queries, _parameters = _install_execution_update_rows(
        monkeypatch,
        existing,
    )

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            execution_id,
            request,
        )

    assert error.value.status_code == 409
    assert error.value.detail == {"code": "RESEARCH_EXECUTION_TERMINAL"}
    assert len(queries) == 1


def test_stale_failed_demotion_remains_recoverable_but_cannot_overwrite_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "runpod-proof-stale-screen-demotion"
    success_request = _committed_screen_finalization_request(execution_id)
    failure_receipt_digest = "sha256:" + "e" * 64
    failed_request = ResearchComputeExecutionRequest(
        **{
            **success_request.model_dump(mode="python"),
            "status": "FAILED",
            "progress": {
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
                "phase": "failed",
                "operator_error": {
                    "code": "RUNPOD_OPERATOR_FAILURE",
                    "message": (
                        "The completed larger-model eligibility screen could not be published."
                    ),
                },
            },
            "failure_receipt_digest": failure_receipt_digest,
        }
    )
    initial = _stored_execution(
        execution_id,
        success_request,
        status="FINALIZING",
        artifact_publication=None,
        completed_at=None,
        teardown_confirmed=False,
    )
    failed = _stored_execution(
        execution_id,
        failed_request,
        failure_receipt_digest=failure_receipt_digest,
    )
    succeeded = _stored_execution(
        execution_id,
        success_request,
        failure_receipt_digest=failure_receipt_digest,
    )
    state, _queries, _parameters = _install_execution_update_rows(
        monkeypatch,
        initial,
        failed,
        succeeded,
    )

    demoted = orchestrator_main.update_research_compute_execution(
        execution_id,
        failed_request,
    )
    assert demoted["status"] == "FAILED"

    recovered = orchestrator_main.update_research_compute_execution(
        execution_id,
        success_request,
    )
    assert recovered["status"] == "SUCCEEDED"
    assert recovered["failure_receipt_digest"] == failure_receipt_digest

    with pytest.raises(HTTPException) as stale:
        orchestrator_main.update_research_compute_execution(
            execution_id,
            failed_request,
        )

    assert stale.value.detail == {"code": "RESEARCH_EXECUTION_TERMINAL"}
    assert state["status"] == "SUCCEEDED"
    assert state["failure_receipt_digest"] == failure_receipt_digest


def test_at10_profile_cannot_omit_the_required_publication_contract() -> None:
    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            name="Required @10 screen",
            workload_id="repository-repair-larger-model-eligibility",
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            branch_width=4,
            complexity_strategy="adaptive",
            status="PROVISIONING",
            resource_profile={
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            },
            started_at="2026-07-29T16:00:00Z",
        )

    with pytest.raises(ValidationError):
        ResearchComputeExecutionRequest(
            name="Required @10 screen from progress",
            workload_id="repository-repair-larger-model-eligibility",
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            branch_width=4,
            complexity_strategy="adaptive",
            status="PROVISIONING",
            progress={
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            },
            started_at="2026-07-29T16:00:00Z",
        )

    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            provider_name="RunPod",
            provider_handle="runpod://pods/at10-uncommitted",
            provider_cli_version="2.7.2",
            resource_profile={
                "gpu_id": "NVIDIA H100 80GB HBM3",
                "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            },
            workload={"id": "repository-repair"},
            result={"workload": "repository-repair"},
            started_at="2026-07-29T16:00:00Z",
            completed_at="2026-07-29T16:20:00Z",
            teardown_confirmed=True,
        )

    predecessor = ResearchComputeExecutionRequest(
        name="Predecessor profile remains legacy",
        workload_id="repository-repair-larger-model-eligibility",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="PROVISIONING",
        resource_profile={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@9",
        },
        started_at="2026-07-29T16:00:00Z",
    )
    assert predecessor.artifact_publication_required is False


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ({"branch_width": 1}, "RESEARCH_EXECUTION_CONFLICT"),
        (
            {"provider_handle": "runpod://pods/authorization-other"},
            "RESEARCH_EXECUTION_AUTHORIZATION_CONFLICT",
        ),
        (
            {
                "resource_profile": {
                    "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
                    "source_contract_digest": "sha256:" + "2" * 64,
                    "maximum_total_cost_usd": 25.0,
                },
            },
            "RESEARCH_EXECUTION_AUTHORIZATION_CONFLICT",
        ),
        (
            {
                "resource_profile": {
                    "profile_id": "qwen2.5-coder-7b-runpod-h100@9",
                    "source_contract_digest": "sha256:" + "1" * 64,
                    "maximum_total_cost_usd": 25.0,
                },
                "progress": {
                    "profile_id": "qwen2.5-coder-7b-runpod-h100@9",
                },
            },
            "RESEARCH_EXECUTION_AUTHORIZATION_CONFLICT",
        ),
    ),
)
def test_atomic_execution_authorization_fields_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    expected_code: str,
) -> None:
    base = ResearchComputeExecutionRequest(
        name="Atomic authorization boundary",
        workload_id="repository-repair-larger-model-eligibility",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="RUNNING",
        provider_handle="runpod://pods/authorization",
        resource_profile={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
            "source_contract_digest": "sha256:" + "1" * 64,
            "maximum_total_cost_usd": 25.0,
        },
        progress={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
        },
        artifact_publication_required=True,
        started_at="2026-07-29T16:00:00Z",
    )
    existing = {
        **base.model_dump(mode="python"),
        "execution_id": "runpod-proof-authorization",
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
    }
    updated = ResearchComputeExecutionRequest(
        **{
            **base.model_dump(mode="python"),
            **mutation,
        }
    )
    queries: list[str] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            return Result([existing])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-authorization",
            updated,
        )

    assert error.value.status_code == 409
    assert error.value.detail == {"code": expected_code}
    assert len(queries) == 1


def test_required_publication_must_be_declared_on_initial_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ResearchComputeExecutionRequest(
        name="Late atomic declaration",
        workload_id="repository-repair-larger-model-eligibility",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="RUNNING",
        provider_handle="runpod://pods/late-declaration",
        artifact_publication_required=True,
        started_at="2026-07-29T16:00:00Z",
    )

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            _query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            return Result([])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    with pytest.raises(HTTPException) as error:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-late-declaration",
            request,
        )
    assert error.value.detail == {
        "code": "RESEARCH_EXECUTION_PUBLICATION_REQUIRES_INITIAL_PROVISIONING"
    }


def test_initial_provisioning_persists_the_required_publication_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ResearchComputeExecutionRequest(
        name="Atomic @10 provisioning",
        workload_id="repository-repair-larger-model-eligibility",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="PROVISIONING",
        resource_profile={
            "profile_id": "qwen2.5-coder-7b-runpod-h100@10",
        },
        artifact_publication_required=True,
        started_at="2026-07-29T16:00:00Z",
    )
    stored = {
        **request.model_dump(mode="python"),
        "execution_id": "runpod-proof-atomic-provisioning",
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
    }
    queries: list[str] = []
    parameters: list[tuple[object, ...]] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            parameters.append(params)
            if "SELECT *" in query:
                return Result([])
            return Result([stored])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    result = orchestrator_main.update_research_compute_execution(
        "runpod-proof-atomic-provisioning",
        request,
    )

    assert result["artifact_publication_required"] is True
    assert "artifact_publication_required" in queries[-1]
    assert parameters[-1][11] is True
    assert parameters[-1][12] is None


def test_failed_execution_attaches_one_operational_receipt_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    request = ResearchComputeExecutionRequest(
        name="Failed model repair observer",
        workload_id="model-repair-group-policy-optimization",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="FAILED",
        provider_handle="runpod://pods/failed-pod",
        resource_profile={"gpu_id": "NVIDIA H100 80GB HBM3"},
        progress={
            "phase": "failed",
            "remote_error": {
                "code": "REMOTE_WORKLOAD_FAILURE",
                "message": "RuntimeError: retained policy update missing",
            },
            "operator_error": {
                "code": "RUNPOD_OPERATOR_FAILURE",
                "message": "The remote workload failed with exit code 1.",
            },
        },
        failure_receipt_digest=digest,
        started_at="2026-07-29T16:00:00Z",
        completed_at="2026-07-29T16:20:00Z",
        teardown_confirmed=True,
    )
    existing = {
        **request.model_dump(),
        "execution_id": "runpod-proof-failed",
        "proof_id": None,
        "failure_receipt_digest": None,
    }
    queries: list[str] = []
    parameters: list[tuple[object, ...]] = []
    selected_rows: list[dict[str, object]] = [existing]
    stored_digest = [digest]

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            parameters.append(params)
            if "SELECT *" in query:
                return Result(selected_rows)
            return Result([{**existing, "failure_receipt_digest": stored_digest[0]}])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    attached = orchestrator_main.update_research_compute_execution(
        "runpod-proof-failed",
        request,
    )

    assert attached["proof_id"] is None
    assert attached["failure_receipt_digest"] == digest
    assert "FOR UPDATE" in queries[0]
    assert "COALESCE" in queries[-1]
    assert parameters[-1][-1] == digest
    assert all("research_compute_proofs" not in query for query in queries)

    existing["failure_receipt_digest"] = digest
    queries.clear()
    replayed = orchestrator_main.update_research_compute_execution(
        "runpod-proof-failed",
        request,
    )
    assert replayed is existing
    assert len(queries) == 1

    for replay_digest in (None, "sha256:" + "b" * 64):
        conflicting = ResearchComputeExecutionRequest(
            **{
                **request.model_dump(),
                "failure_receipt_digest": replay_digest,
            }
        )
        with pytest.raises(HTTPException) as error:
            orchestrator_main.update_research_compute_execution(
                "runpod-proof-failed",
                conflicting,
            )
        assert error.value.status_code == 409
        assert error.value.detail == {"code": "RESEARCH_EXECUTION_RECEIPT_CONFLICT"}

    existing["failure_receipt_digest"] = None
    stored_digest[0] = "sha256:" + "b" * 64
    with pytest.raises(HTTPException) as losing_writer:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-failed",
            request,
        )
    assert losing_writer.value.detail == {"code": "RESEARCH_EXECUTION_RECEIPT_CONFLICT"}

    selected_rows.clear()
    with pytest.raises(HTTPException) as unregistered:
        orchestrator_main.update_research_compute_execution(
            "runpod-proof-failed",
            request,
        )
    assert unregistered.value.detail == {"code": "RESEARCH_EXECUTION_RECEIPT_REQUIRES_REGISTRATION"}


def test_proof_publication_binds_exact_canonical_result_and_receipt() -> None:
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/canonical-proof",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": "qwen2.5-coder-7b-runpod-h100@7",
        },
        "workload": {"id": "repository-repair"},
        "result": {"workload": "repository-repair", "reward_gain": 0.125},
        "started_at": "2026-07-29T16:00:00Z",
        "completed_at": "2026-07-29T16:20:00Z",
        "teardown_confirmed": True,
    }
    publication = _bound_screen_publication(
        proof_fields,
        "runpod-proof-canonical-proof",
    )
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=publication,
    )

    assert request.artifact_publication.run_result_canonical_json_sha256 == canonical_digest(
        request.result
    )
    assert request.artifact_publication.provider_receipt_canonical_json_sha256 == canonical_digest(
        request.model_dump(mode="json", exclude={"artifact_publication"})
    )
    assert canonical_digest(request.model_dump(mode="json")) != (
        request.artifact_publication.provider_receipt_canonical_json_sha256
    )

    wrong_result = dict(publication)
    wrong_result["run_result_canonical_json_sha256"] = "sha256:" + "9" * 64
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **proof_fields,
            artifact_publication=wrong_result,
        )

    wrong_receipt = dict(publication)
    wrong_receipt["provider_receipt_canonical_json_sha256"] = "sha256:" + "8" * 64
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **proof_fields,
            artifact_publication=wrong_receipt,
        )

    marked_result = dict(proof_fields)
    marked_result["result"] = {
        "workload": "repository-repair",
        "artifact_set_committed": True,
    }
    with pytest.raises(ValidationError):
        ResearchComputeProofRequest(
            **marked_result,
            artifact_publication=publication,
        )


def test_proof_receipt_recovers_only_the_exact_post_contract_failure() -> None:
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/recovered-pod",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA RTX PRO 4500 Blackwell",
            "profile_id": "qwen2.5-coder-7b-runpod-h100@7",
        },
        "workload": {
            "id": "repository-repair-restored-continuation-post-training",
            "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "result": {
            "workload": "repository-repair-restored-continuation-post-training",
            "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "reward_gain": 0.1875,
        },
        "started_at": "2026-07-28T14:24:04Z",
        "completed_at": "2026-07-28T15:56:46Z",
        "teardown_confirmed": True,
    }
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=_bound_screen_publication(
            proof_fields,
            "runpod-proof-recovered",
        ),
    )
    execution = {
        "execution_id": "runpod-proof-recovered",
        "status": "FAILED",
        "provider_handle": "runpod://pods/recovered-pod",
        "workload_id": "repository-repair-restored-continuation-post-training",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "started_at": request.started_at,
        "teardown_confirmed": True,
        "progress": {"error": "The remote result did not satisfy the declared proof contract."},
    }

    assert research_proof_can_recover_execution(execution, request) is True
    assert (
        research_proof_can_recover_execution(
            {
                **execution,
                "progress": {
                    "operator_error": {
                        "code": "RUNPOD_OPERATOR_FAILURE",
                        "message": (
                            "The remote result did not satisfy the declared proof contract."
                        ),
                    }
                },
            },
            request,
        )
        is True
    )
    assert (
        research_proof_can_recover_execution(
            {
                **execution,
                "progress": {"error": "A verified local proof receipt is pending ingestion."},
            },
            request,
        )
        is True
    )
    assert (
        research_proof_can_recover_execution(
            {
                **execution,
                "progress": {"error": "The remote workload failed with exit code 1."},
            },
            request,
        )
        is False
    )
    assert (
        research_proof_can_recover_execution(
            {**execution, "teardown_confirmed": False},
            request,
        )
        is False
    )


def test_scientific_recovery_locks_execution_and_preserves_failure_receipt_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_digest = "sha256:" + "f" * 64
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/recovered-lineage",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": "qwen2.5-coder-7b-runpod-h100@7",
        },
        "workload": {
            "id": "repository-repair-restored-continuation-post-training",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "result": {
            "workload": "repository-repair-restored-continuation-post-training",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "reward_gain": 0.125,
        },
        "started_at": "2026-07-29T16:00:00Z",
        "completed_at": "2026-07-29T16:20:00Z",
        "teardown_confirmed": True,
    }
    publication = _bound_screen_publication(
        proof_fields,
        "runpod-proof-recovered-lineage",
    )
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=publication,
    )
    execution = {
        "execution_id": "runpod-proof-recovered-lineage",
        "status": "FAILED",
        "provider_handle": request.provider_handle,
        "workload_id": request.workload["id"],
        "model_id": request.workload["model_id"],
        "started_at": request.started_at,
        "teardown_confirmed": True,
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": failure_digest,
        "artifact_publication_required": True,
        "artifact_publication": None,
        "resource_profile": proof_fields["resource_profile"],
        "progress": {
            "operator_error": {
                "code": "RUNPOD_OPERATOR_FAILURE",
                "message": RECOVERABLE_PROOF_INGESTION_ERROR,
            }
        },
    }
    queries: list[str] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            if "SELECT *" in query and "provider_handle" in query:
                return Result([execution])
            if "SELECT proof_id" in query:
                return Result([])
            if "UPDATE research_compute_executions SET" in query:
                return Result([{"execution_id": execution["execution_id"]}])
            return Result([])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    result = orchestrator_main.ingest_research_compute_proof(request)

    assert result["proof_id"].startswith("research_proof_")
    execution_select = next(
        query for query in queries if "SELECT *" in query and "provider_handle" in query
    )
    assert "FOR UPDATE" in execution_select
    execution_update = next(
        query for query in queries if "UPDATE research_compute_executions SET" in query
    )
    assert "receipt_digest = %s" in execution_update
    assert "failure_receipt_digest" not in execution_update


def test_proof_ingestion_refuses_an_uncommitted_required_artifact_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ResearchComputeProofRequest(
        provider_name="RunPod",
        provider_handle="runpod://pods/uncommitted",
        provider_cli_version="2.7.2",
        resource_profile={
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": "qwen2.5-coder-7b-runpod-h100@7",
        },
        workload={"id": "repository-repair"},
        result={"workload": "repository-repair"},
        started_at="2026-07-29T16:00:00Z",
        completed_at="2026-07-29T16:20:00Z",
        teardown_confirmed=True,
    )
    execution = {
        "execution_id": "runpod-proof-uncommitted",
        "artifact_publication_required": True,
        "artifact_publication": None,
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
            return Result([execution])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    with pytest.raises(HTTPException) as error:
        orchestrator_main.ingest_research_compute_proof(request)

    assert error.value.status_code == 422
    assert error.value.detail == {"code": "RESEARCH_PROOF_ARTIFACT_SET_UNCOMMITTED"}


def test_legacy_execution_and_proof_remain_non_atomic_and_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_request = ResearchComputeExecutionRequest(
        name="Legacy external evaluation",
        workload_id="revision30-post-freeze-external-adapter-evaluation",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        branch_width=4,
        complexity_strategy="adaptive",
        status="SUCCEEDED",
        provider_handle="runpod://pods/legacy-evaluation",
        resource_profile={"gpu_id": "NVIDIA H100 80GB HBM3"},
        progress={"phase": "complete", "elapsed_seconds": 60},
        started_at="2026-07-28T16:00:00Z",
        completed_at="2026-07-28T16:01:00Z",
        teardown_confirmed=True,
    )
    stored_execution = {
        **execution_request.model_dump(mode="python"),
        "execution_id": "runpod-proof-legacy-evaluation",
        "proof_id": None,
        "receipt_digest": None,
        "failure_receipt_digest": None,
        "progress": {
            **execution_request.progress,
            "artifact_set_committed": True,
            "artifact_set_manifest_digest": "sha256:" + "9" * 64,
        },
    }
    execution_queries: list[str] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class ExecutionConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            execution_queries.append(query)
            if "SELECT *" in query:
                return Result([])
            return Result([stored_execution])

    @contextmanager
    def execution_connection():
        yield ExecutionConnection()

    monkeypatch.setattr(orchestrator_main, "connection", execution_connection)

    stored = orchestrator_main.update_research_compute_execution(
        "runpod-proof-legacy-evaluation",
        execution_request,
    )
    observed = research_execution_response(stored)

    assert stored["artifact_publication_required"] is False
    assert stored["artifact_publication"] is None
    assert observed["status"] == "SUCCEEDED"
    assert "artifact_set_committed" not in observed["observer_evidence"]
    assert "artifact_publication_required" in execution_queries[-1]

    proof_request = ResearchComputeProofRequest(
        provider_name="RunPod",
        provider_handle="runpod://pods/legacy-evaluation",
        provider_cli_version="2.7.2",
        resource_profile={"gpu_id": "NVIDIA H100 80GB HBM3"},
        workload={
            "id": "revision30-post-freeze-external-adapter-evaluation",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        result={
            "workload": "revision30-post-freeze-external-adapter-evaluation",
            "elapsed_seconds": 60,
        },
        started_at="2026-07-28T16:00:00Z",
        completed_at="2026-07-28T16:01:00Z",
        teardown_confirmed=True,
    )
    legacy_execution = {
        **stored_execution,
        "status": "FINALIZING",
        "proof_id": None,
    }
    proof_queries: list[str] = []
    proof_parameters: list[tuple[object, ...]] = []

    class ProofConnection:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] = (),
        ) -> Result:
            proof_queries.append(query)
            proof_parameters.append(params)
            if "SELECT *" in query and "provider_handle" in query:
                return Result([legacy_execution])
            if "UPDATE research_compute_executions SET" in query:
                return Result([{"execution_id": legacy_execution["execution_id"]}])
            return Result([])

    @contextmanager
    def proof_connection():
        yield ProofConnection()

    monkeypatch.setattr(orchestrator_main, "connection", proof_connection)

    proof = orchestrator_main.ingest_research_compute_proof(proof_request)
    legacy_receipt = proof_request.model_dump(
        mode="json",
        exclude={"artifact_publication"},
    )

    assert proof["receipt_digest"] == canonical_digest(legacy_receipt)
    proof_insert_index = next(
        index
        for index, query in enumerate(proof_queries)
        if "INSERT INTO research_compute_proofs" in query
    )
    assert proof_parameters[proof_insert_index][7] is None
    legacy_item = {
        "proof_id": proof["proof_id"],
        "execution_id": "runpod-proof-legacy-evaluation",
        "execution_name": "Legacy external evaluation",
        "provider_name": "RunPod",
        "provider_handle": proof_request.provider_handle,
        "provider_cli_version": proof_request.provider_cli_version,
        "resource_profile": proof_request.resource_profile,
        "workload": proof_request.workload,
        "result": proof_request.result,
        "artifact_publication": None,
        "receipt_digest": proof["receipt_digest"],
        "failure_receipt_digest": None,
        "started_at": proof_request.started_at,
        "completed_at": proof_request.completed_at,
        "teardown_confirmed": True,
    }
    detail = research_proof_response(legacy_item, detail=True)
    assert detail["evidence"]["artifact_set_committed"] is False
    assert detail["evidence"]["artifact_set_manifest_digest"] is None
    assert detail["evidence"]["artifact_publication_status"] == "legacy_non_atomic"

    class ListConnection:
        def execute(
            self,
            _query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            return Result([legacy_item])

    @contextmanager
    def list_connection():
        yield ListConnection()

    monkeypatch.setattr(orchestrator_main, "connection", list_connection)

    listed = orchestrator_main.list_research_compute_proofs()
    assert listed["items"][0]["proof_id"] == proof["proof_id"]


def test_proof_ingestion_requires_the_exact_registered_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/missing-execution",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": "qwen2.5-coder-7b-runpod-h100@7",
        },
        "workload": {"id": "repository-repair"},
        "result": {"workload": "repository-repair"},
        "started_at": "2026-07-29T16:00:00Z",
        "completed_at": "2026-07-29T16:20:00Z",
        "teardown_confirmed": True,
    }
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=_bound_screen_publication(
            proof_fields,
            "runpod-proof-missing-execution",
        ),
    )

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    executions: list[dict[str, object]] = []

    class FakeConnection:
        def execute(
            self,
            _query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            return Result(executions)

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    with pytest.raises(HTTPException) as error:
        orchestrator_main.ingest_research_compute_proof(request)

    assert error.value.status_code == 422
    assert error.value.detail == {"code": "RESEARCH_PROOF_EXECUTION_REQUIRED"}

    executions.append(
        {
            "execution_id": "runpod-proof-different-execution",
            "artifact_publication_required": True,
            "artifact_publication": None,
        }
    )
    with pytest.raises(HTTPException) as mismatch:
        orchestrator_main.ingest_research_compute_proof(request)
    assert mismatch.value.detail == {"code": "RESEARCH_PROOF_ARTIFACT_PUBLICATION_MISMATCH"}


@pytest.mark.parametrize(
    ("execution_mutation", "expected_code"),
    (
        (
            {"workload_id": "different-workload"},
            "RESEARCH_PROOF_EXECUTION_LINEAGE_MISMATCH",
        ),
        (
            {"model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
            "RESEARCH_PROOF_EXECUTION_LINEAGE_MISMATCH",
        ),
        (
            {"started_at": "2026-07-29T15:59:59Z"},
            "RESEARCH_PROOF_EXECUTION_LINEAGE_MISMATCH",
        ),
        (
            {
                "resource_profile": {
                    "profile_id": "qwen2.5-coder-7b-runpod-h100@8",
                }
            },
            "RESEARCH_PROOF_EXECUTION_PROFILE_MISMATCH",
        ),
    ),
)
def test_normal_proof_ingestion_rejects_execution_lineage_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    execution_mutation: dict[str, object],
    expected_code: str,
) -> None:
    profile_id = "qwen2.5-coder-7b-runpod-h100@7"
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/exact-lineage",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": profile_id,
        },
        "workload": {
            "id": "repository-repair",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "result": {
            "workload": "repository-repair",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "started_at": "2026-07-29T16:00:00Z",
        "completed_at": "2026-07-29T16:20:00Z",
        "teardown_confirmed": True,
    }
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=_bound_screen_publication(
            proof_fields,
            "runpod-proof-exact-lineage",
        ),
    )
    execution = {
        "execution_id": "runpod-proof-exact-lineage",
        "status": "FINALIZING",
        "provider_name": "RunPod",
        "provider_handle": request.provider_handle,
        "workload_id": "repository-repair",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "resource_profile": {"profile_id": profile_id},
        "progress": {},
        "started_at": request.started_at,
        "teardown_confirmed": True,
        "artifact_publication_required": True,
        "artifact_publication": None,
        "proof_id": None,
        **execution_mutation,
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
            return Result([execution])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    with pytest.raises(HTTPException) as error:
        orchestrator_main.ingest_research_compute_proof(request)

    assert error.value.status_code == 422
    assert error.value.detail == {"code": expected_code}


def test_pilot_proof_verifies_source_screen_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_id = "qwen2.5-coder-7b-runpod-h100@7"
    proof_fields: dict[str, object] = {
        "provider_name": "RunPod",
        "provider_handle": "runpod://pods/pilot-lineage",
        "provider_cli_version": "2.7.2",
        "resource_profile": {
            "gpu_id": "NVIDIA H100 80GB HBM3",
            "profile_id": profile_id,
        },
        "workload": {"id": "repository-repair"},
        "result": {"workload": "repository-repair", "reward_gain": 0.25},
        "started_at": "2026-07-29T16:00:00Z",
        "completed_at": "2026-07-29T16:20:00Z",
        "teardown_confirmed": True,
    }
    publication = _pilot_publication(
        "runpod-proof-pilot-lineage",
        profile_id=profile_id,
        run_result_digest=canonical_digest(proof_fields["result"]),
        provider_receipt_digest=canonical_digest(proof_fields),
    )
    request = ResearchComputeProofRequest(
        **proof_fields,
        artifact_publication=publication,
    )
    primary_execution = {
        "execution_id": "runpod-proof-pilot-lineage",
        "status": "FINALIZING",
        "provider_handle": request.provider_handle,
        "workload_id": "repository-repair",
        "model_id": None,
        "started_at": request.started_at,
        "teardown_confirmed": True,
        "artifact_publication_required": True,
        "artifact_publication": None,
        "proof_id": None,
        "resource_profile": {"profile_id": profile_id},
        "progress": {},
    }
    source_publication = _screen_publication(
        "runpod-proof-screen-source",
        profile_id=profile_id,
    )
    source_publication["artifact_set_manifest_digest"] = "sha256:" + "e" * 64
    source_execution = {
        "execution_id": "runpod-proof-screen-source",
        "status": "SUCCEEDED",
        "teardown_confirmed": True,
        "artifact_publication_required": True,
        "artifact_publication": source_publication,
    }
    queries: list[str] = []

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    class FakeConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            queries.append(query)
            if "WHERE provider_handle" in query and "SELECT *" in query:
                return Result([primary_execution])
            if "SELECT execution_id, status" in query:
                return Result([source_execution])
            if "UPDATE research_compute_executions SET" in query:
                return Result([{"execution_id": primary_execution["execution_id"]}])
            return Result([])

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(orchestrator_main, "connection", fake_connection)

    response = orchestrator_main.ingest_research_compute_proof(request)

    assert response["already_recorded"] is False
    assert any(
        "artifact_publication" in query and "INSERT INTO research_compute_proofs" in query
        for query in queries
    )

    source_execution["status"] = "RUNNING"
    with pytest.raises(HTTPException) as invalid_source:
        orchestrator_main.ingest_research_compute_proof(request)
    assert invalid_source.value.detail == {
        "code": "RESEARCH_ARTIFACT_PUBLICATION_SOURCE_SCREEN_INVALID"
    }


def test_operational_failure_receipt_migration_preserves_scientific_lineage() -> None:
    migration = Path(
        "services/orchestrator/migrations/012_operational_failure_receipts.sql"
    ).read_text(encoding="utf-8")

    assert "ADD COLUMN IF NOT EXISTS failure_receipt_digest" in migration
    assert "SET failure_receipt_digest = receipt_digest" in migration
    assert "status = 'FAILED'" in migration
    assert "proof_id IS NULL" in migration


def test_artifact_publication_migration_adds_nullable_jsonb_to_both_records() -> None:
    migration = Path("services/orchestrator/migrations/013_artifact_publication.sql").read_text(
        encoding="utf-8"
    )

    assert "ALTER TABLE research_compute_executions" in migration
    assert "ALTER TABLE research_compute_proofs" in migration
    assert migration.count("ADD COLUMN IF NOT EXISTS artifact_publication jsonb") == 2
    assert "ADD COLUMN IF NOT EXISTS artifact_publication_required boolean" in migration
    assert "NOT NULL DEFAULT false" in migration
    assert "research_compute_executions_required_publication_check" in migration
    assert "NOT artifact_publication_required" in migration
    assert "status <> 'SUCCEEDED'" in migration
    assert "OR artifact_publication IS NOT NULL" in migration
    assert "artifact_publication_required" in migration
    assert "AND jsonb_typeof(artifact_publication) = 'object'" in migration
    assert "INSERT INTO schema_migrations(version) VALUES (13)" in migration


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


def test_research_execution_response_normalizes_only_present_observer_evidence() -> None:
    bundle_digest = "sha256:" + "b" * 64
    stage_digest = "sha256:" + "c" * 64
    source_digest = "sha256:" + "d" * 64
    response = research_execution_response(
        {
            "execution_id": "runpod-proof-observer",
            "provider_handle": "runpod://pods/observer",
            "resource_profile": {
                "workload_bundle_digest": bundle_digest,
                "bundle_stage_receipt_digest": stage_digest,
                "source_contract_digest": "sha256:" + "e" * 64,
                "internal_launcher_value": "not-public-observer-evidence",
            },
            "progress": {
                "phase": "verifying_runtime",
                "message": "Verifying the runtime and pinned model.",
                "source_contract_digest": source_digest,
                "branch_groups_completed": 3,
            },
        }
    )

    assert response["observer_evidence"] == {
        "phase": "verifying_runtime",
        "message": "Verifying the runtime and pinned model.",
        "source_contract_digest": source_digest,
        "workload_bundle_digest": bundle_digest,
        "bundle_stage_receipt_digest": stage_digest,
        "branch_groups_completed": 3,
    }
    assert "internal_launcher_value" not in response["observer_evidence"]
    assert "gate_results" not in response["observer_evidence"]


def test_execution_response_claims_success_only_after_atomic_artifact_commit() -> None:
    digest = "sha256:" + "a" * 64
    item = {
        "execution_id": "runpod-proof-publication",
        "status": "SUCCEEDED",
        "provider_handle": "runpod://pods/publication",
        "resource_profile": {},
        "progress": {"phase": "complete"},
        "proof_id": "research_proof_publication",
        "receipt_digest": "sha256:" + "b" * 64,
        "artifact_publication_required": True,
    }

    pending = research_execution_response(item)
    assert pending["status"] == "FINALIZING"
    assert pending["proof_id"] is None
    assert pending["receipt_digest"] is None

    published = research_execution_response(
        {
            **item,
            "progress": {
                "phase": "complete",
                "artifact_set_manifest_digest": digest,
                "artifact_set_committed": True,
            },
        }
    )
    assert published["status"] == "FINALIZING"

    published = research_execution_response(
        {
            **item,
            "artifact_publication": _screen_publication("runpod-proof-publication"),
        }
    )
    assert published["status"] == "SUCCEEDED"
    assert published["proof_id"] == "research_proof_publication"
    assert published["observer_evidence"]["artifact_set_manifest_digest"] == digest
    assert published["observer_evidence"]["artifact_set_committed"] is True


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
            "meaningful_post_training": False,
            "post_training_outcome": "NEGATIVE_EXPERIMENT_COMPLETED",
            "dynamic_complexity_progressed": True,
            "elapsed_seconds": 900,
            "promotion_count": 2,
            "reached_complexity_level": 2,
            "maximum_complexity_level": 3,
        },
        "artifact_publication": _screen_publication("runpod-proof-test"),
        "receipt_digest": "sha256:receipt",
        "failure_receipt_digest": "sha256:" + "f" * 64,
        "started_at": "2026-07-26T21:00:00Z",
        "completed_at": "2026-07-26T21:15:00Z",
        "teardown_confirmed": True,
    }

    summary = research_proof_response(item, detail=False)
    detail = research_proof_response(item, detail=True)

    assert summary["execution_id"] == "runpod-proof-test"
    assert summary["learning"]["reward_gain"] == 1.0
    assert summary["learning"]["meaningful_post_training"] is False
    assert summary["learning"]["post_training_outcome"] == "NEGATIVE_EXPERIMENT_COMPLETED"
    assert summary["cost"]["total_usd"] == 0.11
    assert "provider" not in summary
    assert "evidence" not in summary

    assert detail["provider"]["handle"] == "runpod://pods/pod-test"
    assert detail["workload"]["model_id"].endswith("1.5B-Instruct")
    assert detail["curriculum"]["reached_level"] == 2
    assert detail["curriculum"]["dynamic_complexity_progressed"] is True
    assert detail["evidence"]["receipt_digest"] == "sha256:receipt"
    assert detail["evidence"]["failure_receipt_digest"] == "sha256:" + "f" * 64
    assert detail["evidence"]["artifact_set_committed"] is True
    assert detail["evidence"]["artifact_set_manifest_digest"] == "sha256:" + "a" * 64


def test_partial_proof_suppresses_headline_learning_metrics() -> None:
    item = {
        "proof_id": "proof-partial",
        "execution_id": "run-partial",
        "execution_name": "Partial run",
        "provider_name": "RunPod",
        "resource_profile": {},
        "workload": {},
        "result": {
            "initial_reward": 0.25,
            "final_reward": 0.75,
            "reward_gain": 0.5,
            "paired_test_change": {
                "examples": 4,
                "improved": 4,
                "regressed": 0,
            },
            "final_evaluation_partial": True,
        },
        "completed_at": "2026-07-27T12:00:00Z",
        "teardown_confirmed": True,
    }

    learning = research_proof_response(item, detail=False)["learning"]

    assert learning["initial_reward"] is None
    assert learning["final_reward"] is None
    assert learning["reward_gain"] is None


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
        "result": {
            "elapsed_seconds": 900,
            "reward_gain": 1.0,
        },
        "artifact_publication": _screen_publication("runpod-proof-api"),
        "receipt_digest": "sha256:api",
        "failure_receipt_digest": "sha256:" + "f" * 64,
        "started_at": "2026-07-26T21:00:00Z",
        "completed_at": "2026-07-26T21:15:00Z",
        "teardown_confirmed": True,
    }

    class Result(list[dict[str, object]]):
        def fetchone(self) -> dict[str, object] | None:
            return self[0] if self else None

    proof_queries: list[str] = []

    class FakeConnection:
        def execute(
            self,
            query: str,
            _params: tuple[object, ...] = (),
        ) -> Result:
            proof_queries.append(query)
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
    assert proof_detail["evidence"]["failure_receipt_digest"] == ("sha256:" + "f" * 64)
    assert "failure_receipt_digest" not in proof_queries[0]
    assert "e.failure_receipt_digest" in proof_queries[1]


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
            "attempted_policy_update_count": 10,
            "effective_policy_update_count": 8,
            "retained_policy_update_count": 7,
            "retained_checkpoint_update": 40,
            "retention_rollback_count": 2,
            "retention_transaction_revision": "adapter-optimizer-policy-lineage@1",
            "history": [
                {
                    "update": 5,
                    "level": 0,
                    "examples": 8,
                    "exact_successes": 5,
                    "exact_rate": 0.625,
                    "checkpoint_candidate_retained": False,
                    "retention_transaction_disposition": "rollback",
                    "attempted_policy_update_count": 10,
                    "effective_policy_update_count": 8,
                    "retained_policy_update_count": 7,
                    "retention_rollback_count": 2,
                    "task_outcomes": [{"task_id": "hidden"}],
                }
            ],
            "claim_strength": "INCOMPLETE_FINAL_EVALUATION",
            "seed_count": 1,
            "objective": "leave-one-out-group-normalized-reinforce@1",
            "elapsed_seconds": 1804.33,
            "stop_reason": "final_evaluation_reserve",
            "validation_examples": 8,
            "test_examples": 12,
            "mastery_windows": 2,
            "teacher_data_used": False,
            "resumed_from_checkpoint": True,
            "attempt_count": 2,
            "attempt_elapsed_seconds": 412.5,
            "raw_resume_gap_seconds": 31.25,
            "applied_resume_gap_seconds": 31.25,
            "crash_tail_actions_unaccounted": True,
            "crash_tail_cost_accounting": "wall_clock_only",
            "final_evaluation_reserve_exceeded_ceiling": False,
            "final_evaluation_complete": False,
            "final_evaluation_partial": True,
            "final_evaluation_deadline_seconds": 9_600,
            "maximum_measured_final_evaluation_reserve_seconds": 782,
            "discarded_task_groups": 1,
            "discarded_sampled_actions": 12,
            "discarded_post_branch_actions": 10,
            "structural_mirror_disclosures": [
                {
                    "families": ["first", "safe_head"],
                    "splits": ["validation", "test"],
                }
            ],
            "paired_test_change": {
                "examples": 48,
                "improved": 19,
                "regressed": 2,
                "mcnemar_exact_p_value": 0.000221,
            },
            "initial_by_level": {
                "1": {
                    "examples": 12,
                    "split": "test",
                    "exact_rate": 0.25,
                }
            },
            "final_by_level": {
                "1": {
                    "examples": 12,
                    "distinct_semantic_examples": 12,
                    "semantic_universe_size": 12,
                    "split": "test",
                    "exact_rate": 0.333333,
                    "exact_rate_95ci": [0.1377, 0.6094],
                    "checkpoint_rate": 1.0,
                    "checkpoint_rate_95ci": [0.7575, 1.0],
                }
            },
        }
    )

    assert progress["phase"] == "complete"
    assert progress["exact_rate"] == 0.333333
    assert "initial_level_exact_rate" not in progress
    assert progress["exact_rate_source"] == "reached_level"
    assert progress["checkpoint_rate_source"] == "reached_level"
    assert progress["hypothesis_passed"] is False
    assert progress["adapter_persisted"] is True
    assert progress["post_training_completed"] is True
    assert progress["informative_group_rate"] == 0.2
    assert progress["attempted_policy_update_count"] == 10
    assert progress["effective_policy_update_count"] == 8
    assert progress["retained_policy_update_count"] == 7
    assert progress["retained_checkpoint_update"] == 40
    assert progress["retention_rollback_count"] == 2
    assert progress["retention_transaction_revision"] == "adapter-optimizer-policy-lineage@1"
    assert progress["claim_strength"] == "INCOMPLETE_FINAL_EVALUATION"
    assert progress["seed_count"] == 1
    assert progress["exact_rate_95ci"] == [0.1377, 0.6094]
    assert progress["evaluation_examples"] == 12
    assert progress["evaluation_completed"] == 12
    assert progress["evaluation_total"] == 12
    assert progress["evaluation_split"] == "test"
    assert progress["test_examples"] == 12
    assert progress["teacher_data_used"] is False
    assert progress["resumed_from_checkpoint"] is True
    assert progress["attempt_count"] == 2
    assert progress["raw_resume_gap_seconds"] == 31.25
    assert progress["applied_resume_gap_seconds"] == 31.25
    assert progress["crash_tail_actions_unaccounted"] is True
    assert progress["crash_tail_cost_accounting"] == "wall_clock_only"
    assert progress["final_evaluation_reserve_exceeded_ceiling"] is False
    assert progress["final_evaluation_complete"] is False
    assert progress["final_evaluation_partial"] is True
    assert "initial_exact_rate" not in progress
    assert "final_exact_rate" not in progress
    assert progress["distinct_semantic_examples"] == 12
    assert progress["discarded_task_groups"] == 1
    assert progress["discarded_sampled_actions"] == 12
    assert progress["discarded_post_branch_actions"] == 10
    assert progress["structural_mirror_disclosures"][0]["families"] == [
        "first",
        "safe_head",
    ]
    assert progress["validation_history"] == [
        {
            "update": 5,
            "level": 0,
            "examples": 8,
            "exact_successes": 5,
            "exact_rate": 0.625,
            "checkpoint_candidate_retained": False,
            "retention_transaction_disposition": "rollback",
            "attempted_policy_update_count": 10,
            "effective_policy_update_count": 8,
            "retained_policy_update_count": 7,
            "retention_rollback_count": 2,
        }
    ]
    assert "partial evidence" in progress["message"]
    assert "paired_test_change" not in progress


def test_research_result_progress_preserves_external_evaluation_evidence() -> None:
    progress = research_result_progress(
        {
            "workload": "revision30-post-freeze-external-adapter-evaluation",
            "external_evaluation_completed": True,
            "adapter_count": 5,
            "task_count": 9,
            "elapsed_seconds": 812.5,
            "model": {"id": "Qwen/Qwen2.5-Coder-3B-Instruct"},
            "pack": {"pack_id": "revision30-post-freeze-external-pack@1"},
        }
    )

    assert progress == {
        "phase": "complete",
        "message": (
            "External adapter evaluation completed; artifacts persisted "
            "and provider teardown confirmed."
        ),
        "external_evaluation_completed": True,
        "adapter_count": 5,
        "policy_count": 6,
        "task_count": 9,
        "evaluation_completed": 54,
        "evaluation_total": 54,
        "evaluation_split": "post-freeze external pack",
        "elapsed_seconds": 812.5,
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "pack_id": "revision30-post-freeze-external-pack@1",
    }


def test_research_result_progress_falls_back_only_for_exact_rate() -> None:
    progress = research_result_progress(
        {
            "reached_complexity_level": 1,
            "final_reward": 0.625,
            "test_examples": 12,
            "initial_by_level": {
                "1": {
                    "exact_rate": 0.25,
                }
            },
            "final_by_level": {
                "1": {
                    "exact_rate": None,
                    "checkpoint_rate": None,
                    "checkpoint_rate_95ci": [0.1, 0.9],
                }
            },
        }
    )

    assert progress["exact_rate"] == 0.625
    assert progress["initial_level_exact_rate"] == 0.25
    assert progress["exact_rate_source"] == "aggregate_test_mean"
    assert progress["evaluation_split"] == "test"
    assert "evaluation_examples" not in progress
    assert "exact_rate_95ci" not in progress
    assert "checkpoint_rate_95ci" not in progress
    assert "checkpoint_rate" not in progress
    assert "checkpoint_rate_source" not in progress


def test_research_result_progress_does_not_invent_a_test_split_for_ladder_reward() -> None:
    progress = research_result_progress({"final_reward": 0.625})

    assert progress["exact_rate"] == 0.625
    assert progress["exact_rate_source"] == "final_evaluation_reward"
    assert "evaluation_split" not in progress
    assert "evaluation_examples" not in progress


def test_research_result_progress_omits_rate_provenance_without_a_rate() -> None:
    progress = research_result_progress(
        {
            "reached_complexity_level": 1,
            "final_by_level": {"1": {"exact_rate": None}},
        }
    )

    assert "exact_rate" not in progress
    assert "exact_rate_source" not in progress
    assert "checkpoint_rate_source" not in progress
    assert "evaluation_split" not in progress


def test_research_result_progress_suppresses_partial_aggregate_fallbacks() -> None:
    progress = research_result_progress(
        {
            "reached_complexity_level": 1,
            "final_evaluation_partial": True,
            "initial_reward": 0.5,
            "final_reward": 1.0,
            "reward_gain": 0.5,
            "paired_test_change": {
                "examples": 4,
                "improved": 4,
                "regressed": 0,
            },
            "checkpoint_rate": 1.0,
            "final_by_level": {
                "1": {
                    "exact_rate": None,
                    "checkpoint_rate": None,
                }
            },
        }
    )

    assert progress["final_evaluation_partial"] is True
    assert "exact_rate" not in progress
    assert "exact_rate_source" not in progress
    assert "checkpoint_rate" not in progress
    assert "checkpoint_rate_source" not in progress
    assert "evaluation_split" not in progress
    assert "initial_exact_rate" not in progress
    assert "final_exact_rate" not in progress
    assert "reward_gain" not in progress
    assert "paired_test_change" not in progress


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
            "latest_branch_snapshot": {
                "snapshot_id": "update-40-must-not-replace-full-history",
                "siblings": [{"index": index} for index in range(4)],
            },
            "initial_by_level": {"0": {"exact_rate": 0.4}},
            "final_by_level": {"0": {"exact_rate": 0.9}},
            "policy_update_count": 7,
            "attempted_policy_update_count": 7,
            "effective_policy_update_count": 5,
            "retained_policy_update_count": 4,
            "retained_checkpoint_update": 40,
            "retention_rollback_count": 2,
            "retention_transaction_revision": "adapter-optimizer-policy-lineage@1",
            "pending_optimizer_input_group_count": 2,
            "pending_optimizer_input_group_ids": ["group-pending", "anchor-pending"],
            "branch_evidence_complete": True,
            "branch_evidence_group_count": 1,
            "branch_evidence_limit": 1024,
            "branch_evidence_payload_bytes": 23884,
            "branch_evidence_payload_limit_bytes": 16777216,
            "total_sampled_completion_tokens": 128,
            "discarded_sampled_completion_tokens": 8,
            "policy_update_lineage": [
                {
                    "attempted_policy_update_index": 1,
                    "update": 20,
                    "policy_signal_group_ids": ["group-a", "group-b"],
                    "retention_lineage_status": "retained",
                },
                "invalid",
            ],
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
    assert trajectory["attempted_policy_update_count"] == 7
    assert trajectory["effective_policy_update_count"] == 5
    assert trajectory["retained_policy_update_count"] == 4
    assert trajectory["retained_checkpoint_update"] == 40
    assert trajectory["retention_rollback_count"] == 2
    assert trajectory["retention_transaction_revision"] == "adapter-optimizer-policy-lineage@1"
    assert trajectory["pending_optimizer_input_group_count"] == 2
    assert trajectory["pending_optimizer_input_group_ids"] == [
        "group-pending",
        "anchor-pending",
    ]
    assert trajectory["branch_evidence_complete"] is True
    assert trajectory["branch_evidence_group_count"] == 1
    assert trajectory["branch_evidence_limit"] == 1024
    assert trajectory["branch_evidence_payload_bytes"] == 23884
    assert trajectory["branch_evidence_payload_limit_bytes"] == 16777216
    assert trajectory["total_sampled_completion_tokens"] == 128
    assert trajectory["discarded_sampled_completion_tokens"] == 8
    assert trajectory["policy_update_lineage"] == [
        {
            "attempted_policy_update_index": 1,
            "update": 20,
            "policy_signal_group_ids": ["group-a", "group-b"],
            "retention_lineage_status": "retained",
        }
    ]


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


def test_research_trajectory_never_retruncates_terminal_branch_evidence() -> None:
    snapshots = [
        {
            "snapshot_id": f"update-{index // 4 + 1}-snapshot-{index}",
            "task_id": f"group-{index}",
            "siblings": [{"index": sibling} for sibling in range(4)],
        }
        for index in range(64)
    ]
    trajectory = research_trajectory(
        {
            "schema_version": 2,
            "branch_width": 4,
            "complexity_strategy": "adaptive",
            "multi_step": True,
            "branch_evidence_complete": True,
            "branch_evidence_group_count": len(snapshots),
            "branch_snapshots": snapshots,
            "policy_update_lineage": [
                {
                    "attempted_policy_update_index": 1,
                    "policy_signal_group_ids": ["group-0", "group-63"],
                    "branch_snapshot_ids": [
                        "update-1-snapshot-0",
                        "update-16-snapshot-63",
                    ],
                }
            ],
        }
    )

    assert len(trajectory["branch_snapshots"]) == 64
    assert trajectory["branch_snapshots"][0]["task_id"] == "group-0"
    assert trajectory["branch_snapshots"][-1]["task_id"] == "group-63"
    assert trajectory["policy_update_lineage"][0]["policy_signal_group_ids"] == [
        "group-0",
        "group-63",
    ]


@pytest.mark.parametrize("branch_width", (1, 4))
def test_live_trajectory_inherits_static_execution_contract_before_remote_progress(
    branch_width: int,
) -> None:
    source = research_trajectory_source(
        {
            "branch_width": branch_width,
            "complexity_strategy": "adaptive",
            "progress": {
                "phase": "requesting_capacity",
                "message": "Requesting one bounded RunPod worker.",
            },
        },
        None,
    )

    assert source is not None
    trajectory = research_trajectory(source)
    assert trajectory["branch_width"] == branch_width
    assert trajectory["complexity_strategy"] == "adaptive"
    assert trajectory["branch_snapshots"] == []
