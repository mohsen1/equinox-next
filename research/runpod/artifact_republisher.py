"""Replay one committed RunPod pilot proof without touching provider state."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.equinox_core.canonical import (
    CanonicalizationError,
    canonical_bytes,
    canonical_digest,
)
from research.runpod.artifact_publication import (
    ATOMIC_PUBLICATION_PROFILE_ID,
    ArtifactPublicationError,
    PublicationResult,
    build_artifact_publication_envelope,
    build_committed_proof_replay_payload,
    load_committed_artifact_set,
    load_committed_provider_receipt,
)

MAXIMUM_HTTP_RESPONSE_BYTES = 1024 * 1024
MAXIMUM_ATTEMPTS = 3
RECOVERABLE_FAILURE_MESSAGES = frozenset(
    {
        "The remote result did not satisfy the declared proof contract.",
        "A verified local proof receipt is pending ingestion.",
    }
)
RECOVERABLE_SCREEN_FAILURE_MESSAGE = (
    "The completed larger-model eligibility screen could not be published."
)
REGISTERED_EXECUTION_RECEIPT_IDENTITY_KEYS = (
    "gpu_id",
    "gpu_count",
    "image",
    "image_digest",
    "hourly_cost_usd",
    "cloud_type",
    "data_center_ids",
    "persistent_volume_in_gb",
    "network_volume_id",
    "network_volume_data_center_id",
    "profile_id",
    "manifest_digest",
    "source_head_commit",
    "source_contract_digest",
    "volume_readiness_receipt_digest",
    "bundle_stage_receipt_digest",
    "torch_retention_evidence_digest",
    "dependency_quarantine_revision",
    "dependency_lock_digest",
    "dependency_quarantine_evidence_digest",
    "dependency_private_tree_digest",
    "code_materialization_revision",
    "code_materialization_evidence_digest",
    "code_private_tree_digest",
    "bootstrap_source_digest",
    "bundle_handoff_revision",
    "live_stage_activation_revision",
    "live_stage_activation_digest",
    "workload_bundle_digest",
    "workload_bundle_size_bytes",
    "workload_bundle_compression",
    "workload_bundle_path",
)


class ProofReplayHttpError(ArtifactPublicationError):
    """A bounded HTTP request failed with enough detail to classify retryability."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


JsonRequester = Callable[
    [str, str, bytes | None, str | None, float],
    Mapping[str, Any],
]
Sleeper = Callable[[float], None]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _api_root(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    try:
        host = parsed.hostname
        _port = parsed.port
    except ValueError as error:
        raise ArtifactPublicationError("proof replay API root is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or host is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactPublicationError("proof replay API root is invalid")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ArtifactPublicationError(
                "proof replay requires HTTPS for a non-loopback API root"
            )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _api_url(api_root: str, path: str) -> str:
    return _api_root(api_root) + path


def _decode_object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProofReplayHttpError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProofReplayHttpError(f"{name} must be one JSON object")
    return value


def _http_json_request(
    method: str,
    url: str,
    payload: bytes | None,
    internal_token: str | None,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if internal_token is not None:
        headers["Authorization"] = f"Bearer {internal_token}"
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers=headers,
    )
    opener = urllib.request.build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status = response.status
            response_payload = response.read(MAXIMUM_HTTP_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ProofReplayHttpError(
            f"proof replay API returned HTTP {error.code}{suffix}",
            status=error.code,
            retryable=(error.code in {408, 429} or error.code >= 500),
        ) from error
    except (TimeoutError, urllib.error.URLError) as error:
        raise ProofReplayHttpError(
            "proof replay API is unavailable",
            retryable=True,
        ) from error
    expected_statuses = {200} if method == "GET" else {200, 201}
    if status not in expected_statuses or len(response_payload) > MAXIMUM_HTTP_RESPONSE_BYTES:
        raise ProofReplayHttpError(
            "proof replay API returned an invalid response",
            status=status,
        )
    return _decode_object(response_payload, "proof replay API response")


def _request_with_retry(
    requester: JsonRequester,
    method: str,
    url: str,
    payload: bytes | None,
    internal_token: str | None,
    timeout_seconds: float,
    sleeper: Sleeper,
) -> Mapping[str, Any]:
    for attempt in range(1, MAXIMUM_ATTEMPTS + 1):
        try:
            return requester(
                method,
                url,
                payload,
                internal_token,
                timeout_seconds,
            )
        except ProofReplayHttpError as error:
            if not error.retryable or attempt == MAXIMUM_ATTEMPTS:
                raise
            sleeper(0.5 * attempt)
    raise AssertionError("bounded proof replay loop did not terminate")


def _failure_message(execution: Mapping[str, Any]) -> str | None:
    progress = execution.get("progress")
    if not isinstance(progress, Mapping):
        return None
    operator_error = progress.get("operator_error")
    if isinstance(operator_error, Mapping):
        message = operator_error.get("message")
        return message if isinstance(message, str) else None
    error = progress.get("error")
    return error if isinstance(error, str) else None


def _same_timestamp(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        left_timestamp = datetime.fromisoformat(left.replace("Z", "+00:00"))
        right_timestamp = datetime.fromisoformat(right.replace("Z", "+00:00"))
    except ValueError:
        return False
    if left_timestamp.utcoffset() is None or right_timestamp.utcoffset() is None:
        return False
    return left_timestamp.astimezone(UTC) == right_timestamp.astimezone(UTC)


def _resource_profile_identity_matches(
    registered: Any,
    receipt: Any,
) -> bool:
    if not isinstance(registered, Mapping) or not isinstance(receipt, Mapping):
        return False
    return all(
        key in registered and key in receipt and registered[key] == receipt[key]
        for key in REGISTERED_EXECUTION_RECEIPT_IDENTITY_KEYS
    )


def _is_pristine_running_execution(execution: Mapping[str, Any]) -> bool:
    """Return whether RUNNING is an unfinalized observer row with no terminal evidence."""

    return (
        execution.get("status") == "RUNNING"
        and execution.get("completed_at") is None
        and execution.get("teardown_confirmed") is False
        and execution.get("artifact_publication") is None
        and execution.get("proof_id") is None
        and execution.get("receipt_digest") is None
        and execution.get("failure_receipt_digest") is None
    )


def _finalizing_execution_payload(
    execution: Mapping[str, Any],
    publication: PublicationResult,
) -> dict[str, Any]:
    """Advance one exact pristine RUNNING row before replaying terminal evidence."""

    progress = execution.get("progress")
    resource_profile = execution.get("resource_profile")
    required_strings = (
        "name",
        "workload_id",
        "complexity_strategy",
        "provider_name",
        "provider_handle",
        "started_at",
    )
    if (
        not _is_pristine_running_execution(execution)
        or not isinstance(progress, Mapping)
        or not isinstance(resource_profile, Mapping)
        or any(not isinstance(execution.get(field), str) for field in required_strings)
        or not isinstance(execution.get("branch_width"), int)
        or isinstance(execution.get("branch_width"), bool)
    ):
        raise ArtifactPublicationError(
            "registered RUNNING execution is not safe for committed-set recovery"
        )
    finalizing_progress = dict(progress)
    for field in ("error", "remote_error", "operator_error"):
        finalizing_progress.pop(field, None)
    finalizing_progress.update(
        {
            "phase": "finalizing",
            "message": "Committed artifacts verified; replaying finalization.",
            "profile_id": publication.profile_id,
        }
    )
    return {
        "name": execution["name"],
        "workload_id": execution["workload_id"],
        "model_id": execution.get("model_id"),
        "branch_width": execution["branch_width"],
        "complexity_strategy": execution["complexity_strategy"],
        "status": "FINALIZING",
        "provider_name": execution["provider_name"],
        "provider_handle": execution["provider_handle"],
        "resource_profile": dict(resource_profile),
        "progress": finalizing_progress,
        "artifact_publication_required": True,
        "artifact_publication": build_artifact_publication_envelope(publication),
        "failure_receipt_digest": None,
        "started_at": execution["started_at"],
        "completed_at": None,
        "teardown_confirmed": False,
    }


def _validate_finalizing_advance(
    execution: Mapping[str, Any],
    publication: PublicationResult,
) -> None:
    if (
        execution.get("execution_id") != publication.publication_id
        or execution.get("status") != "FINALIZING"
        or execution.get("completed_at") is not None
        or execution.get("teardown_confirmed") is not False
        or execution.get("artifact_publication") != build_artifact_publication_envelope(publication)
        or execution.get("proof_id") is not None
        or execution.get("receipt_digest") is not None
        or execution.get("failure_receipt_digest") is not None
    ):
        raise ArtifactPublicationError(
            "RUNNING recovery did not persist the exact FINALIZING evidence"
        )


def _advance_running_execution(
    execution: Mapping[str, Any],
    publication: PublicationResult,
    *,
    api_root: str,
    internal_token: str,
    timeout_seconds: float,
    requester: JsonRequester,
    sleeper: Sleeper,
) -> Mapping[str, Any]:
    try:
        request_payload = canonical_bytes(_finalizing_execution_payload(execution, publication))
    except CanonicalizationError as error:
        raise ArtifactPublicationError(
            "committed FINALIZING recovery request is not canonical JSON"
        ) from error
    updated = _request_with_retry(
        requester,
        "PUT",
        api_root
        + "/internal/research-compute-executions/"
        + urllib.parse.quote(publication.publication_id, safe=""),
        request_payload,
        internal_token,
        timeout_seconds,
        sleeper,
    )
    _validate_finalizing_advance(updated, publication)
    return updated


def _validate_registered_execution(
    execution: Mapping[str, Any],
    publication: PublicationResult,
    proof_payload: Mapping[str, Any],
) -> None:
    envelope = proof_payload["artifact_publication"]
    resource_profile = execution.get("resource_profile")
    progress = execution.get("progress")
    workload = proof_payload["workload"]
    result = proof_payload["result"]
    expected_model_id = workload.get("model_id") or result.get("model_id")
    if (
        execution.get("execution_id") != publication.publication_id
        or execution.get("provider_name") != "RunPod"
        or execution.get("provider_handle") != proof_payload["provider_handle"]
        or execution.get("workload_id") != workload.get("id")
        or workload.get("id") != result.get("workload")
        or execution.get("model_id") != expected_model_id
        or not _same_timestamp(
            execution.get("started_at"),
            proof_payload["started_at"],
        )
        or execution.get("artifact_publication_required") is not True
        or execution.get("branch_width") != workload.get("static_branch_width")
        or execution.get("complexity_strategy") != "adaptive"
        or not isinstance(resource_profile, Mapping)
        or resource_profile.get("profile_id") != ATOMIC_PUBLICATION_PROFILE_ID
        or not _resource_profile_identity_matches(
            resource_profile,
            proof_payload.get("resource_profile"),
        )
        or (
            isinstance(progress, Mapping)
            and isinstance(progress.get("profile_id"), str)
            and progress["profile_id"] != ATOMIC_PUBLICATION_PROFILE_ID
        )
        or execution.get("artifact_publication") not in (None, envelope)
    ):
        raise ArtifactPublicationError(
            "registered execution does not match the committed pilot proof"
        )
    status = execution.get("status")
    if status == "FAILED":
        if (
            execution.get("teardown_confirmed") is not True
            or _failure_message(execution) not in RECOVERABLE_FAILURE_MESSAGES
        ):
            raise ArtifactPublicationError("registered execution failure is not proof-recoverable")
    elif status == "SUCCEEDED":
        if (
            execution.get("teardown_confirmed") is not True
            or execution.get("artifact_publication") != envelope
            or not isinstance(execution.get("proof_id"), str)
        ):
            raise ArtifactPublicationError(
                "registered successful execution has different proof evidence"
            )
    elif status == "RUNNING":
        if not _is_pristine_running_execution(execution):
            raise ArtifactPublicationError(
                "registered RUNNING execution is not safe for proof replay"
            )
    elif status != "FINALIZING":
        raise ArtifactPublicationError("registered execution is not ready for proof replay")


def _validate_source_screen_execution(
    source_execution: Mapping[str, Any],
    source_publication: PublicationResult,
) -> None:
    source_envelope = build_artifact_publication_envelope(source_publication)
    resource_profile = source_execution.get("resource_profile")
    if (
        source_execution.get("execution_id") != source_publication.publication_id
        or source_execution.get("status") != "SUCCEEDED"
        or source_execution.get("teardown_confirmed") is not True
        or source_execution.get("artifact_publication_required") is not True
        or source_execution.get("artifact_publication") != source_envelope
        or not isinstance(resource_profile, Mapping)
        or resource_profile.get("profile_id") != ATOMIC_PUBLICATION_PROFILE_ID
    ):
        raise ArtifactPublicationError(
            "registered source screen does not match the committed source_screen"
        )


def _validate_registered_screen_execution(
    execution: Mapping[str, Any],
    publication: PublicationResult,
    provider_receipt: Mapping[str, Any],
) -> None:
    envelope = build_artifact_publication_envelope(publication)
    resource_profile = execution.get("resource_profile")
    receipt_resource_profile = provider_receipt.get("resource_profile")
    progress = execution.get("progress")
    workload = provider_receipt.get("workload")
    result = provider_receipt.get("result")
    if (
        not isinstance(resource_profile, Mapping)
        or not isinstance(receipt_resource_profile, Mapping)
        or not isinstance(progress, Mapping)
        or not isinstance(workload, Mapping)
        or not isinstance(result, Mapping)
    ):
        raise ArtifactPublicationError("registered screen execution has incomplete lineage")
    expected_model_id = workload.get("model_id") or result.get("model_id")
    if (
        execution.get("execution_id") != publication.publication_id
        or execution.get("provider_name") != "RunPod"
        or execution.get("provider_handle") != provider_receipt.get("provider_handle")
        or execution.get("workload_id") != workload.get("id")
        or workload.get("id") != result.get("workload")
        or execution.get("model_id") != expected_model_id
        or not _same_timestamp(
            execution.get("started_at"),
            provider_receipt.get("started_at"),
        )
        or execution.get("artifact_publication_required") is not True
        or execution.get("branch_width") != workload.get("static_branch_width")
        or execution.get("complexity_strategy") != "adaptive"
        or resource_profile.get("profile_id") != ATOMIC_PUBLICATION_PROFILE_ID
        or (
            isinstance(progress.get("profile_id"), str)
            and progress["profile_id"] != ATOMIC_PUBLICATION_PROFILE_ID
        )
        or execution.get("artifact_publication") not in (None, envelope)
        or not _resource_profile_identity_matches(
            resource_profile,
            receipt_resource_profile,
        )
        or execution.get("proof_id") is not None
    ):
        raise ArtifactPublicationError("registered execution does not match the committed screen")
    status = execution.get("status")
    if status == "FAILED":
        if (
            execution.get("teardown_confirmed") is not True
            or _failure_message(execution) != RECOVERABLE_SCREEN_FAILURE_MESSAGE
        ):
            raise ArtifactPublicationError(
                "registered screen failure is not publication-recoverable"
            )
    elif status == "SUCCEEDED":
        if (
            execution.get("teardown_confirmed") is not True
            or execution.get("artifact_publication") != envelope
            or not _same_timestamp(
                execution.get("completed_at"),
                provider_receipt.get("completed_at"),
            )
        ):
            raise ArtifactPublicationError(
                "registered successful screen has different publication evidence"
            )
    elif status == "RUNNING":
        if not _is_pristine_running_execution(execution):
            raise ArtifactPublicationError(
                "registered RUNNING screen is not safe for finalization replay"
            )
    elif status != "FINALIZING":
        raise ArtifactPublicationError("registered screen is not ready for finalization replay")


def _screen_execution_payload(
    execution: Mapping[str, Any],
    publication: PublicationResult,
    provider_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    progress = dict(execution["progress"])
    for field in ("error", "remote_error", "operator_error"):
        progress.pop(field, None)
    progress.update(
        {
            "phase": "complete",
            "message": "Eligibility recorded and teardown confirmed.",
            "profile_id": publication.profile_id,
        }
    )
    return {
        "name": execution["name"],
        "workload_id": execution["workload_id"],
        "model_id": execution.get("model_id"),
        "branch_width": execution["branch_width"],
        "complexity_strategy": "adaptive",
        "status": "SUCCEEDED",
        "provider_name": "RunPod",
        "provider_handle": provider_receipt["provider_handle"],
        "resource_profile": dict(execution["resource_profile"]),
        "progress": progress,
        "artifact_publication_required": True,
        "artifact_publication": build_artifact_publication_envelope(publication),
        "failure_receipt_digest": None,
        "started_at": provider_receipt["started_at"],
        "completed_at": provider_receipt["completed_at"],
        "teardown_confirmed": True,
    }


def _validate_final_screen_execution(
    execution: Mapping[str, Any],
    *,
    publication: PublicationResult,
    provider_receipt: Mapping[str, Any],
) -> None:
    if (
        execution.get("execution_id") != publication.publication_id
        or execution.get("status") != "SUCCEEDED"
        or execution.get("provider_handle") != provider_receipt["provider_handle"]
        or execution.get("teardown_confirmed") is not True
        or execution.get("artifact_publication") != build_artifact_publication_envelope(publication)
        or not _same_timestamp(
            execution.get("completed_at"),
            provider_receipt.get("completed_at"),
        )
    ):
        raise ArtifactPublicationError(
            "screen finalization replay completed with inconsistent evidence"
        )


def _validate_ingestion_response(
    response: Mapping[str, Any],
    *,
    expected_receipt_digest: str,
) -> tuple[str, bool]:
    expected_keys = {"proof_id", "receipt_digest", "already_recorded"}
    if (
        set(response) != expected_keys
        or not isinstance(response.get("proof_id"), str)
        or not response["proof_id"]
        or response.get("receipt_digest") != expected_receipt_digest
        or type(response.get("already_recorded")) is not bool
    ):
        raise ArtifactPublicationError("proof replay API response does not match the exact request")
    return response["proof_id"], response["already_recorded"]


def _validate_post_replay_state(
    execution: Mapping[str, Any],
    proof: Mapping[str, Any],
    *,
    publication: PublicationResult,
    proof_payload: Mapping[str, Any],
    proof_id: str,
    receipt_digest: str,
) -> None:
    if (
        execution.get("execution_id") != publication.publication_id
        or execution.get("status") != "SUCCEEDED"
        or execution.get("teardown_confirmed") is not True
        or execution.get("artifact_publication") != proof_payload["artifact_publication"]
        or execution.get("proof_id") != proof_id
        or execution.get("receipt_digest") != receipt_digest
    ):
        raise ArtifactPublicationError(
            "proof replay completed but execution evidence is inconsistent"
        )
    evidence = proof.get("evidence")
    provider = proof.get("provider")
    if (
        proof.get("proof_id") != proof_id
        or proof.get("execution_id") != publication.publication_id
        or proof.get("teardown_confirmed") is not True
        or not isinstance(provider, Mapping)
        or provider.get("handle") != proof_payload["provider_handle"]
        or not isinstance(evidence, Mapping)
        or evidence.get("receipt_digest") != receipt_digest
        or evidence.get("artifact_set_committed") is not True
        or evidence.get("artifact_set_manifest_digest") != publication.artifact_set_manifest_digest
    ):
        raise ArtifactPublicationError("proof replay completed but proof evidence is inconsistent")


def dry_run_committed_proof(
    proof_directory: Path,
    publication_id: str,
) -> dict[str, Any]:
    publication, proof_payload = build_committed_proof_replay_payload(
        proof_directory,
        publication_id,
    )
    try:
        receipt_digest = canonical_digest(proof_payload)
    except CanonicalizationError as error:
        raise ArtifactPublicationError(
            "committed proof replay request is not canonical JSON"
        ) from error
    return {
        "schema_version": 2,
        "publication_id": publication.publication_id,
        "publication_type": publication.publication_type,
        "profile_id": publication.profile_id,
        "artifact_set_manifest_digest": publication.artifact_set_manifest_digest,
        "request_receipt_digest": receipt_digest,
        "dry_run": True,
        "network_attempted": False,
    }


def dry_run_committed_result(
    proof_directory: Path,
    publication_id: str,
) -> dict[str, Any]:
    publication = load_committed_artifact_set(
        proof_directory,
        publication_id,
        expected_profile_id=ATOMIC_PUBLICATION_PROFILE_ID,
    )
    if publication.publication_type == "pilot":
        return dry_run_committed_proof(proof_directory, publication_id)
    envelope = build_artifact_publication_envelope(publication)
    return {
        "schema_version": 2,
        "publication_id": publication.publication_id,
        "publication_type": publication.publication_type,
        "profile_id": publication.profile_id,
        "artifact_set_manifest_digest": publication.artifact_set_manifest_digest,
        "artifact_publication": envelope,
        "dry_run": True,
        "network_attempted": False,
        "registered_execution_required": True,
    }


def republish_committed_screen(
    proof_directory: Path,
    publication_id: str,
    *,
    api_root: str,
    internal_token: str,
    timeout_seconds: float = 30.0,
    requester: JsonRequester | None = None,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Replay one exact terminal screen PUT and verify the persisted envelope."""

    if not isinstance(internal_token, str) or len(internal_token) < 32:
        raise ArtifactPublicationError("screen replay requires EQUINOX_INTERNAL_TOKEN")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not 0 < timeout_seconds <= 300
    ):
        raise ArtifactPublicationError("screen replay timeout is invalid")
    root = _api_root(api_root)
    publication = load_committed_artifact_set(
        proof_directory,
        publication_id,
        expected_profile_id=ATOMIC_PUBLICATION_PROFILE_ID,
        expected_publication_type="screen",
    )
    provider_receipt = load_committed_provider_receipt(publication)
    request_json = requester or _http_json_request
    execution_path = "/v1/research-compute-executions/" + urllib.parse.quote(
        publication.publication_id, safe=""
    )
    execution = _request_with_retry(
        request_json,
        "GET",
        root + execution_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    _validate_registered_screen_execution(
        execution,
        publication,
        provider_receipt,
    )
    if execution.get("status") == "RUNNING":
        execution = _advance_running_execution(
            execution,
            publication,
            api_root=root,
            internal_token=internal_token,
            timeout_seconds=float(timeout_seconds),
            requester=request_json,
            sleeper=sleeper,
        )
        _validate_registered_screen_execution(
            execution,
            publication,
            provider_receipt,
        )
    put_payload = _screen_execution_payload(
        execution,
        publication,
        provider_receipt,
    )
    try:
        request_payload = canonical_bytes(put_payload)
    except CanonicalizationError as error:
        raise ArtifactPublicationError(
            "committed screen finalization request is not canonical JSON"
        ) from error
    updated = _request_with_retry(
        request_json,
        "PUT",
        root + "/internal/research-compute-executions/" + publication.publication_id,
        request_payload,
        internal_token,
        float(timeout_seconds),
        sleeper,
    )
    _validate_final_screen_execution(
        updated,
        publication=publication,
        provider_receipt=provider_receipt,
    )
    final_execution = _request_with_retry(
        request_json,
        "GET",
        root + execution_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    _validate_final_screen_execution(
        final_execution,
        publication=publication,
        provider_receipt=provider_receipt,
    )
    return {
        "schema_version": 2,
        "publication_id": publication.publication_id,
        "publication_type": publication.publication_type,
        "profile_id": publication.profile_id,
        "artifact_set_manifest_digest": publication.artifact_set_manifest_digest,
        "replayed": True,
        "screen_finalized": True,
        "provider_compute_used": False,
        "authorization_consumed": False,
    }


def republish_committed_result(
    proof_directory: Path,
    publication_id: str,
    *,
    api_root: str,
    internal_token: str,
    timeout_seconds: float = 30.0,
    requester: JsonRequester | None = None,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    publication = load_committed_artifact_set(
        proof_directory,
        publication_id,
        expected_profile_id=ATOMIC_PUBLICATION_PROFILE_ID,
    )
    if publication.publication_type == "screen":
        return republish_committed_screen(
            proof_directory,
            publication_id,
            api_root=api_root,
            internal_token=internal_token,
            timeout_seconds=timeout_seconds,
            requester=requester,
            sleeper=sleeper,
        )
    return republish_committed_proof(
        proof_directory,
        publication_id,
        api_root=api_root,
        internal_token=internal_token,
        timeout_seconds=timeout_seconds,
        requester=requester,
        sleeper=sleeper,
    )


def republish_committed_proof(
    proof_directory: Path,
    publication_id: str,
    *,
    api_root: str,
    internal_token: str,
    timeout_seconds: float = 30.0,
    requester: JsonRequester | None = None,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Replay one exact proof POST and verify its persisted API evidence."""

    if not isinstance(internal_token, str) or len(internal_token) < 32:
        raise ArtifactPublicationError("proof replay requires EQUINOX_INTERNAL_TOKEN")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not 0 < timeout_seconds <= 300
    ):
        raise ArtifactPublicationError("proof replay timeout is invalid")
    root = _api_root(api_root)
    publication, proof_payload = build_committed_proof_replay_payload(
        proof_directory,
        publication_id,
    )
    source_publication_id = publication.source_screen_publication_id
    if source_publication_id is None:
        raise ArtifactPublicationError("committed pilot has no source_screen")
    source_publication = load_committed_artifact_set(
        publication.manifest_path.parent,
        source_publication_id,
        expected_profile_id=ATOMIC_PUBLICATION_PROFILE_ID,
        expected_publication_type="screen",
    )
    if source_publication.artifact_set_manifest_digest != publication.source_screen_manifest_digest:
        raise ArtifactPublicationError("committed pilot source_screen changed during proof replay")
    request_json = requester or _http_json_request
    execution_path = "/v1/research-compute-executions/" + urllib.parse.quote(
        publication.publication_id, safe=""
    )
    source_path = "/v1/research-compute-executions/" + urllib.parse.quote(
        source_publication.publication_id, safe=""
    )
    execution = _request_with_retry(
        request_json,
        "GET",
        root + execution_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    _validate_registered_execution(execution, publication, proof_payload)
    source_execution = _request_with_retry(
        request_json,
        "GET",
        root + source_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    _validate_source_screen_execution(source_execution, source_publication)
    if execution.get("status") == "RUNNING":
        execution = _advance_running_execution(
            execution,
            publication,
            api_root=root,
            internal_token=internal_token,
            timeout_seconds=float(timeout_seconds),
            requester=request_json,
            sleeper=sleeper,
        )
        _validate_registered_execution(execution, publication, proof_payload)
    try:
        request_payload = canonical_bytes(proof_payload)
        expected_receipt_digest = canonical_digest(proof_payload)
    except CanonicalizationError as error:
        raise ArtifactPublicationError(
            "committed proof replay request is not canonical JSON"
        ) from error
    response = _request_with_retry(
        request_json,
        "POST",
        root + "/internal/research-compute-proofs",
        request_payload,
        internal_token,
        float(timeout_seconds),
        sleeper,
    )
    proof_id, already_recorded = _validate_ingestion_response(
        response,
        expected_receipt_digest=expected_receipt_digest,
    )
    final_execution = _request_with_retry(
        request_json,
        "GET",
        root + execution_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    proof_path = "/v1/proofs/" + urllib.parse.quote(proof_id, safe="")
    proof = _request_with_retry(
        request_json,
        "GET",
        root + proof_path,
        None,
        None,
        float(timeout_seconds),
        sleeper,
    )
    _validate_post_replay_state(
        final_execution,
        proof,
        publication=publication,
        proof_payload=proof_payload,
        proof_id=proof_id,
        receipt_digest=expected_receipt_digest,
    )
    return {
        "schema_version": 2,
        "publication_id": publication.publication_id,
        "publication_type": publication.publication_type,
        "profile_id": publication.profile_id,
        "artifact_set_manifest_digest": publication.artifact_set_manifest_digest,
        "proof_id": proof_id,
        "receipt_digest": expected_receipt_digest,
        "already_recorded": already_recorded,
        "replayed": True,
        "provider_compute_used": False,
        "authorization_consumed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a verified committed RunPod pilot proof without allocating "
            "provider compute or consuming pilot authorization."
        )
    )
    parser.add_argument("--proof-directory", type=Path, required=True)
    parser.add_argument("--publication-id", required=True)
    parser.add_argument(
        "--api-root",
        default=os.environ.get("EQUINOX_API_ROOT", "http://127.0.0.1:8180"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.dry_run:
            result = dry_run_committed_result(
                args.proof_directory,
                args.publication_id,
            )
        else:
            result = republish_committed_result(
                args.proof_directory,
                args.publication_id,
                api_root=args.api_root,
                internal_token=os.environ.get("EQUINOX_INTERNAL_TOKEN", ""),
                timeout_seconds=args.timeout_seconds,
            )
        payload = canonical_bytes(result) + b"\n"
    except (ArtifactPublicationError, CanonicalizationError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
