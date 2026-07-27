from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .canonical import canonical_digest


class ContractValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    configured = os.getenv("EQUINOX_CONTRACT_PATH")
    path = (
        Path(configured) if configured else Path(__file__).parents[2] / "schemas/contracts.v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def _validator(definition: str) -> Draft202012Validator:
    schema = _schema()
    if definition not in schema["$defs"]:
        raise ContractValidationError(f"unknown contract: {definition}")
    return Draft202012Validator(
        {"$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]},
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def validate_contract(definition: str, value: dict[str, Any]) -> None:
    errors = sorted(_validator(definition).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        error: ValidationError = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ContractValidationError(f"{definition}.{location}: {error.message}")
    if definition == "ProofBundle":
        validate_proof_bundle(value)
    elif definition == "JudgeResult":
        _validate_judge_outcome(value)
    elif definition == "IterationInput":
        _validate_iteration_input(value)


def _validate_iteration_input(value: dict[str, Any]) -> None:
    content = {
        "collection_closure_digest": value["collection_closure_digest"],
        "dataset_digest": value["dataset_digest"],
        "dataset_row_count": value["dataset_row_count"],
        "materializer_version": value["materializer_version"],
        "weights": value["weights"],
    }
    if value["digest"] != canonical_digest(content):
        raise ContractValidationError(
            "IterationInput.digest: does not bind the canonical materialization content"
        )
    if len(value["weights"]) != value["dataset_row_count"]:
        raise ContractValidationError(
            "IterationInput.weights: must contain one weight per dataset row"
        )


def validate_proof_bundle(value: dict[str, Any]) -> None:
    expected_roles = (
        ("task_reference_metadata", "task-reference-metadata"),
        ("task_reference", "task-reference"),
        ("source_render", "source-render"),
        ("candidate_render", "candidate-render"),
        ("action_summary", "action-summary"),
        ("geometry_report", "geometry-report"),
    )
    digests: list[str] = []
    for field, role in expected_roles:
        artifact = value[field]
        if artifact["role"] != role:
            raise ContractValidationError(
                f"ProofBundle.{field}.role: expected {role!r}, received {artifact['role']!r}"
            )
        if artifact.get("ordinal", 0) != 0:
            raise ContractValidationError(f"ProofBundle.{field}.ordinal: expected 0")
        digests.append(artifact["digest"])
    if value["ordered_artifact_digests"] != digests:
        raise ContractValidationError(
            "ProofBundle.ordered_artifact_digests: must exactly match the declared evidence roles"
        )
    content = {key: item for key, item in value.items() if key not in {"proof_bundle_id", "digest"}}
    expected_digest = canonical_digest(content)
    if value["digest"] != expected_digest:
        raise ContractValidationError(
            "ProofBundle.digest: does not bind the canonical proof content"
        )


def _validate_judge_outcome(value: dict[str, Any]) -> None:
    outcome = value["outcome"]
    assessments = value["assessments"]
    ranking = value.get("ranking", [])
    abstained = value["abstained"]
    disagreement = value["disagreement"]
    integrity_flags = value["integrity_flags"]
    reason = value.get("abstention_reason")

    if outcome == "SUCCEEDED":
        if abstained or disagreement or integrity_flags:
            raise ContractValidationError(
                "JudgeResult.outcome: SUCCEEDED cannot carry abstention, disagreement, "
                "or integrity state"
            )
        if not assessments and not ranking:
            raise ContractValidationError(
                "JudgeResult.assessments: a successful result must contain an assessment or ranking"
            )
    elif outcome == "ABSTAINED":
        if not abstained or not reason or assessments or ranking:
            raise ContractValidationError(
                "JudgeResult.outcome: ABSTAINED requires a reason and no scored result"
            )
    elif outcome == "DISAGREEMENT":
        if not disagreement or abstained or assessments or ranking:
            raise ContractValidationError(
                "JudgeResult.outcome: DISAGREEMENT requires disagreement and no scored result"
            )
    elif outcome == "INTEGRITY_VIOLATION":
        if not integrity_flags or abstained or assessments or ranking:
            raise ContractValidationError(
                "JudgeResult.outcome: INTEGRITY_VIOLATION requires integrity flags "
                "and no scored result"
            )
    elif outcome in {"INVALID_RESULT", "INFRA_FAILED"} and (assessments or ranking):
        raise ContractValidationError(
            f"JudgeResult.outcome: {outcome} cannot contain a scored result"
        )


def validate_judge_result(
    value: dict[str, Any],
    *,
    spec: dict[str, Any],
    proof_digest: str,
    group_labels: set[str] | None = None,
) -> None:
    validate_contract("JudgeResult", value)
    if value["judge_spec_id"] != spec["judge_spec_id"]:
        raise ContractValidationError("JudgeResult.judge_spec_id: does not match JudgeSpec")
    if value["proof_bundle_digest"] != proof_digest:
        raise ContractValidationError("JudgeResult.proof_bundle_digest: does not match proof")
    if value["outcome"] != "SUCCEEDED":
        return
    criteria = [assessment["criterion"] for assessment in value["assessments"]]
    if len(criteria) != len(set(criteria)) or set(criteria) != set(spec["criteria"]):
        raise ContractValidationError(
            "JudgeResult.assessments: criteria must exactly match JudgeSpec"
        )
    allowed_roles = set(spec["allowed_evidence_roles"])
    if any(
        not set(assessment["evidence_roles"]).issubset(allowed_roles)
        for assessment in value["assessments"]
    ):
        raise ContractValidationError(
            "JudgeResult.assessments: cites an evidence role outside JudgeSpec"
        )
    if group_labels is not None:
        ranking = value.get("ranking", [])
        if len(ranking) != len(set(ranking)) or set(ranking) != group_labels:
            raise ContractValidationError(
                "JudgeResult.ranking: must be a permutation of the blinded labels"
            )
