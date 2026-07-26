import pytest
from equinox_core import canonical_digest
from equinox_core.contracts import (
    ContractValidationError,
    validate_contract,
    validate_judge_result,
)

from services.execution.app.judge import pointwise_spec


def test_operation_contract_rejects_missing_digest() -> None:
    with pytest.raises(ContractValidationError):
        validate_contract(
            "OperationEnvelope",
            {
                "operation_id": "op_fixture",
                "idempotency_key": "fixture",
                "expected_version": 0,
                "correlation_id": "run_fixture",
            },
        )


def _artifact(role: str, marker: str) -> dict[str, object]:
    return {
        "artifact_id": f"art_{marker}",
        "digest": "sha256:" + marker * 64,
        "role": role,
        "ordinal": 0,
        "media_type": "application/json",
        "viewer_hint": None,
        "visibility": "OPERATOR",
        "trust_class": "PLATFORM_DERIVED",
    }


def test_judge_result_semantics_reject_contradictory_success() -> None:
    spec = pointwise_spec()
    proof_digest = "sha256:" + "f" * 64
    result = {
        "judge_result_id": "judge_result_fixture",
        "judge_spec_id": spec["judge_spec_id"],
        "proof_bundle_digest": proof_digest,
        "outcome": "SUCCEEDED",
        "assessments": [
            {
                "criterion": criterion,
                "score": 0.5,
                "confidence": 0.9,
                "evidence_roles": ["candidate-render"],
            }
            for criterion in spec["criteria"]
        ],
        "preferences": [],
        "ranking": [],
        "tie": False,
        "confidence": 0.9,
        "abstained": True,
        "abstention_reason": "contradiction",
        "disagreement": True,
        "integrity_flags": ["contradiction"],
        "explanation": "invalid",
        "provider_model_identity": "deterministic-judge-fixture",
        "raw_output_artifact": _artifact("judge-raw-output", "a"),
        "parsed_output_artifact": _artifact("judge-parsed-output", "b"),
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "images": 1,
            "latency_ms": 1,
            "cost": 0,
        },
    }
    with pytest.raises(ContractValidationError, match="SUCCEEDED"):
        validate_judge_result(result, spec=spec, proof_digest=proof_digest)


def test_iteration_input_digest_excludes_occurrence_identity() -> None:
    content = {
        "collection_closure_digest": "sha256:" + "a" * 64,
        "dataset_digest": "sha256:" + "b" * 64,
        "dataset_row_count": 1,
        "materializer_version": "branch-jsonl@1",
        "weights": {"sha256:" + "c" * 64: 1.0},
    }
    manifest = {
        "manifest_id": "iteration_input_first",
        "digest": canonical_digest(content),
        "collection_closure_id": "closure_occurrence",
        **content,
        "dataset_artifact_id": "art_dataset",
        "rollout_tree_ids": ["tree_occurrence"],
        "proof_bundle_ids": ["proof_occurrence"],
        "eligibility_decision_ids": ["eligibility_occurrence"],
        "verification_run_ids": ["verification_occurrence"],
        "judge_result_ids": ["judge_occurrence"],
        "reward_signal_ids": ["reward_occurrence"],
    }
    validate_contract("IterationInput", manifest)
    validate_contract(
        "IterationInput",
        {
            **manifest,
            "manifest_id": "iteration_input_second",
        },
    )


def test_iteration_input_rejects_dataset_content_mutation() -> None:
    digest = "sha256:" + "d" * 64
    manifest = {
        "manifest_id": "iteration_input",
        "digest": digest,
        "collection_closure_id": "closure",
        "collection_closure_digest": "sha256:" + "a" * 64,
        "dataset_artifact_id": "art_dataset",
        "dataset_digest": "sha256:" + "b" * 64,
        "dataset_row_count": 1,
        "rollout_tree_ids": ["tree"],
        "proof_bundle_ids": [],
        "eligibility_decision_ids": ["eligibility"],
        "verification_run_ids": ["verification"],
        "judge_result_ids": ["judge"],
        "reward_signal_ids": ["reward"],
        "materializer_version": "branch-jsonl@1",
        "weights": {"sha256:" + "c" * 64: 1.0},
    }
    with pytest.raises(ContractValidationError, match="does not bind"):
        validate_contract("IterationInput", manifest)
