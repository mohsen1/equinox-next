import pytest
from equinox_core.contracts import ContractValidationError, validate_contract


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
