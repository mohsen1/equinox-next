from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


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
    if not errors:
        return
    error: ValidationError = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ContractValidationError(f"{definition}.{location}: {error.message}")
