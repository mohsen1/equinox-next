from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class CanonicalizationError(ValueError):
    """Raised when a value cannot participate in an Equinox scientific digest."""


def _normalize(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise CanonicalizationError(
                f"{path}: integer exceeds the interoperable JSON safe range"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"{path}: non-finite numbers are not canonical JSON")
        return 0 if value == 0 else value
    if isinstance(value, list | tuple):
        return [_normalize(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"{path}: object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise CanonicalizationError(
                    f"{path}: object keys collide after Unicode normalization"
                )
            normalized[normalized_key] = _normalize(item, path=f"{path}.{normalized_key}")
        return normalized
    raise CanonicalizationError(f"{path}: unsupported canonical value {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def content_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
