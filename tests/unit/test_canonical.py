from __future__ import annotations

import math

import pytest
from equinox_core import CanonicalizationError, canonical_bytes, canonical_digest


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_bytes({"value": value})


def test_canonical_json_normalizes_unicode_and_negative_zero() -> None:
    composed = {"caf\u00e9": -0.0}
    decomposed = {"cafe\u0301": 0}

    assert canonical_bytes(composed) == b'{"caf\xc3\xa9":0}'
    assert canonical_bytes(composed) == canonical_bytes(decomposed)
    assert canonical_digest(composed) == canonical_digest(decomposed)


def test_canonical_json_rejects_ambiguous_or_nonportable_objects() -> None:
    with pytest.raises(CanonicalizationError, match="keys must be strings"):
        canonical_bytes({1: "value"})
    with pytest.raises(CanonicalizationError, match="safe range"):
        canonical_bytes({"value": 2**53})
    with pytest.raises(CanonicalizationError, match="unsupported"):
        canonical_bytes({"value": object()})


def test_canonical_json_preserves_array_order_and_sorts_object_keys() -> None:
    assert canonical_bytes({"z": [2, 1], "a": True}) == b'{"a":true,"z":[2,1]}'
