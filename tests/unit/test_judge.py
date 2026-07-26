import json

import pytest

from services.execution.app.judge import MockJudgeProvider, RetryableJudgeError, blinded_order


@pytest.mark.parametrize(
    ("scenario", "outcome"),
    [
        ("valid", "SUCCEEDED"),
        ("low", "SUCCEEDED"),
        ("tie", "SUCCEEDED"),
        ("abstain", "ABSTAINED"),
        ("disagreement", "DISAGREEMENT"),
        ("integrity", "INTEGRITY_VIOLATION"),
    ],
)
def test_mock_judge_fixture_outcomes(scenario: str, outcome: str) -> None:
    response = MockJudgeProvider().invoke(
        {"fixture_scenario": scenario, "deterministic_progress": 0.8, "evidence": []},
        attempt_number=1,
    )
    result = json.loads(response.raw)
    assert result["outcome"] == outcome
    assert bool(result["assessments"]) is (outcome == "SUCCEEDED")
    assert response.model_identity.startswith("mock-judge-")
    assert response.usage["cost"] == 0


def test_retry_and_malformed_provider_paths() -> None:
    provider = MockJudgeProvider()
    with pytest.raises(RetryableJudgeError):
        provider.invoke({"fixture_scenario": "retry"}, attempt_number=1)
    assert (
        json.loads(provider.invoke({"fixture_scenario": "retry"}, attempt_number=2).raw)["outcome"]
        == "SUCCEEDED"
    )
    assert provider.invoke({"fixture_scenario": "malformed"}, attempt_number=1).raw.endswith("[")


def test_blinded_order_is_stable_and_not_identity_order() -> None:
    first = blinded_order("branch-group-fixture", 4)
    assert first == blinded_order("branch-group-fixture", 4)
    assert sorted(first) == [0, 1, 2, 3]
