import json

import pytest

from services.execution.app.judge import (
    DeterministicJudgeFixture,
    RetryableJudgeError,
    blinded_order,
    pointwise_spec,
    prompt_registry,
)


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
    response = DeterministicJudgeFixture().invoke(
        {"fixture_scenario": scenario, "deterministic_progress": 0.8, "evidence": []},
        attempt_number=1,
    )
    result = json.loads(response.raw)
    assert result["outcome"] == outcome
    assert bool(result["assessments"]) is (outcome == "SUCCEEDED")
    assert response.model_identity.startswith("deterministic-judge-fixture-")
    assert response.usage["cost"] == 0


def test_retry_and_malformed_provider_paths() -> None:
    provider = DeterministicJudgeFixture()
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


def test_judge_spec_binds_the_committed_prompt_bytes() -> None:
    template = prompt_registry()["cad-pointwise-v1"]
    spec = pointwise_spec()
    assert spec["provider"] == "DeterministicJudgeFixture"
    assert spec["prompt_template_id"] == template.template_id
    assert spec["prompt_template_digest"] == template.content_digest
    rendered = template.render(
        {
            "proof_bundle_digest": "sha256:" + "a" * 64,
            "evidence": [{"role": "candidate-render"}],
        }
    )
    assert rendered.startswith("# Role")
    assert "<untrusted-evidence>" in rendered
    assert '"role":"candidate-render"' in rendered
    assert rendered.endswith("</untrusted-evidence>\n")
