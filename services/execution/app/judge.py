from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from equinox_core import canonical_digest

CRITERIA = [
    "reference_correspondence",
    "silhouette_and_proportion",
    "feature_presence_and_placement",
    "geometric_coherence",
    "progress_from_source_state",
    "regressions",
]


class RetryableJudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResponse:
    raw: str
    model_identity: str
    usage: dict[str, Any]


class MockJudgeProvider:
    name = "MockJudgeProvider"
    model_identity = "mock-judge-deterministic-2026-07-26"

    def invoke(self, request: dict[str, Any], *, attempt_number: int) -> ProviderResponse:
        scenario = request.get("fixture_scenario", "valid")
        if scenario == "retry" and attempt_number == 1:
            raise RetryableJudgeError("fixture provider overload")
        if scenario == "malformed":
            raw = '{"outcome":"SUCCEEDED","assessments":['
        else:
            raw = json.dumps(self._result(request, scenario), sort_keys=True, separators=(",", ":"))
        return ProviderResponse(
            raw=raw,
            model_identity=self.model_identity,
            usage={
                "input_tokens": 420,
                "output_tokens": max(16, len(raw) // 4),
                "images": len(request.get("evidence", [])),
                "latency_ms": 8 + attempt_number,
                "cost": 0,
            },
        )

    def _result(self, request: dict[str, Any], scenario: str) -> dict[str, Any]:
        score = float(request.get("deterministic_progress", 0.5))
        confidence = 0.92
        outcome = "SUCCEEDED"
        abstained = False
        abstention_reason = None
        disagreement = False
        integrity_flags: list[str] = []
        tie = False

        if scenario == "low":
            score = min(score, 0.18)
        elif scenario == "tie":
            tie = True
            score = 0.5
        elif scenario == "abstain":
            outcome = "ABSTAINED"
            abstained = True
            confidence = 0.42
            abstention_reason = "fixture evidence is intentionally ambiguous"
        elif scenario == "disagreement":
            outcome = "DISAGREEMENT"
            disagreement = True
            confidence = 0.55
        elif scenario == "integrity":
            outcome = "INTEGRITY_VIOLATION"
            integrity_flags = ["candidate_visual_instruction_detected"]
            confidence = 0.99

        assessments = (
            [
                {
                    "criterion": criterion,
                    "score": round(score if criterion != "regressions" else 1 - score, 4),
                    "confidence": confidence,
                    "evidence_roles": ["task-reference", "source-render", "candidate-render"],
                }
                for criterion in CRITERIA
            ]
            if outcome == "SUCCEEDED"
            else []
        )
        return {
            "outcome": outcome,
            "assessments": assessments,
            "confidence": confidence,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "disagreement": disagreement,
            "integrity_flags": integrity_flags,
            "tie": tie,
            "explanation": (
                "Mock assessment cites pinned renders and geometry evidence; "
                "it is deterministic contract evidence, not visual ground truth."
            ),
        }


def pointwise_spec(*, version: int = 1) -> dict[str, Any]:
    prompt_digest = canonical_digest(
        {
            "template": "cad-pointwise",
            "version": version,
            "privileged_instructions": True,
            "untrusted_evidence_delimiters": True,
        }
    )
    return {
        "judge_spec_id": f"cad.pointwise.mock@{version}",
        "provider": "MockJudgeProvider",
        "model_revision": "mock-judge-deterministic-2026-07-26",
        "prompt_template_digest": prompt_digest,
        "rubric_version": f"cad-progress-rubric@{version}",
        "output_schema": "judge-result.v1",
        "mode": "POINTWISE",
        "criteria": CRITERIA,
        "sampling": {"temperature": 0, "samples": 1, "sample_index": 0},
        "presentation": {"blind_identity": True, "randomize_order": False},
        "abstention": {
            "allowed": True,
            "minimum_confidence": 0.7,
            "disagreement_threshold": 0.25,
        },
        "calibration_status": "MOCK_CONTRACT_ONLY",
        "integrity_profile": "untrusted-multimodal-input@1",
        "allowed_evidence_roles": [
            "task-reference",
            "source-render",
            "candidate-render",
            "action-summary",
            "geometry-report",
        ],
    }


def group_spec() -> dict[str, Any]:
    spec = pointwise_spec()
    spec.update(
        {
            "judge_spec_id": "cad.sibling-group.mock@1",
            "prompt_template_digest": canonical_digest(
                {"template": "cad-sibling-group", "version": 1}
            ),
            "rubric_version": "cad-sibling-rubric@1",
            "mode": "GROUP",
            "criteria": ["sibling_preference", "progress", "regressions"],
            "presentation": {"blind_identity": True, "randomize_order": True},
            "allowed_evidence_roles": [
                "task-reference",
                "source-render",
                "sibling-render",
                "geometry-report",
            ],
        }
    )
    return spec


def blinded_order(subject_id: str, count: int) -> list[int]:
    order = list(range(count))
    random.Random(canonical_digest(subject_id)).shuffle(order)
    return order
