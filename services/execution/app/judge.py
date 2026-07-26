from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from hmac import new as hmac_new
from pathlib import Path
from typing import Any

from equinox_core import canonical_bytes, content_digest

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


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    content: str
    content_digest: str

    def render(self, evidence: dict[str, Any]) -> str:
        return (
            f"{self.content.rstrip()}\n\n"
            "<untrusted-evidence>\n"
            f"{canonical_bytes(evidence).decode('utf-8')}\n"
            "</untrusted-evidence>\n"
        )


@lru_cache(maxsize=1)
def prompt_registry() -> dict[str, PromptTemplate]:
    prompts_dir = Path(__file__).parents[1] / "prompts"
    manifest = json.loads((prompts_dir / "manifest.json").read_text(encoding="utf-8"))
    templates: dict[str, PromptTemplate] = {}
    for template_id, entry in manifest["templates"].items():
        prompt_bytes = (prompts_dir / entry["path"]).read_bytes()
        actual_digest = content_digest(prompt_bytes)
        if actual_digest != entry["sha256"]:
            raise RuntimeError(
                f"prompt {template_id} digest mismatch: "
                f"expected {entry['sha256']}, received {actual_digest}"
            )
        templates[template_id] = PromptTemplate(
            template_id=template_id,
            content=prompt_bytes.decode("utf-8"),
            content_digest=actual_digest,
        )
    return templates


class DeterministicJudgeFixture:
    name = "DeterministicJudgeFixture"
    model_identity = "deterministic-judge-fixture-2026-07-26"

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
                "Fixture assessment cites pinned renders and geometry evidence; "
                "it is deterministic contract evidence, not visual ground truth."
            ),
        }


def pointwise_spec(*, version: int = 1) -> dict[str, Any]:
    template = prompt_registry()["cad-pointwise-v1"]
    return {
        "judge_spec_id": f"cad.pointwise.fixture@{version}",
        "provider": "DeterministicJudgeFixture",
        "model_revision": "deterministic-judge-fixture-2026-07-26",
        "prompt_template_id": template.template_id,
        "prompt_template_digest": template.content_digest,
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
        "calibration_status": "FIXTURE_CONTRACT_ONLY",
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
    template = prompt_registry()["cad-group-v1"]
    spec.update(
        {
            "judge_spec_id": "cad.sibling-group.fixture@1",
            "prompt_template_id": template.template_id,
            "prompt_template_digest": template.content_digest,
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
    secret = os.getenv("JUDGE_PRESENTATION_SECRET", "equinox-local-fixture-only").encode()
    seed = hmac_new(secret, subject_id.encode(), sha256).hexdigest()
    random.Random(seed).shuffle(order)
    return order
