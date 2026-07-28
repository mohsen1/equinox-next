"""Blind protocol-eligibility screen for frozen revision-30 study splits."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import repository_repair_rl as frozen
    import repository_repair_study as study
except ModuleNotFoundError:
    from . import repository_repair_rl as frozen
    from . import repository_repair_study as study


SCREEN_REVISION = "revision30-protocol-eligibility@1"


class EligibilityScreenComplete(BaseException):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__("the protocol-eligibility screen completed")


def eligibility_result(
    configuration: study.StudyConfiguration,
    *,
    action_protocol_validity_rate: float | None,
    minimum_protocol_validity_rate: float,
    elapsed_seconds: float,
    device: str,
    validation_examples: int,
) -> dict[str, Any]:
    eligible = (
        action_protocol_validity_rate is not None
        and action_protocol_validity_rate >= minimum_protocol_validity_rate
    )
    return {
        "schema_version": 1,
        "workload": "repository-repair-protocol-eligibility-screen",
        "workload_revision": SCREEN_REVISION,
        "screen_completed": True,
        "protocol_eligible": eligible,
        "training_started": False,
        "optimizer_step_calls": 0,
        "test_split_accessed": False,
        "correctness_metrics_disclosed": False,
        "device": device,
        "model_id": os.environ.get("EQUINOX_RL_MODEL_ID"),
        "study_id": study.STUDY_ID,
        "condition": configuration.condition,
        "optimization_seed": configuration.optimization_seed,
        "validation_seed_base": configuration.validation_seed_base,
        "test_seed_base_reserved_but_unread": configuration.test_seed_base,
        "branch_width": configuration.branch_width,
        "validation_examples": validation_examples,
        "action_protocol_validity_rate": action_protocol_validity_rate,
        "minimum_protocol_validity_rate": minimum_protocol_validity_rate,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "frozen_source_commit": study.FROZEN_SOURCE_COMMIT,
        "frozen_source_sha256": study.FROZEN_SOURCE_SHA256,
        "study_configuration": asdict(configuration),
    }


def install_screen(configuration: study.StudyConfiguration) -> None:
    original_emit_progress = frozen.emit_progress

    def screen_emit_progress(
        phase: str,
        message: str,
        runtime_configuration: Any,
        *,
        preserve_context: bool = False,
        **values: Any,
    ) -> None:
        public_values = values
        if phase == "protocol_evaluation":
            public_values = {
                key: values[key]
                for key in (
                    "elapsed_seconds",
                    "current_level",
                    "evaluation_split",
                    "evaluation_examples",
                    "evaluation_completed",
                    "evaluation_total",
                    "action_protocol_validity_rate",
                    "minimum_protocol_validity_rate",
                )
                if key in values
            }
        original_emit_progress(
            phase,
            message,
            runtime_configuration,
            preserve_context=preserve_context,
            eligibility_screen_revision=SCREEN_REVISION,
            correctness_metrics_disclosed=False,
            test_split_accessed=False,
            **public_values,
        )
        if phase != "protocol_evaluation":
            return
        validity = values.get("action_protocol_validity_rate")
        minimum = values.get("minimum_protocol_validity_rate")
        elapsed = values.get("elapsed_seconds")
        if validity is not None and (
            isinstance(validity, bool) or not isinstance(validity, int | float)
        ):
            raise RuntimeError("the frozen trainer emitted an invalid protocol-validity rate")
        if isinstance(minimum, bool) or not isinstance(minimum, int | float):
            raise RuntimeError("the frozen trainer omitted the protocol-validity threshold")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
            raise RuntimeError("the frozen trainer omitted eligibility elapsed time")
        raise EligibilityScreenComplete(
            eligibility_result(
                configuration,
                action_protocol_validity_rate=(None if validity is None else float(validity)),
                minimum_protocol_validity_rate=float(minimum),
                elapsed_seconds=float(elapsed),
                device="cuda",
                validation_examples=int(runtime_configuration.validation_examples),
            )
        )

    frozen.emit_progress = screen_emit_progress


def main() -> None:
    configuration = study.study_configuration_from_environment()
    if configuration.condition != "k4_train":
        raise ValueError("eligibility screening is only defined for K=4 training conditions")
    study.verify_frozen_sources()
    if "--validate-configuration" in sys.argv:
        frozen.main()
        return
    if "--self-test" in sys.argv:
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "screen_revision": SCREEN_REVISION,
                    "configuration": asdict(configuration),
                },
                sort_keys=True,
            )
        )
        return
    evidence_path = (
        Path(os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp"))
        / "eligibility-unused-study-evidence.json"
    )
    evidence = study.RuntimeEvidence()
    study.install_study_condition(configuration, evidence, evidence_path)
    install_screen(configuration)
    try:
        frozen.main()
    except EligibilityScreenComplete as completed:
        print(json.dumps(completed.result, sort_keys=True))
        return
    raise RuntimeError("the frozen trainer ended without emitting protocol eligibility")


if __name__ == "__main__":
    main()
