from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

NUMERIC_TYPES = (int, float)


def study_report_directory() -> Path:
    configured = os.environ.get("EQUINOX_STUDY_REPORT_DIRECTORY")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "research" / "studies"


def load_study_reports(directory: Path | None = None) -> list[dict[str, Any]]:
    root = directory or study_report_directory()
    if not root.exists():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path} is not a study report")
        if not (
            isinstance(value.get("study_id"), str)
            and isinstance(value.get("report_id"), str)
            and value.get("overall_status") in {"PASS", "FAIL"}
        ):
            continue
        reports.append(value)
    return sorted(
        reports,
        key=lambda report: str(report.get("generated_at", "")),
        reverse=True,
    )


def study_summary(report: dict[str, Any]) -> dict[str, Any]:
    freeze = report.get("freeze")
    freeze = freeze if isinstance(freeze, dict) else {}
    model = freeze.get("model")
    model = model if isinstance(model, dict) else {}
    decisions = report.get("decisions")
    decisions = decisions if isinstance(decisions, dict) else {}
    conditions = report.get("conditions")
    conditions = conditions if isinstance(conditions, dict) else {}
    executions = report.get("executions")
    executions = executions if isinstance(executions, list) else []
    costs = [
        item.get("estimated_cost_usd")
        for item in executions
        if isinstance(item, dict)
        and isinstance(item.get("estimated_cost_usd"), NUMERIC_TYPES)
        and not isinstance(item.get("estimated_cost_usd"), bool)
    ]
    return {
        "study_id": report["study_id"],
        "report_id": report.get("report_id"),
        "generated_at": report.get("generated_at"),
        "overall_status": report.get("overall_status"),
        "workload_revision": freeze.get("frozen_workload_revision"),
        "model_id": model.get("id"),
        "condition_count": len(conditions),
        "execution_count": len(executions),
        "estimated_provider_cost_usd": round(sum(costs), 6),
        "decisions": {
            decision_id: decision.get("status")
            for decision_id, decision in decisions.items()
            if isinstance(decision, dict)
        },
    }


def list_study_summaries(directory: Path | None = None) -> list[dict[str, Any]]:
    return [study_summary(report) for report in load_study_reports(directory)]


def find_study_report(
    study_id: str,
    directory: Path | None = None,
) -> dict[str, Any] | None:
    return next(
        (report for report in load_study_reports(directory) if report.get("study_id") == study_id),
        None,
    )
