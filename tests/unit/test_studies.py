import json
from pathlib import Path

import pytest

from services.orchestrator.app.studies import (
    find_study_report,
    list_study_summaries,
    load_study_reports,
)


def write_report(directory: Path, *, study_id: str, generated_at: str) -> None:
    (directory / f"{study_id.replace('@', '-')}.json").write_text(
        json.dumps(
            {
                "study_id": study_id,
                "report_id": f"{study_id}/report",
                "generated_at": generated_at,
                "overall_status": "FAIL",
                "freeze": {
                    "frozen_workload_revision": "revision-30",
                    "model": {"id": "model"},
                },
                "conditions": {"trained": {}, "control": {}},
                "decisions": {
                    "transfer": {"status": "FAIL"},
                    "teardown": {"status": "PASS"},
                },
                "executions": [
                    {"estimated_cost_usd": 0.25},
                    {"estimated_cost_usd": None},
                    {"estimated_cost_usd": 0.125},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_study_reports_are_live_summaries_of_committed_evidence(tmp_path: Path) -> None:
    write_report(tmp_path, study_id="study@1", generated_at="2026-07-28T00:00:00Z")
    write_report(tmp_path, study_id="study@2", generated_at="2026-07-29T00:00:00Z")
    (tmp_path / "study-definition.json").write_text(
        json.dumps({"study_id": "study@3", "conditions": []}),
        encoding="utf-8",
    )

    summaries = list_study_summaries(tmp_path)

    assert [item["study_id"] for item in summaries] == ["study@2", "study@1"]
    assert summaries[0] == {
        "study_id": "study@2",
        "report_id": "study@2/report",
        "generated_at": "2026-07-29T00:00:00Z",
        "overall_status": "FAIL",
        "workload_revision": "revision-30",
        "model_id": "model",
        "condition_count": 2,
        "execution_count": 3,
        "estimated_provider_cost_usd": 0.375,
        "decisions": {"transfer": "FAIL", "teardown": "PASS"},
    }
    assert find_study_report("study@1", tmp_path)["report_id"] == "study@1/report"
    assert find_study_report("../study@1", tmp_path) is None


def test_study_report_loader_rejects_non_report_json(tmp_path: Path) -> None:
    (tmp_path / "invalid.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="is not a study report"):
        load_study_reports(tmp_path)
