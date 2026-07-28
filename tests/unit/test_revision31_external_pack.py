import json
from collections import Counter
from pathlib import Path

from research.external.revision31_task_pack import (
    PACK_ID,
    ExternalTaskEnvironment,
    apply_golden_edits,
    audit_pack,
    external_tasks,
    pack_manifest,
)


def test_pack_has_120_audited_tasks_balanced_across_domains() -> None:
    tasks = external_tasks()

    assert len(tasks) == 120
    assert Counter(task.domain for task in tasks) == {
        "micro_repository": 40,
        "sqlite_data_repair": 40,
        "filesystem_cli": 40,
    }
    assert audit_pack(tasks) == {
        "audit_revision": "baseline-fails-golden-passes-unique-safe@1",
        "audited_task_count": 120,
        "baseline_failures_confirmed": 120,
        "golden_repairs_confirmed": 120,
        "unique_task_ids": 120,
        "unique_fixture_digests": 120,
    }


def test_every_baseline_fails_and_every_golden_repair_passes() -> None:
    for task in external_tasks():
        with ExternalTaskEnvironment(task) as environment:
            assert environment.verify()[0] is False
            apply_golden_edits(environment)
            assert environment.verify() == (True, [])


def test_external_environment_uses_revision31_recovery_contract() -> None:
    task = external_tasks()[0]
    with ExternalTaskEnvironment(task) as environment:
        rejected = environment.step('{"tool":"read","path":"relative/path.py"}')
        recovery = json.loads(rejected.observation)

    assert rejected.accepted is False
    assert recovery["error"] == "PATH_NOT_OBSERVED_OR_NOT_FOUND"
    assert recovery["next_action"] == {"path": "", "tool": "list"}


def test_pack_matches_sealed_manifest() -> None:
    expected = json.loads(
        Path("research/frozen/revision31-external-pack.json").read_text(encoding="utf-8")
    )

    assert pack_manifest() == expected
    assert expected["pack_id"] == PACK_ID
