from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from research.external.revision30_task_pack import (
    ExternalTaskEnvironment,
    apply_golden_edits,
    external_tasks,
    pack_manifest,
)


def test_external_pack_is_balanced_and_postdates_trainer_freeze() -> None:
    tasks = external_tasks()
    manifest = pack_manifest()

    assert len(tasks) == 9
    assert Counter(task.domain for task in tasks) == {
        "micro_repository": 3,
        "sqlite_data_repair": 3,
        "filesystem_cli": 3,
    }
    assert manifest["trainer_freeze_commit"] == "7195efc"
    assert manifest["authored_at"] == "2026-07-28"
    assert manifest["task_count"] == 9
    assert len({task.task_id for task in tasks}) == len(tasks)
    frozen = json.loads(
        Path("research/frozen/revision30-external-pack.json").read_text(encoding="utf-8")
    )
    assert manifest == frozen


def test_every_external_fixture_fails_initially_and_passes_its_hidden_verifier() -> None:
    for task in external_tasks():
        with ExternalTaskEnvironment(task) as environment:
            initial_passed, initial_failures = environment.verify()
            assert initial_passed is False, task.task_id
            assert initial_failures, task.task_id

            apply_golden_edits(environment)
            final_passed, final_failures = environment.verify()
            assert final_passed is True, task.task_id
            assert final_failures == [], task.task_id


def test_external_environment_rejects_paths_and_keeps_hidden_checks_out_of_prompt() -> None:
    task = external_tasks()[0]
    with ExternalTaskEnvironment(task) as environment:
        prompt = environment.policy_prompt()
        rejected = environment.step('{"tool":"read","path":"../etc/passwd"}')

        assert task.description in prompt
        assert task.golden_edits[0].new not in prompt
        assert rejected.accepted is False
        assert environment.visible_files()
