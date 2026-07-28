"""Sealed post-freeze external task pack for the revision-31 causal study."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from research.external.revision30_task_pack import (
    ExternalTask,
    GoldenEdit,
    Verifier,
    bounded,
    canonical_json,
    pure_function_verifier,
    safe_relative_path,
    sqlite_verifier,
)
from research.external.revision30_task_pack import (
    ExternalTaskEnvironment as Revision30ExternalTaskEnvironment,
)
from research.runpod.repository_repair_env_v31 import ACTION_REMINDER

PACK_ID = "revision31-post-freeze-external-pack@1"
TRAINER_FREEZE_COMMIT = "e6f08aa34ea183302159db68ff6abcdea68e4e6e"
AUTHORED_AT = "2026-07-29"
MAXIMUM_ACTIONS = 18
TASKS_PER_DOMAIN = 40


@dataclass(frozen=True)
class MicroFault:
    fault_id: str
    path: str
    bad_source: str
    expected_source: str
    cases: tuple[tuple[Any, ...], ...]
    old: str
    new: str
    test_id: str


MICRO_FAULTS = (
    MicroFault(
        "normalize_label",
        "src/labels.py",
        "def normalize_label(value):\n    return value.strip()\n",
        "def normalize_label(value):\n    return value.strip().lower()\n",
        ((" Admin ",), ("GUEST",), ("already-normal",), ("ß",)),
        "return value.strip()",
        "return value.strip().lower()",
        "test_normalizes_case_and_space",
    ),
    MicroFault(
        "retry_delay",
        "src/retry.py",
        "def retry_delay(base, index):\n    return base * (2 ** (index + 1))\n",
        "def retry_delay(base, index):\n    return base * (2 ** index)\n",
        ((1, 0), (3, 1), (2, 4), (0, 8)),
        "2 ** (index + 1)",
        "2 ** index",
        "test_retry_index_origin",
    ),
    MicroFault(
        "canonical_route",
        "src/routes.py",
        "def canonical_route(value):\n    return value.strip('/') if value else '/'\n",
        ("def canonical_route(value):\n    return '/' + value.strip('/') if value else '/'\n"),
        (("health",), ("/health/",), ("",), ("///nested/item//",)),
        "return value.strip('/') if value else '/'",
        "return '/' + value.strip('/') if value else '/'",
        "test_route_has_one_leading_slash",
    ),
    MicroFault(
        "lookup_fallback",
        "src/lookup.py",
        "def lookup(records, key, fallback):\n    return records[key]\n",
        "def lookup(records, key, fallback):\n    return records.get(key, fallback)\n",
        (({"a": 1}, "a", 9), ({"a": 1}, "b", 9), ({}, "x", None)),
        "return records[key]",
        "return records.get(key, fallback)",
        "test_lookup_uses_fallback",
    ),
    MicroFault(
        "clamp_bounds",
        "src/bounds.py",
        ("def clamp(value, minimum, maximum):\n    return min(minimum, max(maximum, value))\n"),
        ("def clamp(value, minimum, maximum):\n    return min(maximum, max(minimum, value))\n"),
        ((5, 0, 10), (-2, 0, 10), (22, 0, 10), (3, 3, 3)),
        "return min(minimum, max(maximum, value))",
        "return min(maximum, max(minimum, value))",
        "test_clamp_respects_bounds",
    ),
    MicroFault(
        "page_offset",
        "src/paging.py",
        "def page_offset(page, size):\n    return page * size\n",
        "def page_offset(page, size):\n    return (page - 1) * size\n",
        ((1, 20), (2, 20), (5, 7), (1, 1)),
        "return page * size",
        "return (page - 1) * size",
        "test_pages_are_one_indexed",
    ),
    MicroFault(
        "absolute_distance",
        "src/distance.py",
        "def distance(left, right):\n    return left - right\n",
        "def distance(left, right):\n    return abs(left - right)\n",
        ((1, 4), (4, 1), (-2, 3), (5, 5)),
        "return left - right",
        "return abs(left - right)",
        "test_distance_is_nonnegative",
    ),
    MicroFault(
        "suffix_match",
        "src/files.py",
        "def is_markdown(name):\n    return name.startswith('.md')\n",
        "def is_markdown(name):\n    return name.endswith('.md')\n",
        (("README.md",), (".md-cache",), ("notes.txt",), ("a.md",)),
        "return name.startswith('.md')",
        "return name.endswith('.md')",
        "test_markdown_suffix",
    ),
    MicroFault(
        "largest_value",
        "src/extrema.py",
        "def largest(left, right):\n    return min(left, right)\n",
        "def largest(left, right):\n    return max(left, right)\n",
        ((1, 2), (9, -1), (3, 3), (-8, -2)),
        "return min(left, right)",
        "return max(left, right)",
        "test_largest_selects_maximum",
    ),
    MicroFault(
        "none_coalesce",
        "src/defaults.py",
        "def coalesce(value, fallback):\n    return value or fallback\n",
        ("def coalesce(value, fallback):\n    return value if value is not None else fallback\n"),
        ((None, "x"), (0, 9), ("", "fallback"), ("value", "fallback")),
        "return value or fallback",
        "return value if value is not None else fallback",
        "test_only_none_uses_fallback",
    ),
)


def combined_verifier(verifiers: Sequence[tuple[str, Verifier]]) -> Verifier:
    def verify(root: Path) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for fault_id, verifier in verifiers:
            passed, fault_failures = verifier(root)
            failures.extend(f"{fault_id}:{failure}" for failure in fault_failures)
            if passed and fault_failures:
                raise RuntimeError("a passing verifier reported failures")
        return not failures, failures

    return verify


def micro_repository_tasks() -> list[ExternalTask]:
    tasks: list[ExternalTask] = []
    for index, selected in enumerate(combinations(MICRO_FAULTS, 2)):
        if index >= TASKS_PER_DOMAIN:
            break
        left, right = selected
        tasks.append(
            ExternalTask(
                task_id=f"r31-micro-{index + 1:02d}-{left.fault_id}-{right.fault_id}",
                domain="micro_repository",
                description=(
                    "Repair both independent library regressions without changing their "
                    "public function signatures."
                ),
                known_failing_tests=(left.test_id, right.test_id),
                files={
                    "README.md": (
                        f"Package audit case {index + 1}. Keep public APIs stable and fix "
                        "the two reported semantic regressions.\n"
                    ),
                    left.path: left.bad_source,
                    right.path: right.bad_source,
                },
                verifier_revision="restricted-python-ast-composite@2",
                verifier=combined_verifier(
                    (
                        (
                            left.fault_id,
                            pure_function_verifier(
                                left.path,
                                left.expected_source,
                                left.cases,
                            ),
                        ),
                        (
                            right.fault_id,
                            pure_function_verifier(
                                right.path,
                                right.expected_source,
                                right.cases,
                            ),
                        ),
                    )
                ),
                golden_edits=(
                    GoldenEdit(left.path, left.old, left.new),
                    GoldenEdit(right.path, right.old, right.new),
                ),
            )
        )
    if len(tasks) != TASKS_PER_DOMAIN:
        raise RuntimeError("the micro-repository pack has the wrong size")
    return tasks


def sqlite_account_task(variant: int) -> ExternalTask:
    active_id = variant * 10 + 1
    disabled_id = variant * 10 + 2
    email = f"USER{variant}@EXAMPLE.COM"
    disabled_email = f"KEEP{variant}@EXAMPLE.COM"
    return ExternalTask(
        task_id=f"r31-sqlite-{variant:02d}-active-email-scope",
        domain="sqlite_data_repair",
        description="Normalize email casing for active accounts only.",
        known_failing_tests=("active_email_invariant", "disabled_rows_unchanged"),
        files={
            "README.md": "Apply repair.sql to data/app.db without widening its scope.\n",
            "repair.sql": "UPDATE accounts SET email = lower(email);\n",
            "data/schema.sql": (
                "CREATE TABLE accounts(id INTEGER PRIMARY KEY,email TEXT,status TEXT);\n"
            ),
            "data/sample_rows.json": json.dumps(
                [
                    {"id": active_id, "email": email, "status": "active"},
                    {
                        "id": disabled_id,
                        "email": disabled_email,
                        "status": "disabled",
                    },
                ]
            )
            + "\n",
        },
        sqlite_setup=(
            "CREATE TABLE accounts(id INTEGER PRIMARY KEY,email TEXT,status TEXT);"
            f"INSERT INTO accounts VALUES({active_id},'{email}','active'),"
            f"({disabled_id},'{disabled_email}','disabled');"
        ),
        verifier_revision="sqlite-authorized-hidden-queries@2",
        verifier=sqlite_verifier(
            (
                (
                    "active_email_invariant",
                    f"SELECT email FROM accounts WHERE id={active_id}",
                    ((email.lower(),),),
                ),
                (
                    "disabled_rows_unchanged",
                    f"SELECT email FROM accounts WHERE id={disabled_id}",
                    ((disabled_email,),),
                ),
            )
        ),
        golden_edits=(
            GoldenEdit(
                "repair.sql",
                "UPDATE accounts SET email = lower(email);",
                "UPDATE accounts SET email = lower(email) WHERE status = 'active';",
            ),
        ),
    )


def sqlite_inventory_task(variant: int) -> ExternalTask:
    old_sku = f"old-{variant}"
    live_sku = f"live-{variant}"
    old_quantity = -(variant + 1)
    live_quantity = -(variant + 11)
    return ExternalTask(
        task_id=f"r31-sqlite-{variant:02d}-inventory-state-scope",
        domain="sqlite_data_repair",
        description="Clamp negative discontinued stock without changing live stock.",
        known_failing_tests=("discontinued_stock_floor", "live_stock_unchanged"),
        files={
            "README.md": "Repair only discontinued inventory records.\n",
            "repair.sql": "UPDATE inventory SET quantity = 0 WHERE quantity < 0;\n",
            "data/schema.sql": (
                "CREATE TABLE inventory(sku TEXT PRIMARY KEY,quantity INTEGER,state TEXT);\n"
            ),
            "data/sample_rows.json": json.dumps(
                [
                    {
                        "sku": old_sku,
                        "quantity": old_quantity,
                        "state": "discontinued",
                    },
                    {"sku": live_sku, "quantity": live_quantity, "state": "active"},
                ]
            )
            + "\n",
        },
        sqlite_setup=(
            "CREATE TABLE inventory(sku TEXT PRIMARY KEY,quantity INTEGER,state TEXT);"
            f"INSERT INTO inventory VALUES('{old_sku}',{old_quantity},'discontinued'),"
            f"('{live_sku}',{live_quantity},'active');"
        ),
        verifier_revision="sqlite-authorized-hidden-queries@2",
        verifier=sqlite_verifier(
            (
                (
                    "discontinued_stock_floor",
                    f"SELECT quantity FROM inventory WHERE sku='{old_sku}'",
                    ((0,),),
                ),
                (
                    "live_stock_unchanged",
                    f"SELECT quantity FROM inventory WHERE sku='{live_sku}'",
                    ((live_quantity,),),
                ),
            )
        ),
        golden_edits=(
            GoldenEdit(
                "repair.sql",
                "WHERE quantity < 0",
                "WHERE quantity < 0 AND state = 'discontinued'",
            ),
        ),
    )


def sqlite_event_task(variant: int) -> ExternalTask:
    job_id = 100 + variant
    started = f"2026-02-{variant + 1:02d}T10:00:00Z"
    finished = f"2026-02-{variant + 1:02d}T10:{variant + 10:02d}:00Z"
    return ExternalTask(
        task_id=f"r31-sqlite-{variant:02d}-terminal-event",
        domain="sqlite_data_repair",
        description="Backfill a job timestamp from its terminal event.",
        known_failing_tests=("terminal_timestamp", "event_kind_filter"),
        files={
            "README.md": "Use the latest finished event to repair jobs.finished_at.\n",
            "repair.sql": (
                "UPDATE jobs SET finished_at = "
                "(SELECT happened_at FROM events WHERE events.job_id = jobs.id "
                "ORDER BY happened_at ASC LIMIT 1) WHERE state = 'finished';\n"
            ),
            "data/schema.sql": (
                "CREATE TABLE jobs(id INTEGER PRIMARY KEY,state TEXT,finished_at TEXT);"
                "CREATE TABLE events(job_id INTEGER,kind TEXT,happened_at TEXT);\n"
            ),
            "data/sample_rows.json": json.dumps(
                [{"job_id": job_id, "started": started, "finished": finished}]
            )
            + "\n",
        },
        sqlite_setup=(
            "CREATE TABLE jobs(id INTEGER PRIMARY KEY,state TEXT,finished_at TEXT);"
            "CREATE TABLE events(job_id INTEGER,kind TEXT,happened_at TEXT);"
            f"INSERT INTO jobs VALUES({job_id},'finished',NULL);"
            f"INSERT INTO events VALUES({job_id},'started','{started}'),"
            f"({job_id},'finished','{finished}');"
        ),
        verifier_revision="sqlite-authorized-hidden-queries@2",
        verifier=sqlite_verifier(
            (
                (
                    "terminal_timestamp",
                    f"SELECT finished_at FROM jobs WHERE id={job_id}",
                    ((finished,),),
                ),
            )
        ),
        golden_edits=(
            GoldenEdit(
                "repair.sql",
                "ORDER BY happened_at ASC LIMIT 1",
                "AND kind = 'finished' ORDER BY happened_at DESC LIMIT 1",
            ),
        ),
    )


def sqlite_archive_task(variant: int) -> ExternalTask:
    archived_id = 200 + variant * 2
    active_id = archived_id + 1
    date = f"2026-03-{variant + 1:02d}"
    return ExternalTask(
        task_id=f"r31-sqlite-{variant:02d}-archive-backfill-scope",
        domain="sqlite_data_repair",
        description="Backfill purge dates for archived documents only.",
        known_failing_tests=("archived_date_backfilled", "active_row_unchanged"),
        files={
            "README.md": "Only archived documents may receive the purge date.\n",
            "repair.sql": (
                f"UPDATE documents SET purge_after = '{date}' WHERE purge_after IS NULL;\n"
            ),
            "data/schema.sql": (
                "CREATE TABLE documents(id INTEGER PRIMARY KEY,state TEXT,purge_after TEXT);\n"
            ),
            "data/sample_rows.json": json.dumps(
                [
                    {"id": archived_id, "state": "archived", "purge_after": None},
                    {"id": active_id, "state": "active", "purge_after": None},
                ]
            )
            + "\n",
        },
        sqlite_setup=(
            "CREATE TABLE documents(id INTEGER PRIMARY KEY,state TEXT,purge_after TEXT);"
            f"INSERT INTO documents VALUES({archived_id},'archived',NULL),"
            f"({active_id},'active',NULL);"
        ),
        verifier_revision="sqlite-authorized-hidden-queries@2",
        verifier=sqlite_verifier(
            (
                (
                    "archived_date_backfilled",
                    f"SELECT purge_after FROM documents WHERE id={archived_id}",
                    ((date,),),
                ),
                (
                    "active_row_unchanged",
                    f"SELECT purge_after FROM documents WHERE id={active_id}",
                    ((None,),),
                ),
            )
        ),
        golden_edits=(
            GoldenEdit(
                "repair.sql",
                "WHERE purge_after IS NULL",
                "WHERE purge_after IS NULL AND state = 'archived'",
            ),
        ),
    )


def sqlite_tasks() -> list[ExternalTask]:
    tasks = [
        task
        for variant in range(10)
        for task in (
            sqlite_account_task(variant),
            sqlite_inventory_task(variant),
            sqlite_event_task(variant),
            sqlite_archive_task(variant),
        )
    ]
    if len(tasks) != TASKS_PER_DOMAIN:
        raise RuntimeError("the SQLite pack has the wrong size")
    return tasks


def load_plan(root: Path) -> dict[str, Any]:
    value = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plan must be an object")
    return value


def rename_verifier(
    source_suffix: str,
    target_suffix: str,
    names: tuple[str, str],
) -> Verifier:
    def verify(root: Path) -> tuple[bool, list[str]]:
        sandbox = root / ".verification"
        shutil.copytree(root / "inbox", sandbox)
        try:
            plan = load_plan(root)
            if plan != {
                "command": "rename_suffix",
                "from_suffix": source_suffix,
                "to_suffix": target_suffix,
            }:
                raise ValueError
            for source in sorted(sandbox.glob(f"*{source_suffix}")):
                subprocess.run(
                    ["mv", "--", source.name, source.with_suffix(target_suffix).name],
                    cwd=sandbox,
                    check=True,
                    capture_output=True,
                    timeout=2,
                    env={"PATH": "/usr/bin:/bin"},
                )
            observed = sorted(path.name for path in sandbox.iterdir())
            expected = sorted([f"{name}{target_suffix}" for name in names] + ["keep.csv"])
            return observed == expected, ([] if observed == expected else ["renamed_tree"])
        except (OSError, subprocess.SubprocessError, ValueError):
            return False, ["plan_execution"]
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    return verify


def sort_verifier(delimiter: str, key: int, rows: tuple[str, ...]) -> Verifier:
    def verify(root: Path) -> tuple[bool, list[str]]:
        try:
            plan = load_plan(root)
            if plan != {
                "command": "sort_records",
                "delimiter": delimiter,
                "key": key,
                "numeric": True,
                "path": "data/records.txt",
            }:
                raise ValueError
            completed = subprocess.run(
                ["sort", "-t", delimiter, "-k", f"{key},{key}n", "records.txt"],
                cwd=root / "data",
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            expected = "".join(
                sorted(
                    rows,
                    key=lambda row: int(row.rstrip("\n").split(delimiter)[key - 1]),
                )
            )
            return completed.stdout == expected, (
                [] if completed.stdout == expected else ["numeric_order"]
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False, ["plan_execution"]

    return verify


def grep_verifier(needle: str, expected_lines: tuple[str, ...]) -> Verifier:
    def verify(root: Path) -> tuple[bool, list[str]]:
        try:
            plan = load_plan(root)
            if plan != {
                "command": "grep_incident",
                "fixed_string": True,
                "ignore_case": True,
                "path": "logs/service.log",
                "query": needle,
            }:
                raise ValueError
            completed = subprocess.run(
                ["grep", "-F", "-i", needle, "service.log"],
                cwd=root / "logs",
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            expected = "".join(expected_lines)
            passed = completed.returncode == 0 and completed.stdout == expected
            return passed, ([] if passed else ["literal_case_insensitive_match"])
        except (OSError, subprocess.SubprocessError, ValueError):
            return False, ["plan_execution"]

    return verify


def find_verifier(extension: str, expected: tuple[str, ...]) -> Verifier:
    def verify(root: Path) -> tuple[bool, list[str]]:
        try:
            plan = load_plan(root)
            if plan != {
                "command": "find_extension",
                "extension": extension,
                "max_depth": 1,
                "path": "archive",
            }:
                raise ValueError
            completed = subprocess.run(
                [
                    "find",
                    ".",
                    "-maxdepth",
                    "1",
                    "-type",
                    "f",
                    "-name",
                    f"*{extension}",
                    "-print",
                ],
                cwd=root / "archive",
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
            observed = tuple(
                sorted(line.removeprefix("./") for line in completed.stdout.splitlines())
            )
            return observed == expected, (
                [] if observed == expected else ["bounded_extension_search"]
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False, ["plan_execution"]

    return verify


def filesystem_cli_tasks() -> list[ExternalTask]:
    tasks: list[ExternalTask] = []
    for variant in range(10):
        names = (f"alpha{variant}", f"beta{variant}")
        source_suffix = ".text" if variant % 2 else ".txt"
        target_suffix = ".md"
        tasks.append(
            ExternalTask(
                task_id=f"r31-cli-{variant:02d}-rename-suffix",
                domain="filesystem_cli",
                description="Repair the bounded suffix-rename plan.",
                known_failing_tests=("renamed_tree",),
                files={
                    "plan.json": canonical_json(
                        {
                            "command": "rename_suffix",
                            "from_suffix": ".wrong",
                            "to_suffix": target_suffix,
                        }
                    )
                    + "\n",
                    f"inbox/{names[0]}{source_suffix}": "alpha\n",
                    f"inbox/{names[1]}{source_suffix}": "beta\n",
                    "inbox/keep.csv": "keep\n",
                },
                verifier_revision="allowlisted-cli-subprocess@2",
                verifier=rename_verifier(source_suffix, target_suffix, names),
                golden_edits=(GoldenEdit("plan.json", '".wrong"', json.dumps(source_suffix)),),
            )
        )

        delimiter = "," if variant % 2 == 0 else "|"
        key = 2
        rows = (
            f"alpha{variant}{delimiter}{100 + variant}\n",
            f"gamma{variant}{delimiter}{2 + variant}\n",
            f"beta{variant}{delimiter}{11 + variant}\n",
        )
        tasks.append(
            ExternalTask(
                task_id=f"r31-cli-{variant:02d}-numeric-sort",
                domain="filesystem_cli",
                description="Repair the plan so its numeric field is sorted numerically.",
                known_failing_tests=("numeric_order",),
                files={
                    "plan.json": canonical_json(
                        {
                            "command": "sort_records",
                            "delimiter": delimiter,
                            "key": key,
                            "numeric": False,
                            "path": "data/records.txt",
                        }
                    )
                    + "\n",
                    "data/records.txt": "".join(rows),
                },
                verifier_revision="allowlisted-cli-subprocess@2",
                verifier=sort_verifier(delimiter, key, rows),
                golden_edits=(GoldenEdit("plan.json", '"numeric":false', '"numeric":true'),),
            )
        )

        needle = f"alert[{40 + variant}]"
        expected_lines = (
            f"ALERT[{40 + variant}] retry exhausted\n",
            f"alert[{40 + variant}] queue stalled\n",
        )
        tasks.append(
            ExternalTask(
                task_id=f"r31-cli-{variant:02d}-literal-grep",
                domain="filesystem_cli",
                description="Repair the grep plan for a case-insensitive literal incident code.",
                known_failing_tests=("literal_case_insensitive_match",),
                files={
                    "plan.json": canonical_json(
                        {
                            "command": "grep_incident",
                            "fixed_string": False,
                            "ignore_case": False,
                            "path": "logs/service.log",
                            "query": needle,
                        }
                    )
                    + "\n",
                    "logs/service.log": (
                        "INFO startup\n"
                        + "".join(expected_lines)
                        + f"alert{40 + variant} unrelated\n"
                    ),
                },
                verifier_revision="allowlisted-cli-subprocess@2",
                verifier=grep_verifier(needle, expected_lines),
                golden_edits=(
                    GoldenEdit("plan.json", '"fixed_string":false', '"fixed_string":true'),
                    GoldenEdit("plan.json", '"ignore_case":false', '"ignore_case":true'),
                ),
            )
        )

        extension = ".log" if variant % 2 == 0 else ".trace"
        expected = tuple(sorted((f"worker{variant}{extension}", f"api{variant}{extension}")))
        tasks.append(
            ExternalTask(
                task_id=f"r31-cli-{variant:02d}-bounded-find",
                domain="filesystem_cli",
                description="Repair the bounded file-discovery plan.",
                known_failing_tests=("bounded_extension_search",),
                files={
                    "plan.json": canonical_json(
                        {
                            "command": "find_extension",
                            "extension": ".wrong",
                            "max_depth": 2,
                            "path": "archive",
                        }
                    )
                    + "\n",
                    f"archive/worker{variant}{extension}": "worker\n",
                    f"archive/api{variant}{extension}": "api\n",
                    "archive/keep.txt": "keep\n",
                    f"archive/nested/hidden{variant}{extension}": "hidden\n",
                },
                verifier_revision="allowlisted-cli-subprocess@2",
                verifier=find_verifier(extension, expected),
                golden_edits=(
                    GoldenEdit("plan.json", '".wrong"', json.dumps(extension)),
                    GoldenEdit("plan.json", '"max_depth":2', '"max_depth":1'),
                ),
            )
        )
    if len(tasks) != TASKS_PER_DOMAIN:
        raise RuntimeError("the filesystem/CLI pack has the wrong size")
    return tasks


def external_tasks() -> list[ExternalTask]:
    return [*micro_repository_tasks(), *sqlite_tasks(), *filesystem_cli_tasks()]


class ExternalTaskEnvironment(Revision30ExternalTaskEnvironment):
    """Use the revision-31 path and stale-edit recovery contract."""

    def policy_prompt(self) -> str:
        task_data = {
            "instruction": "Inspect the workspace, repair it, run tests, and finish.",
            "task": {
                "task_id": self.task.task_id,
                "domain": self.task.domain,
                "description": self.task.description,
                "known_failing_tests": self.task.known_failing_tests,
            },
            "transcript": [asdict(step) for step in self.steps[-8:]],
        }
        return (
            "<untrusted-environment-data>\n"
            + canonical_json(task_data)
            + "\n</untrusted-environment-data>\n\n"
            + ACTION_REMINDER
        )

    def execute(self, action: dict[str, str]) -> tuple[bool, str, bool | None]:
        tool = action["tool"]
        if tool == "read":
            path = action["path"]
            if not safe_relative_path(path) or path not in self.visible_files():
                return (
                    False,
                    canonical_json(
                        {
                            "error": "PATH_NOT_OBSERVED_OR_NOT_FOUND",
                            "next_action": {"tool": "list", "path": ""},
                        }
                    ),
                    None,
                )
        if tool == "edit":
            path, old = action["path"], action["old"]
            if safe_relative_path(path) and path in self.visible_files() and old:
                source = (self.root / path).read_text(encoding="utf-8")
                match_count = source.count(old)
                if match_count != 1:
                    return (
                        False,
                        bounded(
                            canonical_json(
                                {
                                    "error": (
                                        "EDIT_TARGET_NOT_FOUND"
                                        if match_count == 0
                                        else "EDIT_TARGET_AMBIGUOUS"
                                    ),
                                    "path": path,
                                    "match_count": match_count,
                                    "current_file": source,
                                    "recovery": (
                                        "Read this path, then derive an edit from its latest "
                                        "content."
                                    ),
                                }
                            )
                        ),
                        None,
                    )
        return super().execute(action)


def apply_golden_edits(environment: ExternalTaskEnvironment) -> None:
    for edit in environment.task.golden_edits:
        step = environment.step(
            canonical_json(
                {
                    "tool": "edit",
                    "path": edit.path,
                    "old": edit.old,
                    "new": edit.new,
                }
            )
        )
        if not step.accepted:
            raise RuntimeError(f"golden edit was rejected for {environment.task.task_id}")


def audit_pack(tasks: Sequence[ExternalTask] | None = None) -> dict[str, Any]:
    audited_tasks = list(tasks or external_tasks())
    task_ids = [task.task_id for task in audited_tasks]
    fixture_digests = [
        hashlib.sha256(canonical_json(task.files).encode()).hexdigest() for task in audited_tasks
    ]
    if len(task_ids) != 120 or len(set(task_ids)) != len(task_ids):
        raise RuntimeError("external task IDs are missing or duplicated")
    if len(set(fixture_digests)) != len(fixture_digests):
        raise RuntimeError("external fixtures are duplicated")
    for task in audited_tasks:
        if any(not safe_relative_path(path) for path in task.files):
            raise RuntimeError(f"{task.task_id} contains an unsafe fixture path")
        with ExternalTaskEnvironment(task) as environment:
            baseline_passed, _ = environment.verify()
            if baseline_passed:
                raise RuntimeError(f"{task.task_id} baseline unexpectedly passes")
            apply_golden_edits(environment)
            golden_passed, failures = environment.verify()
            if not golden_passed or failures:
                raise RuntimeError(f"{task.task_id} golden repair does not pass")
    return {
        "audit_revision": "baseline-fails-golden-passes-unique-safe@1",
        "audited_task_count": len(audited_tasks),
        "baseline_failures_confirmed": len(audited_tasks),
        "golden_repairs_confirmed": len(audited_tasks),
        "unique_task_ids": len(set(task_ids)),
        "unique_fixture_digests": len(set(fixture_digests)),
    }


def task_descriptors(tasks: Sequence[ExternalTask]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "domain": task.domain,
            "description": task.description,
            "known_failing_tests": list(task.known_failing_tests),
            "verifier_revision": task.verifier_revision,
            "fixture_digest": hashlib.sha256(canonical_json(task.files).encode()).hexdigest(),
        }
        for task in tasks
    ]


def pack_manifest(*, run_audit: bool = True) -> dict[str, Any]:
    tasks = external_tasks()
    descriptors = task_descriptors(tasks)
    return {
        "schema_version": 1,
        "pack_id": PACK_ID,
        "trainer_freeze_commit": TRAINER_FREEZE_COMMIT,
        "authored_at": AUTHORED_AT,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_count": len(tasks),
        "domains": {
            domain: sum(task.domain == domain for task in tasks)
            for domain in ("micro_repository", "sqlite_data_repair", "filesystem_cli")
        },
        "task_descriptor_digest": hashlib.sha256(canonical_json(descriptors).encode()).hexdigest(),
        "audit": audit_pack(tasks) if run_audit else None,
        "candidate_execution": {
            "network": "not_available_to_action_protocol",
            "credentials": "not_available_to_action_protocol",
            "python": "restricted_ast_interpreter",
            "sqlite": "authorizer_denies_attach_detach_and_pragma",
            "cli": "fixed_allowlisted_argv_with_two_second_timeout",
        },
    }


if __name__ == "__main__":
    print(json.dumps(pack_manifest(), sort_keys=True, indent=2))
