"""Post-freeze external tasks for revision-30 adapter transfer evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from research.runpod.repository_repair_env import (
    ACTION_REMINDER,
    _evaluate_expression,
    _return_expression,
    parse_action,
)

PACK_ID = "revision30-post-freeze-external-pack@1"
TRAINER_FREEZE_COMMIT = "7195efc"
AUTHORED_AT = "2026-07-28"
MAXIMUM_ACTIONS = 14
Verifier = Callable[[Path], tuple[bool, list[str]]]


@dataclass(frozen=True)
class GoldenEdit:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class ExternalTask:
    task_id: str
    domain: Literal["micro_repository", "sqlite_data_repair", "filesystem_cli"]
    description: str
    known_failing_tests: tuple[str, ...]
    files: dict[str, str]
    verifier_revision: str
    verifier: Verifier = field(repr=False, compare=False)
    golden_edits: tuple[GoldenEdit, ...] = field(repr=False, compare=False)
    sqlite_setup: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ExternalStep:
    index: int
    action: dict[str, str] | None
    accepted: bool
    observation: str
    verifier_passed: bool | None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bounded(value: str, limit: int = 4_000) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def safe_relative_path(value: str, *, allow_empty: bool = False) -> bool:
    if not value:
        return allow_empty
    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and all(part not in {"", "."} for part in path.parts)
    )


def pure_function_verifier(
    path: str,
    expected_source: str,
    cases: Sequence[tuple[Any, ...]],
) -> Verifier:
    expected_expression = _return_expression(expected_source)
    if expected_expression is None:
        raise ValueError("expected source is outside the restricted verifier subset")
    expected_arguments, expected_return = expected_expression

    def verify(root: Path) -> tuple[bool, list[str]]:
        candidate_path = root / path
        try:
            candidate_expression = _return_expression(candidate_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            candidate_expression = None
        if candidate_expression is None:
            return False, ["module_parse"]
        candidate_arguments, candidate_return = candidate_expression
        if candidate_arguments != expected_arguments:
            return False, ["function_contract"]
        failures: list[str] = []
        for index, arguments in enumerate(cases):
            values = dict(zip(candidate_arguments, arguments, strict=True))
            try:
                observed = _evaluate_expression(candidate_return, values)
                expected = _evaluate_expression(expected_return, values)
            except (
                AttributeError,
                IndexError,
                KeyError,
                MemoryError,
                OverflowError,
                RecursionError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                failures.append(f"semantic_case_{index}")
                continue
            if type(observed) is not type(expected) or observed != expected:
                failures.append(f"semantic_case_{index}")
        return not failures, failures

    return verify


def sqlite_verifier(
    checks: Sequence[tuple[str, str, tuple[tuple[Any, ...], ...]]],
) -> Verifier:
    denied_actions = {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    }

    def verify(root: Path) -> tuple[bool, list[str]]:
        source_database = root / "data/app.db"
        candidate_database = root / "data/verification.db"
        shutil.copyfile(source_database, candidate_database)
        connection = sqlite3.connect(candidate_database)
        connection.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_DENY if action in denied_actions else sqlite3.SQLITE_OK
            )
        )
        failures: list[str] = []
        try:
            repair = (root / "repair.sql").read_text(encoding="utf-8")
            connection.executescript(repair)
            connection.commit()
            for check_id, query, expected in checks:
                if tuple(connection.execute(query).fetchall()) != expected:
                    failures.append(check_id)
        except (OSError, sqlite3.Error, UnicodeError):
            failures.append("repair_execution")
        finally:
            connection.close()
            candidate_database.unlink(missing_ok=True)
        return not failures, failures

    return verify


def load_cli_plan(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CLI plan must be an object")
    return payload


def rename_cli_verifier(root: Path) -> tuple[bool, list[str]]:
    sandbox = root / ".verification"
    shutil.copytree(root / "inbox", sandbox)
    try:
        plan = load_cli_plan(root)
        if set(plan) != {"command", "from_suffix", "to_suffix"}:
            raise ValueError
        if (
            plan["command"] != "rename_suffix"
            or plan["from_suffix"] not in {".txt", ".text"}
            or plan["to_suffix"] != ".md"
        ):
            raise ValueError
        for source in sorted(sandbox.glob(f"*{plan['from_suffix']}")):
            target = source.with_suffix(str(plan["to_suffix"]))
            subprocess.run(
                ["mv", "--", source.name, target.name],
                cwd=sandbox,
                check=True,
                capture_output=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin"},
            )
        observed = sorted(path.name for path in sandbox.iterdir())
        expected = ["alpha.md", "beta.md", "keep.csv"]
        return observed == expected, ([] if observed == expected else ["renamed_tree"])
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, ["plan_execution"]
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def sort_cli_verifier(root: Path) -> tuple[bool, list[str]]:
    try:
        plan = load_cli_plan(root)
        if set(plan) != {"command", "delimiter", "key", "numeric", "path"}:
            raise ValueError
        if (
            plan["command"] != "sort_records"
            or plan["delimiter"] != ","
            or plan["key"] != 2
            or not isinstance(plan["numeric"], bool)
            or plan["path"] != "data/records.csv"
        ):
            raise ValueError
        arguments = ["sort", "-t", ",", "-k", "2,2n" if plan["numeric"] else "2,2"]
        arguments.append("records.csv")
        completed = subprocess.run(
            arguments,
            cwd=root / "data",
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        expected = "gamma,2\nbeta,11\nalpha,100\n"
        return completed.stdout == expected, (
            [] if completed.stdout == expected else ["numeric_order"]
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, ["plan_execution"]


def grep_cli_verifier(root: Path) -> tuple[bool, list[str]]:
    try:
        plan = load_cli_plan(root)
        if set(plan) != {"command", "fixed_string", "ignore_case", "path"}:
            raise ValueError
        if (
            plan["command"] != "grep_errors"
            or not isinstance(plan["fixed_string"], bool)
            or not isinstance(plan["ignore_case"], bool)
            or plan["path"] != "logs/service.log"
        ):
            raise ValueError
        arguments = ["grep"]
        if plan["fixed_string"]:
            arguments.append("-F")
        if plan["ignore_case"]:
            arguments.append("-i")
        arguments.extend(["error[42]", "service.log"])
        completed = subprocess.run(
            arguments,
            cwd=root / "logs",
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        expected = "ERROR[42] retry budget exhausted\nerror[42] queue stalled\n"
        passed = completed.returncode == 0 and completed.stdout == expected
        return passed, ([] if passed else ["literal_case_insensitive_match"])
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, ["plan_execution"]


def micro_repository_tasks() -> list[ExternalTask]:
    return [
        ExternalTask(
            task_id="external-micro-label-normalization",
            domain="micro_repository",
            description="Normalize user labels before they become cache keys.",
            known_failing_tests=("test_collapses_case_variants", "test_trims_outer_space"),
            files={
                "README.md": "The cache treats labels case-insensitively.\n",
                "src/labels.py": "def normalize_label(value):\n    return value.strip()\n",
                "src/cache.py": (
                    "from .labels import normalize_label\n\n"
                    "def cache_key(value):\n    return 'label:' + normalize_label(value)\n"
                ),
            },
            verifier_revision="restricted-python-ast@1",
            verifier=pure_function_verifier(
                "src/labels.py",
                "def normalize_label(value):\n    return value.strip().lower()\n",
                ((" Admin ",), ("GUEST",), ("already-normal",), ("ß",)),
            ),
            golden_edits=(
                GoldenEdit("src/labels.py", "return value.strip()", "return value.strip().lower()"),
            ),
        ),
        ExternalTask(
            task_id="external-micro-retry-delay",
            domain="micro_repository",
            description="Return the delay for the current retry index.",
            known_failing_tests=("test_first_retry_uses_base", "test_growth_is_exponential"),
            files={
                "pyproject.toml": '[project]\nname = "retry-window"\nversion = "0.1.0"\n',
                "src/retry.py": (
                    "def retry_delay(base, index):\n    return base * (2 ** (index + 1))\n"
                ),
                "src/worker.py": (
                    "from .retry import retry_delay\n\n"
                    "def next_delay(config, attempt):\n"
                    "    return retry_delay(config['base_delay'], attempt)\n"
                ),
            },
            verifier_revision="restricted-python-ast@1",
            verifier=pure_function_verifier(
                "src/retry.py",
                "def retry_delay(base, index):\n    return base * (2 ** index)\n",
                ((1, 0), (3, 1), (2, 4), (0, 8)),
            ),
            golden_edits=(GoldenEdit("src/retry.py", "2 ** (index + 1)", "2 ** index"),),
        ),
        ExternalTask(
            task_id="external-micro-route-prefix",
            domain="micro_repository",
            description="Canonicalize a route fragment with exactly one leading slash.",
            known_failing_tests=("test_adds_leading_slash", "test_preserves_root"),
            files={
                "README.md": "Route fragments are stored with one leading slash.\n",
                "src/routes.py": (
                    "def canonical_route(value):\n    return value.strip('/') if value else '/'\n"
                ),
                "src/registry.py": (
                    "from .routes import canonical_route\n\n"
                    "def register(routes, value):\n"
                    "    routes[canonical_route(value)] = True\n"
                ),
            },
            verifier_revision="restricted-python-ast@1",
            verifier=pure_function_verifier(
                "src/routes.py",
                "def canonical_route(value):\n"
                "    return '/' + value.strip('/') if value else '/'\n",
                (("health",), ("/health/",), ("",), ("///nested/item//",)),
            ),
            golden_edits=(
                GoldenEdit(
                    "src/routes.py",
                    "return value.strip('/') if value else '/'",
                    "return '/' + value.strip('/') if value else '/'",
                ),
            ),
        ),
    ]


def sqlite_tasks() -> list[ExternalTask]:
    return [
        ExternalTask(
            task_id="external-sqlite-email-normalization",
            domain="sqlite_data_repair",
            description="Normalize email casing for active accounts only.",
            known_failing_tests=("active_email_invariant", "disabled_rows_unchanged"),
            files={
                "README.md": "Apply repair.sql to data/app.db.\n",
                "repair.sql": "UPDATE accounts SET email = lower(email);\n",
                "data/schema.sql": (
                    "CREATE TABLE accounts(id INTEGER PRIMARY KEY,email TEXT,status TEXT);\n"
                ),
                "data/sample_rows.json": (
                    '[{"id":1,"email":"ALICE@EXAMPLE.COM","status":"active"},'
                    '{"id":2,"email":"KEEP@EXAMPLE.COM","status":"disabled"}]\n'
                ),
            },
            sqlite_setup=(
                "CREATE TABLE accounts(id INTEGER PRIMARY KEY,email TEXT,status TEXT);"
                "INSERT INTO accounts VALUES"
                "(1,'ALICE@EXAMPLE.COM','active'),"
                "(2,'KEEP@EXAMPLE.COM','disabled'),"
                "(3,'bob@example.com','active');"
            ),
            verifier_revision="sqlite-authorized-hidden-queries@1",
            verifier=sqlite_verifier(
                (
                    (
                        "active_email_invariant",
                        "SELECT id,email FROM accounts WHERE status='active' ORDER BY id",
                        ((1, "alice@example.com"), (3, "bob@example.com")),
                    ),
                    (
                        "disabled_rows_unchanged",
                        "SELECT id,email FROM accounts WHERE status='disabled'",
                        ((2, "KEEP@EXAMPLE.COM"),),
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
        ),
        ExternalTask(
            task_id="external-sqlite-stock-floor",
            domain="sqlite_data_repair",
            description="Clamp negative stock for discontinued products without changing live stock.",
            known_failing_tests=("discontinued_stock_floor", "live_stock_unchanged"),
            files={
                "README.md": "Repair invalid inventory data with repair.sql.\n",
                "repair.sql": "UPDATE inventory SET quantity = 0 WHERE quantity < 0;\n",
                "data/schema.sql": (
                    "CREATE TABLE inventory(sku TEXT PRIMARY KEY,quantity INTEGER,state TEXT);\n"
                ),
                "data/sample_rows.json": (
                    '[{"sku":"old","quantity":-4,"state":"discontinued"},'
                    '{"sku":"live","quantity":-2,"state":"active"}]\n'
                ),
            },
            sqlite_setup=(
                "CREATE TABLE inventory(sku TEXT PRIMARY KEY,quantity INTEGER,state TEXT);"
                "INSERT INTO inventory VALUES"
                "('old',-4,'discontinued'),('live',-2,'active'),('ok',7,'active');"
            ),
            verifier_revision="sqlite-authorized-hidden-queries@1",
            verifier=sqlite_verifier(
                (
                    (
                        "discontinued_stock_floor",
                        "SELECT quantity FROM inventory WHERE sku='old'",
                        ((0,),),
                    ),
                    (
                        "live_stock_unchanged",
                        "SELECT sku,quantity FROM inventory WHERE state='active' ORDER BY sku",
                        (("live", -2), ("ok", 7)),
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
        ),
        ExternalTask(
            task_id="external-sqlite-job-completion",
            domain="sqlite_data_repair",
            description="Backfill completion timestamps from the final job event.",
            known_failing_tests=("finished_job_timestamp", "running_job_remains_null"),
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
                "data/sample_rows.json": (
                    '[{"job":7,"events":["2026-01-01T10:00:00Z","2026-01-01T10:05:00Z"]}]\n'
                ),
            },
            sqlite_setup=(
                "CREATE TABLE jobs(id INTEGER PRIMARY KEY,state TEXT,finished_at TEXT);"
                "CREATE TABLE events(job_id INTEGER,kind TEXT,happened_at TEXT);"
                "INSERT INTO jobs VALUES(7,'finished',NULL),(8,'running',NULL);"
                "INSERT INTO events VALUES"
                "(7,'started','2026-01-01T10:00:00Z'),"
                "(7,'finished','2026-01-01T10:05:00Z'),"
                "(8,'started','2026-01-01T11:00:00Z');"
            ),
            verifier_revision="sqlite-authorized-hidden-queries@1",
            verifier=sqlite_verifier(
                (
                    (
                        "finished_job_timestamp",
                        "SELECT finished_at FROM jobs WHERE id=7",
                        (("2026-01-01T10:05:00Z",),),
                    ),
                    (
                        "running_job_remains_null",
                        "SELECT finished_at FROM jobs WHERE id=8",
                        ((None,),),
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
        ),
    ]


def filesystem_cli_tasks() -> list[ExternalTask]:
    return [
        ExternalTask(
            task_id="external-cli-rename-suffix",
            domain="filesystem_cli",
            description="Fix a bounded rename plan so Markdown sources receive the .md suffix.",
            known_failing_tests=("renamed_tree",),
            files={
                "plan.json": (
                    '{"command":"rename_suffix","from_suffix":".text","to_suffix":".md"}\n'
                ),
                "inbox/alpha.txt": "alpha\n",
                "inbox/beta.txt": "beta\n",
                "inbox/keep.csv": "keep\n",
            },
            verifier_revision="allowlisted-cli-subprocess@1",
            verifier=rename_cli_verifier,
            golden_edits=(GoldenEdit("plan.json", '".text"', '".txt"'),),
        ),
        ExternalTask(
            task_id="external-cli-numeric-sort",
            domain="filesystem_cli",
            description="Fix a sort plan so the second CSV field is ordered numerically.",
            known_failing_tests=("numeric_order",),
            files={
                "plan.json": (
                    '{"command":"sort_records","delimiter":",","key":2,'
                    '"numeric":false,"path":"data/records.csv"}\n'
                ),
                "data/records.csv": "alpha,100\ngamma,2\nbeta,11\n",
            },
            verifier_revision="allowlisted-cli-subprocess@1",
            verifier=sort_cli_verifier,
            golden_edits=(GoldenEdit("plan.json", '"numeric":false', '"numeric":true'),),
        ),
        ExternalTask(
            task_id="external-cli-literal-grep",
            domain="filesystem_cli",
            description="Fix a grep plan for a case-insensitive literal incident code.",
            known_failing_tests=("literal_case_insensitive_match",),
            files={
                "plan.json": (
                    '{"command":"grep_errors","fixed_string":false,"ignore_case":false,'
                    '"path":"logs/service.log"}\n'
                ),
                "logs/service.log": (
                    "INFO startup\n"
                    "ERROR[42] retry budget exhausted\n"
                    "error[42] queue stalled\n"
                    "ERROR4 unrelated\n"
                ),
            },
            verifier_revision="allowlisted-cli-subprocess@1",
            verifier=grep_cli_verifier,
            golden_edits=(
                GoldenEdit("plan.json", '"fixed_string":false', '"fixed_string":true'),
                GoldenEdit("plan.json", '"ignore_case":false', '"ignore_case":true'),
            ),
        ),
    ]


def external_tasks() -> list[ExternalTask]:
    return [*micro_repository_tasks(), *sqlite_tasks(), *filesystem_cli_tasks()]


class ExternalTaskEnvironment:
    def __init__(self, task: ExternalTask) -> None:
        self.task = task
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=f"equinox-{task.task_id}-")
        self.root = Path(self._temporary_directory.name)
        for relative_path, content in task.files.items():
            destination = self.root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        if task.sqlite_setup is not None:
            database_path = self.root / "data/app.db"
            database_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(database_path) as connection:
                connection.executescript(task.sqlite_setup)
        self.steps: list[ExternalStep] = []
        self.terminal = False
        self.terminal_reason: str | None = None
        self.reward = 0.0

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> ExternalTaskEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def visible_files(self) -> list[str]:
        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
            and ".verification" not in path.parts
            and path.name not in {"app.db", "verification.db"}
        )

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

    def verify(self) -> tuple[bool, list[str]]:
        return self.task.verifier(self.root)

    def execute(self, action: dict[str, str]) -> tuple[bool, str, bool | None]:
        tool = action["tool"]
        if tool == "list":
            requested = action["path"].rstrip("/")
            if not safe_relative_path(requested, allow_empty=True):
                return False, "Rejected unsafe path.", None
            prefix = f"{requested}/" if requested else ""
            matches = [path for path in self.visible_files() if path.startswith(prefix)]
            return (
                (True, bounded(canonical_json({"files": matches})), None)
                if matches
                else (False, "No files found.", None)
            )
        if tool == "read":
            path = action["path"]
            if not safe_relative_path(path) or path not in self.visible_files():
                return False, "File not found or path rejected.", None
            return True, bounded((self.root / path).read_text(encoding="utf-8")), None
        if tool == "search":
            query = action["query"]
            if not query or len(query) > 120:
                return False, "Search query rejected.", None
            matches = []
            for path in self.visible_files():
                for line_number, line in enumerate(
                    (self.root / path).read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if query in line:
                        matches.append({"path": path, "line": line_number, "text": line[:240]})
            return True, bounded(canonical_json({"matches": matches[:40]})), None
        if tool == "edit":
            path, old, new = action["path"], action["old"], action["new"]
            if (
                not safe_relative_path(path)
                or path not in self.visible_files()
                or not old
                or len(new) > 4_000
            ):
                return False, "Edit rejected.", None
            source = (self.root / path).read_text(encoding="utf-8")
            if source.count(old) != 1:
                return False, "Edit target must occur exactly once.", None
            (self.root / path).write_text(source.replace(old, new, 1), encoding="utf-8")
            return True, "Edit applied.", None
        if tool == "test":
            passed, failures = self.verify()
            return (
                True,
                canonical_json({"passed": passed, "failed_checks": failures}),
                passed,
            )
        if tool == "finish":
            passed, failures = self.verify()
            self.terminal = True
            self.terminal_reason = "solved" if passed else "finished_with_failures"
            self.reward = round(1.0 - min(0.05, 0.002 * (len(self.steps) + 1)), 6) if passed else 0
            return (
                True,
                canonical_json({"passed": passed, "failed_checks": failures}),
                passed,
            )
        return False, "Unknown tool.", None

    def step(self, response: str) -> ExternalStep:
        if self.terminal:
            raise RuntimeError("the external task is already terminal")
        action = parse_action(response)
        accepted = False
        observation = "Malformed action."
        verifier_passed: bool | None = None
        if action is not None:
            accepted, observation, verifier_passed = self.execute(action)
        step = ExternalStep(
            index=len(self.steps),
            action=action,
            accepted=accepted,
            observation=observation,
            verifier_passed=verifier_passed,
        )
        self.steps.append(step)
        if len(self.steps) >= MAXIMUM_ACTIONS and not self.terminal:
            self.terminal = True
            self.terminal_reason = "action_limit"
        return step


def pack_manifest() -> dict[str, Any]:
    tasks = external_tasks()
    source_path = Path(__file__)
    return {
        "schema_version": 1,
        "pack_id": PACK_ID,
        "trainer_freeze_commit": TRAINER_FREEZE_COMMIT,
        "authored_at": AUTHORED_AT,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "task_count": len(tasks),
        "domains": {
            domain: sum(task.domain == domain for task in tasks)
            for domain in ("micro_repository", "sqlite_data_repair", "filesystem_cli")
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "description": task.description,
                "known_failing_tests": list(task.known_failing_tests),
                "verifier_revision": task.verifier_revision,
                "fixture_digest": hashlib.sha256(canonical_json(task.files).encode()).hexdigest(),
            }
            for task in tasks
        ],
        "candidate_execution": {
            "network": "not_available_to_action_protocol",
            "credentials": "not_available_to_action_protocol",
            "python": "restricted_ast_interpreter",
            "sqlite": "authorizer_denies_attach_detach_and_pragma",
            "cli": "fixed allowlisted argv with two-second timeout",
        },
    }


def apply_golden_edits(environment: ExternalTaskEnvironment) -> None:
    for edit in environment.task.golden_edits:
        environment.step(
            canonical_json(
                {
                    "tool": "edit",
                    "path": edit.path,
                    "old": edit.old,
                    "new": edit.new,
                }
            )
        )


if __name__ == "__main__":
    print(json.dumps(pack_manifest(), sort_keys=True, indent=2))
