import json
import os
import subprocess
import sys
from pathlib import Path


def run_remote_runner(
    tmp_path: Path,
    failure_mode: str,
    workload_file: str = "repository_repair_rl.py",
    *,
    missing_variables: tuple[str, ...] = (),
    environment_overrides: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        """#!/bin/sh
case "$1" in
  -)
    if [ "$EQUINOX_WORKLOAD_FILE" = "repository_repair_eligibility.py" ] &&
      grep -q '"screen_completed":[[:space:]]*true' "$2"; then
      exit 0
    fi
    if [ "$EQUINOX_WORKLOAD_FILE" != "repository_repair_eligibility.py" ] &&
      grep -q '"experiment_completed":[[:space:]]*true' "$2"; then
      exit 0
    fi
    exit 1
    ;;
  */result_server.py)
    : >"$EQUINOX_REMOTE_WORKDIR/result-server-started"
    if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-exit-cleared" ] &&
      [ -f "$EQUINOX_REMOTE_WORKDIR/exit_code" ]; then
      exit 88
    fi
    exit 0
    ;;
esac
attempt_path="$EQUINOX_REMOTE_WORKDIR/attempts"
attempt=0
if [ -f "$attempt_path" ]; then
  attempt="$(cat "$attempt_path")"
fi
attempt=$((attempt + 1))
printf '%s\\n' "$attempt" >"$attempt_path"
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-exit-cleared" ]; then
  printf '%s\\n' '{"experiment_completed":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-cublas-workspace" ] &&
  [ "$CUBLAS_WORKSPACE_CONFIG" != ":4096:8" ]; then
  printf '%s\n' 'missing deterministic cuBLAS workspace' >&2
  exit 89
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "eligibility-success" ]; then
  printf '%s\n' '{"screen_completed":true,"protocol_eligible":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "invalid-once" ] && [ "$attempt" -eq 1 ]; then
  mkdir -p "$EQUINOX_ADAPTER_PATH/checkpoints"
  printf '%s\\n' '{"checkpoint":"update-1"}' \
    >"$EQUINOX_ADAPTER_PATH/checkpoints/latest.json"
  printf '%s\\n' '{}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "no-checkpoint" ]; then
  printf '%s\\n' 'failure before checkpoint' >&2
  exit 7
fi
if [ "$attempt" -eq 1 ]; then
  mkdir -p "$EQUINOX_ADAPTER_PATH/checkpoints"
  printf '%s\\n' '{"checkpoint":"update-1"}' \
    >"$EQUINOX_ADAPTER_PATH/checkpoints/latest.json"
  if [ "$EQUINOX_FAKE_FAILURE_MODE" = "result-then-fail" ]; then
    printf '%s\\n' '{"experiment_completed":true}'
  fi
  printf '%s\\n' 'transient failure' >&2
  exit 7
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "always-fail" ]; then
  printf '%s\\n' 'second failure' >&2
  exit 9
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "invalid-result" ]; then
  printf '%s\\n' '{}'
  exit 0
fi
printf '%s\\n' '{"experiment_completed":true}'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_tar = fake_bin / "tar"
    fake_tar.write_text(
        """#!/bin/sh
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "archive-fail" ]; then
  exit 23
fi
exec /usr/bin/tar "$@"
""",
        encoding="utf-8",
    )
    fake_tar.chmod(0o755)
    (tmp_path / "result_server.py").write_text("", encoding="utf-8")
    (tmp_path / "repository_repair_rl.py").write_text("", encoding="utf-8")
    (tmp_path / "repository_repair_study.py").write_text("", encoding="utf-8")
    (tmp_path / "repository_repair_eligibility.py").write_text("", encoding="utf-8")
    (tmp_path / "branching_sequence_ladder.py").write_text("", encoding="utf-8")

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EQUINOX_FAKE_FAILURE_MODE": failure_mode,
        "EQUINOX_DURABILITY_PYTHON": sys.executable,
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_WORKLOAD_FILE": workload_file,
        "EQUINOX_RL_MODEL_ID": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "EQUINOX_RL_SEED": "107",
        "EQUINOX_RL_TARGET_SECONDS": "7200",
        "EQUINOX_RL_MAX_RESUME_GAP_SECONDS": "2700",
        "EQUINOX_RL_MAX_UPDATES": "120",
        "EQUINOX_RL_VALIDATION_EXAMPLES": "8",
        "EQUINOX_RL_TEST_EXAMPLES": "12",
        "EQUINOX_RL_MASTERY_WINDOWS": "2",
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": "2",
        "EQUINOX_RL_REPLAY_TASKS_PER_LEVEL": "1",
        "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS": "2400",
        "EQUINOX_STUDY_CONDITION": "k4_train",
        "EQUINOX_STUDY_VALIDATION_SEED_BASE": "20000000",
        "EQUINOX_STUDY_TEST_SEED_BASE": "50000000",
    }
    for variable in missing_variables:
        environment.pop(variable)
    environment.update(environment_overrides or {})
    repository_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["bash", str(repository_root / "research/runpod/remote_runner.sh")],
        check=check,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_remote_runner_refuses_to_start_without_result_transport_token(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        missing_variables=("EQUINOX_RESULT_TOKEN",),
        check=False,
    )

    assert completed.returncode == 78
    assert not (tmp_path / "attempts").exists()
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "78"
    assert "refusing to start" in (tmp_path / "error.log").read_text(encoding="utf-8")


def test_external_eval_remote_runner_refuses_unobservable_work(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "EQUINOX_REMOTE_WORKDIR": str(tmp_path),
        "EQUINOX_WORKLOAD_FILE": "research/runpod/revision30_external_eval.py",
    }
    environment.pop("EQUINOX_RESULT_TOKEN", None)

    completed = subprocess.run(
        [
            "bash",
            str(repository_root / "research/runpod/external_eval_remote_runner.sh"),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 78
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "78"
    assert "unobservable evaluation" in (tmp_path / "error.log").read_text(encoding="utf-8")


def test_remote_runner_serves_progress_serialization_failure(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        environment_overrides={"EQUINOX_DURABILITY_PYTHON": "/bin/false"},
        check=False,
    )

    assert completed.returncode == 70
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "70"
    assert "serialization failed" in (tmp_path / "error.log").read_text(encoding="utf-8")
    assert (tmp_path / "result-server-started").is_file()


def test_remote_runner_retries_once_from_a_persisted_checkpoint(tmp_path: Path) -> None:
    completed = run_remote_runner(tmp_path, "retry-success")

    assert completed.stderr == ""
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "0"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }
    assert (tmp_path / "adapter.tgz").is_file()
    assert not (tmp_path / "adapter.tgz.pending").exists()
    assert not (tmp_path / "adapter" / "checkpoints").exists()
    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "=== attempt 1 · exit 7 ===" in error_log
    assert "transient failure" in error_log
    assert "=== attempt 2 · exit 0 ===" in error_log


def test_remote_runner_supplies_deterministic_cublas_workspace(tmp_path: Path) -> None:
    run_remote_runner(tmp_path, "assert-cublas-workspace")

    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }


def test_remote_runner_does_not_retry_without_a_checkpoint(tmp_path: Path) -> None:
    run_remote_runner(tmp_path, "no-checkpoint")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "7"
    assert not (tmp_path / "result.json").exists()


def test_remote_runner_stops_when_attempt_fence_cannot_be_persisted(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        environment_overrides={"EQUINOX_ATTEMPT_COUNTER_PYTHON": "/bin/false"},
    )

    assert not (tmp_path / "attempts").exists()
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "WORKLOAD_ATTEMPT_PERSISTENCE_FAILED"


def test_remote_runner_stops_after_one_failed_retry(tmp_path: Path) -> None:
    run_remote_runner(tmp_path, "always-fail")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "9"
    assert not (tmp_path / "result.json").exists()
    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "=== attempt 1 · exit 7 ===" in error_log
    assert "=== attempt 2 · exit 9 ===" in error_log


def test_remote_runner_recovers_stderr_from_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "error.attempt-2.log").write_text(
        "CUDA out of memory during backward pass\n",
        encoding="utf-8",
    )

    run_remote_runner(tmp_path, "always-fail")

    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "=== attempt 2 · interrupted before exit ===" in error_log
    assert "CUDA out of memory during backward pass" in error_log
    assert not (tmp_path / "error.attempt-2.log").exists()


def test_remote_runner_deduplicates_stderr_after_atomic_merge_before_unlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "error.log").write_text(
        "=== attempt 2 · exit 137 ===\nCUDA out of memory\n\n",
        encoding="utf-8",
    )
    (tmp_path / "error.attempt-2.log").write_text(
        "CUDA out of memory\n",
        encoding="utf-8",
    )

    run_remote_runner(tmp_path, "always-fail")

    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert error_log.count("CUDA out of memory") == 1
    assert "interrupted before exit" not in error_log
    assert not (tmp_path / "error.attempt-2.log").exists()


def test_remote_runner_falls_back_when_durable_error_merge_fails(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        environment_overrides={"EQUINOX_ERROR_MERGE_PYTHON": "/bin/false"},
    )

    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "attempt 1 · exit 7 · non-durable fallback" in error_log
    assert "transient failure" in error_log
    assert "attempt 2 · exit 0 · non-durable fallback" in error_log
    assert not list(tmp_path.glob("error.attempt-*.log"))


def test_remote_runner_preserves_specific_failed_progress_after_budget_exhaustion(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    specific_progress = {
        "schema_version": 2,
        "phase": "failed",
        "message": "CUDA allocator failed after update 19.",
        "attempt": 2,
        "error": "RuntimeError: CUDA out of memory",
        "update": 19,
    }
    (tmp_path / "progress.json").write_text(
        json.dumps(specific_progress),
        encoding="utf-8",
    )

    run_remote_runner(tmp_path, "always-fail")

    observed = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert observed == specific_progress


def test_remote_runner_preserves_last_known_progress_when_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "training",
                "message": "Collecting K=4 branch group.",
                "attempt": 2,
                "update": 19,
                "current_level": 2,
                "elapsed_seconds": 3100.5,
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(tmp_path, "always-fail")

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "failed"
    assert progress["error"] == "WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED"
    assert progress["update"] == 19
    assert progress["current_level"] == 2
    assert progress["elapsed_seconds"] == 3100.5


def test_remote_runner_uses_the_sequence_workload_progress_schema(tmp_path: Path) -> None:
    run_remote_runner(
        tmp_path,
        "no-checkpoint",
        workload_file="branching_sequence_ladder.py",
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["schema_version"] == 1


def test_remote_runner_requires_common_sequence_configuration(tmp_path: Path) -> None:
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="branching_sequence_ladder.py",
        missing_variables=("EQUINOX_RL_TARGET_SECONDS",),
        check=False,
    )

    assert completed.returncode == 1
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["schema_version"] == 1
    assert progress["error"] == "WORKLOAD_CONFIGURATION_MISSING"


def test_remote_runner_does_not_reset_attempt_budget_after_restart(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "no-checkpoint")
    run_remote_runner(tmp_path, "no-checkpoint")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["attempt"] == 1
    assert progress["error"] == "WORKLOAD_FAILED_BEFORE_FIRST_CHECKPOINT"


def test_remote_runner_resumes_with_the_true_second_attempt_after_restart(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "adapter" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "latest.json").write_text(
        '{"checkpoint":"update-1"}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")
    (tmp_path / "exit_code").write_text("7\n", encoding="utf-8")

    run_remote_runner(tmp_path, "always-fail")

    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "2"
    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "=== attempt 2 · exit 7 ===" in error_log


def test_remote_runner_clears_stale_exit_code_before_resumed_work(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "adapter" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "latest.json").write_text(
        '{"checkpoint":"update-1"}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")
    (tmp_path / "exit_code").write_text("74\n", encoding="utf-8")

    run_remote_runner(tmp_path, "assert-exit-cleared")

    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "0"


def test_remote_runner_recovers_a_completed_pending_result_after_restart(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.pending.json").write_text(
        '{"experiment_completed":true}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")

    run_remote_runner(tmp_path, "always-fail")

    assert not (tmp_path / "attempts").exists()
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    assert progress["attempt"] == 1


def test_remote_runner_marks_an_already_published_result_complete_after_restart(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        '{"experiment_completed":true}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        '{"phase":"finalizing"}\n',
        encoding="utf-8",
    )

    run_remote_runner(tmp_path, "always-fail")

    assert not (tmp_path / "attempts").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    assert progress["attempt"] == 1


def test_remote_runner_promotes_a_completed_result_before_in_process_retry(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "result-then-fail")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"


def test_remote_runner_keeps_result_private_and_checkpoint_resumable_if_archive_fails(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "archive-fail")

    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "result.pending.json").is_file()
    assert (tmp_path / "adapter" / "checkpoints" / "latest.json").is_file()
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "ARTIFACT_ARCHIVE_FAILED"


def test_remote_runner_reports_missing_repository_configuration_structurally(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        missing_variables=("EQUINOX_RL_TARGET_SECONDS",),
        check=False,
    )

    assert completed.returncode == 1
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "failed"
    assert progress["error"] == "WORKLOAD_CONFIGURATION_MISSING"
    assert progress["attempt"] == 2


def test_remote_runner_accepts_study_workload_and_publishes_k1(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_study.py",
        environment_overrides={
            "EQUINOX_STUDY_CONDITION": "k1_train",
            "EQUINOX_STUDY_COMPLETION_BUDGET": "440",
        },
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["branch_width"] == 1


def test_remote_runner_accepts_eligibility_result_without_adapter(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "eligibility-success",
        workload_file="repository_repair_eligibility.py",
    )

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result == {"screen_completed": True, "protocol_eligible": True}
    assert not (tmp_path / "adapter.tgz").exists()


def test_remote_runner_rejects_study_without_matched_completion_budget(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_study.py",
        environment_overrides={"EQUINOX_STUDY_CONDITION": "k4_no_update"},
        check=False,
    )

    assert completed.returncode == 1
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "STUDY_CONFIGURATION_MISSING"


def test_remote_runner_preserves_progress_when_restart_configuration_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "training",
                "message": "Training update 19.",
                "attempt": 2,
                "update": 19,
                "current_level": 2,
                "elapsed_seconds": 3100.5,
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "retry-success",
        missing_variables=("EQUINOX_RL_TARGET_SECONDS",),
        check=False,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "WORKLOAD_CONFIGURATION_MISSING"
    assert progress["update"] == 19
    assert progress["current_level"] == 2
    assert progress["elapsed_seconds"] == 3100.5


def test_remote_runner_rejects_an_invalid_success_result(tmp_path: Path) -> None:
    run_remote_runner(tmp_path, "invalid-result")

    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "65"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "WORKLOAD_RESULT_INVALID"


def test_remote_runner_retries_an_invalid_result_from_a_checkpoint(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "invalid-once")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "0"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }
