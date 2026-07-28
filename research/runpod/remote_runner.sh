#!/usr/bin/env bash
set -uo pipefail

workload_file="${EQUINOX_WORKLOAD_FILE:-}"
target_runtime_seconds="${EQUINOX_RL_TARGET_SECONDS:-}"
model_id="${EQUINOX_RL_MODEL_ID:-}"
optimization_seed="${EQUINOX_RL_SEED:-}"
maximum_updates="${EQUINOX_RL_MAX_UPDATES:-}"
maximum_resume_gap_seconds="${EQUINOX_RL_MAX_RESUME_GAP_SECONDS:-}"
validation_examples="${EQUINOX_RL_VALIDATION_EXAMPLES:-}"
test_examples="${EQUINOX_RL_TEST_EXAMPLES:-}"
mastery_windows="${EQUINOX_RL_MASTERY_WINDOWS:-}"
training_tasks_per_update="${EQUINOX_RL_TRAINING_TASKS_PER_UPDATE:-}"
replay_tasks_per_level="${EQUINOX_RL_REPLAY_TASKS_PER_LEVEL:-}"
maximum_final_evaluation_reserve_seconds="${EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS:-}"
study_condition="${EQUINOX_STUDY_CONDITION:-}"
study_validation_seed_base="${EQUINOX_STUDY_VALIDATION_SEED_BASE:-}"
study_test_seed_base="${EQUINOX_STUDY_TEST_SEED_BASE:-}"
study_completion_budget="${EQUINOX_STUDY_COMPLETION_BUDGET:-}"
branch_width=4
if [[ "$study_condition" == "k1_train" ]]; then
  branch_width=1
fi
work_directory="${EQUINOX_REMOTE_WORKDIR:-/tmp}"
progress_path="$work_directory/progress.json"
result_pending_path="$work_directory/result.pending.json"
result_path="$work_directory/result.json"
error_path="$work_directory/error.log"
exit_code_path="$work_directory/exit_code"
adapter_path="$work_directory/adapter"
attempt_counter_path="$work_directory/workload-attempt-count"
boot_attempt=1
if [[ -z "${EQUINOX_RESULT_TOKEN:-}" ]]; then
  printf '%s\n' \
    "EQUINOX_RESULT_TOKEN is missing; refusing to start an unobservable workload." \
    >"$error_path"
  printf '%s\n' 78 >"$exit_code_path"
  exit 78
fi
if [[ -f "$attempt_counter_path" ]]; then
  boot_attempt_candidate="$(<"$attempt_counter_path")"
  if [[ "$boot_attempt_candidate" =~ ^[12]$ ]]; then
    boot_attempt="$boot_attempt_candidate"
  fi
fi

serve_transport_failure_and_exit() {
  local failure_exit_code="$1"
  if [[ -z "${http_server_pid:-}" ]]; then
    python3 "$work_directory/result_server.py" \
      >"$work_directory/http.log" 2>&1 &
    http_server_pid="$!"
  fi
  wait "$http_server_pid"
  exit "$failure_exit_code"
}

merge_attempt_error_log() {
  local attempt_error_path="$1"
  local attempt="$2"
  local outcome="$3"
  "${EQUINOX_ERROR_MERGE_PYTHON:-${EQUINOX_DURABILITY_PYTHON:-python3}}" - \
    "$error_path" \
    "$attempt_error_path" \
    "$attempt" \
    "$outcome" <<'PY'
import os
import sys

error_path, attempt_path, attempt, outcome = sys.argv[1:]
marker = f"=== attempt {attempt} · {outcome} ===\n".encode()
attempt_prefix = f"=== attempt {attempt} ".encode()
try:
    with open(error_path, "rb") as handle:
        existing = handle.read()
except FileNotFoundError:
    existing = b""
if attempt_prefix not in existing:
    with open(attempt_path, "rb") as handle:
        attempt_error = handle.read()
    pending_path = f"{error_path}.pending"
    with open(pending_path, "wb") as handle:
        handle.write(existing)
        handle.write(marker)
        handle.write(attempt_error)
        if attempt_error and not attempt_error.endswith(b"\n"):
            handle.write(b"\n")
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending_path, error_path)
    directory_fd = os.open(os.path.dirname(error_path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
os.unlink(attempt_path)
PY
}

merge_attempt_error_log_or_fallback() {
  local attempt_error_path="$1"
  local attempt="$2"
  local outcome="$3"
  if merge_attempt_error_log "$attempt_error_path" "$attempt" "$outcome"; then
    return
  fi
  {
    printf '=== attempt %s · %s · non-durable fallback ===\n' "$attempt" "$outcome"
    cat "$attempt_error_path"
    printf '\n'
  } >>"$error_path"
  rm -f -- "$attempt_error_path"
}

write_progress() {
  local schema_version="$1"
  local phase="$2"
  local message="$3"
  local attempt="$4"
  local error_code="${5:-}"
  local preserve_context="${6:-false}"
  local pending_path="$progress_path.pending"
  if ! "${EQUINOX_DURABILITY_PYTHON:-python3}" - \
    "$pending_path" \
    "$schema_version" \
    "$phase" \
    "$message" \
    "$attempt" \
    "$error_code" \
    "$preserve_context" \
    "$progress_path" \
    "$branch_width" <<'PY'
import json
import os
import sys

(
    path,
    schema_version,
    phase,
    message,
    attempt,
    error_code,
    preserve_context,
    live_path,
    branch_width,
) = sys.argv[1:]
payload = {
    "schema_version": int(schema_version),
    "phase": phase,
    "message": message,
    "branch_width": int(branch_width),
    "complexity_strategy": "adaptive",
    "attempt": int(attempt),
    "error": error_code or None,
}
if preserve_context == "true" and os.path.isfile(live_path):
    try:
        with open(live_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        previous = {}
    if isinstance(previous, dict):
        for key in (
            "update",
            "current_level",
            "maximum_level",
            "maximum_updates",
            "elapsed_seconds",
        ):
            if key in previous:
                payload[key] = previous[key]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  then
    printf '%s\n' "Structured progress serialization failed." >>"$error_path"
    printf '%s\n' 70 >"$exit_code_path"
    serve_transport_failure_and_exit 70
  fi
  if ! mv "$pending_path" "$progress_path"; then
    printf '%s\n' "Structured progress publication failed." >>"$error_path"
    printf '%s\n' 70 >"$exit_code_path"
    serve_transport_failure_and_exit 70
  fi
}

serve_boot_failure() {
  local schema_version="$1"
  local error_code="$2"
  local failure_message="$3"
  write_progress \
    "$schema_version" \
    "failed" \
    "$failure_message" \
    "$boot_attempt" \
    "$error_code" \
    "true"
  printf '%s\n' 1 >"$exit_code_path"
  python3 "$work_directory/result_server.py" \
    >"$work_directory/http.log" 2>&1 &
  wait "$!"
  exit 1
}

case "$workload_file" in
  repository_repair_rl.py | repository_repair_study.py)
    progress_schema_version=2
    repository_workload=true
    ;;
  branching_sequence_ladder.py)
    progress_schema_version=1
    repository_workload=false
    ;;
  *)
    serve_boot_failure \
      1 \
      "UNSUPPORTED_WORKLOAD_FILE" \
      "The remote runner received an unsupported workload file."
    ;;
esac

if [[ "$repository_workload" == "true" && -z "$model_id" ]]; then
  serve_boot_failure \
    "$progress_schema_version" \
    "MODEL_ID_MISSING" \
    "Repository repair requires a model ID."
fi

for common_configuration_value in \
  "$target_runtime_seconds" \
  "$optimization_seed"; do
  if [[ -z "$common_configuration_value" ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "WORKLOAD_CONFIGURATION_MISSING" \
      "The workload is missing required runtime configuration."
  fi
done

if [[ "$repository_workload" == "true" ]]; then
  for configuration_value in \
    "$maximum_updates" \
    "$maximum_resume_gap_seconds" \
    "$validation_examples" \
    "$test_examples" \
    "$mastery_windows" \
    "$training_tasks_per_update" \
    "$replay_tasks_per_level" \
    "$maximum_final_evaluation_reserve_seconds"; do
    if [[ -z "$configuration_value" ]]; then
      serve_boot_failure \
        "$progress_schema_version" \
        "WORKLOAD_CONFIGURATION_MISSING" \
        "Repository repair is missing required workload configuration."
    fi
  done
fi
if [[ "$workload_file" == "repository_repair_study.py" ]]; then
  for study_configuration_value in \
    "$study_condition" \
    "$study_validation_seed_base" \
    "$study_test_seed_base"; do
    if [[ -z "$study_configuration_value" ]]; then
      serve_boot_failure \
        "$progress_schema_version" \
        "STUDY_CONFIGURATION_MISSING" \
        "The confirmatory study is missing its condition or fresh split seeds."
    fi
  done
  if [[ "$study_condition" != "k4_train" && -z "$study_completion_budget" ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "STUDY_CONFIGURATION_MISSING" \
      "The matched-compute study condition is missing its completion budget."
  fi
fi

touch "$error_path"
for interrupted_error_path in "$work_directory"/error.attempt-*.log; do
  if [[ ! -f "$interrupted_error_path" ]]; then
    continue
  fi
  interrupted_attempt="${interrupted_error_path##*/error.attempt-}"
  interrupted_attempt="${interrupted_attempt%.log}"
  if [[ "$interrupted_attempt" =~ ^[0-9]+$ ]]; then
    merge_attempt_error_log_or_fallback \
      "$interrupted_error_path" \
      "$interrupted_attempt" \
      "interrupted before exit"
  fi
done

recorded_attempts=0
if [[ -f "$attempt_counter_path" ]]; then
  recorded_attempts="$(<"$attempt_counter_path")"
  if [[ ! "$recorded_attempts" =~ ^[0-2]$ ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "WORKLOAD_ATTEMPT_STATE_INVALID" \
      "The persisted workload attempt state is invalid."
  fi
fi

completed_pending_result_available() {
  if [[ ! -f "$result_pending_path" || -f "$result_path" ]]; then
    return 1
  fi
  if python3 - "$result_pending_path" "$workload_file" 2>/dev/null <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if not isinstance(result, dict):
    raise SystemExit(1)
if (
    sys.argv[2] in {"repository_repair_rl.py", "repository_repair_study.py"}
    and result.get("experiment_completed") is not True
):
    raise SystemExit(1)
if (
    sys.argv[2] == "branching_sequence_ladder.py"
    and result.get("workload") != "branching-sequence-policy-gradient-ladder"
):
    raise SystemExit(1)
PY
  then
    return 0
  fi
  return 1
}

write_recovered_result_progress() {
  local phase="$1"
  local recovery_message="$2"
  local recovery_attempt="$recorded_attempts"
  if ((recovery_attempt < 1)); then
    recovery_attempt=1
  fi
  write_progress \
    "$progress_schema_version" \
    "$phase" \
    "$recovery_message" \
    "$recovery_attempt"
}

pending_result_completed=false
pending_result_recovered=false
if completed_pending_result_available; then
  pending_result_completed=true
  pending_result_recovered=true
  write_recovered_result_progress \
    "finalizing" \
    "Recovered a completed result; packaging artifacts."
elif [[ -f "$result_path" ]]; then
  write_recovered_result_progress \
    "complete" \
    "Previously published result and artifacts are available."
fi

retry_allowed=true
terminal_error=""
terminal_message=""
if [[ "$pending_result_completed" == "true" || -f "$result_path" ]]; then
  retry_allowed=false
elif ((recorded_attempts >= 2)); then
  retry_allowed=false
  terminal_error="WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED"
  terminal_message="The bounded workload attempt budget is exhausted."
elif ((recorded_attempts > 0)) &&
  [[ ! -f "$adapter_path/checkpoints/latest.json" ]]; then
  retry_allowed=false
  terminal_error="WORKLOAD_FAILED_BEFORE_FIRST_CHECKPOINT"
  terminal_message="The workload failed before producing a resumable checkpoint."
fi

if [[ "$retry_allowed" == "true" || "$pending_result_completed" == "true" ]]; then
  rm -f -- "$exit_code_path"
fi

if [[ ! -f "$result_path" && "$pending_result_completed" != "true" ]]; then
  if [[ "$retry_allowed" == "true" ]]; then
    next_attempt="$((recorded_attempts + 1))"
    if ((recorded_attempts > 0)); then
      progress_phase="resuming"
      progress_message="Retrying from the latest checkpoint after a workload failure."
    else
      progress_phase="container_starting"
      progress_message="RunPod container started; preparing the workload."
    fi
    write_progress \
      "$progress_schema_version" \
      "$progress_phase" \
      "$progress_message" \
      "$next_attempt"
  else
    failed_attempt="$recorded_attempts"
    if ((failed_attempt < 1)); then
      failed_attempt=1
    fi
    if ! grep -Eq '"phase"[[:space:]]*:[[:space:]]*"failed"' \
      "$progress_path" 2>/dev/null; then
      write_progress \
        "$progress_schema_version" \
        "failed" \
        "$terminal_message" \
        "$failed_attempt" \
        "$terminal_error" \
        "true"
    fi
  fi
fi
python3 "$work_directory/result_server.py" \
  >"$work_directory/http.log" 2>&1 &
http_server_pid="$!"

workload_exit_code=1
if [[ "$pending_result_completed" == "true" || -f "$result_path" ]]; then
  workload_exit_code=0
elif [[ -f "$exit_code_path" ]]; then
  prior_exit_code="$(<"$exit_code_path")"
  if [[ "$prior_exit_code" =~ ^[0-9]+$ ]]; then
    workload_exit_code="$prior_exit_code"
  fi
fi
while [[ ! -f "$result_path" && "$retry_allowed" == "true" ]] &&
  ((recorded_attempts < 2)); do
  if completed_pending_result_available; then
    workload_exit_code=0
    pending_result_completed=true
    pending_result_recovered=true
    write_recovered_result_progress \
      "finalizing" \
      "Recovered a completed result; packaging artifacts."
    break
  fi
  workload_attempt="$((recorded_attempts + 1))"
  printf '%s\n' "$workload_attempt" >"$attempt_counter_path.pending"
  if ! "${EQUINOX_ATTEMPT_COUNTER_PYTHON:-${EQUINOX_DURABILITY_PYTHON:-python3}}" - \
    "$attempt_counter_path.pending" \
    "$attempt_counter_path" <<'PY'
import os
import sys

pending_path = os.path.abspath(sys.argv[1])
live_path = os.path.abspath(sys.argv[2])
with open(pending_path, "rb") as handle:
    os.fsync(handle.fileno())
os.replace(pending_path, live_path)
directory_fd = os.open(os.path.dirname(live_path), os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
  then
    workload_exit_code=74
    retry_allowed=false
    terminal_error="WORKLOAD_ATTEMPT_PERSISTENCE_FAILED"
    terminal_message="The workload attempt fence could not be persisted."
    write_progress \
      "$progress_schema_version" \
      "failed" \
      "$terminal_message" \
      "$workload_attempt" \
      "$terminal_error"
    break
  fi
  recorded_attempts="$workload_attempt"
  attempt_error_path="$work_directory/error.attempt-$workload_attempt.log"
  EQUINOX_PROGRESS_PATH="$progress_path" \
    EQUINOX_ADAPTER_PATH="$adapter_path" \
    EQUINOX_RL_MODEL_ID="$model_id" \
    EQUINOX_RL_SEED="$optimization_seed" \
    EQUINOX_RL_TARGET_SECONDS="$target_runtime_seconds" \
    EQUINOX_RL_MAX_UPDATES="$maximum_updates" \
    EQUINOX_RL_MAX_RESUME_GAP_SECONDS="$maximum_resume_gap_seconds" \
    EQUINOX_RL_VALIDATION_EXAMPLES="$validation_examples" \
    EQUINOX_RL_TEST_EXAMPLES="$test_examples" \
    EQUINOX_RL_MASTERY_WINDOWS="$mastery_windows" \
    EQUINOX_RL_TRAINING_TASKS_PER_UPDATE="$training_tasks_per_update" \
    EQUINOX_RL_REPLAY_TASKS_PER_LEVEL="$replay_tasks_per_level" \
    EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS="$maximum_final_evaluation_reserve_seconds" \
    EQUINOX_STUDY_CONDITION="$study_condition" \
    EQUINOX_STUDY_VALIDATION_SEED_BASE="$study_validation_seed_base" \
    EQUINOX_STUDY_TEST_SEED_BASE="$study_test_seed_base" \
    EQUINOX_STUDY_COMPLETION_BUDGET="$study_completion_budget" \
    EQUINOX_WORKLOAD_ATTEMPT="$workload_attempt" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python3 "$work_directory/$workload_file" \
    >"$result_pending_path" 2>"$attempt_error_path"
  workload_exit_code="$?"
  merge_attempt_error_log_or_fallback \
    "$attempt_error_path" \
    "$workload_attempt" \
    "exit $workload_exit_code"
  if [[ "$workload_exit_code" == "0" ]]; then
    if completed_pending_result_available; then
      pending_result_completed=true
      break
    else
      workload_exit_code=65
      terminal_error="WORKLOAD_RESULT_INVALID"
      terminal_message="The workload exited successfully without a valid result."
    fi
  fi
  if completed_pending_result_available; then
    workload_exit_code=0
    pending_result_completed=true
    pending_result_recovered=true
    write_recovered_result_progress \
      "finalizing" \
      "Recovered a completed result; packaging artifacts."
    break
  fi
  if ((workload_attempt >= 2)) ||
    [[ ! -f "$adapter_path/checkpoints/latest.json" ]]; then
    break
  fi
  write_progress \
    "$progress_schema_version" \
    "resuming" \
    "Retrying from the latest checkpoint after a workload failure." \
    "$((workload_attempt + 1))"
  terminal_error=""
  terminal_message=""
done

if [[ "$workload_exit_code" != "0" && ! -f "$result_path" ]] &&
  ! grep -Eq '"phase"[[:space:]]*:[[:space:]]*"failed"' \
    "$progress_path" 2>/dev/null; then
  if [[ -n "$terminal_error" ]]; then
    :
  elif [[ ! -f "$adapter_path/checkpoints/latest.json" ]]; then
    terminal_error="WORKLOAD_FAILED_BEFORE_FIRST_CHECKPOINT"
    terminal_message="The workload failed before producing a resumable checkpoint."
  elif ((recorded_attempts >= 2)); then
    terminal_error="WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED"
    terminal_message="The bounded workload attempt budget is exhausted."
  else
    terminal_error="WORKLOAD_FAILED"
    terminal_message="The workload failed."
  fi
  write_progress \
    "$progress_schema_version" \
    "failed" \
    "$terminal_message" \
    "$recorded_attempts" \
    "$terminal_error"
fi

if [[ "$workload_exit_code" == "0" ]] &&
  [[ "$pending_result_completed" == "true" || -f "$result_path" ]]; then
  if [[ -d "$adapter_path" && ! -f "$work_directory/adapter.tgz" ]]; then
    archive_pending_path="$work_directory/adapter.tgz.pending"
    if tar \
      --exclude="adapter/checkpoints" \
      -czf "$archive_pending_path" \
      -C "$work_directory" \
      adapter &&
      mv "$archive_pending_path" "$work_directory/adapter.tgz"; then
      :
    else
      workload_exit_code=74
      terminal_error="ARTIFACT_ARCHIVE_FAILED"
      terminal_message="The trained adapter could not be packaged safely."
      write_progress \
        "$progress_schema_version" \
        "failed" \
        "$terminal_message" \
        "$recorded_attempts" \
        "$terminal_error"
    fi
  fi
  if [[ "$workload_exit_code" == "0" && ! -f "$result_path" ]]; then
    if ! mv "$result_pending_path" "$result_path"; then
      workload_exit_code=74
      terminal_error="RESULT_PUBLICATION_FAILED"
      terminal_message="The completed workload result could not be published safely."
      write_progress \
        "$progress_schema_version" \
        "failed" \
        "$terminal_message" \
        "$recorded_attempts" \
        "$terminal_error"
    fi
  fi
  if [[ "$workload_exit_code" == "0" && -d "$adapter_path/checkpoints" ]]; then
    rm -rf -- "$adapter_path/checkpoints"
  fi
  if [[ "$workload_exit_code" == "0" && "$pending_result_recovered" == "true" ]]; then
    write_recovered_result_progress \
      "complete" \
      "Recovered result and artifacts published."
  fi
fi
printf '%s\n' "$workload_exit_code" >"$exit_code_path"

wait "$http_server_pid"
