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
maximum_workload_attempts="${EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS:-2}"
branch_width=4
if [[ "$study_condition" == "k1_train" || "$study_condition" == k1_* ]]; then
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
non_retryable_failure_path="$work_directory/non-retryable-failure"
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
if error_code:
    payload["remote_error"] = {
        "code": error_code,
        "message": message,
    }
bundle_digest = os.environ.get("EQUINOX_BUNDLE_SHA256", "")
if bundle_digest:
    try:
        bundle_size_bytes = int(os.environ["EQUINOX_BUNDLE_SIZE_BYTES"])
    except (KeyError, ValueError):
        bundle_size_bytes = None
    payload.update(
        {
            "bundle_handoff_revision": os.environ.get(
                "EQUINOX_BUNDLE_HANDOFF_REVISION"
            ),
            "workload_bundle_digest": bundle_digest,
            "workload_bundle_size_bytes": bundle_size_bytes,
            "workload_bundle_path": os.environ.get("EQUINOX_BUNDLE_VOLUME_PATH"),
            "bundle_stage_receipt_digest": os.environ.get(
                "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256"
            ),
            "bootstrap_source_digest": os.environ.get(
                "EQUINOX_BOOTSTRAP_SOURCE_SHA256"
            ),
            "network_volume_id": os.environ.get("EQUINOX_RUNPOD_NETWORK_VOLUME_ID"),
        }
    )
if preserve_context == "true" and os.path.isfile(live_path):
    try:
        with open(live_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        previous = {}
    if isinstance(previous, dict):
        payload = {**previous, **payload}
if phase != "failed":
    payload.pop("error", None)
    payload.pop("remote_error", None)
    payload.pop("operator_error", None)
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

structure_remote_failure() {
  local failure_exit_code="$1"
  if ! "${EQUINOX_DURABILITY_PYTHON:-python3}" - \
    "$progress_path" \
    "$failure_exit_code" <<'PY'
import json
import os
import re
import sys

path, exit_code = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict) or payload.get("phase") != "failed":
    raise SystemExit(0)
remote_error = payload.get("remote_error")
valid_remote_error = (
    isinstance(remote_error, dict)
    and isinstance(remote_error.get("code"), str)
    and re.fullmatch(r"[A-Z][A-Z0-9_]*", remote_error["code"])
    and isinstance(remote_error.get("message"), str)
    and bool(remote_error["message"])
)
if not valid_remote_error:
    malformed_remote_message = (
        remote_error.get("message")
        if isinstance(remote_error, dict)
        and isinstance(remote_error.get("message"), str)
        and remote_error["message"]
        else None
    )
    raw_error = payload.get("error")
    payload_message = payload.get("message")
    if not isinstance(payload_message, str) or not payload_message:
        payload_message = "The remote workload failed."
    stable_code = (
        raw_error
        if isinstance(raw_error, str)
        and re.fullmatch(r"[A-Z][A-Z0-9_]*", raw_error)
        else "REMOTE_WORKLOAD_FAILURE"
    )
    if stable_code == raw_error:
        exact_message = payload_message
    else:
        exact_message = (
            raw_error
            if isinstance(raw_error, str) and raw_error
            else malformed_remote_message
            or payload_message
        )
    remote_error = {
        "code": stable_code,
        "message": exact_message,
    }
else:
    remote_error = dict(remote_error)
remote_error["exit_code"] = int(exit_code)
payload["remote_error"] = remote_error
pending_path = f"{path}.pending"
with open(pending_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(pending_path, path)
directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
  then
    printf '%s\n' "Structured remote failure enrichment failed." >>"$error_path"
    return 1
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
  repository_repair_rl.py | repository_repair_study.py | repository_repair_study_v31.py | repository_repair_eligibility.py | repository_repair_large_model_eligibility.py | repository_repair_large_model_pilot.py)
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

if [[ ! "$maximum_workload_attempts" =~ ^[12]$ ]]; then
  serve_boot_failure \
    "$progress_schema_version" \
    "WORKLOAD_ATTEMPT_LIMIT_INVALID" \
    "The workload attempt limit must be 1 or 2."
fi

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
if [[ "$workload_file" == "repository_repair_study.py" ||
  "$workload_file" == "repository_repair_study_v31.py" ||
  "$workload_file" == "repository_repair_eligibility.py" ]]; then
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
  if [[ "$workload_file" == "repository_repair_study_v31.py" &&
    -z "$study_completion_budget" ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "STUDY_CONFIGURATION_MISSING" \
      "The revision-31 condition is missing its fixed completion budget."
  fi
  if [[ "$workload_file" != "repository_repair_study_v31.py" &&
    "$study_condition" != "k4_train" &&
    -z "$study_completion_budget" ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "STUDY_CONFIGURATION_MISSING" \
      "The matched-compute study condition is missing its completion budget."
  fi
fi

non_retryable_failure_code() {
  local attempt_error_path="$1"
  if grep -Eiq \
    'CUDA([^[:cntrl:]]*)out of memory|CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED' \
    "$attempt_error_path"; then
    printf '%s\n' "GPU_MEMORY_EXHAUSTED"
    return 0
  fi
  if grep -Eiq \
    'MODEL_LOAD_FAILED|MODEL_LOAD_TIMEOUT|model load(ing)? failed|failed to load (the )?model|RevisionNotFoundError|RepositoryNotFoundError|GatedRepoError|not a valid model identifier|safetensors(_rust)?[^[:cntrl:]]*(error|exception)' \
    "$attempt_error_path"; then
    printf '%s\n' "MODEL_LOAD_FAILED"
    return 0
  fi
  return 1
}

persist_non_retryable_failure() {
  local failure_code="$1"
  printf '%s\n' "$failure_code" >"$non_retryable_failure_path.pending"
  mv "$non_retryable_failure_path.pending" "$non_retryable_failure_path"
}

non_retryable_failure_message() {
  case "$1" in
    GPU_MEMORY_EXHAUSTED)
      printf '%s\n' \
        "The workload exhausted GPU memory; retrying unchanged would waste compute."
      ;;
    MODEL_LOAD_FAILED)
      printf '%s\n' \
        "The configured model could not be loaded; retrying unchanged would waste compute."
      ;;
    *)
      printf '%s\n' "The workload encountered a non-retryable failure."
      ;;
  esac
}

touch "$error_path"
for interrupted_error_path in "$work_directory"/error.attempt-*.log; do
  if [[ ! -f "$interrupted_error_path" ]]; then
    continue
  fi
  interrupted_attempt="${interrupted_error_path##*/error.attempt-}"
  interrupted_attempt="${interrupted_attempt%.log}"
  if [[ "$interrupted_attempt" =~ ^[0-9]+$ ]]; then
    interrupted_non_retryable_failure=""
    if interrupted_non_retryable_failure="$(
      non_retryable_failure_code "$interrupted_error_path"
    )"; then
      persist_non_retryable_failure "$interrupted_non_retryable_failure"
    fi
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
if sys.argv[2] in {
    "repository_repair_rl.py",
    "repository_repair_study.py",
    "repository_repair_study_v31.py",
    "repository_repair_large_model_pilot.py",
}:
    if result.get("experiment_completed") is not True:
        raise SystemExit(1)
if sys.argv[2] in {
    "repository_repair_eligibility.py",
    "repository_repair_large_model_eligibility.py",
}:
    if result.get("screen_completed") is not True:
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
    "$recovery_attempt" \
    "" \
    "true"
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
elif [[ -f "$non_retryable_failure_path" ]]; then
  persisted_non_retryable_failure="$(<"$non_retryable_failure_path")"
  case "$persisted_non_retryable_failure" in
    GPU_MEMORY_EXHAUSTED | MODEL_LOAD_FAILED)
      retry_allowed=false
      terminal_error="$persisted_non_retryable_failure"
      terminal_message="$(
        non_retryable_failure_message "$persisted_non_retryable_failure"
      )"
      ;;
    *)
      retry_allowed=false
      terminal_error="WORKLOAD_FAILURE_STATE_INVALID"
      terminal_message="The persisted non-retryable failure state is invalid."
      ;;
  esac
elif ((recorded_attempts >= maximum_workload_attempts)); then
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
      "$next_attempt" \
      "" \
      "true"
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
  ((recorded_attempts < maximum_workload_attempts)); do
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
      "$terminal_error" \
      "true"
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
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python3 "$work_directory/$workload_file" \
    >"$result_pending_path" 2>"$attempt_error_path"
  workload_exit_code="$?"
  detected_non_retryable_failure=""
  if detected_non_retryable_failure="$(
    non_retryable_failure_code "$attempt_error_path"
  )"; then
    persist_non_retryable_failure "$detected_non_retryable_failure"
  fi
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
  if [[ -n "$detected_non_retryable_failure" ]]; then
    retry_allowed=false
    terminal_error="$detected_non_retryable_failure"
    terminal_message="$(
      non_retryable_failure_message "$detected_non_retryable_failure"
    )"
    break
  fi
  if ((workload_attempt >= maximum_workload_attempts)) ||
    [[ ! -f "$adapter_path/checkpoints/latest.json" ]]; then
    break
  fi
  write_progress \
    "$progress_schema_version" \
    "resuming" \
    "Retrying from the latest checkpoint after a workload failure." \
    "$((workload_attempt + 1))" \
    "" \
    "true"
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
  elif ((recorded_attempts >= maximum_workload_attempts)); then
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
    "$terminal_error" \
    "true"
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
        "$terminal_error" \
        "true"
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
        "$terminal_error" \
        "true"
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
if [[ "$workload_exit_code" != "0" && -f "$progress_path" ]]; then
  structure_remote_failure "$workload_exit_code" || true
fi
printf '%s\n' "$workload_exit_code" >"$exit_code_path"

wait "$http_server_pid"
