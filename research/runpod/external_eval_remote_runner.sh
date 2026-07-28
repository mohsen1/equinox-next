#!/usr/bin/env bash
set -uo pipefail

work_directory="${EQUINOX_REMOTE_WORKDIR:-/tmp}"
progress_path="$work_directory/progress.json"
result_pending_path="$work_directory/result.pending.json"
result_path="$work_directory/result.json"
error_path="$work_directory/error.log"
exit_code_path="$work_directory/exit_code"
input_timeout_seconds="${EQUINOX_EXTERNAL_INPUT_TIMEOUT_SECONDS:-1800}"
workload_file="${EQUINOX_WORKLOAD_FILE:-}"
expected_workload_file="research/runpod/revision30_external_eval.py"

durable_json() {
  local destination="$1"
  local payload="$2"
  python3 - "$destination" "$payload" <<'PY'
import json
import os
import sys

destination, encoded = sys.argv[1:]
payload = json.loads(encoded)
pending = destination + ".pending"
with open(pending, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(pending, destination)
directory_descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
}

write_progress() {
  local phase="$1"
  local message="$2"
  local error_code="${3:-}"
  local elapsed_seconds="${4:-0}"
  local payload
  payload="$(
    python3 - "$phase" "$message" "$error_code" "$elapsed_seconds" <<'PY'
import json
import sys

phase, message, error_code, elapsed = sys.argv[1:]
print(
    json.dumps(
        {
            "schema_version": 1,
            "phase": phase,
            "message": message,
            "error": error_code or None,
            "workload": "revision30-post-freeze-external-adapter-evaluation",
            "pack_id": "revision30-post-freeze-external-pack@1",
            "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "elapsed_seconds": int(elapsed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
  )"
  durable_json "$progress_path" "$payload"
}

publish_exit_code() {
  local value="$1"
  printf '%s\n' "$value" >"$exit_code_path.pending"
  python3 - "$exit_code_path.pending" "$exit_code_path" <<'PY'
import os
import sys

pending, destination = sys.argv[1:]
with open(pending, "rb") as handle:
    os.fsync(handle.fileno())
os.replace(pending, destination)
directory_descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
}

if [[ -z "${EQUINOX_RESULT_TOKEN:-}" ]]; then
  printf '%s\n' \
    "EQUINOX_RESULT_TOKEN is missing; refusing to start an unobservable evaluation." \
    >"$error_path"
  publish_exit_code 78
  exit 78
fi
if [[ "$workload_file" != "$expected_workload_file" ]]; then
  printf '%s\n' "The external evaluator received an unsupported workload file." >"$error_path"
  write_progress \
    "failed" \
    "The external evaluator received an unsupported workload file." \
    "UNSUPPORTED_WORKLOAD_FILE"
  publish_exit_code 78
  exit 78
fi
if [[ ! "$input_timeout_seconds" =~ ^[1-9][0-9]*$ ]] ||
  ((input_timeout_seconds > 3600)); then
  printf '%s\n' "External input timeout must be between 1 and 3600 seconds." >"$error_path"
  write_progress \
    "failed" \
    "External input timeout configuration is invalid." \
    "INPUT_TIMEOUT_INVALID"
  publish_exit_code 78
  exit 78
fi

rm -f -- "$exit_code_path" "$result_pending_path"
touch "$error_path"
write_progress \
  "awaiting_inputs" \
  "Container is ready for the frozen manifest and adapter archives."

EQUINOX_REMOTE_WORKDIR="$work_directory" \
  python3 "$work_directory/research/runpod/external_eval_transport.py" \
  >"$work_directory/http.log" 2>&1 &
http_server_pid="$!"

started_epoch="$(date +%s)"
while [[ ! -f "$work_directory/inputs/ready.json" ]]; do
  if ! kill -0 "$http_server_pid" 2>/dev/null; then
    printf '%s\n' "Authenticated external-evaluation transport stopped." >>"$error_path"
    write_progress \
      "failed" \
      "Authenticated input transport stopped unexpectedly." \
      "INPUT_TRANSPORT_FAILED" \
      "$(( $(date +%s) - started_epoch ))"
    publish_exit_code 70
    wait "$http_server_pid" 2>/dev/null || true
    exit 70
  fi
  elapsed_seconds="$(( $(date +%s) - started_epoch ))"
  if ((elapsed_seconds >= input_timeout_seconds)); then
    printf '%s\n' "Timed out waiting for the sealed evaluation inputs." >>"$error_path"
    write_progress \
      "failed" \
      "Timed out waiting for the sealed evaluation inputs." \
      "INPUT_UPLOAD_TIMEOUT" \
      "$elapsed_seconds"
    publish_exit_code 75
    wait "$http_server_pid"
    exit 75
  fi
  if ((elapsed_seconds > 0 && elapsed_seconds % 60 < 5)); then
    write_progress \
      "awaiting_inputs" \
      "Container is ready for the frozen manifest and adapter archives." \
      "" \
      "$elapsed_seconds"
  fi
  sleep 5
done

write_progress \
  "input_verification" \
  "All sealed inputs arrived; starting digest verification." \
  "" \
  "$(( $(date +%s) - started_epoch ))"

cd "$work_directory"
PYTHONPATH="$work_directory" \
  EQUINOX_PROGRESS_PATH="$progress_path" \
  CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  python3 -m research.runpod.revision30_external_eval \
  >"$result_pending_path" 2>>"$error_path"
workload_exit_code="$?"

if [[ "$workload_exit_code" == "0" ]] &&
  python3 - "$result_pending_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if (
    not isinstance(result, dict)
    or result.get("external_evaluation_completed") is not True
    or result.get("every_adapter_reported") is not True
    or not isinstance(result.get("result_digest"), str)
):
    raise SystemExit(1)
PY
then
  python3 - "$result_pending_path" "$result_path" <<'PY'
import os
import sys

pending, destination = sys.argv[1:]
with open(pending, "rb") as handle:
    os.fsync(handle.fileno())
os.replace(pending, destination)
directory_descriptor = os.open(os.path.dirname(destination), os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
PY
else
  if [[ "$workload_exit_code" == "0" ]]; then
    workload_exit_code=65
    printf '%s\n' "External evaluator returned an invalid result." >>"$error_path"
  fi
  write_progress \
    "failed" \
    "External adapter evaluation failed." \
    "EXTERNAL_EVALUATION_FAILED" \
    "$(( $(date +%s) - started_epoch ))"
fi

publish_exit_code "$workload_exit_code"
wait "$http_server_pid"
exit "$workload_exit_code"
