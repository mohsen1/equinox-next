#!/usr/bin/env bash
set -uo pipefail

workload_file="${EQUINOX_WORKLOAD_FILE:?missing EQUINOX_WORKLOAD_FILE}"
target_runtime_seconds="${EQUINOX_RL_TARGET_SECONDS:?missing EQUINOX_RL_TARGET_SECONDS}"
work_directory="${EQUINOX_REMOTE_WORKDIR:-/tmp}"
progress_path="$work_directory/progress.json"
result_pending_path="$work_directory/result.pending.json"
result_path="$work_directory/result.json"
error_path="$work_directory/error.log"
exit_code_path="$work_directory/exit_code"
adapter_path="$work_directory/adapter"

printf '%s\n' \
  '{"schema_version":1,"phase":"container_starting","message":"RunPod container started; preparing the workload.","branch_width":4,"complexity_strategy":"adaptive"}' \
  >"$progress_path"
python3 -m http.server 8000 --directory "$work_directory" \
  >"$work_directory/http.log" 2>&1 &
http_server_pid="$!"

EQUINOX_PROGRESS_PATH="$progress_path" \
  EQUINOX_ADAPTER_PATH="$adapter_path" \
  EQUINOX_RL_TARGET_SECONDS="$target_runtime_seconds" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  python3 "$work_directory/$workload_file" \
  >"$result_pending_path" 2>"$error_path"
workload_exit_code="$?"

if [[ "$workload_exit_code" == "0" ]]; then
  mv "$result_pending_path" "$result_path"
  if [[ -d "$adapter_path" ]]; then
    tar -czf "$work_directory/adapter.tgz" -C "$work_directory" adapter
  fi
fi
printf '%s\n' "$workload_exit_code" >"$exit_code_path"

wait "$http_server_pid"
