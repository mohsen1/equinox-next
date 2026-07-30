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
filesystem_python="${EQUINOX_FILESYSTEM_PYTHON:-python3}"
branch_width=4
if [[ "$study_condition" == "k1_train" || "$study_condition" == k1_* ]]; then
  branch_width=1
fi
requested_work_directory="${EQUINOX_REMOTE_WORKDIR:-/tmp}"
requested_code_root="${EQUINOX_CODE_ROOT:-$requested_work_directory}"
requested_dependency_root="${EQUINOX_DEPENDENCY_ROOT:-}"
case "$workload_file" in
  repository_repair_large_model_eligibility.py | repository_repair_large_model_pilot.py)
    requires_private_code_root=true
    ;;
  *)
    requires_private_code_root=false
    ;;
esac

if ! exec 14<"$requested_work_directory"; then
  printf '%s\n' "The persistent work directory could not be opened." >&2
  exit 78
fi
if ! "$filesystem_python" - "$requested_work_directory" 14 <<'PY'
import os
import re
import stat
import sys

path = sys.argv[1]
descriptor = int(sys.argv[2])
metadata = os.fstat(descriptor)
if not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("persistent work root is not a directory")
if re.fullmatch(r"(?:/proc/self/fd|/dev/fd)/[0-9]+", path):
    return_code = 0
else:
    path_metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        raise SystemExit("persistent work root changed while being opened")
for entry in os.scandir(descriptor):
    entry_metadata = entry.stat(follow_symlinks=False)
    if stat.S_ISLNK(entry_metadata.st_mode) or not (
        stat.S_ISREG(entry_metadata.st_mode) or stat.S_ISDIR(entry_metadata.st_mode)
    ):
        raise SystemExit(f"unsafe persistent work entry: {entry.name}")
PY
then
  printf '%s\n' "The persistent work directory failed no-follow validation." >&2
  exit 78
fi

if [[ "$requires_private_code_root" == "true" && -z "${EQUINOX_CODE_ROOT:-}" ]]; then
  printf '%s\n' "The live-stage workload has no immutable private code root." >&2
  exit 78
fi
if ! exec 18<"$requested_code_root"; then
  printf '%s\n' "The immutable code root could not be opened." >&2
  exit 78
fi
if ! "$filesystem_python" - \
  "$requested_code_root" \
  18 \
  14 \
  "$requires_private_code_root" \
  "$workload_file" <<'PY'
import os
import re
import stat
import sys

path, code_descriptor_raw, work_descriptor_raw, private_required, workload = sys.argv[1:]
code_descriptor = int(code_descriptor_raw)
work_descriptor = int(work_descriptor_raw)
metadata = os.fstat(code_descriptor)
if not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("code root is not a directory")
if re.fullmatch(r"(?:/proc/self/fd|/dev/fd)/[0-9]+", path) is None:
    path_metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        raise SystemExit("code root changed while being opened")
if private_required == "true":
    work_metadata = os.fstat(work_descriptor)
    if (
        metadata.st_dev == work_metadata.st_dev
        and metadata.st_ino == work_metadata.st_ino
    ):
        raise SystemExit("live-stage code root aliases persistent state")
    if metadata.st_mode & 0o222:
        raise SystemExit("live-stage code root is writable")
for name in ("result_server.py", workload):
    if "/" in name or name in {"", ".", ".."}:
        raise SystemExit("unsafe code member name")
    entry_metadata = os.stat(name, dir_fd=code_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(entry_metadata.st_mode):
        raise SystemExit(f"code member is not a regular file: {name}")
    if private_required == "true" and entry_metadata.st_mode & 0o222:
        raise SystemExit(f"live-stage code member is writable: {name}")
PY
then
  printf '%s\n' "The immutable code root failed no-follow validation." >&2
  exit 78
fi
if [[ "$requires_private_code_root" == "true" ]]; then
  if [[ -z "$requested_dependency_root" ]] ||
    ! exec 16<"$requested_dependency_root"; then
    printf '%s\n' "The live-stage workload has no immutable dependency root." >&2
    exit 78
  fi
  if ! "$filesystem_python" - \
    "$requested_dependency_root" \
    16 \
    18 \
    14 <<'PY'
import os
import re
import stat
import sys

path = sys.argv[1]
dependency_descriptor = int(sys.argv[2])
code_descriptor = int(sys.argv[3])
work_descriptor = int(sys.argv[4])
metadata = os.fstat(dependency_descriptor)
if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o222:
    raise SystemExit("dependency root is not a read-only directory")
if re.fullmatch(r"(?:/proc/self/fd|/dev/fd)/[0-9]+", path) is None:
    path_metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        raise SystemExit("dependency root changed while being opened")
for other_descriptor in (code_descriptor, work_descriptor):
    other = os.fstat(other_descriptor)
    if metadata.st_dev == other.st_dev and metadata.st_ino == other.st_ino:
        raise SystemExit("dependency root aliases another execution boundary")
PY
  then
    printf '%s\n' "The immutable dependency root failed no-follow validation." >&2
    exit 78
  fi
fi

proof_id="${EQUINOX_PROOF_ID:-}"
requested_runtime_root="${EQUINOX_RUNTIME_ROOT:-}"
requested_anchor_root="${EQUINOX_RUNTIME_ANCHOR_ROOT:-}"
if [[ "$requires_private_code_root" == "true" ]] &&
  {
    [[ -z "$requested_runtime_root" ]] ||
      [[ "${EQUINOX_PRIVATE_RUNTIME_REVISION:-}" != "bootstrap-private-runtime@1" ]] ||
      [[ -z "$requested_anchor_root" ]] ||
      [[ "${EQUINOX_PRIVATE_ANCHOR_REVISION:-}" != "bootstrap-private-anchor@1" ]]
  }; then
  printf '%s\n' \
    "The live-stage workload has no bootstrap-owned private runtime and rollback anchor." \
    >&2
  exit 78
fi
if [[ -n "$requested_runtime_root" ]]; then
  private_runtime_root="$requested_runtime_root"
else
  private_runtime_root="/tmp/equinox-runtime/$proof_id"
  if ! "$filesystem_python" - "$proof_id" <<'PY'
import os
import re
import stat
import sys

proof_id = sys.argv[1]
if re.fullmatch(r"[A-Za-z0-9._-]+", proof_id) is None:
    raise SystemExit("proof identity is missing or malformed")
base = "/tmp/equinox-runtime"
try:
    os.mkdir(base, 0o700)
except FileExistsError:
    pass
base_metadata = os.lstat(base)
if not stat.S_ISDIR(base_metadata.st_mode) or stat.S_ISLNK(base_metadata.st_mode):
    raise SystemExit("private runtime base is unsafe")
os.chmod(base, 0o700)
base_descriptor = os.open(
    base,
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    try:
        os.mkdir(proof_id, 0o700, dir_fd=base_descriptor)
        os.fsync(base_descriptor)
    except FileExistsError:
        pass
    runtime_descriptor = os.open(
        proof_id,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=base_descriptor,
    )
    try:
        metadata = os.fstat(runtime_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit("private runtime root is unsafe")
        os.fchmod(runtime_descriptor, 0o700)
        os.fsync(runtime_descriptor)
    finally:
        os.close(runtime_descriptor)
finally:
    os.close(base_descriptor)
PY
  then
    printf '%s\n' "The private runtime root could not be established." >&2
    exit 78
  fi
fi
if [[ -n "$requested_anchor_root" ]]; then
  private_anchor_root="$requested_anchor_root"
else
  if ! private_anchor_root="$(
    "$filesystem_python" - "$proof_id" "${EQUINOX_RESULT_TOKEN:-}" <<'PY'
import hashlib
import hmac
import os
import re
import stat
import sys

proof_id, token = sys.argv[1:]
if re.fullmatch(r"[A-Za-z0-9._-]+", proof_id) is None or not token:
    raise SystemExit("fallback anchor identity is missing or malformed")
base = "/tmp/equinox-runner-state"
try:
    os.mkdir(base, 0o700)
except FileExistsError:
    pass
base_metadata = os.lstat(base)
if (
    not stat.S_ISDIR(base_metadata.st_mode)
    or stat.S_ISLNK(base_metadata.st_mode)
    or base_metadata.st_uid != os.geteuid()
):
    raise SystemExit("fallback anchor base is unsafe")
os.chmod(base, 0o700)
base_descriptor = os.open(
    base,
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
anchor_name = hmac.new(
    token.encode(),
    b"equinox/fallback-anchor-root/v1\0" + proof_id.encode(),
    hashlib.sha256,
).hexdigest()
try:
    try:
        os.mkdir(anchor_name, 0o700, dir_fd=base_descriptor)
        os.fsync(base_descriptor)
    except FileExistsError:
        pass
    anchor_descriptor = os.open(
        anchor_name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=base_descriptor,
    )
    try:
        metadata = os.fstat(anchor_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise SystemExit("fallback anchor root is unsafe")
        os.fchmod(anchor_descriptor, 0o700)
        os.fsync(anchor_descriptor)
    finally:
        os.close(anchor_descriptor)
finally:
    os.close(base_descriptor)
print(f"{base}/{anchor_name}")
PY
  )"
  then
    printf '%s\n' "The private rollback anchor could not be established." >&2
    exit 78
  fi
fi
if ! exec 19<"$private_runtime_root"; then
  printf '%s\n' "The private runtime root could not be pinned." >&2
  exit 78
fi
if ! exec 13<"$private_anchor_root"; then
  printf '%s\n' "The private rollback anchor could not be pinned." >&2
  exit 78
fi
if ! "$filesystem_python" - "$private_anchor_root" 13 14 18 19 <<'PY'
import os
import re
import stat
import sys

path = sys.argv[1]
descriptor = int(sys.argv[2])
metadata = os.fstat(descriptor)
if (
    not stat.S_ISDIR(metadata.st_mode)
    or stat.S_IMODE(metadata.st_mode) != 0o700
    or metadata.st_uid != os.geteuid()
):
    raise SystemExit("private rollback anchor root is unsafe")
if re.fullmatch(r"(?:/proc/self/fd|/dev/fd)/[0-9]+", path) is None:
    path_metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        raise SystemExit("private rollback anchor root changed while being pinned")
for raw_other in sys.argv[3:]:
    other = os.fstat(int(raw_other))
    if other.st_dev == metadata.st_dev and other.st_ino == metadata.st_ino:
        raise SystemExit("private rollback anchor aliases another execution root")
PY
then
  printf '%s\n' "The private rollback anchor failed identity validation." >&2
  exit 78
fi
if [[ "$requires_private_code_root" == "true" ]] &&
  ! "$filesystem_python" - 13 16 <<'PY'
import os
import sys

anchor = os.fstat(int(sys.argv[1]))
dependency = os.fstat(int(sys.argv[2]))
if anchor.st_dev == dependency.st_dev and anchor.st_ino == dependency.st_ino:
    raise SystemExit("private rollback anchor aliases the dependency root")
PY
then
  printf '%s\n' "The private rollback anchor aliases immutable dependencies." >&2
  exit 78
fi
if ! "$filesystem_python" - "$private_runtime_root" 19 14 18 <<'PY'
import os
import re
import stat
import sys

path = sys.argv[1]
descriptor = int(sys.argv[2])
descriptor_metadata = os.fstat(descriptor)
if not stat.S_ISDIR(descriptor_metadata.st_mode):
    raise SystemExit("private runtime root is not a directory")
if re.fullmatch(r"(?:/proc/self/fd|/dev/fd)/[0-9]+", path) is None:
    path_metadata = os.lstat(path)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_dev != descriptor_metadata.st_dev
        or path_metadata.st_ino != descriptor_metadata.st_ino
    ):
        raise SystemExit("private runtime root changed while being pinned")
for raw_other in sys.argv[3:]:
    other = os.fstat(int(raw_other))
    if (
        other.st_dev == descriptor_metadata.st_dev
        and other.st_ino == descriptor_metadata.st_ino
    ):
        raise SystemExit("private runtime root aliases a persistent or code root")
PY
then
  printf '%s\n' "The private runtime root failed identity validation." >&2
  exit 78
fi
if [[ "$requires_private_code_root" == "true" ]] &&
  ! "$filesystem_python" - 19 16 <<'PY'
import os
import sys

runtime = os.fstat(int(sys.argv[1]))
dependency = os.fstat(int(sys.argv[2]))
if runtime.st_dev == dependency.st_dev and runtime.st_ino == dependency.st_ino:
    raise SystemExit("private runtime root aliases the dependency root")
PY
then
  printf '%s\n' "The private runtime root aliases immutable dependencies." >&2
  exit 78
fi
if ! "$filesystem_python" - 14 "$workload_file" <<'PY'
import os
import stat
import sys

descriptor = int(sys.argv[1])
workload = sys.argv[2]
common = {"remote_runner.sh", "result_server.py"}
support = {
    "branching_sequence_ladder.py": set(),
    "repository_repair_rl.py": {"repository_repair_env.py"},
    "repository_repair_study.py": {
        "repository_repair_env.py",
        "repository_repair_rl.py",
    },
    "repository_repair_study_v31.py": {
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
    },
    "repository_repair_eligibility.py": {
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
    },
}
larger = {
    "larger-model-dependencies.lock",
    "larger-model-eligibility.json",
    "larger_model_gate.py",
    "repository_repair_env.py",
    "repository_repair_env_v31.py",
    "repository_repair_env_v32.py",
    "repository_repair_env_v33.py",
    "repository_repair_large_model_eligibility.py",
    "repository_repair_large_model_pilot.py",
    "repository_repair_large_model_study.py",
    "repository_repair_large_model_trainer.py",
    "retention_checkpoint_probe.py",
}
support["repository_repair_large_model_eligibility.py"] = larger
support["repository_repair_large_model_pilot.py"] = larger
try:
    allowed = common | support[workload] | {workload}
except KeyError as error:
    raise SystemExit("persistent workload is unsupported") from error
allowed.update(
    {
        "bundle-stage-receipt.json",
        "code-materialization-evidence.json",
        "dependency-quarantine-evidence.json",
        "live-stage-activation.json",
        "live-stage-plan.json",
        "live-stage-state.json",
        "torch-retention-evidence.json",
        "volume-readiness-receipt.json",
    }
)
for entry in os.scandir(descriptor):
    metadata = entry.stat(follow_symlinks=False)
    if entry.name not in allowed or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"unexpected persistent launch entry: {entry.name}")
PY
then
  printf '%s\n' "The persistent launch entry set failed validation." >&2
  exit 78
fi

if [[ -d /proc/self/fd/19 && -d /proc/self/fd/18 ]]; then
  work_directory="/proc/self/fd/19"
  persistent_work_directory="/proc/self/fd/14"
  code_root="/proc/self/fd/18"
  dependency_root="/proc/self/fd/16"
else
  # macOS exposes directory descriptors through /dev/fd but does not support
  # child lookup through them. Production RunPod workers use the Linux branch.
  work_directory="$private_runtime_root"
  persistent_work_directory="$requested_work_directory"
  code_root="$requested_code_root"
  dependency_root="$requested_dependency_root"
fi
export EQUINOX_REMOTE_WORKDIR="$work_directory"
export EQUINOX_RUNTIME_ROOT="$work_directory"
export EQUINOX_RUNTIME_ROOT_FD=19
export EQUINOX_RUNTIME_ANCHOR_ROOT_FD=13
if [[ "$requires_private_code_root" == "true" ]]; then
  export EQUINOX_CODE_ROOT="$code_root"
  export EQUINOX_DEPENDENCY_ROOT="$dependency_root"
  export PYTHONPATH="$code_root:$dependency_root"
fi
progress_path="$work_directory/progress.json"
result_pending_path="$work_directory/result.pending.json"
result_path="$work_directory/result.json"
error_path="$work_directory/error.log"
exit_code_path="$work_directory/exit_code"
adapter_path="$work_directory/adapter"
attempt_counter_path="$work_directory/workload-attempt-count"
non_retryable_failure_path="$work_directory/non-retryable-failure"
live_stage_handoff_path="$work_directory/live-stage-handoff.json"
resume_state_ready=false

update_resume_manifest() {
  local mode="$1"
  shift
  "$filesystem_python" - 19 13 "$mode" "$workload_file" "$@" <<'PY'
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
anchor_root_descriptor = int(sys.argv[2])
mode = sys.argv[3]
workload = sys.argv[4]
changed_roots = tuple(sys.argv[5:])
manifest_name = "runner-resume-state.json"
revision = "token-bound-runner-resume@1"
maximum_files = 200_000
maximum_bytes = 16 * 1024 * 1024 * 1024
token = os.environ.get("EQUINOX_RESULT_TOKEN", "")
if not token:
    raise SystemExit("resume authentication key is missing")
if mode not in {"initialize", "validate", "update"}:
    raise SystemExit("resume manifest mode is invalid")

bundle_files = {
    "branching_sequence_ladder.py": {
        "branching_sequence_ladder.py",
        "remote_runner.sh",
        "result_server.py",
    },
    "repository_repair_rl.py": {
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "result_server.py",
    },
    "repository_repair_study.py": {
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "result_server.py",
    },
    "repository_repair_study_v31.py": {
        "remote_runner.sh",
        "repository_repair_env.py",
        "repository_repair_env_v31.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "repository_repair_study_v31.py",
        "result_server.py",
    },
    "repository_repair_eligibility.py": {
        "remote_runner.sh",
        "repository_repair_eligibility.py",
        "repository_repair_env.py",
        "repository_repair_rl.py",
        "repository_repair_study.py",
        "result_server.py",
    },
}
larger_model_files = {
    "larger-model-dependencies.lock",
    "larger-model-eligibility.json",
    "larger_model_gate.py",
    "remote_runner.sh",
    "repository_repair_env.py",
    "repository_repair_env_v31.py",
    "repository_repair_env_v32.py",
    "repository_repair_env_v33.py",
    "repository_repair_large_model_eligibility.py",
    "repository_repair_large_model_pilot.py",
    "repository_repair_large_model_study.py",
    "repository_repair_large_model_trainer.py",
    "result_server.py",
    "retention_checkpoint_probe.py",
}
bundle_files["repository_repair_large_model_eligibility.py"] = larger_model_files
bundle_files["repository_repair_large_model_pilot.py"] = larger_model_files
try:
    baseline_names = set(bundle_files[workload])
except KeyError as error:
    raise SystemExit("resume manifest workload is unsupported") from error
baseline_names.update(
    {
        "bundle-stage-receipt.json",
        "code-materialization-evidence.json",
        "dependency-quarantine-evidence.json",
        "live-stage-activation.json",
        "live-stage-plan.json",
        "live-stage-state.json",
        "torch-retention-evidence.json",
        "volume-readiness-receipt.json",
    }
)
# Mutable runtime state is isolated from the shared bundle/evidence directory.
# No launch artifact is permitted to appear in the private runtime root.
baseline_names = set()

runtime_names = {
    "adapter.tgz",
    "attempts",
    "eligibility-unused-study-evidence.json",
    "error.log",
    "exit_code",
    "live-stage-handoff.json",
    "non-retryable-failure",
    "progress.json",
    "result.json",
    "result.pending.json",
    "revision31-study-runtime-evidence.json",
    "study-runtime-evidence.json",
    "workload-attempt-count",
}
attempt_error_pattern = re.compile(r"error[.]attempt-[12][.]log")


def canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


launch_identity = {
    "proof_id": os.environ.get("EQUINOX_PROOF_ID", ""),
    "run_identity": os.environ.get("EQUINOX_RUN_IDENTITY", ""),
    "private_runtime_revision": os.environ.get(
        "EQUINOX_PRIVATE_RUNTIME_REVISION",
        "",
    ),
    "workload_file": workload,
    "model_id": os.environ.get("EQUINOX_RL_MODEL_ID", ""),
    "optimization_seed": os.environ.get("EQUINOX_RL_SEED", ""),
    "study_condition": os.environ.get("EQUINOX_STUDY_CONDITION", ""),
    "bundle_digest": os.environ.get("EQUINOX_BUNDLE_SHA256", ""),
    "bundle_activation_digest": os.environ.get(
        "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
        "",
    ),
    "source_contract_digest": os.environ.get(
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
        "",
    ),
    "code_private_tree_digest": os.environ.get(
        "EQUINOX_CODE_PRIVATE_TREE_SHA256",
        "",
    ),
    "dependency_private_tree_digest": os.environ.get(
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
        "",
    ),
    "dependency_lock_digest": os.environ.get(
        "EQUINOX_DEPENDENCY_LOCK_SHA256",
        "",
    ),
}
launch_identity_digest = (
    "sha256:" + hashlib.sha256(canonical_json(launch_identity)).hexdigest()
)
if not re.fullmatch(r"[A-Za-z0-9._-]+", launch_identity["proof_id"]):
    raise SystemExit("resume manifest proof identity is missing or malformed")
anchor_root_metadata = os.fstat(anchor_root_descriptor)
if (
    not stat.S_ISDIR(anchor_root_metadata.st_mode)
    or stat.S_IMODE(anchor_root_metadata.st_mode) != 0o700
    or anchor_root_metadata.st_uid != os.geteuid()
):
    raise SystemExit("private resume anchor root is unsafe")
anchor_name = (
    hmac.new(
        token.encode(),
        b"equinox/private-anchor-name/v1\0" + canonical_json(launch_identity),
        hashlib.sha256,
    ).hexdigest()
    + ".json"
)


def artifact_domain(path):
    if path == "progress.json":
        return "equinox/progress/v1"
    if path in {"result.json", "result.pending.json"}:
        return "equinox/result/v1"
    if path == "adapter.tgz" or path == "adapter" or path.startswith("adapter/"):
        return "equinox/checkpoint/v1"
    if (
        path == "workload-attempt-count"
        or path == "attempts"
        or attempt_error_pattern.fullmatch(path) is not None
    ):
        return "equinox/attempt/v1"
    return "equinox/journal/v1"


def authenticated_entry(path, size, digest):
    material = {
        "domain": artifact_domain(path),
        "launch_identity_digest": launch_identity_digest,
        "path": path,
        "sha256": digest,
        "size_bytes": size,
    }
    return {
        **material,
        "hmac": (
            "hmac-sha256:"
            + hmac.new(
                token.encode(),
                (material["domain"] + "\0").encode() + canonical_json(material),
                hashlib.sha256,
            ).hexdigest()
        ),
    }


def runtime_top(name):
    return name in runtime_names or attempt_error_pattern.fullmatch(name) is not None


for changed in changed_roots:
    if changed == "adapter":
        continue
    if "/" in changed or not runtime_top(changed):
        raise SystemExit(f"unsafe resume transition root: {changed}")

commit_expectations = {
    "result.pending.json": os.environ.get(
        "EQUINOX_EXPECTED_PENDING_RESULT_COMMIT",
        "",
    ),
    "result.json": os.environ.get("EQUINOX_EXPECTED_RESULT_COMMIT", ""),
    "adapter.tgz": os.environ.get("EQUINOX_EXPECTED_ADAPTER_COMMIT", ""),
}
commit_identity_pattern = re.compile(
    r"[0-9]+:[0-9]+:[0-9]+:sha256:[0-9a-f]{64}"
)
if any(
    value and commit_identity_pattern.fullmatch(value) is None
    for value in commit_expectations.values()
):
    raise SystemExit("artifact commit identity is malformed")
observed_commit_identities = {}


def read_regular(name, maximum):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError(f"{name} has an unsafe type or size")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, metadata.st_size - len(payload)),
            )
            if not chunk:
                raise ValueError(f"{name} was truncated")
            payload += chunk
        return payload
    finally:
        os.close(descriptor)


def load_manifest():
    try:
        payload = read_regular(manifest_name, 64 * 1024 * 1024)
    except FileNotFoundError:
        return None
    value = json.loads(payload)
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ValueError("resume manifest is not one canonical JSON object")
    if set(value) != {
        "schema_version",
        "revision",
        "workload_file",
        "launch_identity_digest",
        "generation",
        "parent_manifest_sha256",
        "directories",
        "artifacts",
        "hmac",
    }:
        raise ValueError("resume manifest has unknown or missing fields")
    if not isinstance(value.get("artifacts"), list) or any(
        not isinstance(entry, dict)
        or set(entry)
        != {
            "domain",
            "launch_identity_digest",
            "path",
            "sha256",
            "size_bytes",
            "hmac",
        }
        for entry in value["artifacts"]
    ):
        raise ValueError("resume manifest artifact schema is invalid")
    supplied_hmac = value.get("hmac")
    material = {key: item for key, item in value.items() if key != "hmac"}
    expected_hmac = (
        "hmac-sha256:"
        + hmac.new(
            token.encode(),
            b"equinox/runner-resume-manifest/v1\0" + canonical_json(material),
            hashlib.sha256,
        ).hexdigest()
    )
    if (
        not isinstance(supplied_hmac, str)
        or not hmac.compare_digest(supplied_hmac, expected_hmac)
        or value.get("schema_version") != 1
        or value.get("revision") != revision
        or value.get("workload_file") != workload
        or value.get("launch_identity_digest") != launch_identity_digest
    ):
        raise ValueError("resume manifest authentication failed")
    return value


def anchor_hmac(material):
    return (
        "hmac-sha256:"
        + hmac.new(
            token.encode(),
            b"equinox/runner-private-anchor/v1\0" + canonical_json(material),
            hashlib.sha256,
        ).hexdigest()
    )


def load_anchor():
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(anchor_name, flags, dir_fd=anchor_root_descriptor)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024:
            raise ValueError("private resume anchor has an unsafe type or size")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, metadata.st_size - len(payload))
            if not chunk:
                raise ValueError("private resume anchor was truncated")
            payload += chunk
    finally:
        os.close(descriptor)
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "revision",
            "proof_id",
            "launch_identity_digest",
            "generation",
            "manifest_digest",
            "hmac",
        }
        or payload != canonical_json(value) + b"\n"
    ):
        raise ValueError("private resume anchor schema is invalid")
    supplied = value["hmac"]
    material = {key: item for key, item in value.items() if key != "hmac"}
    if (
        value.get("schema_version") != 1
        or value.get("revision") != "private-runner-rollback-anchor@1"
        or value.get("proof_id") != launch_identity["proof_id"]
        or value.get("launch_identity_digest") != launch_identity_digest
        or not isinstance(supplied, str)
        or not hmac.compare_digest(supplied, anchor_hmac(material))
    ):
        raise ValueError("private resume anchor authentication failed")
    return value


def write_anchor(generation, manifest_digest):
    material = {
        "schema_version": 1,
        "revision": "private-runner-rollback-anchor@1",
        "proof_id": launch_identity["proof_id"],
        "launch_identity_digest": launch_identity_digest,
        "generation": generation,
        "manifest_digest": manifest_digest,
    }
    value = {**material, "hmac": anchor_hmac(material)}
    payload = canonical_json(value) + b"\n"
    pending_name = ".anchor-" + secrets.token_hex(16)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(
        pending_name,
        flags,
        0o600,
        dir_fd=anchor_root_descriptor,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.replace(
            pending_name,
            anchor_name,
            src_dir_fd=anchor_root_descriptor,
            dst_dir_fd=anchor_root_descriptor,
        )
        os.fsync(anchor_root_descriptor)
    finally:
        os.close(descriptor)
        try:
            os.unlink(pending_name, dir_fd=anchor_root_descriptor)
        except FileNotFoundError:
            pass


observed_files = []
observed_directories = []
observed_bytes = 0


def scan_directory(descriptor, prefix):
    global observed_bytes
    for entry in sorted(os.scandir(descriptor), key=lambda item: item.name):
        if entry.name in {"", ".", ".."} or "/" in entry.name:
            raise ValueError("runtime tree contains an unsafe name")
        metadata = entry.stat(follow_symlinks=False)
        relative = f"{prefix}/{entry.name}" if prefix else entry.name
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.append(relative)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(entry.name, flags, dir_fd=descriptor)
            try:
                scan_directory(child, relative)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"runtime artifact has an unsafe type: {relative}")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        file_descriptor = os.open(entry.name, flags, dir_fd=descriptor)
        digest = hashlib.sha256()
        try:
            before = os.fstat(file_descriptor)
            remaining = before.st_size
            while remaining:
                chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"runtime artifact was truncated: {relative}")
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
            raise ValueError(f"runtime artifact changed while hashing: {relative}")
        observed_bytes += before.st_size
        observed_files.append(
            authenticated_entry(
                relative,
                before.st_size,
                "sha256:" + digest.hexdigest(),
            )
        )
        if len(observed_files) > maximum_files or observed_bytes > maximum_bytes:
            raise ValueError("runtime artifacts exceed their safety bound")


for entry in sorted(os.scandir(directory_descriptor), key=lambda item: item.name):
    name = entry.name
    metadata = entry.stat(follow_symlinks=False)
    if name == manifest_name:
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("resume manifest has an unsafe type")
        continue
    if name in baseline_names:
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"baseline artifact has an unsafe type: {name}")
        continue
    if name == "adapter":
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("adapter runtime root has an unsafe type")
        continue
    if runtime_top(name):
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"runtime artifact has an unsafe type: {name}")
        continue
    raise ValueError(f"unexpected persistent work entry: {name}")

for entry in sorted(os.scandir(directory_descriptor), key=lambda item: item.name):
    name = entry.name
    if name == "adapter":
        observed_directories.append(name)
        adapter_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            scan_directory(adapter_descriptor, name)
        finally:
            os.close(adapter_descriptor)
    elif runtime_top(name):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        file_descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        digest = hashlib.sha256()
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"runtime artifact has an unsafe type: {name}")
            remaining = before.st_size
            while remaining:
                chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"runtime artifact was truncated: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
            raise ValueError(f"runtime artifact changed while hashing: {name}")
        observed_bytes += before.st_size
        observed_commit_identities[name] = (
            f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
            f"sha256:{digest.hexdigest()}"
        )
        observed_files.append(
            authenticated_entry(
                name,
                before.st_size,
                "sha256:" + digest.hexdigest(),
            )
        )
        if len(observed_files) > maximum_files or observed_bytes > maximum_bytes:
            raise ValueError("runtime artifacts exceed their safety bound")

observed_paths = {str(entry["path"]) for entry in observed_files}
for protected_path, expected_identity in commit_expectations.items():
    if expected_identity:
        if observed_commit_identities.get(protected_path) != expected_identity:
            raise ValueError(f"{protected_path} changed before authenticated commit")
    elif (
        mode == "update"
        and protected_path in changed_roots
        and protected_path in observed_paths
    ):
        raise ValueError(f"{protected_path} has no pinned commit identity")

observed_files.sort(key=lambda entry: str(entry["path"]))
observed_directories.sort()
old = load_manifest()
anchor = load_anchor()
if old is None:
    if (
        mode != "initialize"
        or observed_files
        or observed_directories
        or anchor is not None
    ):
        raise ValueError("unauthenticated resumable artifacts were present")
    next_generation = 0
    parent_manifest_sha256 = None
elif mode == "initialize":
    old_generation = old.get("generation")
    if not isinstance(old_generation, int) or isinstance(old_generation, bool):
        raise ValueError("resume manifest generation is invalid")
    old_digest = "sha256:" + hashlib.sha256(
        canonical_json(old) + b"\n"
    ).hexdigest()
    if anchor is None or not (
        (
            anchor.get("generation") == old_generation
            and anchor.get("manifest_digest") == old_digest
        )
        or (
            anchor.get("generation") == old_generation - 1
            and old.get("parent_manifest_sha256") == anchor.get("manifest_digest")
        )
    ):
        raise ValueError("resume manifest rollback anchor did not match")
    expected_files = old.get("artifacts")
    expected_directories = old.get("directories")
    if expected_files != observed_files or expected_directories != observed_directories:
        raise ValueError("authenticated resume artifacts changed")
    next_generation = old_generation
    parent_manifest_sha256 = old.get("parent_manifest_sha256")
elif mode == "validate":
    old_generation = old.get("generation")
    old_digest = "sha256:" + hashlib.sha256(
        canonical_json(old) + b"\n"
    ).hexdigest()
    if (
        not isinstance(old_generation, int)
        or isinstance(old_generation, bool)
        or anchor is None
        or anchor.get("generation") != old_generation
        or anchor.get("manifest_digest") != old_digest
    ):
        raise ValueError("resume manifest rollback anchor did not match")
    if old.get("artifacts") != observed_files or old.get("directories") != observed_directories:
        raise ValueError("authenticated resume artifacts changed")
    next_generation = old_generation
    parent_manifest_sha256 = old.get("parent_manifest_sha256")
else:
    old_generation = old.get("generation")
    old_digest = "sha256:" + hashlib.sha256(
        canonical_json(old) + b"\n"
    ).hexdigest()
    if (
        not isinstance(old_generation, int)
        or isinstance(old_generation, bool)
        or anchor is None
        or anchor.get("generation") != old_generation
        or anchor.get("manifest_digest") != old_digest
    ):
        raise ValueError("resume manifest rollback anchor did not match")
    old_files = {
        str(entry["path"]): entry
        for entry in old.get("artifacts", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    new_files = {str(entry["path"]): entry for entry in observed_files}
    old_directories = set(old.get("directories", []))
    new_directories = set(observed_directories)

    def permitted(path):
        return any(path == root or path.startswith(root + "/") for root in changed_roots)

    for path in set(old_files) | set(new_files):
        if old_files.get(path) != new_files.get(path) and not permitted(path):
            raise ValueError(f"unapproved runtime artifact transition: {path}")
    for path in old_directories | new_directories:
        if (path in old_directories) != (path in new_directories) and not permitted(path):
            raise ValueError(f"unapproved runtime directory transition: {path}")
    next_generation = old_generation + 1
    parent_manifest_sha256 = old_digest

material = {
    "schema_version": 1,
    "revision": revision,
    "workload_file": workload,
    "launch_identity_digest": launch_identity_digest,
    "generation": next_generation,
    "parent_manifest_sha256": parent_manifest_sha256,
    "directories": observed_directories,
    "artifacts": observed_files,
}
material["hmac"] = (
    "hmac-sha256:"
    + hmac.new(
        token.encode(),
        b"equinox/runner-resume-manifest/v1\0" + canonical_json(material),
        hashlib.sha256,
    ).hexdigest()
)
payload = canonical_json(material) + b"\n"
pending_name = ".runner-resume-" + secrets.token_hex(16)
flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
pending_descriptor = os.open(
    pending_name,
    flags,
    0o600,
    dir_fd=directory_descriptor,
)
try:
    os.write(pending_descriptor, payload)
    os.fsync(pending_descriptor)
    os.replace(
        pending_name,
        manifest_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(pending_descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
manifest_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
write_anchor(next_generation, manifest_digest)
PY
}

safe_atomic_write() {
  local name="$1"
  local mode="${2:-0600}"
  local content="${3:-}"
  "$filesystem_python" - 19 "$name" "$mode" "$content" <<'PY'
import os
import re
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
name = sys.argv[2]
mode = int(sys.argv[3], 8)
payload = sys.argv[4].encode()
if re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
    raise SystemExit("unsafe persistent artifact name")
try:
    existing = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
except FileNotFoundError:
    existing = None
if existing is not None and not stat.S_ISREG(existing.st_mode):
    raise SystemExit("persistent artifact target has an unsafe type")
pending_name = ".pending-" + secrets.token_hex(16)
flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
descriptor = os.open(pending_name, flags, mode, dir_fd=directory_descriptor)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(
        pending_name,
        name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
PY
  if [[ "$resume_state_ready" == "true" ]]; then
    update_resume_manifest update "$name"
  fi
}

safe_read_regular() {
  local name="$1"
  local maximum_bytes="${2:-67108864}"
  "$filesystem_python" - 19 "$name" "$maximum_bytes" <<'PY'
import os
import re
import stat
import sys

directory_descriptor = int(sys.argv[1])
name = sys.argv[2]
maximum = int(sys.argv[3])
if re.fullmatch(r"[A-Za-z0-9._-]+", name) is None or maximum <= 0:
    raise SystemExit(1)
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
except FileNotFoundError:
    raise SystemExit(1) from None
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        raise SystemExit(1)
    remaining = metadata.st_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise SystemExit(1)
        sys.stdout.buffer.write(chunk)
        remaining -= len(chunk)
finally:
    os.close(descriptor)
PY
}

safe_regular_exists() {
  local name="$1"
  "$filesystem_python" - 19 "$name" <<'PY'
import os
import re
import stat
import sys

descriptor = int(sys.argv[1])
name = sys.argv[2]
if re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
    raise SystemExit(1)
try:
    metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
except FileNotFoundError:
    raise SystemExit(1) from None
raise SystemExit(0 if stat.S_ISREG(metadata.st_mode) else 2)
PY
}

safe_unlink_regular() {
  local name="$1"
  "$filesystem_python" - 19 "$name" <<'PY'
import os
import re
import stat
import sys

descriptor = int(sys.argv[1])
name = sys.argv[2]
if re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
    raise SystemExit("unsafe persistent artifact name")
try:
    metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
except FileNotFoundError:
    raise SystemExit(0) from None
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("persistent artifact has an unsafe type")
os.unlink(name, dir_fd=descriptor)
os.fsync(descriptor)
PY
  if [[ "$resume_state_ready" == "true" ]]; then
    update_resume_manifest update "$name"
  fi
}

safe_write_line() {
  local name="$1"
  local value="$2"
  safe_atomic_write "$name" 0600 "$value"$'\n'
}

boot_attempt=1
if [[ -z "${EQUINOX_RESULT_TOKEN:-}" ]]; then
  safe_write_line \
    "error.log" \
    "EQUINOX_RESULT_TOKEN is missing; refusing to start an unobservable workload."
  safe_write_line "exit_code" 78
  exit 78
fi
if ! update_resume_manifest initialize; then
  printf '%s\n' \
    "The persistent runtime state was not authenticated for this launch." \
    >&2
  exit 78
fi
resume_state_ready=true
if safe_regular_exists "workload-attempt-count"; then
  boot_attempt_candidate="$(safe_read_regular "workload-attempt-count" 16)"
  if [[ "$boot_attempt_candidate" =~ ^[12]$ ]]; then
    boot_attempt="$boot_attempt_candidate"
  fi
fi

case "$workload_file" in
  repository_repair_rl.py | repository_repair_study.py | repository_repair_study_v31.py | repository_repair_large_model_pilot.py)
    adapter_enabled=true
    ;;
  *)
    adapter_enabled=false
    ;;
esac
if [[ "$adapter_enabled" == "true" ]]; then
  if ! "$filesystem_python" - 19 <<'PY'
import os
import stat
import sys

directory_descriptor = int(sys.argv[1])
try:
    metadata = os.stat("adapter", dir_fd=directory_descriptor, follow_symlinks=False)
except FileNotFoundError:
    os.mkdir("adapter", mode=0o700, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)
    metadata = os.stat("adapter", dir_fd=directory_descriptor, follow_symlinks=False)
if not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("persistent adapter root is not a real directory")
PY
  then
    safe_write_line "error.log" "The persistent adapter root has an unsafe type."
    safe_write_line "exit_code" 78
    exit 78
  fi
  if ! update_resume_manifest update adapter; then
    printf '%s\n' "The persistent adapter root could not be authenticated." >&2
    exit 78
  fi
  if ! exec 17<"$work_directory/adapter"; then
    safe_write_line "error.log" "The persistent adapter root could not be pinned."
    safe_write_line "exit_code" 78
    exit 78
  fi
  if ! "$filesystem_python" - 19 17 <<'PY'
import os
import stat
import sys

parent_descriptor = int(sys.argv[1])
adapter_descriptor = int(sys.argv[2])
path_metadata = os.stat("adapter", dir_fd=parent_descriptor, follow_symlinks=False)
descriptor_metadata = os.fstat(adapter_descriptor)
if (
    not stat.S_ISDIR(path_metadata.st_mode)
    or not stat.S_ISDIR(descriptor_metadata.st_mode)
    or path_metadata.st_dev != descriptor_metadata.st_dev
    or path_metadata.st_ino != descriptor_metadata.st_ino
):
    raise SystemExit("persistent adapter root changed while being pinned")


def validate_tree(descriptor):
    for entry in os.scandir(descriptor):
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(entry.name, flags, dir_fd=descriptor)
            try:
                validate_tree(child)
            finally:
                os.close(child)
        elif not stat.S_ISREG(metadata.st_mode):
            raise SystemExit(f"unsafe adapter entry: {entry.name}")


validate_tree(adapter_descriptor)
PY
  then
    safe_write_line "error.log" "The persistent adapter tree failed no-follow validation."
    safe_write_line "exit_code" 78
    exit 78
  fi
  if [[ -d /proc/self/fd/17 ]]; then
    adapter_path="/proc/self/fd/17"
  else
    adapter_path="$work_directory/adapter"
  fi
  export EQUINOX_ADAPTER_ROOT_FD=17
fi

adapter_checkpoint_available() {
  if [[ "$adapter_enabled" != "true" ]]; then
    return 1
  fi
  "$filesystem_python" - 17 <<'PY'
import os
import stat
import sys

adapter_descriptor = int(sys.argv[1])
flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
try:
    checkpoints_descriptor = os.open("checkpoints", flags, dir_fd=adapter_descriptor)
except (FileNotFoundError, NotADirectoryError, OSError):
    raise SystemExit(1) from None
try:
    latest_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    latest_descriptor = os.open("latest.json", latest_flags, dir_fd=checkpoints_descriptor)
    try:
        metadata = os.fstat(latest_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise SystemExit(1)
    finally:
        os.close(latest_descriptor)
finally:
    os.close(checkpoints_descriptor)
PY
}

checkpoint_replay_handoff() {
  local mode="$1"
  if [[ "$adapter_enabled" != "true" ]]; then
    return 1
  fi
  "$filesystem_python" - 19 17 13 "$mode" "$workload_file" <<'PY'
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys

runtime_descriptor = int(sys.argv[1])
adapter_descriptor = int(sys.argv[2])
anchor_root_descriptor = int(sys.argv[3])
mode = sys.argv[4]
workload = sys.argv[5]
if mode not in {"capture", "load"}:
    raise SystemExit("checkpoint replay handoff mode is invalid")
key_hex = os.environ.get("EQUINOX_CHECKPOINT_AUTHENTICATION_KEY", "")
result_token = os.environ.get("EQUINOX_RESULT_TOKEN", "")
run_identity = os.environ.get("EQUINOX_RUN_IDENTITY", "")
proof_id = os.environ.get("EQUINOX_PROOF_ID", "")
if (
    re.fullmatch(r"[0-9a-f]{64}", key_hex) is None
    or not result_token
    or re.fullmatch(r"[A-Za-z0-9._-]+", proof_id) is None
    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{15,191}", run_identity) is None
):
    raise SystemExit("checkpoint authentication key is unavailable")
checkpoint_key = bytes.fromhex(key_hex)
digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
checkpoint_revision = "launch-bound-checkpoint-manifest@1"
anchor_revision = "private-checkpoint-replay-anchor@1"


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


launch_identity = {
    "proof_id": proof_id,
    "run_identity": run_identity,
    "private_runtime_revision": os.environ.get(
        "EQUINOX_PRIVATE_RUNTIME_REVISION",
        "",
    ),
    "workload_file": workload,
    "model_id": os.environ.get("EQUINOX_RL_MODEL_ID", ""),
    "optimization_seed": os.environ.get("EQUINOX_RL_SEED", ""),
    "study_condition": os.environ.get("EQUINOX_STUDY_CONDITION", ""),
    "bundle_digest": os.environ.get("EQUINOX_BUNDLE_SHA256", ""),
    "bundle_activation_digest": os.environ.get(
        "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
        "",
    ),
    "source_contract_digest": os.environ.get(
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
        "",
    ),
    "code_private_tree_digest": os.environ.get(
        "EQUINOX_CODE_PRIVATE_TREE_SHA256",
        "",
    ),
    "dependency_private_tree_digest": os.environ.get(
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
        "",
    ),
    "dependency_lock_digest": os.environ.get(
        "EQUINOX_DEPENDENCY_LOCK_SHA256",
        "",
    ),
}
launch_identity_digest = (
    "sha256:" + hashlib.sha256(canonical(launch_identity)).hexdigest()
)


def read_regular_at(directory_descriptor, name, maximum_bytes):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{name} is not a bounded single-link regular file")
        chunks = []
        observed_bytes = 0
        while observed_bytes < before.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, before.st_size - observed_bytes),
            )
            if not chunk:
                raise ValueError(f"{name} was truncated")
            chunks.append(chunk)
            observed_bytes += len(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{name} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def canonical_json_at(directory_descriptor, name, maximum_bytes):
    payload = read_regular_at(directory_descriptor, name, maximum_bytes)
    value = json.loads(payload)
    if not isinstance(value, dict) or payload != canonical(value) + b"\n":
        raise ValueError(f"{name} is not one canonical JSON object")
    return value, payload


resume_manifest, _ = canonical_json_at(
    runtime_descriptor,
    "runner-resume-state.json",
    64 * 1024 * 1024,
)
if (
    set(resume_manifest)
    != {
        "schema_version",
        "revision",
        "workload_file",
        "launch_identity_digest",
        "generation",
        "parent_manifest_sha256",
        "directories",
        "artifacts",
        "hmac",
    }
    or resume_manifest.get("schema_version") != 1
    or resume_manifest.get("revision") != "token-bound-runner-resume@1"
    or resume_manifest.get("workload_file") != workload
    or resume_manifest.get("launch_identity_digest") != launch_identity_digest
):
    raise ValueError("runner resume manifest identity is invalid")
resume_material = {
    name: value for name, value in resume_manifest.items() if name != "hmac"
}
expected_resume_hmac = (
    "hmac-sha256:"
    + hmac.new(
        result_token.encode(),
        b"equinox/runner-resume-manifest/v1\0" + canonical(resume_material),
        hashlib.sha256,
    ).hexdigest()
)
if not isinstance(resume_manifest.get("hmac"), str) or not hmac.compare_digest(
    resume_manifest["hmac"],
    expected_resume_hmac,
):
    raise ValueError("runner resume manifest authentication failed")

progress_payload = read_regular_at(
    runtime_descriptor,
    "progress.json",
    64 * 1024 * 1024,
)
progress_digest = "sha256:" + hashlib.sha256(progress_payload).hexdigest()
progress_entries = [
    entry
    for entry in resume_manifest.get("artifacts", [])
    if isinstance(entry, dict) and entry.get("path") == "progress.json"
]
if len(progress_entries) != 1:
    raise ValueError("authenticated progress entry is missing")
progress_entry = progress_entries[0]
progress_material = {
    "domain": "equinox/progress/v1",
    "launch_identity_digest": launch_identity_digest,
    "path": "progress.json",
    "sha256": progress_digest,
    "size_bytes": len(progress_payload),
}
expected_progress_hmac = (
    "hmac-sha256:"
    + hmac.new(
        result_token.encode(),
        b"equinox/progress/v1\0" + canonical(progress_material),
        hashlib.sha256,
    ).hexdigest()
)
if (
    set(progress_entry) != {*progress_material, "hmac"}
    or any(progress_entry.get(name) != value for name, value in progress_material.items())
    or not isinstance(progress_entry.get("hmac"), str)
    or not hmac.compare_digest(progress_entry["hmac"], expected_progress_hmac)
):
    raise ValueError("structured progress is not authenticated by the runner manifest")
progress = json.loads(progress_payload)
if not isinstance(progress, dict):
    raise ValueError("structured progress is not an object")
checkpoint_generation = progress.get("checkpoint_generation")
checkpoint_manifest_digest = progress.get("checkpoint_manifest_digest")
if (
    not isinstance(checkpoint_generation, int)
    or isinstance(checkpoint_generation, bool)
    or checkpoint_generation < 1
    or not isinstance(checkpoint_manifest_digest, str)
    or digest_pattern.fullmatch(checkpoint_manifest_digest) is None
):
    raise ValueError("structured progress has no valid checkpoint replay binding")

directory_flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
checkpoints_descriptor = os.open(
    "checkpoints",
    directory_flags,
    dir_fd=adapter_descriptor,
)
try:
    pointer, _ = canonical_json_at(
        checkpoints_descriptor,
        "latest.json",
        16 * 1024,
    )
finally:
    os.close(checkpoints_descriptor)
if (
    set(pointer)
    != {
        "schema_version",
        "revision",
        "run_identity",
        "checkpoint",
        "generation",
        "checkpoint_manifest_digest",
        "hmac",
    }
    or pointer.get("schema_version") != 2
    or pointer.get("revision") != checkpoint_revision
    or pointer.get("run_identity") != run_identity
    or not isinstance(pointer.get("checkpoint"), str)
    or re.fullmatch(r"[A-Za-z0-9._-]+", pointer["checkpoint"]) is None
    or not isinstance(pointer.get("generation"), int)
    or isinstance(pointer.get("generation"), bool)
    or pointer["generation"] < 1
    or not isinstance(pointer.get("checkpoint_manifest_digest"), str)
    or digest_pattern.fullmatch(pointer["checkpoint_manifest_digest"]) is None
):
    raise ValueError("checkpoint pointer identity is invalid")
material = {key: value for key, value in pointer.items() if key != "hmac"}
expected_hmac = (
    "hmac-sha256:"
    + hmac.new(checkpoint_key, canonical(material), hashlib.sha256).hexdigest()
)
if not isinstance(pointer.get("hmac"), str) or not hmac.compare_digest(
    pointer["hmac"],
    expected_hmac,
):
    raise ValueError("checkpoint pointer authentication failed")
if (
    pointer["generation"] != checkpoint_generation
    or pointer["checkpoint_manifest_digest"] != checkpoint_manifest_digest
):
    raise ValueError("checkpoint pointer and structured progress do not match")

anchor_root_metadata = os.fstat(anchor_root_descriptor)
if (
    not stat.S_ISDIR(anchor_root_metadata.st_mode)
    or stat.S_IMODE(anchor_root_metadata.st_mode) != 0o700
    or anchor_root_metadata.st_uid != os.geteuid()
):
    raise ValueError("private checkpoint replay anchor root is unsafe")
anchor_name = (
    "checkpoint-"
    + hmac.new(
        checkpoint_key,
        b"equinox/checkpoint-replay-anchor-name/v1\0"
        + canonical(launch_identity),
        hashlib.sha256,
    ).hexdigest()
    + ".json"
)


def anchor_hmac(material):
    return (
        "hmac-sha256:"
        + hmac.new(
            checkpoint_key,
            b"equinox/checkpoint-replay-anchor/v1\0" + canonical(material),
            hashlib.sha256,
        ).hexdigest()
    )


def load_anchor():
    try:
        value, _ = canonical_json_at(
            anchor_root_descriptor,
            anchor_name,
            16 * 1024,
        )
    except FileNotFoundError:
        return None
    if (
        set(value)
        != {
            "schema_version",
            "revision",
            "proof_id",
            "run_identity",
            "launch_identity_digest",
            "checkpoint_generation",
            "checkpoint_manifest_digest",
            "hmac",
        }
        or value.get("schema_version") != 1
        or value.get("revision") != anchor_revision
        or value.get("proof_id") != proof_id
        or value.get("run_identity") != run_identity
        or value.get("launch_identity_digest") != launch_identity_digest
        or not isinstance(value.get("checkpoint_generation"), int)
        or isinstance(value.get("checkpoint_generation"), bool)
        or value["checkpoint_generation"] < 1
        or not isinstance(value.get("checkpoint_manifest_digest"), str)
        or digest_pattern.fullmatch(value["checkpoint_manifest_digest"]) is None
    ):
        raise ValueError("private checkpoint replay anchor identity is invalid")
    supplied_hmac = value["hmac"]
    anchor_material = {
        name: item for name, item in value.items() if name != "hmac"
    }
    if not isinstance(supplied_hmac, str) or not hmac.compare_digest(
        supplied_hmac,
        anchor_hmac(anchor_material),
    ):
        raise ValueError("private checkpoint replay anchor authentication failed")
    return value


def write_anchor():
    anchor_material = {
        "schema_version": 1,
        "revision": anchor_revision,
        "proof_id": proof_id,
        "run_identity": run_identity,
        "launch_identity_digest": launch_identity_digest,
        "checkpoint_generation": checkpoint_generation,
        "checkpoint_manifest_digest": checkpoint_manifest_digest,
    }
    anchor = {**anchor_material, "hmac": anchor_hmac(anchor_material)}
    payload = canonical(anchor) + b"\n"
    pending_name = ".checkpoint-replay-" + secrets.token_hex(16)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(
        pending_name,
        flags,
        0o600,
        dir_fd=anchor_root_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.link(
            pending_name,
            anchor_name,
            src_dir_fd=anchor_root_descriptor,
            dst_dir_fd=anchor_root_descriptor,
            follow_symlinks=False,
        )
        os.fsync(anchor_root_descriptor)
    finally:
        os.close(descriptor)
        try:
            os.unlink(pending_name, dir_fd=anchor_root_descriptor)
        except FileNotFoundError:
            pass


try:
    anchor = load_anchor()
    if mode == "capture" and anchor is None:
        try:
            write_anchor()
        except FileExistsError:
            pass
        anchor = load_anchor()
    if anchor is None:
        raise ValueError("private checkpoint replay anchor is missing")
    anchored_generation = anchor["checkpoint_generation"]
    anchored_digest = anchor["checkpoint_manifest_digest"]
    if checkpoint_generation < anchored_generation:
        raise ValueError("checkpoint replay generation regressed")
    if (
        checkpoint_generation != anchored_generation
        or checkpoint_manifest_digest != anchored_digest
    ):
        raise ValueError("checkpoint replay binding does not match its private anchor")
    print(checkpoint_generation, checkpoint_manifest_digest)
finally:
    os.close(anchor_root_descriptor)
PY
}

safe_archive_adapter() {
  local archive_python="${EQUINOX_ARCHIVE_PYTHON:-$filesystem_python}"
  "$archive_python" - 19 17 <<'PY'
import gzip
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tarfile

work_descriptor = int(sys.argv[1])
adapter_descriptor = int(sys.argv[2])
maximum_files = 4096
maximum_bytes = 4 * 1024 * 1024 * 1024
observed_files = 0
observed_bytes = 0


def read_regular_at(descriptor, name, maximum):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_descriptor = os.open(name, flags, dir_fd=descriptor)
    try:
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise RuntimeError(f"{name} has an unsafe type or size")
        payload = b""
        while len(payload) < before.st_size:
            chunk = os.read(
                file_descriptor,
                min(1024 * 1024, before.st_size - len(payload)),
            )
            if not chunk:
                raise RuntimeError(f"{name} was truncated")
            payload += chunk
        after = os.fstat(file_descriptor)
        path_metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            any(
                getattr(before, field) != getattr(after, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            or path_metadata.st_dev != before.st_dev
            or path_metadata.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{name} changed while being read")
        return payload
    finally:
        os.close(file_descriptor)


result_name = "result.pending.json"
try:
    result_payload = read_regular_at(work_descriptor, result_name, 64 * 1024 * 1024)
except FileNotFoundError:
    result_name = "result.json"
    result_payload = read_regular_at(work_descriptor, result_name, 64 * 1024 * 1024)
try:
    result = json.loads(
        result_payload,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {value}")
        ),
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise RuntimeError("completed result is not valid JSON") from error
if not isinstance(result, dict):
    raise RuntimeError("completed result is not an object")
expected_manifest = result.get("adapter_manifest")
adapter_persisted = result.get("adapter_persisted")


def collect(descriptor, prefix):
    global observed_bytes, observed_files
    entries = []
    for entry in sorted(os.scandir(descriptor), key=lambda item: item.name):
        metadata = entry.stat(follow_symlinks=False)
        relative = f"{prefix}/{entry.name}"
        if stat.S_ISDIR(metadata.st_mode):
            if relative == "adapter/checkpoints":
                continue
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(entry.name, flags, dir_fd=descriptor)
            try:
                entries.append((relative, metadata, None))
                entries.extend(collect(child, relative))
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode):
            observed_files += 1
            observed_bytes += metadata.st_size
            if observed_files > maximum_files or observed_bytes > maximum_bytes:
                raise RuntimeError("adapter archive exceeds its safety bound")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            file_descriptor = os.open(entry.name, flags, dir_fd=descriptor)
            entries.append((relative, metadata, file_descriptor))
        else:
            raise RuntimeError(f"unsafe adapter archive entry: {relative}")
    return entries


entries = collect(adapter_descriptor, "adapter")
try:
    regular_entries = {
        relative.removeprefix("adapter/"): (metadata, descriptor)
        for relative, metadata, descriptor in entries
        if descriptor is not None
    }
    directory_entries = {
        relative for relative, _metadata, descriptor in entries if descriptor is None
    }
    if adapter_persisted is not True:
        if expected_manifest is not None or regular_entries or directory_entries:
            raise RuntimeError(
                "adapter bytes exist without completed-result persistence evidence"
            )
        try:
            existing = os.stat(
                "adapter.tgz",
                dir_fd=work_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            raise RuntimeError("an adapter archive exists without adapter evidence")
        print("none")
        raise SystemExit(0)
    manifest_keys = {
        "schema_version",
        "model_id",
        "model_revision",
        "workload_revision",
        "objective_id",
        "training_configuration",
        "files",
        "digest",
    }
    if (
        not isinstance(expected_manifest, dict)
        or set(expected_manifest) != manifest_keys
        or expected_manifest.get("schema_version") != 1
        or not all(
            isinstance(expected_manifest.get(name), str)
            and bool(expected_manifest[name])
            for name in (
                "model_id",
                "model_revision",
                "workload_revision",
                "objective_id",
            )
        )
        or not isinstance(expected_manifest.get("training_configuration"), dict)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(expected_manifest.get("digest")),
        )
        is None
    ):
        raise RuntimeError("completed result adapter manifest schema is invalid")
    manifest_content = {
        name: value
        for name, value in expected_manifest.items()
        if name != "digest"
    }
    canonical_manifest_content = json.dumps(
        manifest_content,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if expected_manifest["digest"] != (
        "sha256:" + hashlib.sha256(canonical_manifest_content).hexdigest()
    ):
        raise RuntimeError("completed result adapter manifest digest is invalid")
    manifest_entry = regular_entries.get("adapter-manifest.json")
    if manifest_entry is None:
        raise RuntimeError("adapter tree omits adapter-manifest.json")
    manifest_metadata, manifest_descriptor = manifest_entry
    os.lseek(manifest_descriptor, 0, os.SEEK_SET)
    manifest_payload = os.read(manifest_descriptor, manifest_metadata.st_size + 1)
    os.lseek(manifest_descriptor, 0, os.SEEK_SET)
    canonical_expected_manifest = (
        json.dumps(
            expected_manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    if manifest_payload != canonical_expected_manifest:
        raise RuntimeError("adapter manifest differs from the completed result")
    files = expected_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("adapter manifest declares no files")
    declared = {}
    declared_bytes = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError("adapter manifest file evidence is invalid")
        path = item.get("path")
        parts = path.split("/") if isinstance(path, str) else []
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in parts)
            or path == "adapter-manifest.json"
            or path in declared
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None
        ):
            raise RuntimeError("adapter manifest file evidence is invalid")
        declared[path] = item
        declared_bytes += item["size_bytes"]
        if declared_bytes > maximum_bytes:
            raise RuntimeError("adapter manifest exceeds its byte bound")
    if set(regular_entries) != set(declared) | {"adapter-manifest.json"}:
        raise RuntimeError("adapter tree and manifest file sets differ")
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(declared):
        raise RuntimeError("adapter manifest omits required LoRA files")
    for path, item in declared.items():
        metadata, descriptor = regular_entries[path]
        if metadata.st_size != item["size_bytes"]:
            raise RuntimeError(f"adapter manifest size differs for {path}")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"adapter entry was truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            digest.hexdigest() != item["sha256"]
            or after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
        ):
            raise RuntimeError(f"adapter manifest digest differs for {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)

    try:
        existing = os.stat("adapter.tgz", dir_fd=work_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or existing.st_size <= 0:
            raise RuntimeError("existing adapter archive has an unsafe type")
    if existing is None:
        pending_name = ".adapter-" + secrets.token_hex(16) + ".tgz"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        output_descriptor = os.open(pending_name, flags, 0o400, dir_fd=work_descriptor)
        try:
            with os.fdopen(output_descriptor, "wb", closefd=False) as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    with tarfile.open(
                        fileobj=compressed,
                        mode="w|",
                        format=tarfile.PAX_FORMAT,
                    ) as archive:
                        root = tarfile.TarInfo("adapter")
                        root.type = tarfile.DIRTYPE
                        root.mode = 0o700
                        root.mtime = 0
                        archive.addfile(root)
                        for relative, metadata, file_descriptor in entries:
                            info = tarfile.TarInfo(relative)
                            info.mode = stat.S_IMODE(metadata.st_mode) & 0o777
                            info.mtime = 0
                            if file_descriptor is None:
                                info.type = tarfile.DIRTYPE
                                archive.addfile(info)
                            else:
                                current = os.fstat(file_descriptor)
                                if (
                                    not stat.S_ISREG(current.st_mode)
                                    or current.st_dev != metadata.st_dev
                                    or current.st_ino != metadata.st_ino
                                    or current.st_size != metadata.st_size
                                ):
                                    raise RuntimeError(
                                        "adapter entry changed before archiving"
                                    )
                                os.lseek(file_descriptor, 0, os.SEEK_SET)
                                info.size = current.st_size
                                with os.fdopen(os.dup(file_descriptor), "rb") as source:
                                    archive.addfile(info, source)
                raw.flush()
                os.fsync(raw.fileno())
            pending_metadata = os.stat(
                pending_name,
                dir_fd=work_descriptor,
                follow_symlinks=False,
            )
            output_metadata = os.fstat(output_descriptor)
            if (
                pending_metadata.st_dev != output_metadata.st_dev
                or pending_metadata.st_ino != output_metadata.st_ino
            ):
                raise RuntimeError("adapter archive changed before publication")
            os.replace(
                pending_name,
                "adapter.tgz",
                src_dir_fd=work_descriptor,
                dst_dir_fd=work_descriptor,
            )
            os.fsync(work_descriptor)
            published = os.stat(
                "adapter.tgz",
                dir_fd=work_descriptor,
                follow_symlinks=False,
            )
            if (
                published.st_dev != output_metadata.st_dev
                or published.st_ino != output_metadata.st_ino
            ):
                raise RuntimeError("adapter archive publication changed inode identity")
        finally:
            os.close(output_descriptor)
            try:
                os.unlink(pending_name, dir_fd=work_descriptor)
            except FileNotFoundError:
                pass

    archive_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    archive_descriptor = os.open("adapter.tgz", archive_flags, dir_fd=work_descriptor)
    try:
        archive_before = os.fstat(archive_descriptor)
        if (
            not stat.S_ISREG(archive_before.st_mode)
            or archive_before.st_size <= 0
            or archive_before.st_size > maximum_bytes
        ):
            raise RuntimeError("adapter archive has an unsafe type or size")
        expected_archive_files = {
            "adapter/" + path: item for path, item in declared.items()
        }
        expected_archive_files["adapter/adapter-manifest.json"] = {
            "size_bytes": len(canonical_expected_manifest),
            "sha256": hashlib.sha256(canonical_expected_manifest).hexdigest(),
        }
        expected_archive_directories = {"adapter", *directory_entries}
        with os.fdopen(os.dup(archive_descriptor), "rb") as raw_archive:
            with tarfile.open(fileobj=raw_archive, mode="r:gz") as archive:
                members = archive.getmembers()
                names = {member.name for member in members}
                if len(names) != len(members):
                    raise RuntimeError("adapter archive has duplicate members")
                observed_archive_files = set()
                observed_archive_directories = set()
                for member in members:
                    if member.isdir():
                        if member.size != 0:
                            raise RuntimeError("adapter archive directory is invalid")
                        observed_archive_directories.add(member.name.rstrip("/"))
                        continue
                    if not member.isfile() or member.issparse():
                        raise RuntimeError("adapter archive contains a special member")
                    expected = expected_archive_files.get(member.name)
                    if expected is None or member.size != expected["size_bytes"]:
                        raise RuntimeError("adapter archive file evidence differs")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("adapter archive file could not be read")
                    digest = hashlib.sha256()
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError("adapter archive member was truncated")
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if digest.hexdigest() != expected["sha256"]:
                        raise RuntimeError("adapter archive member digest differs")
                    observed_archive_files.add(member.name)
                if (
                    observed_archive_files != set(expected_archive_files)
                    or observed_archive_directories != expected_archive_directories
                ):
                    raise RuntimeError("adapter archive member sets differ")
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        archive_digest = hashlib.sha256()
        remaining = archive_before.st_size
        while remaining:
            chunk = os.read(archive_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("adapter archive was truncated")
            archive_digest.update(chunk)
            remaining -= len(chunk)
        archive_after = os.fstat(archive_descriptor)
        archive_path = os.stat(
            "adapter.tgz",
            dir_fd=work_descriptor,
            follow_symlinks=False,
        )
        if (
            any(
                getattr(archive_before, field) != getattr(archive_after, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            or archive_path.st_dev != archive_before.st_dev
            or archive_path.st_ino != archive_before.st_ino
        ):
            raise RuntimeError("adapter archive changed while being verified")
        print(
            f"{archive_before.st_dev}:{archive_before.st_ino}:"
            f"{archive_before.st_size}:sha256:{archive_digest.hexdigest()}"
        )
    finally:
        os.close(archive_descriptor)
finally:
    for _, _, descriptor in entries:
        if descriptor is not None:
            os.close(descriptor)
PY
}

safe_remove_adapter_checkpoints() {
  if [[ "$adapter_enabled" != "true" ]]; then
    return 0
  fi
  "$filesystem_python" - 17 <<'PY'
import os
import stat
import sys

adapter_descriptor = int(sys.argv[1])
directory_flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def remove_directory(parent_descriptor, name):
    descriptor = os.open(name, directory_flags, dir_fd=parent_descriptor)
    try:
        for entry in os.scandir(descriptor):
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                remove_directory(descriptor, entry.name)
            elif stat.S_ISREG(metadata.st_mode):
                os.unlink(entry.name, dir_fd=descriptor)
            else:
                raise RuntimeError(f"unsafe checkpoint cleanup entry: {entry.name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


try:
    remove_directory(adapter_descriptor, "checkpoints")
except FileNotFoundError:
    pass
PY
  local cleanup_status="$?"
  if [[ "$cleanup_status" != "0" ]]; then
    return "$cleanup_status"
  fi
}

serve_transport_failure_and_exit() {
  local failure_exit_code="$1"
  if [[ -z "${http_server_pid:-}" ]]; then
    EQUINOX_REMOTE_WORKDIR_FD=19 \
      python3 "$code_root/result_server.py" \
      >/dev/null 2>&1 &
    http_server_pid="$!"
  fi
  wait "$http_server_pid"
  exit "$failure_exit_code"
}

merge_attempt_error_log() {
  local attempt_error_name="$1"
  local attempt="$2"
  local outcome="$3"
  "${EQUINOX_ERROR_MERGE_PYTHON:-${EQUINOX_DURABILITY_PYTHON:-python3}}" - \
    19 \
    "error.log" \
    "$attempt_error_name" \
    "$attempt" \
    "$outcome" <<'PY'
import os
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
error_name, attempt_name, attempt, outcome = sys.argv[2:]
marker = f"=== attempt {attempt} · {outcome} ===\n".encode()
attempt_prefix = f"=== attempt {attempt} ".encode()


def read_regular(name, maximum):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise RuntimeError("attempt log has an unsafe type or size")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("attempt log was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


try:
    existing = read_regular(error_name, 64 * 1024 * 1024)
except FileNotFoundError:
    existing = b""
if attempt_prefix not in existing:
    attempt_error = read_regular(attempt_name, 64 * 1024 * 1024)
    pending_name = ".error-merge-" + secrets.token_hex(16)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(pending_name, flags, 0o600, dir_fd=directory_descriptor)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(existing)
            handle.write(marker)
            handle.write(attempt_error)
            if attempt_error and not attempt_error.endswith(b"\n"):
                handle.write(b"\n")
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            pending_name,
            error_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        os.close(descriptor)
        try:
            os.unlink(pending_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
os.unlink(attempt_name, dir_fd=directory_descriptor)
os.fsync(directory_descriptor)
PY
}

merge_attempt_error_log_fallback() {
  local attempt_error_name="$1"
  local attempt="$2"
  local outcome="$3"
  "$filesystem_python" - \
    19 \
    "error.log" \
    "$attempt_error_name" \
    "$attempt" \
    "$outcome" <<'PY'
import os
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
error_name, attempt_name, attempt, outcome = sys.argv[2:]


def read_regular(name):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return b""
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024 * 1024:
            raise RuntimeError("fallback log source has an unsafe type or size")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, metadata.st_size - len(payload)))
            if not chunk:
                raise RuntimeError("fallback log source was truncated")
            payload += chunk
        return payload
    finally:
        os.close(descriptor)


existing = read_regular(error_name)
attempt_error = read_regular(attempt_name)
marker = f"=== attempt {attempt} · {outcome} · non-durable fallback ===\n".encode()
payload = existing + marker + attempt_error
if attempt_error and not attempt_error.endswith(b"\n"):
    payload += b"\n"
payload += b"\n"
pending_name = ".error-fallback-" + secrets.token_hex(16)
flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
descriptor = os.open(pending_name, flags, 0o600, dir_fd=directory_descriptor)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(
        pending_name,
        error_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
try:
    os.unlink(attempt_name, dir_fd=directory_descriptor)
except FileNotFoundError:
    pass
os.fsync(directory_descriptor)
PY
}

merge_attempt_error_log_or_fallback() {
  local attempt_error_name="$1"
  local attempt="$2"
  local outcome="$3"
  if merge_attempt_error_log "$attempt_error_name" "$attempt" "$outcome"; then
    :
  else
    merge_attempt_error_log_fallback "$attempt_error_name" "$attempt" "$outcome"
  fi
  update_resume_manifest update "error.log" "$attempt_error_name"
}

write_progress() {
  local schema_version="$1"
  local phase="$2"
  local message="$3"
  local attempt="$4"
  local error_code="${5:-}"
  local preserve_context="${6:-false}"
  if ! "${EQUINOX_DURABILITY_PYTHON:-python3}" - \
    19 \
    "$schema_version" \
    "$phase" \
    "$message" \
    "$attempt" \
    "$error_code" \
    "$preserve_context" \
    "$branch_width" \
    "${requires_live_stage_handoff:-false}" <<'PY'
import json
import os
import secrets
import stat
import sys

(
    directory_descriptor_raw,
    schema_version,
    phase,
    message,
    attempt,
    error_code,
    preserve_context,
    branch_width,
    requires_live_stage_handoff,
) = sys.argv[1:]
directory_descriptor = int(directory_descriptor_raw)
live_name = "progress.json"
handoff_name = "live-stage-handoff.json"


def load_optional_json(name):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024 * 1024:
            raise ValueError(f"{name} has an unsafe type or size")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, metadata.st_size - len(payload)))
            if not chunk:
                raise ValueError(f"{name} was truncated")
            payload += chunk
        value = json.loads(payload)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise TypeError(f"{name} is not an object")
    return value


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
if (
    requires_live_stage_handoff != "true"
    and load_optional_json(handoff_name) is None
):
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
                "workload_bundle_path": os.environ.get(
                    "EQUINOX_BUNDLE_VOLUME_PATH"
                ),
                "bundle_stage_receipt_digest": os.environ.get(
                    "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256"
                ),
                "bootstrap_source_digest": os.environ.get(
                    "EQUINOX_BOOTSTRAP_SOURCE_SHA256"
                ),
                "network_volume_id": os.environ.get(
                    "EQUINOX_RUNPOD_NETWORK_VOLUME_ID"
                ),
            }
        )
if preserve_context == "true":
    try:
        previous = load_optional_json(live_name) or {}
    except (OSError, ValueError, TypeError):
        previous = {}
    if isinstance(previous, dict):
        payload = {**previous, **payload}
live_stage_handoff = load_optional_json(handoff_name)
if live_stage_handoff is not None:
    payload.update(live_stage_handoff)
if phase != "failed":
    payload.pop("error", None)
    payload.pop("remote_error", None)
    payload.pop("operator_error", None)
encoded = (
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
pending_name = ".progress-" + secrets.token_hex(16)
flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
descriptor = os.open(pending_name, flags, 0o600, dir_fd=directory_descriptor)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        existing = os.stat(live_name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise RuntimeError("structured progress target has an unsafe type")
    os.replace(
        pending_name,
        live_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
PY
  then
    safe_write_line "error.log" "Structured progress serialization failed."
    safe_write_line "exit_code" 70
    serve_transport_failure_and_exit 70
  fi
  if ! update_resume_manifest update "progress.json"; then
    printf '%s\n' "Structured progress authentication failed." >&2
    exit 70
  fi
}

validate_private_materialization() {
  "$filesystem_python" - 14 18 16 <<'PY'
import hashlib
import json
import os
import re
import stat
import sys

work_descriptor = int(sys.argv[1])
code_descriptor = int(sys.argv[2])
dependency_descriptor = int(sys.argv[3])
digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
maximum_files = 200_000
maximum_bytes = 16 * 1024 * 1024 * 1024


def canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def required_environment(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is missing")
    return value


def required_digest(name):
    value = required_environment(name)
    if digest_pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is not a tagged SHA-256 digest")
    return value


def read_json_at(name):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=work_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024 * 1024:
            raise ValueError(f"{name} has an unsafe type or size")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, metadata.st_size - len(payload)))
            if not chunk:
                raise ValueError(f"{name} was truncated")
            payload += chunk
        value = json.loads(payload)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ValueError(f"{name} is not one canonical JSON object")
    return value


def scan_tree(root_descriptor):
    entries = []
    installed_bytes = 0

    def visit(descriptor, prefix):
        nonlocal installed_bytes
        for entry in sorted(os.scandir(descriptor), key=lambda item: item.name):
            if entry.name in {"", ".", ".."} or "/" in entry.name:
                raise ValueError("private tree contains an unsafe name")
            metadata = entry.stat(follow_symlinks=False)
            path = f"{prefix}/{entry.name}" if prefix else entry.name
            if stat.S_ISDIR(metadata.st_mode):
                if metadata.st_mode & 0o222:
                    raise ValueError(f"private directory is writable: {path}")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                child = os.open(entry.name, flags, dir_fd=descriptor)
                try:
                    visit(child, path)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
                raise ValueError(f"private file has an unsafe type or mode: {path}")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            file_descriptor = os.open(entry.name, flags, dir_fd=descriptor)
            digest = hashlib.sha256()
            try:
                before = os.fstat(file_descriptor)
                remaining = before.st_size
                while remaining:
                    chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"private file was truncated: {path}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                after = os.fstat(file_descriptor)
            finally:
                os.close(file_descriptor)
            stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in stable):
                raise ValueError(f"private file changed while being read: {path}")
            entries.append(
                {
                    "path": path,
                    "size_bytes": before.st_size,
                    "sha256": "sha256:" + digest.hexdigest(),
                }
            )
            installed_bytes += before.st_size
            if len(entries) > maximum_files or installed_bytes > maximum_bytes:
                raise ValueError("private tree exceeds its safety bound")

    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_mode & 0o222:
        raise ValueError("private tree root has an unsafe type or mode")
    visit(root_descriptor, "")
    return entries, installed_bytes


def verify_root_identity(evidence, descriptor, kind):
    private_tree_digest = evidence.get("private_tree_digest")
    if (
        not isinstance(private_tree_digest, str)
        or digest_pattern.fullmatch(private_tree_digest) is None
        or evidence.get("source_tree_digest") != private_tree_digest
    ):
        raise ValueError(f"{kind} tree digest is invalid")
    expected_root = (
        f"/tmp/equinox-quarantine/{kind}/"
        f"{private_tree_digest.removeprefix('sha256:')}"
    )
    if evidence.get("private_root") != expected_root:
        raise ValueError(f"{kind} root is not content-addressed")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    expected_descriptor = os.open(expected_root, flags)
    try:
        expected_metadata = os.fstat(expected_descriptor)
        observed_metadata = os.fstat(descriptor)
        if (
            expected_metadata.st_dev != observed_metadata.st_dev
            or expected_metadata.st_ino != observed_metadata.st_ino
        ):
            raise ValueError(f"{kind} descriptor does not match its receipt")
    finally:
        os.close(expected_descriptor)
    return private_tree_digest


code_path = required_environment("EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH")
dependency_path = required_environment("EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH")
if os.path.basename(code_path) != "code-materialization-evidence.json":
    raise ValueError("code evidence path is invalid")
if os.path.basename(dependency_path) != "dependency-quarantine-evidence.json":
    raise ValueError("dependency evidence path is invalid")
code = read_json_at("code-materialization-evidence.json")
dependencies = read_json_at("dependency-quarantine-evidence.json")
dependency_lock_digest = required_digest("EQUINOX_DEPENDENCY_LOCK_SHA256")
if (
    dependency_lock_digest
    != "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
):
    raise ValueError("dependency lock identity is unsupported")
for evidence, expected_revision, digest_name, tree_name, descriptor, kind in (
    (
        code,
        "private-code-materialization@1",
        "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256",
        "EQUINOX_CODE_PRIVATE_TREE_SHA256",
        code_descriptor,
        "code",
    ),
    (
        dependencies,
        "hash-locked-private-dependencies@1",
        "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256",
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
        dependency_descriptor,
        "dependencies",
    ),
):
    if evidence.get("schema_version") != 1 or evidence.get("revision") != expected_revision:
        raise ValueError(f"{kind} materialization revision is invalid")
    if evidence.get("profile_id") != required_environment(
        "EQUINOX_LARGER_MODEL_PROFILE_ID"
    ) or evidence.get("ready") is not True:
        raise ValueError(f"{kind} materialization identity is invalid")
    material = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    observed_evidence_digest = "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()
    if (
        evidence.get("evidence_digest") != observed_evidence_digest
        or observed_evidence_digest != required_digest(digest_name)
    ):
        raise ValueError(f"{kind} materialization evidence digest is invalid")
    private_tree_digest = verify_root_identity(evidence, descriptor, kind)
    if private_tree_digest != required_digest(tree_name):
        raise ValueError(f"{kind} private tree digest changed")
    entries, installed_bytes = scan_tree(descriptor)
    observed_tree_digest = "sha256:" + hashlib.sha256(canonical_json(entries)).hexdigest()
    if observed_tree_digest != private_tree_digest:
        raise ValueError(f"{kind} private tree bytes changed")
    if (
        evidence.get("installed_file_count") != len(entries)
        or evidence.get("installed_bytes") != installed_bytes
    ):
        raise ValueError(f"{kind} private tree totals changed")
    if kind == "code":
        if evidence.get("files") != entries:
            raise ValueError("code private file inventory changed")
        lock_entries = [
            entry
            for entry in entries
            if entry.get("path") == "larger-model-dependencies.lock"
        ]
        if (
            len(lock_entries) != 1
            or lock_entries[0].get("sha256") != dependency_lock_digest
        ):
            raise ValueError("code tree does not contain the authorized dependency lock")
        if (
            evidence.get("bundle_digest")
            != required_digest("EQUINOX_BUNDLE_SHA256")
            or evidence.get("bundle_size_bytes")
            != int(required_environment("EQUINOX_BUNDLE_SIZE_BYTES"))
            or evidence.get("source_contract_digest")
            != required_digest("EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256")
            or evidence.get("source_root")
            != required_environment("EQUINOX_BUNDLE_VOLUME_PATH")
        ):
            raise ValueError("code materialization does not bind the workload bundle")
PY
}

validate_live_stage_handoff() {
  if ! "${EQUINOX_DURABILITY_PYTHON:-python3}" - \
    19 <<'PY'
import hashlib
import json
import os
import re
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
identity_name = "live-stage-handoff.json"
progress_name = "progress.json"
activation_revision = "authenticated-proxy-stage-activation@2"
bundle_handoff_revision = "runpod-volume-bundle-handoff@1"
maximum_bundle_bytes = 2 * 1024 * 1024
digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
safe_identity_pattern = re.compile(r"[A-Za-z0-9._@-]+")
safe_provider_pattern = re.compile(r"[A-Za-z0-9._-]+")


def required_environment(name):
    value = os.environ.get(name)
    if value is None or not value:
        raise ValueError(f"{name} is missing")
    return value


def required_digest(name):
    value = required_environment(name)
    if digest_pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is not a tagged SHA-256 digest")
    return value


def required_positive_integer(name, maximum=None):
    raw_value = required_environment(name)
    if re.fullmatch(r"[1-9][0-9]*", raw_value) is None:
        raise ValueError(f"{name} is not a positive integer")
    value = int(raw_value)
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return value


profile_id = required_environment("EQUINOX_LARGER_MODEL_PROFILE_ID")
if safe_identity_pattern.fullmatch(profile_id) is None:
    raise ValueError("EQUINOX_LARGER_MODEL_PROFILE_ID is malformed")
head_commit = required_environment("EQUINOX_SOURCE_HEAD_COMMIT")
if re.fullmatch(r"[0-9a-f]{40}", head_commit) is None:
    raise ValueError("EQUINOX_SOURCE_HEAD_COMMIT is not a full Git commit")
source_contract_digest = required_digest(
    "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256"
)
bootstrap_source_digest = required_digest("EQUINOX_BOOTSTRAP_SOURCE_SHA256")
workload_bundle_digest = required_digest("EQUINOX_BUNDLE_SHA256")
workload_bundle_size_bytes = required_positive_integer(
    "EQUINOX_BUNDLE_SIZE_BYTES",
    maximum_bundle_bytes,
)
workload_bundle_path = required_environment("EQUINOX_BUNDLE_VOLUME_PATH")
expected_bundle_path = (
    f"/workspace/equinox-state/workload-bundles/{profile_id}/"
    f"{workload_bundle_digest.removeprefix('sha256:')}.tar.xz"
)
if workload_bundle_path != expected_bundle_path:
    raise ValueError("EQUINOX_BUNDLE_VOLUME_PATH is not content-addressed")
bundle_stage_receipt_digest = required_digest(
    "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256"
)
volume_readiness_receipt_digest = required_digest(
    "EQUINOX_VOLUME_READINESS_RECEIPT_SHA256"
)
torch_retention_evidence_digest = required_digest(
    "EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256"
)
dependency_lock_digest = required_digest("EQUINOX_DEPENDENCY_LOCK_SHA256")
if (
    dependency_lock_digest
    != "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
):
    raise ValueError("EQUINOX_DEPENDENCY_LOCK_SHA256 is unsupported")
dependency_quarantine_revision = "hash-locked-private-dependencies@1"
dependency_quarantine_evidence_digest = required_digest(
    "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256"
)
dependency_private_tree_digest = required_digest(
    "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256"
)
code_materialization_revision = "private-code-materialization@1"
code_materialization_evidence_digest = required_digest(
    "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256"
)
code_private_tree_digest = required_digest("EQUINOX_CODE_PRIVATE_TREE_SHA256")
network_volume_id = required_environment("EQUINOX_RUNPOD_NETWORK_VOLUME_ID")
if safe_provider_pattern.fullmatch(network_volume_id) is None:
    raise ValueError("EQUINOX_RUNPOD_NETWORK_VOLUME_ID is malformed")
network_volume_data_center_id = required_environment(
    "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID"
)
if safe_provider_pattern.fullmatch(network_volume_data_center_id) is None:
    raise ValueError("EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID is malformed")
network_volume_size_gb = required_positive_integer(
    "EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB"
)
observed_bundle_handoff_revision = required_environment(
    "EQUINOX_BUNDLE_HANDOFF_REVISION"
)
if observed_bundle_handoff_revision != bundle_handoff_revision:
    raise ValueError("EQUINOX_BUNDLE_HANDOFF_REVISION is unsupported")

activation_body = {
    "revision": activation_revision,
    "profile_id": profile_id,
    "head_commit": head_commit,
    "source_contract_digest": source_contract_digest,
    "bootstrap_source_digest": bootstrap_source_digest,
    "workload_bundle_digest": workload_bundle_digest,
    "workload_bundle_size_bytes": workload_bundle_size_bytes,
    "workload_bundle_path": workload_bundle_path,
    "bundle_stage_receipt_digest": bundle_stage_receipt_digest,
    "volume_readiness_receipt_digest": volume_readiness_receipt_digest,
    "torch_retention_evidence_digest": torch_retention_evidence_digest,
    "dependency_lock_digest": dependency_lock_digest,
    "dependency_quarantine_revision": dependency_quarantine_revision,
    "dependency_quarantine_evidence_digest": dependency_quarantine_evidence_digest,
    "dependency_private_tree_digest": dependency_private_tree_digest,
    "code_materialization_revision": code_materialization_revision,
    "code_materialization_evidence_digest": code_materialization_evidence_digest,
    "code_private_tree_digest": code_private_tree_digest,
    "network_volume_id": network_volume_id,
    "network_volume_data_center_id": network_volume_data_center_id,
    "network_volume_size_gb": network_volume_size_gb,
}
serialized_activation = json.dumps(
    activation_body,
    sort_keys=True,
    separators=(",", ":"),
).encode()
expected_activation_digest = (
    "sha256:" + hashlib.sha256(serialized_activation).hexdigest()
)
observed_activation_digest = required_digest("EQUINOX_BUNDLE_ACTIVATION_DIGEST")
if observed_activation_digest != expected_activation_digest:
    raise ValueError("EQUINOX_BUNDLE_ACTIVATION_DIGEST does not bind the handoff")

identity = {
    key: value for key, value in activation_body.items() if key != "revision"
}
identity.update(
    {
        "live_stage_activation_revision": activation_revision,
        "bundle_handoff_revision": bundle_handoff_revision,
        "bundle_activation_digest": observed_activation_digest,
    }
)


def load_json_object(name, description):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} is unreadable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise ValueError(f"{description} has an unsafe type or size")
        payload_bytes = b""
        while len(payload_bytes) < metadata.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, metadata.st_size - len(payload_bytes)),
            )
            if not chunk:
                raise ValueError(f"{description} was truncated")
            payload_bytes += chunk
        payload = json.loads(payload_bytes)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} is unreadable") from error
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


try:
    identity_metadata = os.stat(
        identity_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
except FileNotFoundError:
    identity_metadata = None
if identity_metadata is not None:
    if not stat.S_ISREG(identity_metadata.st_mode):
        raise ValueError("persisted live-stage handoff has an unsafe type")
    if load_json_object(identity_name, "persisted live-stage handoff") != identity:
        raise ValueError("live-stage handoff changed across restart")
else:
    encoded_identity = (
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    pending_name = ".live-stage-handoff-" + secrets.token_hex(16)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    pending_descriptor = os.open(
        pending_name,
        flags,
        0o444,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(pending_descriptor, "wb", closefd=False) as handle:
            os.fchmod(handle.fileno(), 0o444)
            handle.write(encoded_identity)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                pending_name,
                identity_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = os.stat(
                identity_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or load_json_object(identity_name, "persisted live-stage handoff")
                != identity
            ):
                raise ValueError("live-stage handoff changed while being persisted")
        os.fsync(directory_descriptor)
    finally:
        os.close(pending_descriptor)
        try:
            os.unlink(pending_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass

try:
    progress_metadata = os.stat(
        progress_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
except FileNotFoundError:
    progress_metadata = None
if progress_metadata is not None:
    if not stat.S_ISREG(progress_metadata.st_mode):
        raise ValueError("structured progress has an unsafe type")
    progress = load_json_object(progress_name, "structured progress")
    for key, expected_value in identity.items():
        if progress.get(key) != expected_value:
            raise ValueError(f"structured progress changed live-stage field {key}")
PY
  then
    safe_write_line "error.log" "Authenticated live-stage handoff validation failed."
    return 1
  fi
  if ! update_resume_manifest update "live-stage-handoff.json"; then
    printf '%s\n' "The live-stage handoff could not be resume-authenticated." >&2
    return 1
  fi
}

structure_remote_failure() {
  local failure_exit_code="$1"
  if ! "${EQUINOX_DURABILITY_PYTHON:-python3}" - \
    19 \
    "$failure_exit_code" <<'PY'
import json
import os
import re
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
exit_code = sys.argv[2]
name = "progress.json"
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(name, flags, dir_fd=directory_descriptor)
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024 * 1024:
        raise RuntimeError("structured progress has an unsafe type or size")
    payload_bytes = b""
    while len(payload_bytes) < metadata.st_size:
        chunk = os.read(
            descriptor,
            min(1024 * 1024, metadata.st_size - len(payload_bytes)),
        )
        if not chunk:
            raise RuntimeError("structured progress was truncated")
        payload_bytes += chunk
    payload = json.loads(payload_bytes)
finally:
    os.close(descriptor)
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
encoded = (
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
).encode()
pending_name = ".progress-failure-" + secrets.token_hex(16)
write_flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
pending_descriptor = os.open(
    pending_name,
    write_flags,
    0o600,
    dir_fd=directory_descriptor,
)
try:
    with os.fdopen(pending_descriptor, "wb", closefd=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(
        pending_name,
        name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(pending_descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
PY
  then
    safe_write_line "error.log" "Structured remote failure enrichment failed."
    return 1
  fi
  update_resume_manifest update "progress.json"
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
  safe_write_line "exit_code" 1
  EQUINOX_REMOTE_WORKDIR_FD=19 \
    python3 "$code_root/result_server.py" \
    >/dev/null 2>&1 &
  wait "$!"
  exit 1
}

case "$workload_file" in
  repository_repair_large_model_eligibility.py | repository_repair_large_model_pilot.py)
    progress_schema_version=2
    repository_workload=true
    requires_live_stage_handoff=true
    ;;
  repository_repair_rl.py | repository_repair_study.py | repository_repair_study_v31.py | repository_repair_eligibility.py)
    progress_schema_version=2
    repository_workload=true
    requires_live_stage_handoff=false
    ;;
  branching_sequence_ladder.py)
    progress_schema_version=1
    repository_workload=false
    requires_live_stage_handoff=false
    ;;
  *)
    serve_boot_failure \
      1 \
      "UNSUPPORTED_WORKLOAD_FILE" \
      "The remote runner received an unsupported workload file."
    ;;
esac

if [[ "$requires_live_stage_handoff" == "true" ]] &&
  ! validate_private_materialization; then
  serve_boot_failure \
    "$progress_schema_version" \
    "PRIVATE_MATERIALIZATION_INVALID" \
    "The private code or dependency materialization is missing, mutable, or inconsistent."
fi

if [[ "$requires_live_stage_handoff" == "true" ]] &&
  ! validate_live_stage_handoff; then
  serve_boot_failure \
    "$progress_schema_version" \
    "LIVE_STAGE_HANDOFF_INVALID" \
    "The authenticated live-stage handoff evidence is missing, malformed, or inconsistent."
fi

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
  local attempt_error_name="$1"
  if safe_read_regular "$attempt_error_name" 67108864 | grep -Eiq \
    'CUDA([^[:cntrl:]]*)out of memory|CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED' \
    -; then
    printf '%s\n' "GPU_MEMORY_EXHAUSTED"
    return 0
  fi
  if safe_read_regular "$attempt_error_name" 67108864 | grep -Eiq \
    'MODEL_LOAD_FAILED|MODEL_LOAD_TIMEOUT|model load(ing)? failed|failed to load (the )?model|RevisionNotFoundError|RepositoryNotFoundError|GatedRepoError|not a valid model identifier|safetensors(_rust)?[^[:cntrl:]]*(error|exception)' \
    -; then
    printf '%s\n' "MODEL_LOAD_FAILED"
    return 0
  fi
  return 1
}

persist_non_retryable_failure() {
  local failure_code="$1"
  safe_write_line "non-retryable-failure" "$failure_code"
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

if ! safe_regular_exists "error.log"; then
  safe_atomic_write "error.log"
fi
for interrupted_attempt in 1 2; do
  interrupted_error_name="error.attempt-$interrupted_attempt.log"
  if ! safe_regular_exists "$interrupted_error_name"; then
    continue
  fi
  interrupted_non_retryable_failure=""
  if interrupted_non_retryable_failure="$(
    non_retryable_failure_code "$interrupted_error_name"
  )"; then
    persist_non_retryable_failure "$interrupted_non_retryable_failure"
  fi
  merge_attempt_error_log_or_fallback \
    "$interrupted_error_name" \
    "$interrupted_attempt" \
    "interrupted before exit"
done

recorded_attempts=0
if safe_regular_exists "workload-attempt-count"; then
  recorded_attempts="$(safe_read_regular "workload-attempt-count" 16)"
  if [[ ! "$recorded_attempts" =~ ^[0-2]$ ]]; then
    serve_boot_failure \
      "$progress_schema_version" \
      "WORKLOAD_ATTEMPT_STATE_INVALID" \
      "The persisted workload attempt state is invalid."
  fi
fi

completed_result_identity() {
  local result_name="$1"
  "$filesystem_python" - 19 "$workload_file" "$result_name" <<'PY'
import hashlib
import json
import os
import stat
import sys

directory_descriptor = int(sys.argv[1])
workload = sys.argv[2]
result_name = sys.argv[3]
if result_name not in {"result.pending.json", "result.json"}:
    raise SystemExit("completed result name is invalid")
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(result_name, flags, dir_fd=directory_descriptor)


def reject_constant(_value):
    raise ValueError("non-finite JSON number")


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def validate_value(value, *, depth=0, budget=None):
    if budget is None:
        budget = [0]
    budget[0] += 1
    if depth > 64 or budget[0] > 2_000_000:
        raise ValueError("completed result structure exceeds its bound")
    if value is None or type(value) in {bool, int, float}:
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("completed result string exceeds its bound")
        return
    if isinstance(value, list):
        for item in value:
            validate_value(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key.encode("utf-8")) > 4096:
                raise ValueError("completed result key is invalid")
            validate_value(item, depth=depth + 1, budget=budget)
        return
    raise ValueError("completed result contains an unsupported value")


try:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > 64 * 1024 * 1024
    ):
        raise ValueError("completed result has an unsafe type or size")
    payload = b""
    digest = hashlib.sha256()
    while len(payload) < before.st_size:
        chunk = os.read(descriptor, min(1024 * 1024, before.st_size - len(payload)))
        if not chunk:
            raise ValueError("completed result was truncated")
        digest.update(chunk)
        payload += chunk
    after = os.fstat(descriptor)
    result = json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
finally:
    os.close(descriptor)
path_metadata = os.stat(result_name, dir_fd=directory_descriptor, follow_symlinks=False)
stable_fields = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
if (
    any(getattr(before, field) != getattr(after, field) for field in stable_fields)
    or path_metadata.st_dev != before.st_dev
    or path_metadata.st_ino != before.st_ino
):
    raise ValueError("completed result changed while being validated")
if not isinstance(result, dict):
    raise ValueError("completed result is not an object")
validate_value(result)
schema_version = result.get("schema_version")
if schema_version is not None and (
    type(schema_version) is not int or schema_version < 1
):
    raise ValueError("completed result schema version is invalid")
declared_workload = result.get("workload")
if declared_workload is not None and (
    not isinstance(declared_workload, str)
    or not declared_workload
    or len(declared_workload) > 256
):
    raise ValueError("completed result workload identity is invalid")
if workload in {
    "repository_repair_rl.py",
    "repository_repair_study.py",
    "repository_repair_study_v31.py",
    "repository_repair_large_model_pilot.py",
}:
    if result.get("experiment_completed") is not True:
        raise ValueError("completed result lacks its experiment completion marker")
if workload in {
    "repository_repair_eligibility.py",
    "repository_repair_large_model_eligibility.py",
}:
    if result.get("screen_completed") is not True:
        raise ValueError("completed result lacks its screen completion marker")
if (
    workload == "branching_sequence_ladder.py"
    and result.get("workload") != "branching-sequence-policy-gradient-ladder"
):
    raise ValueError("completed result has the wrong workload identity")
adapter_persisted = result.get("adapter_persisted")
adapter_manifest = result.get("adapter_manifest")
if adapter_persisted is True and not isinstance(adapter_manifest, dict):
    raise ValueError("completed result omits its adapter manifest")
if adapter_persisted is False and adapter_manifest is not None:
    raise ValueError("completed result has contradictory adapter evidence")
print(
    f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
    f"sha256:{digest.hexdigest()}"
)
PY
}

completed_pending_result_available() {
  if ! safe_regular_exists "result.pending.json" ||
    safe_regular_exists "result.json"; then
    return 1
  fi
  if completed_result_identity "result.pending.json" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

discard_uncommitted_pending_result() {
  "$filesystem_python" - 19 <<'PY'
import os
import stat
import sys

directory_descriptor = int(sys.argv[1])
name = "result.pending.json"
try:
    metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
except FileNotFoundError:
    raise SystemExit(0)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("uncommitted result target has an unsafe type")
os.unlink(name, dir_fd=directory_descriptor)
os.fsync(directory_descriptor)
PY
}

promote_completed_result() {
  local expected_identity="$1"
  "$filesystem_python" - 19 "$expected_identity" <<'PY'
import hashlib
import os
import re
import stat
import sys

directory_descriptor = int(sys.argv[1])
expected_identity = sys.argv[2]
if re.fullmatch(
    r"[0-9]+:[0-9]+:[0-9]+:sha256:[0-9a-f]{64}",
    expected_identity,
) is None:
    raise SystemExit("completed result commit identity is invalid")
source = "result.pending.json"
destination = "result.json"
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source, flags, dir_fd=directory_descriptor)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError("completed result source has an unsafe type or size")
    digest = hashlib.sha256()
    remaining = before.st_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("completed result source was truncated")
        digest.update(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    observed_identity = (
        f"{before.st_dev}:{before.st_ino}:{before.st_size}:"
        f"sha256:{digest.hexdigest()}"
    )
    source_path = os.stat(source, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        observed_identity != expected_identity
        or any(
            getattr(before, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )
        or source_path.st_dev != before.st_dev
        or source_path.st_ino != before.st_ino
    ):
        raise ValueError("completed result changed before publication")
    try:
        destination_metadata = os.stat(
            destination,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        destination_metadata = None
    if destination_metadata is not None:
        raise ValueError("completed result destination already exists")
    os.replace(
        source,
        destination,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
    published = os.stat(destination, dir_fd=directory_descriptor, follow_symlinks=False)
    if published.st_dev != before.st_dev or published.st_ino != before.st_ino:
        raise ValueError("completed result publication changed inode identity")
    print(observed_identity)
finally:
    os.close(descriptor)
PY
}

progress_is_failed() {
  safe_read_regular "progress.json" 67108864 2>/dev/null |
    grep -Eq '"phase"[[:space:]]*:[[:space:]]*"failed"'
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
pending_result_commit_identity=""
if completed_pending_result_available; then
  if ! pending_result_commit_identity="$(
    completed_result_identity "result.pending.json"
  )"; then
    printf '%s\n' "The recovered completed result could not be pinned." >&2
    exit 74
  fi
  pending_result_completed=true
  pending_result_recovered=true
  write_recovered_result_progress \
    "finalizing" \
    "Recovered a completed result; packaging artifacts."
elif safe_regular_exists "result.json"; then
  write_recovered_result_progress \
    "complete" \
    "Previously published result and artifacts are available."
fi

retry_allowed=true
terminal_error=""
terminal_message=""
if [[ "$pending_result_completed" == "true" ]] ||
  safe_regular_exists "result.json"; then
  retry_allowed=false
elif safe_regular_exists "non-retryable-failure"; then
  persisted_non_retryable_failure="$(
    safe_read_regular "non-retryable-failure" 128
  )"
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
  ! adapter_checkpoint_available; then
  retry_allowed=false
  terminal_error="WORKLOAD_FAILED_BEFORE_FIRST_CHECKPOINT"
  terminal_message="The workload failed before producing a resumable checkpoint."
fi

if [[ "$retry_allowed" == "true" || "$pending_result_completed" == "true" ]]; then
  safe_unlink_regular "exit_code"
fi

if ! safe_regular_exists "result.json" &&
  [[ "$pending_result_completed" != "true" ]]; then
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
    if ! progress_is_failed; then
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
EQUINOX_REMOTE_WORKDIR_FD=19 \
  python3 "$code_root/result_server.py" \
  >/dev/null 2>&1 &
http_server_pid="$!"

scientific_workload_command=(python3 "$code_root/$workload_file")
if [[ "$requires_live_stage_handoff" == "true" ]]; then
  scientific_workload_command=(
    python3
    -c
    '
import json
import os
import runpy
import sys

workload_path = os.path.abspath(sys.argv[1])
identity_path = os.path.abspath(sys.argv[2])
progress_path = os.path.abspath(os.environ["EQUINOX_PROGRESS_PATH"])
with open(identity_path, encoding="utf-8") as handle:
    live_stage_handoff = json.load(handle)
if not isinstance(live_stage_handoff, dict):
    raise TypeError("The persisted live-stage handoff is not an object.")

replace = os.replace


def replace_with_live_stage_handoff(source, destination, *args, **kwargs):
    if (
        not args
        and not kwargs
        and os.path.abspath(os.fspath(destination)) == progress_path
    ):
        with open(source, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Structured scientific progress is not an object.")
        payload.update(live_stage_handoff)
        with open(source, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return replace(source, destination, *args, **kwargs)


os.replace = replace_with_live_stage_handoff
sys.path.insert(0, os.path.dirname(workload_path))
sys.argv = [workload_path, *sys.argv[3:]]
runpy.run_path(workload_path, run_name="__main__")
'
    "$code_root/$workload_file"
    "$live_stage_handoff_path"
  )
fi

run_scientific_attempt() {
  local attempt_error_name="$1"
  shift
  "$filesystem_python" - 19 "$attempt_error_name" "$@" <<'PY'
import os
import secrets
import stat
import subprocess
import sys

directory_descriptor = int(sys.argv[1])
attempt_error_name = sys.argv[2]
command = sys.argv[3:]
if not command:
    raise SystemExit("scientific command is empty")
if not attempt_error_name.startswith("error.attempt-") or not attempt_error_name.endswith(".log"):
    raise SystemExit("attempt error name is invalid")
result_name = "result.pending.json"
try:
    existing_result = os.stat(
        result_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
except FileNotFoundError:
    existing_result = None
if existing_result is not None:
    if not stat.S_ISREG(existing_result.st_mode):
        raise SystemExit("pending result target has an unsafe type")
    os.unlink(result_name, dir_fd=directory_descriptor)
try:
    existing_error = os.stat(
        attempt_error_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
except FileNotFoundError:
    existing_error = None
if existing_error is not None:
    raise SystemExit("attempt error target already exists")

result_pending_name = ".result-" + secrets.token_hex(16)
write_flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
result_descriptor = os.open(
    result_pending_name,
    write_flags,
    0o600,
    dir_fd=directory_descriptor,
)
error_descriptor = os.open(
    attempt_error_name,
    write_flags,
    0o600,
    dir_fd=directory_descriptor,
)
try:
    inherited = []
    for candidate in (14, 16, 17, 18, 19):
        try:
            os.fstat(candidate)
        except OSError:
            continue
        inherited.append(candidate)
    child_environment = dict(os.environ)
    for private_name in (
        "EQUINOX_RESULT_TOKEN",
        "EQUINOX_RUNTIME_ANCHOR_ROOT",
        "EQUINOX_RUNTIME_ANCHOR_ROOT_FD",
        "EQUINOX_PRIVATE_ANCHOR_REVISION",
    ):
        child_environment.pop(private_name, None)
    completed = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=result_descriptor,
        stderr=error_descriptor,
        close_fds=True,
        pass_fds=tuple(inherited),
        env=child_environment,
    )
    os.fsync(result_descriptor)
    os.fsync(error_descriptor)
    os.replace(
        result_pending_name,
        result_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(result_descriptor)
    os.close(error_descriptor)
    try:
        os.unlink(result_pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
raise SystemExit(completed.returncode)
PY
}

workload_exit_code=1
unset EQUINOX_EXPECTED_CHECKPOINT_GENERATION
unset EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256
if [[ "$pending_result_completed" == "true" ]] ||
  safe_regular_exists "result.json"; then
  workload_exit_code=0
elif safe_regular_exists "exit_code"; then
  prior_exit_code="$(safe_read_regular "exit_code" 32)"
  if [[ "$prior_exit_code" =~ ^[0-9]+$ ]]; then
    workload_exit_code="$prior_exit_code"
  fi
fi
while ! safe_regular_exists "result.json" &&
  [[ "$retry_allowed" == "true" ]] &&
  ((recorded_attempts < maximum_workload_attempts)); do
  if [[ "$requires_live_stage_handoff" == "true" ]] &&
    ! validate_live_stage_handoff; then
    workload_exit_code=66
    retry_allowed=false
    terminal_error="LIVE_STAGE_HANDOFF_INVALID"
    terminal_message="The authenticated live-stage handoff evidence changed before scientific work."
    write_progress \
      "$progress_schema_version" \
      "failed" \
      "$terminal_message" \
      "$((recorded_attempts + 1))" \
      "$terminal_error" \
      "true"
    break
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
  workload_attempt="$((recorded_attempts + 1))"
  unset EQUINOX_EXPECTED_CHECKPOINT_GENERATION
  unset EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256
  if [[ "$workload_file" == "repository_repair_large_model_pilot.py" &&
    "$workload_attempt" == "2" ]]; then
    checkpoint_replay_pair=""
    if ! update_resume_manifest validate ||
      ! checkpoint_replay_pair="$(checkpoint_replay_handoff load)"; then
      workload_exit_code=74
      retry_allowed=false
      terminal_error="CHECKPOINT_REPLAY_HANDOFF_INVALID"
      terminal_message="The retry checkpoint did not match its authenticated private handoff."
      write_progress \
        "$progress_schema_version" \
        "failed" \
        "$terminal_message" \
        "$workload_attempt" \
        "$terminal_error" \
        "true"
      break
    fi
    if ! read -r checkpoint_generation checkpoint_manifest_digest <<<"$checkpoint_replay_pair" ||
      [[ ! "$checkpoint_generation" =~ ^[1-9][0-9]*$ ]] ||
      [[ ! "$checkpoint_manifest_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
      workload_exit_code=74
      retry_allowed=false
      terminal_error="CHECKPOINT_REPLAY_HANDOFF_INVALID"
      terminal_message="The authenticated retry checkpoint handoff was malformed."
      write_progress \
        "$progress_schema_version" \
        "failed" \
        "$terminal_message" \
        "$workload_attempt" \
        "$terminal_error" \
        "true"
      break
    fi
    export EQUINOX_EXPECTED_CHECKPOINT_GENERATION="$checkpoint_generation"
    export EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256="$checkpoint_manifest_digest"
  fi
  if ! "${EQUINOX_ATTEMPT_COUNTER_PYTHON:-$filesystem_python}" - \
    19 \
    "$workload_attempt" <<'PY'
import os
import secrets
import stat
import sys

directory_descriptor = int(sys.argv[1])
name = "workload-attempt-count"
payload = (sys.argv[2] + "\n").encode()
if len(payload) >= 16:
    raise SystemExit("attempt fence payload is invalid")
try:
    existing = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
except FileNotFoundError:
    existing = None
if existing is not None and not stat.S_ISREG(existing.st_mode):
    raise SystemExit("attempt fence target has an unsafe type")
pending_name = ".attempt-" + secrets.token_hex(16)
flags = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
descriptor = os.open(pending_name, flags, 0o600, dir_fd=directory_descriptor)
try:
    os.write(descriptor, payload)
    os.fsync(descriptor)
    os.replace(
        pending_name,
        name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    os.fsync(directory_descriptor)
finally:
    os.close(descriptor)
    try:
        os.unlink(pending_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
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
  if ! update_resume_manifest update "workload-attempt-count"; then
    printf '%s\n' "The workload attempt fence could not be authenticated." >&2
    exit 74
  fi
  recorded_attempts="$workload_attempt"
  attempt_error_name="error.attempt-$workload_attempt.log"
  EQUINOX_PROGRESS_PATH="$progress_path" \
    EQUINOX_ADAPTER_PATH="$adapter_path" \
    EQUINOX_REMOTE_WORKDIR_FD=19 \
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
    run_scientific_attempt \
    "$attempt_error_name" \
    "${scientific_workload_command[@]}"
  workload_exit_code="$?"
  pending_result_commit_identity=""
  if safe_regular_exists "result.pending.json"; then
    if pending_result_commit_identity="$(
      completed_result_identity "result.pending.json" 2>/dev/null
    )"; then
      :
    elif ! discard_uncommitted_pending_result; then
      printf '%s\n' "The uncommitted workload result could not be discarded safely." >&2
      exit 74
    fi
  fi
  if ! EQUINOX_EXPECTED_PENDING_RESULT_COMMIT="$pending_result_commit_identity" \
    update_resume_manifest update \
    "progress.json" \
    "result.pending.json" \
    "$attempt_error_name" \
    "adapter" \
    "attempts" \
    "eligibility-unused-study-evidence.json" \
    "revision31-study-runtime-evidence.json" \
    "study-runtime-evidence.json"; then
    printf '%s\n' "Scientific output authentication failed." >&2
    exit 74
  fi
  detected_non_retryable_failure=""
  if detected_non_retryable_failure="$(
    non_retryable_failure_code "$attempt_error_name"
  )"; then
    persist_non_retryable_failure "$detected_non_retryable_failure"
  fi
  checkpoint_replay_capture_failed=false
  if [[ "$workload_file" == "repository_repair_large_model_pilot.py" &&
    "$workload_attempt" == "1" &&
    -z "$detected_non_retryable_failure" ]] &&
    ! completed_pending_result_available &&
    adapter_checkpoint_available; then
    if ! update_resume_manifest validate ||
      ! checkpoint_replay_handoff capture >/dev/null; then
      checkpoint_replay_capture_failed=true
      retry_allowed=false
      terminal_error="CHECKPOINT_REPLAY_HANDOFF_INVALID"
      terminal_message="The first attempt did not produce an authenticated checkpoint replay handoff."
    fi
  fi
  merge_attempt_error_log_or_fallback \
    "$attempt_error_name" \
    "$workload_attempt" \
    "exit $workload_exit_code"
  if [[ "$checkpoint_replay_capture_failed" == "true" ]]; then
    workload_exit_code=74
    break
  fi
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
    ! adapter_checkpoint_available; then
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

if [[ "$workload_exit_code" != "0" ]] &&
  ! safe_regular_exists "result.json" &&
  ! progress_is_failed; then
  if [[ -n "$terminal_error" ]]; then
    :
  elif ! adapter_checkpoint_available; then
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
  {
    [[ "$pending_result_completed" == "true" ]] ||
      safe_regular_exists "result.json"
  }; then
  result_commit_identity=""
  if safe_regular_exists "result.pending.json"; then
    if [[ -z "$pending_result_commit_identity" ]] &&
      ! pending_result_commit_identity="$(
        completed_result_identity "result.pending.json"
      )"; then
      workload_exit_code=74
      terminal_error="RESULT_VALIDATION_FAILED"
      terminal_message="The completed workload result could not be validated safely."
    fi
  elif safe_regular_exists "result.json"; then
    if ! result_commit_identity="$(
      completed_result_identity "result.json"
    )"; then
      workload_exit_code=74
      terminal_error="RESULT_VALIDATION_FAILED"
      terminal_message="The published workload result could not be validated safely."
    fi
  else
    workload_exit_code=74
    terminal_error="RESULT_VALIDATION_FAILED"
    terminal_message="The completed workload result disappeared before publication."
  fi
  adapter_commit_identity=""
  if [[ "$adapter_enabled" == "true" && "$workload_exit_code" == "0" ]]; then
    if adapter_commit_identity="$(safe_archive_adapter)"; then
      if [[ "$adapter_commit_identity" == "none" ]]; then
        adapter_commit_identity=""
      fi
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
  if [[ "$workload_exit_code" == "0" && "$adapter_enabled" == "true" ]] &&
    ! safe_remove_adapter_checkpoints; then
    workload_exit_code=74
    terminal_error="ARTIFACT_CLEANUP_FAILED"
    terminal_message="The persisted checkpoint tree could not be removed safely."
    write_progress \
      "$progress_schema_version" \
      "failed" \
      "$terminal_message" \
      "$recorded_attempts" \
      "$terminal_error" \
      "true"
  fi
  if [[ "$workload_exit_code" == "0" ]] &&
    ! safe_regular_exists "result.json"; then
    if ! result_commit_identity="$(
      promote_completed_result "$pending_result_commit_identity"
    )"; then
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
  if [[ "$workload_exit_code" == "0" ]] &&
    ! EQUINOX_EXPECTED_RESULT_COMMIT="$result_commit_identity" \
      EQUINOX_EXPECTED_ADAPTER_COMMIT="$adapter_commit_identity" \
      update_resume_manifest update \
      "result.pending.json" \
      "result.json" \
      "adapter.tgz" \
      "adapter"; then
    workload_exit_code=74
    terminal_error="RESULT_AUTHENTICATION_FAILED"
    terminal_message="The completed result and adapter could not be authenticated together."
    printf '%s\n' "$terminal_message" >&2
  fi
  if [[ "$workload_exit_code" == "0" ]]; then
    if [[ "$pending_result_recovered" == "true" ]]; then
      completion_message="Recovered result and artifacts published."
    else
      completion_message="Result and artifacts published."
    fi
    write_recovered_result_progress "complete" "$completion_message"
  fi
fi
if [[ "$workload_exit_code" != "0" ]] &&
  safe_regular_exists "progress.json"; then
  structure_remote_failure "$workload_exit_code" || true
fi
safe_write_line "exit_code" "$workload_exit_code"

wait "$http_server_pid"
