"""Authenticated, allowlisted result transport for the bounded RunPod worker."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import BinaryIO
from urllib.parse import urlsplit

TOKEN = os.environ["EQUINOX_RESULT_TOKEN"]
MANIFEST_NAME = "runner-resume-state.json"
MANIFEST_REVISION = "token-bound-runner-resume@1"
ROUTES = {
    "/adapter.tgz": ("adapter.tgz", "application/gzip", 4 * 1024 * 1024 * 1024),
    "/error.log": ("error.log", "text/plain; charset=utf-8", 64 * 1024 * 1024),
    "/exit_code": ("exit_code", "text/plain; charset=utf-8", 128),
    "/integrity.json": (MANIFEST_NAME, "application/json", 64 * 1024 * 1024),
    "/progress.json": ("progress.json", "application/json", 24 * 1024 * 1024),
    "/result.json": ("result.json", "application/json", 64 * 1024 * 1024),
}
SAFE_FD_PATH = re.compile(r"(?:/proc/self/fd|/dev/fd)/([0-9]+)")
ARTIFACT_FIELDS = {
    "domain",
    "launch_identity_digest",
    "path",
    "sha256",
    "size_bytes",
    "hmac",
}
PROGRESS_REQUIRED_FIELDS = {
    "schema_version",
    "phase",
    "message",
    "branch_width",
    "complexity_strategy",
    "attempt",
}
PROGRESS_ALLOWED_FIELDS = PROGRESS_REQUIRED_FIELDS | {
    "action_protocol_validity_rate",
    "active_complexity",
    "attempted_policy_update_count",
    "attempted_priority_training_family_ids",
    "authorization_digest",
    "baseline_failure_family_ids",
    "baseline_validation",
    "best_validation",
    "bootstrap_source_digest",
    "branch_evidence_complete",
    "branch_evidence_group_count",
    "branch_evidence_limit",
    "branch_evidence_payload_bytes",
    "branch_evidence_payload_limit_bytes",
    "branch_groups_completed",
    "branch_groups_total",
    "branch_snapshots",
    "bundle_activation_digest",
    "bundle_handoff_revision",
    "bundle_stage_receipt_digest",
    "checkpoint_generation",
    "checkpoint_manifest_digest",
    "checkpoint_name",
    "checkpoint_rate",
    "checkpoint_rate_95ci",
    "checkpoint_stage",
    "claim_strength",
    "code_materialization_evidence_digest",
    "code_materialization_revision",
    "code_private_tree_digest",
    "completion_budget",
    "consecutive_malformed_windows",
    "consecutive_uninformative_groups",
    "correctness_metrics_disclosed",
    "current_level",
    "curriculum_decision",
    "curriculum_failure_family_ids",
    "curriculum_history",
    "curriculum_policy",
    "curriculum_target_expansion_count",
    "dependency_lock_digest",
    "dependency_private_tree_digest",
    "dependency_quarantine_evidence_digest",
    "dependency_quarantine_revision",
    "deterministic_runtime",
    "discarded_sampled_completion_tokens",
    "effective_policy_update_count",
    "elapsed_seconds",
    "eligibility_screen_revision",
    "error",
    "evaluation_complete",
    "evaluation_completed",
    "evaluation_examples",
    "evaluation_split",
    "evaluation_total",
    "exact_rate",
    "exact_rate_95ci",
    "expected_evaluation_examples",
    "final_evaluation_reserve_seconds",
    "frontier_probe_task_groups",
    "gpu_id",
    "gpu_memory_gb",
    "gpu_memory_gib",
    "gpu_name",
    "gpu_total_memory_bytes",
    "head_commit",
    "hypothesis_passed",
    "informative_group_rate",
    "larger_model_profile_id",
    "larger_model_screen_revision",
    "latest_branch_snapshot",
    "learning_rate",
    "live_stage_activation_revision",
    "mastery_streak",
    "mastery_windows",
    "maximum_level",
    "maximum_sampled_complexity_level",
    "maximum_updates",
    "minimum_protocol_validity_rate",
    "model_id",
    "model_revision",
    "multi_step",
    "network_volume_data_center_id",
    "network_volume_id",
    "network_volume_size_gb",
    "operator_error",
    "optimization_seed",
    "optimizer_update_count",
    "paired_test_change",
    "pending_informative_group_count",
    "pending_optimizer_input_group_count",
    "pending_optimizer_input_group_ids",
    "pending_policy_example_count",
    "pending_training_example_count",
    "policy_mutation_enabled",
    "policy_update_count",
    "policy_update_lineage",
    "priority_training_family_ids",
    "profile_id",
    "promotion_count",
    "provider_remaining_seconds",
    "recent_action_protocol_groups",
    "recent_action_protocol_window_complete",
    "recent_malformed_action_rate",
    "reference_kl_coefficient",
    "regression_streak",
    "remote_error",
    "restored_continuations",
    "retained_policy_update_count",
    "retention_rollback_count",
    "retention_transaction_disposition",
    "retention_transaction_revision",
    "rollback_applied",
    "source_contract_digest",
    "stop_reason",
    "study_condition",
    "study_id",
    "targeted_training_family_ids",
    "teacher_data_used",
    "test_examples",
    "test_examples_accessed",
    "test_split_accessed",
    "torch_retention_evidence_digest",
    "total_sampled_actions",
    "total_sampled_completion_tokens",
    "training_remaining_seconds",
    "training_started",
    "update",
    "validation_examples",
    "validation_history",
    "volume_readiness_receipt_digest",
    "workload_bundle_digest",
    "workload_bundle_path",
    "workload_bundle_size_bytes",
}
PROGRESS_PHASE_RANKS = {
    "container_starting": 0,
    "resuming": 0,
    "dependency_setup": 1,
    "model_loading": 2,
    "baseline_evaluation": 3,
    "protocol_evaluation": 4,
    "checkpoint_baseline_evaluation": 5,
    "checkpoint_validation_evaluation": 5,
    "eligibility_branch_collection": 5,
    "retention_guard_baseline_evaluation": 5,
    "training": 5,
    "validation_evaluation": 5,
    "checkpointing": 5,
    "finalizing": 6,
    "baseline_test_evaluation": 6,
    "final_test_evaluation": 6,
    "test_split_isolation_violation": 6,
    "complete": 7,
    "failed": 7,
}
PROGRESS_GLOBAL_COUNTERS = {
    "branch_groups_completed",
    "checkpoint_generation",
    "optimizer_update_count",
    "policy_update_count",
    "retained_policy_update_count",
    "total_sampled_actions",
    "total_sampled_completion_tokens",
}
PROGRESS_LOCAL_COUNTERS = {"evaluation_completed"}
MAXIMUM_PROGRESS_DEPTH = 12
MAXIMUM_PROGRESS_NODES = 1_000_000
MAXIMUM_PROGRESS_LIST_ITEMS = 4096
MAXIMUM_PROGRESS_OBJECT_FIELDS = 512
MAXIMUM_PROGRESS_STRING_BYTES = 256 * 1024
MAXIMUM_PROGRESS_ELAPSED_SECONDS = 31 * 24 * 60 * 60


@dataclass(frozen=True)
class ProgressState:
    digest: str
    attempt: int
    phase: str
    phase_rank: int
    update_floor: int | None
    elapsed_floor: float | None
    global_counter_floors: tuple[tuple[str, int], ...]
    local_counters: tuple[tuple[str, int], ...]
    terminal_base_digest: str
    terminal_exit_enriched: bool


PROGRESS_LOCK = threading.Lock()
PROGRESS_STATE: ProgressState | None = None


def _open_root_descriptor() -> int:
    inherited = os.environ.get("EQUINOX_REMOTE_WORKDIR_FD", "")
    if inherited:
        if not inherited.isascii() or not inherited.isdigit():
            raise RuntimeError("EQUINOX_REMOTE_WORKDIR_FD is invalid")
        descriptor = os.dup(int(inherited))
    else:
        path = os.environ.get("EQUINOX_REMOTE_WORKDIR", "/tmp")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        match = SAFE_FD_PATH.fullmatch(path)
        descriptor = os.dup(int(match.group(1))) if match else os.open(path, flags)
        if match is None:
            path_metadata = os.lstat(path)
            descriptor_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(path_metadata.st_mode)
                or stat.S_ISLNK(path_metadata.st_mode)
                or path_metadata.st_dev != descriptor_metadata.st_dev
                or path_metadata.st_ino != descriptor_metadata.st_ino
            ):
                os.close(descriptor)
                raise RuntimeError("result root changed while being opened")
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("result root is not a directory")
    return descriptor


ROOT_DESCRIPTOR = _open_root_descriptor()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_progress_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_bounded_json(
    value: object,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> None:
    if budget is None:
        budget = [MAXIMUM_PROGRESS_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAXIMUM_PROGRESS_DEPTH:
        raise FileNotFoundError("progress.json")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAXIMUM_PROGRESS_STRING_BYTES:
            raise FileNotFoundError("progress.json")
        return
    if _is_integer(value):
        if abs(value) > (2**63 - 1):
            raise FileNotFoundError("progress.json")
        return
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 1e15:
            raise FileNotFoundError("progress.json")
        return
    if isinstance(value, list):
        if len(value) > MAXIMUM_PROGRESS_LIST_ITEMS:
            raise FileNotFoundError("progress.json")
        for item in value:
            _validate_bounded_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        if len(value) > MAXIMUM_PROGRESS_OBJECT_FIELDS:
            raise FileNotFoundError("progress.json")
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key.encode("utf-8")) > 128
                or any(ord(character) < 0x20 for character in key)
            ):
                raise FileNotFoundError("progress.json")
            _validate_bounded_json(item, depth=depth + 1, budget=budget)
        return
    raise FileNotFoundError("progress.json")


def _optional_nonnegative_integer(
    progress: dict[str, object],
    field: str,
    *,
    maximum: int = 10**12,
) -> int | None:
    value = progress.get(field)
    if value is None:
        return None
    if not _is_integer(value) or value < 0 or value > maximum:
        raise FileNotFoundError("progress.json")
    return value


def _parse_progress(payload: bytes) -> dict[str, object]:
    try:
        progress = json.loads(payload)
        canonical = _canonical_progress_json(progress) + b"\n"
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise FileNotFoundError("progress.json") from error
    if (
        not isinstance(progress, dict)
        or payload != canonical
        or not PROGRESS_REQUIRED_FIELDS.issubset(progress)
        or not set(progress).issubset(PROGRESS_ALLOWED_FIELDS)
    ):
        raise FileNotFoundError("progress.json")
    _validate_bounded_json(progress)

    schema_version = progress["schema_version"]
    phase = progress["phase"]
    message = progress["message"]
    branch_width = progress["branch_width"]
    attempt = progress["attempt"]
    if (
        not _is_integer(schema_version)
        or schema_version not in {1, 2}
        or not isinstance(phase, str)
        or phase not in PROGRESS_PHASE_RANKS
        or not isinstance(message, str)
        or not message
        or len(message.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in message)
        or not _is_integer(branch_width)
        or branch_width < 1
        or branch_width > 64
        or progress["complexity_strategy"] != "adaptive"
        or not _is_integer(attempt)
        or attempt not in {1, 2}
    ):
        raise FileNotFoundError("progress.json")

    _optional_nonnegative_integer(progress, "update", maximum=10_000_000)
    for field in PROGRESS_GLOBAL_COUNTERS | PROGRESS_LOCAL_COUNTERS:
        _optional_nonnegative_integer(progress, field)
    elapsed = progress.get("elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(float(elapsed))
        or elapsed < 0
        or elapsed > MAXIMUM_PROGRESS_ELAPSED_SECONDS
    ):
        raise FileNotFoundError("progress.json")
    error = progress.get("error")
    if error is not None and (
        not isinstance(error, str)
        or not error
        or len(error.encode("utf-8")) > MAXIMUM_PROGRESS_STRING_BYTES
    ):
        raise FileNotFoundError("progress.json")
    if phase != "failed" and error is not None:
        raise FileNotFoundError("progress.json")
    remote_error = progress.get("remote_error")
    if remote_error is not None:
        if (
            phase != "failed"
            or not isinstance(remote_error, dict)
            or set(remote_error) not in (
                {"code", "message"},
                {"code", "message", "exit_code"},
            )
            or not isinstance(remote_error.get("code"), str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", str(remote_error["code"])) is None
            or not isinstance(remote_error.get("message"), str)
            or not remote_error["message"]
            or len(str(remote_error["message"]).encode("utf-8"))
            > MAXIMUM_PROGRESS_STRING_BYTES
            or (
                "exit_code" in remote_error
                and (
                    not _is_integer(remote_error["exit_code"])
                    or not 0 <= int(remote_error["exit_code"]) <= 255
                )
            )
        ):
            raise FileNotFoundError("progress.json")
    return progress


def _progress_state(
    progress: dict[str, object],
    payload: bytes,
    previous: ProgressState | None,
) -> ProgressState:
    attempt = int(progress["attempt"])
    phase = str(progress["phase"])
    phase_rank = PROGRESS_PHASE_RANKS[phase]
    digest = hashlib.sha256(payload).hexdigest()
    update = _optional_nonnegative_integer(progress, "update", maximum=10_000_000)
    raw_elapsed = progress.get("elapsed_seconds")
    elapsed = None if raw_elapsed is None else float(raw_elapsed)
    terminal_base = dict(progress)
    terminal_base.pop("remote_error", None)
    terminal_base_digest = hashlib.sha256(
        _canonical_progress_json(terminal_base)
    ).hexdigest()
    remote_error = progress.get("remote_error")
    terminal_exit_enriched = (
        isinstance(remote_error, dict) and "exit_code" in remote_error
    )
    observed_global_counters = {
        field: value
        for field in PROGRESS_GLOBAL_COUNTERS
        if (value := _optional_nonnegative_integer(progress, field)) is not None
    }
    observed_local_counters = {
        field: value
        for field in PROGRESS_LOCAL_COUNTERS
        if (value := _optional_nonnegative_integer(progress, field)) is not None
    }
    if previous is None:
        return ProgressState(
            digest=digest,
            attempt=attempt,
            phase=phase,
            phase_rank=phase_rank,
            update_floor=update,
            elapsed_floor=elapsed,
            global_counter_floors=tuple(sorted(observed_global_counters.items())),
            local_counters=tuple(sorted(observed_local_counters.items())),
            terminal_base_digest=terminal_base_digest,
            terminal_exit_enriched=terminal_exit_enriched,
        )
    if hmac.compare_digest(digest, previous.digest):
        return previous

    if attempt < previous.attempt or attempt > previous.attempt + 1:
        raise FileNotFoundError("progress.json")
    if previous.phase == "complete":
        raise FileNotFoundError("progress.json")
    terminal_enrichment = (
        previous.phase == "failed"
        and phase == "failed"
        and attempt == previous.attempt
        and previous.terminal_base_digest == terminal_base_digest
        and not previous.terminal_exit_enriched
        and terminal_exit_enriched
    )
    if (
        previous.phase == "failed"
        and attempt == previous.attempt
        and not terminal_enrichment
    ):
        raise FileNotFoundError("progress.json")

    same_attempt = attempt == previous.attempt
    checkpoint_restore_transition = (
        same_attempt
        and previous.phase == "model_loading"
        and phase == "resuming"
    )
    if (
        same_attempt
        and phase_rank < previous.phase_rank
        and not checkpoint_restore_transition
    ):
        raise FileNotFoundError("progress.json")
    if update is not None and previous.update_floor is not None and update < previous.update_floor:
        raise FileNotFoundError("progress.json")
    reset_elapsed_scope = (
        attempt > previous.attempt
        or (
            same_attempt
            and previous.phase in {"container_starting", "resuming"}
            and phase == "dependency_setup"
        )
    )
    if (
        elapsed is not None
        and previous.elapsed_floor is not None
        and elapsed < previous.elapsed_floor
        and not reset_elapsed_scope
    ):
        raise FileNotFoundError("progress.json")

    previous_global_counters = dict(previous.global_counter_floors)
    for field, value in observed_global_counters.items():
        if field in previous_global_counters and value < previous_global_counters[field]:
            raise FileNotFoundError("progress.json")

    previous_local_counters = dict(previous.local_counters)
    same_local_scope = same_attempt and phase == previous.phase and update == previous.update_floor
    if same_local_scope:
        for field, value in observed_local_counters.items():
            if field in previous_local_counters and value < previous_local_counters[field]:
                raise FileNotFoundError("progress.json")

    advanced = (
        terminal_enrichment
        or checkpoint_restore_transition
        or attempt > previous.attempt
        or (same_attempt and phase_rank > previous.phase_rank)
        or (
            update is not None and (previous.update_floor is None or update > previous.update_floor)
        )
        or (
            elapsed is not None
            and (previous.elapsed_floor is None or elapsed > previous.elapsed_floor)
        )
        or any(
            field not in previous_global_counters or value > previous_global_counters[field]
            for field, value in observed_global_counters.items()
        )
        or (
            same_local_scope
            and any(
                field not in previous_local_counters or value > previous_local_counters[field]
                for field, value in observed_local_counters.items()
            )
        )
    )
    if not advanced:
        raise FileNotFoundError("progress.json")

    global_counter_floors = dict(previous_global_counters)
    for field, value in observed_global_counters.items():
        global_counter_floors[field] = max(global_counter_floors.get(field, value), value)
    if same_local_scope:
        local_counters = dict(previous_local_counters)
        for field, value in observed_local_counters.items():
            local_counters[field] = max(local_counters.get(field, value), value)
    else:
        local_counters = observed_local_counters
    return ProgressState(
        digest=digest,
        attempt=attempt,
        phase=phase,
        phase_rank=phase_rank,
        update_floor=(
            update
            if previous.update_floor is None
            else (previous.update_floor if update is None else max(previous.update_floor, update))
        ),
        elapsed_floor=(
            elapsed
            if previous.elapsed_floor is None or reset_elapsed_scope
            else (
                previous.elapsed_floor if elapsed is None else max(previous.elapsed_floor, elapsed)
            )
        ),
        global_counter_floors=tuple(sorted(global_counter_floors.items())),
        local_counters=tuple(sorted(local_counters.items())),
        terminal_base_digest=terminal_base_digest,
        terminal_exit_enriched=terminal_exit_enriched,
    )


def _launch_identity_digest() -> str:
    proof_id = os.environ.get("EQUINOX_PROOF_ID", "")
    if re.fullmatch(r"[A-Za-z0-9._-]+", proof_id) is None:
        raise FileNotFoundError("launch identity is unavailable")
    identity = {
        "proof_id": proof_id,
        "run_identity": os.environ.get("EQUINOX_RUN_IDENTITY", ""),
        "private_runtime_revision": os.environ.get(
            "EQUINOX_PRIVATE_RUNTIME_REVISION",
            "",
        ),
        "workload_file": os.environ.get("EQUINOX_WORKLOAD_FILE", ""),
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
    return "sha256:" + hashlib.sha256(_canonical_json(identity)).hexdigest()


def _artifact_domain(path: str) -> str:
    if path == "progress.json":
        return "equinox/progress/v1"
    if path in {"result.json", "result.pending.json"}:
        return "equinox/result/v1"
    if path == "adapter.tgz" or path.startswith("adapter/"):
        return "equinox/checkpoint/v1"
    if path == "workload-attempt-count" or re.fullmatch(
        r"error[.]attempt-[12][.]log",
        path,
    ):
        return "equinox/attempt/v1"
    return "equinox/journal/v1"


def _read_regular_at(name: str, maximum_bytes: int) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=ROOT_DESCRIPTOR)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
            raise FileNotFoundError(name)
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, metadata.st_size - len(payload)),
            )
            if not chunk:
                raise FileNotFoundError(name)
            payload += chunk
        return payload
    finally:
        os.close(descriptor)


def _authenticated_artifact(name: str) -> dict[str, object]:
    payload = _read_regular_at(MANIFEST_NAME, 64 * 1024 * 1024)
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FileNotFoundError(MANIFEST_NAME) from error
    expected_manifest_fields = {
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
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_fields
        or payload != _canonical_json(manifest) + b"\n"
        or manifest.get("schema_version") != 1
        or manifest.get("revision") != MANIFEST_REVISION
        or manifest.get("workload_file") != os.environ.get("EQUINOX_WORKLOAD_FILE", "")
        or manifest.get("launch_identity_digest") != _launch_identity_digest()
        or not isinstance(manifest.get("generation"), int)
        or isinstance(manifest.get("generation"), bool)
        or int(manifest["generation"]) < 0
        or (
            manifest.get("parent_manifest_sha256") is not None
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(manifest.get("parent_manifest_sha256")),
            )
            is None
        )
        or not isinstance(manifest.get("directories"), list)
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise FileNotFoundError(MANIFEST_NAME)
    supplied_manifest_hmac = manifest["hmac"]
    material = {key: value for key, value in manifest.items() if key != "hmac"}
    expected_manifest_hmac = (
        "hmac-sha256:"
        + hmac.new(
            TOKEN.encode(),
            b"equinox/runner-resume-manifest/v1\0" + _canonical_json(material),
            hashlib.sha256,
        ).hexdigest()
    )
    if not isinstance(supplied_manifest_hmac, str) or not hmac.compare_digest(
        supplied_manifest_hmac,
        expected_manifest_hmac,
    ):
        raise FileNotFoundError(MANIFEST_NAME)
    if name == MANIFEST_NAME:
        return {
            "path": MANIFEST_NAME,
            "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    matches = [
        entry
        for entry in manifest["artifacts"]
        if isinstance(entry, dict) and entry.get("path") == name
    ]
    if len(matches) != 1 or set(matches[0]) != ARTIFACT_FIELDS:
        raise FileNotFoundError(name)
    entry = matches[0]
    entry_material = {key: value for key, value in entry.items() if key != "hmac"}
    expected_domain = _artifact_domain(name)
    expected_artifact_hmac = (
        "hmac-sha256:"
        + hmac.new(
            TOKEN.encode(),
            (expected_domain + "\0").encode() + _canonical_json(entry_material),
            hashlib.sha256,
        ).hexdigest()
    )
    if (
        entry.get("domain") != expected_domain
        or entry.get("launch_identity_digest") != _launch_identity_digest()
        or not isinstance(entry.get("hmac"), str)
        or not hmac.compare_digest(str(entry["hmac"]), expected_artifact_hmac)
    ):
        raise FileNotFoundError(name)
    return entry


@contextmanager
def _open_progress_route(maximum_bytes: int) -> Iterator[tuple[BinaryIO, int]]:
    global PROGRESS_STATE

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    with PROGRESS_LOCK:
        descriptor = os.open("progress.json", flags, dir_fd=ROOT_DESCRIPTOR)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > maximum_bytes
            ):
                raise FileNotFoundError("progress.json")
            payload = b""
            while len(payload) < before.st_size:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, before.st_size - len(payload)),
                )
                if not chunk:
                    raise FileNotFoundError("progress.json")
                payload += chunk
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
                raise FileNotFoundError("progress.json")
        finally:
            os.close(descriptor)
        progress = _parse_progress(payload)
        next_state = _progress_state(progress, payload, PROGRESS_STATE)
        PROGRESS_STATE = next_state
        with io.BytesIO(payload) as handle:
            yield handle, len(payload)


@contextmanager
def _open_route(name: str, maximum_bytes: int) -> Iterator[tuple[BinaryIO, int]]:
    if name == "progress.json":
        with _open_progress_route(maximum_bytes) as route:
            yield route
        return
    authenticated = _authenticated_artifact(name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=ROOT_DESCRIPTOR)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 0
            or metadata.st_size > maximum_bytes
            or (authenticated is not None and metadata.st_size != authenticated.get("size_bytes"))
        ):
            raise FileNotFoundError(name)
        digest = hashlib.sha256()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise FileNotFoundError(name)
            digest.update(chunk)
            remaining -= len(chunk)
        observed_digest = "sha256:" + digest.hexdigest()
        if authenticated is not None and not hmac.compare_digest(
            observed_digest, str(authenticated.get("sha256", ""))
        ):
            raise FileNotFoundError(name)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle, metadata.st_size
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(metadata, key) != getattr(after, key) for key in stable_fields):
            raise ConnectionError("result artifact changed while being served")
    finally:
        os.close(descriptor)


class ResultHandler(BaseHTTPRequestHandler):
    server_version = "EquinoxResultTransport/2"

    def _authorized(self) -> bool:
        provided = self.headers.get("Authorization", "")
        return hmac.compare_digest(provided, f"Bearer {TOKEN}")

    def _serve(self, *, include_body: bool) -> None:
        if not self._authorized():
            self.send_error(HTTPStatus.UNAUTHORIZED)
            return
        route = ROUTES.get(urlsplit(self.path).path)
        if route is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        relative_path, media_type, maximum_bytes = route
        try:
            route_context = _open_route(relative_path, maximum_bytes)
            handle, size = route_context.__enter__()
        except (FileNotFoundError, NotADirectoryError, OSError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if include_body:
                remaining = size
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        return
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        finally:
            route_context.__exit__(None, None, None)

    def do_GET(self) -> None:
        self._serve(include_body=True)

    def do_HEAD(self) -> None:
        self._serve(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return


class ResultHTTPServer(ThreadingHTTPServer):
    """Bind without the HTTPServer reverse-DNS lookup that can stall readiness."""

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


if __name__ == "__main__":
    host = os.environ.get("EQUINOX_RESULT_HOST", "0.0.0.0")
    port = int(os.environ.get("EQUINOX_RESULT_PORT", "8000"))
    ResultHTTPServer((host, port), ResultHandler).serve_forever()
