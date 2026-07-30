import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

LARGE_MODEL_WORKLOADS = {
    "repository_repair_large_model_eligibility.py",
    "repository_repair_large_model_pilot.py",
}
LIVE_STAGE_ACTIVATION_REVISION = "authenticated-proxy-stage-activation@2"
BUNDLE_HANDOFF_REVISION = "runpod-volume-bundle-handoff@1"
DEPENDENCY_QUARANTINE_REVISION = "hash-locked-private-dependencies@1"
CODE_MATERIALIZATION_REVISION = "private-code-materialization@1"
DEPENDENCY_LOCK_DIGEST = (
    "sha256:bb143bf631b6509c07881a606dc96cd244099debdca09f8f700abc90dad2e34f"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _tree_entries(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _materialize_test_tree(staging: Path, kind: str) -> tuple[Path, list[dict[str, object]], str]:
    entries = _tree_entries(staging)
    digest = "sha256:" + hashlib.sha256(_canonical_json(entries)).hexdigest()
    destination = Path("/tmp/equinox-quarantine") / kind / digest.removeprefix("sha256:")
    if destination.exists():
        destination.chmod(0o755)
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, destination)
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return destination, entries, digest


def materialization_environment(
    tmp_path: Path,
    code_staging: Path,
    dependency_staging: Path,
) -> tuple[Path, Path, dict[str, str]]:
    code_root, code_entries, code_tree_digest = _materialize_test_tree(code_staging, "code")
    dependency_root, dependency_entries, dependency_tree_digest = _materialize_test_tree(
        dependency_staging,
        "dependencies",
    )
    base = live_stage_environment()
    code_evidence = {
        "schema_version": 1,
        "revision": CODE_MATERIALIZATION_REVISION,
        "profile_id": base["EQUINOX_LARGER_MODEL_PROFILE_ID"],
        "bundle_digest": base["EQUINOX_BUNDLE_SHA256"],
        "bundle_size_bytes": int(base["EQUINOX_BUNDLE_SIZE_BYTES"]),
        "source_contract_digest": base[
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256"
        ],
        "source_root": base["EQUINOX_BUNDLE_VOLUME_PATH"],
        "private_root": str(code_root),
        "source_tree_digest": code_tree_digest,
        "private_tree_digest": code_tree_digest,
        "files": code_entries,
        "installed_file_count": len(code_entries),
        "installed_bytes": sum(int(entry["size_bytes"]) for entry in code_entries),
        "ready": True,
    }
    code_evidence["evidence_digest"] = (
        "sha256:" + hashlib.sha256(_canonical_json(code_evidence)).hexdigest()
    )
    distributions = [
        {
            "name": "fake-dependency",
            "version": "1.0",
            "record_path": "fake_dependency-1.0.dist-info/RECORD",
            "record_digest": "sha256:" + "8" * 64,
            "file_count": len(dependency_entries),
            "files_digest": dependency_tree_digest,
        }
    ]
    dependency_evidence = {
        "schema_version": 1,
        "revision": DEPENDENCY_QUARANTINE_REVISION,
        "profile_id": base["EQUINOX_LARGER_MODEL_PROFILE_ID"],
        "source_root": "/workspace/equinox-state/python",
        "private_root": str(dependency_root),
        "source_tree_digest": dependency_tree_digest,
        "private_tree_digest": dependency_tree_digest,
        "record_closure_digest": (
            "sha256:" + hashlib.sha256(_canonical_json(distributions)).hexdigest()
        ),
        "distributions": distributions,
        "installed_file_count": len(dependency_entries),
        "installed_bytes": sum(
            int(entry["size_bytes"]) for entry in dependency_entries
        ),
        "ready": True,
    }
    dependency_evidence["evidence_digest"] = (
        "sha256:" + hashlib.sha256(_canonical_json(dependency_evidence)).hexdigest()
    )
    code_evidence_path = tmp_path / "code-materialization-evidence.json"
    dependency_evidence_path = tmp_path / "dependency-quarantine-evidence.json"
    code_evidence_path.write_bytes(_canonical_json(code_evidence) + b"\n")
    dependency_evidence_path.write_bytes(_canonical_json(dependency_evidence) + b"\n")
    return (
        code_root,
        dependency_root,
        {
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH": str(code_evidence_path),
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": str(
                code_evidence["evidence_digest"]
            ),
            "EQUINOX_CODE_PRIVATE_TREE_SHA256": code_tree_digest,
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH": str(
                dependency_evidence_path
            ),
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": str(
                dependency_evidence["evidence_digest"]
            ),
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": dependency_tree_digest,
        },
    )


def live_stage_environment(
    **activation_overrides: str | int,
) -> dict[str, str]:
    activation_body: dict[str, str | int] = {
        "revision": LIVE_STAGE_ACTIVATION_REVISION,
        "profile_id": "qwen2.5-coder-7b-runpod-h100@9",
        "head_commit": "1" * 40,
        "source_contract_digest": "sha256:" + "2" * 64,
        "bootstrap_source_digest": "sha256:" + "3" * 64,
        "workload_bundle_digest": "sha256:" + "4" * 64,
        "workload_bundle_size_bytes": 1048576,
        "workload_bundle_path": (
            "/workspace/equinox-state/workload-bundles/"
            "qwen2.5-coder-7b-runpod-h100@9/"
            f"{'4' * 64}.tar.xz"
        ),
        "bundle_stage_receipt_digest": "sha256:" + "5" * 64,
        "volume_readiness_receipt_digest": "sha256:" + "6" * 64,
        "torch_retention_evidence_digest": "sha256:" + "7" * 64,
        "dependency_lock_digest": DEPENDENCY_LOCK_DIGEST,
        "dependency_quarantine_revision": DEPENDENCY_QUARANTINE_REVISION,
        "dependency_quarantine_evidence_digest": "sha256:" + "8" * 64,
        "dependency_private_tree_digest": "sha256:" + "9" * 64,
        "code_materialization_revision": CODE_MATERIALIZATION_REVISION,
        "code_materialization_evidence_digest": "sha256:" + "a" * 64,
        "code_private_tree_digest": "sha256:" + "b" * 64,
        "network_volume_id": "euh248b2p5",
        "network_volume_data_center_id": "EU-FR-1",
        "network_volume_size_gb": 50,
    }
    activation_body.update(activation_overrides)
    activation_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                activation_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    return {
        "EQUINOX_LARGER_MODEL_PROFILE_ID": str(activation_body["profile_id"]),
        "EQUINOX_SOURCE_HEAD_COMMIT": str(activation_body["head_commit"]),
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256": str(
            activation_body["source_contract_digest"]
        ),
        "EQUINOX_BOOTSTRAP_SOURCE_SHA256": str(activation_body["bootstrap_source_digest"]),
        "EQUINOX_BUNDLE_SHA256": str(activation_body["workload_bundle_digest"]),
        "EQUINOX_BUNDLE_SIZE_BYTES": str(activation_body["workload_bundle_size_bytes"]),
        "EQUINOX_BUNDLE_VOLUME_PATH": str(activation_body["workload_bundle_path"]),
        "EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256": str(activation_body["bundle_stage_receipt_digest"]),
        "EQUINOX_VOLUME_READINESS_RECEIPT_SHA256": str(
            activation_body["volume_readiness_receipt_digest"]
        ),
        "EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256": str(
            activation_body["torch_retention_evidence_digest"]
        ),
        "EQUINOX_DEPENDENCY_LOCK_SHA256": str(
            activation_body["dependency_lock_digest"]
        ),
        "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256": str(
            activation_body["dependency_quarantine_evidence_digest"]
        ),
        "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256": str(
            activation_body["dependency_private_tree_digest"]
        ),
        "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256": str(
            activation_body["code_materialization_evidence_digest"]
        ),
        "EQUINOX_CODE_PRIVATE_TREE_SHA256": str(
            activation_body["code_private_tree_digest"]
        ),
        "EQUINOX_RUNPOD_NETWORK_VOLUME_ID": str(activation_body["network_volume_id"]),
        "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID": str(
            activation_body["network_volume_data_center_id"]
        ),
        "EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB": str(activation_body["network_volume_size_gb"]),
        "EQUINOX_BUNDLE_HANDOFF_REVISION": BUNDLE_HANDOFF_REVISION,
        "EQUINOX_BUNDLE_ACTIVATION_DIGEST": activation_digest,
    }


def live_stage_progress_evidence(tmp_path: Path | None = None) -> dict[str, str | int]:
    activation_overrides: dict[str, str] = {}
    if tmp_path is not None:
        evidence_root = (
            tmp_path.parent / f".{tmp_path.name}-runner-assets/persistent"
        )
        dependency_evidence = json.loads(
            (evidence_root / "dependency-quarantine-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        code_evidence = json.loads(
            (evidence_root / "code-materialization-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        activation_overrides = {
            "dependency_quarantine_evidence_digest": dependency_evidence[
                "evidence_digest"
            ],
            "dependency_private_tree_digest": dependency_evidence[
                "private_tree_digest"
            ],
            "code_materialization_evidence_digest": code_evidence["evidence_digest"],
            "code_private_tree_digest": code_evidence["private_tree_digest"],
        }
    environment = live_stage_environment(**activation_overrides)
    return {
        "profile_id": environment["EQUINOX_LARGER_MODEL_PROFILE_ID"],
        "head_commit": environment["EQUINOX_SOURCE_HEAD_COMMIT"],
        "source_contract_digest": environment["EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256"],
        "bootstrap_source_digest": environment["EQUINOX_BOOTSTRAP_SOURCE_SHA256"],
        "workload_bundle_digest": environment["EQUINOX_BUNDLE_SHA256"],
        "workload_bundle_size_bytes": int(environment["EQUINOX_BUNDLE_SIZE_BYTES"]),
        "workload_bundle_path": environment["EQUINOX_BUNDLE_VOLUME_PATH"],
        "bundle_stage_receipt_digest": environment["EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256"],
        "volume_readiness_receipt_digest": environment["EQUINOX_VOLUME_READINESS_RECEIPT_SHA256"],
        "torch_retention_evidence_digest": environment["EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256"],
        "dependency_lock_digest": environment["EQUINOX_DEPENDENCY_LOCK_SHA256"],
        "dependency_quarantine_revision": DEPENDENCY_QUARANTINE_REVISION,
        "dependency_quarantine_evidence_digest": environment[
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256"
        ],
        "dependency_private_tree_digest": environment[
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256"
        ],
        "code_materialization_revision": CODE_MATERIALIZATION_REVISION,
        "code_materialization_evidence_digest": environment[
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256"
        ],
        "code_private_tree_digest": environment["EQUINOX_CODE_PRIVATE_TREE_SHA256"],
        "network_volume_id": environment["EQUINOX_RUNPOD_NETWORK_VOLUME_ID"],
        "network_volume_data_center_id": environment[
            "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID"
        ],
        "network_volume_size_gb": int(environment["EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB"]),
        "live_stage_activation_revision": LIVE_STAGE_ACTIVATION_REVISION,
        "bundle_handoff_revision": BUNDLE_HANDOFF_REVISION,
        "bundle_activation_digest": environment["EQUINOX_BUNDLE_ACTIVATION_DIGEST"],
    }


def checkpoint_authentication_key(tmp_path: Path) -> str:
    return hashlib.sha256(
        ("checkpoint-key:" + str(tmp_path.resolve())).encode()
    ).hexdigest()


def rewrite_authenticated_checkpoint_pair(
    tmp_path: Path,
    *,
    generation: int,
    manifest_digest: str,
) -> None:
    pointer_path = tmp_path / "adapter/checkpoints/latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer.update(
        {
            "generation": generation,
            "checkpoint_manifest_digest": manifest_digest,
        }
    )
    material = {name: value for name, value in pointer.items() if name != "hmac"}
    pointer["hmac"] = (
        "hmac-sha256:"
        + hmac.new(
            bytes.fromhex(checkpoint_authentication_key(tmp_path)),
            _canonical_json(material),
            hashlib.sha256,
        ).hexdigest()
    )
    pointer_path.write_bytes(_canonical_json(pointer) + b"\n")
    progress_path = tmp_path / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(
        {
            "checkpoint_generation": generation,
            "checkpoint_manifest_digest": manifest_digest,
        }
    )
    progress_path.write_bytes(_canonical_json(progress) + b"\n")


def authenticate_runtime_state(
    tmp_path: Path,
    environment: dict[str, str],
    workload_file: str,
) -> None:
    launch_identity = {
        "proof_id": environment["EQUINOX_PROOF_ID"],
        "run_identity": environment.get("EQUINOX_RUN_IDENTITY", ""),
        "private_runtime_revision": environment.get(
            "EQUINOX_PRIVATE_RUNTIME_REVISION",
            "",
        ),
        "workload_file": workload_file,
        "model_id": environment.get("EQUINOX_RL_MODEL_ID", ""),
        "optimization_seed": environment.get("EQUINOX_RL_SEED", ""),
        "study_condition": environment.get("EQUINOX_STUDY_CONDITION", ""),
        "bundle_digest": environment.get("EQUINOX_BUNDLE_SHA256", ""),
        "bundle_activation_digest": environment.get(
            "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
            "",
        ),
        "source_contract_digest": environment.get(
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
            "",
        ),
        "code_private_tree_digest": environment.get(
            "EQUINOX_CODE_PRIVATE_TREE_SHA256",
            "",
        ),
        "dependency_private_tree_digest": environment.get(
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
            "",
        ),
        "dependency_lock_digest": environment.get(
            "EQUINOX_DEPENDENCY_LOCK_SHA256",
            "",
        ),
    }
    identity_digest = (
        "sha256:" + hashlib.sha256(_canonical_json(launch_identity)).hexdigest()
    )
    baseline_names = {
        "bundle-stage-receipt.json",
        "code-materialization-evidence.json",
        "dependency-quarantine-evidence.json",
        "live-stage-activation.json",
        "live-stage-plan.json",
        "live-stage-state.json",
        "runner-resume-state.json",
        "torch-retention-evidence.json",
        "volume-readiness-receipt.json",
    }

    def domain(path: str) -> str:
        if path == "progress.json":
            return "equinox/progress/v1"
        if path in {"result.json", "result.pending.json"}:
            return "equinox/result/v1"
        if path == "adapter.tgz" or path.startswith("adapter/"):
            return "equinox/checkpoint/v1"
        if path == "workload-attempt-count" or path == "attempts" or re.fullmatch(
            r"error[.]attempt-[12][.]log",
            path,
        ):
            return "equinox/attempt/v1"
        return "equinox/journal/v1"

    directories = [
        str(path.relative_to(tmp_path))
        for path in sorted(tmp_path.rglob("*"))
        if path.is_dir() and path.name not in baseline_names
    ]
    artifacts: list[dict[str, object]] = []
    for path in sorted(tmp_path.rglob("*")):
        if (
            not path.is_file()
            or (path.parent == tmp_path and path.name in baseline_names)
        ):
            continue
        relative = str(path.relative_to(tmp_path))
        payload = path.read_bytes()
        artifact_material: dict[str, object] = {
            "domain": domain(relative),
            "launch_identity_digest": identity_digest,
            "path": relative,
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        artifacts.append(
            {
                **artifact_material,
                "hmac": (
                    "hmac-sha256:"
                    + hmac.new(
                        environment["EQUINOX_RESULT_TOKEN"].encode(),
                        (str(artifact_material["domain"]) + "\0").encode()
                        + _canonical_json(artifact_material),
                        hashlib.sha256,
                    ).hexdigest()
                ),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "revision": "token-bound-runner-resume@1",
        "workload_file": workload_file,
        "launch_identity_digest": identity_digest,
        "generation": 0,
        "parent_manifest_sha256": None,
        "directories": directories,
        "artifacts": artifacts,
    }
    manifest["hmac"] = (
        "hmac-sha256:"
        + hmac.new(
            environment["EQUINOX_RESULT_TOKEN"].encode(),
            b"equinox/runner-resume-manifest/v1\0" + _canonical_json(manifest),
            hashlib.sha256,
        ).hexdigest()
    )
    manifest_payload = _canonical_json(manifest) + b"\n"
    (tmp_path / "runner-resume-state.json").write_bytes(manifest_payload)
    anchor_material: dict[str, object] = {
        "schema_version": 1,
        "revision": "private-runner-rollback-anchor@1",
        "proof_id": environment["EQUINOX_PROOF_ID"],
        "launch_identity_digest": identity_digest,
        "generation": 0,
        "manifest_digest": (
            "sha256:" + hashlib.sha256(manifest_payload).hexdigest()
        ),
    }
    anchor = {
        **anchor_material,
        "hmac": (
            "hmac-sha256:"
            + hmac.new(
                environment["EQUINOX_RESULT_TOKEN"].encode(),
                b"equinox/runner-private-anchor/v1\0"
                + _canonical_json(anchor_material),
                hashlib.sha256,
            ).hexdigest()
        ),
    }
    anchor_root = Path(environment["EQUINOX_RUNTIME_ANCHOR_ROOT"])
    anchor_root.mkdir(mode=0o700, exist_ok=True)
    anchor_root.chmod(0o700)
    anchor_name = (
        hmac.new(
            environment["EQUINOX_RESULT_TOKEN"].encode(),
            b"equinox/private-anchor-name/v1\0" + _canonical_json(launch_identity),
            hashlib.sha256,
        ).hexdigest()
        + ".json"
    )
    (anchor_root / anchor_name).write_bytes(_canonical_json(anchor) + b"\n")


def run_remote_runner(
    tmp_path: Path,
    failure_mode: str,
    workload_file: str = "repository_repair_rl.py",
    *,
    missing_variables: tuple[str, ...] = (),
    environment_overrides: dict[str, str] | None = None,
    authenticate_existing_state: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    asset_root = tmp_path.parent / f".{tmp_path.name}-runner-assets"
    asset_root.mkdir(exist_ok=True)
    anchor_root = asset_root / "anchor"
    anchor_root.mkdir(mode=0o700, exist_ok=True)
    anchor_root.chmod(0o700)
    persistent_root = asset_root / "persistent"
    persistent_root.mkdir(exist_ok=True)
    fake_bin = asset_root / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        """#!/bin/sh
case "$1" in
  -)
    if { [ "$EQUINOX_WORKLOAD_FILE" = "repository_repair_eligibility.py" ] ||
      [ "$EQUINOX_WORKLOAD_FILE" = "repository_repair_large_model_eligibility.py" ]; } &&
      grep -q '"screen_completed":[[:space:]]*true' "$2"; then
      exit 0
    fi
    if [ "$EQUINOX_WORKLOAD_FILE" != "repository_repair_eligibility.py" ] &&
      [ "$EQUINOX_WORKLOAD_FILE" != "repository_repair_large_model_eligibility.py" ] &&
      grep -q '"experiment_completed":[[:space:]]*true' "$2"; then
      exit 0
    fi
    exit 1
    ;;
  */result_server.py)
    : >"$EQUINOX_FAKE_SERVER_MARKER"
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
write_fake_checkpoint() {
  mkdir -p "$EQUINOX_ADAPTER_PATH/checkpoints"
  if [ -z "${EQUINOX_CHECKPOINT_AUTHENTICATION_KEY:-}" ]; then
    printf '%s\\n' '{"checkpoint":"update-1"}' \
      >"$EQUINOX_ADAPTER_PATH/checkpoints/latest.json"
    return
  fi
  "$EQUINOX_REAL_PYTHON" - \
    "$EQUINOX_ADAPTER_PATH/checkpoints/latest.json" \
    "$EQUINOX_PROGRESS_PATH" <<'PY'
import hashlib
import hmac
import json
import os
import sys

pointer_path, progress_path = sys.argv[1:]
generation = int(os.environ.get("EQUINOX_FAKE_POINTER_CHECKPOINT_GENERATION", "3"))
manifest_digest = os.environ.get(
    "EQUINOX_FAKE_POINTER_CHECKPOINT_MANIFEST_SHA256",
    "sha256:" + "c" * 64,
)
material = {
    "schema_version": 2,
    "revision": "launch-bound-checkpoint-manifest@1",
    "run_identity": os.environ["EQUINOX_RUN_IDENTITY"],
    "checkpoint": "update-0003",
    "generation": generation,
    "checkpoint_manifest_digest": manifest_digest,
}
pointer = {
    **material,
    "hmac": (
        "hmac-sha256:"
        + hmac.new(
            bytes.fromhex(os.environ["EQUINOX_CHECKPOINT_AUTHENTICATION_KEY"]),
            json.dumps(
                material,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            hashlib.sha256,
        ).hexdigest()
    ),
}
with open(pointer_path + ".pending", "w", encoding="utf-8") as handle:
    json.dump(
        pointer,
        handle,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(pointer_path + ".pending", pointer_path)
if os.environ.get("EQUINOX_FAKE_OMIT_PROGRESS_CHECKPOINT_BINDING") != "1":
    with open(progress_path, encoding="utf-8") as handle:
        progress = json.load(handle)
    progress["checkpoint_generation"] = int(
        os.environ.get(
            "EQUINOX_FAKE_PROGRESS_CHECKPOINT_GENERATION",
            str(generation),
        )
    )
    progress["checkpoint_manifest_digest"] = os.environ.get(
        "EQUINOX_FAKE_PROGRESS_CHECKPOINT_MANIFEST_SHA256",
        manifest_digest,
    )
    progress["checkpoint_name"] = "update-0003"
    with open(progress_path + ".pending", "w", encoding="utf-8") as handle:
        json.dump(
            progress,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(progress_path + ".pending", progress_path)
    if os.environ.get("EQUINOX_FAKE_HARDLINK_PROGRESS") == "1":
        os.link(
            progress_path,
            os.path.join(os.path.dirname(pointer_path), "progress-hardlink"),
        )
PY
}
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-authenticated-checkpoint-retry" ]; then
  if [ "$attempt" -eq 1 ] &&
    { [ -n "${EQUINOX_EXPECTED_CHECKPOINT_GENERATION+x}" ] ||
      [ -n "${EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256+x}" ]; }; then
    printf '%s\n' 'first attempt received checkpoint replay variables' >&2
    exit 92
  fi
  if [ "$attempt" -eq 2 ] &&
    { [ "$EQUINOX_EXPECTED_CHECKPOINT_GENERATION" != \
        "${EQUINOX_FAKE_POINTER_CHECKPOINT_GENERATION:-3}" ] ||
      [ "$EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256" != \
        "${EQUINOX_FAKE_POINTER_CHECKPOINT_MANIFEST_SHA256:-sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}" ]; }; then
    printf '%s\n' 'retry received the wrong checkpoint replay variables' >&2
    exit 93
  fi
  if [ "$attempt" -eq 2 ]; then
    printf '%s\n' 'authenticated retry binding observed' >&2
    exit 9
  fi
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "scientific-progress-overwrite" ]; then
  exec "$EQUINOX_REAL_PYTHON" "$@"
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-exit-cleared" ]; then
  printf '%s\\n' '{"experiment_completed":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-cublas-workspace" ] &&
  [ "$CUBLAS_WORKSPACE_CONFIG" != ":4096:8" ]; then
  printf '%s\n' 'missing deterministic cuBLAS workspace' >&2
  exit 89
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-token-scrubbed" ]; then
  if [ -n "${EQUINOX_RESULT_TOKEN:-}" ]; then
    printf '%s\n' 'scientific child received transport token' >&2
    exit 91
  fi
  if [ -e /proc/self/fd/13 ] || [ -e /dev/fd/13 ]; then
    printf '%s\n' 'scientific child received rollback anchor descriptor' >&2
    exit 92
  fi
  if [ -n "${EQUINOX_RUNTIME_ANCHOR_ROOT:-}" ] ||
    [ -n "${EQUINOX_RUNTIME_ANCHOR_ROOT_FD:-}" ]; then
    printf '%s\n' 'scientific child received rollback anchor identity' >&2
    exit 93
  fi
  printf '%s\n' '{"experiment_completed":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "adapter-success" ] ||
  [ "$EQUINOX_FAKE_FAILURE_MODE" = "adapter-manifest-mismatch" ] ||
  [ "$EQUINOX_FAKE_FAILURE_MODE" = "replace-result-after-pin" ] ||
  [ "$EQUINOX_FAKE_FAILURE_MODE" = "replace-adapter-after-pin" ]; then
  "$EQUINOX_REAL_PYTHON" - <<'PY'
import hashlib
import json
import os
import subprocess
import sys

adapter_root = os.path.realpath(os.environ["EQUINOX_ADAPTER_PATH"])
runtime_root = os.path.realpath(os.environ["EQUINOX_REMOTE_WORKDIR"])
os.makedirs(adapter_root, exist_ok=True)
files = {
    "adapter_config.json": b"{}\\n",
    "adapter_model.safetensors": b"verified-adapter-bytes",
}
for name, payload in files.items():
    with open(os.path.join(adapter_root, name), "wb") as handle:
        handle.write(payload)
manifest_content = {
    "schema_version": 1,
    "model_id": os.environ["EQUINOX_RL_MODEL_ID"],
    "model_revision": "fixture-revision",
    "workload_revision": "fixture-workload@1",
    "objective_id": "fixture-objective@1",
    "training_configuration": {},
    "files": [
        {
            "path": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(files.items())
    ],
}
canonical = lambda value: json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode()
manifest = {
    **manifest_content,
    "digest": "sha256:" + hashlib.sha256(canonical(manifest_content)).hexdigest(),
}
with open(
    os.path.join(adapter_root, "adapter-manifest.json"),
    "wb",
) as handle:
    handle.write(canonical(manifest) + b"\\n")
mode = os.environ["EQUINOX_FAKE_FAILURE_MODE"]
if mode == "adapter-manifest-mismatch":
    with open(
        os.path.join(adapter_root, "adapter_model.safetensors"),
        "ab",
    ) as handle:
        handle.write(b"-tampered")
if mode in {"replace-result-after-pin", "replace-adapter-after-pin"}:
    watcher = r'''
import json
import os
import sys
import time

root, mode = sys.argv[1:]
deadline = time.monotonic() + 5
marker = os.path.join(root, "." + mode + "-done")
while time.monotonic() < deadline:
    try:
        if mode == "replace-result-after-pin":
            manifest_path = os.path.join(root, "runner-resume-state.json")
            with open(manifest_path, encoding="utf-8") as handle:
                state = json.load(handle)
            ready = any(
                item.get("path") == "result.pending.json"
                for item in state.get("artifacts", [])
                if isinstance(item, dict)
            )
            target = os.path.join(root, "result.pending.json")
            payload = b'{"experiment_completed":true,"schema_version":99}\\n'
        else:
            ready = os.path.isfile(os.path.join(root, "result.json"))
            target = os.path.join(root, "adapter.tgz")
            payload = b"forged-adapter-archive"
        if ready and os.path.isfile(target):
            pending = target + ".attacker"
            with open(pending, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pending, target)
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("replaced\\n")
            raise SystemExit(0)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    time.sleep(0.0005)
raise SystemExit(2)
'''
    subprocess.Popen(
        [sys.executable, "-c", watcher, runtime_root, mode],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
result = {
    "schema_version": 2,
    "experiment_completed": True,
    "adapter_persisted": True,
    "adapter_manifest": manifest,
}
sys.stdout.buffer.write(canonical(result) + b"\\n")
PY
  exit $?
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "eligibility-success" ]; then
  printf '%s\n' '{"screen_completed":true,"protocol_eligible":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "assert-live-stage-before-workload" ]; then
  if [ ! -s "$EQUINOX_REMOTE_WORKDIR/live-stage-handoff.json" ] ||
    ! grep -q '"bundle_activation_digest"' \
      "$EQUINOX_REMOTE_WORKDIR/progress.json"; then
    printf '%s\n' 'scientific workload started before handoff evidence' >&2
    exit 90
  fi
  printf '%s\n' '{"screen_completed":true,"protocol_eligible":true}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "invalid-once" ] && [ "$attempt" -eq 1 ]; then
  write_fake_checkpoint
  printf '%s\\n' '{}'
  exit 0
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "cuda-oom" ]; then
  write_fake_checkpoint
  printf '%s\\n' 'torch.OutOfMemoryError: CUDA out of memory' >&2
  exit 7
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "model-load-failed" ]; then
  write_fake_checkpoint
  printf '%s\\n' 'RevisionNotFoundError: failed to load model' >&2
  exit 7
fi
if [ "$EQUINOX_FAKE_FAILURE_MODE" = "no-checkpoint" ]; then
  printf '%s\\n' 'failure before checkpoint' >&2
  exit 7
fi
if [ "$attempt" -eq 1 ]; then
  write_fake_checkpoint
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
    code_staging = asset_root / "code"
    if code_staging.exists():
        code_staging.chmod(0o755)
        for code_file in code_staging.iterdir():
            code_file.chmod(0o644)
    code_staging.mkdir(exist_ok=True)
    (code_staging / "result_server.py").write_text("", encoding="utf-8")
    (code_staging / "repository_repair_rl.py").write_text("", encoding="utf-8")
    (code_staging / "repository_repair_study.py").write_text("", encoding="utf-8")
    (code_staging / "repository_repair_eligibility.py").write_text(
        "",
        encoding="utf-8",
    )
    (code_staging / "repository_repair_large_model_eligibility.py").write_text(
        "",
        encoding="utf-8",
    )
    (code_staging / "repository_repair_large_model_pilot.py").write_text(
        "",
        encoding="utf-8",
    )
    (code_staging / "branching_sequence_ladder.py").write_text("", encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[2]
    (code_staging / "larger-model-dependencies.lock").write_bytes(
        (repository_root / "research/runpod/larger-model-dependencies.lock").read_bytes()
    )
    (code_staging / "_fixture_identity").write_text(
        str(tmp_path.resolve()),
        encoding="utf-8",
    )
    if failure_mode == "scientific-progress-overwrite":
        (code_staging / workload_file).write_text(
            """import json
import os

progress_path = os.environ["EQUINOX_PROGRESS_PATH"]
pending_path = progress_path + ".tmp"
with open(pending_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 2,
            "phase": "scientific_overwrite",
            "message": "Scientific progress without preserved context.",
        },
        handle,
    )
os.replace(pending_path, progress_path)
print(json.dumps({"screen_completed": True, "protocol_eligible": True}))
""",
            encoding="utf-8",
        )
    for code_file in code_staging.iterdir():
        code_file.chmod(0o444)
    code_staging.chmod(0o555)
    dependency_staging = asset_root / "dependencies"
    if dependency_staging.exists():
        dependency_staging.chmod(0o755)
        for dependency_file in dependency_staging.iterdir():
            dependency_file.chmod(0o644)
    dependency_staging.mkdir(exist_ok=True)
    (dependency_staging / "fake_dependency.py").write_text(
        "VERSION = '1.0'\n",
        encoding="utf-8",
    )
    (dependency_staging / "_fixture_identity").write_text(
        str(tmp_path.resolve()),
        encoding="utf-8",
    )
    for dependency_file in dependency_staging.iterdir():
        dependency_file.chmod(0o444)
    dependency_staging.chmod(0o555)
    code_root = code_staging
    dependency_root = dependency_staging
    materialization_env: dict[str, str] = {}
    if workload_file in LARGE_MODEL_WORKLOADS:
        code_root, dependency_root, materialization_env = materialization_environment(
            persistent_root,
            code_staging,
            dependency_staging,
        )

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EQUINOX_FAKE_FAILURE_MODE": failure_mode,
        "EQUINOX_FAKE_SERVER_MARKER": str(asset_root / "result-server-started"),
        "EQUINOX_REAL_PYTHON": sys.executable,
        "EQUINOX_DURABILITY_PYTHON": sys.executable,
        "EQUINOX_FILESYSTEM_PYTHON": sys.executable,
        "EQUINOX_RESULT_TOKEN": "test-token",
        "EQUINOX_PROOF_ID": (
            "runpod-proof-test-"
            + hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:24]
        ),
        "EQUINOX_RUN_IDENTITY": (
            "run-identity-"
            + hashlib.sha256(str(tmp_path.resolve()).encode()).hexdigest()[:24]
        ),
        "EQUINOX_REMOTE_WORKDIR": str(persistent_root),
        "EQUINOX_RUNTIME_ROOT": str(tmp_path),
        "EQUINOX_PRIVATE_RUNTIME_REVISION": "bootstrap-private-runtime@1",
        "EQUINOX_RUNTIME_ANCHOR_ROOT": str(anchor_root),
        "EQUINOX_PRIVATE_ANCHOR_REVISION": "bootstrap-private-anchor@1",
        "EQUINOX_CODE_ROOT": str(code_root),
        "EQUINOX_DEPENDENCY_ROOT": str(dependency_root),
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
    if workload_file in LARGE_MODEL_WORKLOADS:
        environment.update(
            live_stage_environment(
                dependency_quarantine_evidence_digest=materialization_env[
                    "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256"
                ],
                dependency_private_tree_digest=materialization_env[
                    "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256"
                ],
                code_materialization_evidence_digest=materialization_env[
                    "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256"
                ],
                code_private_tree_digest=materialization_env[
                    "EQUINOX_CODE_PRIVATE_TREE_SHA256"
                ],
            )
        )
        environment.update(materialization_env)
    if workload_file == "repository_repair_large_model_pilot.py":
        environment.update(
            {
                "EQUINOX_CHECKPOINT_AUTHENTICATION_KEY": (
                    checkpoint_authentication_key(tmp_path)
                ),
                "EQUINOX_FAKE_POINTER_CHECKPOINT_GENERATION": "3",
                "EQUINOX_FAKE_POINTER_CHECKPOINT_MANIFEST_SHA256": (
                    "sha256:" + "c" * 64
                ),
            }
        )
    if failure_mode == "archive-fail":
        environment["EQUINOX_ARCHIVE_PYTHON"] = "/bin/false"
    for variable in missing_variables:
        environment.pop(variable)
    environment.update(environment_overrides or {})
    if authenticate_existing_state:
        authenticate_runtime_state(tmp_path, environment, workload_file)
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


def test_remote_runner_does_not_expose_transport_token_to_science(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "assert-token-scrubbed")

    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }


def test_remote_runner_rejects_unbound_result_without_starting_science(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        '{"experiment_completed":true}\n',
        encoding="utf-8",
    )

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        check=False,
    )

    assert completed.returncode == 78
    assert not (tmp_path / "attempts").exists()
    assert "unauthenticated resumable artifacts" in completed.stderr


@pytest.mark.parametrize("artifact_name", ("progress.json", "result.pending.json"))
def test_remote_runner_rejects_symlinked_runtime_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-{artifact_name}.outside"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / artifact_name).symlink_to(outside)

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        check=False,
    )

    assert completed.returncode == 78
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert not (tmp_path / "attempts").exists()


def test_remote_runner_rejects_symlinked_adapter_without_traversal(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-adapter-outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_text("outside\n", encoding="utf-8")
    (tmp_path / "adapter").symlink_to(outside)

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        check=False,
    )

    assert completed.returncode == 78
    assert marker.read_text(encoding="utf-8") == "outside\n"
    assert not (tmp_path / "attempts").exists()


def test_remote_runner_rejects_unexpected_shared_volume_entry(
    tmp_path: Path,
) -> None:
    persistent_root = (
        tmp_path.parent / f".{tmp_path.name}-runner-assets/persistent"
    )
    persistent_root.mkdir(parents=True)
    (persistent_root / "forged-result.json").write_text(
        '{"experiment_completed":true}\n',
        encoding="utf-8",
    )

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        check=False,
    )

    assert completed.returncode == 78
    assert not (tmp_path / "attempts").exists()
    assert "persistent launch entry set failed" in completed.stderr


def test_remote_runner_rejects_same_run_manifest_rollback(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "no-checkpoint")
    saved_runtime = tmp_path.parent / f".{tmp_path.name}-saved-runtime"
    shutil.copytree(tmp_path, saved_runtime)

    run_remote_runner(tmp_path, "no-checkpoint")
    for path in sorted(
        tmp_path.rglob("*"),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    for source in saved_runtime.rglob("*"):
        relative = source.relative_to(saved_runtime)
        destination = tmp_path / relative
        if source.is_dir():
            destination.mkdir()
        else:
            shutil.copy2(source, destination)

    completed = run_remote_runner(
        tmp_path,
        "no-checkpoint",
        check=False,
    )

    assert completed.returncode == 78
    assert "rollback anchor did not match" in completed.stderr


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
    marker = tmp_path.parent / f".{tmp_path.name}-runner-assets/result-server-started"
    assert marker.is_file()


def test_remote_runner_retries_once_from_a_persisted_checkpoint(tmp_path: Path) -> None:
    completed = run_remote_runner(tmp_path, "retry-success")

    assert completed.stderr == ""
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "2"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "0"
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "experiment_completed": True
    }
    assert not (tmp_path / "adapter.tgz").exists()
    assert not (tmp_path / "adapter.tgz.pending").exists()
    assert not (tmp_path / "adapter" / "checkpoints").exists()
    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "=== attempt 1 · exit 7 ===" in error_log
    assert "transient failure" in error_log
    assert "=== attempt 2 · exit 0 ===" in error_log


def test_remote_runner_commits_only_a_manifest_validated_adapter_archive(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "adapter-success")

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["adapter_persisted"] is True
    archive_path = tmp_path / "adapter.tgz"
    assert archive_path.is_file()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        assert {
            member.name
            for member in archive.getmembers()
            if member.isfile()
        } == {
            "adapter/adapter-manifest.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
    integrity = json.loads(
        (tmp_path / "runner-resume-state.json").read_text(encoding="utf-8")
    )
    artifacts = {
        entry["path"]: entry
        for entry in integrity["artifacts"]
        if entry["path"] in {"result.json", "adapter.tgz"}
    }
    assert set(artifacts) == {"result.json", "adapter.tgz"}
    for path, entry in artifacts.items():
        payload = (tmp_path / path).read_bytes()
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_remote_runner_rejects_adapter_bytes_that_differ_from_the_result_manifest(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "adapter-manifest-mismatch", check=False)

    assert not (tmp_path / "adapter.tgz").exists()
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "result.pending.json").is_file()
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "ARTIFACT_ARCHIVE_FAILED"


@pytest.mark.parametrize(
    ("failure_mode", "target_name", "forged_payload"),
    (
        (
            "replace-result-after-pin",
            "result.pending.json",
            b'{"experiment_completed":true,"schema_version":99}\n',
        ),
        (
            "replace-adapter-after-pin",
            "adapter.tgz",
            b"forged-adapter-archive",
        ),
    ),
)
def test_remote_runner_never_hmac_commits_a_replacement_after_pinning(
    tmp_path: Path,
    failure_mode: str,
    target_name: str,
    forged_payload: bytes,
) -> None:
    run_remote_runner(tmp_path, failure_mode, check=False)
    marker = tmp_path / f".{failure_mode}-done"
    deadline = time.monotonic() + 3
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.is_file()

    integrity = json.loads(
        (tmp_path / "runner-resume-state.json").read_text(encoding="utf-8")
    )
    signed = [
        entry
        for entry in integrity["artifacts"]
        if entry["path"] in {target_name, "result.json"}
    ]
    forged_digest = "sha256:" + hashlib.sha256(forged_payload).hexdigest()
    assert all(entry["sha256"] != forged_digest for entry in signed)
    if target_name == "result.pending.json":
        assert not any(entry["path"] == "result.json" for entry in signed)
    else:
        assert (tmp_path / target_name).read_bytes() == forged_payload


def test_remote_runner_honors_a_single_attempt_budget(tmp_path: Path) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "7"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED"


@pytest.mark.parametrize("maximum_attempts", ("0", "3", "many"))
def test_remote_runner_rejects_an_invalid_attempt_budget(
    tmp_path: Path,
    maximum_attempts: str,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": maximum_attempts},
        check=False,
    )

    assert completed.returncode == 1
    assert not (tmp_path / "attempts").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "WORKLOAD_ATTEMPT_LIMIT_INVALID"


@pytest.mark.parametrize(
    ("failure_mode", "expected_error"),
    (
        ("cuda-oom", "GPU_MEMORY_EXHAUSTED"),
        ("model-load-failed", "MODEL_LOAD_FAILED"),
    ),
)
def test_remote_runner_does_not_retry_a_deterministic_expensive_failure(
    tmp_path: Path,
    failure_mode: str,
    expected_error: str,
) -> None:
    run_remote_runner(tmp_path, failure_mode)

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "non-retryable-failure").read_text(
        encoding="utf-8"
    ).strip() == expected_error
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == expected_error
    assert "waste compute" in progress["message"]


def test_remote_runner_retains_a_non_retryable_failure_across_restart(
    tmp_path: Path,
) -> None:
    run_remote_runner(tmp_path, "cuda-oom")
    run_remote_runner(tmp_path, "retry-success")

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "GPU_MEMORY_EXHAUSTED"


def test_remote_runner_rejects_unbound_interrupted_oom_state(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "adapter" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "latest.json").write_text(
        '{"checkpoint":"update-1"}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")
    (tmp_path / "error.attempt-1.log").write_text(
        "torch.OutOfMemoryError: CUDA out of memory\n",
        encoding="utf-8",
    )

    completed = run_remote_runner(tmp_path, "retry-success", check=False)

    assert completed.returncode == 78
    assert not (tmp_path / "attempts").exists()
    assert not (tmp_path / "runner-resume-state.json").exists()
    assert "not authenticated" in completed.stderr


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
    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

    observed = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert observed == {
        **specific_progress,
        "remote_error": {
            "code": "REMOTE_WORKLOAD_FAILURE",
            "message": "RuntimeError: CUDA out of memory",
            "exit_code": 1,
        },
    }


def test_remote_runner_preserves_last_known_progress_when_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    validation_history = [
        {
            "update": 5,
            "level": 0,
            "fixed_guard_paired_change": {"improved": 0, "regressed": 1},
        }
    ]
    branch_snapshots = [
        {
            "snapshot_id": "update-5-snapshot-a",
            "siblings": [
                {
                    "index": index,
                    "steps": [{"step_id": f"sibling-{index}-1", "tool": "edit"}],
                }
                for index in range(4)
            ],
        }
    ]
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
                "validation_history": validation_history,
                "branch_snapshots": branch_snapshots,
                "latest_branch_snapshot": branch_snapshots[-1],
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "failed"
    assert progress["error"] == "WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED"
    assert progress["update"] == 19
    assert progress["current_level"] == 2
    assert progress["elapsed_seconds"] == 3100.5
    assert progress["validation_history"] == validation_history
    assert progress["branch_snapshots"] == branch_snapshots
    assert progress["latest_branch_snapshot"] == branch_snapshots[-1]
    assert progress["remote_error"] == {
        "code": "WORKLOAD_ATTEMPT_BUDGET_EXHAUSTED",
        "message": "The bounded workload attempt budget is exhausted.",
        "exit_code": 1,
    }


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

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "phase": "failed",
                "error": "PREVIOUS_REMOTE_FAILURE",
                "remote_error": {
                    "code": "PREVIOUS_REMOTE_FAILURE",
                    "message": "Previous attempt failed.",
                },
                "operator_error": {
                    "code": "PREVIOUS_OPERATOR_FAILURE",
                    "message": "Previous operator failed.",
                },
                "validation_history": [{"update": 3}],
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "assert-exit-cleared",
        authenticate_existing_state=True,
    )

    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "0"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    assert progress["validation_history"] == [{"update": 3}]
    assert "error" not in progress
    assert "remote_error" not in progress
    assert "operator_error" not in progress


def test_remote_runner_uses_progress_message_for_a_stable_legacy_error_code(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "failed",
                "message": "No policy-bearing checkpoint passed retention.",
                "error": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["remote_error"] == {
        "code": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
        "message": "No policy-bearing checkpoint passed retention.",
        "exit_code": 1,
    }


def test_remote_runner_uses_default_for_a_non_string_legacy_progress_message(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "failed",
                "message": {"unexpected": "object"},
                "error": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["remote_error"] == {
        "code": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
        "message": "The remote workload failed.",
        "exit_code": 1,
    }


@pytest.mark.parametrize(
    "malformed_remote_error",
    [
        {"code": 17, "message": "Malformed remote code."},
        {"code": "REMOTE_WORKLOAD_FAILURE", "message": None},
    ],
)
def test_remote_runner_reconstructs_malformed_structured_remote_errors(
    tmp_path: Path,
    malformed_remote_error: dict[str, object],
) -> None:
    (tmp_path / "workload-attempt-count").write_text("2\n", encoding="utf-8")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "phase": "failed",
                "message": "No policy-bearing checkpoint passed retention.",
                "error": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
                "remote_error": malformed_remote_error,
            }
        ),
        encoding="utf-8",
    )

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["remote_error"] == {
        "code": "PILOT_PRODUCED_NO_RETAINED_POLICY_UPDATE",
        "message": "No policy-bearing checkpoint passed retention.",
        "exit_code": 1,
    }


def test_remote_runner_recovers_a_completed_pending_result_after_restart(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.pending.json").write_text(
        '{"experiment_completed":true}\n',
        encoding="utf-8",
    )
    (tmp_path / "workload-attempt-count").write_text("1\n", encoding="utf-8")

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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

    run_remote_runner(
        tmp_path,
        "always-fail",
        authenticate_existing_state=True,
    )

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
        authenticate_existing_state=True,
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


def test_remote_runner_accepts_large_model_eligibility_without_study_or_adapter(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "eligibility-success",
        workload_file="repository_repair_large_model_eligibility.py",
        missing_variables=(
            "EQUINOX_STUDY_CONDITION",
            "EQUINOX_STUDY_VALIDATION_SEED_BASE",
            "EQUINOX_STUDY_TEST_SEED_BASE",
        ),
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result == {"screen_completed": True, "protocol_eligible": True}
    assert not (tmp_path / "adapter.tgz").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    expected_evidence = live_stage_progress_evidence(tmp_path)
    assert {key: progress[key] for key in expected_evidence} == expected_evidence
    assert all(value is not None for value in expected_evidence.values())
    assert (
        json.loads((tmp_path / "live-stage-handoff.json").read_text(encoding="utf-8"))
        == expected_evidence
    )


@pytest.mark.parametrize(
    ("missing_variable", "expected_error"),
    (
        ("EQUINOX_LARGER_MODEL_PROFILE_ID", "PRIVATE_MATERIALIZATION_INVALID"),
        ("EQUINOX_SOURCE_HEAD_COMMIT", "LIVE_STAGE_HANDOFF_INVALID"),
        (
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        ("EQUINOX_BOOTSTRAP_SOURCE_SHA256", "LIVE_STAGE_HANDOFF_INVALID"),
        ("EQUINOX_BUNDLE_SHA256", "PRIVATE_MATERIALIZATION_INVALID"),
        ("EQUINOX_BUNDLE_SIZE_BYTES", "PRIVATE_MATERIALIZATION_INVALID"),
        ("EQUINOX_BUNDLE_VOLUME_PATH", "PRIVATE_MATERIALIZATION_INVALID"),
        ("EQUINOX_BUNDLE_STAGE_RECEIPT_SHA256", "LIVE_STAGE_HANDOFF_INVALID"),
        (
            "EQUINOX_VOLUME_READINESS_RECEIPT_SHA256",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        (
            "EQUINOX_TORCH_RETENTION_EVIDENCE_SHA256",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        ("EQUINOX_DEPENDENCY_LOCK_SHA256", "PRIVATE_MATERIALIZATION_INVALID"),
        ("EQUINOX_RUNPOD_NETWORK_VOLUME_ID", "LIVE_STAGE_HANDOFF_INVALID"),
        (
            "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        ("EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB", "LIVE_STAGE_HANDOFF_INVALID"),
        ("EQUINOX_BUNDLE_HANDOFF_REVISION", "LIVE_STAGE_HANDOFF_INVALID"),
        ("EQUINOX_BUNDLE_ACTIVATION_DIGEST", "LIVE_STAGE_HANDOFF_INVALID"),
        (
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_PATH",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_CODE_MATERIALIZATION_EVIDENCE_SHA256",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_CODE_PRIVATE_TREE_SHA256",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_PATH",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_DEPENDENCY_QUARANTINE_EVIDENCE_SHA256",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_DEPENDENCY_PRIVATE_TREE_SHA256",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
    ),
)
def test_large_model_workload_rejects_missing_live_stage_evidence_before_science(
    tmp_path: Path,
    missing_variable: str,
    expected_error: str,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "eligibility-success",
        workload_file="repository_repair_large_model_eligibility.py",
        missing_variables=(missing_variable,),
        check=False,
    )

    assert completed.returncode == 1
    assert not (tmp_path / "attempts").exists()
    assert not (tmp_path / "workload-attempt-count").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == expected_error
    assert all(value is not None for value in progress.values())


@pytest.mark.parametrize(
    ("variable", "malformed_value", "expected_error"),
    (
        (
            "EQUINOX_LARGER_MODEL_PROFILE_ID",
            "../wrong-profile",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        ("EQUINOX_SOURCE_HEAD_COMMIT", "abc123", "LIVE_STAGE_HANDOFF_INVALID"),
        (
            "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256",
            "2" * 64,
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        ("EQUINOX_BUNDLE_SIZE_BYTES", "0", "PRIVATE_MATERIALIZATION_INVALID"),
        (
            "EQUINOX_BUNDLE_SIZE_BYTES",
            str(2 * 1024 * 1024 + 1),
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_BUNDLE_VOLUME_PATH",
            "/workspace/not-content-addressed.tar.xz",
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_DEPENDENCY_LOCK_SHA256",
            "sha256:" + "0" * 64,
            "PRIVATE_MATERIALIZATION_INVALID",
        ),
        (
            "EQUINOX_RUNPOD_NETWORK_VOLUME_ID",
            "volume/escape",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        (
            "EQUINOX_RUNPOD_NETWORK_VOLUME_DATA_CENTER_ID",
            "EU/FR/1",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        (
            "EQUINOX_RUNPOD_NETWORK_VOLUME_SIZE_GB",
            "-1",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        (
            "EQUINOX_BUNDLE_HANDOFF_REVISION",
            "old-handoff@1",
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
        (
            "EQUINOX_BUNDLE_ACTIVATION_DIGEST",
            "sha256:" + "f" * 64,
            "LIVE_STAGE_HANDOFF_INVALID",
        ),
    ),
)
def test_large_model_workload_rejects_malformed_live_stage_evidence_before_science(
    tmp_path: Path,
    variable: str,
    malformed_value: str,
    expected_error: str,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "eligibility-success",
        workload_file="repository_repair_large_model_eligibility.py",
        environment_overrides={variable: malformed_value},
        check=False,
    )

    assert completed.returncode == 1
    assert not (tmp_path / "attempts").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == expected_error


def test_large_model_science_starts_after_exact_handoff_progress(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "assert-live-stage-before-workload",
        workload_file="repository_repair_large_model_eligibility.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    expected_evidence = live_stage_progress_evidence(tmp_path)
    assert {key: progress[key] for key in expected_evidence} == expected_evidence


def test_large_model_scientific_progress_cannot_drop_handoff_evidence(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "scientific-progress-overwrite",
        workload_file="repository_repair_large_model_eligibility.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    expected_evidence = live_stage_progress_evidence(tmp_path)
    assert {key: progress[key] for key in expected_evidence} == expected_evidence


def test_large_model_restart_requires_the_same_live_stage_handoff(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )
    persisted_evidence = json.loads(
        (tmp_path / "live-stage-handoff.json").read_text(encoding="utf-8")
    )

    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2"},
    )

    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "complete"
    assert {key: progress[key] for key in persisted_evidence} == persisted_evidence
    assert (
        json.loads((tmp_path / "live-stage-handoff.json").read_text(encoding="utf-8"))
        == persisted_evidence
    )


def test_large_model_restart_rejects_changed_handoff_before_second_attempt(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )
    original_evidence = json.loads(
        (tmp_path / "live-stage-handoff.json").read_text(encoding="utf-8")
    )
    changed_environment = {
        "EQUINOX_LARGER_MODEL_SOURCE_CONTRACT_SHA256": "sha256:" + "a" * 64,
        "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2",
    }

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides=changed_environment,
        check=False,
    )

    assert completed.returncode == 78
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "workload-attempt-count").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert {key: progress[key] for key in original_evidence} == original_evidence
    assert "resume manifest authentication failed" in completed.stderr


def test_large_model_restart_rejects_tampered_progress_handoff(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
    )
    progress_path = tmp_path / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["head_commit"] = "f" * 40
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    completed = run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2"},
        check=False,
    )

    assert completed.returncode == 78
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    observed = json.loads(progress_path.read_text(encoding="utf-8"))
    assert observed["head_commit"] == "f" * 40
    assert "authenticated resume artifacts changed" in completed.stderr


def test_remote_runner_accepts_large_model_pilot_without_study_configuration(
    tmp_path: Path,
) -> None:
    run_remote_runner(
        tmp_path,
        "retry-success",
        workload_file="repository_repair_large_model_pilot.py",
        missing_variables=(
            "EQUINOX_STUDY_CONDITION",
            "EQUINOX_STUDY_VALIDATION_SEED_BASE",
            "EQUINOX_STUDY_TEST_SEED_BASE",
        ),
    )

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result == {"experiment_completed": True}
    assert not (tmp_path / "adapter.tgz").exists()


def test_large_model_retry_uses_exact_private_checkpoint_replay_handoff(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={
            "EQUINOX_EXPECTED_CHECKPOINT_GENERATION": "999",
            "EQUINOX_EXPECTED_CHECKPOINT_MANIFEST_SHA256": (
                "sha256:" + "e" * 64
            ),
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2",
        },
        check=False,
    )

    assert completed.returncode == 0
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "9"
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "2"
    error_log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "authenticated retry binding observed" in error_log
    assert "first attempt received checkpoint replay variables" not in error_log
    assert "retry received the wrong checkpoint replay variables" not in error_log
    checkpoint_secret = checkpoint_authentication_key(tmp_path)
    assert checkpoint_secret not in error_log
    assert checkpoint_secret not in completed.stdout
    assert checkpoint_secret not in completed.stderr


def test_large_model_retry_rejects_missing_progress_checkpoint_binding(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={
            "EQUINOX_FAKE_OMIT_PROGRESS_CHECKPOINT_BINDING": "1",
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2",
        },
        check=False,
    )

    assert completed.returncode == 0
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "CHECKPOINT_REPLAY_HANDOFF_INVALID"
    assert "checkpoint_generation" not in progress
    assert "checkpoint_manifest_digest" not in progress


def test_large_model_retry_rejects_pointer_progress_checkpoint_mismatch(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={
            "EQUINOX_FAKE_PROGRESS_CHECKPOINT_MANIFEST_SHA256": (
                "sha256:" + "d" * 64
            ),
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2",
        },
        check=False,
    )

    assert completed.returncode == 0
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "CHECKPOINT_REPLAY_HANDOFF_INVALID"


def test_large_model_retry_rejects_hardlinked_progress_checkpoint_binding(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={
            "EQUINOX_FAKE_HARDLINK_PROGRESS": "1",
            "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2",
        },
        check=False,
    )

    assert completed.returncode == 0
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "CHECKPOINT_REPLAY_HANDOFF_INVALID"


def test_large_model_retry_rejects_missing_checkpoint_authentication_key(
    tmp_path: Path,
) -> None:
    completed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2"},
        missing_variables=("EQUINOX_CHECKPOINT_AUTHENTICATION_KEY",),
        check=False,
    )

    assert completed.returncode == 0
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "CHECKPOINT_REPLAY_HANDOFF_INVALID"


def test_large_model_retry_rejects_generation_regression_against_private_anchor(
    tmp_path: Path,
) -> None:
    first_attempt = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "1"},
        check=False,
    )
    assert first_attempt.returncode == 0
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    anchor_root = tmp_path.parent / f".{tmp_path.name}-runner-assets/anchor"
    assert len(list(anchor_root.glob("checkpoint-*.json"))) == 1

    rewrite_authenticated_checkpoint_pair(
        tmp_path,
        generation=2,
        manifest_digest="sha256:" + "b" * 64,
    )
    resumed = run_remote_runner(
        tmp_path,
        "assert-authenticated-checkpoint-retry",
        workload_file="repository_repair_large_model_pilot.py",
        environment_overrides={"EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS": "2"},
        authenticate_existing_state=True,
        check=False,
    )

    assert resumed.returncode == 0
    assert (tmp_path / "attempts").read_text(encoding="utf-8").strip() == "1"
    assert (tmp_path / "exit_code").read_text(encoding="utf-8").strip() == "74"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["error"] == "CHECKPOINT_REPLAY_HANDOFF_INVALID"
    assert "authenticated retry binding observed" not in (
        tmp_path / "error.log"
    ).read_text(encoding="utf-8")


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
        authenticate_existing_state=True,
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
