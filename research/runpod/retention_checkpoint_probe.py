"""Prove a real AdamW retention checkpoint can persist, restore, and advance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
HEAD_COMMIT = re.compile(r"[0-9a-f]{40}")
EVIDENCE_REVISION = "real-adamw-persist-restore-advance@7"
TEST_ID = "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"


def canonical_json(value: Any) -> bytes:
    """Return compact, stable JSON suitable for an evidence digest."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return a tagged SHA-256 digest for one source or checkpoint file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def evidence_digest(evidence: Mapping[str, Any]) -> str:
    """Digest evidence while excluding only its self-reported digest."""

    material = {key: value for key, value in evidence.items() if key != "evidence_digest"}
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def _optimizer_state_material(value: Any, torch_module: Any) -> Any:
    if torch_module.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        raw_bytes = bytes(tensor.view(torch_module.uint8).reshape(-1).tolist())
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
    if isinstance(value, Mapping):
        items = [
            {
                "key": _optimizer_state_material(key, torch_module),
                "value": _optimizer_state_material(item, torch_module),
            }
            for key, item in value.items()
        ]
        items.sort(key=lambda item: canonical_json(item["key"]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_optimizer_state_material(item, torch_module) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_optimizer_state_material(item, torch_module) for item in value],
        }
    if value is None or type(value) in {bool, int, float, str}:
        return {"kind": type(value).__name__, "value": value}
    raise RuntimeError(f"unsupported optimizer-state value: {type(value).__name__}")


def optimizer_state_digest(state: Mapping[str, Any], torch_module: Any) -> str:
    """Hash tensor bytes and the complete typed AdamW state structure."""

    if not isinstance(state, Mapping):
        raise RuntimeError("optimizer state must be a mapping")
    material = _optimizer_state_material(state, torch_module)
    return "sha256:" + hashlib.sha256(canonical_json(material)).hexdigest()


def optimizer_states_equal(left: Any, right: Any, torch_module: Any) -> bool:
    """Compare complete optimizer state without dtype or container coercion."""

    left_is_tensor = torch_module.is_tensor(left)
    right_is_tensor = torch_module.is_tensor(right)
    if left_is_tensor or right_is_tensor:
        if not left_is_tensor or not right_is_tensor:
            return False
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch_module.equal(left.detach().cpu(), right.detach().cpu()))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(optimizer_states_equal(left[key], right[key], torch_module) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                optimizer_states_equal(a, b, torch_module) for a, b in zip(left, right, strict=True)
            )
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and len(left) == len(right)
            and all(
                optimizer_states_equal(a, b, torch_module) for a, b in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def optimizer_state_tensor_devices(
    state: Mapping[str, Any],
    torch_module: Any,
) -> dict[str, list[str]]:
    """Return the exact device placement of each named optimizer-state tensor."""

    parameter_states = state.get("state") if isinstance(state, Mapping) else None
    if not isinstance(parameter_states, Mapping) or not parameter_states:
        raise RuntimeError("optimizer state does not contain parameter state")
    devices: dict[str, set[str]] = {}
    for parameter_state in parameter_states.values():
        if not isinstance(parameter_state, Mapping):
            raise RuntimeError("optimizer parameter state must be a mapping")
        for name, value in parameter_state.items():
            if torch_module.is_tensor(value):
                devices.setdefault(str(name), set()).add(str(value.device))
    if not devices:
        raise RuntimeError("optimizer state does not contain tensors")
    return {name: sorted(values) for name, values in sorted(devices.items())}


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _source_layout(
    source_root: Path,
    *,
    flat_bundle: bool,
) -> tuple[Path, Path, Path]:
    source_root = source_root.resolve()
    if flat_bundle:
        return (
            source_root,
            source_root / "larger-model-eligibility.json",
            source_root / "repository_repair_large_model_trainer.py",
        )
    return (
        source_root / "research/runpod",
        source_root / "research/studies/larger-model-eligibility.json",
        source_root / "research/runpod/repository_repair_large_model_trainer.py",
    )


def _import_trainer(source_directory: Path, *, flat_bundle: bool) -> Any:
    if flat_bundle:
        source_text = str(source_directory)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        return importlib.import_module("repository_repair_large_model_trainer")
    return importlib.import_module("research.runpod.repository_repair_large_model_trainer")


def _verify_sources(
    source_directory: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    source_contract = manifest.get("source_contract", {}).get("files")
    if not isinstance(source_contract, dict) or not source_contract:
        raise RuntimeError("retention probe source contract is unavailable")
    observed: dict[str, str] = {}
    for name, expected in source_contract.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise RuntimeError("retention probe source contract is malformed")
        source_path = source_directory / name
        if not source_path.is_file():
            raise RuntimeError(f"retention probe source is unavailable: {name}")
        digest = sha256_file(source_path)
        if digest != f"sha256:{expected}":
            raise RuntimeError(f"retention probe source contract mismatch: {name}")
        observed[name] = digest
    return observed


@contextmanager
def _checkpoint_workspace(storage_parent: Path | None) -> Any:
    if storage_parent is None:
        with tempfile.TemporaryDirectory(prefix="equinox-retention-roundtrip-") as directory:
            yield Path(directory)
        return

    try:
        resolved_parent = storage_parent.resolve(strict=True)
        parent_metadata = storage_parent.lstat()
    except OSError as error:
        raise RuntimeError("retention checkpoint storage root is unavailable") from error
    if not resolved_parent.is_dir() or resolved_parent != storage_parent.absolute():
        raise RuntimeError("retention checkpoint storage root must be a real directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(resolved_parent, flags)
    name = ".equinox-retention-" + secrets.token_hex(16)
    try:
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise RuntimeError("retention checkpoint storage root must be a directory")
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        workspace = resolved_parent / name
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace)
    finally:
        os.close(parent_descriptor)


def _restore_and_advance_checkpoint(
    trainer: Any,
    torch_module: Any,
    *,
    checkpoint_path: Path,
    device: str,
    expected_optimizer_state_digest: str,
    expected_optimizer_state_devices: Mapping[str, Any],
    expected_retained_weight: list[float],
    expected_optimizer_state: Mapping[str, Any] | None = None,
    checkpoint_authentication_key: bytes,
    checkpoint_authentication_identity: dict[str, str],
    checkpoint_generation: int,
    checkpoint_manifest_digest: str,
) -> dict[str, Any]:
    authenticated = trainer.load_authenticated_checkpoint(
        checkpoints_root=str(checkpoint_path.parent.parent),
        authentication_key=checkpoint_authentication_key,
        authentication_identity=checkpoint_authentication_identity,
        expected_generation=checkpoint_generation,
        expected_manifest_digest=checkpoint_manifest_digest,
    )
    private_checkpoint_path = Path(authenticated.private_directory) / "training-state.pt"
    try:
        resumed_state = torch_module.load(
            private_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    finally:
        shutil.rmtree(authenticated.private_directory, ignore_errors=True)
    restored = trainer.validate_retained_transaction_state(
        resumed_state["retained_transaction"],
        best_validation_update=2,
        retained_policy_update_count=1,
    )
    if optimizer_state_digest(
        restored["optimizer_state"], torch_module
    ) != expected_optimizer_state_digest or (
        expected_optimizer_state is not None
        and not optimizer_states_equal(
            restored["optimizer_state"],
            expected_optimizer_state,
            torch_module,
        )
    ):
        raise RuntimeError("retention checkpoint changed optimizer state during persistence")
    resumed_parameter = torch_module.nn.Parameter(torch_module.tensor([-9.0], device=device))
    resumed_optimizer = torch_module.optim.AdamW([resumed_parameter], lr=0.01)
    effective_count, pending, pending_policy, pending_groups, observation = (
        trainer.restore_retention_transaction(
            restored,
            restore_trainable_state=lambda state: resumed_parameter.data.copy_(state["weight"]),
            optimizer=resumed_optimizer,
        )
    )
    restored_optimizer_state = resumed_optimizer.state_dict()
    optimizer_state_digest_after_restore = optimizer_state_digest(
        restored_optimizer_state,
        torch_module,
    )
    optimizer_state_devices_after_restore = optimizer_state_tensor_devices(
        restored_optimizer_state,
        torch_module,
    )
    restored_weight_before_resume_step = resumed_parameter.detach().tolist()
    if (
        restored_weight_before_resume_step != expected_retained_weight
        or effective_count != 1
        or pending != []
        or pending_policy != 0
        or pending_groups != []
        or observation != {"exact_rate": 0.5}
        or not resumed_optimizer.state
        or optimizer_state_digest_after_restore != expected_optimizer_state_digest
        or optimizer_state_devices_after_restore != expected_optimizer_state_devices
    ):
        raise RuntimeError("retention checkpoint did not restore its exact transaction")

    resumed_parameter.square().sum().backward()
    resumed_optimizer.step()
    safely_retained = trainer.capture_retention_transaction(
        capture_trainable_state=lambda: {"weight": resumed_parameter.detach().clone()},
        optimizer=resumed_optimizer,
        effective_policy_update_count=2,
        update=3,
        retained_observation={"exact_rate": 0.75},
    )
    advanced_weight_after_resume_step = resumed_parameter.detach().tolist()
    optimizer_state_digest_after_resume_step = optimizer_state_digest(
        resumed_optimizer.state_dict(),
        torch_module,
    )
    if (
        safely_retained["effective_policy_update_count"] != 2
        or safely_retained["retained_observation"] != {"exact_rate": 0.75}
        or restored_weight_before_resume_step == advanced_weight_after_resume_step
        or not resumed_optimizer.state
        or optimizer_state_digest_after_resume_step == optimizer_state_digest_after_restore
    ):
        raise RuntimeError("resumed AdamW state could not be retained safely")
    return {
        "restored_weight_before_resume_step": restored_weight_before_resume_step,
        "advanced_weight_after_resume_step": advanced_weight_after_resume_step,
        "effective_policy_update_count_before_resume_step": effective_count,
        "effective_policy_update_count_after_resume_step": safely_retained[
            "effective_policy_update_count"
        ],
        "retained_observation_after_resume_step": safely_retained["retained_observation"],
        "optimizer_state_entries_after_resume_step": len(resumed_optimizer.state),
        "optimizer_state_digest_after_restore": optimizer_state_digest_after_restore,
        "optimizer_state_digest_after_resume_step": (optimizer_state_digest_after_resume_step),
        "optimizer_parameter_device": str(resumed_parameter.device),
        "optimizer_state_devices_after_restore": (optimizer_state_devices_after_restore),
        "checkpoint_sha256": authenticated.source_training_state_digest,
        "checkpoint_size_bytes": authenticated.source_training_state_size_bytes,
        "checkpoint_source_device": authenticated.source_training_state_device,
        "checkpoint_inode_after_reopen": authenticated.source_training_state_inode,
        "checkpoint_authentication_revision": (trainer.CHECKPOINT_AUTHENTICATION_REVISION),
        "checkpoint_authentication_mechanism_digest": (
            trainer.checkpoint_authentication_mechanism_digest()
        ),
        "checkpoint_generation": authenticated.generation,
        "checkpoint_manifest_digest": authenticated.manifest_digest,
        "checkpoint_authenticated_private_resume": True,
        "checkpoint_resume_process_pid": os.getpid(),
        "checkpoint_resume_parent_process_pid": os.getppid(),
    }


def _resume_checkpoint_in_fresh_process(
    *,
    probe_path: Path,
    source_directory: Path,
    checkpoint_path: Path,
    device: str,
    expected_optimizer_state_digest: str,
    expected_optimizer_state_devices: Mapping[str, Any],
    expected_retained_weight: list[float],
    checkpoint_authentication_key: bytes,
    checkpoint_authentication_identity: dict[str, str],
    checkpoint_generation: int,
    checkpoint_manifest_digest: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(probe_path),
        "--resume-roundtrip-checkpoint",
        str(checkpoint_path),
        "--resume-source-directory",
        str(source_directory),
        "--resume-device",
        device,
        "--expected-optimizer-state-digest",
        expected_optimizer_state_digest,
        "--expected-optimizer-state-devices-json",
        json.dumps(expected_optimizer_state_devices, sort_keys=True),
        "--expected-retained-weight-json",
        json.dumps(expected_retained_weight),
    ]
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_KEY": (checkpoint_authentication_key.hex()),
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_IDENTITY": json.dumps(
                checkpoint_authentication_identity,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_GENERATION": str(checkpoint_generation),
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_MANIFEST_SHA256": (checkpoint_manifest_digest),
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=child_environment,
        timeout=120,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise RuntimeError(f"fresh retention resume process failed: {diagnostic}")
    try:
        result = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("fresh retention resume process returned invalid evidence") from error
    if not isinstance(result, dict):
        raise RuntimeError("fresh retention resume process returned invalid evidence")
    return result


def _checkpoint_roundtrip(
    trainer: Any,
    torch_module: Any,
    *,
    device: str,
    storage_parent: Path | None = None,
    storage_scope: str | None = None,
    fresh_process_probe_path: Path | None = None,
    fresh_process_source_directory: Path | None = None,
    checkpoint_authentication_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    with _checkpoint_workspace(storage_parent) as temporary_root:
        parameter = torch_module.nn.Parameter(torch_module.tensor([1.0], device=device))
        optimizer = torch_module.optim.AdamW([parameter], lr=0.01)
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        retained_weight = parameter.detach().clone()
        transaction = trainer.capture_retention_transaction(
            capture_trainable_state=lambda: {"weight": parameter.detach().clone()},
            optimizer=optimizer,
            effective_policy_update_count=1,
            update=2,
            retained_observation={"exact_rate": 0.5},
        )
        expected_optimizer_state = copy.deepcopy(transaction["optimizer_state"])
        optimizer_state_digest_before_persist = optimizer_state_digest(
            expected_optimizer_state,
            torch_module,
        )
        optimizer_state_devices_before_persist = optimizer_state_tensor_devices(
            expected_optimizer_state,
            torch_module,
        )

        parameter.square().sum().backward()
        optimizer.step()
        checkpoints_root = temporary_root / "checkpoints"
        latest = checkpoints_root / "latest.json"

        def save_adapter(target: str) -> None:
            Path(target, "adapter_config.json").write_text("{}", encoding="utf-8")
            Path(target, "adapter_model.safetensors").write_bytes(b"adapter")

        authentication_key = secrets.token_bytes(32)
        authentication_identity = checkpoint_authentication_identity or {
            "run_identity": f"retention-probe-{secrets.token_hex(16)}",
            "profile_id": "retention-probe-local",
            "authorization_digest": "sha256:" + "1" * 64,
            "source_head_commit": "2" * 40,
            "workload_bundle_digest": "sha256:" + "3" * 64,
            "workload_revision": trainer.WORKLOAD_REVISION,
            "model_revision": "4" * 40,
            "objective_id": trainer.OBJECTIVE_ID,
        }
        commit = trainer.persist_checkpoint(
            checkpoints_root=str(checkpoints_root),
            latest_checkpoint_path=str(latest),
            checkpoint_name="update-0002",
            state={
                "checkpoint_generation": 1,
                "retained_transaction": transaction,
            },
            save_adapter=save_adapter,
            save_state=torch_module.save,
            generation=1,
            authentication_key=authentication_key,
            authentication_identity=authentication_identity,
        )
        workspace_descriptor = os.open(
            temporary_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(workspace_descriptor)
        finally:
            os.close(workspace_descriptor)
        checkpoint_path = checkpoints_root / "update-0002" / "training-state.pt"
        persisted_checkpoint = trainer.load_authenticated_checkpoint(
            checkpoints_root=str(checkpoints_root),
            authentication_key=authentication_key,
            authentication_identity=authentication_identity,
            expected_generation=commit.generation,
            expected_manifest_digest=commit.manifest_digest,
        )
        checkpoint_inode_before_reopen = persisted_checkpoint.source_training_state_inode
        checkpoint_device_before_reopen = persisted_checkpoint.source_training_state_device
        checkpoint_size_before_reopen = persisted_checkpoint.source_training_state_size_bytes
        checkpoint_digest_before_reopen = persisted_checkpoint.source_training_state_digest
        shutil.rmtree(persisted_checkpoint.private_directory, ignore_errors=True)
        retained_weight_values = retained_weight.tolist()
        if fresh_process_probe_path is not None:
            if fresh_process_source_directory is None:
                raise RuntimeError("fresh retention resume source directory is unavailable")
            resumed_evidence = _resume_checkpoint_in_fresh_process(
                probe_path=fresh_process_probe_path,
                source_directory=fresh_process_source_directory,
                checkpoint_path=checkpoint_path,
                device=device,
                expected_optimizer_state_digest=optimizer_state_digest_before_persist,
                expected_optimizer_state_devices=(optimizer_state_devices_before_persist),
                expected_retained_weight=retained_weight_values,
                checkpoint_authentication_key=authentication_key,
                checkpoint_authentication_identity=authentication_identity,
                checkpoint_generation=commit.generation,
                checkpoint_manifest_digest=commit.manifest_digest,
            )
        else:
            resumed_evidence = _restore_and_advance_checkpoint(
                trainer,
                torch_module,
                checkpoint_path=checkpoint_path,
                device=device,
                expected_optimizer_state_digest=optimizer_state_digest_before_persist,
                expected_optimizer_state_devices=(optimizer_state_devices_before_persist),
                expected_retained_weight=retained_weight_values,
                expected_optimizer_state=expected_optimizer_state,
                checkpoint_authentication_key=authentication_key,
                checkpoint_authentication_identity=authentication_identity,
                checkpoint_generation=commit.generation,
                checkpoint_manifest_digest=commit.manifest_digest,
            )
        if (
            resumed_evidence["checkpoint_sha256"] != checkpoint_digest_before_reopen
            or resumed_evidence["checkpoint_size_bytes"] != checkpoint_size_before_reopen
            or resumed_evidence["checkpoint_source_device"] != checkpoint_device_before_reopen
        ):
            raise RuntimeError("retention checkpoint changed across the fresh resume")
        if checkpoint_size_before_reopen <= 0:
            raise RuntimeError("retention checkpoint is empty")

        evidence = {
            "optimizer_state_digest_before_persist": optimizer_state_digest_before_persist,
            "optimizer_state_devices_before_persist": (optimizer_state_devices_before_persist),
            **resumed_evidence,
        }
        if storage_parent is None:
            for key in (
                "checkpoint_inode_after_reopen",
                "checkpoint_resume_process_pid",
                "checkpoint_resume_parent_process_pid",
            ):
                evidence.pop(key, None)
        if storage_parent is not None:
            storage_root = storage_parent.resolve(strict=True)
            storage_root_metadata = storage_root.stat()
            root_device = storage_root_metadata.st_dev
            checkpoint_device = resumed_evidence["checkpoint_source_device"]
            if root_device <= 0 or checkpoint_device != root_device:
                raise RuntimeError("retention checkpoint escaped its persistent filesystem")
            storage_evidence = {
                "checkpoint_storage_scope": storage_scope,
                "checkpoint_storage_root": str(storage_root),
                "checkpoint_storage_root_device": root_device,
                "checkpoint_storage_checkpoint_device": checkpoint_device,
                "checkpoint_storage_root_inode": storage_root_metadata.st_ino,
                "checkpoint_inode_before_reopen": checkpoint_inode_before_reopen,
                "checkpoint_inode_after_reopen": resumed_evidence.get(
                    "checkpoint_inode_after_reopen"
                ),
                "checkpoint_persist_process_pid": os.getpid(),
                "checkpoint_resume_process_pid": resumed_evidence.get(
                    "checkpoint_resume_process_pid"
                ),
                "checkpoint_resume_parent_process_pid": resumed_evidence.get(
                    "checkpoint_resume_parent_process_pid"
                ),
                "checkpoint_reopened_after_fsync": True,
                "checkpoint_source_device": resumed_evidence.get("checkpoint_source_device"),
                "checkpoint_authentication_revision": resumed_evidence.get(
                    "checkpoint_authentication_revision"
                ),
                "checkpoint_authentication_mechanism_digest": resumed_evidence.get(
                    "checkpoint_authentication_mechanism_digest"
                ),
                "checkpoint_generation": resumed_evidence.get("checkpoint_generation"),
                "checkpoint_authenticated_private_resume": resumed_evidence.get(
                    "checkpoint_authenticated_private_resume"
                ),
            }
            if (
                storage_evidence["checkpoint_inode_before_reopen"]
                != storage_evidence["checkpoint_inode_after_reopen"]
                or storage_evidence["checkpoint_persist_process_pid"]
                != storage_evidence["checkpoint_resume_parent_process_pid"]
                or storage_evidence["checkpoint_persist_process_pid"]
                == storage_evidence["checkpoint_resume_process_pid"]
            ):
                raise RuntimeError("retention checkpoint was not reopened by a fresh child process")
            storage_evidence["checkpoint_storage_evidence_digest"] = (
                "sha256:" + hashlib.sha256(canonical_json(storage_evidence)).hexdigest()
            )
            evidence.update(storage_evidence)
        return evidence


def _legacy_probe(
    source_root: Path,
    head_commit: str,
    source_archive: Path,
) -> dict[str, Any]:
    import torch

    source_directory, manifest_path, trainer_path = _source_layout(
        source_root,
        flat_bundle=False,
    )
    probe_path = Path(__file__).resolve()
    environment_path = source_directory / "repository_repair_env.py"
    for source in (
        probe_path,
        manifest_path,
        trainer_path,
        environment_path,
        source_archive,
    ):
        if not source.is_file():
            raise RuntimeError(f"retention probe source is unavailable: {source.name}")
    manifest = _load_json_object(manifest_path, "retention probe manifest")
    source_sha256 = _verify_sources(source_directory, manifest)
    trainer = _import_trainer(source_directory, flat_bundle=False)
    return {
        "status": "passed",
        "test_id": TEST_ID,
        "profile_id": manifest["profile_id"],
        "head_commit": head_commit,
        "source_archive_digest": sha256_file(source_archive),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "probe_sha256": sha256_file(probe_path),
        "trainer_sha256": sha256_file(trainer_path),
        "environment_sha256": sha256_file(environment_path),
        "source_sha256": source_sha256,
        **_checkpoint_roundtrip(trainer, torch, device="cpu"),
    }


def run_flat_bundle_probe(
    flat_bundle_root: Path,
    *,
    head_commit: str,
    workload_bundle_digest: str,
    workload_bundle_size_bytes: int,
    workload_bundle_path: str,
    bundle_stage_receipt_digest: str,
    bootstrap_source_digest: str,
    volume_readiness_receipt: Path,
    dependency_lock_digest: str,
    dependency_quarantine_evidence: Path,
    dependency_quarantine_evidence_digest: str,
    dependency_private_tree_digest: str,
    code_materialization_evidence: Path,
    code_materialization_evidence_digest: str,
    code_private_tree_digest: str,
    network_volume_id: str,
    data_center_id: str,
    volume_size_gb: int,
) -> dict[str, Any]:
    """Run and bind the integrated H100 checkpoint probe to one exact handoff."""

    if not HEAD_COMMIT.fullmatch(head_commit):
        raise RuntimeError("retention probe HEAD commit is invalid")
    for value, name in (
        (workload_bundle_digest, "workload bundle digest"),
        (bundle_stage_receipt_digest, "bundle stage receipt digest"),
        (bootstrap_source_digest, "bootstrap source digest"),
        (
            dependency_quarantine_evidence_digest,
            "dependency quarantine evidence digest",
        ),
        (dependency_lock_digest, "dependency lock digest"),
        (dependency_private_tree_digest, "dependency private tree digest"),
        (
            code_materialization_evidence_digest,
            "code materialization evidence digest",
        ),
        (code_private_tree_digest, "code private tree digest"),
    ):
        if not isinstance(value, str) or not TAGGED_SHA256.fullmatch(value):
            raise RuntimeError(f"retention probe {name} is invalid")
    if (
        type(workload_bundle_size_bytes) is not int
        or not 0 < workload_bundle_size_bytes <= 2 * 1024 * 1024
    ):
        raise RuntimeError("retention probe workload bundle size is invalid")
    if (
        not isinstance(workload_bundle_path, str)
        or workload_bundle_digest[7:] not in workload_bundle_path
        or not workload_bundle_path.endswith(".tar.xz")
    ):
        raise RuntimeError("retention probe workload bundle path is invalid")
    if (
        not isinstance(network_volume_id, str)
        or not network_volume_id
        or not isinstance(data_center_id, str)
        or not data_center_id
        or type(volume_size_gb) is not int
        or volume_size_gb <= 0
    ):
        raise RuntimeError("retention probe volume plan is invalid")

    flat_bundle_root = flat_bundle_root.resolve()
    source_directory, manifest_path, trainer_path = _source_layout(
        flat_bundle_root,
        flat_bundle=True,
    )
    probe_path = source_directory / "retention_checkpoint_probe.py"
    environment_path = source_directory / "repository_repair_env.py"
    for source in (probe_path, manifest_path, trainer_path, environment_path):
        if not source.is_file():
            raise RuntimeError(f"retention probe source is unavailable: {source.name}")

    source_text = str(source_directory)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    gate = importlib.import_module("larger_model_gate")
    manifest = gate.load_manifest(manifest_path)
    if dependency_lock_digest != manifest["materialization"]["dependency_lock"]["digest"]:
        raise RuntimeError("retention probe dependency lock digest is invalid")
    source_verification = gate.verify_source_contract(manifest, source_directory)
    dependency_evidence = _load_json_object(
        dependency_quarantine_evidence,
        "dependency quarantine evidence",
    )
    code_evidence = _load_json_object(
        code_materialization_evidence,
        "code materialization evidence",
    )
    verified_dependencies = gate.verify_dependency_quarantine_evidence(
        manifest,
        dependency_evidence,
        workload_bundle_path=workload_bundle_path,
    )
    verified_code = gate.verify_code_materialization_evidence(
        manifest,
        code_evidence,
        workload_bundle_digest=workload_bundle_digest,
        workload_bundle_size_bytes=workload_bundle_size_bytes,
        workload_bundle_path=workload_bundle_path,
    )
    for observed, expected, name in (
        (
            verified_dependencies["evidence_digest"],
            dependency_quarantine_evidence_digest,
            "dependency quarantine evidence digest",
        ),
        (
            verified_dependencies["private_tree_digest"],
            dependency_private_tree_digest,
            "dependency private tree digest",
        ),
        (
            verified_code["evidence_digest"],
            code_materialization_evidence_digest,
            "code materialization evidence digest",
        ),
        (
            verified_code["private_tree_digest"],
            code_private_tree_digest,
            "code private tree digest",
        ),
    ):
        if observed != expected:
            raise RuntimeError(f"retention probe {name} does not match its evidence")
    torch_module = importlib.import_module("torch")
    source_sha256 = {
        name: f"sha256:{digest}" for name, digest in source_verification["files"].items()
    }
    hardware = gate.require_cuda_hardware(manifest, torch_module)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if cuda_version != gate.expected_cuda_version(manifest):
        raise RuntimeError("retention probe CUDA build does not match the profile")
    receipt = _load_json_object(
        volume_readiness_receipt,
        "volume readiness receipt",
    )
    receipt_verification = gate.verify_volume_readiness_receipt(
        manifest,
        receipt,
        {
            "id": network_volume_id,
            "dataCenterId": data_center_id,
            "size": volume_size_gb,
        },
    )
    for key, expected in {
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_evidence_digest": (dependency_quarantine_evidence_digest),
        "dependency_private_tree_digest": dependency_private_tree_digest,
        "code_materialization_evidence_digest": code_materialization_evidence_digest,
        "code_private_tree_digest": code_private_tree_digest,
    }.items():
        if receipt_verification.get(key) != expected:
            raise RuntimeError(f"retention probe volume receipt {key} is inconsistent")
    model_cache = Path(manifest["artifact_readiness"]["cache_directory"]).resolve(strict=True)
    persistent_work_directory = flat_bundle_root.resolve(strict=True)
    if (
        model_cache.stat().st_dev <= 0
        or persistent_work_directory.stat().st_dev != model_cache.stat().st_dev
    ):
        raise RuntimeError("retention checkpoint work directory is not on the model volume")
    trainer = _import_trainer(source_directory, flat_bundle=True)
    evidence: dict[str, Any] = {
        "schema_version": 3,
        "evidence_revision": EVIDENCE_REVISION,
        "status": "passed",
        "test_id": TEST_ID,
        "profile_id": manifest["profile_id"],
        "head_commit": head_commit,
        "source_contract_digest": source_verification["source_contract_digest"],
        "workload_bundle_digest": workload_bundle_digest,
        "workload_bundle_size_bytes": workload_bundle_size_bytes,
        "workload_bundle_path": workload_bundle_path,
        "bundle_stage_receipt_digest": bundle_stage_receipt_digest,
        "bootstrap_source_digest": bootstrap_source_digest,
        "volume_readiness_receipt_digest": receipt_verification["receipt_digest"],
        "dependency_lock_digest": dependency_lock_digest,
        "dependency_quarantine_revision": manifest["materialization"]["dependency_lock"][
            "revision"
        ],
        "dependency_quarantine_evidence_digest": (dependency_quarantine_evidence_digest),
        "dependency_private_tree_digest": dependency_private_tree_digest,
        "code_materialization_revision": manifest["materialization"]["code"]["revision"],
        "code_materialization_evidence_digest": code_materialization_evidence_digest,
        "code_private_tree_digest": code_private_tree_digest,
        "network_volume_id": network_volume_id,
        "network_volume_data_center_id": data_center_id,
        "network_volume_size_gb": volume_size_gb,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(cuda_version),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "gpu_name": hardware.gpu_name,
        "gpu_total_memory_bytes": hardware.total_memory_bytes,
        "bf16_supported": hardware.bf16_supported,
        "probe_sha256": sha256_file(probe_path),
        "trainer_sha256": sha256_file(trainer_path),
        "environment_sha256": sha256_file(environment_path),
        "source_sha256": source_sha256,
        **_checkpoint_roundtrip(
            trainer,
            torch_module,
            device="cuda:0",
            storage_parent=persistent_work_directory,
            storage_scope="runpod-network-volume",
            fresh_process_probe_path=probe_path,
            fresh_process_source_directory=source_directory,
            checkpoint_authentication_identity={
                "run_identity": ("retention-readiness-" + workload_bundle_digest[7:39]),
                "profile_id": manifest["profile_id"],
                "authorization_digest": bundle_stage_receipt_digest,
                "source_head_commit": head_commit,
                "workload_bundle_digest": workload_bundle_digest,
                "workload_revision": trainer.WORKLOAD_REVISION,
                "model_revision": manifest["model"]["revision"],
                "objective_id": trainer.OBJECTIVE_ID,
            },
        ),
    }
    evidence["evidence_digest"] = evidence_digest(evidence)
    return evidence


def run_probe(
    source_root: Path,
    head_commit: str,
    source_archive: Path,
) -> dict[str, Any]:
    """Retain the historical CPU-stage probe API for predecessor receipts."""

    if not HEAD_COMMIT.fullmatch(head_commit):
        raise RuntimeError("retention probe HEAD commit is invalid")
    return _legacy_probe(source_root, head_commit, source_archive)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-root", type=Path)
    mode.add_argument("--flat-bundle-root", type=Path)
    mode.add_argument("--resume-roundtrip-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--head-commit")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--workload-bundle-digest")
    parser.add_argument("--workload-bundle-size-bytes", type=_positive_int)
    parser.add_argument("--workload-bundle-path")
    parser.add_argument("--bundle-stage-receipt-digest")
    parser.add_argument("--bootstrap-source-digest")
    parser.add_argument("--volume-readiness-receipt", type=Path)
    parser.add_argument("--dependency-quarantine-evidence", type=Path)
    parser.add_argument("--dependency-lock-digest")
    parser.add_argument("--dependency-quarantine-evidence-digest")
    parser.add_argument("--dependency-private-tree-digest")
    parser.add_argument("--code-materialization-evidence", type=Path)
    parser.add_argument("--code-materialization-evidence-digest")
    parser.add_argument("--code-private-tree-digest")
    parser.add_argument("--network-volume-id")
    parser.add_argument("--data-center-id")
    parser.add_argument("--volume-size-gb", type=_positive_int)
    parser.add_argument("--resume-source-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-device", help=argparse.SUPPRESS)
    parser.add_argument("--expected-optimizer-state-digest", help=argparse.SUPPRESS)
    parser.add_argument("--expected-optimizer-state-devices-json", help=argparse.SUPPRESS)
    parser.add_argument("--expected-retained-weight-json", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.resume_roundtrip_checkpoint is not None:
        checkpoint_key_hex = os.environ.pop(
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_KEY",
            "",
        )
        checkpoint_identity_json = os.environ.pop(
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_IDENTITY",
            "",
        )
        checkpoint_generation_text = os.environ.pop(
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_GENERATION",
            "",
        )
        checkpoint_manifest_digest = os.environ.pop(
            "EQUINOX_RETENTION_PROBE_CHECKPOINT_MANIFEST_SHA256",
            "",
        )
        required_resume = {
            "resume_source_directory": arguments.resume_source_directory,
            "resume_device": arguments.resume_device,
            "expected_optimizer_state_digest": (arguments.expected_optimizer_state_digest),
            "expected_optimizer_state_devices_json": (
                arguments.expected_optimizer_state_devices_json
            ),
            "expected_retained_weight_json": arguments.expected_retained_weight_json,
            "checkpoint_key_hex": checkpoint_key_hex,
            "checkpoint_identity_json": checkpoint_identity_json,
            "checkpoint_generation": checkpoint_generation_text,
            "checkpoint_manifest_digest": checkpoint_manifest_digest,
        }
        missing = sorted(name for name, value in required_resume.items() if value is None)
        if missing:
            parser.error("resume-roundtrip mode is missing: " + ", ".join(missing))
        try:
            expected_devices = json.loads(arguments.expected_optimizer_state_devices_json)
            expected_weight = json.loads(arguments.expected_retained_weight_json)
            checkpoint_identity = json.loads(checkpoint_identity_json)
        except json.JSONDecodeError as error:
            parser.error(f"resume-roundtrip evidence is invalid JSON: {error}")
        if (
            not isinstance(expected_devices, dict)
            or not isinstance(expected_weight, list)
            or not expected_weight
            or not isinstance(arguments.expected_optimizer_state_digest, str)
            or not TAGGED_SHA256.fullmatch(arguments.expected_optimizer_state_digest)
            or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_key_hex)
            or not isinstance(checkpoint_identity, dict)
            or not checkpoint_identity
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in checkpoint_identity.items()
            )
            or not checkpoint_generation_text.isascii()
            or not checkpoint_generation_text.isdigit()
            or int(checkpoint_generation_text) < 1
            or not TAGGED_SHA256.fullmatch(checkpoint_manifest_digest)
        ):
            parser.error("resume-roundtrip expected evidence is invalid")
        torch_module = importlib.import_module("torch")
        trainer = _import_trainer(
            arguments.resume_source_directory.resolve(strict=True),
            flat_bundle=True,
        )
        output = _restore_and_advance_checkpoint(
            trainer,
            torch_module,
            checkpoint_path=arguments.resume_roundtrip_checkpoint,
            device=arguments.resume_device,
            expected_optimizer_state_digest=(arguments.expected_optimizer_state_digest),
            expected_optimizer_state_devices=expected_devices,
            expected_retained_weight=expected_weight,
            checkpoint_authentication_key=bytes.fromhex(checkpoint_key_hex),
            checkpoint_authentication_identity=checkpoint_identity,
            checkpoint_generation=int(checkpoint_generation_text),
            checkpoint_manifest_digest=checkpoint_manifest_digest,
        )
    elif arguments.source_root is not None:
        if arguments.head_commit is None:
            parser.error("--source-root requires --head-commit")
        if arguments.source_archive is None:
            parser.error("--source-root requires --source-archive")
        output = run_probe(
            arguments.source_root,
            arguments.head_commit,
            arguments.source_archive,
        )
    else:
        if arguments.head_commit is None:
            parser.error("--flat-bundle-root requires --head-commit")
        required = {
            "workload_bundle_digest": arguments.workload_bundle_digest,
            "workload_bundle_size_bytes": arguments.workload_bundle_size_bytes,
            "workload_bundle_path": arguments.workload_bundle_path,
            "bundle_stage_receipt_digest": arguments.bundle_stage_receipt_digest,
            "bootstrap_source_digest": arguments.bootstrap_source_digest,
            "volume_readiness_receipt": arguments.volume_readiness_receipt,
            "dependency_quarantine_evidence": (arguments.dependency_quarantine_evidence),
            "dependency_lock_digest": arguments.dependency_lock_digest,
            "dependency_quarantine_evidence_digest": (
                arguments.dependency_quarantine_evidence_digest
            ),
            "dependency_private_tree_digest": (arguments.dependency_private_tree_digest),
            "code_materialization_evidence": (arguments.code_materialization_evidence),
            "code_materialization_evidence_digest": (
                arguments.code_materialization_evidence_digest
            ),
            "code_private_tree_digest": arguments.code_private_tree_digest,
            "network_volume_id": arguments.network_volume_id,
            "data_center_id": arguments.data_center_id,
            "volume_size_gb": arguments.volume_size_gb,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            parser.error("flat-bundle mode is missing: " + ", ".join(missing))
        output = run_flat_bundle_probe(
            arguments.flat_bundle_root,
            head_commit=arguments.head_commit,
            **required,
        )
    sys.stdout.buffer.write(canonical_json(output) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
