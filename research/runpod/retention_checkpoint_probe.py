"""Exercise the transactional retention checkpoint with real CPU Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a tagged SHA-256 digest for one source or checkpoint file."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(
    source_root: Path,
    head_commit: str,
    source_archive: Path,
) -> dict[str, Any]:
    """Persist, restore, and advance a real AdamW retention transaction."""

    import torch

    from research.runpod import repository_repair_large_model_trainer as trainer

    source_root = source_root.resolve()
    probe_path = Path(__file__).resolve()
    manifest_path = source_root / "research/studies/larger-model-eligibility.json"
    trainer_path = source_root / "research/runpod/repository_repair_large_model_trainer.py"
    environment_path = source_root / "research/runpod/repository_repair_env.py"
    for source in (
        probe_path,
        manifest_path,
        trainer_path,
        environment_path,
        source_archive,
    ):
        if not source.is_file():
            raise RuntimeError(f"retention probe source is unavailable: {source.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_contract = manifest.get("source_contract", {}).get("files")
    if not isinstance(source_contract, dict) or not source_contract:
        raise RuntimeError("retention probe source contract is unavailable")
    source_sha256: dict[str, str] = {}
    for name, expected in source_contract.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise RuntimeError("retention probe source contract is malformed")
        source_path = source_root / "research/runpod" / name
        observed = sha256_file(source_path)
        if observed != f"sha256:{expected}":
            raise RuntimeError(f"retention probe source contract mismatch: {name}")
        source_sha256[name] = observed

    with tempfile.TemporaryDirectory(prefix="equinox-retention-roundtrip-") as directory:
        temporary_root = Path(directory)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
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

        parameter.square().sum().backward()
        optimizer.step()
        checkpoints_root = temporary_root / "checkpoints"
        latest = checkpoints_root / "latest.json"

        def save_adapter(target: str) -> None:
            Path(target, "adapter_config.json").write_text("{}", encoding="utf-8")
            Path(target, "adapter_model.safetensors").write_bytes(b"adapter")

        trainer.persist_checkpoint(
            checkpoints_root=str(checkpoints_root),
            latest_checkpoint_path=str(latest),
            checkpoint_name="update-0002",
            state={"retained_transaction": transaction},
            save_adapter=save_adapter,
            save_state=torch.save,
        )
        checkpoint_path = checkpoints_root / "update-0002" / "training-state.pt"
        resumed_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        restored = trainer.validate_retained_transaction_state(
            resumed_state["retained_transaction"],
            best_validation_update=2,
            retained_policy_update_count=1,
        )
        resumed_parameter = torch.nn.Parameter(torch.tensor([-9.0]))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=0.01)
        effective_count, pending, pending_policy, pending_groups, observation = (
            trainer.restore_retention_transaction(
                restored,
                restore_trainable_state=lambda state: resumed_parameter.data.copy_(state["weight"]),
                optimizer=resumed_optimizer,
            )
        )
        if (
            not torch.equal(resumed_parameter.detach(), retained_weight)
            or effective_count != 1
            or pending != []
            or pending_policy != 0
            or pending_groups != []
            or observation != {"exact_rate": 0.5}
        ):
            raise RuntimeError("retention checkpoint did not restore its exact transaction")
        restored_weight_before_resume_step = resumed_parameter.detach().tolist()

        resumed_parameter.square().sum().backward()
        resumed_optimizer.step()
        safely_retained = trainer.capture_retention_transaction(
            capture_trainable_state=lambda: {"weight": resumed_parameter.detach().clone()},
            optimizer=resumed_optimizer,
            effective_policy_update_count=2,
            update=3,
            retained_observation={"exact_rate": 0.75},
        )
        if safely_retained["effective_policy_update_count"] != 2 or safely_retained[
            "retained_observation"
        ] != {"exact_rate": 0.75}:
            raise RuntimeError("resumed AdamW state could not be retained safely")

        return {
            "status": "passed",
            "test_id": (
                "test_transaction_round_trips_real_optimizer_checkpoint_when_torch_is_available"
            ),
            "profile_id": manifest["profile_id"],
            "head_commit": head_commit,
            "source_archive_digest": sha256_file(source_archive),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "probe_sha256": sha256_file(probe_path),
            "trainer_sha256": sha256_file(trainer_path),
            "environment_sha256": sha256_file(environment_path),
            "source_sha256": source_sha256,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "restored_weight_before_resume_step": restored_weight_before_resume_step,
            "advanced_weight_after_resume_step": resumed_parameter.detach().tolist(),
            "effective_policy_update_count_before_resume_step": effective_count,
            "effective_policy_update_count_after_resume_step": safely_retained[
                "effective_policy_update_count"
            ],
            "retained_observation_after_resume_step": safely_retained["retained_observation"],
            "optimizer_state_entries_after_resume_step": len(resumed_optimizer.state),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--head-commit", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    arguments = parser.parse_args(argv)
    output = run_probe(
        arguments.source_root,
        arguments.head_commit,
        arguments.source_archive,
    )
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
