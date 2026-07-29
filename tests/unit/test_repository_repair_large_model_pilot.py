from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runpod import larger_model_gate as gate
from research.runpod import repository_repair_large_model_pilot as pilot


def test_authorization_digest_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EQUINOX_LARGER_MODEL_AUTHORIZATION_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="authorization digest"):
        pilot.require_authorization_digest()

    digest = "sha256:" + "a" * 64
    monkeypatch.setenv("EQUINOX_LARGER_MODEL_AUTHORIZATION_DIGEST", digest)
    assert pilot.require_authorization_digest() == digest


def test_offline_snapshot_requires_exact_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = gate.load_manifest()
    manifest["artifact_readiness"]["cache_directory"] = str(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="complete pinned 7B snapshot"):
        pilot.require_offline_snapshot(manifest)


def test_actual_cuda_profile_is_rejected_before_snapshot_hash() -> None:
    class WrongCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA L40"

        @staticmethod
        def get_device_properties(_index: int) -> object:
            return SimpleNamespace(total_memory=46_999_999_999)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    snapshot_hash_started = False

    def snapshot_verifier(_manifest: dict[str, object]) -> tuple[Path, str]:
        nonlocal snapshot_hash_started
        snapshot_hash_started = True
        return Path("/workspace/model"), "sha256:unreachable"

    with pytest.raises(gate.GateError, match="does not match"):
        pilot.require_pre_model_readiness(
            gate.load_manifest(),
            torch_module=SimpleNamespace(cuda=WrongCuda()),
            snapshot_verifier=snapshot_verifier,
        )
    assert snapshot_hash_started is False


def test_v32_pilot_constants_do_not_modify_frozen_sources() -> None:
    assert pilot.MODEL_REVISION == gate.MODEL_REVISION
    assert pilot.WORKLOAD_REVISION == "runpod-repository-repair-large-model-pilot@1"
    assert pilot.OBJECTIVE_ID == "verified-fix-coverage-retention-policy-gradient@15"
    assert Path(pilot.__file__).name == "repository_repair_large_model_pilot.py"
    assert os.path.basename(pilot.frozen.__file__) == "repository_repair_rl.py"


def test_runtime_configuration_pins_the_pilot_optimization_seed() -> None:
    manifest = gate.load_manifest()
    runtime = SimpleNamespace(
        model_id=manifest["model"]["id"],
        model_revision=manifest["model"]["revision"],
        target_runtime_seconds=manifest["pilot"]["target_runtime_seconds"],
        maximum_updates=manifest["pilot"]["maximum_updates"],
        validation_examples=manifest["pilot"]["validation_examples"],
        test_examples=manifest["pilot"]["test_examples"],
        training_tasks_per_update=manifest["pilot"]["training_tasks_per_update"],
        replay_tasks_per_level=manifest["pilot"]["replay_tasks_per_level"],
        mastery_windows=manifest["pilot"]["mastery_windows"],
        maximum_final_evaluation_reserve_seconds=manifest["pilot"][
            "maximum_final_evaluation_reserve_seconds"
        ],
        optimization_seed=manifest["pilot_limits"]["optimization_seed"],
        workload_attempt=1,
    )

    pilot.validate_runtime_configuration(runtime, manifest)
    runtime.optimization_seed += 1
    with pytest.raises(ValueError, match="optimization_seed"):
        pilot.validate_runtime_configuration(runtime, manifest)


def test_pilot_result_records_seed_and_source_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = gate.load_manifest()
    optimization_seed = manifest["pilot_limits"]["optimization_seed"]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(
        pilot.gate,
        "require_cuda_hardware",
        lambda _manifest, _torch: gate.CUDAHardware(
            gpu_name="NVIDIA L40",
            total_memory_bytes=48_000_000_000,
            bf16_supported=True,
        ),
    )
    monkeypatch.setattr(pilot, "RETAINED_CHECKPOINT_UPDATE", 3)
    monkeypatch.setattr(pilot, "RETAINED_POLICY_UPDATE_COUNT", 1)
    monkeypatch.setenv(
        "EQUINOX_LARGER_MODEL_EXPECTED_SNAPSHOT_DIGEST",
        gate.expected_snapshot_digest(manifest),
    )
    monkeypatch.setenv("EQUINOX_RUNPOD_NETWORK_VOLUME_ID", "network-volume-123")

    result = pilot.augment_result(
        {
            "seed": optimization_seed,
            "policy_update_count": 1,
            "best_validation": {"update": 3},
        },
        manifest=manifest,
        authorization_digest="sha256:" + "a" * 64,
        snapshot=Path("/workspace/model"),
    )

    assert result["optimization_seed"] == optimization_seed
    assert result["source_contract_digest"] == gate.expected_source_contract_digest(manifest)


def test_paid_dependency_setup_fails_closed_instead_of_running_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = gate.load_manifest()
    monkeypatch.setattr(
        pilot.importlib.metadata,
        "version",
        lambda _package: "unexpected",
    )
    monkeypatch.setattr(
        pilot.frozen,
        "ensure_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("network installer called")),
    )

    pilot.install_memory_profile(manifest)

    with pytest.raises(RuntimeError, match="manifest-pinned dependencies"):
        pilot.frozen.ensure_dependencies()


def test_final_evaluation_requires_a_policy_update_in_the_retained_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = gate.load_manifest()
    monkeypatch.setattr(pilot, "OBSERVED_POLICY_UPDATE_COUNT", 0)
    monkeypatch.setattr(pilot, "POLICY_UPDATE_COUNTS_BY_TRAINING_UPDATE", {})
    monkeypatch.setattr(pilot, "RETAINED_CHECKPOINT_UPDATE", 0)
    monkeypatch.setattr(pilot, "RETAINED_POLICY_UPDATE_COUNT", 0)

    pilot.observe_retained_update_evidence(
        "training",
        {"update": 2, "policy_update_count": 0},
    )
    pilot.observe_retained_update_evidence(
        "training",
        {"update": 3, "policy_update_count": 1},
    )
    pilot.observe_retained_update_evidence(
        "finalizing",
        {"best_validation": {"update": 2}},
    )

    with pytest.raises(RuntimeError, match="NO_RETAINED_POLICY_UPDATE"):
        pilot.require_retained_policy_update(manifest)

    pilot.observe_retained_update_evidence(
        "finalizing",
        {"best_validation": {"update": 3}},
    )
    assert pilot.require_retained_policy_update(manifest) == (3, 1)
