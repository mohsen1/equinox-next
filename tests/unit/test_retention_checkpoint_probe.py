from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runpod import retention_checkpoint_probe as probe
from research.runpod.retention_checkpoint_probe import (
    optimizer_state_digest,
    optimizer_state_tensor_devices,
    optimizer_states_equal,
)


@dataclass
class FakeTensor:
    raw_bytes: list[int]
    dtype: str = "torch.float32"
    shape: tuple[int, ...] = (1,)
    device: str = "cpu"

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def contiguous(self) -> FakeTensor:
        return self

    def view(self, _dtype: object) -> FakeTensor:
        return FakeTensor(
            self.raw_bytes,
            dtype="torch.uint8",
            shape=(len(self.raw_bytes),),
            device=self.device,
        )

    def reshape(self, *_shape: int) -> FakeTensor:
        return self

    def tolist(self) -> list[int]:
        return list(self.raw_bytes)


class FakeTorch:
    uint8 = object()

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, FakeTensor)

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        return left.raw_bytes == right.raw_bytes


def optimizer_state() -> dict[str, object]:
    return {
        "state": {
            0: {
                "step": FakeTensor([1, 0, 0, 0]),
                "exp_avg": FakeTensor([10, 20, 30, 40]),
                "exp_avg_sq": FakeTensor([50, 60, 70, 80]),
            }
        },
        "param_groups": [
            {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
                "params": [0],
            }
        ],
    }


def test_optimizer_state_digest_and_deep_equality_cover_moments_and_groups() -> None:
    expected = optimizer_state()
    restored = copy.deepcopy(expected)

    assert optimizer_states_equal(expected, restored, FakeTorch)
    assert optimizer_state_digest(expected, FakeTorch) == optimizer_state_digest(
        restored,
        FakeTorch,
    )

    restored["state"][0]["exp_avg"] = FakeTensor([10, 20, 30, 41])
    assert not optimizer_states_equal(expected, restored, FakeTorch)
    assert optimizer_state_digest(expected, FakeTorch) != optimizer_state_digest(
        restored,
        FakeTorch,
    )

    restored = copy.deepcopy(expected)
    restored["param_groups"][0]["lr"] = 0.02
    assert not optimizer_states_equal(expected, restored, FakeTorch)
    assert optimizer_state_digest(expected, FakeTorch) != optimizer_state_digest(
        restored,
        FakeTorch,
    )


def test_optimizer_state_device_evidence_covers_step_and_moments() -> None:
    state = optimizer_state()
    state["state"][0]["exp_avg"].device = "cuda:0"
    state["state"][0]["exp_avg_sq"].device = "cuda:0"

    assert optimizer_state_tensor_devices(state, FakeTorch) == {
        "exp_avg": ["cuda:0"],
        "exp_avg_sq": ["cuda:0"],
        "step": ["cpu"],
    }


def test_retention_resume_uses_a_fresh_interpreter_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_result = {
        "checkpoint_inode_after_reopen": 91,
        "checkpoint_resume_process_pid": 201,
        "checkpoint_resume_parent_process_pid": 200,
    }
    observed_command: list[str] = []
    observed_environment: dict[str, str] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed_command.extend(command)
        environment = kwargs.pop("env")
        assert isinstance(environment, dict)
        observed_environment.update(environment)
        assert kwargs == {"check": False, "capture_output": True, "timeout": 120}
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(expected_result).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    result = probe._resume_checkpoint_in_fresh_process(
        probe_path=tmp_path / "retention_checkpoint_probe.py",
        source_directory=tmp_path,
        checkpoint_path=tmp_path / "training-state.pt",
        device="cuda:0",
        expected_optimizer_state_digest="sha256:" + "a" * 64,
        expected_optimizer_state_devices={
            "exp_avg": ["cuda:0"],
            "exp_avg_sq": ["cuda:0"],
            "step": ["cpu"],
        },
        expected_retained_weight=[0.99],
        checkpoint_authentication_key=b"k" * 32,
        checkpoint_authentication_identity={
            "run_identity": "retention-probe-unit-test",
            "profile_id": "test@1",
        },
        checkpoint_generation=3,
        checkpoint_manifest_digest="sha256:" + "b" * 64,
    )

    assert result == expected_result
    assert observed_command[0] == probe.sys.executable
    assert "--resume-roundtrip-checkpoint" in observed_command
    assert "--resume-device" in observed_command
    assert observed_environment["EQUINOX_RETENTION_PROBE_CHECKPOINT_KEY"] == (
        b"k" * 32
    ).hex()
    assert (
        observed_environment["EQUINOX_RETENTION_PROBE_CHECKPOINT_GENERATION"]
        == "3"
    )
    assert observed_environment[
        "EQUINOX_RETENTION_PROBE_CHECKPOINT_MANIFEST_SHA256"
    ] == ("sha256:" + "b" * 64)
