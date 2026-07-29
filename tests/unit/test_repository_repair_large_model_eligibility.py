from __future__ import annotations

import contextlib
import copy
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runpod import larger_model_gate as gate
from research.runpod import repository_repair_large_model_eligibility as eligibility


def profile() -> dict[str, object]:
    return copy.deepcopy(gate.load_manifest())


def baseline(
    *,
    checkpoints_by_level: tuple[int, int, int, int] = (7, 7, 7, 7),
) -> list[eligibility.BaselineOutcome]:
    outcomes: list[eligibility.BaselineOutcome] = []
    for level, checkpoint_count in enumerate(checkpoints_by_level):
        for index in range(8):
            outcomes.append(
                eligibility.BaselineOutcome(
                    level=level,
                    checkpoint_reached=index < checkpoint_count,
                    solved=index == 0,
                )
            )
    return outcomes


def collection(
    *,
    checkpointed: bool = True,
    informative: bool = False,
    solved: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=object() if checkpointed else None,
        exclusion_reason=None if checkpointed else "PREFIX_CHECKPOINT_NOT_REACHED",
        prefix=SimpleNamespace(terminal=False),
        informative=informative,
        solved_siblings=solved if checkpointed else 0,
        siblings=[object()] * (4 if checkpointed else 0),
    )


def passing_evidence() -> eligibility.ScreenEvidence:
    manifest = gate.load_manifest()
    return eligibility.ScreenEvidence(
        baseline_outcomes=baseline(),
        branch_collections=[
            collection(informative=index < 2)
            if index < 6
            else collection(checkpointed=False, solved=0)
            for index in range(8)
        ],
        total_actions=1_000,
        accepted_actions=995,
        schema_valid_actions=1_000,
        optimizer_step_calls=1,
        restored_parameter_tensors=64,
        optimizer_state_restored=True,
        capacity_smoke_completed=True,
        gradient_checkpointing_enabled=True,
        pinned_snapshot_ready=True,
        offline_mode_active=True,
        gpu_name=manifest["hardware"]["gpu_id"],
        gpu_total_memory_bytes=manifest["hardware"]["minimum_gpu_memory_gb"] * 1024**3,
        peak_reserved_vram_bytes=int(manifest["hardware"]["minimum_gpu_memory_gb"] * 1024**3 * 0.8),
        bf16_supported=True,
        base_parameter_count=manifest["model"]["parameter_count"],
        base_bf16_parameter_count=manifest["model"]["parameter_count"],
        trainable_parameter_count=40_000_000,
        model_parameter_count_with_adapter=manifest["model"]["parameter_count"] + 40_000_000,
    )


def test_passing_screen_result_matches_the_pilot_authorization_contract() -> None:
    manifest = gate.load_manifest()
    result = eligibility.build_screen_result(
        passing_evidence(),
        manifest,
        completed_at="2026-07-29T12:00:00Z",
    )

    assert result["screen_completed"] is True
    assert result["eligible"] is True
    assert result["completed_baseline_examples"] == 32
    assert result["per_level_checkpoint_rates"] == dict.fromkeys(
        eligibility.SCREEN_LEVELS,
        0.875,
    )
    assert result["branch_groups"] == 8
    assert result["informative_groups"] == 2
    assert result["solved_siblings"] == 6
    assert result["failed_siblings"] == 18
    assert result["policy_mutation_detected"] is False
    assert result["persistent_policy_updates"] == 0
    assert result["test_split_accessed"] is False
    assert result["training_microbatch_size"] == 1
    assert result["maximum_input_tokens"] == 1_536
    assert result["gate_results"] == dict.fromkeys(gate.REQUIRED_GATE_RESULTS, True)
    assert result["digest"] == gate.result_digest(result)
    authorization = gate.verify_pilot_authorization(
        manifest,
        result,
        {
            "provider": "runpod",
            "provider_handle": "runpod://pods/screen-123",
            "profile_id": manifest["profile_id"],
            "model_id": manifest["model"]["id"],
            "model_revision": manifest["model"]["revision"],
            "screen_workload_revision": manifest["screen"]["workload_revision"],
            "gpu_id": manifest["hardware"]["gpu_id"],
            "result_digest": gate.result_digest(result),
            "teardown_confirmed": True,
            "completed_at": "2026-07-29T12:00:00Z",
        },
        now=datetime(2026, 7, 29, 13, 0, tzinfo=UTC),
    )
    assert authorization["screen_result_digest"] == result["digest"]


def test_every_level_must_pass_even_when_overall_checkpoint_rate_passes() -> None:
    evidence = passing_evidence()
    evidence.baseline_outcomes = baseline(checkpoints_by_level=(5, 8, 8, 8))

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["baseline_checkpoint_rate"] == 0.90625
    assert result["per_level_checkpoint_rates"]["0"] == 0.625
    assert result["gate_results"]["baseline_checkpoint_rate"] is False
    assert result["eligible"] is False
    assert result["screen_completed"] is True


def test_accepted_action_gate_rejects_repeated_rejected_action_loops() -> None:
    evidence = passing_evidence()
    evidence.repeated_rejected_loop_count = 1

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["action_protocol_validity"] == 0.995
    assert result["gate_results"]["action_protocol_validity"] is False
    assert result["ineligible_reasons"] == ["action_protocol_validity"]


def test_baseline_stops_only_after_twelve_when_a_level_gate_is_impossible() -> None:
    manifest = gate.load_manifest()
    evidence = eligibility.ScreenEvidence(
        baseline_outcomes=[
            *[
                eligibility.BaselineOutcome(
                    level=0,
                    checkpoint_reached=index < 5,
                    solved=index == 0,
                )
                for index in range(8)
            ],
            *[
                eligibility.BaselineOutcome(
                    level=1,
                    checkpoint_reached=True,
                    solved=False,
                )
                for _ in range(4)
            ],
        ]
    )

    assert (
        eligibility.baseline_impossible(evidence, manifest)
        == "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )
    evidence.baseline_outcomes.pop()
    assert eligibility.baseline_impossible(evidence, manifest) is None


def test_incomplete_branch_collection_is_not_counted_as_a_checkpoint() -> None:
    evidence = passing_evidence()
    evidence.branch_collections[0].prefix.terminal = True

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["branch_checkpoint_rate"] == 0.625
    assert result["gate_results"]["branch_checkpoint_rate"] is False
    assert result["eligible"] is False


def test_snapshot_readiness_requires_four_exact_local_shards(tmp_path: Path) -> None:
    manifest = profile()
    manifest["artifact_readiness"]["cache_directory"] = str(tmp_path)
    manifest["model"]["safetensors_bytes"] = 10
    model_cache = "models--" + manifest["model"]["id"].replace("/", "--")
    snapshot = tmp_path / "hub" / model_cache / "snapshots" / manifest["model"]["revision"]
    snapshot.mkdir(parents=True)
    for index, size in enumerate((1, 2, 3, 4), start=1):
        (snapshot / f"model-{index:05d}-of-00004.safetensors").write_bytes(b"x" * size)
    shard_names = {f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)}
    for name in eligibility.REQUIRED_SNAPSHOT_FILES - {"model.safetensors.index.json"}:
        (snapshot / name).write_text("ready", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"layer.{index}": name for index, name in enumerate(sorted(shard_names))
                }
            }
        ),
        encoding="utf-8",
    )

    assert eligibility.verify_pinned_snapshot(manifest) == (True, str(snapshot))

    (snapshot / "model-00004-of-00004.safetensors").write_bytes(b"xxxxx")
    assert eligibility.verify_pinned_snapshot(manifest) == (False, None)


def test_offline_mode_requires_both_huggingface_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert eligibility.offline_mode_active() is False

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert eligibility.offline_mode_active() is True


def test_paid_screen_requires_exact_prewarmed_dependencies() -> None:
    manifest = gate.load_manifest()

    assert (
        eligibility.verify_prewarmed_dependencies(
            manifest,
            version_reader=lambda package: manifest["runtime"]["dependencies"][package],
        )
        == manifest["runtime"]["dependencies"]
    )

    with pytest.raises(RuntimeError, match="does not match"):
        eligibility.verify_prewarmed_dependencies(
            manifest,
            version_reader=lambda _package: "0.0.0",
        )


def test_adam_step_restores_parameters_and_optimizer_state() -> None:
    class FakeTensor:
        shape = (1,)

        def __init__(self, value: float) -> None:
            self.value = value
            self.requires_grad = True

        def detach(self) -> FakeTensor:
            return self

        def clone(self) -> FakeTensor:
            return FakeTensor(self.value)

        def copy_(self, other: FakeTensor) -> None:
            self.value = other.value

    class FakeAdamW:
        def __init__(self, parameter: FakeTensor) -> None:
            self.param_groups = [{"params": [parameter]}]
            self.step_count = 0

        def state_dict(self) -> dict[str, object]:
            return {"state": {"step": self.step_count}, "param_groups": [{"lr": 0.1}]}

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.step_count = int(state["state"]["step"])

        def step(self, closure: object = None) -> None:
            del closure
            self.step_count += 1
            self.param_groups[0]["params"][0].value += 0.25

    fake_torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=FakeAdamW),
        no_grad=contextlib.nullcontext,
        equal=lambda left, right: left.value == right.value,
    )
    evidence = eligibility.ScreenEvidence()
    parameter = FakeTensor(1.0)
    optimizer = FakeAdamW(parameter)

    eligibility.install_no_persistent_mutation(evidence, fake_torch)
    optimizer.step()

    assert parameter.value == 1.0
    assert optimizer.step_count == 0
    assert evidence.optimizer_step_calls == 1
    assert evidence.restored_parameter_tensors == 1
    assert evidence.policy_mutation_detected is False
    assert evidence.optimizer_state_restored is True


def test_runtime_configuration_is_bound_to_exact_model_and_screen_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = gate.load_manifest()
    runtime = SimpleNamespace(
        model_id=manifest["model"]["id"],
        model_revision=manifest["model"]["revision"],
        validation_examples=manifest["screen"]["validation_examples"],
        training_tasks_per_update=manifest["screen"]["training_tasks_per_update"],
        maximum_updates=manifest["screen"]["maximum_updates"],
        workload_attempt=1,
    )
    eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.model_revision = "floating-main"
    with pytest.raises(ValueError, match="model_revision"):
        eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.model_revision = manifest["model"]["revision"]
    monkeypatch.setenv("EQUINOX_ADAPTER_PATH", "/tmp/old-adapter")
    with pytest.raises(ValueError, match="must not load"):
        eligibility.validate_runtime_configuration(runtime, manifest)


def test_runner_adapter_path_is_disarmed_only_when_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "adapter"
    monkeypatch.setenv("EQUINOX_ADAPTER_PATH", str(empty_path))
    eligibility.disarm_empty_adapter_path()
    assert "EQUINOX_ADAPTER_PATH" not in eligibility.os.environ

    empty_path.mkdir()
    (empty_path / "checkpoint.json").write_text("state", encoding="utf-8")
    monkeypatch.setenv("EQUINOX_ADAPTER_PATH", str(empty_path))
    with pytest.raises(ValueError, match="pre-existing adapter state"):
        eligibility.disarm_empty_adapter_path()


def test_runtime_hooks_install_balanced_v32_k4_screen_and_isolate_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = gate.load_manifest()
    for name, value in {
        "EQUINOX_RL_MODEL_ID": manifest["model"]["id"],
        "EQUINOX_RL_VALIDATION_EXAMPLES": str(manifest["screen"]["validation_examples"]),
        "EQUINOX_RL_TRAINING_TASKS_PER_UPDATE": str(
            manifest["screen"]["training_tasks_per_update"]
        ),
        "EQUINOX_RL_MAX_UPDATES": "1",
        "EQUINOX_RL_TEST_EXAMPLES": "4",
        "EQUINOX_RL_TARGET_SECONDS": "2400",
        "EQUINOX_RL_MAX_FINAL_EVALUATION_RESERVE_SECONDS": "1200",
        "EQUINOX_RL_SEED": "73",
        "EQUINOX_WORKLOAD_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("EQUINOX_ADAPTER_PATH", raising=False)

    frozen_attributes = (
        "SUPPORTED_MODELS",
        "SYSTEM_PROMPT",
        "ENVIRONMENT_REVISION",
        "ACTION_PROTOCOL_REVISION",
        "BRANCH_WIDTH",
        "TRAINING_MICROBATCH_SIZE",
        "MAX_INPUT_TOKENS",
        "RepositoryRepairEnvironment",
        "branch_checkpoint_reached",
        "make_tasks",
        "fixed_retention_guard_example_count",
        "family_balanced_validation_tasks",
        "collect_greedy_trajectory",
        "collect_branch_group",
    )
    with monkeypatch.context() as restore:
        for name in frozen_attributes:
            value = getattr(eligibility.frozen, name)
            restore.setattr(
                eligibility.frozen,
                name,
                copy.copy(value) if name == "SUPPORTED_MODELS" else value,
            )
        restore.setattr(
            eligibility.frozen_environment,
            "BRANCH_WIDTH",
            eligibility.frozen_environment.BRANCH_WIDTH,
        )
        evidence = eligibility.ScreenEvidence()
        runtime = eligibility.prepare_runtime(evidence, manifest)
        count = eligibility.frozen.fixed_retention_guard_example_count(
            0,
            runtime.validation_examples,
        )
        tasks = eligibility.frozen.family_balanced_validation_tasks(0, count, 40_000)

        assert count == 32
        assert Counter(task.level for task in tasks) == {0: 8, 1: 8, 2: 8, 3: 8}
        assert eligibility.frozen.BRANCH_WIDTH == 4
        assert eligibility.frozen.TRAINING_MICROBATCH_SIZE == 1
        assert eligibility.frozen.MAX_INPUT_TOKENS == 1_536
        assert (
            eligibility.frozen.RepositoryRepairEnvironment.__mro__[1]
            is eligibility.revision32.RepositoryRepairEnvironment
        )
        assert (
            eligibility.frozen.branch_checkpoint_reached(
                SimpleNamespace(),
                SimpleNamespace(terminal=True),
            )
            is False
        )
        with pytest.raises(RuntimeError, match="test split"):
            eligibility.frozen.make_tasks(0, 1, 190_000, split="test")
        assert evidence.test_split_accessed is True
