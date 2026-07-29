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


class FakeTokenizer:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.calls: list[dict[str, object]] = []
        self.pad_token = None

    def __call__(self, _prompt: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"input_ids": [list(range(self.token_count))]}


def profile() -> dict[str, object]:
    return copy.deepcopy(gate.load_manifest())


def test_prompt_tokenizer_returns_full_in_budget_encoding_without_truncation() -> None:
    tokenizer = FakeTokenizer(2_048)
    guarded = eligibility.fail_closed_prompt_tokenizer(tokenizer)

    encoded = guarded(
        "task-generic prompt",
        truncation=True,
        max_length=2_048,
    )
    guarded.pad_token = "eos"

    assert len(encoded["input_ids"][0]) == 2_048
    assert tokenizer.calls == [{"truncation": False, "max_length": 2_048}]
    assert tokenizer.pad_token == "eos"
    assert eligibility.fail_closed_prompt_tokenizer(guarded) is guarded


def test_prompt_tokenizer_fails_before_silent_right_truncation() -> None:
    tokenizer = FakeTokenizer(2_049)
    guarded = eligibility.fail_closed_prompt_tokenizer(tokenizer)

    with pytest.raises(RuntimeError, match="PROMPT_INPUT_TOKEN_LIMIT_EXCEEDED"):
        guarded(
            "task-generic prompt",
            truncation=True,
            max_length=2_048,
        )

    assert tokenizer.calls == [{"truncation": False, "max_length": 2_048}]


def test_capacity_smoke_covers_the_larger_prompt_envelope_and_continuation() -> None:
    manifest = profile()
    manifest["screen"]["maximum_input_tokens"] = 1_024
    manifest["pilot"]["maximum_input_tokens"] = 2_048

    assert eligibility.capacity_smoke_sequence_tokens(manifest) == 2_240


def baseline(
    *,
    checkpoint_count: int = 7,
    localized_count: int = 4,
) -> list[eligibility.BaselineOutcome]:
    return [
        eligibility.BaselineOutcome(
            level=0,
            checkpoint_reached=index < checkpoint_count,
            solved=index == 0,
            all_fault_sources_observed=index < localized_count,
        )
        for index in range(8)
    ]


def collection(
    *,
    checkpointed: bool = True,
    informative: bool = False,
    solved: int = 1,
    localized: int = 2,
) -> SimpleNamespace:
    fault = SimpleNamespace(path="fault.py")
    root_step = SimpleNamespace(
        accepted=True,
        tool="list",
        action={"tool": "list", "path": ""},
    )
    localized_step = SimpleNamespace(
        accepted=True,
        tool="read",
        action={"tool": "read", "path": fault.path},
    )
    siblings = [
        SimpleNamespace(steps=[root_step, *([localized_step] if index < localized else [])])
        for index in range(4 if checkpointed else 0)
    ]
    return SimpleNamespace(
        snapshot=object() if checkpointed else None,
        exclusion_reason=None if checkpointed else "PREFIX_CHECKPOINT_NOT_REACHED",
        task=SimpleNamespace(level=0, faults=[fault]),
        prefix=SimpleNamespace(terminal=False, steps=[root_step]),
        informative=informative,
        solved_siblings=solved if checkpointed else 0,
        siblings=siblings,
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
        capacity_smoke_completed_at=10.0,
        baseline_completed_at=60.0,
        gradient_checkpointing_enabled=True,
        pinned_snapshot_ready=True,
        pinned_snapshot_digest=gate.expected_snapshot_digest(manifest),
        offline_mode_active=True,
        gpu_name=manifest["hardware"]["gpu_id"],
        gpu_total_memory_bytes=manifest["hardware"]["minimum_cuda_memory_bytes"],
        peak_reserved_vram_bytes=int(manifest["hardware"]["minimum_cuda_memory_bytes"] * 0.8),
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
    assert result["completed_baseline_examples"] == 8
    assert result["minimum_completed_baseline_examples"] == 8
    assert result["screen_levels"] == [0]
    assert result["per_level_checkpoint_rates"] == {"0": 0.875}
    assert result["shared_prefix_checkpoint_strategy"] == "repository_root_observed@1"
    assert result["baseline_all_fault_sources_observed_examples"] == 4
    assert result["baseline_all_fault_sources_observed_rate"] == 0.5
    assert result["per_level_all_fault_sources_observed_rates"] == {"0": 0.5}
    assert result["branch_groups"] == 8
    assert result["all_fault_sources_observed_branch_groups"] == 6
    assert result["branch_all_fault_sources_observed_rate"] == 0.75
    assert result["all_fault_sources_observed_siblings"] == 12
    assert result["sibling_all_fault_sources_observed_rate"] == 0.5
    assert result["informative_groups"] == 2
    assert result["solved_siblings"] == 6
    assert result["failed_siblings"] == 18
    assert result["policy_mutation_detected"] is False
    assert result["persistent_policy_updates"] == 0
    assert result["test_split_accessed"] is False
    assert result["training_microbatch_size"] == 1
    assert result["maximum_input_tokens"] == 2_048
    assert result["capacity_smoke_sequence_tokens"] == 2_240
    assert result["optimization_seed"] == 137
    assert result["optimization_seed"] == manifest["screen_limits"]["optimization_seed"]
    assert result["source_contract_digest"] == gate.expected_source_contract_digest(manifest)
    assert result["environment_revision"] == manifest["interface"]["environment_revision"]
    assert result["action_protocol_revision"] == manifest["interface"]["action_protocol_revision"]
    assert (
        result["terminal_submission_contract"]
        == manifest["interface"]["terminal_submission_contract"]
    )
    assert result["action_counts_by_tool"] == {}
    assert result["accepted_action_counts_by_tool"] == {}
    assert result["rejected_action_counts_by_tool"] == {}
    assert result["rejection_reason_counts"] == {}
    assert result["repeated_rejected_pair_counts_by_tool"] == {}
    assert result["terminal_reason_counts"] == {}
    assert result["pinned_snapshot_digest"] == gate.expected_snapshot_digest(manifest)
    assert result["baseline_runtime_seconds"] == 50.0
    assert result["predicted_final_evaluation_seconds"] == 1_406.25
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
            "resource_profile": {
                "image": manifest["runtime"]["image"],
                "image_digest": manifest["runtime"]["image_digest"],
                "network_volume_id": "volume-123",
            },
        },
        now=datetime(2026, 7, 29, 13, 0, tzinfo=UTC),
    )
    assert authorization["screen_result_digest"] == result["digest"]
    assert authorization["pinned_snapshot_digest"] == result["pinned_snapshot_digest"]


def test_eighth_branch_group_completes_before_the_policy_update_path() -> None:
    evidence = passing_evidence()
    final_collection = evidence.branch_collections.pop()

    with pytest.raises(eligibility.EligibilityScreenComplete) as completed:
        eligibility.record_screen_branch_collection(
            evidence,
            gate.load_manifest(),
            final_collection,
        )

    assert len(evidence.branch_collections) == 8
    assert completed.value.result["screen_completed"] is True
    assert completed.value.result["eligible"] is True
    assert completed.value.result["persistent_policy_updates"] == 0


def test_branch_progress_contains_cumulative_observer_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = passing_evidence()
    final_collection = evidence.branch_collections.pop()
    evidence.runtime_configuration = SimpleNamespace(
        optimization_seed=137,
        validation_examples=8,
    )
    emitted: dict[str, object] = {}

    monkeypatch.setattr(
        eligibility.frozen,
        "serialize_branch_group",
        lambda _collection, *, update, optimizer_update: {
            "snapshot_id": f"screen-group-{update}",
            "update": update,
            "optimizer_update": optimizer_update,
        },
    )

    def capture_progress(
        phase: str,
        _message: str,
        _runtime_configuration: object,
        **values: object,
    ) -> None:
        emitted.update(values)
        emitted["phase"] = phase

    monkeypatch.setattr(eligibility.frozen, "emit_progress", capture_progress)

    with pytest.raises(eligibility.EligibilityScreenComplete):
        eligibility.record_screen_branch_collection(
            evidence,
            gate.load_manifest(),
            final_collection,
        )

    assert emitted["phase"] == "eligibility_branch_collection"
    assert emitted["preserve_context"] is True
    assert emitted["evaluation_completed"] == 8
    assert emitted["evaluation_total"] == 8
    assert emitted["evaluation_split"] == "train"
    assert emitted["branch_snapshots"] == [
        {
            "snapshot_id": f"screen-group-{index}",
            "update": index,
            "optimizer_update": None,
        }
        for index in range(1, 9)
    ]
    assert emitted["latest_branch_snapshot"] == emitted["branch_snapshots"][-1]
    assert emitted["policy_update_count"] == 0
    assert emitted["optimizer_update_count"] == 0


def test_deadline_excluded_collection_finishes_ineligible_without_test_access() -> None:
    evidence = passing_evidence()
    evidence.branch_collections = []
    deadline_collection = collection(checkpointed=False)
    deadline_collection.exclusion_reason = "TRAINING_DEADLINE_REACHED"

    with pytest.raises(eligibility.EligibilityScreenComplete) as completed:
        eligibility.record_screen_branch_collection(
            evidence,
            gate.load_manifest(),
            deadline_collection,
        )

    assert completed.value.result["eligible"] is False
    assert completed.value.result["test_split_accessed"] is False
    assert completed.value.result["early_stop_reason"] == "screen_collection_deadline"
    assert completed.value.result["branch_groups"] == 1


def test_screen_rejects_a_collected_probe_outside_level_zero() -> None:
    evidence = passing_evidence()
    evidence.branch_collections = []
    wrong_level = collection()
    wrong_level.task.level = 1

    with pytest.raises(RuntimeError, match="outside level 0"):
        eligibility.record_screen_branch_collection(
            evidence,
            gate.load_manifest(),
            wrong_level,
        )

    assert evidence.branch_collections == []


def test_starting_level_checkpoint_gate_requires_six_of_eight_examples() -> None:
    evidence = passing_evidence()
    evidence.baseline_outcomes = baseline(checkpoint_count=5)

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["baseline_checkpoint_rate"] == 0.625
    assert result["per_level_checkpoint_rates"]["0"] == 0.625
    assert result["gate_results"]["baseline_checkpoint_rate"] is False
    assert result["eligible"] is False
    assert result["screen_completed"] is True


def test_protocol_gate_separates_schema_validity_from_semantic_acceptance() -> None:
    evidence = passing_evidence()
    evidence.accepted_actions = 850

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["action_protocol_validity"] == 1.0
    assert result["schema_valid_action_rate"] == 1.0
    assert result["semantic_acceptance_rate"] == 0.85
    assert result["accepted_action_rate"] == 0.85
    assert result["gate_results"]["action_protocol_validity"] is True

    evidence.schema_valid_actions = 980
    result = eligibility.build_screen_result(evidence, gate.load_manifest())
    assert result["gate_results"]["action_protocol_validity"] is False
    assert result["ineligible_reasons"] == ["action_protocol_validity"]


def test_protocol_gate_keeps_repeated_semantic_rejection_loops_as_telemetry() -> None:
    evidence = passing_evidence()
    evidence.repeated_rejected_loop_count = 1

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["action_protocol_validity"] == 1.0
    assert result["semantic_acceptance_rate"] == 0.995
    assert result["repeated_rejected_loop_count"] == 1
    assert result["gate_results"]["action_protocol_validity"] is True
    assert result["eligible"] is True


def test_action_diagnostics_are_bounded_aggregate_counts_without_task_data() -> None:
    evidence = passing_evidence()
    secret_path = "src/private_fault.py"
    secret_content = "return secret_fix"
    invalid = SimpleNamespace(
        tool=None,
        action=None,
        accepted=False,
        observation="Invalid action. Return exactly one allowed JSON object.",
        terminal=False,
        terminal_reason=None,
    )
    rejected_list = SimpleNamespace(
        tool="list",
        action={"tool": "list", "path": secret_path},
        accepted=False,
        observation=json.dumps(
            {
                "error": "PATH_EVIDENCE_REQUIRED",
                "detail": secret_content,
            }
        ),
        terminal=False,
        terminal_reason=None,
    )
    unknown_structured = SimpleNamespace(
        tool="edit",
        action={
            "tool": "edit",
            "path": secret_path,
            "old": secret_content,
            "new": "fixed",
        },
        accepted=False,
        observation=json.dumps({"error": secret_content}),
        terminal=False,
        terminal_reason=None,
    )
    unknown_plain = SimpleNamespace(
        tool="search",
        action={"tool": "search", "query": secret_content},
        accepted=False,
        observation=secret_content,
        terminal=False,
        terminal_reason=None,
    )
    solved = SimpleNamespace(
        tool="finish",
        action={"tool": "finish"},
        accepted=True,
        observation="{}",
        terminal=True,
        terminal_reason="solved",
    )

    previous = None
    for result in (
        invalid,
        invalid,
        rejected_list,
        rejected_list,
        unknown_structured,
        unknown_plain,
        solved,
    ):
        eligibility._record_action_telemetry(evidence, result, previous)
        previous = result

    result = eligibility.build_screen_result(evidence, gate.load_manifest())
    telemetry = {
        key: result[key]
        for key in (
            "action_counts_by_tool",
            "accepted_action_counts_by_tool",
            "rejected_action_counts_by_tool",
            "rejection_reason_counts",
            "repeated_rejected_pair_counts_by_tool",
            "terminal_reason_counts",
        )
    }

    assert telemetry == {
        "action_counts_by_tool": {
            "edit": 1,
            "finish": 1,
            "invalid": 2,
            "list": 2,
            "search": 1,
        },
        "accepted_action_counts_by_tool": {"finish": 1},
        "rejected_action_counts_by_tool": {
            "edit": 1,
            "invalid": 2,
            "list": 2,
            "search": 1,
        },
        "rejection_reason_counts": {
            "INVALID_ACTION": 2,
            "OTHER_REJECTION": 1,
            "OTHER_STRUCTURED_ERROR": 1,
            "PATH_EVIDENCE_REQUIRED": 2,
        },
        "repeated_rejected_pair_counts_by_tool": {
            "invalid": 1,
            "list": 1,
        },
        "terminal_reason_counts": {"solved": 1},
    }
    serialized = json.dumps(telemetry, sort_keys=True)
    assert secret_path not in serialized
    assert secret_content not in serialized


def test_action_diagnostics_reject_unbounded_keys_at_result_boundary() -> None:
    evidence = passing_evidence()
    evidence.rejection_reason_counts = {"task-content-as-key": 1}

    with pytest.raises(RuntimeError, match="unbounded key"):
        eligibility.build_screen_result(evidence, gate.load_manifest())


def test_runtime_gate_is_derived_from_capacity_and_baseline_timestamps() -> None:
    manifest = gate.load_manifest()
    evidence = passing_evidence()
    maximum_predicted = manifest["screen"]["thresholds"][
        "maximum_predicted_final_evaluation_seconds"
    ]

    evidence.baseline_completed_at = evidence.capacity_smoke_completed_at
    missing_measurement = eligibility.build_screen_result(evidence, manifest)
    assert missing_measurement["predicted_final_evaluation_seconds"] == 0.0
    assert missing_measurement["gate_results"]["pilot_runtime_feasible"] is False

    multiplier = (
        manifest["pilot"]["test_examples"]
        * eligibility.pilot_final_evaluation_horizon_scale()
        * 2
        * manifest["pilot"]["final_evaluation_safety_factor"]
        / (manifest["screen"]["baseline_examples_per_level"] * len(eligibility.SCREEN_LEVELS))
    )
    assert eligibility.pilot_final_evaluation_horizon_scale() == 6.25
    evidence.baseline_completed_at = (
        evidence.capacity_smoke_completed_at + maximum_predicted / multiplier + 0.001
    )
    too_slow = eligibility.build_screen_result(evidence, manifest)
    assert too_slow["predicted_final_evaluation_seconds"] > maximum_predicted
    assert too_slow["gate_results"]["pilot_runtime_feasible"] is False


def test_hardware_gate_uses_the_exact_cuda_byte_floor() -> None:
    manifest = gate.load_manifest()
    evidence = passing_evidence()
    assert evidence.gpu_total_memory_bytes == 78_000_000_000
    assert eligibility.build_screen_result(evidence, manifest)["gate_results"]["hardware_verified"]

    evidence.gpu_total_memory_bytes -= 1
    result = eligibility.build_screen_result(evidence, manifest)
    assert result["gate_results"]["hardware_verified"] is False
    assert result["eligible"] is False


def test_actual_cuda_profile_is_rejected_before_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA A40"

        @staticmethod
        def get_device_properties(_index: int) -> object:
            return SimpleNamespace(total_memory=80_000_000_000)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    snapshot_hash_started = False

    def snapshot_verifier(_manifest: dict[str, object]) -> tuple[bool, None, None]:
        nonlocal snapshot_hash_started
        snapshot_hash_started = True
        return False, None, None

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with pytest.raises(gate.GateError, match="does not match"):
        eligibility.verify_pre_model_readiness(
            eligibility.ScreenEvidence(),
            gate.load_manifest(),
            torch_module=SimpleNamespace(cuda=WrongCuda()),
            snapshot_verifier=snapshot_verifier,
        )
    assert snapshot_hash_started is False


def test_baseline_stops_only_after_all_eight_l0_examples() -> None:
    manifest = gate.load_manifest()
    evidence = eligibility.ScreenEvidence(
        baseline_outcomes=[
            eligibility.BaselineOutcome(
                level=0,
                checkpoint_reached=index < 5,
                solved=index == 0,
            )
            for index in range(8)
        ]
    )

    assert (
        eligibility.baseline_impossible(evidence, manifest)
        == "LEVEL_0_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )
    evidence.baseline_outcomes.pop()
    assert eligibility.baseline_impossible(evidence, manifest) is None


def test_zero_solve_baseline_with_zero_floor_proceeds_to_branch_feasibility() -> None:
    manifest = profile()
    manifest["screen"]["thresholds"]["minimum_baseline_exact_rate"] = 0.0
    evidence = eligibility.ScreenEvidence(
        baseline_outcomes=[
            eligibility.BaselineOutcome(
                level=0,
                checkpoint_reached=True,
                solved=False,
            )
            for _ in range(8)
        ],
        repeated_rejected_loop_count=15,
    )

    assert eligibility.baseline_impossible(evidence, manifest) is None

    evidence.branch_collections.append(collection(informative=False, solved=0))
    assert eligibility.branch_collection_impossible(evidence, manifest) is None


def test_branch_checkpoint_gate_stops_at_first_mathematically_impossible_group() -> None:
    manifest = gate.load_manifest()
    evidence = eligibility.ScreenEvidence(
        branch_collections=[collection(checkpointed=False) for _ in range(2)],
    )
    assert eligibility.branch_collection_impossible(evidence, manifest) is None

    evidence.branch_collections.append(collection(checkpointed=False))
    assert (
        eligibility.branch_collection_impossible(evidence, manifest)
        == "BRANCH_CHECKPOINT_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )


def test_branch_fail_fast_covers_informative_rate_and_action_gates() -> None:
    manifest = gate.load_manifest()
    no_signal = eligibility.ScreenEvidence(
        branch_collections=[collection(informative=False) for _ in range(7)],
    )
    assert (
        eligibility.branch_collection_impossible(no_signal, manifest)
        == "INFORMATIVE_GROUP_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )

    too_many_malformed_actions = passing_evidence()
    too_many_malformed_actions.branch_collections.pop()
    too_many_malformed_actions.total_actions = 1_000
    too_many_malformed_actions.schema_valid_actions = 900
    assert (
        eligibility.branch_collection_impossible(too_many_malformed_actions, manifest)
        == "ACTION_PROTOCOL_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )


def test_branch_fail_fast_accounts_for_joint_sibling_rate_headroom() -> None:
    manifest = gate.load_manifest()
    evidence = eligibility.ScreenEvidence(
        branch_collections=[collection(informative=index < 2, solved=4) for index in range(7)],
    )

    assert (
        eligibility.branch_collection_impossible(evidence, manifest)
        == "SOLVED_SIBLING_RATE_GATE_MATHEMATICALLY_IMPOSSIBLE"
    )


def test_incomplete_branch_collection_is_not_counted_as_a_checkpoint() -> None:
    evidence = passing_evidence()
    evidence.branch_collections[0].prefix.terminal = True

    result = eligibility.build_screen_result(evidence, gate.load_manifest())

    assert result["branch_checkpoint_rate"] == 0.625
    assert result["gate_results"]["branch_checkpoint_rate"] is False
    assert result["eligible"] is False


def test_snapshot_readiness_propagates_the_exact_snapshot_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = profile()
    manifest["artifact_readiness"]["cache_directory"] = str(tmp_path)
    model_cache = "models--" + manifest["model"]["id"].replace("/", "--")
    snapshot = tmp_path / "hub" / model_cache / "snapshots" / manifest["model"]["revision"]
    snapshot.mkdir(parents=True)
    expected_digest = gate.expected_snapshot_digest(manifest)

    def verify_exact(
        observed_manifest: dict[str, object],
        observed_snapshot: Path,
    ) -> dict[str, object]:
        assert observed_manifest == manifest
        assert observed_snapshot == snapshot
        assert gate.expected_snapshot_digest(observed_manifest) == expected_digest
        return {
            "snapshot_path": str(snapshot),
            "snapshot_digest": expected_digest,
        }

    monkeypatch.setattr(eligibility.gate, "verify_local_snapshot", verify_exact)
    assert eligibility.verify_pinned_snapshot(manifest) == (
        True,
        str(snapshot),
        expected_digest,
    )

    monkeypatch.setattr(
        eligibility.gate,
        "verify_local_snapshot",
        lambda _manifest, _snapshot: (_ for _ in ()).throw(
            gate.GateError("snapshot digest mismatch")
        ),
    )
    assert eligibility.verify_pinned_snapshot(manifest) == (False, None, None)


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
        test_examples=eligibility.SCREEN_RUNTIME_TEST_EXAMPLES,
        training_tasks_per_update=manifest["screen"]["training_tasks_per_update"],
        maximum_updates=manifest["screen"]["maximum_updates"],
        optimization_seed=manifest["screen_limits"]["optimization_seed"],
        workload_attempt=1,
    )
    eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.model_revision = "floating-main"
    with pytest.raises(ValueError, match="model_revision"):
        eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.model_revision = manifest["model"]["revision"]
    runtime.optimization_seed += 1
    with pytest.raises(ValueError, match="optimization_seed"):
        eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.optimization_seed = manifest["screen_limits"]["optimization_seed"]
    runtime.test_examples += 1
    with pytest.raises(ValueError, match="test_examples"):
        eligibility.validate_runtime_configuration(runtime, manifest)

    runtime.test_examples = eligibility.SCREEN_RUNTIME_TEST_EXAMPLES
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


def test_runtime_hooks_install_l0_root_checkpoint_v33_k4_screen_and_isolate_test_split(
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
        "EQUINOX_RL_SEED": "137",
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
        "SHARED_PREFIX_CHECKPOINT_STRATEGY",
        "RepositoryRepairEnvironment",
        "branch_checkpoint_diagnostic_actions",
        "branch_checkpoint_reached",
        "training_level_allocation",
        "make_tasks",
        "bounded_final_evaluation_reserve",
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
        restore.setattr(
            eligibility.frozen_environment,
            "ENVIRONMENT_REVISION",
            eligibility.frozen_environment.ENVIRONMENT_REVISION,
        )
        restore.setattr(
            eligibility.frozen_environment,
            "ACTION_PROTOCOL_REVISION",
            eligibility.frozen_environment.ACTION_PROTOCOL_REVISION,
        )
        evidence = eligibility.ScreenEvidence()
        runtime = eligibility.prepare_runtime(evidence, manifest)
        count = eligibility.frozen.fixed_retention_guard_example_count(
            0,
            runtime.validation_examples,
        )
        tasks = eligibility.frozen.family_balanced_validation_tasks(0, count, 40_000)

        assert count == 8
        assert Counter(task.level for task in tasks) == {0: 8}
        assert eligibility.frozen.BRANCH_WIDTH == 4
        assert eligibility.frozen.TRAINING_MICROBATCH_SIZE == 1
        assert eligibility.frozen.MAX_INPUT_TOKENS == 2_048
        assert eligibility.frozen.bounded_final_evaluation_reserve(10_000, 300) == (
            0,
            False,
        )
        assert (
            eligibility.frozen.training_level_allocation(
                0,
                8,
                probe_level=1,
            )
            == [0] * 8
        )
        assert (
            eligibility.frozen.SHARED_PREFIX_CHECKPOINT_STRATEGY
            == eligibility.SCREEN_SHARED_PREFIX_CHECKPOINT_STRATEGY
        )
        assert eligibility.frozen.branch_checkpoint_diagnostic_actions(tasks[0]) == 1
        assert (
            manifest["interface"]["environment_revision"] == eligibility.frozen.ENVIRONMENT_REVISION
        )
        assert (
            manifest["interface"]["action_protocol_revision"]
            == eligibility.frozen.ACTION_PROTOCOL_REVISION
        )
        assert (
            manifest["interface"]["environment_revision"]
            == eligibility.frozen_environment.ENVIRONMENT_REVISION
        )
        assert (
            manifest["interface"]["action_protocol_revision"]
            == eligibility.frozen_environment.ACTION_PROTOCOL_REVISION
        )
        assert (
            eligibility.frozen.RepositoryRepairEnvironment.__mro__[1]
            is eligibility.revision33.RepositoryRepairEnvironment
        )
        environment = eligibility.frozen.RepositoryRepairEnvironment(tasks[0])
        prompt = environment.policy_prompt("shared_prefix")
        assert eligibility.ROOT_CHECKPOINT_PHASE_INSTRUCTION in prompt
        assert "Read each relevant implementation file before the checkpoint." not in prompt
        assert eligibility.frozen.branch_checkpoint_reached(tasks[0], environment) is False
        root_result = environment.step('{"tool":"list","path":""}')
        assert root_result.accepted is True
        assert eligibility.frozen.branch_checkpoint_reached(tasks[0], environment) is True
        assert eligibility.all_fault_sources_observed(tasks[0], environment) is False
        fault_path = tasks[0].faults[0].path
        read_result = environment.step(f'{{"tool":"read","path":"{fault_path}"}}')
        assert read_result.accepted is True
        assert eligibility.all_fault_sources_observed(tasks[0], environment) is True
        assert evidence.action_counts_by_tool == {"list": 1, "read": 1}
        assert evidence.accepted_action_counts_by_tool == {"list": 1, "read": 1}
        assert evidence.rejected_action_counts_by_tool == {}
        assert evidence.rejection_reason_counts == {}
        environment.terminal = True
        assert eligibility.frozen.branch_checkpoint_reached(tasks[0], environment) is False
        isolation_progress: dict[str, object] = {}

        def capture_isolation_progress(
            phase: str,
            _message: str,
            _runtime_configuration: object,
            **values: object,
        ) -> None:
            isolation_progress.update(values)
            isolation_progress["phase"] = phase

        restore.setattr(
            eligibility.frozen,
            "emit_progress",
            capture_isolation_progress,
        )
        with pytest.raises(RuntimeError, match="test split"):
            eligibility.frozen.make_tasks(0, 1, 190_000, split="test")
        assert evidence.test_split_accessed is True
        assert isolation_progress["phase"] == "test_split_isolation_violation"
        assert isolation_progress["test_split_accessed"] is True
        assert isolation_progress["test_examples_accessed"] == 0


def test_progress_reports_the_current_test_isolation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: dict[str, object] = {}

    def capture_progress(
        _phase: str,
        _message: str,
        _runtime_configuration: object,
        *,
        preserve_context: bool = False,
        **values: object,
    ) -> None:
        emitted.update(values)
        emitted["preserve_context"] = preserve_context

    monkeypatch.setattr(eligibility.frozen, "emit_progress", capture_progress)
    evidence = eligibility.ScreenEvidence(test_split_accessed=True)
    eligibility.install_progress_hook(evidence, gate.load_manifest())

    eligibility.frozen.emit_progress(
        "baseline_evaluation",
        "blocked",
        SimpleNamespace(),
    )

    assert emitted["test_split_accessed"] is True


def test_pretest_finalization_returns_an_ineligible_screen_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eligibility.frozen,
        "emit_progress",
        lambda *_args, **_kwargs: None,
    )
    evidence = passing_evidence()
    evidence.branch_collections = evidence.branch_collections[:2]
    eligibility.install_progress_hook(evidence, gate.load_manifest())

    with pytest.raises(eligibility.EligibilityScreenComplete) as completed:
        eligibility.frozen.emit_progress(
            "finalizing",
            "generic trainer attempted final evaluation",
            SimpleNamespace(),
        )

    assert completed.value.result["screen_completed"] is True
    assert completed.value.result["eligible"] is False
    assert completed.value.result["early_stop_reason"] == "screen_collection_incomplete"
    assert completed.value.result["test_split_accessed"] is False
