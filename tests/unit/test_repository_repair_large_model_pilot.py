from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runpod import larger_model_gate as gate
from research.runpod import repository_repair_large_model_pilot as pilot


def test_pilot_installs_the_shared_fail_closed_tokenizer_guard() -> None:
    raw_tokenizer = object()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: raw_tokenizer,
        )
    )

    pilot.install_fail_closed_tokenizer_loader(transformers)
    guarded = transformers.AutoTokenizer.from_pretrained("model", revision="revision")

    assert isinstance(guarded, pilot.eligibility.FailClosedPromptTokenizer)


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
            return "NVIDIA H100 80GB HBM3"

        @staticmethod
        def get_device_properties(_index: int) -> object:
            return SimpleNamespace(total_memory=77_999_999_999)

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


def test_v33_pilot_constants_do_not_modify_frozen_sources() -> None:
    assert pilot.MODEL_REVISION == gate.MODEL_REVISION
    assert pilot.WORKLOAD_REVISION == "runpod-repository-repair-large-model-pilot@4"
    assert pilot.OBJECTIVE_ID == "verified-repair-chain-root-branch-retention-policy-gradient@17"
    assert pilot.REWARD_CONTRACT_REVISION == "correctness-gated-efficiency@1"
    assert pilot.SHARED_PREFIX_CHECKPOINT_STRATEGY == "repository_root_observed@1"
    assert Path(pilot.__file__).name == "repository_repair_large_model_pilot.py"
    assert os.path.basename(pilot.frozen.__file__) == "repository_repair_rl.py"


def test_shared_prefix_prompt_requests_only_the_root_listing() -> None:
    task = pilot.frozen.make_task(0, seed=73)
    environment = pilot.PilotRepositoryRepairEnvironment(task)

    shared_prefix = environment.policy_prompt("shared_prefix")
    continuation = environment.policy_prompt("continuation")
    unchanged_continuation = pilot.interface.RepositoryRepairEnvironment(task).policy_prompt(
        "continuation"
    )

    assert shared_prefix.startswith("PHASE INSTRUCTION\n")
    assert 'Return exactly {"tool":"list","path":""}' in shared_prefix
    assert "Read each relevant implementation file before the checkpoint" not in shared_prefix
    shared_data = json.loads(
        shared_prefix.split("<untrusted-environment-data>\n", 1)[1].splitlines()[0]
    )
    assert shared_data["interface_state"]["mechanical_action_space"]["list_paths"] == [""]
    assert continuation == unchanged_continuation


def test_root_checkpoint_requires_an_accepted_complete_nonterminal_listing() -> None:
    task = pilot.frozen.make_task(0, seed=74)
    environment = pilot.PilotRepositoryRepairEnvironment(task)
    fault_path = task.faults[0].path

    rejected_read = environment.step(
        pilot.frozen.encode_action({"tool": "read", "path": fault_path}),
        allowed_tools=pilot.frozen.DIAGNOSTIC_TOOLS,
    )
    assert rejected_read.accepted is False
    assert pilot.repository_root_observed_checkpoint(task, environment) is False

    listing = environment.step(
        pilot.frozen.encode_action({"tool": "list", "path": ""}),
        allowed_tools=pilot.frozen.DIAGNOSTIC_TOOLS,
    )
    assert listing.accepted is True
    assert json.loads(listing.observation) == {"files": sorted(task.files)}
    assert pilot.repository_root_observed_checkpoint(task, environment) is True

    environment.terminal = True
    assert pilot.repository_root_observed_checkpoint(task, environment) is False


def test_branching_precedes_localization_and_keeps_coverage_as_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = pilot.frozen.make_task(0, seed=75)
    fault_path = task.faults[0].path
    sampling_seed = 5_000
    monkeypatch.setattr(
        pilot.frozen,
        "RepositoryRepairEnvironment",
        pilot.PilotRepositoryRepairEnvironment,
    )
    monkeypatch.setattr(
        pilot.frozen,
        "branch_checkpoint_reached",
        pilot.repository_root_observed_checkpoint,
    )

    def sample_one(
        _prompt: str,
        _stochastic: bool,
        seed: int,
    ) -> pilot.frozen.GeneratedAction:
        if seed == sampling_seed:
            action = {"tool": "list", "path": ""}
        elif seed == sampling_seed + 10_000:
            action = {"tool": "read", "path": fault_path}
        else:
            action = {"tool": "finish"}
        return pilot.frozen.GeneratedAction(
            response=pilot.frozen.encode_action(action),
        )

    collection = pilot.frozen.collect_branch_group(
        task,
        sample_one,
        stochastic=True,
        sampling_seed=sampling_seed,
    )
    serialized = pilot.serialize_pilot_branch_group(collection, update=1)

    assert collection.snapshot is not None
    assert len(collection.prefix.steps) == 1
    assert len(collection.siblings) == 4
    assert collection.prefix.steps[0].action == {"tool": "list", "path": ""}
    assert serialized["checkpoint"]["static_branch_width"] == 4
    assert serialized["shared_prefix"]["checkpoint_strategy"] == "repository_root_observed@1"
    assert serialized["shared_prefix"]["required_diagnostic_actions"] == 1
    assert serialized["shared_prefix"]["required_fault_source_reads"] == 0
    assert serialized["shared_prefix"]["observed_fault_source_paths"] == []
    assert serialized["shared_prefix"]["repository_root_observed"] is True
    telemetry = serialized["localization_telemetry"]
    assert telemetry["strategy"] == "all_fault_sources_observed"
    assert telemetry["branch_admission_gate"] is False
    assert telemetry["prefix"]["all_fault_sources_observed"] is False
    assert telemetry["siblings"][0]["observed_fault_source_paths"] == [fault_path]
    assert telemetry["siblings"][0]["all_fault_sources_observed"] is True
    assert all(
        sibling["all_fault_sources_observed"] is False for sibling in telemetry["siblings"][1:]
    )


def test_bootstrap_contract_pins_static_k_and_truthful_training_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = (
        (pilot.frozen, "RepositoryRepairEnvironment"),
        (pilot.frozen, "BRANCH_WIDTH"),
        (pilot.frozen_environment, "BRANCH_WIDTH"),
        (pilot.frozen, "SHARED_PREFIX_CHECKPOINT_STRATEGY"),
        (pilot.frozen, "MINIMUM_PREFIX_ACCEPTED_ACTIONS"),
        (pilot.frozen, "branch_checkpoint_reached"),
        (pilot.frozen, "fault_fixing_edit_actions"),
        (pilot.frozen, "POLICY_CREDIT_SCOPE"),
        (pilot.frozen, "serialize_branch_group"),
    )
    for target, name in tracked:
        monkeypatch.setattr(target, name, getattr(target, name))

    pilot.install_bootstrap_checkpoint_contract(4)

    assert pilot.frozen.BRANCH_WIDTH == 4
    assert pilot.frozen_environment.BRANCH_WIDTH == 4
    assert pilot.frozen.MINIMUM_PREFIX_ACCEPTED_ACTIONS == 1
    assert pilot.frozen.SHARED_PREFIX_CHECKPOINT_STRATEGY == "repository_root_observed@1"
    assert pilot.frozen.POLICY_CREDIT_SCOPE == pilot.POLICY_CREDIT_SCOPE
    assert pilot.frozen.fault_fixing_edit_actions is pilot.fault_fixing_edit_and_fresh_read_actions
    with pytest.raises(ValueError, match="static K=4"):
        pilot.install_bootstrap_checkpoint_contract(1)


def test_policy_credit_is_limited_to_fresh_read_and_fix_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = pilot.frozen.make_task(0, seed=76)
    fault = task.faults[0]
    unrelated_path = next(path for path in sorted(task.files) if path != fault.path)
    sampling_seed = 6_000
    monkeypatch.setattr(
        pilot.frozen,
        "RepositoryRepairEnvironment",
        pilot.PilotRepositoryRepairEnvironment,
    )
    monkeypatch.setattr(
        pilot.frozen,
        "branch_checkpoint_reached",
        pilot.repository_root_observed_checkpoint,
    )
    monkeypatch.setattr(
        pilot.frozen,
        "fault_fixing_edit_actions",
        pilot.fault_fixing_edit_and_fresh_read_actions,
    )

    successful_actions = (
        {"tool": "search", "query": fault.old},
        {"tool": "read", "path": unrelated_path},
        {"tool": "read", "path": fault.path},
        {
            "tool": "edit",
            "path": fault.path,
            "old": fault.old,
            "new": fault.new,
        },
        {"tool": "test"},
    )

    def sample_one(
        _prompt: str,
        _stochastic: bool,
        seed: int,
    ) -> pilot.frozen.GeneratedAction:
        if seed == sampling_seed:
            action = {"tool": "list", "path": ""}
        elif sampling_seed + 10_000 <= seed < sampling_seed + 10_005:
            action = successful_actions[seed - sampling_seed - 10_000]
        else:
            action = {"tool": "finish"}
        return pilot.frozen.GeneratedAction(
            response=pilot.frozen.encode_action(action),
            input_ids=(seed,),
        )

    collection = pilot.frozen.collect_branch_group(
        task,
        sample_one,
        stochastic=True,
        sampling_seed=sampling_seed,
    )

    credited = pilot.fault_fixing_edit_and_fresh_read_actions(collection, 0)
    credited_tools = [json.loads(action.response)["tool"] for action in credited]
    examples = pilot.frozen.policy_examples(collection)
    serialized = pilot.serialize_pilot_branch_group(collection, update=1)

    assert collection.solved_siblings == 1
    assert collection.informative is True
    assert credited_tools == ["read", "edit"]
    assert pilot.fault_fixing_edit_and_fresh_read_actions(collection, 1) == []
    assert [example.weight for example in examples] == [0.5, 0.5]
    assert sum(example.weight for example in examples) == 1.0
    assert [step["policy_signal"] for step in serialized["siblings"][0]["steps"]] == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert serialized["policy_credit_scope"] == pilot.POLICY_CREDIT_SCOPE


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
            gpu_name="NVIDIA H100 80GB HBM3",
            total_memory_bytes=80_000_000_000,
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
            "reward_contract": {"revision": "correctness-gated-efficiency@1"},
            "training_configuration": {
                "shared_prefix_checkpoint": "stale",
                "minimum_shared_prefix_actions": 2,
                "policy_credit_scope": "stale",
                "learning_signal": "stale",
            },
        },
        manifest=manifest,
        authorization_digest="sha256:" + "a" * 64,
        snapshot=Path("/workspace/model"),
    )

    assert result["optimization_seed"] == optimization_seed
    assert result["source_contract_digest"] == gate.expected_source_contract_digest(manifest)
    assert (
        result["terminal_submission_contract"]
        == manifest["interface"]["terminal_submission_contract"]
    )
    assert result["training_configuration"]["shared_prefix_checkpoint"] == (
        "repository_root_observed@1"
    )
    assert result["training_configuration"]["minimum_shared_prefix_actions"] == 1
    assert result["training_configuration"]["policy_credit_scope"] == pilot.POLICY_CREDIT_SCOPE
    assert result["training_configuration"]["learning_signal"] == pilot.LEARNING_SIGNAL


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
