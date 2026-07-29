from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from research.runpod.larger_model_gate import load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/runpod-rl-proof"
SCREEN = REPOSITORY_ROOT / "scripts/screen-larger-model"
PILOT = REPOSITORY_ROOT / "scripts/run-larger-model-pilot"


def screen_proof_assertion() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    marker = 'if [[ "$larger_model_mode" == "screen" ]]; then\n  proof_assertion=\''
    start = source.index(marker) + len(marker)
    end = source.index("\n  '\nelif [[", start)
    return source[start:end]


def run_screen_proof_assertion(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "jq",
            "-e",
            "--arg",
            "expected_model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--arg",
            "expected_snapshot_digest",
            "sha256:snapshot",
            "--arg",
            "expected_source_contract_digest",
            "sha256:source",
            "--argjson",
            "expected_optimization_seed",
            "137",
            screen_proof_assertion(),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def test_larger_model_operator_scripts_are_valid_and_executable() -> None:
    subprocess.run(
        ["bash", "-n", str(LAUNCHER), str(SCREEN), str(PILOT)],
        check=True,
    )
    assert os.access(SCREEN, os.X_OK)
    assert os.access(PILOT, os.X_OK)


def test_screen_preflight_verifies_model_inventory_and_immutable_caps() -> None:
    source = SCREEN.read_text(encoding="utf-8")
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    manifest = load_manifest()

    assert "verify-model" in source
    assert "gpu list --include-unavailable" in source
    assert "EQUINOX_RUNPOD_PREFLIGHT_ONLY=1" in source
    larger_model_case = launcher_source[
        launcher_source.index('if [[ -n "$larger_model_mode" ]]; then') : launcher_source.index(
            'elif [[ -z "$study_condition" ]]; then'
        )
    ]
    assert 'contract_workload_file="$workload_file"' not in larger_model_case
    assert 'network-volume get "$network_volume_id"' in launcher_source
    assert "network_volume_data_center_id" in launcher_source
    assert "runpodctl datacenter list" in launcher_source
    assert "jq -r '.model.revision' <<<\"$larger_model_manifest\"" in launcher_source
    assert (
        f"EQUINOX_RUNPOD_MAX_TOTAL_COST={manifest['screen_limits']['maximum_total_cost_usd']}"
        in source
    )
    assert "EQUINOX_RUNPOD_MAX_WORKLOAD_ATTEMPTS=1" in source


def test_pilot_requires_exact_screen_authorization_before_launcher() -> None:
    source = PILOT.read_text(encoding="utf-8")

    assert "larger_model_gate.py" in source
    assert "authorize" in source
    assert "EQUINOX_LARGER_MODEL_SCREEN_RESULT" in source
    assert "EQUINOX_LARGER_MODEL_SCREEN_RECEIPT" in source
    assert "EQUINOX_LARGER_MODEL_AUTHORIZATION_DIGEST" in source
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    assert ".retained_checkpoint_update == .best_validation.update" in launcher_source
    assert ".effective_policy_update_count <= .policy_update_count" in launcher_source


def test_paid_larger_model_worker_is_offline_and_cannot_invoke_pip() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "EQUINOX_RUNPOD_NETWORK_VOLUME_ID" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "PIP_NO_INDEX=1" in source
    assert "PYTHONPATH=/workspace/equinox-state/python" in source
    assert "EQUINOX_RUNPOD_MODEL_LOAD_TIMEOUT_SECONDS" in source
    assert "EQUINOX_RUNPOD_STALE_PROGRESS_SECONDS" in source


def test_provider_query_failure_is_unknown_and_teardown_requires_zero_spend() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "printf '%s\\n' \"unknown\"" in source
    assert "consecutive_absent >= 3" in source
    assert "currentSpendPerHr" in source
    assert "reconcile_pod_id_by_name" in source
    assert 'if [[ -z "$pod_id" ]]; then\n      reconcile_pod_id_by_name || true' in source
    assert "pod_is_present" not in source
    assert "teardown_confirmed=true" in source
    assert 'and (.id | type) == "string"' in source
    assert 'and (.name | type) == "string"' in source
    assert "ambiguous_create_safe_after_epoch" in source
    assert "current_epoch >= ambiguous_create_safe_after_epoch" in source


def test_paid_provider_commands_and_cleanup_are_hard_bounded() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'timeout --kill-after=5 "$duration"' in source
    assert 'create_response="$(run_provider_command 90 runpodctl' in source
    assert 'pod_response="$(run_provider_command 30 runpodctl pod get' in source
    assert "trap '' INT TERM HUP" in source
    assert "cleanup_reserve_seconds=120" in source
    assert "ambiguous_create_visibility_seconds=300" in source
    assert "ambiguous_create_reconciliation_seconds=330" in source
    assert "artifact_retrieval_reserve_seconds=180" in source
    assert (
        "result_deadline_epoch=$((provider_deadline_epoch - cleanup_reserve_seconds "
        "- artifact_retrieval_reserve_seconds))"
    ) in source


def test_operator_lease_and_authorization_are_shared_and_durable() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '".local/state/equinox/runpod"' in source
    assert 'operator_lock_directory="$operator_state_root/operator.lock"' in source
    assert "durable_write_stdin" in source
    assert "durable_create_stdin" in source
    assert "os.O_EXCL" in source
    assert "os.fsync" in source
    assert "an unresolved shared RunPod operator lease exists" in source


def test_ambiguous_create_preserves_any_valid_returned_pod_id() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    valid_id_branch = source[
        source.index('if [[ "$pod_id" =~ ^[a-zA-Z0-9_-]+$ ]]') : source.index(
            'failure_message="RunPod did not report a bounded hourly rate."'
        )
    ]

    assert 'provider_handle="runpod://pods/$pod_id"' in valid_id_branch
    assert 'pod_id=""' not in valid_id_branch.split("else", maxsplit=1)[0]
    assert "cleanup will delete that exact pod" in valid_id_branch


def test_larger_model_direct_launch_caps_are_manifest_pinned() -> None:
    manifest = load_manifest()
    launcher = LAUNCHER.read_text(encoding="utf-8")
    screen = SCREEN.read_text(encoding="utf-8")
    pilot = PILOT.read_text(encoding="utf-8")

    assert ".runtime.image_digest" in launcher
    for key in (
        "boot_timeout_seconds",
        "stale_progress_timeout_seconds",
        "target_runtime_seconds",
        "retry_reserve_seconds",
        "maximum_final_evaluation_reserve_seconds",
        "optimization_seed",
    ):
        assert key in launcher
    assert (
        f"EQUINOX_RUNPOD_STALE_PROGRESS_SECONDS={manifest['screen_limits']['stale_progress_timeout_seconds']}"
        in screen
    )
    assert (
        f"EQUINOX_RUNPOD_STALE_PROGRESS_SECONDS={manifest['pilot_limits']['stale_progress_timeout_seconds']}"
        in pilot
    )
    assert "EQUINOX_RL_SEED=137" in screen
    assert "EQUINOX_RL_SEED=137" in pilot
    assert "verify-sources" in screen
    assert "verify-sources" in pilot
    assert "verify-sources" in launcher
    assert ".source_contract_digest == $expected_source_contract_digest" in launcher


def test_total_cost_and_phase_deadlines_are_enforced_during_polling() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "accrued_cost" in source
    assert "maximum_total_cost" in source
    assert "preparation_started_epoch" in source
    assert "model_load_timeout_seconds" in source
    assert "last_progress_change_epoch" in source
    assert (
        'curl --fail --silent --show-error --max-time 30 \\\n    -H "$remote_authorization"'
        in source
    )


def test_completed_ineligible_screen_is_valid_evidence_for_failed_metric_gates() -> None:
    gate_results = {
        gate_name: True for gate_name in load_manifest()["authorization"]["required_gate_results"]
    }
    gate_results["peak_reserved_vram_within_limit"] = False
    gate_results["pilot_runtime_feasible"] = False
    failed_gates = sorted(gate_name for gate_name, passed in gate_results.items() if not passed)
    payload: dict[str, object] = {
        "device": "cuda",
        "workload": "repository-repair-larger-model-eligibility-screen",
        "workload_revision": "larger-model-eligibility-screen@1",
        "profile_id": "qwen2.5-coder-7b-runpod-h100@1",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "optimization_seed": 137,
        "source_contract_digest": "sha256:source",
        "screen_completed": True,
        "eligible": False,
        "policy_mutation_detected": False,
        "optimizer_state_restored": True,
        "test_split_accessed": False,
        "branch_width": 4,
        "gpu_id": "NVIDIA H100 80GB HBM3",
        "gpu_total_memory_bytes": 80_000_000_000,
        "peak_reserved_vram_fraction": 0.9,
        "pinned_snapshot_digest": "sha256:snapshot",
        "predicted_final_evaluation_seconds": 1_500,
        "completed_baseline_examples": 32,
        "per_level_checkpoint_rates": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
        "informative_groups": 2,
        "solved_siblings": 2,
        "failed_siblings": 2,
        "gate_results": gate_results,
        "ineligibility_reasons": failed_gates,
        "ineligible_reasons": failed_gates,
    }

    assert run_screen_proof_assertion(payload).returncode == 0
    payload["ineligibility_reasons"] = ["pilot_runtime_feasible"]
    payload["ineligible_reasons"] = ["pilot_runtime_feasible"]
    assert run_screen_proof_assertion(payload).returncode != 0
