from research.runpod.revision31_report import (
    branching_gate,
    external_transfer_gate,
    internal_learning_gate,
)

SEEDS = (137, 269, 443, 617, 887)


def training_result(net: int, regressed: int = 0) -> dict:
    return {
        "paired_test_change": {
            "net_improved": net,
            "improved": net + regressed,
            "regressed": regressed,
        }
    }


def external_adapter(successes: int, net: int, regressed: int = 0) -> dict:
    return {
        "exact_successes": successes,
        "paired_change_vs_base": {
            "net_improved": net,
            "improved": net + regressed,
            "regressed": regressed,
        },
    }


def test_internal_gate_applies_preregistered_thresholds() -> None:
    passing = {f"k4_adaptive_seed{seed}": training_result(2) for seed in SEEDS}

    assert internal_learning_gate(passing)["status"] == "PASS"

    passing["k4_adaptive_seed137"] = training_result(-1, 1)
    passing["k4_adaptive_seed269"] = training_result(-1, 1)
    assert internal_learning_gate(passing)["status"] == "FAIL"


def test_external_gate_uses_all_five_primary_seeds_and_aggregate_mcnemar() -> None:
    passing = {f"k4_adaptive_seed{seed}": external_adapter(24, 8) for seed in SEEDS}

    decision = external_transfer_gate(passing)

    assert decision["status"] == "PASS"
    assert decision["evidence"]["median_gain_tasks"] == 8
    assert decision["evidence"]["aggregate_mcnemar_exact_p_value"] <= 0.05


def test_branching_gate_uses_both_curriculum_policies_within_seed() -> None:
    external = {}
    for seed in SEEDS:
        for curriculum in ("scheduled_dynamic", "adaptive"):
            external[f"k1_{curriculum}_seed{seed}"] = external_adapter(10, 0)
            external[f"k4_{curriculum}_seed{seed}"] = external_adapter(14, 4)

    decision = branching_gate(external)

    assert decision["status"] == "PASS"
    assert decision["evidence"]["median_k4_main_effect_tasks"] == 4


def test_gates_remain_incomplete_when_evidence_is_missing() -> None:
    assert internal_learning_gate({})["status"] == "INCOMPLETE"
    assert external_transfer_gate({})["status"] == "INCOMPLETE"
    assert branching_gate({})["status"] == "INCOMPLETE"
