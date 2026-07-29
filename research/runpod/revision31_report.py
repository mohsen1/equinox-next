"""Compute the preregistered revision-31 graduation decision."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - macOS Python 3.9.
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - Python 3.9 compatibility.

from research.runpod.revision31_external_eval import (
    WORKLOAD as EXTERNAL_WORKLOAD,
)
from research.runpod.revision31_study_operator import (
    STUDY_ID,
    all_condition_ids,
    condition_from_id,
    load_manifest,
)

REPORT_ID = f"{STUDY_ID}/graduation-report@1"
TRAINING_WORKLOAD_REVISION = "runpod-repository-repair-causal-credit@31"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
NUMERIC_TYPES = (int, float)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an object")
    return value


def training_results(receipt_directory: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in sorted(receipt_directory.glob("runpod-proof-*.result.json")):
        result = read_json(path)
        study = result.get("study")
        if (
            result.get("workload_revision") != TRAINING_WORKLOAD_REVISION
            or not isinstance(study, dict)
            or study.get("study_id") != STUDY_ID
        ):
            continue
        condition = study.get("condition")
        seed = study.get("optimization_seed")
        if not isinstance(condition, str) or not isinstance(seed, int):
            raise ValueError(f"{path} has invalid revision-31 condition evidence")
        condition_id = f"{condition}_seed{seed}"
        if condition_id in results:
            raise ValueError(f"multiple successful results exist for {condition_id}")
        results[condition_id] = result
    return results


def external_results(receipt_directory: Path) -> list[dict[str, Any]]:
    return [
        result
        for path in sorted(receipt_directory.glob("runpod-proof-*.result.json"))
        if (result := read_json(path)).get("workload") == EXTERNAL_WORKLOAD
    ]


def external_adapters_by_condition(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    adapters = {}
    base_outcomes: list[dict[str, Any]] | None = None
    for result in results:
        if (
            result.get("external_evaluation_completed") is not True
            or result.get("every_adapter_reported") is not True
            or result.get("task_count") != 120
        ):
            continue
        base = result.get("base")
        if not isinstance(base, dict) or not isinstance(base.get("task_outcomes"), list):
            continue
        if base_outcomes is None:
            base_outcomes = base["task_outcomes"]
        elif base["task_outcomes"] != base_outcomes:
            raise ValueError("external shards produced different frozen-base outcomes")
        for adapter in result.get("adapters", []):
            condition_id = adapter.get("condition_id") if isinstance(adapter, dict) else None
            if not isinstance(condition_id, str) or condition_id in adapters:
                raise ValueError("external adapter results are missing or duplicated")
            adapters[condition_id] = adapter
    return adapters


def receipt_by_execution(receipt_directory: Path) -> dict[str, dict[str, Any]]:
    receipts = {}
    for path in sorted(receipt_directory.glob("runpod-proof-*.json")):
        if path.name.endswith(".result.json"):
            continue
        receipts[path.stem] = read_json(path)
    return receipts


def execution_id_for_result(
    receipt_directory: Path,
    condition_id: str,
) -> str | None:
    for path in sorted(receipt_directory.glob("runpod-proof-*.result.json")):
        result = read_json(path)
        study = result.get("study")
        if not isinstance(study, dict):
            continue
        observed = f"{study.get('condition')}_seed{study.get('optimization_seed')}"
        if study.get("study_id") == STUDY_ID and observed == condition_id:
            return path.name.removesuffix(".result.json")
    return None


def exact_cost(receipt: dict[str, Any]) -> float | None:
    profile = receipt.get("resource_profile")
    rate = profile.get("hourly_cost_usd") if isinstance(profile, dict) else None
    started_at = receipt.get("started_at")
    completed_at = receipt.get("completed_at")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, NUMERIC_TYPES)
        or not isinstance(started_at, str)
        or not isinstance(completed_at, str)
    ):
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(float(rate) * max(0.0, (finish - start).total_seconds()) / 3_600, 6)


def decision(complete: bool, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS" if complete and passed else "FAIL" if complete else "INCOMPLETE",
        "evidence": evidence,
    }


def paired_metrics(result: dict[str, Any]) -> tuple[int, int, int]:
    paired = result.get("paired_test_change")
    if not isinstance(paired, dict):
        raise ValueError("training result omits paired test evidence")
    values = tuple(paired.get(key) for key in ("net_improved", "improved", "regressed"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("training paired test evidence is invalid")
    return values


def mcnemar_exact_p_value(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    smaller_tail = (
        sum(math.comb(discordant, count) for count in range(min(improved, regressed) + 1))
        / 2**discordant
    )
    return min(1.0, 2 * smaller_tail)


def operational_gate(
    expected_ids: list[str],
    training: dict[str, dict[str, Any]],
    receipt_directory: Path,
    receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = {}
    for condition_id in expected_ids:
        result = training.get(condition_id)
        execution_id = execution_id_for_result(receipt_directory, condition_id)
        receipt = receipts.get(execution_id) if execution_id else None
        rows[condition_id] = {
            "completed": result is not None,
            "final_evaluation_complete": (
                result.get("final_evaluation_complete") if result else None
            ),
            "adapter_persisted": result.get("adapter_persisted") if result else None,
            "action_protocol_validity_rate": (
                result.get("action_protocol_validity_rate") if result else None
            ),
            "execution_id": execution_id,
            "teardown_confirmed": (
                receipt.get("teardown_confirmed") if isinstance(receipt, dict) else None
            ),
        }
    complete = len(training) == len(expected_ids)
    passed = complete and all(
        row["final_evaluation_complete"] is True
        and row["adapter_persisted"] is True
        and isinstance(row["action_protocol_validity_rate"], NUMERIC_TYPES)
        and not isinstance(row["action_protocol_validity_rate"], bool)
        and row["action_protocol_validity_rate"] >= 0.99
        and row["teardown_confirmed"] is True
        for row in rows.values()
    )
    return decision(complete, passed, {"conditions": rows})


def internal_learning_gate(training: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = [f"k4_adaptive_seed{seed}" for seed in (137, 269, 443, 617, 887)]
    rows = {}
    for condition_id in ids:
        result = training.get(condition_id)
        if result is None:
            rows[condition_id] = None
            continue
        net, improved, regressed = paired_metrics(result)
        rows[condition_id] = {
            "net_improved": net,
            "improved": improved,
            "regressed": regressed,
            "clean_positive_gain": net > 0 and regressed == 0,
        }
    complete = all(row is not None for row in rows.values())
    observed = [row for row in rows.values() if row is not None]
    positive = sum(row["clean_positive_gain"] for row in observed)
    median_gain = statistics.median(row["net_improved"] for row in observed) if observed else None
    regressions = sum(row["regressed"] for row in observed)
    passed = (
        complete
        and positive >= 4
        and median_gain is not None
        and median_gain >= 1
        and regressions <= 2
    )
    return decision(
        complete,
        passed,
        {
            "seeds": rows,
            "positive_clean_gain_seeds": positive,
            "median_gain_tasks": median_gain,
            "total_regressed_tasks": regressions,
        },
    )


def external_transfer_gate(
    external: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = [f"k4_adaptive_seed{seed}" for seed in (137, 269, 443, 617, 887)]
    rows = {}
    aggregate_improved = 0
    aggregate_regressed = 0
    for condition_id in ids:
        adapter = external.get(condition_id)
        paired = adapter.get("paired_change_vs_base") if isinstance(adapter, dict) else None
        if not isinstance(paired, dict):
            rows[condition_id] = None
            continue
        row = {
            "net_improved": paired.get("net_improved"),
            "improved": paired.get("improved"),
            "regressed": paired.get("regressed"),
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in row.values()):
            raise ValueError("external paired evidence is invalid")
        rows[condition_id] = row
        aggregate_improved += row["improved"]
        aggregate_regressed += row["regressed"]
    complete = all(row is not None for row in rows.values())
    observed = [row for row in rows.values() if row is not None]
    positive = sum(row["net_improved"] > 0 for row in observed)
    median_gain = statistics.median(row["net_improved"] for row in observed) if observed else None
    regressing_seeds = sum(row["net_improved"] < 0 for row in observed)
    p_value = mcnemar_exact_p_value(aggregate_improved, aggregate_regressed)
    passed = (
        complete
        and positive >= 4
        and median_gain is not None
        and median_gain >= 6
        and regressing_seeds <= 1
        and p_value <= 0.05
    )
    return decision(
        complete,
        passed,
        {
            "seeds": rows,
            "positive_gain_seeds": positive,
            "median_gain_tasks": median_gain,
            "regressing_seeds": regressing_seeds,
            "aggregate_improved": aggregate_improved,
            "aggregate_regressed": aggregate_regressed,
            "aggregate_mcnemar_exact_p_value": p_value,
        },
    )


def branching_gate(
    external: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    effects = {}
    for seed in (137, 269, 443, 617, 887):
        cell_effects = {}
        for curriculum in ("scheduled_dynamic", "adaptive"):
            k1 = external.get(f"k1_{curriculum}_seed{seed}")
            k4 = external.get(f"k4_{curriculum}_seed{seed}")
            if not isinstance(k1, dict) or not isinstance(k4, dict):
                cell_effects[curriculum] = None
                continue
            k1_successes = k1.get("exact_successes")
            k4_successes = k4.get("exact_successes")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (k1_successes, k4_successes)
            ):
                raise ValueError("external exact-success evidence is invalid")
            cell_effects[curriculum] = k4_successes - k1_successes
        values = [value for value in cell_effects.values() if value is not None]
        effects[str(seed)] = {
            "by_curriculum": cell_effects,
            "main_effect_tasks": statistics.mean(values) if len(values) == 2 else None,
        }
    complete = all(row["main_effect_tasks"] is not None for row in effects.values())
    values = [
        row["main_effect_tasks"] for row in effects.values() if row["main_effect_tasks"] is not None
    ]
    positive = sum(value > 0 for value in values)
    median_effect = statistics.median(values) if values else None
    passed = complete and positive >= 4 and median_effect is not None and median_effect >= 3
    return decision(
        complete,
        passed,
        {
            "seeds": effects,
            "positive_k4_main_effect_seeds": positive,
            "median_k4_main_effect_tasks": median_effect,
        },
    )


def build_report(repository_root: Path) -> dict[str, Any]:
    manifest = load_manifest(repository_root / "research/studies/revision31-causal-study.json")
    receipt_directory = repository_root / "var/research-proofs"
    expected_ids = all_condition_ids(manifest)
    training = training_results(receipt_directory)
    external = external_adapters_by_condition(external_results(receipt_directory))
    receipts = receipt_by_execution(receipt_directory)
    gates = {
        "operational": operational_gate(
            expected_ids,
            training,
            receipt_directory,
            receipts,
        ),
        "internal_learning": internal_learning_gate(training),
        "external_transfer": external_transfer_gate(external),
        "branching": branching_gate(external),
    }
    required_passed = all(gate["status"] == "PASS" for gate in gates.values())
    all_complete = all(gate["status"] != "INCOMPLETE" for gate in gates.values())
    gates["larger_model"] = decision(
        all_complete,
        required_passed,
        {
            "required_gates": [
                "operational",
                "internal_learning",
                "external_transfer",
                "branching",
            ],
            "authorized": required_passed,
        },
    )
    condition_rows = {}
    for condition_id in expected_ids:
        declared = condition_from_id(manifest, condition_id)
        result = training.get(condition_id)
        condition_rows[condition_id] = {
            **declared,
            "status": "SUCCEEDED" if result else "PENDING",
            "paired_test_change": result.get("paired_test_change") if result else None,
            "action_protocol_validity_rate": (
                result.get("action_protocol_validity_rate") if result else None
            ),
            "external": (
                {
                    "exact_successes": external[condition_id].get("exact_successes"),
                    "paired_change_vs_base": external[condition_id].get("paired_change_vs_base"),
                }
                if condition_id in external
                else None
            ),
        }
    execution_rows = []
    for execution_id, receipt in sorted(receipts.items()):
        result = receipt.get("result")
        study = result.get("study") if isinstance(result, dict) else None
        if not (
            (isinstance(study, dict) and study.get("study_id") == STUDY_ID)
            or (isinstance(result, dict) and result.get("workload") == EXTERNAL_WORKLOAD)
        ):
            continue
        execution_rows.append(
            {
                "execution_id": execution_id,
                "provider_handle": receipt.get("provider_handle"),
                "gpu_id": (
                    receipt.get("resource_profile", {}).get("gpu_id")
                    if isinstance(receipt.get("resource_profile"), dict)
                    else None
                ),
                "estimated_cost_usd": exact_cost(receipt),
                "started_at": receipt.get("started_at"),
                "completed_at": receipt.get("completed_at"),
                "teardown_confirmed": receipt.get("teardown_confirmed"),
            }
        )
    overall_status = (
        "PASS" if all_complete and required_passed else "FAIL" if all_complete else "INCOMPLETE"
    )
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "report_id": REPORT_ID,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "overall_status": overall_status,
        "freeze": {
            "freeze_tag": "rl-revision-31",
            "frozen_workload_revision": TRAINING_WORKLOAD_REVISION,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "external_pack_id": "revision31-post-freeze-external-pack@1",
        },
        "conditions": condition_rows,
        "external_adapter_count": len(external),
        "decisions": gates,
        "executions": execution_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    report = build_report(repository_root)
    if report["overall_status"] == "INCOMPLETE" and not arguments.allow_incomplete:
        raise SystemExit("revision-31 evidence is incomplete")
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
