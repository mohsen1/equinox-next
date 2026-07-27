"""Aggregate independent RunPod receipts without upgrading single-seed claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

EXPECTED_WORKLOAD = "runpod-repository-repair-loo-reinforce@2"
EXPECTED_OBJECTIVE = "leave-one-out-group-normalized-reinforce@1"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 3:
        raise ValueError("at least three independent optimization seeds are required")
    seeds = [item.get("seed") for item in results]
    if any(not isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("study receipts require distinct integer optimization seeds")
    for result in results:
        if (
            result.get("workload_revision") != EXPECTED_WORKLOAD
            or result.get("objective_id") != EXPECTED_OBJECTIVE
            or result.get("teacher_data_used") is not False
            or result.get("adapter_persisted") is not True
        ):
            raise ValueError("study receipt identity or persistence evidence is invalid")
        observations = list(result.get("final_by_level", {}).values())
        if not observations or any(item.get("split") != "test" for item in observations):
            raise ValueError("study receipt does not contain the final test split")

    gains = [float(item["reward_gain"]) for item in results]
    if not all(math.isfinite(value) for value in gains):
        raise ValueError("study reward gains must be finite")
    mean_gain = statistics.mean(gains)
    standard_deviation = statistics.stdev(gains)
    standard_error = standard_deviation / math.sqrt(len(gains))
    critical_value = 4.303 if len(gains) == 3 else 2.776 if len(gains) == 4 else 1.96
    interval = [
        mean_gain - critical_value * standard_error,
        mean_gain + critical_value * standard_error,
    ]
    content = {
        "schema_version": 1,
        "workload_revision": EXPECTED_WORKLOAD,
        "objective_id": EXPECTED_OBJECTIVE,
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "reward_gains": gains,
        "mean_reward_gain": mean_gain,
        "sample_standard_deviation": standard_deviation,
        "mean_reward_gain_95ci": interval,
        "all_runs_observed_branching": all(
            item.get("restored_branching_observed") is True for item in results
        ),
        "all_runs_had_policy_updates": all(
            int(item.get("policy_update_count", 0)) > 0 for item in results
        ),
    }
    return {
        **content,
        "replicated_positive_gain": (
            interval[0] > 0
            and content["all_runs_observed_branching"]
            and content["all_runs_had_policy_updates"]
        ),
        "study_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    arguments = parser.parse_args()
    result = summarize([json.loads(path.read_text(encoding="utf-8")) for path in arguments.results])
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
