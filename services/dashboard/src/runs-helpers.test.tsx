import { describe, expect, it } from "vitest";

import {
  complexityForLevel,
  formatDuration,
  formatRateInterval,
  intervalValue,
  observerStages,
  phasePercentage,
  resultItems,
  runPercentage,
} from "./runs-helpers";
import type { ResearchComputeExecution } from "./types";

function execution(
  progress: ResearchComputeExecution["progress"],
): ResearchComputeExecution {
  return {
    execution_id: "run-test",
    name: "Repository repair",
    workload_id: "repository-repair",
    model_id: "Qwen/Qwen2.5-Coder-3B-Instruct",
    branch_width: 4,
    complexity_strategy: "adaptive",
    status: "RUNNING",
    provider_name: "RunPod",
    provider_handle: "runpod://pods/test",
    resource_profile: {},
    progress,
    proof_id: null,
    receipt_digest: null,
    started_at: "2026-07-27T12:00:00Z",
    updated_at: "2026-07-27T12:10:00Z",
    completed_at: null,
    teardown_confirmed: false,
  };
}

describe("run observer formatting", () => {
  it("normalizes rounded duration rollover", () => {
    expect(formatDuration(119.7)).toBe("2m 0s");
    expect(formatDuration(59.4)).toBe("59 sec");
    expect(formatDuration(12_822)).toBe("3h 33m");
  });

  it("validates confidence intervals and includes sample size", () => {
    expect(intervalValue([0.42, 0.91])).toEqual([0.42, 0.91]);
    expect(intervalValue([0.42, Number.NaN])).toBeNull();
    expect(intervalValue([0.42])).toBeNull();
    expect(formatRateInterval(0.75, [0.42, 0.91], 8, "Pending")).toBe(
      "75.0% · 95% CI 42.0–91.0% · n=8",
    );
    expect(formatRateInterval(null, null, null, "Pending")).toBe("Pending");
    expect(formatRateInterval(0.5, null, null, "Pending")).toBe("50.0%");
  });

  it("shows retry, reserve, and partial-evaluation evidence", () => {
    const items = resultItems(
      execution({
        resumed_from_checkpoint: true,
        attempt_count: 2,
        final_evaluation_reserve_exceeded_ceiling: true,
        final_evaluation_partial: true,
      }),
      "INCOMPLETE_FINAL_EVALUATION",
    );
    const values = items.map((item) => `${item.label}: ${item.value}`);

    expect(values).toContain("Attempt: Resumed from checkpoint · 2");
    expect(values).toContain(
      "Crash tail: Wall-clock cost retained · action count unavailable",
    );
    expect(values).toContain(
      "Evaluation reserve: Estimate exceeded ceiling · training stopped",
    );
    expect(values).toContain(
      "Evaluation: Incomplete · workload deadline reached",
    );
    expect(values).toContain("Baseline test: Not reported");
    expect(values).toContain("Final test: Not reported");
  });

  it("qualifies legacy headline rates from a partial evaluation", () => {
    const items = resultItems(
      execution({
        final_evaluation_partial: true,
        initial_exact_rate: 0.25,
        final_exact_rate: 0.5,
        paired_test_change: {
          improved: 4,
          regressed: 0,
          examples: 4,
          mcnemar_exact_p_value: 0.125,
        },
      }),
      "INCOMPLETE_FINAL_EVALUATION",
    );

    expect(items).toContainEqual({
      label: "Baseline test",
      value: "25.0% · partial",
    });
    expect(items).toContainEqual({
      label: "Final test",
      value: "50.0% · partial",
    });
    expect(items.some((item) => item.label === "Test pairs")).toBe(false);
  });

  it("keeps a resumed run in the learning stage", () => {
    expect(observerStages("RUNNING", "container_starting", false, 2)).toEqual([
      expect.objectContaining({ label: "Allocate", state: "complete" }),
      expect.objectContaining({ label: "Prepare", state: "complete" }),
      expect.objectContaining({ label: "Learn", state: "current" }),
      expect.objectContaining({ label: "Finalize", state: "pending" }),
      expect.objectContaining({ label: "Release", state: "pending" }),
    ]);
    expect(phasePercentage("RUNNING", "container_starting", 2)).toBe(48);
  });

  it("does not round a small p-value down to zero", () => {
    const items = resultItems(
      execution({
        paired_test_change: {
          improved: 12,
          regressed: 0,
          examples: 12,
          mcnemar_exact_p_value: 0.00001,
        },
      }),
      "EXPLORATORY_SINGLE_SEED",
    );

    expect(items).toContainEqual(
      expect.objectContaining({
        label: "Test pairs",
        value: "12 improved · 0 regressed · n=12 · p < 0.0001",
      }),
    );
  });

  it("omits paired statistics when no test pair was observed", () => {
    const items = resultItems(
      execution({
        final_evaluation_partial: true,
        paired_test_change: {
          improved: 0,
          regressed: 0,
          examples: 0,
          mcnemar_exact_p_value: 1,
        },
      }),
      "INCOMPLETE_FINAL_EVALUATION",
    );

    expect(items.some((item) => item.label === "Test pairs")).toBe(false);
  });

  it("renders terminal zero-update runs as complete", () => {
    expect(runPercentage("SUCCEEDED", 0, 120, "complete", 1)).toBe(100);
    expect(runPercentage("FAILED", 0, 120, "failed", 1)).toBe(0);
    expect(runPercentage("FAILED", 30, 120, "failed", 1)).toBe(25);
  });

  it("keeps a zero-update final evaluation visibly active", () => {
    expect(runPercentage("RUNNING", 0, 120, "finalizing", 1)).toBe(85);
    expect(runPercentage("RUNNING", 120, 120, "finalizing", 1)).toBe(100);
  });

  it("does not show stale complexity from a different evaluation level", () => {
    const complexity = {
      level: 1,
      file_count: 6,
      fault_count: 1,
      dependency_depth: 2,
      repair_horizon: 10,
    };

    expect(complexityForLevel(3, complexity)).toBeNull();
    expect(complexityForLevel(1, complexity)).toEqual(complexity);
  });

  it("shows a recovered result in the finalize stage", () => {
    expect(observerStages("RUNNING", "complete", false, 1)).toEqual([
      expect.objectContaining({ label: "Allocate", state: "complete" }),
      expect.objectContaining({ label: "Prepare", state: "complete" }),
      expect.objectContaining({ label: "Learn", state: "complete" }),
      expect.objectContaining({ label: "Finalize", state: "current" }),
      expect.objectContaining({ label: "Release", state: "pending" }),
    ]);
    expect(runPercentage("RUNNING", null, 120, "complete", 1)).toBe(92);
  });

  it("attributes a first-attempt boot failure to preparation", () => {
    expect(observerStages("FAILED", "failed", true, 1)).toEqual([
      expect.objectContaining({ label: "Allocate", state: "complete" }),
      expect.objectContaining({ label: "Prepare", state: "failed" }),
      expect.objectContaining({ label: "Learn", state: "pending" }),
      expect.objectContaining({ label: "Finalize", state: "pending" }),
      expect.objectContaining({ label: "Release", state: "complete" }),
    ]);
  });

  it("attributes a failed run with learning evidence to learning", () => {
    expect(observerStages("FAILED", "failed", true, 1, true)).toEqual([
      expect.objectContaining({ label: "Allocate", state: "complete" }),
      expect.objectContaining({ label: "Prepare", state: "complete" }),
      expect.objectContaining({ label: "Learn", state: "failed" }),
      expect.objectContaining({ label: "Finalize", state: "pending" }),
      expect.objectContaining({ label: "Release", state: "complete" }),
    ]);
  });
});
