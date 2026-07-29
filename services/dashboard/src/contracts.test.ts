import { describe, expect, it } from "vitest";
import {
  decodeResearchComputeExecution,
  decodeResearchStudyReport,
  decodeResearchTrajectoryResponse,
  decodeStudiesResponse,
} from "./contracts";

function k1Execution() {
  return {
    execution_id: "runpod-proof-k1",
    name: "K=1 training · seed 211",
    workload_id: "repository-repair-restored-continuation-post-training",
    model_id: "Qwen/Qwen2.5-Coder-3B-Instruct",
    branch_width: 1,
    complexity_strategy: "adaptive",
    status: "RUNNING",
    provider_name: "RunPod",
    provider_handle: "runpod://pods/k1",
    resource_profile: {},
    progress: {},
    started_at: "2026-07-28T17:00:00Z",
    updated_at: "2026-07-28T17:01:00Z",
    completed_at: null,
    teardown_confirmed: false,
  };
}

describe("research contracts", () => {
  it("accepts a static K=1 ablation execution and trajectory", () => {
    expect(decodeResearchComputeExecution(k1Execution()).branch_width).toBe(1);
    expect(
      decodeResearchTrajectoryResponse({
        execution: k1Execution(),
        trajectory: {
          schema_version: 2,
          branch_width: 1,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [],
          initial_by_level: {},
          final_by_level: {},
        },
      }).trajectory?.branch_width,
    ).toBe(1);
  });

  it("still rejects dynamic or undeclared branch widths", () => {
    expect(() =>
      decodeResearchComputeExecution({
        ...k1Execution(),
        branch_width: 2,
      }),
    ).toThrow("branch_width must be 1 or 4");
  });

  it("accepts explicit missing study evidence without turning it into zero", () => {
    const decoded = decodeResearchStudyReport({
      study_id: "study@1",
      report_id: "study@1/report@1",
      generated_at: "2026-07-29T10:00:00Z",
      overall_status: "FAIL",
      aggregation_policy: "Per-seed evidence",
      report_digest: "sha256:report",
      freeze: {},
      conditions: {
        failed_condition: {
          seed: 211,
          branch_width: 4,
          initial_successes: null,
          final_successes: null,
        },
      },
      decisions: {
        advantage: { status: "FAIL", evidence: {} },
      },
      failure_count: {
        provider_executions: 1,
        operator_attempts: 0,
      },
      executions: [
        {
          execution_id: "execution-1",
          outcome: "FAILED",
          estimated_cost_usd: null,
          teardown_confirmed: true,
        },
      ],
    });

    expect(decoded.conditions.failed_condition.initial_successes).toBeNull();
    expect(decoded.executions[0]?.estimated_cost_usd).toBeNull();
  });

  it("rejects partial study responses at the API boundary", () => {
    expect(() =>
      decodeStudiesResponse({
        items: [
          {
            study_id: "study@1",
            generated_at: null,
            overall_status: "FAIL",
          },
        ],
      }),
    ).toThrow("study 0.generated_at");
    expect(() =>
      decodeResearchStudyReport({
        study_id: "study@1",
        report_id: "study@1/report@1",
        generated_at: "2026-07-29T10:00:00Z",
        overall_status: "FAIL",
      }),
    ).toThrow("study report.aggregation_policy");
  });
});
