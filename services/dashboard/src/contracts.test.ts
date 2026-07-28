import { describe, expect, it } from "vitest";
import {
  decodeResearchComputeExecution,
  decodeResearchTrajectoryResponse,
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
});
