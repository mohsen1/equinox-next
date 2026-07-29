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
    proof_id: null,
    receipt_digest: null,
    failure_receipt_digest: null,
    started_at: "2026-07-28T17:00:00Z",
    updated_at: "2026-07-28T17:01:00Z",
    completed_at: null,
    teardown_confirmed: false,
  };
}

function independentSnapshot() {
  return {
    snapshot_id: "independent-snapshot",
    task_id: "independent-task",
    checkpoint: null,
    shared_prefix: null,
    initial_state: {
      state_id: "initial-task",
      payload_digest: "sha256:initial",
      fidelity: "logical_restore",
      role: "matched_task_initial_state_not_decision_checkpoint",
    },
    rollout_topology: {
      revision: "independent-prefix-k4@1",
      static_group_width: 4,
      independent_model_generated_prefixes: true,
      shared_model_generated_prefix: false,
      sibling_group_relative_credit: true,
      initial_state_matching: "same_task_initial_state",
    },
    comparison_condition: {
      schema_version: 1,
      condition_id: "independent_prefix_grpo_k4",
      short_label: "Independent · K=4",
      prefix_topology: "independent",
      branch_width: 4,
      group_credit: "sibling_relative",
      shared_prefix: false,
    },
    sampled_completion_tokens: 28,
    siblings: Array.from({ length: 4 }, (_, index) => ({
      index,
      completion_tokens: 7,
      steps: [{ step_id: `trajectory-${index}-step-0` }],
    })),
  };
}

function decodeIndependentSnapshot(snapshot: unknown) {
  return decodeResearchTrajectoryResponse({
    execution: { ...k1Execution(), branch_width: 4 },
    trajectory: {
      schema_version: 2,
      branch_width: 4,
      complexity_strategy: "adaptive",
      checkpoints: [],
      promotions: [],
      branch_snapshots: [snapshot],
      branch_evidence_complete: true,
      branch_evidence_group_count: 1,
      total_sampled_completion_tokens: 28,
      discarded_sampled_completion_tokens: 0,
      initial_by_level: {},
      final_by_level: {},
    },
  });
}

function optimizerEvidence(
  taskId: string,
  signalGroupIds: string[],
  optimizerInputGroupIds: string[],
) {
  return {
    update: 2,
    applied: true,
    policy_signal_applied: true,
    attempted_policy_update_index: 1,
    optimizer_input_group_count: optimizerInputGroupIds.length,
    optimizer_input_group_ids: optimizerInputGroupIds,
    optimizer_input_consumed_by_update: 2,
    policy_signal_group_count: signalGroupIds.length,
    policy_signal_group_ids: signalGroupIds,
    ...(signalGroupIds.includes(taskId)
      ? { policy_signal_consumed_by_update: 2 }
      : {}),
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

  it("accepts exact independent-prefix trajectory evidence", () => {
    const execution = { ...k1Execution(), branch_width: 4 };
    const trajectory = decodeResearchTrajectoryResponse({
      execution,
      trajectory: {
        schema_version: 2,
        branch_width: 4,
        complexity_strategy: "adaptive",
        checkpoints: [],
        promotions: [],
        branch_snapshots: [independentSnapshot()],
        branch_evidence_complete: true,
        branch_evidence_group_count: 1,
        total_sampled_completion_tokens: 28,
        discarded_sampled_completion_tokens: 0,
        initial_by_level: {},
        final_by_level: {},
      },
    }).trajectory;

    expect(
      trajectory?.branch_snapshots[0]?.comparison_condition?.prefix_topology,
    ).toBe("independent");
    expect(trajectory?.branch_snapshots[0]?.shared_prefix).toBeNull();
  });

  it("rejects independent-prefix evidence with incomplete token accounting", () => {
    const snapshot = independentSnapshot();
    snapshot.siblings[3]!.completion_tokens = 6;

    expect(() =>
      decodeResearchTrajectoryResponse({
        execution: { ...k1Execution(), branch_width: 4 },
        trajectory: {
          schema_version: 2,
          branch_width: 4,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [snapshot],
          branch_evidence_complete: true,
          branch_evidence_group_count: 1,
          total_sampled_completion_tokens: 28,
          discarded_sampled_completion_tokens: 0,
          initial_by_level: {},
          final_by_level: {},
        },
      }),
    ).toThrow("independent completion tokens do not reconcile");

    const fractional = independentSnapshot();
    fractional.siblings[0]!.completion_tokens = 6.5;
    expect(() =>
      decodeResearchTrajectoryResponse({
        execution: { ...k1Execution(), branch_width: 4 },
        trajectory: {
          schema_version: 2,
          branch_width: 4,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [fractional],
          branch_evidence_complete: true,
          branch_evidence_group_count: 1,
          total_sampled_completion_tokens: 27.5,
          discarded_sampled_completion_tokens: 0,
          initial_by_level: {},
          final_by_level: {},
        },
      }),
    ).toThrow("expected integer");
  });

  it("rejects untyped or role-tampered independent rollout evidence", () => {
    const { comparison_condition: _comparisonCondition, ...withoutCondition } =
      independentSnapshot();
    expect(() => decodeIndependentSnapshot(withoutCondition)).toThrow(
      "independent rollout requires a typed comparison condition",
    );

    const wrongInitialRole = independentSnapshot();
    wrongInitialRole.initial_state.role = "decision_checkpoint";
    expect(() => decodeIndependentSnapshot(wrongInitialRole)).toThrow(
      "not a faithful independent-prefix K=4 snapshot",
    );
  });

  it("validates reconstructable policy update lineage", () => {
    const trajectory = decodeResearchTrajectoryResponse({
      execution: k1Execution(),
      trajectory: {
        schema_version: 2,
        branch_width: 1,
        complexity_strategy: "adaptive",
        checkpoints: [],
        promotions: [],
        branch_snapshots: [
          {
            snapshot_id: "snapshot-a",
            task_id: "group-a",
            sampled_completion_tokens: 10,
            optimizer_update: optimizerEvidence(
              "group-a",
              ["group-a", "group-b"],
              ["group-a", "group-b", "anchor-c"],
            ),
          },
          {
            snapshot_id: "snapshot-b",
            task_id: "group-b",
            sampled_completion_tokens: 20,
            optimizer_update: optimizerEvidence(
              "group-b",
              ["group-a", "group-b"],
              ["group-a", "group-b", "anchor-c"],
            ),
          },
          {
            snapshot_id: "snapshot-c",
            task_id: "anchor-c",
            sampled_completion_tokens: 30,
            optimizer_update: optimizerEvidence(
              "anchor-c",
              ["group-a", "group-b"],
              ["group-a", "group-b", "anchor-c"],
            ),
          },
        ],
        branch_evidence_complete: true,
        branch_evidence_group_count: 3,
        total_sampled_completion_tokens: 65,
        discarded_sampled_completion_tokens: 5,
        policy_update_lineage: [
          {
            schema_version: 2,
            attempted_policy_update_index: 1,
            update: 2,
            policy_signal_group_count: 2,
            policy_signal_group_ids: ["group-a", "group-b"],
            optimizer_input_group_count: 3,
            optimizer_input_group_ids: ["group-a", "group-b", "anchor-c"],
            branch_snapshot_ids: ["snapshot-a", "snapshot-b", "snapshot-c"],
          },
        ],
        initial_by_level: {},
        final_by_level: {},
      },
    }).trajectory;

    expect(
      trajectory?.policy_update_lineage?.[0]?.policy_signal_group_ids,
    ).toEqual(["group-a", "group-b"]);
    expect(trajectory?.total_sampled_completion_tokens).toBe(65);
  });

  it("rejects policy lineage whose consumed group was not persisted", () => {
    expect(() =>
      decodeResearchTrajectoryResponse({
        execution: k1Execution(),
        trajectory: {
          schema_version: 2,
          branch_width: 4,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [{ snapshot_id: "snapshot-a", task_id: "group-a" }],
          branch_evidence_complete: true,
          branch_evidence_group_count: 1,
          policy_update_lineage: [
            {
              attempted_policy_update_index: 1,
              update: 2,
              policy_signal_group_ids: ["group-a", "missing-group"],
              optimizer_input_group_ids: ["group-a"],
              branch_snapshot_ids: ["snapshot-a"],
            },
          ],
          initial_by_level: {},
          final_by_level: {},
        },
      }),
    ).toThrow("signal group has no branch evidence");
  });

  it("rejects optimizer inputs that do not exactly map to branch snapshots", () => {
    expect(() =>
      decodeResearchTrajectoryResponse({
        execution: k1Execution(),
        trajectory: {
          schema_version: 2,
          branch_width: 1,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [
            { snapshot_id: "snapshot-a", task_id: "group-a" },
            { snapshot_id: "snapshot-b", task_id: "anchor-b" },
          ],
          policy_update_lineage: [
            {
              schema_version: 1,
              attempted_policy_update_index: 1,
              update: 2,
              policy_signal_group_ids: ["group-a"],
              optimizer_input_group_ids: ["group-a", "anchor-b"],
              branch_snapshot_ids: ["snapshot-b", "snapshot-a"],
            },
          ],
          initial_by_level: {},
          final_by_level: {},
        },
      }),
    ).toThrow("do not exactly match branch snapshots");
  });

  it("rejects schema-2 lineage that omits a consumed anchor binding", () => {
    expect(() =>
      decodeResearchTrajectoryResponse({
        execution: k1Execution(),
        trajectory: {
          schema_version: 2,
          branch_width: 1,
          complexity_strategy: "adaptive",
          checkpoints: [],
          promotions: [],
          branch_snapshots: [
            {
              snapshot_id: "snapshot-a",
              task_id: "group-a",
              optimizer_update: optimizerEvidence(
                "group-a",
                ["group-a"],
                ["group-a", "anchor-b"],
              ),
            },
            {
              snapshot_id: "snapshot-b",
              task_id: "anchor-b",
              optimizer_update: null,
            },
          ],
          policy_update_lineage: [
            {
              schema_version: 2,
              attempted_policy_update_index: 1,
              update: 2,
              policy_signal_group_count: 1,
              policy_signal_group_ids: ["group-a"],
              optimizer_input_group_count: 2,
              optimizer_input_group_ids: ["group-a", "anchor-b"],
              branch_snapshot_ids: ["snapshot-a", "snapshot-b"],
            },
          ],
          initial_by_level: {},
          final_by_level: {},
        },
      }),
    ).toThrow("not exactly bound to branch snapshots");
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
