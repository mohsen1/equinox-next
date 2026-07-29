import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  buildTrajectorySteps,
  resolveBranchSnapshot,
  TrajectoryOutline,
} from "./pages/research-trajectory";
import {
  BranchOutline,
  buildBranchFlow,
  branchSnapshotLabel,
  defaultBranchActionId,
  isIndependentPrefixSnapshot,
  ResearchBranchWorkspace,
} from "./pages/research-branches";
import type {
  ResearchBranchSnapshot,
  ResearchBranchStep,
  ResearchTrajectory,
} from "./types";

const observation = {
  level: 0,
  examples: 30,
  exact_rate: 1,
  format_rate: 1,
  mean_reward: 1,
  per_domain: {
    sqlite_repair: { exact_rate: 1, mean_reward: 1 },
  },
};

const trajectory: ResearchTrajectory = {
  schema_version: 1,
  branch_width: 4,
  complexity_strategy: "adaptive",
  maximum_level: 3,
  reached_level: 1,
  updates_completed: 40,
  initial_exact_rate: 0.4,
  final_exact_rate: 0.9,
  exact_gain: 0.5,
  stop_reason: "maximum_level_mastered",
  checkpoints: [
    {
      ...observation,
      update: 20,
      informative_group_rate: 0.25,
      mastery_streak: 1,
    },
    {
      ...observation,
      update: 40,
      informative_group_rate: 0.2,
      mastery_streak: 2,
    },
  ],
  promotions: [
    {
      update: 40,
      from_level: 0,
      to_level: 1,
      exact_rate: 1,
      minimum_domain_exact_rate: 1,
      mastery_windows: 2,
    },
  ],
  branch_snapshots: [
    {
      snapshot_id: "update-20-sqlite_repair",
      update: 20,
      level: 0,
      domain: "sqlite_repair",
      prompt: "Repair one row.",
      expected_action: "UPDATE jobs SET state = 'done' WHERE id = 4;",
      best_sibling_index: 1,
      learning_signal: true,
      siblings: [
        {
          index: 0,
          response: "ACTION: UPDATE jobs SET state = 'done';",
          action: "UPDATE jobs SET state = 'done';",
          format_valid: true,
          passed: false,
          reward: 0.4,
          advantage: -1,
          policy_signal: true,
        },
        {
          index: 1,
          response: "ACTION: UPDATE jobs SET state = 'done' WHERE id = 4;",
          action: "UPDATE jobs SET state = 'done' WHERE id = 4;",
          format_valid: true,
          passed: true,
          reward: 1,
          advantage: 1,
          policy_signal: true,
        },
        {
          index: 2,
          response: "invalid",
          action: null,
          format_valid: false,
          passed: false,
          reward: 0,
          advantage: -0.5,
          policy_signal: true,
        },
        {
          index: 3,
          response: "ACTION: SELECT 1;",
          action: "SELECT 1;",
          format_valid: true,
          passed: false,
          reward: 0.1,
          advantage: -0.25,
          policy_signal: true,
        },
      ],
    },
  ],
  initial_by_level: { "0": { ...observation, exact_rate: 0.4 } },
  final_by_level: { "0": { ...observation, exact_rate: 0.9 } },
  policy_update_count: 4,
  informative_group_rate: 0.2,
  total_sampled_completions: 960,
};

function branchStep(
  index: number,
  tool: string,
  terminal = false,
): ResearchBranchStep {
  return {
    step_id: `step-${index}`,
    index,
    tool,
    action: { tool, ...(tool === "read" ? { path: "src/repair.py" } : {}) },
    accepted: true,
    observation: terminal ? '{"passed":1,"failed":0}' : "Accepted",
    state_digest_before: `sha256:before-${index}`,
    state_digest_after: `sha256:after-${index}`,
    verifier_passed: terminal,
    fixed_faults: terminal ? 1 : 0,
    total_faults: 1,
    terminal,
    terminal_reason: terminal ? "solved" : null,
    reward: terminal ? 0.94 : 0,
  };
}

const multiStepSnapshot: ResearchBranchSnapshot = {
  schema_version: 2,
  snapshot_id: "update-5-snapshot-repo",
  update: 5,
  level: 1,
  domain: "micro_repository",
  task_id: "repo-1-test",
  task: {
    description: "Repair the repository.",
    known_failing_tests: ["test_repair"],
    complexity: {
      level: 1,
      file_count: 6,
      fault_count: 1,
      dependency_depth: 2,
      repair_horizon: 10,
    },
  },
  checkpoint: {
    checkpoint_id: "snapshot-repo",
    payload_digest: "sha256:snapshot",
    fidelity: "logical_restore",
    environment_revision: "repository-repair-simulator@1",
    verifier_revision: "repository-repair-hidden-state@1",
    action_protocol_revision: "repository-repair-json-tools@2",
    static_branch_width: 4,
  },
  shared_prefix: {
    policy_generated: true,
    accepted_diagnostic_actions: 2,
    steps: [branchStep(0, "list"), branchStep(1, "read")],
  },
  best_sibling_index: 0,
  learning_signal: true,
  replay: false,
  optimizer_update: {
    applied: true,
    policy_signal_applied: true,
    reference_anchor_applied: true,
    objective_id: "verified-fix-accumulated-retention-policy-gradient@12",
    adapter_revision: "update-5",
    learning_rate: 0.00004,
    policy_loss: 0.12,
    reinforce_loss: 0.11,
    reference_kl: 0.1,
    reference_kl_coefficient: 0.1,
    reference_anchor_scope: "all_accepted_actions_including_greedy_prefix",
    policy_credit_scope: "fault_fixing_edits_from_verified_successful_siblings",
    failed_sibling_policy_weight: 0,
    gradient_norm: 0.34,
    training_examples: 8,
    reference_examples: 24,
    effective_batch_weight: 2.5,
    informative_group_count: 1,
  },
  siblings: Array.from({ length: 4 }, (_, index) => ({
    index,
    sampling_seed: 100 + index,
    passed: index === 0,
    return: index === 0 ? 0.94 : 0,
    advantage: index === 0 ? 1.5 : -0.5,
    policy_signal: index === 0,
    terminal_reason: index === 0 ? "solved" : "finished_with_failures",
    trajectory_digest: `sha256:sibling-${index}`,
    completion_tokens: 24,
    effective_batch_weight: index === 0 ? 1 : 0,
    reward_components: {
      hidden_correctness: index === 0,
      public_verifier_progress: index === 0 ? 1 : 0,
      accepted_action_cost: index === 0 ? -0.03 : -0.02,
      token_cost: 0,
      verifier_submission_cost: 0,
      malformed_action_penalty: 0,
      terminal_aggregate: index === 0 ? 0.97 : 0,
      accepted_action_count: index === 0 ? 6 : 4,
      malformed_action_count: 0,
      verifier_submission_count: 1,
    },
    failure_classification:
      index === 0
        ? []
        : [{ category: "valid_candidate_failure", source: "typed" as const }],
    steps: [
      {
        ...branchStep(2, "edit"),
        policy_signal: index === 0,
        effective_batch_weight: index === 0 ? 1 : 0,
      },
      {
        ...branchStep(3, "finish", true),
        policy_signal: false,
        effective_batch_weight: 0,
      },
    ],
  })),
};

const independentPrefixSnapshot: ResearchBranchSnapshot = {
  ...multiStepSnapshot,
  snapshot_id: "update-5-independent-repo",
  checkpoint: null,
  shared_prefix: null,
  initial_state: {
    state_id: "initial-repo",
    payload_digest: "sha256:initial-repo",
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
  sampled_completion_tokens: 96,
};

describe("research trajectory", () => {
  it("builds a real checkpoint sequence with promotions", () => {
    const steps = buildTrajectorySteps(trajectory);

    expect(steps.map((step) => step.id)).toEqual([
      "baseline",
      "update-20",
      "update-40",
      "final",
    ]);
    expect(steps[2].promotion?.to_level).toBe(1);
  });

  it("provides a keyboard-operable outline equivalent", () => {
    const steps = buildTrajectorySteps(trajectory);
    const html = renderToStaticMarkup(
      <TrajectoryOutline
        steps={steps}
        selectedId="update-40"
        select={() => undefined}
      />,
    );

    expect(html).toContain("<caption>Training checkpoints</caption>");
    expect(html).toContain("Update 40");
    expect(html).toContain("Level 0 → 1");
    expect(html).toContain('class="selected"');
  });

  it("maps one persisted task to four real sibling nodes", () => {
    const snapshot = trajectory.branch_snapshots[0];
    const flow = buildBranchFlow(snapshot, 1);

    expect(flow.nodes).toHaveLength(5);
    expect(flow.edges).toHaveLength(4);
    expect(
      flow.nodes.find((node) => node.data.siblingIndex === 1)?.data,
    ).toMatchObject({ passed: true, best: true, reward: 1 });
    expect(
      flow.edges.find((edge) => edge.target.endsWith("sibling-1"))?.className,
    ).toContain("branch-edge-best");
  });

  it("labels non-mutating eligibility snapshots as branch groups", () => {
    const eligibilitySnapshot: ResearchBranchSnapshot = {
      ...multiStepSnapshot,
      update: 3,
      optimizer_update: null,
    };
    const legacyFlow = buildBranchFlow(
      {
        ...trajectory.branch_snapshots[0],
        update: 3,
        optimizer_update: null,
      },
      0,
    );
    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={eligibilitySnapshot}
        selectedSibling={eligibilitySnapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(branchSnapshotLabel(eligibilitySnapshot)).toBe(
      "Branch group 3 · Level 1",
    );
    expect(legacyFlow.nodes[0]?.data.detail).toBe("Branch group 3 · Level 0");
    expect(html).toContain("Branch group 3 · Level 1");
    expect(html).not.toContain("Update 3 · Level 1");
  });

  it("keeps policy update labels for training snapshots", () => {
    expect(branchSnapshotLabel(multiStepSnapshot)).toBe("Update 5 · Level 1");

    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={multiStepSnapshot}
        selectedSibling={multiStepSnapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(html).toContain("Update 5 · Level 1");
    expect(html).not.toContain("Branch group 5 · Level 1");
  });

  it("shows every optimizer input with its signal role and branch snapshot", () => {
    const signalGroupIds = ["prior-signal", "current-signal"];
    const optimizerInputGroupIds = [
      "prior-signal",
      "prior-reference-anchor",
      "current-signal",
      "current-reference-anchor",
    ];
    const optimizerSnapshot: ResearchBranchSnapshot = {
      ...multiStepSnapshot,
      task_id: "current-signal",
      optimizer_update: {
        ...multiStepSnapshot.optimizer_update!,
        update: 5,
        attempted_policy_update_index: 1,
        policy_signal_group_count: signalGroupIds.length,
        policy_signal_group_ids: signalGroupIds,
        optimizer_input_group_count: optimizerInputGroupIds.length,
        optimizer_input_group_ids: optimizerInputGroupIds,
        optimizer_input_consumed_by_update: 5,
        policy_signal_consumed_by_update: 5,
      },
    };
    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={optimizerSnapshot}
        selectedSibling={optimizerSnapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        policyUpdateLineage={[
          {
            schema_version: 2,
            attempted_policy_update_index: 1,
            update: 5,
            retention_lineage_status: "pending",
            effective_policy_update_count_after_apply: 1,
            retained_policy_update_count_before_validation: 0,
            policy_signal_group_count: signalGroupIds.length,
            policy_signal_group_ids: signalGroupIds,
            optimizer_input_group_count: optimizerInputGroupIds.length,
            optimizer_input_group_ids: optimizerInputGroupIds,
            branch_snapshot_ids: [
              "snapshot-prior-signal",
              "snapshot-prior-reference-anchor",
              optimizerSnapshot.snapshot_id,
              "snapshot-current-reference-anchor",
            ],
          },
        ]}
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(html).toContain("<caption>Optimizer inputs</caption>");
    expect(html).toContain("prior-reference-anchor");
    expect(html).toContain("snapshot-prior-reference-anchor");
    expect(html).toContain("Signal");
    expect(html).toContain("Anchor only");
  });

  it("keeps every consumed branch URL-addressable beyond the old 40-item window", () => {
    const snapshots = Array.from({ length: 64 }, (_, index) => ({
      ...multiStepSnapshot,
      snapshot_id: `update-${Math.floor(index / 4) + 1}-snapshot-${index}`,
      task_id: `group-${index}`,
      update: Math.floor(index / 4) + 1,
      collection_index: (index % 4) + 1,
      collection_count: 4,
      replay: index === 3,
      optimizer_update: {
        ...multiStepSnapshot.optimizer_update!,
        update: Math.floor(index / 4) + 1,
        attempted_policy_update_index: Math.floor(index / 4) + 1,
        policy_signal_group_ids: [`group-${index}`],
      },
    }));
    const selected = resolveBranchSnapshot(snapshots, "update-1-snapshot-0");

    expect(selected?.task_id).toBe("group-0");
    expect(branchSnapshotLabel(snapshots[3]!)).toBe(
      "Update 1 · Group 4/4 · Level 1 · Replay",
    );
    const graph = buildBranchFlow(selected!, 0, "sibling-0-3");
    const outline = renderToStaticMarkup(
      <BranchOutline
        snapshot={selected!}
        selectedIndex={0}
        selectedActionId="sibling-0-3"
        select={() => undefined}
        selectAction={() => undefined}
      />,
    );
    expect(
      graph.nodes.some((node) => node.data.actionId === "sibling-0-3"),
    ).toBe(true);
    expect(outline).toContain("Sibling 1");
    expect(resolveBranchSnapshot(snapshots, "missing")?.task_id).toBe(
      "group-63",
    );
  });

  it("renders an accessible K=4 branch outline", () => {
    const html = renderToStaticMarkup(
      <BranchOutline
        snapshot={trajectory.branch_snapshots[0]}
        selectedIndex={1}
        select={() => undefined}
      />,
    );

    expect(html).toContain("<caption>K=4 sibling actions</caption>");
    expect(html.match(/Sibling [1-4]/g)).toHaveLength(4);
    expect(html).toContain("Sibling 2 · Best");
    expect(html).toContain("Passed");
  });

  it("renders one shared prefix and four multi-step continuation lanes", () => {
    const flow = buildBranchFlow(multiStepSnapshot, 0, "sibling-0-3");

    expect(flow.nodes).toHaveLength(12);
    expect(flow.edges).toHaveLength(11);
    expect(flow.nodes.filter((node) => node.data.prefix)).toHaveLength(2);
    expect(
      flow.nodes.filter((node) => node.data.siblingIndex === 0),
    ).toHaveLength(2);
    expect(
      flow.nodes.find((node) => node.data.actionId === "sibling-0-3")?.selected,
    ).toBe(true);
  });

  it("provides an outline for every prefix and continuation action", () => {
    const html = renderToStaticMarkup(
      <BranchOutline
        snapshot={multiStepSnapshot}
        selectedIndex={0}
        selectedActionId="sibling-0-3"
        select={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(html).toContain(
      "<caption>Shared prefix and K=4 continuation steps</caption>",
    );
    expect(html.match(/>Shared</g)).toHaveLength(2);
    expect(html.match(/Sibling [1-4]/g)).toHaveLength(8);
    expect(html).toContain('class="selected"');
  });

  it("renders four full independent trajectories from one matched initial state", () => {
    const flow = buildBranchFlow(independentPrefixSnapshot, 2, "sibling-2-3");
    const html = renderToStaticMarkup(
      <BranchOutline
        snapshot={independentPrefixSnapshot}
        selectedIndex={2}
        selectedActionId="sibling-2-3"
        select={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(isIndependentPrefixSnapshot(independentPrefixSnapshot)).toBe(true);
    expect(flow.nodes).toHaveLength(9);
    expect(flow.edges).toHaveLength(8);
    expect(flow.nodes.some((node) => node.type === "checkpoint")).toBe(false);
    expect(flow.nodes.find((node) => node.type === "initial")?.data.title).toBe(
      "Matched initial state",
    );
    expect(
      flow.nodes.filter((node) => node.data.siblingIndex === 2),
    ).toHaveLength(2);
    expect(
      flow.nodes.find((node) => node.data.actionId === "sibling-2-3")?.selected,
    ).toBe(true);
    expect(html).toContain(
      "<caption>K=4 independent trajectories from one matched initial state</caption>",
    );
    expect(html.match(/Trajectory [1-4]/g)).toHaveLength(8);
    expect(html).not.toContain(">Shared<");
  });

  it("refuses to render an untyped independent rollout as legacy evidence", () => {
    const untypedIndependentSnapshot: ResearchBranchSnapshot = {
      ...independentPrefixSnapshot,
      comparison_condition: undefined,
    };

    expect(() =>
      buildBranchFlow(untypedIndependentSnapshot, 0, "sibling-0-2"),
    ).toThrow("requires its typed comparison condition");
    expect(() =>
      renderToStaticMarkup(
        <BranchOutline
          snapshot={untypedIndependentSnapshot}
          selectedIndex={0}
          selectedActionId="sibling-0-2"
          select={() => undefined}
          selectAction={() => undefined}
        />,
      ),
    ).toThrow("requires its typed comparison condition");
  });

  it("keeps independent lane selection URL-addressable at action granularity", () => {
    const sibling = independentPrefixSnapshot.siblings[3]!;
    expect(defaultBranchActionId(independentPrefixSnapshot, sibling)).toBe(
      "sibling-3-2",
    );

    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={independentPrefixSnapshot}
        selectedSibling={sibling}
        selectedActionId="sibling-3-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );
    expect(html).toContain(
      'aria-label="Matched initial state and 4 independent trajectories"',
    );
    expect(html).toContain("Trajectory 4 · step 2");
    expect(html).toContain('class="selected"');
    expect(html).not.toContain("Checkpoint");
  });

  it("centers and labels the K=1 ablation as one continuation", () => {
    const k1Snapshot: ResearchBranchSnapshot = {
      ...multiStepSnapshot,
      checkpoint: {
        ...multiStepSnapshot.checkpoint!,
        static_branch_width: 1,
      },
      siblings: [multiStepSnapshot.siblings[0]],
    };
    const flow = buildBranchFlow(k1Snapshot, 0, "sibling-0-3");
    const prefixNode = flow.nodes.find((node) => node.data.prefix);
    const continuationNode = flow.nodes.find(
      (node) => node.data.siblingIndex === 0,
    );
    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={k1Snapshot}
        selectedSibling={k1Snapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(prefixNode?.position.x).toBe(0);
    expect(continuationNode?.position.x).toBe(0);
    expect(html).toContain(
      'aria-label="Shared prefix and 1 restored continuation"',
    );
    expect(html).toContain(
      "<caption>Shared prefix and K=1 continuation steps</caption>",
    );
  });

  it("exposes reward and optimizer provenance in the branch inspector", () => {
    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={multiStepSnapshot}
        selectedSibling={multiStepSnapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(html).toContain("Reward · 0.970");
    expect(html).toContain("Hidden correctness");
    expect(html).toContain("Optimizer · applied");
    expect(html).toContain("update-5");
    expect(html).toContain("Reference KL");
    expect(html).toContain("8 / 24");
    expect(html).toContain("Verified fault-fixing edits");
    expect(html).toContain("Batch weight");
    expect(html).toContain("Policy signal</dt><dd>None");

    const creditedEditHtml = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={multiStepSnapshot}
        selectedSibling={multiStepSnapshot.siblings[0]}
        selectedActionId="sibling-0-2"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );
    expect(creditedEditHtml).toContain("Policy signal</dt><dd>Eligible");
    expect(creditedEditHtml).toContain("Batch weight</dt><dd>+1.000");
  });

  it("distinguishes a restorative anchor-only optimizer step", () => {
    const html = renderToStaticMarkup(
      <ResearchBranchWorkspace
        snapshot={{
          ...multiStepSnapshot,
          optimizer_update: {
            ...multiStepSnapshot.optimizer_update!,
            policy_signal_applied: false,
            training_examples: 0,
          },
        }}
        selectedSibling={multiStepSnapshot.siblings[0]}
        selectedActionId="sibling-0-3"
        view="outline"
        selectSibling={() => undefined}
        selectAction={() => undefined}
      />,
    );

    expect(html).toContain("Optimizer · anchor only");
    expect(html).toContain("0 / 24");
  });

  it.each([
    {
      status: "pending" as const,
      expected: "Optimizer · attempted · pending validation",
      resolutionUpdate: undefined,
    },
    {
      status: "retained" as const,
      expected: "Optimizer · retained",
      resolutionUpdate: 5,
    },
    {
      status: "rolled_back" as const,
      expected: "Optimizer · rolled back",
      resolutionUpdate: 6,
    },
  ])(
    "shows $status transactional optimizer lineage",
    ({ status, expected, resolutionUpdate }) => {
      const html = renderToStaticMarkup(
        <ResearchBranchWorkspace
          snapshot={{
            ...multiStepSnapshot,
            optimizer_update: {
              ...multiStepSnapshot.optimizer_update!,
              retention_lineage_status: status,
              retention_resolution_update: resolutionUpdate,
              retention_resolution_reason:
                status === "rolled_back"
                  ? "retention_guard_regression"
                  : "retention_guard_improvement",
            },
          }}
          selectedSibling={multiStepSnapshot.siblings[0]}
          selectedActionId="sibling-0-3"
          view="outline"
          selectSibling={() => undefined}
          selectAction={() => undefined}
        />,
      );

      expect(html).toContain(expected);
      expect(html).toContain(
        status === "rolled_back" ? "Rolled Back" : friendlyLineage(status),
      );
      if (resolutionUpdate !== undefined) {
        expect(html).toContain(`at update ${resolutionUpdate}`);
      }
    },
  );
});

function friendlyLineage(status: "pending" | "retained"): string {
  return status === "pending" ? "Pending" : "Retained";
}
