import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  buildTrajectorySteps,
  TrajectoryOutline,
} from "./pages/research-trajectory";
import {
  BranchOutline,
  buildBranchFlow,
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
    objective_id: "verified-success-accumulated-retention-policy-gradient@11",
    adapter_revision: "update-5",
    learning_rate: 0.00004,
    policy_loss: 0.12,
    reinforce_loss: 0.11,
    reference_kl: 0.1,
    reference_kl_coefficient: 0.1,
    reference_anchor_scope: "all_accepted_actions_including_greedy_prefix",
    policy_credit_scope: "accepted_actions_from_verified_successful_siblings",
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
    effective_batch_weight: index === 0 ? 0.25 : 0,
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
    steps: [branchStep(2, "edit"), branchStep(3, "finish", true)],
  })),
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
    expect(html).toContain("Verified successes only");
    expect(html).toContain("Batch weight");
    expect(html).toContain("Policy signal</dt><dd>Eligible");
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
});
