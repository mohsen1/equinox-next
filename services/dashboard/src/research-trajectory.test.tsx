import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  buildTrajectorySteps,
  TrajectoryOutline,
} from "./pages/research-trajectory";
import { BranchOutline, buildBranchFlow } from "./pages/research-branches";
import type { ResearchTrajectory } from "./types";

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
      teacher_fallback: false,
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
  teacher_update_count: 2,
  informative_group_rate: 0.2,
  teacher_fallback_rate: 0.1,
  total_sampled_completions: 960,
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
});
