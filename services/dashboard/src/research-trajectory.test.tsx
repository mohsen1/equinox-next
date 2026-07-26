import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  buildTrajectorySteps,
  TrajectoryOutline,
} from "./pages/research-trajectory";
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
});
