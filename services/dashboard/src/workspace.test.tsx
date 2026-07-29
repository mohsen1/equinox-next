import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ProofDetailPage, ProofsPage } from "./pages/proofs";
import { ResearchRunPage, RunsPage } from "./pages/runs";
import { BrowserRouter, Route, Routes } from "./router";

function response(body: unknown) {
  return {
    ok: true,
    json: async () => body,
  };
}

function largerModelEligibilityExecution(eligible: boolean) {
  const timestamp = new Date().toISOString();
  return {
    execution_id: `runpod-proof-larger-model-${eligible ? "eligible" : "ineligible"}`,
    name: "Qwen 7B eligibility screen",
    workload_id: "repository-repair-larger-model-eligibility",
    model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
    branch_width: 4,
    complexity_strategy: "adaptive",
    status: "SUCCEEDED",
    provider_name: "RunPod",
    provider_handle: "runpod://pods/larger-model-screen",
    resource_profile: {
      gpu_id: "NVIDIA L40",
      cloud_type: "SECURE",
      hourly_cost_usd: 0.69,
      maximum_hourly_cost_usd: 1,
    },
    progress: {
      phase: "complete",
      screen_completed: true,
      eligible,
      larger_model_profile_id: "qwen25-coder-7b-l40@1",
      model_revision: "c03e6d358207e414f1eca0bb1891e29f1db0e242",
      checkpoint_admission_rate: 0.875,
      informative_group_rate: 0.25,
      peak_cuda_memory_bytes: 19_750_000_000,
      policy_mutation_verified: true,
      policy_mutation_enabled: false,
      elapsed_seconds: 521,
    },
    proof_id: null,
    receipt_digest: "sha256:eligibility",
    started_at: timestamp,
    updated_at: timestamp,
    completed_at: timestamp,
    teardown_confirmed: true,
  };
}

describe("Runs and Proofs workspaces", () => {
  let container: HTMLDivElement;
  let root: Root | undefined;

  beforeEach(() => {
    Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
    container = document.createElement("div");
    document.body.append(container);
  });

  afterEach(async () => {
    if (root) await act(async () => root?.unmount());
    container.remove();
    vi.unstubAllGlobals();
  });

  it("keeps Runs focused on name, status, progress, GPU, cost, and recency", async () => {
    const updatedAt = new Date(Date.now() - 4 * 60 * 1_000).toISOString();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          items: [],
          research_items: [
            {
              execution_id: "runpod-proof-test",
              name: "Repository repair post-training",
              model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
              branch_width: 4,
              complexity_strategy: "adaptive",
              status: "RUNNING",
              provider_name: "RunPod",
              provider_handle: "runpod://pods/test",
              resource_profile: {},
              allocated_gpu: "NVIDIA A40",
              cost: {
                total_usd: 0.11,
                estimated: true,
                hourly_rate_usd: 0.44,
                elapsed_seconds: 900,
              },
              progress: {
                update: 12,
                maximum_updates: 80,
                current_level: 1,
                maximum_level: 3,
              },
              proof_id: null,
              receipt_digest: null,
              started_at: updatedAt,
              updated_at: updatedAt,
              completed_at: null,
              teardown_confirmed: false,
            },
          ],
        }),
      ),
    );
    window.history.replaceState(null, "", "/runs");
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <RunsPage />
        </BrowserRouter>,
      );
      await Promise.resolve();
    });

    expect(container.textContent).not.toContain("active ·");
    expect(container.textContent).not.toContain("total");
    expect(container.textContent).toContain("Repository repair post-training");
    expect(container.textContent).toContain("Running");
    expect(container.textContent).toContain("Update 12 of 80");
    expect(container.textContent).toContain("NVIDIA A40");
    expect(container.textContent).toContain("$0.11 est.");
    expect(container.textContent).toContain("K=4 · adaptive");
    const time = container.querySelector("time");
    expect(time?.textContent).toBe("4 min ago");
    expect(time?.getAttribute("aria-label")).not.toBe(time?.textContent);
  });

  it("shows validation history, protocol health, and provider reserves", async () => {
    const timestamp = new Date().toISOString();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          execution_id: "runpod-proof-observer",
          name: "Repository repair post-training",
          workload_id: "repository-repair",
          model_id: "Qwen/Qwen2.5-Coder-3B-Instruct",
          branch_width: 4,
          complexity_strategy: "adaptive",
          status: "RUNNING",
          provider_name: "RunPod",
          provider_handle: "runpod://pods/observer",
          resource_profile: {
            gpu_id: "NVIDIA A40",
            cloud_type: "SECURE",
            hourly_cost_usd: 0.44,
            maximum_hourly_cost_usd: 0.5,
          },
          progress: {
            phase: "training",
            update: 6,
            maximum_updates: 120,
            policy_update_count: 4,
            current_level: 0,
            maximum_level: 3,
            curriculum_decision: {
              frontier_probe_level_used: 1,
              next_frontier_probe_level: 2,
              frontier_probe_decision: "all_siblings_solved_raise_probe",
            },
            exact_rate: 0.75,
            exact_rate_95ci: [0.40927, 0.928522],
            validation_examples: 8,
            evaluation_split: "validation",
            informative_group_rate: 0.5,
            action_protocol_validity_rate: 0.99,
            recent_malformed_action_rate: 0.01,
            total_sampled_actions: 214,
            elapsed_seconds: 681,
            training_remaining_seconds: 750,
            final_evaluation_reserve_seconds: 2700,
            active_complexity: {
              level: 0,
              file_count: 4,
              fault_count: 1,
              dependency_depth: 1,
              repair_horizon: 8,
            },
            baseline_validation: {
              level: 0,
              examples: 8,
              exact_successes: 5,
              exact_rate: 0.625,
              exact_rate_95ci: [0.305738, 0.863158],
            },
            best_validation: {
              update: 5,
              level: 0,
              exact_successes: 6,
              exact_rate: 0.75,
              exact_rate_95ci: [0.40927, 0.928522],
            },
            validation_history: [
              {
                update: 5,
                level: 0,
                examples: 8,
                exact_successes: 6,
                exact_rate: 0.75,
                exact_rate_95ci: [0.40927, 0.928522],
                mastery_streak: 0,
                fixed_guard_levels: [0, 1],
                fixed_guard_paired_change: {
                  examples: 16,
                  improved: 1,
                  regressed: 0,
                  unchanged: 15,
                  net_improved: 1,
                  mcnemar_exact_p_value: 1,
                },
                curriculum_paired_change: {
                  examples: 8,
                  improved: 0,
                  regressed: 1,
                  unchanged: 7,
                  net_improved: -1,
                  mcnemar_exact_p_value: 1,
                },
              },
            ],
          },
          proof_id: null,
          receipt_digest: null,
          started_at: timestamp,
          updated_at: timestamp,
          completed_at: null,
          teardown_confirmed: false,
        }),
      ),
    );
    window.history.replaceState(
      null,
      "",
      "/runs/research/runpod-proof-observer",
    );
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <Routes>
            <Route
              path="/runs/research/:executionId"
              element={<ResearchRunPage />}
            />
          </Routes>
        </BrowserRouter>,
      );
      await Promise.resolve();
    });

    expect(container.textContent).toContain("6 / 120 · 4 policy");
    expect(container.textContent).toContain("Level 1 → 2 · 4/4 solved");
    expect(container.textContent).toContain("+12.5 pts vs baseline");
    expect(container.textContent).toContain("99.0% valid");
    expect(container.textContent).toContain("1.0% malformed recent");
    expect(container.textContent).toContain("12m 30s training");
    expect(container.textContent).toContain("45m 0s evaluation");
    expect(container.textContent).toContain(
      "Fixed 0, 1 · +1 / −0 · rotating +0 / −1",
    );
    expect(container.textContent).toContain("Best retained");
  });

  it.each([
    { eligible: true, decision: "Eligible", tone: "positive" },
    { eligible: false, decision: "Ineligible", tone: "negative" },
  ])(
    "shows the larger-model screen as a concise $decision decision",
    async ({ eligible, decision, tone }) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => response(largerModelEligibilityExecution(eligible))),
      );
      window.history.replaceState(
        null,
        "",
        `/runs/research/runpod-proof-larger-model-${eligible ? "eligible" : "ineligible"}`,
      );
      root = createRoot(container);

      await act(async () => {
        root?.render(
          <BrowserRouter>
            <Routes>
              <Route
                path="/runs/research/:executionId"
                element={<ResearchRunPage />}
              />
            </Routes>
          </BrowserRouter>,
        );
        await Promise.resolve();
      });

      const eligibilitySection = [
        ...container.querySelectorAll(".section"),
      ].find(
        (section) => section.querySelector("h2")?.textContent === "Eligibility",
      );
      const headings = [...container.querySelectorAll("h2")].map(
        (heading) => heading.textContent,
      );

      expect(eligibilitySection).toBeDefined();
      expect(eligibilitySection?.querySelector(".status")?.textContent).toBe(
        decision,
      );
      expect(
        eligibilitySection?.querySelector(".status")?.classList.contains(tone),
      ).toBe(true);
      expect(eligibilitySection?.textContent).toContain(
        "qwen25-coder-7b-l40@1",
      );
      expect(
        eligibilitySection?.querySelector(
          '[title="c03e6d358207e414f1eca0bb1891e29f1db0e242"]',
        ),
      ).not.toBeNull();
      expect(eligibilitySection?.textContent).toContain(
        "Checkpoint admission87.5%",
      );
      expect(eligibilitySection?.textContent).toContain(
        "Informative branching25.0%",
      );
      expect(eligibilitySection?.textContent).toContain(
        "Peak GPU memory19.8 GB",
      );
      expect(eligibilitySection?.textContent).toContain(
        "Policy mutationNone · verified",
      );
      expect(container.querySelector(".observer-stage")?.textContent).toContain(
        "ScreenLoad · sample · admit",
      );
      expect(headings).not.toContain("Progress");
      expect(headings).not.toContain("Validation");
      expect(headings).not.toContain("Result");
    },
  );

  it("links each proof row to focused proof evidence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          items: [
            {
              proof_id: "research_proof_test",
              execution_id: "runpod-proof-test",
              run_name: "Repository repair post-training",
              completed_at: new Date().toISOString(),
              learning: {
                initial_reward: 0,
                final_reward: 1,
                reward_gain: 1,
                hypothesis_passed: true,
              },
              hardware: {
                provider: "RunPod",
                gpu: "NVIDIA A40",
                hourly_rate_usd: 0.44,
              },
              cost: {
                total_usd: 0.11,
                estimated: true,
                hourly_rate_usd: 0.44,
                elapsed_seconds: 900,
              },
              teardown_confirmed: true,
            },
          ],
        }),
      ),
    );
    window.history.replaceState(null, "", "/proofs");
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <ProofsPage />
        </BrowserRouter>,
      );
      await Promise.resolve();
    });

    expect(container.querySelector(".proof-row")?.getAttribute("href")).toBe(
      "/proofs/research_proof_test",
    );
    expect(container.textContent).toContain("NVIDIA A40");
    expect(container.textContent).toContain("Released");
  });

  it("links proof evidence back to the run and trajectory", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          proof_id: "research_proof_test",
          execution_id: "runpod-proof-test",
          run_name: "Repository repair post-training",
          started_at: "2026-07-26T21:00:00Z",
          completed_at: "2026-07-26T21:15:00Z",
          runtime_seconds: 900,
          learning: {
            initial_reward: 0,
            final_reward: 1,
            reward_gain: 1,
            hypothesis_passed: true,
          },
          hardware: {
            provider: "RunPod",
            gpu: "NVIDIA A40",
            image: "runpod/pytorch:test",
            cloud_type: "SECURE",
            hourly_rate_usd: 0.44,
          },
          cost: {
            total_usd: 0.11,
            estimated: true,
            hourly_rate_usd: 0.44,
            elapsed_seconds: 900,
          },
          provider: {
            name: "RunPod",
            handle: "runpod://pods/test",
            cli_version: "2.7.2",
          },
          workload: {
            id: "repository-repair",
            model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            branch_width: 4,
            complexity_strategy: "adaptive",
          },
          curriculum: {
            promotion_count: 2,
            reached_level: 2,
            maximum_level: 3,
            updates_completed: 49,
            stop_reason: "target_runtime",
            retention_passed: true,
          },
          evidence: {
            receipt_digest: "sha256:receipt",
            teardown_confirmed: true,
          },
          teardown_confirmed: true,
        }),
      ),
    );
    window.history.replaceState(null, "", "/proofs/research_proof_test");
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <Routes>
            <Route path="/proofs/:proofId" element={<ProofDetailPage />} />
          </Routes>
        </BrowserRouter>,
      );
      await Promise.resolve();
    });

    expect(
      container.querySelector('a[href="/runs/research/runpod-proof-test"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        'a[href="/runs/research/runpod-proof-test/trajectory"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("sha256:receipt");
    expect(container.textContent).toContain("$0.11 est.");
    expect(container.textContent).toContain("Confirmed");
  });
});
