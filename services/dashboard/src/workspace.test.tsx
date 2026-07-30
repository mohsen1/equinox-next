import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ProofDetailPage, ProofsPage } from "./pages/proofs";
import {
  integratedObserverStages,
  observerExecutionStatus,
  ResearchRunPage,
  RunsPage,
} from "./pages/runs";
import { BrowserRouter, Route, Routes } from "./router";
import type { ResearchComputeExecution } from "./types";

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
      gpu_id: "NVIDIA H100 80GB HBM3",
      cloud_type: "SECURE",
      hourly_cost_usd: 3.49,
      maximum_hourly_cost_usd: 4,
    },
    progress: {
      phase: "complete",
      screen_completed: true,
      eligible,
      larger_model_profile_id: "qwen2.5-coder-7b-runpod-h100@1",
      model_revision: "c03e6d358207e414f1eca0bb1891e29f1db0e242",
      checkpoint_admission_rate: 0.875,
      informative_group_rate: 0.25,
      peak_reserved_vram_fraction: 0.411,
      policy_mutation_detected: false,
      policy_mutation_enabled: false,
      gate_results: {
        policy_unchanged: true,
        optimizer_state_restored: true,
        test_split_isolated: true,
      },
      artifact_set_manifest_digest: `sha256:${"a".repeat(64)}`,
      artifact_set_committed: true,
      elapsed_seconds: 521,
    },
    proof_id: null,
    receipt_digest: "sha256:eligibility",
    failure_receipt_digest: null,
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
              failure_receipt_digest: null,
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
            attempted_policy_update_count: 4,
            effective_policy_update_count: 3,
            retained_policy_update_count: 2,
            retention_rollback_count: 1,
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
                update: 4,
                level: 0,
                examples: 8,
                exact_successes: 5,
                exact_rate: 0.625,
                exact_rate_95ci: [0.305738, 0.863158],
                retention_transaction_disposition: "provisional",
              },
              {
                update: 5,
                level: 0,
                examples: 8,
                exact_successes: 6,
                exact_rate: 0.75,
                exact_rate_95ci: [0.40927, 0.928522],
                mastery_streak: 0,
                retention_transaction_disposition: "retain",
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
                  improved: 1,
                  regressed: 0,
                  unchanged: 7,
                  net_improved: 1,
                  mcnemar_exact_p_value: 1,
                },
              },
              {
                update: 6,
                level: 0,
                examples: 8,
                exact_successes: 5,
                exact_rate: 0.625,
                exact_rate_95ci: [0.305738, 0.863158],
                retention_transaction_disposition: "rollback",
                checkpoint_candidate_retained: false,
              },
            ],
          },
          proof_id: null,
          receipt_digest: null,
          failure_receipt_digest: null,
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

    expect(container.textContent).toContain(
      "6 / 120 · 4 attempted · 3 effective · 2 retained · 1 rolled back",
    );
    expect(container.textContent).toContain("Level 1 → 2 · 4/4 solved");
    expect(container.textContent).toContain("+12.5 pts vs baseline");
    expect(container.textContent).toContain("99.0% valid");
    expect(container.textContent).toContain("1.0% malformed recent");
    expect(container.textContent).toContain("12m 30s training");
    expect(container.textContent).toContain("45m 0s evaluation");
    expect(container.textContent).toContain(
      "Fixed 0, 1 · +1 / −0 · rotating +1 / −0",
    );
    expect(container.textContent).toContain("Best retained");
    expect(container.textContent).toContain("Provisional");
    expect(container.textContent).toContain("Rolled back");
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
        "qwen2.5-coder-7b-runpod-h100@1",
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
        "Peak GPU memory41.1% reserved",
      );
      expect(eligibilitySection?.textContent).toContain(
        "Policy mutationNone · verified",
      );
      expect(observerStageLabels(container)).toEqual([
        "Prepare",
        "Stage bundle",
        "Verify runtime/model",
        "Screen",
        "Teardown",
        "Publish",
      ]);
      expect(
        container.querySelector('[aria-label="Publish: Complete"]'),
      ).not.toBeNull();
      expect(container.textContent).toContain("Artifact setsha256:aaaaa");
      expect(container.textContent).toContain("committed");
      expect(headings).not.toContain("Progress");
      expect(headings).not.toContain("Validation");
      expect(headings).not.toContain("Result");
    },
  );

  it("shows exact integrated stage and digest evidence without claiming publication", async () => {
    const timestamp = new Date().toISOString();
    const bundleDigest = `sha256:${"b".repeat(64)}`;
    const stageDigest = `sha256:${"s".repeat(64)}`;
    const sourceDigest = `sha256:${"c".repeat(64)}`;
    const execution = {
      execution_id: "runpod-proof-integrated-observer",
      name: "Integrated 7B screen and training",
      workload_id: "repository-repair-larger-model-pilot",
      model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
      branch_width: 4,
      complexity_strategy: "adaptive",
      status: "RUNNING",
      provider_name: "RunPod",
      provider_handle: "runpod://pods/integrated-observer",
      resource_profile: {
        gpu_id: "NVIDIA H100 80GB HBM3",
        cloud_type: "SECURE",
      },
      observer_evidence: {
        phase: "verifying_runtime",
        message: "Verifying runtime and pinned model snapshot.",
        profile_id: "qwen2.5-coder-7b-runpod-h100@6",
        source_contract_digest: sourceDigest,
        workload_bundle_digest: bundleDigest,
        workload_bundle_size_bytes: 2_500_000,
        workload_bundle_path:
          "/workspace/equinox-state/workload-bundles/qwen-7b/bundle.tar.xz",
        bundle_stage_receipt_digest: stageDigest,
        network_volume_id: "network-volume-7b",
        network_volume_data_center_id: "EU-RO-1",
        network_volume_size_gb: 80,
      },
      progress: {
        phase: "verifying_runtime",
        message: "Verifying runtime and pinned model snapshot.",
      },
      proof_id: null,
      receipt_digest: null,
      failure_receipt_digest: null,
      started_at: timestamp,
      updated_at: timestamp,
      completed_at: null,
      teardown_confirmed: false,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(execution)),
    );
    window.history.replaceState(
      null,
      "",
      "/runs/research/runpod-proof-integrated-observer",
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

    const lifecycle = container.querySelector(".observer-stage");
    expect(observerStageLabels(container)).toEqual([
      "Prepare",
      "Stage bundle",
      "Verify runtime/model",
      "Screen",
      "Run",
      "Teardown",
      "Publish",
    ]);
    expect(
      lifecycle?.querySelector('[aria-label="Verify runtime/model: Current"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain(
      "Verifying runtime and pinned model snapshot.",
    );
    expect(container.querySelector(`[title="${bundleDigest}"]`)).not.toBeNull();
    expect(container.querySelector(`[title="${stageDigest}"]`)).not.toBeNull();
    expect(container.querySelector(`[title="${sourceDigest}"]`)).not.toBeNull();
    expect(container.textContent).toContain("Bundle size2.5 MB");
    expect(container.textContent).toContain(
      "/workspace/equinox-state/workload-bundles/qwen-7b/bundle.tar.xz",
    );
    expect(container.textContent).not.toContain("Mock");
  });

  it("keeps an unreceipted training result in publication", () => {
    const training = {
      ...largerModelEligibilityExecution(true),
      workload_id: "repository-repair-larger-model-pilot",
      status: "SUCCEEDED",
      proof_id: null,
      receipt_digest: null,
      artifact_publication_required: true,
      progress: {
        ...largerModelEligibilityExecution(true).progress,
        artifact_set_committed: false,
      },
    } as ResearchComputeExecution;

    expect(observerExecutionStatus(training)).toBe("FINALIZING");
    expect(integratedObserverStages(training).at(-1)).toMatchObject({
      label: "Publish",
      state: "current",
    });
  });

  it("keeps legacy successes successful and marks their evidence non-atomic", async () => {
    const legacy = {
      ...largerModelEligibilityExecution(true),
      workload_id: "revision30-post-freeze-external-adapter-evaluation",
      status: "SUCCEEDED",
      artifact_publication_required: false,
      progress: {
        phase: "complete",
      },
    } as ResearchComputeExecution;

    expect(observerExecutionStatus(legacy)).toBe("SUCCEEDED");
    expect(integratedObserverStages(legacy).at(-1)).toMatchObject({
      label: "Publish",
      state: "complete",
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(legacy)),
    );
    window.history.replaceState(
      null,
      "",
      `/runs/research/${legacy.execution_id}`,
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

    expect(container.textContent).toContain("Legacy · non-atomic");
    expect(container.textContent).not.toContain("Publication pending");
  });

  it("labels missing evidence as incomplete after an eligibility screen fails", async () => {
    const failed = {
      ...largerModelEligibilityExecution(false),
      execution_id: "runpod-proof-larger-model-failed",
      status: "FAILED",
      failure_receipt_digest: `sha256:${"f".repeat(64)}`,
      progress: {
        phase: "eligibility_branch_collection",
        error:
          "The completed eligibility screen did not expose its full terminal branch tree.",
        remote_error: {
          code: "REMOTE_WORKLOAD_FAILURE",
          message:
            "The completed eligibility screen did not expose its full terminal branch tree.",
          exit_code: 1,
        },
        operator_error: {
          code: "RUNPOD_OPERATOR_FAILURE",
          message: "The remote workload failed with exit code 1.",
        },
        larger_model_profile_id: "qwen2.5-coder-7b-runpod-h100@5",
        policy_mutation_enabled: false,
        branch_groups_completed: 7,
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(failed)),
    );
    window.history.replaceState(
      null,
      "",
      "/runs/research/runpod-proof-larger-model-failed",
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

    const eligibilitySection = [...container.querySelectorAll(".section")].find(
      (section) => section.querySelector("h2")?.textContent === "Eligibility",
    );
    const identitySection = [...container.querySelectorAll(".section")].find(
      (section) => section.querySelector("h2")?.textContent === "Identity",
    );

    expect(eligibilitySection?.querySelector(".status")?.textContent).toBe(
      "Incomplete",
    );
    expect(eligibilitySection?.textContent).toContain("RevisionUnavailable");
    expect(eligibilitySection?.textContent).toContain(
      "Checkpoint admissionIncomplete",
    );
    expect(eligibilitySection?.textContent).toContain(
      "Informative branchingIncomplete",
    );
    expect(eligibilitySection?.textContent).toContain(
      "Peak GPU memoryUnavailable",
    );
    expect(eligibilitySection?.textContent).toContain(
      "Policy mutationDisabled · verification unavailable",
    );
    expect(identitySection?.textContent).toContain("ProofNot produced");
    const failureNotice = container.querySelector(".notice.negative");
    expect(failureNotice?.textContent).toContain(
      "RemoteREMOTE_WORKLOAD_FAILURE",
    );
    expect(failureNotice?.textContent).toContain("full terminal branch tree");
    expect(failureNotice?.textContent).toContain(
      "OperatorRUNPOD_OPERATOR_FAILURE",
    );
    expect(failureNotice?.textContent).toContain("Process exit 1");
    expect(failureNotice?.textContent).toContain("sha256:fffff");
    expect(failureNotice?.textContent).toContain("TeardownConfirmed");
    expect(container.textContent).not.toContain("Awaiting model load");
    expect(container.textContent).not.toContain("Awaiting samples");
    expect(container.textContent).not.toContain("ProofPending");
  });

  it("keeps the operational receipt visible after scientific recovery", async () => {
    const failureDigest = `sha256:${"f".repeat(64)}`;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          ...largerModelEligibilityExecution(true),
          proof_id: "research_proof_recovered",
          receipt_digest: `sha256:${"a".repeat(64)}`,
          failure_receipt_digest: failureDigest,
        }),
      ),
    );
    window.history.replaceState(
      null,
      "",
      "/runs/research/runpod-proof-larger-model-eligible",
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

    const identitySection = [...container.querySelectorAll(".section")].find(
      (section) => section.querySelector("h2")?.textContent === "Identity",
    );
    expect(identitySection?.textContent).toContain("Prior failure receipt");
    expect(
      identitySection?.querySelector(`[title="${failureDigest}"]`),
    ).not.toBeNull();
  });

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
                meaningful_post_training: true,
                post_training_outcome: "MEANINGFUL_POST_TRAINING",
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
    expect(container.textContent).toContain("Meaningful");
    expect(container.textContent).toContain("Released");
  });

  it("links proof evidence back to the run and trajectory", async () => {
    const priorFailureDigest = `sha256:${"f".repeat(64)}`;
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
            hypothesis_passed: false,
            meaningful_post_training: false,
            post_training_outcome: "NEGATIVE_EXPERIMENT_COMPLETED",
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
            dynamic_complexity_progressed: true,
            reached_level: 2,
            maximum_level: 3,
            updates_completed: 49,
            stop_reason: "target_runtime",
            retention_passed: true,
          },
          evidence: {
            receipt_digest: "sha256:receipt",
            failure_receipt_digest: priorFailureDigest,
            teardown_confirmed: true,
            artifact_set_manifest_digest: null,
            artifact_set_committed: false,
            artifact_publication_status: "legacy_non_atomic",
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
    expect(container.textContent).toContain("Prior failure receipt");
    expect(
      container.querySelector(`[title="${priorFailureDigest}"]`),
    ).not.toBeNull();
    expect(container.textContent).toContain("$0.11 est.");
    expect(container.textContent).toContain("Confirmed");
    expect(container.textContent).toContain("Negative result");
    expect(container.textContent).toContain("Level 2 of 3 · progressed");
    expect(container.textContent).toContain("Legacy · non-atomic");
  });
});

function observerStageLabels(container: HTMLElement): Array<string | null> {
  return [
    ...container.querySelectorAll(".observer-stage-item > span:first-child"),
  ].map((label) => label.textContent);
}
