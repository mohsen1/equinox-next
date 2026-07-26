import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ProofDetailPage, ProofsPage } from "./pages/proofs";
import { RunsPage } from "./pages/workspace";
import { BrowserRouter, Route, Routes } from "./router";

function response(body: unknown) {
  return {
    ok: true,
    json: async () => body,
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
    expect(container.textContent).toContain("adaptive complexity · K=4");
    const time = container.querySelector("time");
    expect(time?.textContent).toBe("4 min ago");
    expect(time?.getAttribute("aria-label")).not.toBe(time?.textContent);
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
