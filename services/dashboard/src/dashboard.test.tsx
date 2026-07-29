import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DashboardPage,
  estimatedStudyCost,
  sortConditions,
  unknownStudyCostCount,
} from "./pages/dashboard";
import { BrowserRouter } from "./router";
import type { ResearchStudyReport } from "./types";

function response(body: unknown) {
  return {
    ok: true,
    json: async () => body,
  };
}

const report: ResearchStudyReport = {
  study_id: "branch-aware-study@1",
  report_id: "branch-aware-study@1/report@1",
  generated_at: "2026-07-29T10:00:00Z",
  overall_status: "FAIL",
  aggregation_policy: "Per-seed evidence",
  report_digest: "sha256:study-report",
  freeze: {
    frozen_workload_revision: "repository-repair@3",
    frozen_source_commit: "abc123",
    model: { id: "Qwen/Qwen2.5-Coder-7B-Instruct" },
  },
  conditions: {
    "k4-seed-307": {
      seed: 307,
      branch_width: 4,
      status: "SUCCEEDED",
      initial_successes: 0.25,
      final_successes: 0.5,
      sampled_completions: 64,
    },
    "k1-seed-113": {
      seed: 113,
      branch_width: 1,
      status: "SUCCEEDED",
      initial_successes: 0.25,
      final_successes: 0.375,
      sampled_completions: 32,
    },
    "k4-seed-211": {
      seed: 211,
      branch_width: 4,
      status: "FAILED",
      initial_successes: null,
      final_successes: null,
      paired_improved: null,
      paired_regressed: null,
      sampled_completions: null,
    },
  },
  decisions: {
    training_advantage: { status: "FAIL", evidence: {} },
    teardown_complete: { status: "PASS", evidence: {} },
  },
  failure_count: {
    provider_executions: 0,
    operator_attempts: 0,
  },
  executions: [
    {
      execution_id: "execution-1",
      outcome: "SUCCEEDED",
      estimated_cost_usd: 0.42,
      teardown_confirmed: true,
    },
    {
      execution_id: "execution-2",
      outcome: "FAILED",
      estimated_cost_usd: null,
      teardown_confirmed: true,
    },
  ],
};

describe("Dashboard", () => {
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

  it("renders real research runs and API-backed study evidence", async () => {
    const updatedAt = new Date(Date.now() - 60_000).toISOString();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const path = String(input);
        if (path.endsWith("/v1/runs")) {
          return response({
            items: [
              {
                run_id: "fixture-run",
                status: "SUCCEEDED",
              },
            ],
            research_items: [
              {
                execution_id: "runpod-proof-real",
                name: "Real H100 pilot",
                workload_id: "repository-repair",
                model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
                branch_width: 4,
                complexity_strategy: "adaptive",
                status: "RUNNING",
                provider_name: "RunPod",
                provider_handle: "runpod://pods/real",
                resource_profile: {},
                allocated_gpu: "NVIDIA H100 80GB HBM3",
                cost: {
                  total_usd: 0.42,
                  estimated: true,
                  hourly_rate_usd: 2.99,
                  elapsed_seconds: 506,
                },
                progress: {},
                proof_id: null,
                receipt_digest: null,
                failure_receipt_digest: null,
                started_at: updatedAt,
                updated_at: updatedAt,
                completed_at: null,
                teardown_confirmed: false,
              },
            ],
          });
        }
        if (path.endsWith("/v1/studies")) {
          return response({
            items: [
              {
                study_id: report.study_id,
                report_id: report.report_id,
                generated_at: report.generated_at,
                overall_status: report.overall_status,
                workload_revision: "repository-repair@3",
                model_id: "Qwen/Qwen2.5-Coder-7B-Instruct",
                condition_count: 3,
                execution_count: 2,
                estimated_provider_cost_usd: 0.42,
                decisions: {
                  training_advantage: "FAIL",
                  teardown_complete: "PASS",
                },
              },
            ],
          });
        }
        if (path.includes("/v1/studies/")) return response(report);
        throw new Error(`Unexpected request: ${path}`);
      }),
    );
    window.history.replaceState(null, "", "/dashboard");
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <BrowserRouter>
          <DashboardPage />
        </BrowserRouter>,
      );
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("Real H100 pilot");
    expect(container.textContent).toContain("NVIDIA H100 80GB HBM3");
    expect(container.textContent).toContain("$0.42 est.");
    expect(container.textContent).toContain(
      "The latest study did not pass its preregistered gates.",
    );
    expect(container.textContent).toContain("1 of 2 gates passed");
    const missingCondition = [...container.querySelectorAll("tbody tr")].find(
      (row) => row.textContent?.includes("k4-seed-211"),
    );
    expect(missingCondition?.textContent).toContain("—");
    expect(missingCondition?.textContent).not.toContain("+0 / −0");
    const unknownCost = [...container.querySelectorAll("tbody tr")].find(
      (row) => row.textContent?.includes("execution-2"),
    );
    expect(unknownCost?.textContent).toContain("Unavailable");
    expect(container.textContent).not.toContain("Local fixture");
    expect(container.textContent).not.toContain("Mock providers");
  });

  it("sorts evidence without mutating the API response", () => {
    const rows = Object.entries(report.conditions).map(([id, condition]) => ({
      id,
      ...condition,
    }));
    const sorted = sortConditions(rows, "seed");

    expect(sorted.map((row) => row.seed)).toEqual([113, 211, 307]);
    expect(rows.map((row) => row.seed)).toEqual([307, 113, 211]);
    expect(sortConditions(rows, "final_successes", true).at(-1)?.id).toBe(
      "k4-seed-211",
    );
    expect(estimatedStudyCost(report)).toBe(0.42);
    expect(unknownStudyCostCount(report)).toBe(1);
  });
});
