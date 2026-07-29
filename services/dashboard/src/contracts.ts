import type {
  EnvironmentsResponse,
  ResearchComputeExecution,
  ResearchProofDetail,
  ResearchProofSummary,
  ResearchStudyReport,
  ResearchStudySummary,
  ResearchTrajectoryResponse,
  RunSummary,
} from "./types";

export interface RunsResponse {
  items: RunSummary[];
  research_items: ResearchComputeExecution[];
}

export interface ProofsResponse {
  items: ResearchProofSummary[];
}

export interface StudiesResponse {
  items: ResearchStudySummary[];
}

export function decodeRunsResponse(value: unknown): RunsResponse {
  const root = record(value, "runs response");
  const items = array(root.items, "runs");
  const researchItems = array(root.research_items, "research runs");
  return {
    items: items.map((item, index) => {
      const run = record(item, `run ${index}`);
      requiredString(run.run_id, `run ${index}.run_id`);
      requiredString(run.status, `run ${index}.status`);
      return run as unknown as RunSummary;
    }),
    research_items: researchItems.map(decodeResearchComputeExecution),
  };
}

export function decodeResearchComputeExecution(
  value: unknown,
): ResearchComputeExecution {
  const item = record(value, "research execution");
  requiredString(item.execution_id, "execution_id");
  requiredString(item.name, "name");
  requiredString(item.status, "status");
  requiredString(item.started_at, "started_at");
  requiredString(item.updated_at, "updated_at");
  record(item.resource_profile, "resource_profile");
  record(item.progress, "progress");
  nullableDigest(item.failure_receipt_digest, "failure_receipt_digest");
  if (item.branch_width !== 1 && item.branch_width !== 4) {
    throw new Error("Invalid API response: branch_width must be 1 or 4.");
  }
  if (item.complexity_strategy !== "adaptive") {
    throw new Error(
      "Invalid API response: complexity_strategy must be adaptive.",
    );
  }
  return item as unknown as ResearchComputeExecution;
}

export function decodeStudiesResponse(value: unknown): StudiesResponse {
  const root = record(value, "studies response");
  return {
    items: array(root.items, "studies").map((value, index) => {
      const item = record(value, `study ${index}`);
      requiredString(item.study_id, `study ${index}.study_id`);
      requiredString(item.generated_at, `study ${index}.generated_at`);
      studyStatus(item.overall_status, `study ${index}.overall_status`);
      finiteNumber(item.condition_count, `study ${index}.condition_count`);
      finiteNumber(item.execution_count, `study ${index}.execution_count`);
      finiteNumber(
        item.estimated_provider_cost_usd,
        `study ${index}.estimated_provider_cost_usd`,
      );
      const decisions = record(item.decisions, `study ${index}.decisions`);
      for (const [decisionId, status] of Object.entries(decisions)) {
        studyStatus(status, `study ${index}.decisions.${decisionId}`);
      }
      return item as unknown as ResearchStudySummary;
    }),
  };
}

export function decodeResearchStudyReport(value: unknown): ResearchStudyReport {
  const item = record(value, "study report");
  requiredString(item.study_id, "study report.study_id");
  requiredString(item.report_id, "study report.report_id");
  requiredString(item.generated_at, "study report.generated_at");
  requiredString(item.aggregation_policy, "study report.aggregation_policy");
  requiredString(item.report_digest, "study report.report_digest");
  studyStatus(item.overall_status, "study report.overall_status");
  const freeze = record(item.freeze, "study report.freeze");
  if (freeze.model !== undefined && freeze.model !== null) {
    record(freeze.model, "study report.freeze.model");
  }
  const conditions = record(item.conditions, "study report.conditions");
  for (const [conditionId, value] of Object.entries(conditions)) {
    const condition = record(value, `study condition ${conditionId}`);
    for (const field of [
      "seed",
      "branch_width",
      "sampled_completions",
      "policy_updates",
      "optimizer_updates",
      "initial_successes",
      "final_successes",
      "gain",
      "paired_improved",
      "paired_regressed",
      "paired_p_value",
    ]) {
      nullableFiniteNumber(
        condition[field],
        `study condition ${conditionId}.${field}`,
      );
    }
  }
  const decisions = record(item.decisions, "study report.decisions");
  for (const [decisionId, value] of Object.entries(decisions)) {
    const decision = record(value, `study decision ${decisionId}`);
    studyStatus(decision.status, `study decision ${decisionId}.status`);
    record(decision.evidence, `study decision ${decisionId}.evidence`);
  }
  const failureCount = record(item.failure_count, "study report.failure_count");
  finiteNumber(
    failureCount.provider_executions,
    "study report.failure_count.provider_executions",
  );
  finiteNumber(
    failureCount.operator_attempts,
    "study report.failure_count.operator_attempts",
  );
  for (const [index, value] of array(
    item.executions,
    "study report.executions",
  ).entries()) {
    const execution = record(value, `study execution ${index}`);
    requiredString(
      execution.execution_id,
      `study execution ${index}.execution_id`,
    );
    requiredString(execution.outcome, `study execution ${index}.outcome`);
    nullableFiniteNumber(
      execution.estimated_cost_usd,
      `study execution ${index}.estimated_cost_usd`,
    );
    if (typeof execution.teardown_confirmed !== "boolean") {
      throw new Error(
        `Invalid API response: study execution ${index}.teardown_confirmed.`,
      );
    }
  }
  return item as unknown as ResearchStudyReport;
}

export function decodeProofsResponse(value: unknown): ProofsResponse {
  const root = record(value, "proofs response");
  return {
    items: array(root.items, "proofs").map((item, index) =>
      decodeProofSummary(item, `proof ${index}`),
    ),
  };
}

export function decodeProofDetail(value: unknown): ResearchProofDetail {
  const proof = decodeProofSummary(value, "proof detail");
  const item = record(value, "proof detail");
  requiredString(item.started_at, "started_at");
  record(item.provider, "provider");
  record(item.workload, "workload");
  record(item.curriculum, "curriculum");
  const evidence = record(item.evidence, "evidence");
  nullableDigest(
    evidence.failure_receipt_digest,
    "evidence.failure_receipt_digest",
  );
  return { ...item, ...proof } as unknown as ResearchProofDetail;
}

export function decodeEnvironmentsResponse(
  value: unknown,
): EnvironmentsResponse {
  const root = record(value, "environments response");
  const branching = record(root.branching, "branching");
  if (branching.mode !== "static") {
    throw new Error("Invalid API response: branching mode must be static.");
  }
  return {
    items: array(root.items, "environments").map((value, index) => {
      const item = record(value, `environment ${index}`);
      requiredString(
        item.environment_id,
        `environment ${index}.environment_id`,
      );
      requiredString(item.name, `environment ${index}.name`);
      record(item.complexity, `environment ${index}.complexity`);
      return item as unknown as EnvironmentsResponse["items"][number];
    }),
    complexity_schema: record(root.complexity_schema, "complexity schema"),
    branching: branching as unknown as EnvironmentsResponse["branching"],
  };
}

export function decodeResearchTrajectoryResponse(
  value: unknown,
): ResearchTrajectoryResponse {
  const root = record(value, "trajectory response");
  const execution = decodeResearchComputeExecution(root.execution);
  if (root.trajectory === null) return { execution, trajectory: null };
  const trajectory = record(root.trajectory, "trajectory");
  if (trajectory.branch_width !== 1 && trajectory.branch_width !== 4) {
    throw new Error(
      "Invalid API response: trajectory branch_width must be 1 or 4.",
    );
  }
  array(trajectory.checkpoints, "trajectory checkpoints");
  array(trajectory.promotions, "trajectory promotions");
  array(trajectory.branch_snapshots, "trajectory branch snapshots");
  record(trajectory.initial_by_level, "initial level results");
  record(trajectory.final_by_level, "final level results");
  return {
    execution,
    trajectory:
      trajectory as unknown as ResearchTrajectoryResponse["trajectory"],
  };
}

function decodeProofSummary(
  value: unknown,
  label: string,
): ResearchProofSummary {
  const item = record(value, label);
  requiredString(item.proof_id, `${label}.proof_id`);
  requiredString(item.completed_at, `${label}.completed_at`);
  record(item.learning, `${label}.learning`);
  record(item.hardware, `${label}.hardware`);
  record(item.cost, `${label}.cost`);
  if (typeof item.teardown_confirmed !== "boolean") {
    throw new Error(`Invalid API response: ${label}.teardown_confirmed.`);
  }
  return item as unknown as ResearchProofSummary;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid API response: expected ${label}.`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`Invalid API response: expected ${label} array.`);
  }
  return value;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.length) {
    throw new Error(`Invalid API response: expected ${label}.`);
  }
  return value;
}

function nullableDigest(value: unknown, label: string): void {
  if (
    value !== null &&
    (typeof value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value))
  ) {
    throw new Error(`Invalid API response: expected ${label}.`);
  }
}

function studyStatus(value: unknown, label: string): "PASS" | "FAIL" {
  if (value !== "PASS" && value !== "FAIL") {
    throw new Error(`Invalid API response: expected ${label}.`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid API response: expected ${label}.`);
  }
  return value;
}

function nullableFiniteNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  return finiteNumber(value, label);
}
