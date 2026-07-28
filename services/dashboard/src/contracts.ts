import type {
  EnvironmentsResponse,
  ResearchComputeExecution,
  ResearchProofDetail,
  ResearchProofSummary,
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
  record(item.evidence, "evidence");
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
