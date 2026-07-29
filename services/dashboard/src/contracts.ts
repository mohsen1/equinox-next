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
  const branchSnapshots = array(
    trajectory.branch_snapshots,
    "trajectory branch snapshots",
  );
  branchSnapshots.forEach(validateComparisonBranchSnapshot);
  const branchTaskIds = new Set<string>();
  const branchSnapshotIds = new Set<string>();
  const branchSnapshotIdByTaskId = new Map<string, string>();
  const optimizerBindingsByAttempt = new Map<
    number,
    {
      groupIds: string[];
      snapshotIds: string[];
      declaredInputIds: string[];
      signalIds: string[];
      update: number;
    }
  >();
  branchSnapshots.forEach((value, index) => {
    const snapshot = record(value, `trajectory branch snapshot ${index}`);
    const snapshotId = requiredString(
      snapshot.snapshot_id,
      `trajectory branch snapshot ${index}.snapshot_id`,
    );
    if (branchSnapshotIds.has(snapshotId)) {
      throw new Error(
        "Invalid API response: branch snapshot identities must be unique.",
      );
    }
    branchSnapshotIds.add(snapshotId);
    if (typeof snapshot.task_id !== "string") return;
    const taskId = requiredString(
      snapshot.task_id,
      `trajectory branch snapshot ${index}.task_id`,
    );
    if (branchTaskIds.has(taskId)) {
      throw new Error(
        "Invalid API response: branch task identities must be unique.",
      );
    }
    branchTaskIds.add(taskId);
    branchSnapshotIdByTaskId.set(taskId, snapshotId);

    if (
      snapshot.optimizer_update === null ||
      snapshot.optimizer_update === undefined
    ) {
      return;
    }
    const optimizerUpdate = record(
      snapshot.optimizer_update,
      `trajectory branch snapshot ${index}.optimizer update`,
    );
    if (
      optimizerUpdate.attempted_policy_update_index === null ||
      optimizerUpdate.attempted_policy_update_index === undefined
    ) {
      return;
    }
    const attempt = nonnegativeInteger(
      optimizerUpdate.attempted_policy_update_index,
      `trajectory branch snapshot ${index}.optimizer attempt`,
    );
    if (attempt < 1) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index}.optimizer attempt must be positive.`,
      );
    }
    if (optimizerUpdate.optimizer_input_group_ids === undefined) return;
    const declaredInputIds = array(
      optimizerUpdate.optimizer_input_group_ids,
      `trajectory branch snapshot ${index}.optimizer inputs`,
    ).map((groupId, groupIndex) =>
      requiredString(
        groupId,
        `trajectory branch snapshot ${index}.optimizer input ${groupIndex}`,
      ),
    );
    if (
      new Set(declaredInputIds).size !== declaredInputIds.length ||
      nonnegativeInteger(
        optimizerUpdate.optimizer_input_group_count,
        `trajectory branch snapshot ${index}.optimizer input count`,
      ) !== declaredInputIds.length ||
      !declaredInputIds.includes(taskId)
    ) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} has invalid optimizer input identities.`,
      );
    }
    const update = nonnegativeInteger(
      optimizerUpdate.update,
      `trajectory branch snapshot ${index}.optimizer update`,
    );
    if (
      update < 1 ||
      nonnegativeInteger(
        optimizerUpdate.optimizer_input_consumed_by_update,
        `trajectory branch snapshot ${index}.optimizer consumed update`,
      ) !== update
    ) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} has invalid optimizer consumption evidence.`,
      );
    }
    const signalIds = array(
      optimizerUpdate.policy_signal_group_ids,
      `trajectory branch snapshot ${index}.policy signal inputs`,
    ).map((groupId, groupIndex) =>
      requiredString(
        groupId,
        `trajectory branch snapshot ${index}.policy signal input ${groupIndex}`,
      ),
    );
    const signalConsumed = optimizerUpdate.policy_signal_consumed_by_update;
    if (
      new Set(signalIds).size !== signalIds.length ||
      nonnegativeInteger(
        optimizerUpdate.policy_signal_group_count,
        `trajectory branch snapshot ${index}.policy signal count`,
      ) !== signalIds.length ||
      signalIds.some((groupId) => !declaredInputIds.includes(groupId)) ||
      (signalIds.includes(taskId)
        ? nonnegativeInteger(
            signalConsumed,
            `trajectory branch snapshot ${index}.policy signal consumed update`,
          ) !== update
        : signalConsumed !== undefined)
    ) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} has invalid policy signal subset evidence.`,
      );
    }
    const binding = optimizerBindingsByAttempt.get(attempt) ?? {
      groupIds: [],
      snapshotIds: [],
      declaredInputIds,
      signalIds,
      update,
    };
    if (
      binding.update !== update ||
      binding.declaredInputIds.length !== declaredInputIds.length ||
      binding.declaredInputIds.some(
        (groupId, groupIndex) => groupId !== declaredInputIds[groupIndex],
      ) ||
      binding.signalIds.length !== signalIds.length ||
      binding.signalIds.some(
        (groupId, groupIndex) => groupId !== signalIds[groupIndex],
      )
    ) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} disagrees with its optimizer attempt evidence.`,
      );
    }
    binding.groupIds.push(taskId);
    binding.snapshotIds.push(snapshotId);
    optimizerBindingsByAttempt.set(attempt, binding);
  });
  if (trajectory.branch_evidence_complete !== undefined) {
    if (typeof trajectory.branch_evidence_complete !== "boolean") {
      throw new Error(
        "Invalid API response: expected trajectory branch evidence completeness.",
      );
    }
    const groupCount = nonnegativeInteger(
      trajectory.branch_evidence_group_count,
      "trajectory branch evidence group count",
    );
    if (
      trajectory.branch_evidence_complete &&
      groupCount !== branchSnapshots.length
    ) {
      throw new Error(
        "Invalid API response: complete branch evidence count does not match snapshots.",
      );
    }
  }
  if (trajectory.total_sampled_completion_tokens !== undefined) {
    const totalCompletionTokens = nonnegativeInteger(
      trajectory.total_sampled_completion_tokens,
      "trajectory total sampled completion tokens",
    );
    const discardedCompletionTokens = nonnegativeInteger(
      trajectory.discarded_sampled_completion_tokens,
      "trajectory discarded sampled completion tokens",
    );
    const persistedCompletionTokens = branchSnapshots.reduce<number>(
      (total, value, index) => {
        const snapshot = record(value, `trajectory branch snapshot ${index}`);
        return (
          total +
          nonnegativeInteger(
            snapshot.sampled_completion_tokens,
            `trajectory branch snapshot ${index}.sampled completion tokens`,
          )
        );
      },
      0,
    );
    if (
      persistedCompletionTokens + discardedCompletionTokens !==
      totalCompletionTokens
    ) {
      throw new Error(
        "Invalid API response: sampled completion tokens do not reconcile with branch evidence.",
      );
    }
  }
  if (trajectory.policy_update_lineage !== undefined) {
    array(
      trajectory.policy_update_lineage,
      "trajectory policy update lineage",
    ).forEach((value, index) => {
      const item = record(value, `trajectory policy update ${index}`);
      if (
        item.schema_version !== undefined &&
        item.schema_version !== 1 &&
        item.schema_version !== 2
      ) {
        throw new Error(
          `Invalid API response: policy update ${index} has an unsupported lineage schema.`,
        );
      }
      const attempt = nonnegativeInteger(
        item.attempted_policy_update_index,
        `policy update ${index}.attempt`,
      );
      const update = nonnegativeInteger(
        item.update,
        `policy update ${index}.update`,
      );
      if (attempt < 1 || update < 1) {
        throw new Error(
          `Invalid API response: policy update ${index} identities must be positive.`,
        );
      }
      const signalGroups = array(
        item.policy_signal_group_ids,
        `policy update ${index}.signal groups`,
      ).map((groupId, groupIndex) =>
        requiredString(
          groupId,
          `policy update ${index}.signal group ${groupIndex}`,
        ),
      );
      const optimizerInputs = array(
        item.optimizer_input_group_ids,
        `policy update ${index}.optimizer inputs`,
      ).map((groupId, groupIndex) =>
        requiredString(
          groupId,
          `policy update ${index}.optimizer input ${groupIndex}`,
        ),
      );
      const lineageBranchSnapshots = array(
        item.branch_snapshot_ids,
        `policy update ${index}.branch snapshots`,
      ).map((snapshotId, snapshotIndex) =>
        requiredString(
          snapshotId,
          `policy update ${index}.branch snapshot ${snapshotIndex}`,
        ),
      );
      if (
        new Set(signalGroups).size !== signalGroups.length ||
        new Set(optimizerInputs).size !== optimizerInputs.length ||
        new Set(lineageBranchSnapshots).size !== lineageBranchSnapshots.length
      ) {
        throw new Error(
          `Invalid API response: policy update ${index} lineage identities must be unique.`,
        );
      }
      signalGroups.forEach((id) => {
        if (!branchTaskIds.has(id)) {
          throw new Error(
            `Invalid API response: policy update ${index} signal group has no branch evidence.`,
          );
        }
      });
      if (signalGroups.some((groupId) => !optimizerInputs.includes(groupId))) {
        throw new Error(
          `Invalid API response: policy update ${index} signal groups must be optimizer inputs.`,
        );
      }
      const expectedSnapshotIds = optimizerInputs.map((groupId) => {
        const snapshotId = branchSnapshotIdByTaskId.get(groupId);
        if (!snapshotId) {
          throw new Error(
            `Invalid API response: policy update ${index} optimizer input has no branch evidence.`,
          );
        }
        return snapshotId;
      });
      if (
        expectedSnapshotIds.length !== lineageBranchSnapshots.length ||
        expectedSnapshotIds.some(
          (snapshotId, snapshotIndex) =>
            snapshotId !== lineageBranchSnapshots[snapshotIndex],
        )
      ) {
        throw new Error(
          `Invalid API response: policy update ${index} optimizer inputs do not exactly match branch snapshots.`,
        );
      }
      lineageBranchSnapshots.forEach((id) => {
        if (!branchSnapshotIds.has(id)) {
          throw new Error(
            `Invalid API response: policy update ${index} references a missing branch snapshot.`,
          );
        }
      });
      if (item.schema_version === 2) {
        if (
          nonnegativeInteger(
            item.policy_signal_group_count,
            `policy update ${index}.signal group count`,
          ) !== signalGroups.length ||
          nonnegativeInteger(
            item.optimizer_input_group_count,
            `policy update ${index}.optimizer input count`,
          ) !== optimizerInputs.length
        ) {
          throw new Error(
            `Invalid API response: policy update ${index} lineage counts are inconsistent.`,
          );
        }
        const binding = optimizerBindingsByAttempt.get(attempt);
        if (
          !binding ||
          binding.groupIds.length !== optimizerInputs.length ||
          binding.groupIds.some(
            (groupId, groupIndex) => groupId !== optimizerInputs[groupIndex],
          ) ||
          binding.declaredInputIds.length !== optimizerInputs.length ||
          binding.declaredInputIds.some(
            (groupId, groupIndex) => groupId !== optimizerInputs[groupIndex],
          ) ||
          binding.signalIds.length !== signalGroups.length ||
          binding.signalIds.some(
            (groupId, groupIndex) => groupId !== signalGroups[groupIndex],
          ) ||
          binding.update !== update ||
          binding.snapshotIds.some(
            (snapshotId, snapshotIndex) =>
              snapshotId !== lineageBranchSnapshots[snapshotIndex],
          )
        ) {
          throw new Error(
            `Invalid API response: policy update ${index} optimizer evidence is not exactly bound to branch snapshots.`,
          );
        }
      }
    });
  }
  record(trajectory.initial_by_level, "initial level results");
  record(trajectory.final_by_level, "final level results");
  return {
    execution,
    trajectory:
      trajectory as unknown as ResearchTrajectoryResponse["trajectory"],
  };
}

function validateComparisonBranchSnapshot(value: unknown, index: number): void {
  const snapshot = record(value, `trajectory branch snapshot ${index}`);
  const rolloutTopologyValue =
    snapshot.rollout_topology &&
    typeof snapshot.rollout_topology === "object" &&
    !Array.isArray(snapshot.rollout_topology)
      ? (snapshot.rollout_topology as Record<string, unknown>)
      : null;
  const hasIndependentRolloutMarker =
    rolloutTopologyValue?.revision === "independent-prefix-k4@1";
  if (snapshot.comparison_condition === undefined) {
    if (hasIndependentRolloutMarker) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} independent rollout requires a typed comparison condition.`,
      );
    }
    return;
  }

  const condition = record(
    snapshot.comparison_condition,
    `trajectory branch snapshot ${index}.comparison condition`,
  );
  requiredString(
    condition.condition_id,
    `trajectory branch snapshot ${index}.comparison condition id`,
  );
  requiredString(
    condition.short_label,
    `trajectory branch snapshot ${index}.comparison condition label`,
  );
  requiredString(
    condition.group_credit,
    `trajectory branch snapshot ${index}.comparison group credit`,
  );
  const topology = condition.prefix_topology;
  if (topology !== "shared" && topology !== "independent") {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} has an unknown prefix topology.`,
    );
  }
  if (topology !== "independent" && hasIndependentRolloutMarker) {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} rollout topology does not match its comparison condition.`,
    );
  }
  const branchWidth = nonnegativeInteger(
    condition.branch_width,
    `trajectory branch snapshot ${index}.comparison branch width`,
  );
  if (branchWidth !== 1 && branchWidth !== 4) {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} comparison branch width must be 1 or 4.`,
    );
  }
  if (typeof condition.shared_prefix !== "boolean") {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} comparison shared-prefix flag is invalid.`,
    );
  }

  const siblings = array(
    snapshot.siblings,
    `trajectory branch snapshot ${index}.siblings`,
  );
  if (siblings.length !== branchWidth) {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} does not match its comparison branch width.`,
    );
  }
  let siblingCompletionTokens = 0;
  siblings.forEach((value, siblingIndex) => {
    const sibling = record(
      value,
      `trajectory branch snapshot ${index}.sibling ${siblingIndex}`,
    );
    if (sibling.index !== siblingIndex) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} sibling order is invalid.`,
      );
    }
    array(
      sibling.steps,
      `trajectory branch snapshot ${index}.sibling ${siblingIndex}.steps`,
    );
    siblingCompletionTokens += nonnegativeInteger(
      sibling.completion_tokens,
      `trajectory branch snapshot ${index}.sibling ${siblingIndex}.completion tokens`,
    );
  });

  const sampledCompletionTokens = nonnegativeInteger(
    snapshot.sampled_completion_tokens,
    `trajectory branch snapshot ${index}.sampled completion tokens`,
  );
  if (topology === "independent") {
    const rolloutTopology = record(
      snapshot.rollout_topology,
      `trajectory branch snapshot ${index}.rollout topology`,
    );
    const initialState = record(
      snapshot.initial_state,
      `trajectory branch snapshot ${index}.initial state`,
    );
    requiredString(
      initialState.state_id,
      `trajectory branch snapshot ${index}.initial state id`,
    );
    requiredString(
      initialState.payload_digest,
      `trajectory branch snapshot ${index}.initial state payload digest`,
    );
    requiredString(
      initialState.fidelity,
      `trajectory branch snapshot ${index}.initial state fidelity`,
    );
    const initialStateRole = requiredString(
      initialState.role,
      `trajectory branch snapshot ${index}.initial state role`,
    );
    if (
      branchWidth !== 4 ||
      condition.schema_version !== 1 ||
      condition.condition_id !== "independent_prefix_grpo_k4" ||
      condition.group_credit !== "sibling_relative" ||
      condition.shared_prefix !== false ||
      snapshot.shared_prefix !== null ||
      snapshot.checkpoint !== null ||
      rolloutTopology.revision !== "independent-prefix-k4@1" ||
      rolloutTopology.static_group_width !== 4 ||
      rolloutTopology.independent_model_generated_prefixes !== true ||
      rolloutTopology.shared_model_generated_prefix !== false ||
      rolloutTopology.sibling_group_relative_credit !== true ||
      rolloutTopology.initial_state_matching !== "same_task_initial_state" ||
      initialStateRole !== "matched_task_initial_state_not_decision_checkpoint"
    ) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} is not a faithful independent-prefix K=4 snapshot.`,
      );
    }
    if (siblingCompletionTokens !== sampledCompletionTokens) {
      throw new Error(
        `Invalid API response: trajectory branch snapshot ${index} independent completion tokens do not reconcile.`,
      );
    }
    return;
  }

  const sharedPrefix = record(
    snapshot.shared_prefix,
    `trajectory branch snapshot ${index}.shared prefix`,
  );
  const prefixCompletionTokens = nonnegativeInteger(
    sharedPrefix.completion_tokens,
    `trajectory branch snapshot ${index}.shared prefix completion tokens`,
  );
  array(
    sharedPrefix.steps,
    `trajectory branch snapshot ${index}.shared prefix steps`,
  );
  if (
    condition.shared_prefix !== true ||
    prefixCompletionTokens + siblingCompletionTokens !== sampledCompletionTokens
  ) {
    throw new Error(
      `Invalid API response: trajectory branch snapshot ${index} shared-prefix completion tokens do not reconcile.`,
    );
  }
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

function nonnegativeFiniteNumber(value: unknown, label: string): number {
  const number = finiteNumber(value, label);
  if (number < 0) {
    throw new Error(`Invalid API response: expected nonnegative ${label}.`);
  }
  return number;
}

function nonnegativeInteger(value: unknown, label: string): number {
  const number = nonnegativeFiniteNumber(value, label);
  if (!Number.isInteger(number)) {
    throw new Error(`Invalid API response: expected integer ${label}.`);
  }
  return number;
}

function nullableFiniteNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  return finiteNumber(value, label);
}
