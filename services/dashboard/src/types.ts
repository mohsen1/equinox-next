export type RunStatus =
  | "QUEUED"
  | "PROVISIONING"
  | "PREPARING"
  | "RUNNING"
  | "FINALIZING"
  | "SUCCEEDED"
  | "CANCEL_REQUESTED"
  | "CANCELING"
  | "CANCELED"
  | "FAILED";

export interface RunSummary {
  run_id: string;
  source_run_id: string | null;
  name: string;
  algorithm: "independent_rollout_baseline" | "bpo_local_metric";
  status: RunStatus;
  desired_state: string;
  manifest: {
    environment?: {
      id?: string;
      snapshot_fidelity?: string;
      status?: string;
    };
    complexity?: ComplexityConfig;
    [key: string]: unknown;
  };
  manifest_digest: string;
  cleanup_status: string;
  collection_batch_count: number;
  iteration_count: number;
  rollout_tree_count: number;
  verification_run_count: number;
  abstention_count: number;
  retry_count: number;
  created_at: string;
  updated_at: string;
  providers: {
    policy_compute: string;
    judge: string;
    execution: string;
  };
  cost: { execution_credits: number; judge_credits: number };
}

export type ResearchComputeStatus =
  | "PROVISIONING"
  | "RUNNING"
  | "FINALIZING"
  | "SUCCEEDED"
  | "FAILED";

export interface ResearchComputeExecution {
  execution_id: string;
  name: string;
  workload_id: string;
  model_id: string | null;
  branch_width: 4;
  complexity_strategy: "adaptive";
  status: ResearchComputeStatus;
  provider_name: "RunPod";
  provider_handle: string | null;
  resource_profile: {
    gpu_id?: string;
    cloud_type?: string;
    image?: string;
    hourly_cost_usd?: number;
    maximum_hourly_cost_usd?: number;
    [key: string]: unknown;
  };
  progress: {
    phase?: string;
    message?: string;
    update?: number;
    maximum_updates?: number;
    exact_rate?: number;
    format_rate?: number;
    mean_reward?: number;
    current_level?: number;
    maximum_level?: number;
    promotion_count?: number;
    sampled_completions?: number;
    elapsed_seconds?: number;
    stop_reason?: string;
    initial_exact_rate?: number;
    final_exact_rate?: number;
    reward_gain?: number;
    hypothesis_passed?: boolean;
    adapter_persisted?: boolean;
    post_training_completed?: boolean;
    informative_group_rate?: number;
    teacher_fallback_rate?: number;
    policy_update_count?: number;
    teacher_update_count?: number;
    error?: string;
    [key: string]: unknown;
  };
  proof_id: string | null;
  receipt_digest: string | null;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  teardown_confirmed: boolean;
}

export interface ComplexityConfig {
  strategy: "adaptive";
  minimum_level: number;
  initial_level: number;
  maximum_level: number;
  sampling_band: number;
  mastery_threshold: number;
  evaluation_window: number;
  promotion_step: number;
}

export interface ComplexityState {
  available?: true;
  run_id: string;
  environment_id: string;
  strategy: "adaptive";
  minimum_level: number;
  current_level: number;
  maximum_level: number;
  sampling_band: number;
  mastery_threshold: number;
  evaluation_window: number;
  promotion_step: number;
  window_attempts: number;
  window_successes: number;
  promotion_count: number;
  last_accuracy: number | null;
  active_range: [number, number];
  window_progress: {
    attempts: number;
    required: number;
    successes: number;
  };
}

export interface UnavailableComplexityState {
  available: false;
  run_id: string;
  environment_id: string;
  reason: "legacy_run";
}

export type ComplexityResponse = ComplexityState | UnavailableComplexityState;

export interface EnvironmentSpec {
  environment_id: string;
  name: string;
  short_name: string;
  summary: string;
  status: "LOCAL_FIXTURE" | "CONFIGURATION_DRAFT";
  launch_enabled: boolean;
  action_space: string;
  verifier: string;
  snapshot_strategy: string;
  task_revision: string;
  complexity: {
    dimensions: string[];
    default_initial_level: number;
    default_max_level: number;
  };
}

export interface EnvironmentsResponse {
  items: EnvironmentSpec[];
  complexity_schema: Record<string, unknown>;
  branching: {
    mode: "static";
    branch_width: number;
    note: string;
  };
}

export interface ArtifactRef {
  artifact_id: string;
  digest: string;
  role: string;
  media_type: string;
  viewer_hint?: string | null;
  visibility: string;
  trust_class: string;
}

export interface ProofBundle {
  proof_bundle_id: string;
  digest: string;
  subject_id: string;
  task_reference: ArtifactRef;
  source_render: ArtifactRef;
  candidate_render: ArtifactRef;
  action_summary: ArtifactRef;
  geometry_report: ArtifactRef;
  renderer: Record<string, unknown>;
}

export interface GraphState {
  id: string;
  type: "state";
  sequence: number;
  semantic_status: string;
  payload: {
    state_id: string;
    observation?: ArtifactRef;
    logical_state?: ArtifactRef;
    [key: string]: unknown;
  };
  created_at?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  branch_member_id: string | null;
  outcome: string;
  action: { kind: string; [key: string]: unknown };
  verification_run_id: string;
  proof_bundle_id: string;
  operation_id?: string;
  action_artifact_id?: string;
  runtime_cursor_id?: string;
  cursor_version?: number;
  created_at?: string;
}

export interface RunRolloutTree {
  rollout_tree_id: string;
  collection_batch_id: string;
  task_revision: string;
  root_state_id: string;
  status: string;
  digest: string;
  created_at: string;
  collection_status: string;
  collection_created_at: string;
  state_count: number;
  transition_count: number;
  sibling_count: number;
  excluded_count: number;
  exception_count: number;
}

export interface RolloutGraph {
  tree: {
    rollout_tree_id?: string;
    collection_batch_id?: string;
    run_id?: string;
    run_name?: string;
    algorithm?: RunSummary["algorithm"];
    status?: string;
    [key: string]: unknown;
  };
  nodes: GraphState[];
  edges: GraphEdge[];
  branch_groups: Array<Record<string, unknown>>;
  branch_members: Array<{
    branch_member_id: string;
    branch_group_id: string;
    sibling_index: number;
    status: string;
    failure_mode: string | null;
    eligibility_status?: string | null;
    eligibility_reason?: string | null;
    terminal_outcome?: string | null;
    verification_status?: string | null;
    retry_count?: number;
  }>;
  decision_checkpoints: Array<Record<string, unknown>>;
  environment_snapshots: Array<Record<string, unknown>>;
  accessible_outline: Array<Record<string, unknown>>;
}

export interface VerificationStep {
  step_run_id: string;
  step_id: string;
  step_type: string;
  status: string;
  attempt_count: number;
  cache_status: string;
  evidence_roles: string[];
  metrics: Record<string, unknown>;
  failure: Record<string, unknown> | null;
  cost: Record<string, number>;
}

export interface VerificationDetail {
  verification_run: Record<string, unknown>;
  proof_bundle: { manifest: ProofBundle };
  steps: VerificationStep[];
  judge: {
    result: {
      outcome: string;
      confidence: number;
      abstained: boolean;
      disagreement: boolean;
      integrity_flags: string[];
      assessments: Array<{
        criterion: string;
        score: number;
        confidence: number;
        evidence_roles: string[];
      }>;
      explanation: string;
      provider_model_identity: string;
      usage: Record<string, number>;
    };
    spec: Record<string, unknown>;
  } | null;
  model_assessment_notice: string;
}
