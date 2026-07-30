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

export interface ResearchObserverEvidence {
  phase?: string;
  message?: string;
  profile_id?: string;
  head_commit?: string;
  source_contract_digest?: string;
  bootstrap_source_digest?: string;
  workload_bundle_digest?: string;
  workload_bundle_size_bytes?: number;
  workload_bundle_path?: string;
  bundle_stage_receipt_digest?: string;
  volume_readiness_receipt_digest?: string;
  torch_retention_evidence_digest?: string;
  bundle_activation_digest?: string;
  artifact_set_manifest_digest?: string;
  artifact_set_committed?: boolean;
  network_volume_id?: string;
  network_volume_data_center_id?: string;
  network_volume_size_gb?: number;
  model_snapshot_digest?: string;
  pinned_snapshot_digest?: string;
  model_revision?: string;
  gate_results?: Record<string, boolean>;
  screen_completed?: boolean;
  eligible?: boolean;
  larger_model_eligible?: boolean;
  evaluation_completed?: number;
  evaluation_total?: number;
  branch_groups_completed?: number;
  branch_groups_total?: number;
  update?: number;
  maximum_updates?: number;
  current_level?: number;
  maximum_level?: number;
}

export interface ResearchComputeExecution {
  execution_id: string;
  name: string;
  workload_id: string;
  model_id: string | null;
  branch_width: 1 | 4;
  complexity_strategy: "adaptive";
  status: ResearchComputeStatus;
  provider_name: "RunPod";
  provider_handle: string | null;
  resource_profile: {
    gpu_id?: string;
    cloud_type?: string;
    image?: string;
    image_digest?: string;
    hourly_cost_usd?: number;
    maximum_hourly_cost_usd?: number;
    profile_id?: string;
    head_commit?: string;
    source_contract_digest?: string;
    bootstrap_source_digest?: string;
    workload_bundle_digest?: string;
    workload_bundle_size_bytes?: number;
    workload_bundle_path?: string;
    bundle_stage_receipt_digest?: string;
    volume_readiness_receipt_digest?: string;
    torch_retention_evidence_digest?: string;
    bundle_activation_digest?: string;
    artifact_set_manifest_digest?: string;
    artifact_set_committed?: boolean;
    network_volume_id?: string;
    network_volume_data_center_id?: string;
    network_volume_size_gb?: number;
    [key: string]: unknown;
  };
  allocated_gpu?: string | null;
  cost?: EstimatedComputeCost;
  observer_evidence?: ResearchObserverEvidence;
  artifact_publication_required?: boolean;
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
    initial_level_exact_rate?: number;
    final_exact_rate?: number;
    reward_gain?: number;
    hypothesis_passed?: boolean;
    meaningful_post_training?: boolean;
    post_training_outcome?: string;
    dynamic_complexity_progressed?: boolean;
    adapter_persisted?: boolean;
    post_training_completed?: boolean;
    informative_group_rate?: number;
    policy_update_count?: number;
    attempted_policy_update_count?: number;
    effective_policy_update_count?: number;
    retained_policy_update_count?: number;
    retained_checkpoint_update?: number;
    retention_rollback_count?: number;
    retention_transaction_revision?: string;
    optimizer_update_count?: number;
    pending_informative_group_count?: number;
    pending_informative_group_ids?: string[];
    pending_policy_example_count?: number;
    pending_training_example_count?: number;
    frontier_probe_task_groups?: number;
    maximum_sampled_complexity_level?: number;
    action_protocol_validity_rate?: number;
    recent_malformed_action_rate?: number;
    consecutive_uninformative_groups?: number;
    regression_streak?: number;
    training_remaining_seconds?: number;
    final_evaluation_reserve_seconds?: number;
    provider_remaining_seconds?: number;
    rollback_applied?: boolean;
    baseline_validation?: ResearchValidationSummary;
    best_validation?: ResearchValidationSummary;
    validation_history?: ResearchValidationSummary[];
    curriculum_history?: ResearchTrajectoryPromotion[];
    active_complexity?: ResearchComplexity;
    evaluation_completed?: number;
    evaluation_total?: number;
    claim_strength?: string;
    seed_count?: number;
    objective?: string;
    error?: string;
    screen_completed?: boolean;
    eligible?: boolean;
    larger_model_eligible?: boolean;
    larger_model_profile_id?: string;
    profile_id?: string;
    head_commit?: string;
    source_contract_digest?: string;
    bootstrap_source_digest?: string;
    workload_bundle_digest?: string;
    workload_bundle_size_bytes?: number;
    workload_bundle_path?: string;
    bundle_stage_receipt_digest?: string;
    volume_readiness_receipt_digest?: string;
    torch_retention_evidence_digest?: string;
    bundle_activation_digest?: string;
    artifact_set_manifest_digest?: string;
    artifact_set_committed?: boolean;
    network_volume_id?: string;
    network_volume_data_center_id?: string;
    network_volume_size_gb?: number;
    model_snapshot_digest?: string;
    pinned_snapshot_digest?: string;
    model_revision?: string;
    checkpoint_admission_rate?: number;
    branch_checkpoint_rate?: number;
    peak_gpu_memory_gb?: number;
    peak_cuda_memory_gb?: number;
    peak_gpu_memory_bytes?: number;
    peak_cuda_memory_bytes?: number;
    peak_reserved_vram_gb?: number;
    peak_reserved_vram_bytes?: number;
    peak_reserved_vram_fraction?: number;
    policy_mutation_detected?: boolean;
    policy_mutation_verified?: boolean;
    no_policy_mutation_verified?: boolean;
    policy_parameters_unchanged?: boolean;
    policy_unchanged?: boolean;
    policy_mutation_enabled?: boolean;
    gate_results?: Record<string, boolean>;
    branch_groups_completed?: number;
    branch_groups_total?: number;
    [key: string]: unknown;
  };
  proof_id: string | null;
  receipt_digest: string | null;
  failure_receipt_digest: string | null;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  teardown_confirmed: boolean;
}

export interface EstimatedComputeCost {
  total_usd: number | null;
  estimated: boolean;
  hourly_rate_usd: number | null;
  elapsed_seconds: number | null;
}

export interface ResearchStudySummary {
  study_id: string;
  report_id: string | null;
  generated_at: string;
  overall_status: "PASS" | "FAIL";
  workload_revision: string | null;
  model_id: string | null;
  condition_count: number;
  execution_count: number;
  estimated_provider_cost_usd: number;
  decisions: Record<string, "PASS" | "FAIL">;
}

export interface ResearchStudyCondition {
  role?: string;
  status?: string;
  seed?: number | null;
  branch_width?: number | null;
  policy_mutation_enabled?: boolean;
  sampled_completions?: number | null;
  policy_updates?: number | null;
  optimizer_updates?: number | null;
  initial_successes?: number | null;
  final_successes?: number | null;
  gain?: number | null;
  paired_improved?: number | null;
  paired_regressed?: number | null;
  paired_p_value?: number | null;
}

export interface ResearchStudyReport {
  study_id: string;
  report_id: string;
  generated_at: string;
  overall_status: "PASS" | "FAIL";
  aggregation_policy: string;
  report_digest: string;
  freeze: {
    frozen_workload_revision?: string;
    frozen_source_commit?: string;
    model?: { id?: string; revision?: string };
  };
  conditions: Record<string, ResearchStudyCondition>;
  decisions: Record<
    string,
    {
      status: "PASS" | "FAIL";
      evidence: Record<string, unknown>;
    }
  >;
  external_evaluation?: {
    pack_id?: string;
    task_count?: number;
    domain_task_counts?: Record<string, number>;
  };
  failure_count: {
    provider_executions: number;
    operator_attempts: number;
  };
  executions: Array<{
    execution_id: string;
    condition_id?: string | null;
    outcome: string;
    gpu_id?: string | null;
    estimated_cost_usd?: number | null;
    teardown_confirmed: boolean;
  }>;
}

export interface ResearchProofSummary {
  proof_id: string;
  execution_id: string | null;
  run_name: string | null;
  completed_at: string;
  learning: {
    initial_reward?: number | null;
    final_reward?: number | null;
    reward_gain?: number | null;
    hypothesis_passed?: boolean | null;
    meaningful_post_training?: boolean | null;
    post_training_outcome?: string | null;
    claim_strength?: string | null;
    seed_count?: number | null;
  };
  hardware: {
    provider: string;
    gpu: string | null;
    image?: string | null;
    cloud_type?: string | null;
    hourly_rate_usd: number | null;
  };
  cost: EstimatedComputeCost;
  teardown_confirmed: boolean;
}

export interface ResearchProofDetail extends ResearchProofSummary {
  started_at: string;
  runtime_seconds: number | null;
  provider: {
    name: string;
    handle: string;
    cli_version: string;
  };
  workload: {
    id?: string | null;
    revision?: string | null;
    algorithm?: string | null;
    objective?: string | null;
    model_id?: string | null;
    model_revision?: string | null;
    branch_width?: number | null;
    complexity_strategy?: string | null;
    task_domains?: string[] | null;
    snapshot_fidelity?: string | null;
    multi_step?: boolean | null;
    restored_continuations?: boolean | null;
  };
  curriculum: {
    promotion_count?: number | null;
    dynamic_complexity_progressed?: boolean | null;
    promotions?: Array<{
      update?: number;
      from_level?: number;
      to_level?: number;
      exact_rate?: number;
    }> | null;
    reached_level?: number | null;
    maximum_level?: number | null;
    updates_completed?: number | null;
    stop_reason?: string | null;
    retention_passed?: boolean | null;
  };
  evidence: {
    receipt_digest: string;
    failure_receipt_digest: string | null;
    teardown_confirmed: boolean;
    artifact_set_manifest_digest: string | null;
    artifact_set_committed: boolean;
    artifact_publication_status: "committed" | "legacy_non_atomic";
  };
}

export interface ResearchLevelObservation {
  level: number;
  examples: number;
  exact_rate: number;
  format_rate?: number;
  checkpoint_rate?: number;
  mean_actions?: number;
  mean_reward: number;
  per_domain?: Record<
    string,
    {
      exact_rate: number;
      mean_reward: number;
    }
  >;
}

export interface ResearchComplexity {
  level: number;
  file_count: number;
  fault_count: number;
  dependency_depth: number;
  repair_horizon: number;
}

export interface ResearchValidationSummary {
  update?: number;
  level?: number;
  examples?: number;
  exact_successes?: number;
  exact_rate?: number;
  exact_rate_95ci?: [number, number];
  checkpoint_successes?: number;
  checkpoint_rate?: number;
  checkpoint_rate_95ci?: [number, number];
  action_protocol_validity_rate?: number;
  mastered?: boolean;
  mastery_streak?: number;
  regression_streak?: number;
  best_checkpoint?: number;
  curriculum_baseline_exact_successes?: number;
  curriculum_baseline_exact_rate?: number;
  curriculum_exact_successes?: number;
  curriculum_exact_rate?: number;
  curriculum_paired_change?: {
    examples: number;
    improved: number;
    regressed: number;
    unchanged: number;
    net_improved: number;
    mcnemar_exact_p_value: number;
  };
  fixed_guard_levels?: number[];
  fixed_guard_paired_change?: {
    examples: number;
    improved: number;
    regressed: number;
    unchanged: number;
    net_improved: number;
    mcnemar_exact_p_value: number;
  };
  retention_guard_passed?: boolean;
  checkpoint_candidate_retained?: boolean;
  retention_transaction_disposition?: "retain" | "provisional" | "rollback";
  attempted_policy_update_count?: number;
  effective_policy_update_count?: number;
  retained_policy_update_count?: number;
  retention_rollback_count?: number;
  elapsed_seconds?: number;
}

export interface ResearchTrajectoryCheckpoint extends ResearchLevelObservation {
  update: number;
  training_branch_pass_rate?: number;
  loss?: number;
  policy_loss?: number;
  gradient_norm?: number;
  informative_group_rate?: number;
  policy_update_count?: number;
  attempted_policy_update_count?: number;
  effective_policy_update_count?: number;
  retained_policy_update_count?: number;
  retention_rollback_count?: number;
  retention_transaction_disposition?: "retain" | "provisional" | "rollback";
  mastery_streak?: number;
  regression_streak?: number;
  best_checkpoint?: number;
  elapsed_seconds?: number;
}

export interface ResearchTrajectoryPromotion {
  update: number;
  from_level: number;
  to_level: number;
  exact_rate: number;
  minimum_domain_exact_rate?: number;
  checkpoint_rate?: number;
  mastery_windows: number;
  reason?: string;
  from_complexity?: ResearchComplexity;
  to_complexity?: ResearchComplexity;
  changed_dimensions?: Record<string, { from: number; to: number }>;
  replay_probability?: number;
  minimum_level?: number;
  maximum_level?: number;
}

export interface ResearchBranchStep {
  step_id: string;
  index: number;
  tool: string | null;
  action: Record<string, string> | null;
  accepted: boolean;
  observation: string;
  state_digest_before: string;
  state_digest_after: string;
  verifier_passed: boolean;
  fixed_faults: number;
  total_faults: number;
  terminal: boolean;
  terminal_reason: string | null;
  reward: number;
  policy_signal?: boolean;
  effective_batch_weight?: number;
}

export interface ResearchBranchSibling {
  index: number;
  response?: string;
  action?: string | null;
  format_valid?: boolean;
  passed: boolean;
  reward?: number;
  return?: number;
  advantage: number;
  policy_signal: boolean;
  sampling_seed?: number;
  terminal_reason?: string | null;
  trajectory_digest?: string;
  completion_tokens?: number;
  effective_batch_weight?: number;
  reward_components?: {
    hidden_correctness: boolean;
    public_verifier_progress: number;
    accepted_action_cost: number;
    token_cost: number;
    verifier_submission_cost: number;
    malformed_action_penalty: number;
    terminal_aggregate: number;
    accepted_action_count: number;
    malformed_action_count: number;
    verifier_submission_count: number;
  };
  failure_classification?: Array<{
    category: string;
    source: "typed" | "heuristic";
  }>;
  steps?: ResearchBranchStep[];
}

export interface ResearchComparisonCondition {
  schema_version: 1;
  condition_id: string;
  short_label: string;
  prefix_topology: "shared" | "independent";
  branch_width: 1 | 4;
  group_credit: string;
  shared_prefix: boolean;
}

export interface ResearchBranchSnapshot {
  schema_version?: 1 | 2;
  snapshot_id: string;
  update: number;
  collection_index?: number;
  collection_count?: number;
  level: number;
  domain: string;
  task_id?: string;
  task?: {
    description: string;
    known_failing_tests: string[];
    family_ids?: string[];
    complexity: {
      level: number;
      file_count: number;
      fault_count: number;
      dependency_depth: number;
      repair_horizon: number;
    };
  };
  checkpoint?: {
    checkpoint_id: string;
    payload_digest: string;
    fidelity: string;
    environment_revision: string;
    verifier_revision: string;
    action_protocol_revision: string;
    static_branch_width: 1 | 4;
  } | null;
  shared_prefix?: {
    policy_generated: boolean;
    completion_tokens?: number;
    accepted_diagnostic_actions: number;
    steps: ResearchBranchStep[];
  } | null;
  initial_state?: {
    state_id: string;
    payload_digest: string;
    fidelity: string;
    role: "matched_task_initial_state_not_decision_checkpoint";
  } | null;
  rollout_topology?: {
    revision: string;
    static_group_width: 1 | 4;
    independent_model_generated_prefixes: boolean;
    shared_model_generated_prefix: boolean;
    sibling_group_relative_credit: boolean;
    initial_state_matching: string;
  };
  comparison_condition?: ResearchComparisonCondition;
  prompt?: string;
  expected_action?: string;
  best_sibling_index: number | null;
  learning_signal: boolean;
  excluded?: boolean;
  exclusion_reason?: string | null;
  replay?: boolean;
  curriculum_role?: string;
  sampled_completion_tokens?: number;
  optimizer_update?: {
    update?: number;
    applied: boolean;
    policy_signal_applied?: boolean;
    reference_anchor_applied?: boolean;
    objective_id?: string;
    adapter_revision?: string;
    retention_transaction_revision?: string | null;
    retention_lineage_status?: "pending" | "retained" | "rolled_back" | null;
    retention_transaction_disposition?:
      | "retain"
      | "provisional"
      | "rollback"
      | null;
    retention_resolution_update?: number;
    retention_resolution_reason?: string;
    attempted_policy_update_index?: number | null;
    effective_policy_update_count_after_apply?: number | null;
    retained_policy_update_count_before_validation?: number | null;
    effective_policy_update_count_after_resolution?: number;
    retained_policy_update_count_after_resolution?: number;
    retention_rollback_count_after_resolution?: number;
    learning_rate?: number;
    policy_loss?: number;
    reinforce_loss?: number;
    reference_kl?: number;
    reference_kl_coefficient?: number;
    reference_anchor_scope?: string;
    policy_credit_scope?: string;
    failed_sibling_policy_weight?: number | "signed_trajectory_advantage";
    gradient_norm?: number;
    training_examples?: number;
    reference_examples?: number;
    effective_batch_weight?: number;
    informative_group_count?: number;
    minimum_informative_groups?: number;
    policy_signal_group_count?: number;
    policy_signal_group_ids?: string[];
    optimizer_input_group_count?: number;
    optimizer_input_group_ids?: string[];
    pending_informative_group_count?: number;
    pending_informative_group_ids?: string[];
    pending_optimizer_input_group_count?: number;
    pending_optimizer_input_group_ids?: string[];
    pending_policy_examples?: number;
    pending_training_examples?: number;
    optimizer_input_consumed_by_update?: number;
    policy_signal_consumed_by_update?: number;
    policy_signal_suppressed_reason?: string | null;
  } | null;
  siblings: ResearchBranchSibling[];
}

export interface ResearchPolicyUpdateLineage {
  schema_version: 1 | 2;
  attempted_policy_update_index: number;
  update: number;
  adapter_revision?: string;
  objective_id?: string;
  retention_transaction_revision?: string | null;
  retention_lineage_status: "pending" | "retained" | "rolled_back" | null;
  retention_transaction_disposition?:
    | "retain"
    | "provisional"
    | "rollback"
    | null;
  retention_resolution_update?: number;
  retention_resolution_reason?: string;
  effective_policy_update_count_after_apply: number;
  retained_policy_update_count_before_validation: number;
  effective_policy_update_count_after_resolution?: number;
  retained_policy_update_count_after_resolution?: number;
  retention_rollback_count_after_resolution?: number;
  policy_signal_group_count: number;
  policy_signal_group_ids: string[];
  optimizer_input_group_count?: number;
  optimizer_input_group_ids: string[];
  branch_snapshot_ids: string[];
  training_examples?: number;
  reference_examples?: number;
  policy_loss?: number;
  reinforce_loss?: number;
  reference_kl?: number;
  gradient_norm?: number;
}

export interface ResearchTrajectory {
  schema_version: 1 | 2;
  phase?: string;
  message?: string;
  artifact_set_manifest_digest?: string;
  artifact_set_committed?: boolean;
  branch_width: 1 | 4;
  complexity_strategy: "adaptive";
  multi_step?: boolean;
  restored_continuations?: boolean;
  comparison_condition?: ResearchComparisonCondition;
  rollout_topology?: string;
  prefix_gradient?: boolean | null;
  replay_enabled?: boolean | null;
  environment_revision?: string | null;
  verifier_revision?: string | null;
  action_protocol_revision?: string | null;
  snapshot_fidelity?: string | null;
  maximum_level?: number;
  reached_level?: number;
  maximum_sampled_level?: number;
  updates_completed?: number;
  initial_exact_rate?: number;
  final_exact_rate?: number;
  exact_gain?: number;
  stop_reason?: string;
  best_validation?: ResearchValidationSummary;
  rollback_applied?: boolean;
  action_protocol_validity_rate?: number;
  checkpoints: ResearchTrajectoryCheckpoint[];
  promotions: ResearchTrajectoryPromotion[];
  branch_snapshots: ResearchBranchSnapshot[];
  evaluation_completed?: number;
  evaluation_total?: number;
  branch_groups_completed?: number;
  branch_groups_total?: number;
  branch_evidence_complete?: boolean;
  branch_evidence_group_count?: number;
  branch_evidence_limit?: number;
  branch_evidence_payload_bytes?: number;
  branch_evidence_payload_limit_bytes?: number;
  policy_update_lineage?: ResearchPolicyUpdateLineage[];
  initial_by_level: Record<string, ResearchLevelObservation>;
  final_by_level: Record<string, ResearchLevelObservation>;
  policy_update_count?: number;
  attempted_policy_update_count?: number;
  effective_policy_update_count?: number;
  retained_policy_update_count?: number;
  retained_checkpoint_update?: number;
  retention_rollback_count?: number;
  retention_transaction_revision?: string;
  optimizer_update_count?: number;
  pending_informative_group_count?: number;
  pending_informative_group_ids?: string[];
  pending_optimizer_input_group_count?: number;
  pending_optimizer_input_group_ids?: string[];
  pending_policy_example_count?: number;
  pending_training_example_count?: number;
  frontier_probe_task_groups?: number;
  informative_group_rate?: number;
  total_sampled_completions?: number;
  total_sampled_actions?: number;
  total_sampled_completion_tokens?: number;
  total_post_branch_actions?: number;
  discarded_sampled_completion_tokens?: number;
}

export interface ResearchTrajectoryResponse {
  execution: ResearchComputeExecution;
  trajectory: ResearchTrajectory | null;
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
