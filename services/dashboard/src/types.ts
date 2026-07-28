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
    environment?: { snapshot_fidelity?: string };
    task_revision?: string;
    seed?: number;
    [key: string]: unknown;
  };
  manifest_digest: string;
  cleanup_status: string;
  collection_batch_count: number;
  iteration_count: number;
  committed_iteration_count: number;
  rollout_tree_count: number;
  verification_run_count: number;
  proof_count: number;
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
  study: {
    study_id: string;
    condition: string;
    protocol_revision: string;
    research_question: string;
  };
  learning_outcome: string;
  evidence_strength: string;
  summary: string;
}

export interface StudyCondition {
  condition: string;
  seeds: number[];
  run_count: number;
  completed_count: number;
  learning_outcomes: string[];
  test_result: number | null;
  test_result_label: string;
  execution_credits: number;
}

export interface StudySummary {
  study_id: string;
  research_question: string;
  protocol_revision: string;
  run_count: number;
  conditions: StudyCondition[];
  comparison: {
    status: "MATCHED" | "INCOMPLETE";
    paired_seeds: number[];
    protocol_digest: string | null;
    constraints: string[];
    learning_claim: string;
  };
  updated_at: string;
  runs?: RunSummary[];
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
  payload: { state_id: string };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  branch_member_id: string | null;
  lane: string;
  local_step: number;
  outcome: string;
  action: { kind: string };
  action_label: string;
  verification_run_id: string;
  proof_bundle_id: string;
}

export interface RolloutGraph {
  tree: Record<string, unknown>;
  nodes: GraphState[];
  edges: GraphEdge[];
  branch_groups: Array<Record<string, unknown>>;
  branch_members: Array<{
    branch_member_id: string;
    branch_group_id: string;
    sibling_index: number;
    status: string;
    failure_mode: string | null;
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
