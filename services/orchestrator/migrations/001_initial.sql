CREATE TABLE IF NOT EXISTS schema_migrations (
  version integer PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
  run_id text PRIMARY KEY,
  source_run_id text REFERENCES runs(run_id),
  name text NOT NULL,
  algorithm text NOT NULL CHECK (algorithm IN ('independent_rollout_baseline', 'bpo_local_metric')),
  status text NOT NULL CHECK (
    status IN ('QUEUED', 'PROVISIONING', 'PREPARING', 'RUNNING', 'FINALIZING',
               'SUCCEEDED', 'CANCEL_REQUESTED', 'CANCELING', 'CANCELED', 'FAILED')
  ),
  desired_state text NOT NULL DEFAULT 'RUNNING' CHECK (desired_state IN ('RUNNING', 'CANCELED')),
  manifest jsonb NOT NULL,
  manifest_digest text NOT NULL,
  version integer NOT NULL DEFAULT 0,
  cleanup_status text NOT NULL DEFAULT 'NOT_REQUIRED',
  warning jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_attempts (
  attempt_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  attempt_number integer NOT NULL,
  status text NOT NULL,
  heartbeat_at timestamptz,
  provider_name text NOT NULL CHECK (provider_name = 'MockRunPodProvider'),
  allocation_id text,
  failure jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS compute_allocations (
  allocation_id text PRIMARY KEY,
  run_attempt_id text NOT NULL REFERENCES run_attempts(attempt_id),
  provider_name text NOT NULL CHECK (provider_name = 'MockRunPodProvider'),
  desired_state text NOT NULL,
  observed_state text NOT NULL,
  resource_profile text NOT NULL,
  provider_handle text NOT NULL,
  cleanup_warning jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS policy_versions (
  policy_version_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  ordinal integer NOT NULL,
  artifact_digest text NOT NULL,
  behavior_manifest jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, ordinal)
);

CREATE TABLE IF NOT EXISTS collection_batches (
  collection_batch_id text PRIMARY KEY,
  run_attempt_id text NOT NULL REFERENCES run_attempts(attempt_id),
  behavior_policy_version_id text NOT NULL REFERENCES policy_versions(policy_version_id),
  verification_plan_id text NOT NULL,
  judge_spec_id text NOT NULL,
  reward_pipeline_id text NOT NULL,
  status text NOT NULL,
  version integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rollout_trees (
  rollout_tree_id text PRIMARY KEY,
  collection_batch_id text NOT NULL REFERENCES collection_batches(collection_batch_id),
  task_revision text NOT NULL,
  root_state_id text,
  status text NOT NULL,
  digest text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS states (
  state_id text PRIMARY KEY,
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  parent_transition_id text,
  environment_version text NOT NULL,
  task_revision text NOT NULL,
  logical_state_artifact_id text NOT NULL,
  observation_artifact_id text NOT NULL,
  logical_state_digest text NOT NULL,
  semantic_status text NOT NULL,
  sequence integer NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (rollout_tree_id, sequence)
);

CREATE TABLE IF NOT EXISTS transitions (
  transition_id text PRIMARY KEY,
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  branch_member_id text,
  source_state_id text NOT NULL REFERENCES states(state_id),
  destination_state_id text REFERENCES states(state_id),
  runtime_cursor_id text NOT NULL,
  cursor_version integer NOT NULL,
  cursor_fencing_token integer NOT NULL,
  action_artifact_id text NOT NULL,
  operation_id text NOT NULL,
  outcome text NOT NULL CHECK (outcome IN ('CONTINUED', 'TERMINATED', 'TRUNCATED', 'INVALID_ACTION')),
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (operation_id),
  UNIQUE (runtime_cursor_id, cursor_version)
);

CREATE TABLE IF NOT EXISTS environment_snapshots (
  snapshot_id text PRIMARY KEY,
  source_state_id text NOT NULL REFERENCES states(state_id),
  logical_state_artifact_id text NOT NULL,
  logical_state_digest text NOT NULL,
  requested_fidelity text NOT NULL,
  obtained_fidelity text NOT NULL CHECK (obtained_fidelity = 'logical_restore'),
  fidelity_probe_passed boolean NOT NULL CHECK (fidelity_probe_passed),
  rng_state text NOT NULL,
  payload jsonb NOT NULL,
  ephemeral boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS decision_checkpoints (
  checkpoint_id text PRIMARY KEY,
  snapshot_id text NOT NULL REFERENCES environment_snapshots(snapshot_id),
  source_state_id text NOT NULL REFERENCES states(state_id),
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  behavior_policy_version_id text NOT NULL REFERENCES policy_versions(policy_version_id),
  policy_context_artifact_id text NOT NULL,
  turns_remaining integer NOT NULL CHECK (turns_remaining > 0),
  budget_reservation_id text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS branch_groups (
  branch_group_id text PRIMARY KEY,
  checkpoint_id text NOT NULL REFERENCES decision_checkpoints(checkpoint_id),
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  width integer NOT NULL CHECK (width = 4),
  status text NOT NULL,
  presentation_order jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS branch_members (
  branch_member_id text PRIMARY KEY,
  branch_group_id text NOT NULL REFERENCES branch_groups(branch_group_id),
  sibling_index integer NOT NULL CHECK (sibling_index BETWEEN 0 AND 3),
  runtime_cursor_id text NOT NULL,
  rollout_id text NOT NULL,
  status text NOT NULL,
  failure_mode text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (branch_group_id, sibling_index)
);

CREATE TABLE IF NOT EXISTS verification_runs (
  verification_run_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  subject_type text NOT NULL,
  subject_id text NOT NULL,
  plan_id text NOT NULL,
  plan_version integer NOT NULL,
  status text NOT NULL,
  proof_bundle_id text,
  rejudges_verification_run_id text REFERENCES verification_runs(verification_run_id),
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS verifier_step_runs (
  step_run_id text PRIMARY KEY,
  verification_run_id text NOT NULL REFERENCES verification_runs(verification_run_id),
  step_id text NOT NULL,
  step_type text NOT NULL,
  status text NOT NULL,
  attempt_count integer NOT NULL,
  cache_status text NOT NULL,
  evidence_roles jsonb NOT NULL,
  metrics jsonb NOT NULL,
  failure jsonb,
  cost jsonb NOT NULL,
  started_at timestamptz NOT NULL,
  completed_at timestamptz,
  UNIQUE (verification_run_id, step_id)
);

CREATE TABLE IF NOT EXISTS evidence_bundles (
  proof_bundle_id text PRIMARY KEY,
  verification_run_id text NOT NULL REFERENCES verification_runs(verification_run_id),
  subject_type text NOT NULL,
  subject_id text NOT NULL,
  digest text NOT NULL UNIQUE,
  manifest jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS judge_specs (
  judge_spec_id text PRIMARY KEY,
  provider_name text NOT NULL CHECK (provider_name = 'MockJudgeProvider'),
  digest text NOT NULL UNIQUE,
  spec jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS judge_invocations (
  judge_invocation_id text PRIMARY KEY,
  verification_run_id text NOT NULL REFERENCES verification_runs(verification_run_id),
  step_run_id text NOT NULL REFERENCES verifier_step_runs(step_run_id),
  judge_spec_id text NOT NULL REFERENCES judge_specs(judge_spec_id),
  proof_bundle_digest text NOT NULL,
  sample_index integer NOT NULL,
  status text NOT NULL,
  raw_output_artifact_id text,
  accepted_result_id text,
  usage jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (step_run_id, judge_spec_id, proof_bundle_digest, sample_index)
);

CREATE TABLE IF NOT EXISTS judge_results (
  judge_result_id text PRIMARY KEY,
  judge_invocation_id text NOT NULL UNIQUE REFERENCES judge_invocations(judge_invocation_id),
  outcome text NOT NULL,
  result jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS metric_observations (
  metric_observation_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  subject_type text NOT NULL,
  subject_id text NOT NULL,
  descriptor text NOT NULL,
  value jsonb NOT NULL,
  unit text NOT NULL,
  source_step_run_id text NOT NULL REFERENCES verifier_step_runs(step_run_id),
  judge_result_id text REFERENCES judge_results(judge_result_id),
  accepted boolean NOT NULL CHECK (accepted),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_step_run_id, descriptor, subject_id)
);

CREATE TABLE IF NOT EXISTS reward_signals (
  reward_signal_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  subject_type text NOT NULL,
  subject_id text NOT NULL,
  name text NOT NULL,
  value double precision NOT NULL,
  reward_pipeline_id text NOT NULL,
  metric_observation_ids jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (subject_id, name, reward_pipeline_id)
);

CREATE TABLE IF NOT EXISTS eligibility_decisions (
  decision_id text PRIMARY KEY,
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  branch_member_id text REFERENCES branch_members(branch_member_id),
  status text NOT NULL,
  reason_code text NOT NULL,
  version text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE NULLS NOT DISTINCT (rollout_tree_id, branch_member_id, version)
);

CREATE TABLE IF NOT EXISTS iteration_inputs (
  manifest_id text PRIMARY KEY,
  digest text NOT NULL UNIQUE,
  manifest jsonb NOT NULL,
  artifact_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_iterations (
  training_iteration_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  collection_batch_id text NOT NULL REFERENCES collection_batches(collection_batch_id),
  status text NOT NULL,
  input_policy_version_id text NOT NULL REFERENCES policy_versions(policy_version_id),
  output_policy_version_id text REFERENCES policy_versions(policy_version_id),
  iteration_input_id text REFERENCES iteration_inputs(manifest_id),
  expected_policy_ordinal integer NOT NULL,
  commit_operation_id text UNIQUE,
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  committed_at timestamptz
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id text PRIMARY KEY,
  digest text NOT NULL,
  object_key text NOT NULL,
  media_type text NOT NULL,
  size_bytes bigint NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (digest)
);

CREATE TABLE IF NOT EXISTS artifact_refs (
  artifact_ref_id text PRIMARY KEY,
  artifact_id text NOT NULL REFERENCES artifacts(artifact_id),
  entity_type text NOT NULL,
  entity_id text NOT NULL,
  role text NOT NULL,
  ordinal integer NOT NULL DEFAULT 0,
  viewer_hint text,
  visibility text NOT NULL,
  trust_class text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (entity_type, entity_id, role, ordinal)
);

CREATE TABLE IF NOT EXISTS operations (
  operation_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  idempotency_key text NOT NULL UNIQUE,
  request_digest text NOT NULL,
  operation_type text NOT NULL,
  operation_input jsonb NOT NULL,
  expected_version integer NOT NULL,
  status text NOT NULL,
  result_digest text,
  result jsonb,
  budget_reservation_id text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS budget_ledger (
  ledger_entry_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  operation_id text REFERENCES operations(operation_id),
  entry_type text NOT NULL CHECK (entry_type IN ('RESERVE', 'CHARGE', 'RELEASE', 'ADJUST')),
  dimensions jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
  event_id text PRIMARY KEY,
  event_type text NOT NULL,
  schema_version integer NOT NULL,
  aggregate_type text NOT NULL,
  aggregate_id text NOT NULL,
  aggregate_sequence integer NOT NULL,
  run_id text NOT NULL REFERENCES runs(run_id),
  correlation_id text NOT NULL,
  causation_id text,
  producer text NOT NULL,
  payload jsonb NOT NULL,
  occurred_at timestamptz NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (aggregate_type, aggregate_id, aggregate_sequence)
);

CREATE TABLE IF NOT EXISTS outbox (
  outbox_id bigserial PRIMARY KEY,
  event_id text NOT NULL UNIQUE REFERENCES events(event_id),
  topic text NOT NULL,
  payload jsonb NOT NULL,
  published_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_events_run_cursor ON events(run_id, recorded_at, event_id);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(outbox_id) WHERE published_at IS NULL;

INSERT INTO schema_migrations(version) VALUES (1) ON CONFLICT DO NOTHING;

-- Recovery: restore the PostgreSQL volume, verify artifact digests against MinIO, then
-- resume reconciliation. This local migration is intentionally forward-only.
