CREATE TABLE IF NOT EXISTS policy_decisions (
  policy_decision_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  branch_member_id text REFERENCES branch_members(branch_member_id),
  behavior_policy_version_id text NOT NULL REFERENCES policy_versions(policy_version_id),
  source_state_id text NOT NULL REFERENCES states(state_id),
  observation_artifact_id text NOT NULL REFERENCES artifacts(artifact_id),
  action_artifact_id text NOT NULL REFERENCES artifacts(artifact_id),
  decision_kind text NOT NULL CHECK (
    decision_kind IN ('FIXTURE_SCRIPT', 'MODEL_SAMPLE', 'OPERATOR')
  ),
  decision_digest text NOT NULL,
  rng_receipt jsonb NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE NULLS NOT DISTINCT (
    rollout_tree_id,
    branch_member_id,
    source_state_id
  ),
  UNIQUE (decision_digest)
);

ALTER TABLE transitions
  ADD COLUMN IF NOT EXISTS policy_decision_id text REFERENCES policy_decisions(policy_decision_id);

CREATE UNIQUE INDEX IF NOT EXISTS transitions_policy_decision_id_key
  ON transitions(policy_decision_id)
  WHERE policy_decision_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS collection_candidates (
  collection_candidate_id text PRIMARY KEY,
  collection_batch_id text NOT NULL REFERENCES collection_batches(collection_batch_id),
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  branch_member_id text REFERENCES branch_members(branch_member_id),
  terminal_transition_id text NOT NULL REFERENCES transitions(transition_id),
  eligibility_decision_id text NOT NULL REFERENCES eligibility_decisions(decision_id),
  proof_bundle_id text NOT NULL REFERENCES evidence_bundles(proof_bundle_id),
  candidate_digest text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE NULLS NOT DISTINCT (
    collection_batch_id,
    rollout_tree_id,
    branch_member_id
  ),
  UNIQUE (eligibility_decision_id),
  UNIQUE (candidate_digest)
);

CREATE TABLE IF NOT EXISTS collection_memberships (
  collection_membership_id text PRIMARY KEY,
  collection_batch_id text NOT NULL REFERENCES collection_batches(collection_batch_id),
  collection_candidate_id text NOT NULL REFERENCES collection_candidates(collection_candidate_id),
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  weight double precision NOT NULL CHECK (
    weight > 0
    AND weight <> 'Infinity'::double precision
    AND weight <> 'NaN'::double precision
  ),
  membership_digest text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (collection_batch_id, collection_candidate_id),
  UNIQUE (collection_batch_id, ordinal),
  UNIQUE (membership_digest)
);

CREATE TABLE IF NOT EXISTS collection_closures (
  collection_closure_id text PRIMARY KEY,
  collection_batch_id text NOT NULL UNIQUE REFERENCES collection_batches(collection_batch_id),
  closure_digest text NOT NULL UNIQUE,
  member_count integer NOT NULL CHECK (member_count > 0),
  manifest_artifact_id text NOT NULL REFERENCES artifacts(artifact_id),
  manifest jsonb NOT NULL,
  closed_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE iteration_inputs
  ADD COLUMN IF NOT EXISTS collection_closure_id text REFERENCES collection_closures(collection_closure_id),
  ADD COLUMN IF NOT EXISTS dataset_artifact_id text REFERENCES artifacts(artifact_id),
  ADD COLUMN IF NOT EXISTS dataset_digest text,
  ADD COLUMN IF NOT EXISTS dataset_row_count integer CHECK (dataset_row_count > 0);

ALTER TABLE training_iterations
  ADD COLUMN IF NOT EXISTS update_kind text NOT NULL DEFAULT 'SIMULATED_POLICY_COMMIT'
    CHECK (update_kind IN ('SIMULATED_POLICY_COMMIT', 'OPTIMIZED_MODEL'));

ALTER TABLE policy_versions
  ADD COLUMN IF NOT EXISTS update_kind text NOT NULL DEFAULT 'LEGACY_DECLARATION'
    CHECK (
      update_kind IN (
        'INITIAL_FIXTURE',
        'SIMULATED_POLICY_COMMIT',
        'OPTIMIZED_MODEL',
        'LEGACY_DECLARATION'
      )
    );

CREATE INDEX IF NOT EXISTS policy_decisions_policy_idx
  ON policy_decisions(behavior_policy_version_id, created_at);

CREATE INDEX IF NOT EXISTS collection_candidates_batch_idx
  ON collection_candidates(collection_batch_id, rollout_tree_id);

CREATE INDEX IF NOT EXISTS collection_memberships_batch_idx
  ON collection_memberships(collection_batch_id, ordinal);

CREATE UNIQUE INDEX IF NOT EXISTS iteration_inputs_collection_closure_id_key
  ON iteration_inputs(collection_closure_id)
  WHERE collection_closure_id IS NOT NULL;
