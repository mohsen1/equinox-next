ALTER TABLE states
  DROP CONSTRAINT IF EXISTS states_rollout_tree_id_sequence_state_id_key;
ALTER TABLE states
  DROP CONSTRAINT IF EXISTS states_rollout_tree_id_sequence_key;
ALTER TABLE states
  ADD CONSTRAINT states_rollout_tree_id_sequence_key
  UNIQUE (rollout_tree_id, sequence);

ALTER TABLE transitions
  ADD COLUMN IF NOT EXISTS cursor_fencing_token integer;
UPDATE transitions
  SET cursor_fencing_token = cursor_version + 2
  WHERE cursor_fencing_token IS NULL;
ALTER TABLE transitions
  ALTER COLUMN cursor_fencing_token SET NOT NULL;

ALTER TABLE eligibility_decisions
  DROP CONSTRAINT IF EXISTS eligibility_decisions_rollout_tree_id_branch_member_id_version_key;
ALTER TABLE eligibility_decisions
  ADD CONSTRAINT eligibility_decisions_rollout_tree_id_branch_member_id_version_key
  UNIQUE NULLS NOT DISTINCT (rollout_tree_id, branch_member_id, version);

INSERT INTO schema_migrations(version) VALUES (2) ON CONFLICT DO NOTHING;
