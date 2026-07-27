ALTER TABLE rollout_trees
  ADD COLUMN IF NOT EXISTS next_state_sequence integer;

UPDATE rollout_trees rt
SET next_state_sequence = sequences.next_sequence
FROM (
  SELECT
    rollout_tree_id,
    COALESCE(max(sequence), -1) + 1 AS next_sequence
  FROM states
  GROUP BY rollout_tree_id
) AS sequences
WHERE rt.rollout_tree_id = sequences.rollout_tree_id
  AND rt.next_state_sequence IS NULL;

UPDATE rollout_trees
SET next_state_sequence = 1
WHERE next_state_sequence IS NULL;

ALTER TABLE rollout_trees
  ALTER COLUMN next_state_sequence SET DEFAULT 1,
  ALTER COLUMN next_state_sequence SET NOT NULL,
  ADD CONSTRAINT rollout_trees_next_state_sequence_check
    CHECK (next_state_sequence >= 0);

CREATE TABLE IF NOT EXISTS event_sequences (
  aggregate_type text NOT NULL,
  aggregate_id text NOT NULL,
  next_sequence integer NOT NULL CHECK (next_sequence > 0),
  PRIMARY KEY (aggregate_type, aggregate_id)
);

INSERT INTO event_sequences(aggregate_type, aggregate_id, next_sequence)
SELECT aggregate_type, aggregate_id, max(aggregate_sequence) + 1
FROM events
GROUP BY aggregate_type, aggregate_id
ON CONFLICT (aggregate_type, aggregate_id) DO UPDATE
SET next_sequence = GREATEST(
  event_sequences.next_sequence,
  EXCLUDED.next_sequence
);

ALTER TABLE complexity_observations
  ADD COLUMN IF NOT EXISTS collection_closure_id text
    REFERENCES collection_closures(collection_closure_id),
  ADD COLUMN IF NOT EXISTS behavior_policy_version_id text
    REFERENCES policy_versions(policy_version_id),
  ADD COLUMN IF NOT EXISTS task_set_digest text,
  ADD COLUMN IF NOT EXISTS success_definition text,
  ADD COLUMN IF NOT EXISTS ordered_outcomes jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS complexity_observations_closure_key
  ON complexity_observations(collection_closure_id)
  WHERE collection_closure_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS episodes (
  episode_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  collection_batch_id text NOT NULL REFERENCES collection_batches(collection_batch_id),
  rollout_tree_id text NOT NULL REFERENCES rollout_trees(rollout_tree_id),
  branch_member_id text REFERENCES branch_members(branch_member_id),
  behavior_policy_version_id text NOT NULL REFERENCES policy_versions(policy_version_id),
  terminal_transition_id text REFERENCES transitions(transition_id),
  status text NOT NULL CHECK (
    status IN ('ACTIVE', 'TERMINATED', 'TRUNCATED', 'EXCLUDED')
  ),
  terminal_reason text,
  step_count integer NOT NULL CHECK (step_count >= 0),
  return_value double precision CHECK (
    return_value IS NULL
    OR (
      return_value <> 'Infinity'::double precision
      AND return_value <> '-Infinity'::double precision
      AND return_value <> 'NaN'::double precision
    )
  ),
  policy_lag integer NOT NULL DEFAULT 0 CHECK (policy_lag >= 0),
  manifest_digest text NOT NULL,
  manifest jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  UNIQUE NULLS NOT DISTINCT (rollout_tree_id, branch_member_id)
);

ALTER TABLE collection_candidates
  ADD COLUMN IF NOT EXISTS episode_id text REFERENCES episodes(episode_id);

CREATE UNIQUE INDEX IF NOT EXISTS collection_candidates_episode_key
  ON collection_candidates(episode_id)
  WHERE episode_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reward_metric_links (
  reward_signal_id text NOT NULL REFERENCES reward_signals(reward_signal_id),
  metric_observation_id text NOT NULL REFERENCES metric_observations(metric_observation_id),
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  PRIMARY KEY (reward_signal_id, metric_observation_id),
  UNIQUE (reward_signal_id, ordinal)
);

INSERT INTO reward_metric_links(reward_signal_id, metric_observation_id, ordinal)
SELECT
  reward.reward_signal_id,
  source.value,
  source.ordinal - 1
FROM reward_signals AS reward
CROSS JOIN LATERAL jsonb_array_elements_text(
  reward.metric_observation_ids
) WITH ORDINALITY AS source(value, ordinal)
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS events_run_sequence_idx
  ON events(run_id, aggregate_type, aggregate_id, aggregate_sequence);

CREATE INDEX IF NOT EXISTS episodes_run_status_idx
  ON episodes(run_id, status, created_at);
