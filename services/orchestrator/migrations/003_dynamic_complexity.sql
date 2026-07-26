CREATE TABLE IF NOT EXISTS complexity_states (
  run_id text PRIMARY KEY REFERENCES runs(run_id),
  environment_id text NOT NULL,
  strategy text NOT NULL CHECK (strategy = 'adaptive'),
  minimum_level integer NOT NULL CHECK (minimum_level >= 0),
  current_level integer NOT NULL CHECK (current_level >= minimum_level),
  maximum_level integer NOT NULL CHECK (maximum_level >= current_level),
  sampling_band integer NOT NULL CHECK (sampling_band > 0),
  mastery_threshold double precision NOT NULL
    CHECK (mastery_threshold > 0 AND mastery_threshold <= 1),
  evaluation_window integer NOT NULL CHECK (evaluation_window > 0),
  promotion_step integer NOT NULL CHECK (promotion_step > 0),
  window_attempts integer NOT NULL DEFAULT 0 CHECK (window_attempts >= 0),
  window_successes integer NOT NULL DEFAULT 0
    CHECK (window_successes >= 0 AND window_successes <= window_attempts),
  promotion_count integer NOT NULL DEFAULT 0 CHECK (promotion_count >= 0),
  last_accuracy double precision,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS complexity_observations (
  operation_id text PRIMARY KEY,
  run_id text NOT NULL REFERENCES runs(run_id),
  request_digest text NOT NULL,
  level integer NOT NULL CHECK (level >= 0),
  successes integer NOT NULL CHECK (successes >= 0),
  attempts integer NOT NULL CHECK (attempts > 0 AND successes <= attempts),
  promoted boolean NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS complexity_observations_run_id_idx
  ON complexity_observations(run_id, created_at);

INSERT INTO schema_migrations(version) VALUES (3) ON CONFLICT DO NOTHING;
