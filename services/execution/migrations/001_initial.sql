CREATE TABLE IF NOT EXISTS schema_migrations (
  version integer PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS operations (
  operation_id text PRIMARY KEY,
  idempotency_key text NOT NULL UNIQUE,
  request_digest text NOT NULL,
  operation_type text NOT NULL,
  status text NOT NULL,
  request jsonb NOT NULL,
  result jsonb,
  result_digest text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id text PRIMARY KEY,
  operation_id text NOT NULL UNIQUE REFERENCES operations(operation_id),
  profile text NOT NULL,
  status text NOT NULL,
  max_attempts integer NOT NULL,
  attempt_count integer NOT NULL DEFAULT 0,
  lease_owner text,
  lease_expires_at timestamptz,
  fencing_token bigint NOT NULL DEFAULT 0,
  result jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id text PRIMARY KEY,
  job_id text NOT NULL REFERENCES jobs(job_id),
  attempt_number integer NOT NULL,
  fencing_token bigint NOT NULL,
  status text NOT NULL,
  outcome text,
  failure jsonb,
  cost jsonb NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  UNIQUE (job_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id text PRIMARY KEY,
  runtime_instance_id text NOT NULL UNIQUE,
  task_revision text NOT NULL,
  logical_state jsonb NOT NULL,
  current_state_digest text NOT NULL,
  status text NOT NULL,
  session_root text NOT NULL UNIQUE,
  rng_state text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runtime_cursors (
  cursor_id text PRIMARY KEY,
  session_id text NOT NULL REFERENCES sessions(session_id),
  current_scientific_state_id text NOT NULL,
  version integer NOT NULL DEFAULT 0,
  lease_owner text NOT NULL,
  lease_expires_at timestamptz NOT NULL,
  fencing_token bigint NOT NULL,
  observed_status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS snapshots (
  operational_snapshot_id text PRIMARY KEY,
  source_session_id text NOT NULL REFERENCES sessions(session_id),
  source_scientific_state_id text NOT NULL,
  logical_state jsonb NOT NULL,
  logical_state_digest text NOT NULL,
  requested_fidelity text NOT NULL,
  obtained_fidelity text NOT NULL CHECK (obtained_fidelity = 'logical_restore'),
  fidelity_probe_passed boolean NOT NULL CHECK (fidelity_probe_passed),
  rng_state text NOT NULL,
  ephemeral boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accepted_step_results (
  accepted_step_result_id text PRIMARY KEY,
  operation_id text NOT NULL REFERENCES operations(operation_id),
  verification_run_id text NOT NULL,
  step_id text NOT NULL,
  result_digest text NOT NULL,
  result jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (verification_run_id, step_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

INSERT INTO schema_migrations(version) VALUES (1) ON CONFLICT DO NOTHING;

-- Recovery: operational work may be replayed from orchestrator intents. Never reconstruct
-- scientific lineage directly from these tables.
