CREATE TABLE IF NOT EXISTS research_compute_executions (
  execution_id text PRIMARY KEY,
  name text NOT NULL,
  workload_id text NOT NULL,
  model_id text,
  branch_width integer NOT NULL CHECK (branch_width = 4),
  complexity_strategy text NOT NULL CHECK (complexity_strategy = 'adaptive'),
  status text NOT NULL CHECK (
    status IN ('PROVISIONING', 'RUNNING', 'FINALIZING', 'SUCCEEDED', 'FAILED')
  ),
  provider_name text NOT NULL CHECK (provider_name = 'RunPod'),
  provider_handle text UNIQUE,
  resource_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  progress jsonb NOT NULL DEFAULT '{}'::jsonb,
  proof_id text REFERENCES research_compute_proofs(proof_id),
  receipt_digest text,
  started_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  teardown_confirmed boolean NOT NULL DEFAULT false,
  CHECK (
    (status IN ('SUCCEEDED', 'FAILED')) = (completed_at IS NOT NULL)
  ),
  CHECK (completed_at IS NULL OR completed_at >= started_at),
  CHECK (status != 'SUCCEEDED' OR teardown_confirmed)
);

CREATE INDEX IF NOT EXISTS research_compute_executions_status_idx
  ON research_compute_executions(status, updated_at DESC);

INSERT INTO schema_migrations(version) VALUES (5) ON CONFLICT DO NOTHING;
