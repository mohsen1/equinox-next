CREATE TABLE IF NOT EXISTS research_compute_proofs (
  proof_id text PRIMARY KEY,
  provider_name text NOT NULL CHECK (provider_name = 'RunPod'),
  provider_handle text NOT NULL UNIQUE,
  provider_cli_version text NOT NULL,
  resource_profile jsonb NOT NULL,
  workload jsonb NOT NULL,
  result jsonb NOT NULL,
  receipt_digest text NOT NULL,
  started_at timestamptz NOT NULL,
  completed_at timestamptz NOT NULL,
  teardown_confirmed boolean NOT NULL CHECK (teardown_confirmed),
  ingested_at timestamptz NOT NULL DEFAULT now(),
  CHECK (completed_at >= started_at)
);

INSERT INTO schema_migrations(version) VALUES (4) ON CONFLICT DO NOTHING;
