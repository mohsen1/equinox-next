ALTER TABLE research_compute_executions
  ADD COLUMN IF NOT EXISTS artifact_publication jsonb;

ALTER TABLE research_compute_executions
  ADD COLUMN IF NOT EXISTS artifact_publication_required boolean
  NOT NULL DEFAULT false;

ALTER TABLE research_compute_proofs
  ADD COLUMN IF NOT EXISTS artifact_publication jsonb;

ALTER TABLE research_compute_executions
  DROP CONSTRAINT IF EXISTS research_compute_executions_artifact_publication_check;

ALTER TABLE research_compute_executions
  ADD CONSTRAINT research_compute_executions_artifact_publication_check
  CHECK (
    artifact_publication IS NULL
    OR (
      artifact_publication_required
      AND jsonb_typeof(artifact_publication) = 'object'
    )
  );

ALTER TABLE research_compute_executions
  DROP CONSTRAINT IF EXISTS research_compute_executions_required_publication_check;

ALTER TABLE research_compute_executions
  ADD CONSTRAINT research_compute_executions_required_publication_check
  CHECK (
    NOT artifact_publication_required
    OR status <> 'SUCCEEDED'
    OR artifact_publication IS NOT NULL
  );

ALTER TABLE research_compute_proofs
  DROP CONSTRAINT IF EXISTS research_compute_proofs_artifact_publication_check;

ALTER TABLE research_compute_proofs
  ADD CONSTRAINT research_compute_proofs_artifact_publication_check
  CHECK (
    artifact_publication IS NULL
    OR jsonb_typeof(artifact_publication) = 'object'
  );

INSERT INTO schema_migrations(version) VALUES (13) ON CONFLICT DO NOTHING;
