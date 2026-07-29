ALTER TABLE research_compute_executions
  ADD COLUMN IF NOT EXISTS failure_receipt_digest text;

ALTER TABLE research_compute_executions
  DROP CONSTRAINT IF EXISTS research_compute_executions_failure_receipt_digest_check;

ALTER TABLE research_compute_executions
  ADD CONSTRAINT research_compute_executions_failure_receipt_digest_check
  CHECK (
    failure_receipt_digest IS NULL
    OR failure_receipt_digest ~ '^sha256:[0-9a-f]{64}$'
  );

UPDATE research_compute_executions
SET failure_receipt_digest = receipt_digest
WHERE status = 'FAILED'
  AND proof_id IS NULL
  AND failure_receipt_digest IS NULL
  AND receipt_digest ~ '^sha256:[0-9a-f]{64}$';

INSERT INTO schema_migrations(version) VALUES (12) ON CONFLICT DO NOTHING;
