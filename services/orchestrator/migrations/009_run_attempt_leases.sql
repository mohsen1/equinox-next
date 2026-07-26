ALTER TABLE run_attempts
  ADD COLUMN IF NOT EXISTS claim_id text,
  ADD COLUMN IF NOT EXISTS lease_owner text,
  ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz,
  ADD COLUMN IF NOT EXISTS fencing_token bigint NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS next_eligible_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS retry_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_retries integer NOT NULL DEFAULT 3;

CREATE UNIQUE INDEX IF NOT EXISTS run_attempts_claim_id_key
  ON run_attempts(claim_id)
  WHERE claim_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS run_attempts_claimable_idx
  ON run_attempts(status, next_eligible_at, lease_expires_at, created_at);

ALTER TABLE run_attempts
  ADD CONSTRAINT run_attempts_retry_bounds_check
  CHECK (
    retry_count >= 0
    AND max_retries >= 0
    AND retry_count <= max_retries
  );
