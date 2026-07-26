ALTER TABLE jobs
  ADD COLUMN IF NOT EXISTS retry_policy_id text NOT NULL DEFAULT 'bounded-exponential@1',
  ADD COLUMN IF NOT EXISTS next_eligible_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS deadline_at timestamptz,
  ADD COLUMN IF NOT EXISTS result_digest text;

ALTER TABLE attempts
  ADD COLUMN IF NOT EXISTS lease_owner text,
  ADD COLUMN IF NOT EXISTS failure_class text,
  ADD COLUMN IF NOT EXISTS usage jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS jobs_reclaim_idx
  ON jobs(status, next_eligible_at, lease_expires_at);

CREATE INDEX IF NOT EXISTS attempts_active_lease_idx
  ON attempts(job_id, status, fencing_token);
