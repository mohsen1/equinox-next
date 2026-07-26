ALTER TABLE snapshots
  DROP CONSTRAINT IF EXISTS snapshots_fidelity_probe_passed_check;

ALTER TABLE snapshots
  ADD COLUMN IF NOT EXISTS captured_cursor_version integer,
  ADD COLUMN IF NOT EXISTS captured_fencing_token bigint,
  ADD COLUMN IF NOT EXISTS fidelity_probe_receipt jsonb;
