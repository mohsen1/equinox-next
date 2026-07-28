ALTER TABLE research_compute_executions
  DROP CONSTRAINT IF EXISTS research_compute_executions_branch_width_check;

ALTER TABLE research_compute_executions
  ADD CONSTRAINT research_compute_executions_branch_width_check
  CHECK (branch_width IN (1, 4));

INSERT INTO schema_migrations(version) VALUES (11) ON CONFLICT DO NOTHING;
