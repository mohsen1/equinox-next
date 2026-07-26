ALTER TABLE run_attempts
  DROP CONSTRAINT IF EXISTS run_attempts_provider_name_check;

ALTER TABLE compute_allocations
  DROP CONSTRAINT IF EXISTS compute_allocations_provider_name_check;

UPDATE run_attempts
SET provider_name = 'LocalFixtureComputeProvider'
WHERE provider_name = 'MockRunPodProvider';

UPDATE compute_allocations
SET provider_name = 'LocalFixtureComputeProvider'
WHERE provider_name = 'MockRunPodProvider';

ALTER TABLE run_attempts
  ADD CONSTRAINT run_attempts_provider_name_check
  CHECK (provider_name = 'LocalFixtureComputeProvider');

ALTER TABLE compute_allocations
  ADD CONSTRAINT compute_allocations_provider_name_check
  CHECK (provider_name = 'LocalFixtureComputeProvider');
