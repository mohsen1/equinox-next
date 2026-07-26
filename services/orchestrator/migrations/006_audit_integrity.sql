ALTER TABLE judge_specs
  DROP CONSTRAINT IF EXISTS judge_specs_provider_name_check;

UPDATE judge_specs
SET provider_name = 'DeterministicJudgeFixture'
WHERE provider_name = 'MockJudgeProvider';

ALTER TABLE judge_specs
  ADD CONSTRAINT judge_specs_provider_name_check
  CHECK (provider_name = 'DeterministicJudgeFixture');
