# ADR-0002: Use a deterministic mock behind the production judge contract

Status: accepted for local contract proof
Date: 2026-07-26

## Decision

The local profile registers only `MockJudgeProvider`. Every invocation uses the same
versioned multimodal request and structured response schemas intended for a future strong
provider. A judge specification pins proof-bundle digest, prompt-template digest, rubric,
schema, sampling, sample index, provider model identity, evidence-role allowlist, and
integrity profile.

The prompt keeps privileged instructions separate from untrusted evidence manifests.
Candidate roles are data, not instructions. The provider receives no tools, browsing,
shell, credentials, hidden calibration cases, or writable product systems.

## Mock behavior

Deterministic fixtures cover valid high and low assessments, tie, abstention, malformed
output, retryable provider failure, disagreement, and integrity violation. Raw provider
output and parsed output are distinct artifacts. Malformed, abstained, disputed, failed,
or integrity-violating results create no quality metric or zero reward.

Pointwise and group results are model assessments. Group inputs blind policy, provider,
branch, and candidate identity and persist the deterministic presentation permutation.

## Calibration and reward

The mock passes contract and recovery conformance only. It cannot receive a
human-alignment or production-ready calibration label. Judge results become accepted
metric observations only after schema and integrity checks; named reward signals are
created later by a deterministic reward-pipeline version.

## Rejudging

A new judge specification over a stored proof bundle creates a new verification run and
new invocation. It does not rerun CAD execution, geometry, or rendering and never mutates
the result referenced by a committed iteration.

## Deferred decision

A real multimodal provider, model, data-governance policy, calibration pack, and
admission threshold must be selected from current primary evidence and measured CAD cases
before CAD research beta. No real adapter ships in the local provider registry.
