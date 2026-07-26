from __future__ import annotations

from dataclasses import dataclass

from equinox_core import make_id


@dataclass(frozen=True)
class FixtureAllocation:
    allocation_id: str
    provider_handle: str
    resource_profile: str


class LocalFixtureComputeProvider:
    name = "LocalFixtureComputeProvider"

    def allocate(self, resource_profile: str) -> FixtureAllocation:
        allocation_id = make_id("allocation")
        return FixtureAllocation(
            allocation_id=allocation_id,
            provider_handle=f"fixture://policy/{allocation_id}",
            resource_profile=resource_profile,
        )

    def release(self, _: str) -> None:
        return


POLICY_COMPUTE_PROVIDERS = {
    "LocalFixtureComputeProvider": LocalFixtureComputeProvider()
}
JUDGE_PROVIDER_NAMES = {"DeterministicJudgeFixture"}


def assert_local_registry() -> None:
    if set(POLICY_COMPUTE_PROVIDERS) != {"LocalFixtureComputeProvider"}:
        raise RuntimeError("local profile contains a non-fixture policy-compute provider")
    if {"DeterministicJudgeFixture"} != JUDGE_PROVIDER_NAMES:
        raise RuntimeError("local profile contains a non-fixture judge provider")
