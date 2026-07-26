from __future__ import annotations

from dataclasses import dataclass

from equinox_core import make_id


@dataclass(frozen=True)
class MockAllocation:
    allocation_id: str
    provider_handle: str
    resource_profile: str


class MockRunPodProvider:
    name = "MockRunPodProvider"

    def allocate(self, resource_profile: str) -> MockAllocation:
        allocation_id = make_id("allocation")
        return MockAllocation(
            allocation_id=allocation_id,
            provider_handle=f"mock://policy/{allocation_id}",
            resource_profile=resource_profile,
        )

    def release(self, _: str) -> None:
        return


POLICY_COMPUTE_PROVIDERS = {"MockRunPodProvider": MockRunPodProvider()}
JUDGE_PROVIDER_NAMES = {"MockJudgeProvider"}


def assert_local_registry() -> None:
    if set(POLICY_COMPUTE_PROVIDERS) != {"MockRunPodProvider"}:
        raise RuntimeError("local profile contains a non-mock policy-compute provider")
    if {"MockJudgeProvider"} != JUDGE_PROVIDER_NAMES:
        raise RuntimeError("local profile contains a non-mock judge provider")
