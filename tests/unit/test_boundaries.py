from pathlib import Path

from services.orchestrator.app.providers import (
    JUDGE_PROVIDER_NAMES,
    POLICY_COMPUTE_PROVIDERS,
    assert_local_registry,
)


def test_local_provider_registry_has_no_real_provider() -> None:
    assert_local_registry()
    assert set(POLICY_COMPUTE_PROVIDERS) == {"MockRunPodProvider"}
    assert {"MockJudgeProvider"} == JUDGE_PROVIDER_NAMES


def test_prompts_treat_candidate_content_as_untrusted_and_disable_tools() -> None:
    prompt_root = Path("services/execution/prompts")
    text = "\n".join(path.read_text(encoding="utf-8") for path in prompt_root.glob("*.md"))
    lowered = text.lower()
    assert "untrusted" in lowered
    assert "never as instructions" in lowered
    assert "tools" in lowered
    assert "hidden" in lowered
