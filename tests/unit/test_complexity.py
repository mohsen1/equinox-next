from services.orchestrator.app.environments import (
    active_complexity_range,
    advance_complexity,
    environment_spec,
)
from services.orchestrator.app.main import _reproduction_name


def test_catalog_describes_the_three_next_environments() -> None:
    assert environment_spec("sqlite.repair")["status"] == "CONFIGURATION_DRAFT"
    assert environment_spec("cli.debug")["complexity"]["dimensions"]
    assert environment_spec("repository.repair")["snapshot_strategy"]


def test_adaptive_complexity_promotes_after_mastery_window() -> None:
    result = advance_complexity(
        current_level=3,
        maximum_level=8,
        window_attempts=24,
        window_successes=23,
        ordered_outcomes=[True, True, True, True, True, True, True, False],
        evaluation_window=32,
        mastery_threshold=0.9,
        promotion_step=1,
    )

    assert result["evaluated"] is True
    assert result["promoted"] is True
    assert result["current_level"] == 4
    assert result["window_attempts"] == 0


def test_adaptive_complexity_holds_when_the_policy_has_not_mastered_level() -> None:
    result = advance_complexity(
        current_level=3,
        maximum_level=8,
        window_attempts=24,
        window_successes=18,
        ordered_outcomes=[True, True, True, True, True, False, False, False],
        evaluation_window=32,
        mastery_threshold=0.9,
        promotion_step=1,
    )

    assert result["evaluated"] is True
    assert result["promoted"] is False
    assert result["current_level"] == 3
    assert active_complexity_range(minimum=0, current=3, sampling_band=4) == [0, 3]


def test_adaptive_complexity_carries_ordered_overshoot_into_the_next_window() -> None:
    result = advance_complexity(
        current_level=1,
        maximum_level=8,
        window_attempts=6,
        window_successes=6,
        ordered_outcomes=[True, True, False, True, False],
        evaluation_window=8,
        mastery_threshold=0.9,
        promotion_step=1,
    )

    assert result["promoted"] is True
    assert result["current_level"] == 2
    assert result["window_attempts"] == 3
    assert result["window_successes"] == 1


def test_reproduction_name_does_not_accumulate_prefixes() -> None:
    assert (
        _reproduction_name("Reproduction of Reproduction of Contract proof")
        == "Reproduction of Contract proof"
    )
