from copy import deepcopy

import pytest

from services.execution.app import cad


def test_cad_fixture_is_multi_turn_and_deterministic() -> None:
    state = cad.initial_state()
    for action in (
        {"kind": "create_base"},
        {"kind": "add_boss"},
        {"kind": "add_bore"},
        {"kind": "add_hole_pattern"},
        {"kind": "fillet_edges"},
        {"kind": "submit"},
    ):
        state = cad.apply_action(state, action)

    assert state["submitted"] is True
    assert state["turn"] == 6
    report = cad.geometry_report(state)
    assert report["geometry_valid"] is True
    assert report["constraints_passed"] is True
    assert cad.progress_score(state) == 1
    assert cad.render_svg(state, label="candidate") == cad.render_svg(state, label="candidate")


def test_valid_negative_geometry_is_evidence_not_infrastructure_failure() -> None:
    state = cad.initial_state()
    for action in (
        {"kind": "create_base"},
        {"kind": "add_boss"},
        {"kind": "add_bore"},
        {"kind": "oversize_bore"},
    ):
        state = cad.apply_action(state, action)

    report = cad.geometry_report(state)
    assert report["candidate_failure"] is True
    assert report["geometry_valid"] is False
    assert "central bore exceeds the allowed diameter" in report["violations"]


def test_submitted_fixture_requires_every_feature_and_exact_parameters() -> None:
    missing_fillet = deepcopy(cad.TARGET_STATE)
    missing_fillet["features"] = [
        feature for feature in missing_fillet["features"] if feature["kind"] != "fillet"
    ]
    report = cad.geometry_report(missing_fillet)
    assert report["constraints_passed"] is False
    assert report["topology_valid"] is False

    wrong_boss = deepcopy(cad.TARGET_STATE)
    next(feature for feature in wrong_boss["features"] if feature["kind"] == "boss")["diameter"] = (
        35
    )
    report = cad.geometry_report(wrong_boss)
    assert report["constraints_passed"] is False
    assert "feature parameters" in report["violations"][-1]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("boss", "diameter"),
        ("boss", "height"),
        ("boss", "x"),
        ("hole_pattern", "count"),
        ("hole_pattern", "x_pitch"),
        ("hole_pattern", "diameter"),
        ("bore", "depth"),
        ("bore", "x"),
        ("fillet", "radius"),
        ("fillet", "edges"),
    ],
)
def test_renderer_changes_when_semantic_feature_changes(kind: str, field: str) -> None:
    state = deepcopy(cad.TARGET_STATE)
    state["submitted"] = False
    before = cad.render_svg(state, label="Candidate")
    feature = next(item for item in state["features"] if item["kind"] == kind)
    feature[field] = feature[field] + 1
    assert cad.render_svg(state, label="Candidate") != before


def test_renderer_ignores_occurrence_labels_and_episode_turn() -> None:
    state = deepcopy(cad.TARGET_STATE)
    before = cad.render_svg(state, label="Reference")
    state["turn"] += 10
    assert cad.render_svg(state, label="Candidate") == before


def test_renderer_changes_when_base_dimensions_change() -> None:
    state = deepcopy(cad.TARGET_STATE)
    before = cad.render_svg(state, label="Candidate")
    state["dimensions"]["width"] += 1
    assert cad.render_svg(state, label="Candidate") != before


def test_fixture_rejects_extra_action_fields_and_non_finite_state() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        cad.apply_action(cad.initial_state(), {"kind": "create_base", "width": 80})

    invalid = cad.initial_state()
    invalid["dimensions"]["width"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cad.geometry_report(invalid)
