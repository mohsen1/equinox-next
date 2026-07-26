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
