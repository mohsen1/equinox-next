from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from equinox_core import canonical_digest

TASK_REVISION = "mounting-plate@sha256:fixture-v1"
ENVIRONMENT_VERSION = "cad.reconstruction@1.0.0"
TARGET_STATE: dict[str, Any] = {
    "dimensions": {"width": 80.0, "depth": 60.0, "height": 6.0},
    "features": [
        {"kind": "boss", "diameter": 34.0, "height": 8.0, "x": 0.0, "y": 0.0},
        {"kind": "bore", "diameter": 18.0, "depth": 14.0, "x": 0.0, "y": 0.0},
        {"kind": "hole_pattern", "diameter": 6.0, "count": 4, "x_pitch": 60.0, "y_pitch": 40.0},
        {"kind": "fillet", "radius": 3.0, "edges": 4},
    ],
    "submitted": True,
    "turn": 6,
}

ALLOWED_ACTIONS = {
    "create_base",
    "add_boss",
    "add_bore",
    "add_hole_pattern",
    "fillet_edges",
    "oversize_bore",
    "submit",
}


def initial_state() -> dict[str, Any]:
    return {
        "dimensions": {"width": 0.0, "depth": 0.0, "height": 0.0},
        "features": [],
        "submitted": False,
        "turn": 0,
    }


def apply_action(source: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    if set(action) != {"kind"}:
        raise ValueError("CAD fixture actions require exactly one 'kind' field")
    kind = action.get("kind")
    if kind not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported CAD action: {kind}")
    if source.get("submitted"):
        raise ValueError("terminal CAD state cannot accept another action")

    candidate = deepcopy(source)
    candidate["turn"] = int(source.get("turn", 0)) + 1
    features: list[dict[str, Any]] = candidate["features"]

    if kind == "create_base":
        candidate["dimensions"] = {"width": 80.0, "depth": 60.0, "height": 6.0}
    elif kind == "add_boss":
        _replace(
            features, "boss", {"kind": "boss", "diameter": 34.0, "height": 8.0, "x": 0.0, "y": 0.0}
        )
    elif kind == "add_bore":
        _replace(
            features, "bore", {"kind": "bore", "diameter": 18.0, "depth": 14.0, "x": 0.0, "y": 0.0}
        )
    elif kind == "oversize_bore":
        _replace(
            features, "bore", {"kind": "bore", "diameter": 52.0, "depth": 14.0, "x": 0.0, "y": 0.0}
        )
    elif kind == "add_hole_pattern":
        _replace(
            features,
            "hole_pattern",
            {"kind": "hole_pattern", "diameter": 6.0, "count": 4, "x_pitch": 60.0, "y_pitch": 40.0},
        )
    elif kind == "fillet_edges":
        _replace(features, "fillet", {"kind": "fillet", "radius": 3.0, "edges": 4})
    elif kind == "submit":
        candidate["submitted"] = True

    return candidate


def _replace(features: list[dict[str, Any]], kind: str, feature: dict[str, Any]) -> None:
    features[:] = [existing for existing in features if existing.get("kind") != kind]
    features.append(feature)


def geometry_report(state: dict[str, Any]) -> dict[str, Any]:
    _validate_state(state)
    dimensions = state["dimensions"]
    features = {feature["kind"]: feature for feature in state["features"]}
    base_valid = dimensions == TARGET_STATE["dimensions"]
    bore = features.get("bore")
    bore_safe = bore is None or float(bore["diameter"]) <= 30.0

    expected = {feature["kind"]: feature for feature in TARGET_STATE["features"]}
    matched = sum(features.get(kind) == feature for kind, feature in expected.items())
    feature_ratio = matched / len(expected)
    required_features_present = all(kind in features for kind in expected)
    topology_valid = not state["submitted"] or required_features_present
    feature_constraints_valid = all(
        kind not in features or features[kind] == target for kind, target in expected.items()
    )
    geometry_valid = bool(base_valid and bore_safe and feature_constraints_valid)
    constraints_passed = geometry_valid and topology_valid
    boss = features.get("boss")
    holes = features.get("hole_pattern")
    base_volume = dimensions["width"] * dimensions["depth"] * dimensions["height"]
    boss_volume = (
        math.pi * (float(boss["diameter"]) / 2) ** 2 * float(boss["height"]) if boss else 0
    )
    bore_volume = (
        math.pi * (float(bore["diameter"]) / 2) ** 2 * float(bore["depth"]) if bore else 0
    )
    hole_volume = (
        int(holes["count"])
        * math.pi
        * (float(holes["diameter"]) / 2) ** 2
        * dimensions["height"]
        if holes
        else 0
    )
    return {
        "schema_version": 1,
        "state_digest": canonical_digest(state),
        "syntax_valid": True,
        "execution_valid": True,
        "geometry_valid": geometry_valid,
        "topology_valid": topology_valid,
        "dimensions_valid": base_valid,
        "constraints_passed": constraints_passed,
        "candidate_failure": not constraints_passed,
        "feature_correspondence": round(feature_ratio, 4),
        "fixture_volume_estimate_mm3": round(
            max(0.0, base_volume + boss_volume - bore_volume - hole_volume),
            3,
        ),
        "fixture_face_count_estimate": 6 + 4 * len(features),
        "violations": [
            message
            for condition, message in [
                (not base_valid, "base dimensions do not match the task constraint"),
                (not bore_safe, "central bore exceeds the allowed diameter"),
                (not topology_valid, "submitted topology is missing a required feature"),
                (
                    not feature_constraints_valid,
                    "one or more feature parameters do not match the task constraints",
                ),
            ]
            if condition
        ],
    }


def _validate_state(state: dict[str, Any]) -> None:
    if set(state) != {"dimensions", "features", "submitted", "turn"}:
        raise ValueError("CAD state contains missing or unknown fields")
    dimensions = state["dimensions"]
    if set(dimensions) != {"width", "depth", "height"}:
        raise ValueError("CAD dimensions must contain width, depth, and height")
    if any(
        not isinstance(value, int | float) or not math.isfinite(float(value)) or value < 0
        for value in dimensions.values()
    ):
        raise ValueError("CAD dimensions must be finite non-negative numbers")
    if not isinstance(state["turn"], int) or state["turn"] < 0:
        raise ValueError("CAD turn must be a non-negative integer")
    if not isinstance(state["submitted"], bool) or not isinstance(state["features"], list):
        raise ValueError("CAD state lifecycle fields are invalid")
    expected_keys = {
        "boss": {"kind", "diameter", "height", "x", "y"},
        "bore": {"kind", "diameter", "depth", "x", "y"},
        "hole_pattern": {"kind", "diameter", "count", "x_pitch", "y_pitch"},
        "fillet": {"kind", "radius", "edges"},
    }
    seen: set[str] = set()
    for feature in state["features"]:
        if not isinstance(feature, dict) or feature.get("kind") not in expected_keys:
            raise ValueError("CAD state contains an unknown feature")
        kind = feature["kind"]
        if kind in seen:
            raise ValueError(f"CAD state contains duplicate {kind!r} features")
        seen.add(kind)
        if set(feature) != expected_keys[kind]:
            raise ValueError(f"CAD {kind!r} feature has missing or unknown fields")
        for key, value in feature.items():
            if key == "kind":
                continue
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"CAD {kind!r}.{key} must be finite")


def progress_score(state: dict[str, Any]) -> float:
    report = geometry_report(state)
    base = 0.2 if report["dimensions_valid"] else 0.0
    features = 0.7 * report["feature_correspondence"]
    terminal = 0.1 if state["submitted"] and report["constraints_passed"] else 0.0
    penalty = 0.35 if not report["geometry_valid"] else 0.0
    return round(max(0.0, min(1.0, base + features + terminal - penalty)), 4)


def render_svg(state: dict[str, Any], *, label: str) -> bytes:
    del label
    _validate_state(state)
    features = {feature["kind"]: feature for feature in state["features"]}
    width = state["dimensions"]["width"]
    depth = state["dimensions"]["depth"]
    height = state["dimensions"]["height"]
    has_base = width > 0 and depth > 0
    boss = features.get("boss")
    bore = features.get("bore")
    holes = features.get("hole_pattern")
    fillet = features.get("fillet")
    corner_radius = min(18.0, max(2.0, float(fillet["radius"]) * 3)) if fillet else 2
    top_width = max(1.0, min(208.0, float(width) * 2.2))
    top_depth = max(1.0, min(176.0, float(depth) * 2.2))
    top_x = 160 - top_width / 2
    top_y = 192 - top_depth / 2

    top_shapes = ""
    if has_base:
        top_shapes += (
            f'<rect x="{top_x:.3f}" y="{top_y:.3f}" width="{top_width:.3f}" '
            f'height="{top_depth:.3f}" rx="{corner_radius:.3f}" class="solid"/>'
        )
        if fillet:
            edge_count = int(fillet["edges"])
            for index in range(max(0, min(edge_count, 8))):
                marker_x = top_x + 5 + index * 8
                top_shapes += (
                    f'<circle cx="{marker_x:.3f}" cy="{top_y + 5:.3f}" '
                    'r="1.5" class="feature"/>'
                )
    if boss:
        boss_x = 160 + float(boss["x"]) * 2.2
        boss_y = 192 - float(boss["y"]) * 2.2
        boss_radius = float(boss["diameter"]) * 1.1
        top_shapes += (
            f'<circle cx="{boss_x:.3f}" cy="{boss_y:.3f}" '
            f'r="{boss_radius:.3f}" class="feature"/>'
        )
    if bore:
        radius = min(54, float(bore["diameter"]) * 1.25)
        bore_x = 160 + float(bore["x"]) * 2.2
        bore_y = 192 - float(bore["y"]) * 2.2
        top_shapes += (
            f'<circle cx="{bore_x:.3f}" cy="{bore_y:.3f}" '
            f'r="{radius:.3f}" class="cut" data-depth="{float(bore["depth"]):.3f}"/>'
        )
    if holes:
        count = int(holes["count"])
        pitch_x = float(holes["x_pitch"]) * 1.1
        pitch_y = float(holes["y_pitch"]) * 1.1
        hole_radius = float(holes["diameter"]) * 1.15
        positions = [
            (
                160 + pitch_x * math.cos((2 * math.pi * index / count) - math.pi / 2),
                192 + pitch_y * math.sin((2 * math.pi * index / count) - math.pi / 2),
            )
            for index in range(count)
        ] if count > 0 else []
        for x, y in positions:
            top_shapes += f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{hole_radius:.3f}" class="cut"/>'

    iso_shapes = ""
    if has_base:
        iso_rise = max(6.0, min(30.0, float(height) * 3))
        iso_scale_x = max(70.0, min(125.0, float(width) * 1.5))
        iso_scale_y = max(45.0, min(90.0, float(depth)))
        iso_shapes = (
            f'<path d="M330 208 L{330 + iso_scale_x:.3f} {208 - iso_scale_y:.3f} '
            f'L{330 + iso_scale_x * 2:.3f} 208 L{330 + iso_scale_x:.3f} '
            f'{208 + iso_scale_y:.3f} Z" class="solid"/>'
            f'<path d="M330 208 L330 {208 + iso_rise:.3f} '
            f'L{330 + iso_scale_x:.3f} {208 + iso_scale_y + iso_rise:.3f} '
            f'L{330 + iso_scale_x:.3f} {208 + iso_scale_y:.3f} Z" class="side"/>'
            f'<path d="M{330 + iso_scale_x:.3f} {208 + iso_scale_y:.3f} '
            f'L{330 + iso_scale_x * 2:.3f} 208 L{330 + iso_scale_x * 2:.3f} '
            f'{208 + iso_rise:.3f} L{330 + iso_scale_x:.3f} '
            f'{208 + iso_scale_y + iso_rise:.3f} Z" class="side2"/>'
        )
    if boss:
        boss_cx = 455 + float(boss["x"])
        boss_cy = 207 - float(boss["y"]) * 0.5 - float(boss["height"]) * 1.5
        boss_rx = float(boss["diameter"]) * 1.2
        iso_shapes += (
            f'<ellipse cx="{boss_cx:.3f}" cy="{boss_cy:.3f}" '
            f'rx="{boss_rx:.3f}" ry="{boss_rx * 0.55:.3f}" class="feature"/>'
        )
    if bore:
        radius = min(32, float(bore["diameter"]) * 0.8)
        bore_cx = 455 + float(bore["x"])
        bore_cy = 207 - float(bore["y"]) * 0.5 - float(bore["depth"]) * 0.15
        iso_shapes += (
            f'<ellipse cx="{bore_cx:.3f}" cy="{bore_cy:.3f}" '
            f'rx="{radius:.3f}" ry="{radius * 0.55:.3f}" class="cut"/>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480" role="img" aria-label="CAD geometry evidence">
<style>
  .bg{{fill:#eef1f4}} .grid{{stroke:#d5dce3;stroke-width:1}} .frame{{fill:none;stroke:#7c8996;stroke-width:1.5}}
  .solid{{fill:#8ca9bc;stroke:#263b49;stroke-width:3}} .side{{fill:#65869b;stroke:#263b49;stroke-width:3}}
  .side2{{fill:#7596aa;stroke:#263b49;stroke-width:3}} .feature{{fill:#b6c9d5;stroke:#263b49;stroke-width:3}}
  .cut{{fill:#eef1f4;stroke:#263b49;stroke-width:3}} .label{{font:600 14px ui-monospace,monospace;fill:#263b49}}
  .meta{{font:12px ui-monospace,monospace;fill:#586b78}}
</style>
<rect width="640" height="480" class="bg"/>
<path d="M0 40H640M0 80H640M0 120H640M0 160H640M0 200H640M0 240H640M0 280H640M0 320H640M0 360H640M0 400H640M0 440H640" class="grid"/>
<rect x="36" y="78" width="248" height="248" class="frame"/><rect x="306" y="78" width="298" height="248" class="frame"/>
<text x="48" y="104" class="label">TOP</text><text x="318" y="104" class="label">ISOMETRIC</text>
{top_shapes}{iso_shapes}
<text x="36" y="396" class="meta">camera:isometric-orthographic@1 · material:drafting-blue@1</text>
<text x="36" y="418" class="meta">lighting:three-point-neutral@1 · 640×480</text>
</svg>""".encode()


def task_reference() -> dict[str, Any]:
    return {
        "task_revision": TASK_REVISION,
        "title": "Mounting plate with raised boss",
        "dimensions_mm": {"base": [80, 60, 6], "boss_diameter": 34, "bore_diameter": 18},
        "constraints": [
            "four 6 mm corner holes on a 60 × 40 mm pattern",
            "central 34 mm raised boss",
            "18 mm through bore",
            "3 mm outside-edge fillet",
        ],
        "hidden": False,
        "synthetic_fixture": True,
    }
