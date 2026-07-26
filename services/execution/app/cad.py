from __future__ import annotations

import html
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
    dimensions = state["dimensions"]
    features = {feature["kind"]: feature for feature in state["features"]}
    base_valid = dimensions == TARGET_STATE["dimensions"]
    bore = features.get("bore")
    bore_safe = bore is None or float(bore["diameter"]) <= 30.0
    geometry_valid = bool(base_valid and bore_safe)

    expected = {feature["kind"]: feature for feature in TARGET_STATE["features"]}
    matched = sum(features.get(kind) == feature for kind, feature in expected.items())
    feature_ratio = matched / len(expected)
    topology_valid = all(
        [
            not state["submitted"] or "boss" in features,
            not state["submitted"] or "bore" in features,
            not state["submitted"] or "hole_pattern" in features,
        ]
    )
    constraints_passed = geometry_valid and topology_valid
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
        "volume_mm3": round(
            dimensions["width"] * dimensions["depth"] * dimensions["height"],
            3,
        ),
        "face_count": 6 + 4 * len(features),
        "violations": [
            message
            for condition, message in [
                (not base_valid, "base dimensions do not match the task constraint"),
                (not bore_safe, "central bore exceeds the allowed diameter"),
                (not topology_valid, "submitted topology is missing a required feature"),
            ]
            if condition
        ],
    }


def progress_score(state: dict[str, Any]) -> float:
    report = geometry_report(state)
    base = 0.2 if report["dimensions_valid"] else 0.0
    features = 0.7 * report["feature_correspondence"]
    terminal = 0.1 if state["submitted"] and report["constraints_passed"] else 0.0
    penalty = 0.35 if not report["geometry_valid"] else 0.0
    return round(max(0.0, min(1.0, base + features + terminal - penalty)), 4)


def render_svg(state: dict[str, Any], *, label: str) -> bytes:
    safe_label = html.escape(label)
    features = {feature["kind"]: feature for feature in state["features"]}
    width = state["dimensions"]["width"]
    depth = state["dimensions"]["depth"]
    has_base = width > 0 and depth > 0
    boss = features.get("boss")
    bore = features.get("bore")
    holes = features.get("hole_pattern")
    fillet = features.get("fillet")
    corner_radius = 10 if fillet else 2

    top_shapes = ""
    if has_base:
        top_shapes += (
            f'<rect x="72" y="126" width="176" height="132" rx="{corner_radius}" class="solid"/>'
        )
    if boss:
        top_shapes += '<circle cx="160" cy="192" r="38" class="feature"/>'
    if bore:
        radius = min(54, float(bore["diameter"]) * 1.25)
        top_shapes += f'<circle cx="160" cy="192" r="{radius}" class="cut"/>'
    if holes:
        for x, y in [(94, 148), (226, 148), (94, 236), (226, 236)]:
            top_shapes += f'<circle cx="{x}" cy="{y}" r="7" class="cut"/>'

    iso_shapes = ""
    if has_base:
        iso_shapes = (
            '<path d="M330 208 L430 150 L555 205 L455 266 Z" class="solid"/>'
            '<path d="M330 208 L330 226 L455 286 L455 266 Z" class="side"/>'
            '<path d="M455 266 L555 205 L555 223 L455 286 Z" class="side2"/>'
        )
    if boss:
        iso_shapes += '<ellipse cx="445" cy="207" rx="42" ry="23" class="feature"/>'
    if bore:
        radius = min(32, float(bore["diameter"]) * 0.8)
        iso_shapes += f'<ellipse cx="445" cy="207" rx="{radius}" ry="{radius * 0.55}" class="cut"/>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480" role="img" aria-label="{safe_label}">
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
<text x="36" y="370" class="label">{safe_label}</text>
<text x="36" y="396" class="meta">camera:isometric-orthographic@1 · material:drafting-blue@1</text>
<text x="36" y="418" class="meta">lighting:three-point-neutral@1 · 640×480 · turn:{state["turn"]}</text>
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
