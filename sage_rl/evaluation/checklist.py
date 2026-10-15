"""Deterministic layout checks; visual and simulation evidence are explicit inputs.

Geometry uses Sage's bottom-centre positions and metre dimensions. Support uses
real mesh downward rays; the legacy bbox helper is diagnostic only.
"""

from __future__ import annotations

import math
from typing import Any

from sage_rl.evaluation.categories import ALIASES, category_matches
from sage_rl.evaluation.semantics import object_matches


def footprint(obj: dict) -> list[tuple[float, float]]:
    p, d = obj["position"], obj["dimensions"]
    theta = math.radians(obj.get("rotation", {}).get("z", 0))
    c, s = math.cos(theta), math.sin(theta)
    return [
        (p["x"] + c * x - s * y, p["y"] + s * x + c * y)
        for x, y in (
            (-d["width"] / 2, -d["length"] / 2),
            (d["width"] / 2, -d["length"] / 2),
            (d["width"] / 2, d["length"] / 2),
            (-d["width"] / 2, d["length"] / 2),
        )
    ]


def inside_footprint(
    point: tuple[float, float], obj: dict, tolerance: float = 0
) -> bool:
    p, d = obj["position"], obj["dimensions"]
    theta = math.radians(obj.get("rotation", {}).get("z", 0))
    dx, dy = point[0] - p["x"], point[1] - p["y"]
    x, y = (
        math.cos(theta) * dx + math.sin(theta) * dy,
        -math.sin(theta) * dx + math.cos(theta) * dy,
    )
    return (
        abs(x) <= d["width"] / 2 + tolerance and abs(y) <= d["length"] / 2 + tolerance
    )


def support_geometry_evidence(child: dict, parent: dict, tolerance: float) -> dict:
    """Explain the existing top-face OBB heuristic, without claiming mesh contact.

    A multi-level surface, protruding backboard, or tilted mesh can disagree with
    this approximation. ``place_id`` is intent, not physical support evidence.
    """
    top = parent["position"]["z"] + parent["dimensions"]["height"]
    gap = child["position"]["z"] - top
    corners = [inside_footprint(p, parent, tolerance) for p in footprint(child)]
    return {
        "child_id": child["id"], "parent_id": parent["id"],
        "child_bottom_z_m": child["position"]["z"],
        "parent_bbox_top_z_m": top, "vertical_gap_m": gap,
        "height_within_tolerance": abs(gap) <= tolerance,
        "footprint_corners_inside": corners,
        "passed": abs(gap) <= tolerance and all(corners),
        "method": "oriented_bounding_box_top_face_approximation",
    }


def supported_by(child: dict, parent: dict, tolerance: float) -> bool:
    from sage_rl.evaluation.mesh_support import query_mesh_support
    return query_mesh_support(child, parent)["passed"]


def _check(key, section, passed, *, evidence, critical=False):
    return {
        "id": key,
        "section": section,
        "status": "unknown" if passed is None else "pass" if passed else "fail",
        "evidence": evidence,
        "critical": critical,
    }


def match_objects(
    requirements: list[dict], objects: list[dict]
) -> tuple[dict, list[dict]]:
    """Maximum bipartite matching of required slots; no object is reused."""
    slots = [
        (r["id"], r["category"]) for r in requirements for _ in range(r["count"][1])
    ]
    owner: dict[int, int] = {}

    def assign(slot, seen):
        for index, obj in enumerate(objects):
            if index in seen or not object_matches(slots[slot][1], obj):
                continue
            seen.add(index)
            if index not in owner or assign(owner[index], seen):
                owner[index] = slot
                return True
        return False

    for slot in range(len(slots)):
        assign(slot, set())
    matched = {r["id"]: [] for r in requirements}
    for index, slot in owner.items():
        matched[slots[slot][0]].append(objects[index])
    return matched, [obj for i, obj in enumerate(objects) if i not in owner]


def evaluate_task(
    task: dict,
    layout: dict | None,
    *,
    physics: dict | None = None,
    visual: dict | None = None,
) -> dict[str, Any]:
    """Evaluate final state, preserving unknown evidence instead of passing it."""
    from sage_rl.evaluation.mesh_support import attach_mesh_root, query_mesh_support, MeshUnavailable
    if layout: attach_mesh_root(layout)
    rooms = layout.get("rooms", []) if layout else []
    objects = [obj for room in rooms for obj in room.get("objects", [])]
    declared_objects = objects
    loaded_ids = (physics or {}).get("loaded_object_ids")
    objects = [obj for obj in objects if obj["id"] in set(loaded_ids or [])]
    scoped = task.get('evaluation', {}).get('requirement_contract') == 'surface_scoped_minimum_v1'
    scope_audit = None
    if scoped:
        from sage_rl.evaluation.scoped_requirements import match_scoped
        matched, unmatched, scopes, scope_audit = match_scoped(task, objects)
    else:
        matched, unmatched = match_objects(task["objects"], objects)
    checks = []
    for requirement in task["objects"]:
        count = len(matched[requirement["id"]])
        low, high = requirement["count"]
        extras = sum(
            object_matches(requirement["category"], obj) for obj in unmatched
        )
        if scoped:
            extras = max(0, len(scopes[requirement['id']]) - count)
        checks.append(
            _check(
                "object:" + requirement["id"],
                "objects",
                low <= count and count + extras <= high,
                evidence={
                    "matched_ids": [obj["id"] for obj in matched[requirement["id"]]],
                    "count": count,
                    "extra_matching": extras,
                    "expected": [low, high],
                    "semantic_source": "registered_category_and_loaded_instance",
                },
                critical=requirement.get("critical", False),
            )
        )
    from sage_rl.evaluation.furniture import assess_furniture
    furniture = assess_furniture(task, objects, matched)
    checks.append(_check("furniture_coverage", "furniture", furniture['score'] == 1., evidence=furniture))
    checks.append(_check("furniture_capacity", "furniture", not furniture['excess_ids'],
                         evidence={"excess_ids": furniture['excess_ids']}))
    decorative_unmatched = [obj for obj in unmatched if obj['id'] not in furniture['exempt_ids']]
    extra_limit = task.get("preferences", {}).get("max_extra_objects")
    if task.get("preferences", {}).get("no_extra_objects") or (extra_limit is not None):
        checks.append(
            _check(
                "no_extra_objects",
                "objects",
                len(decorative_unmatched) <= (extra_limit or 0),
                evidence={"unmatched_ids": [obj["id"] for obj in decorative_unmatched],
                          "count": len(decorative_unmatched), "limit": extra_limit or 0},
            )
        )
    missing = sorted({o["id"] for o in declared_objects} - set(loaded_ids or []))
    checks.append(_check("loaded_instances", "physics", None if loaded_ids is None else not missing,
                         evidence={"missing_ids": missing, "loaded_ids": loaded_ids}, critical=True))
    tolerance = task.get("evaluation", {}).get("support_tolerance_m", 0.08)
    for relation in task.get("relations", []):
        children, parents = matched[relation["subject"]], matched[relation["object"]]
        predicate = relation["predicate"]

        def holds(child, parent):
            if predicate == "supported_by":
                return supported_by(child, parent, tolerance)
            distance = math.hypot(
                child["position"]["x"] - parent["position"]["x"],
                child["position"]["y"] - parent["position"]["y"],
            )
            return (
                distance <= relation["distance_m"]
                if predicate == "near"
                else distance >= relation["distance_m"]
            )

        pairs = []
        if predicate == 'supported_by':
            child_status = []
            for child in children:
                outcomes = []
                for parent in parents:
                    try:
                        evidence = query_mesh_support(child, parent)
                    except MeshUnavailable as exc:
                        evidence = dict(child_id=child['id'], parent_id=parent['id'], passed=None, error=str(exc))
                    pairs.append(evidence); outcomes.append(evidence['passed'])
                child_status.append(True if True in outcomes else (None if None in outcomes else False))
            passed = False if not children or not parents or False in child_status else (None if None in child_status else True)
        else:
            passed = bool(children and parents) and all(any(holds(c, p) for p in parents) for c in children)
        checks.append(
            _check(
                f"relation:{relation['subject']}:{predicate}:{relation['object']}",
                "relations",
                passed,
                evidence={
                    "subjects": [o["id"] for o in children],
                    "targets": [o["id"] for o in parents],
                    "method": "mesh_downward_rays_v2_winding_agnostic" if predicate == "supported_by" else "distance",
                    "tolerance_m": tolerance,
                    **({"support_pairs": pairs} if predicate == "supported_by" else {}),
                },
            )
        )
    for constraint in task.get("constraints", []):
        kind = constraint["type"]
        if kind == "only_on_surface":
            surfaces = matched[constraint["supporter"]]
            allowed = {
                obj["id"] for req in constraint["allowed"] for obj in matched[req]
            }
            unexpected = [
                obj["id"]
                for obj in objects
                if obj["id"] not in allowed
                and any(
                    obj["id"] != parent["id"] and supported_by(obj, parent, tolerance)
                    for parent in surfaces
                )
            ]
            checks.append(
                _check(
                    "surface:" + constraint["supporter"],
                    "relations",
                    bool(surfaces) and not unexpected,
                    evidence={"unexpected_ids": unexpected},
                )
            )
        elif kind == "bounds":
            failures = []
            tol = constraint.get("tolerance_m", 0.1)
            for room in rooms:
                p, d = room["position"], room["dimensions"]
                for obj in room.get("objects", []):
                    within = all(
                        p["x"] - tol <= x <= p["x"] + d["width"] + tol
                        and p["y"] - tol <= y <= p["y"] + d["length"] + tol
                        for x, y in footprint(obj)
                    )
                    within &= (
                        obj["position"]["z"] >= p.get("z", 0) - tol
                        and obj["position"]["z"] + obj["dimensions"]["height"]
                        <= p.get("z", 0) + d["height"] + tol
                    )
                    if not within:
                        failures.append(obj["id"])
            checks.append(
                _check(
                    "bounds",
                    "physics",
                    bool(rooms) and not failures,
                    evidence={"violating_ids": failures, "tolerance_m": tol},
                )
            )
        elif kind == "stable":
            valid = physics is not None and physics.get("status") == "success"
            passed = not physics.get("unstable_objects") if valid else None
            checks.append(
                _check(
                    "stable", "physics", passed, evidence=physics or {}, critical=False
                )
            )
        elif kind == "door_access_clear":
            # Require measured evidence; a clear door AABB alone is not reachability.
            access = (physics or {}).get("door_access")
            checks.append(
                _check(
                    "door_access_clear",
                    "physics",
                    access.get("pass") if access else None,
                    evidence=access or {},
                    critical=True,
                )
            )
    from sage_rl.evaluation.scoring import score_task
    result = score_task(task, rooms, objects, matched, unmatched, checks, visual, physics)
    if scope_audit is not None:
        result['requirement_matching'] = scope_audit
    return result
