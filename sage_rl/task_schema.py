"""Task contracts shared by collection, verification and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import math

RELATIONS = {"supported_by", "near", "away_from"}
CONSTRAINTS = {"bounds", "door_access_clear", "stable", "only_on_surface"}


def task_key(task: dict[str, Any]) -> str:
    """Identity is explicit; file names are provenance, never group keys."""
    return f"{task['task_id']}:{task.get('condition_id', 'default')}"


def validate_task(task: dict[str, Any]) -> None:
    for key in ("task_id", "family_id", "room_desc", "scene_type"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError(f"Task requires nonempty {key}")
    if task.get("split") not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev or test")
    environment = task["environment"]
    for key in ("max_tool_calls",):
        if environment[key] <= 0:
            raise ValueError(f"environment.{key} must be positive")
    preferences = task["preferences"]
    mode = preferences["object_mode"]
    if mode not in {"strict", "open"}:
        raise ValueError("object_mode must be strict or open")
    if mode == "strict":
        if not isinstance(preferences["max_extra_objects"], int) or preferences["max_extra_objects"] < 0:
            raise ValueError("Strict tasks require a nonnegative extra-object limit")
    else:
        richness = preferences["richness"]
        if any(not isinstance(richness[k], int) or richness[k] <= 0 for k in
               ("extra_count_target", "extra_category_target", "per_category_cap", "max_total_objects")):
            raise ValueError("Open richness targets must be positive integers")
        if not math.isfinite(richness["objects_per_m2"]) or richness["objects_per_m2"] <= 0:
            raise ValueError("objects_per_m2 must be finite and positive")
    from sage_rl.evaluation.categories import canonical_category, category_matches
    furniture = task['furniture']
    entries = furniture['core'] + furniture['optional']
    categories = [entry['category'] for entry in entries]
    if not furniture['core'] or not 1 <= furniture['min_optional_types'] <= len(furniture['optional']):
        raise ValueError("Furniture needs core categories and a feasible optional coverage target")
    if len(categories) != len(set(categories)) or any(canonical_category(c) != c for c in categories):
        raise ValueError("Furniture categories must be unique canonical names")
    if any(category_matches(a, b) or category_matches(b, a)
           for i, a in enumerate(categories) for b in categories[i + 1:]):
        raise ValueError("Furniture category aliases overlap")
    if any(not isinstance(entry['max_count'], int) or entry['max_count'] < 1 for entry in entries):
        raise ValueError("Furniture category limits must be positive integers")
    objects = task["objects"]
    ids = [obj["id"] for obj in objects]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Object requirement IDs must be nonempty and unique")
    for obj in objects:
        low, high = obj["count"]
        if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high):
            raise ValueError(f"Invalid count range for {obj['id']}")
        if not obj["category"]:
            raise ValueError("Object category must be nonempty")
    for relation in task.get("relations", []):
        if relation["predicate"] not in RELATIONS:
            raise ValueError(f"Unsupported relation: {relation['predicate']}")
        if relation["subject"] not in ids or relation["object"] not in ids:
            raise ValueError("Relation references an unknown object requirement")
        if relation["subject"] == relation["object"]:
            raise ValueError("Self relations are not supported")
        if relation["predicate"] != "supported_by" and relation["distance_m"] <= 0:
            raise ValueError("Distance relations require a positive distance_m")
    for constraint in task.get("constraints", []):
        if constraint["type"] not in CONSTRAINTS:
            raise ValueError(f"Unsupported constraint: {constraint['type']}")
        if constraint["type"] == "only_on_surface":
            if constraint["supporter"] not in ids or not set(
                constraint["allowed"]
            ) <= set(ids):
                raise ValueError("Surface constraint references unknown requirements")


def load_tasks(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    identities: set[str] = set()
    families: dict[str, str] = {}
    for task in records:
        validate_task(task)
        key = task_key(task)
        if key in identities:
            raise ValueError(f"Duplicate task identity: {key}")
        identities.add(key)
        family = task["family_id"]
        if family in families and families[family] != task["split"]:
            raise ValueError(f"Task family crosses data splits: {family}")
        families[family] = task["split"]
    selected = [task for task in records if split is None or task["split"] == split]
    if not selected:
        raise ValueError(f"No tasks for split {split!r} in {path}")
    return selected
