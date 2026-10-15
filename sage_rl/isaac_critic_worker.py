"""Evaluate an Efficiency scene through its Isaac renderer and a local VLM."""

import base64
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlparse


def evaluate(layout_id, demand, task_data=None):
    root = Path(os.environ["SAGE_EFFICIENCY_ROOT"]).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "server"))
    from constants import RESULTS_DIR
    from isaac_room_render import render_isaac_views
    from isaacsim.isaac_mcp.server import get_isaac_connection
    from sage_rl.evaluation.visual import VISUAL_DIMENSIONS, VISUAL_PROMPT, visual_reward
    from sage_rl.evaluation.access import door_access
    from utils import dict_to_floor_plan
    from PIL import Image
    import numpy as np
    from openai import OpenAI

    endpoint = os.environ.get("SAGE_VLM_BASE_URL", "http://127.0.0.1:1081/v1")
    if urlparse(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The final critic requires a loopback VLM endpoint")
    scene_dir = Path(RESULTS_DIR) / layout_id
    layout_data = json.loads((scene_dir / f"{layout_id}.json").read_text())
    layout = dict_to_floor_plan(layout_data)
    if not layout.rooms:
        raise ValueError("Cannot evaluate a layout without rooms")
    output_dir = scene_dir / "rl_final_critic"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    physics_results = []
    for room in layout.rooms:
        physics = get_isaac_connection().send_command("validate_room_stability", {
            "scene_save_dir": str(scene_dir), "room_id": room.id,
        })
        if physics.get("status") != "success":
            return {"success": False, "environment_error": "Isaac validation failed: " + str(physics),
                    "error": "Isaac did not complete final validation", "physics": physics}
        raw_room = next(r for r in layout_data['rooms'] if r['id'] == room.id)
        clearance = next((c.get('clearance_m', .65) for c in (task_data or {}).get('constraints', [])
                          if c['type'] == 'door_access_clear'), .65)
        physics['door_access'] = door_access(raw_room, clearance_m=clearance)
        physics_results.append(physics)
        views = render_isaac_views(layout, room.id, resolution=768, views="five")
        if len(views) != 5:
            raise ValueError(f"Expected five Isaac views, got {len(views)}")
        for index, pixels in enumerate(views):
            path = output_dir / f"{room.id}_{index}.png"
            Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8)).save(path)
            paths.append(path)
    prompt = VISUAL_PROMPT + json.dumps(demand)
    content = [{"type": "text", "text": prompt}]
    for path in paths:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encoded}"}})
    from sage_rl.evaluation.physics_evidence import combine_room_physics
    result = {"success": False, "layout_id": layout_id,
              "views": [str(p) for p in paths], "backend": "isaac",
              "physics": combine_room_physics(physics_results)}
    try:
        with OpenAI(base_url=endpoint, max_retries=0,
                    api_key=os.environ.get("SAGE_VLM_API_KEY", "EMPTY"), timeout=600) as client:
            response = client.chat.completions.create(
                model=os.environ.get("SAGE_VLM_MODEL", "qwen-vl-local"),
                messages=[{"role": "user", "content": content}],
                temperature=0, max_tokens=1024,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        scores = json.loads(response.choices[0].message.content)
        grades = {}
        for key in VISUAL_DIMENSIONS:
            value = float(scores[key])
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f"Invalid VLM grade {key}: {value}")
            grades[key] = {"mean": value}
        mean = sum(item["mean"] for item in grades.values()) / len(grades)
        result.update(success=True, grades=grades, mean_grade=mean,
                      appearance_score=mean / 10, appearance_reward=visual_reward(mean / 10), reason=scores.get("reason", ""), model=response.model)
    except Exception as exc:
        # The completed scene and hard evidence survive a one-shot visual failure.
        result["error"] = repr(exc)
        result["visual_missing"] = True
    (output_dir / "score.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    try:
        result = evaluate(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
    except Exception as exc:
        result = {"success": False, "error": repr(exc)}
    print("SAGE_APPEARANCE_RESULT:" + json.dumps(result), flush=True)
    sys.exit(0 if result["success"] or result.get("visual_missing") or result.get("environment_error") else 1)
