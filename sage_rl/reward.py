"""Terminal task reward and saved layout access."""
import json
import os
from pathlib import Path
from typing import Any, Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_layout_data(layout_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not layout_id:
        return None
    safe_layout_id = Path(str(layout_id)).name
    results_dir = Path(
        os.environ.get("SAGE_RESULTS_DIR", str(PROJECT_ROOT / "server" / "results"))
    )
    layout_path = results_dir / safe_layout_id / f"{safe_layout_id}.json"
    try:
        with layout_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        payload['_mesh_root'] = str(layout_path.parent)
        return payload
    return None



def compute_episode_reward(*, layout_id=None, appearance_result=None, layout_data=None, task_data=None,
                           tool_call_count=None, tool_call_limit_reached=False):
    from sage_rl.evaluation.checklist import evaluate_task
    if layout_id is None:
        return 0.0, {"reward_source": "checklist", "strict_success": False,
                     "evaluable": True, "reason": "no_layout"}
    result = evaluate_task(task_data, layout_data,
                           physics=(appearance_result or {}).get("physics"),
                           visual=appearance_result)
    apply_tool_penalty = (task_data or {}).get('evaluation', {}).get('tool_limit_penalty', True)
    penalty = .1 if tool_call_limit_reached and apply_tool_penalty else 0.
    result["tool_usage"] = {
        "count": tool_call_count,
        "limit_reached": tool_call_limit_reached,
        "penalty": penalty,
        "penalty_enabled": apply_tool_penalty,
    }
    result["reward_before_tool_penalty"] = result["reward"]
    result["reward"] = round(max(0., result["reward"] - penalty), 6)
    result["reward_without_visual"] = round(max(0., result["reward_without_visual"] - penalty), 6)
    return result["reward"], result
