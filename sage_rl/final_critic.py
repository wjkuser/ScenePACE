"""Run the Isaac and local VLM final critic in the scene environment."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPEARANCE_SERVER_DIR = PROJECT_ROOT / "server"

DEFAULT_RENDER_PYTHON = ".runtime/envs/scene/bin/python"
RESULT_PREFIX = "SAGE_APPEARANCE_RESULT:"
_SIMGEN_ENV_BLOCKLIST = frozenset({"PYTHONPATH", "PYTHONHOME"})
_RENDER_PYTHON_CACHE: Optional[str] = None


def resolve_appearance_server_dir() -> Path:
    """Return the server tree that owns the appearance evaluation worker."""
    configured = os.environ.get("SAGE_APPEARANCE_SERVER_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_APPEARANCE_SERVER_DIR


def _simgen_subprocess_env(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build env for simgen subprocess; strip AReaL/vllm PYTHONPATH pollution."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _SIMGEN_ENV_BLOCKLIST
    }
    if overrides:
        env.update(overrides)
    return env


def _python_has_scene_dependencies(python_path: str) -> bool:
    if not os.path.isfile(python_path):
        return False
    completed = subprocess.run(
        [python_path, "-c", "import numpy, PIL, openai"],
        capture_output=True,
        text=True,
        check=False,
        env=_simgen_subprocess_env(),
    )
    return completed.returncode == 0


def resolve_render_python() -> str:
    """Return scene Python; Isaac rendering runs in its separate service."""
    global _RENDER_PYTHON_CACHE
    if _RENDER_PYTHON_CACHE and _python_has_scene_dependencies(_RENDER_PYTHON_CACHE):
        return _RENDER_PYTHON_CACHE

    candidates: list[str] = []
    for value in (
        os.environ.get("SAGE_PYTHON_BIN"),
        DEFAULT_RENDER_PYTHON,
    ):
        if value and value not in candidates:
            candidates.append(value)

    for candidate in candidates:
        if _python_has_scene_dependencies(candidate):
            _RENDER_PYTHON_CACHE = candidate
            return candidate

    checked = ", ".join(candidates) or "(none)"
    raise RuntimeError(
        "No scene Python found. Set SAGE_PYTHON_BIN to the scene environment. "
        f"Checked: {checked}"
    )


def verify_final_critic_setup() -> str:
    """Check the scene interpreter and the shared Efficiency renderer."""
    render_python = resolve_render_python()
    server_dir = resolve_appearance_server_dir()
    worker = Path(__file__).with_name("isaac_critic_worker.py")
    if not worker.is_file():
        raise RuntimeError(f"appearance eval worker missing: {worker}")
    if not (server_dir / "isaac_room_render.py").is_file():
        raise RuntimeError(f"Efficiency Isaac renderer missing: {server_dir}")
    print(
        f"[final_critic] preflight ok render_python={render_python} "
        f"server_dir={server_dir}",
        flush=True,
    )
    return render_python


def _parse_worker_output(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(
        "appearance eval worker returned no result payload; "
        f"stdout tail: {stdout[-500:]}"
    )


def run_final_appearance_critic_sync(
    layout_id: str,
    user_demand: str,
    task_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Request Isaac views and local VLM evaluation from scene Python."""
    if not layout_id:
        return {"success": False, "error": "layout_id is empty"}

    render_python = resolve_render_python()
    server_dir = resolve_appearance_server_dir()
    worker = Path(__file__).with_name("isaac_critic_worker.py")
    env = _simgen_subprocess_env(
        {
            "SAGE_EFFICIENCY_ROOT": str(server_dir.parent),
            "CUDA_VISIBLE_DEVICES": os.environ.get("SAGE_SCENE_CUDA_DEVICES", "7"),
            "SAGE_RENDER_BACKEND": "isaac",
        }
    )

    cmd = [render_python, str(worker), layout_id, user_demand]
    if task_data is not None:
        cmd.append(json.dumps(task_data))
    print(f"[final_critic] appearance eval: {' '.join(cmd[:3])} ...", flush=True)
    def run_once():
        return subprocess.run(
            cmd,
            cwd=str(server_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(os.environ.get("SAGE_FINAL_CRITIC_TIMEOUT", "1800")),
        )

    completed = run_once()
    # Isaac Kit may be restarted by a service supervisor after a GPU or
    # extension crash. Give it one bounded recovery attempt before rejecting
    # the rollout as an unavailable environment.
    combined = f"{completed.stdout}\n{completed.stderr}".lower()
    transient = any(marker in combined for marker in (
        "connection refused", "could not connect to isaac", "not connected to isaac",
    ))
    if completed.returncode != 0 and transient:
        delay = float(os.environ.get("SAGE_FINAL_CRITIC_RETRY_DELAY", "20"))
        time.sleep(delay)
        completed = run_once()
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return {
            "success": False,
            "error": f"appearance eval worker failed (code={completed.returncode}): {stderr}",
        }

    try:
        result = _parse_worker_output(completed.stdout)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    print(
        f"[final_critic] layout_id={layout_id} "
        f"success={result.get('success')} "
        f"appearance_reward={result.get('appearance_reward')}",
        flush=True,
    )
    return result


async def run_final_appearance_critic(
    layout_id: str,
    user_demand: str,
    task_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        run_final_appearance_critic_sync,
        layout_id,
        user_demand,
        task_data,
    )


def save_final_critic_log(result: Dict[str, Any], log_dir: Optional[Path] = None) -> None:
    if not result.get("success"):
        return
    target_dir = log_dir or Path(os.environ.get(
        "SAGE_RL_CRITIC_LOG_DIR", "outputs/final_critics"))
    target_dir.mkdir(parents=True, exist_ok=True)
    layout_id = result.get("layout_id", "unknown")
    path = target_dir / f"{layout_id}_appearance.json"
    payload = {
        "layout_id": layout_id,
        "grades": result.get("grades"),
        "mean_grade": result.get("mean_grade"),
        "appearance_score": result.get("appearance_score"),
        "appearance_reward": result.get("appearance_reward"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
