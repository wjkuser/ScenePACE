"""Route RL episodes through the isolated Efficiency generation pipeline."""

import os
from pathlib import Path
import shlex

from sage_rl.agent.sage_room_agent import SageRoomAgent
from client import client_generation_room_desc as efficiency_client

_server_parameters = efficiency_client.StdioServerParameters


def _scene_server_parameters(*args, **kwargs):
    env = dict(kwargs.get("env") or os.environ)
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("SAGE_SCENE_CUDA_DEVICES", "7")
    kwargs["env"] = env
    script_args = kwargs.get("args")
    if kwargs.get("command") == "bash" and script_args and script_args[0] == "-c":
        tokens = shlex.split(script_args[1])
        # Recognize the exact launcher emitted by Efficiency; do not interpret
        # arbitrary shell commands or change JavaScript server launchers.
        if len(tokens) == 6 and tokens[0] == "cd" and tokens[2:4] == ["&&", "exec"]:
            kwargs["command"] = tokens[4]
            kwargs["cwd"] = tokens[1]
            script_args = [tokens[5]]
    if script_args and str(script_args[0]).endswith(".py"):
        entry = Path(__file__).resolve().parents[1] / "scene_entry.py"
        kwargs["args"] = [str(entry), *script_args]
    return _server_parameters(*args, **kwargs)


class _ScenePython:
    """Override only the client module's subprocess interpreter lookup."""

    def __init__(self, original):
        self.original = original
        self.executable = os.environ.get("SAGE_PYTHON_BIN", ".runtime/envs/scene/bin/python")

    def __getattr__(self, name):
        return getattr(self.original, name)


class SageEfficiencyRoomAgent(SageRoomAgent):
    def __init__(self, server_paths=None, **kwargs):
        if not isinstance(efficiency_client.sys, _ScenePython):
            efficiency_client.sys = _ScenePython(efficiency_client.sys)
            efficiency_client.StdioServerParameters = _scene_server_parameters
        root = Path(os.environ.get("SAGE_EFFICIENCY_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
        os.environ.update({
            "SAGE_EFFICIENCY_ROOT": str(root),
            "SAGE_ASSET_MODE": "generation",
            "SAGE_RETRIEVAL_ONLY": "0",
            "SAGE_RENDER_BACKEND": "isaac",
            "PHYSICS_CRITIC_ENABLED": "true",
            "SEMANTIC_CRITIC_ENABLED": "true",
        })
        os.environ.setdefault("SAGE_RESULTS_DIR", "outputs/results")
        os.environ.setdefault("SAGE_APPEARANCE_SERVER_DIR", str(root / "server"))
        super().__init__(server_paths=server_paths or [str(root / "server/layout_wo_robot.py")], **kwargs)
