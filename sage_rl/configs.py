from dataclasses import dataclass, field
import sys
import os
from pathlib import Path

AREAL_ROOT = Path(os.environ.get("AREAL_ROOT", str(Path(__file__).resolve().parents[1] / "third_party/AReaL")))
if str(AREAL_ROOT) not in sys.path:
    sys.path.insert(0, str(AREAL_ROOT))

from areal.api.cli_args import GRPOConfig  # noqa: E402


@dataclass
class SageRLConfig(GRPOConfig):
    task_visits: int = field(default=2, metadata={"help": "Fresh online groups per training task."})
    tasks_per_step: int = field(default=1, metadata={"help": "Distinct prompt groups in one GRPO update."})
    workflow: str = field(
        default="sage_rl.agent.sage_efficiency_room_agent.SageEfficiencyRoomAgent",
        metadata={"help": "AReaL agent workflow for SAGE room generation."},
    )
    max_tool_calls: int = field(
        default=8,
        metadata={"help": "Maximum MCP tool calls per rollout."},
    )
    use_final_appearance_critic: bool = field(
        default=True,
        metadata={
            "help": "Run eval_appearance-style critic on final layout for reward."
        },
    )
    server_paths: list[str] = field(
        default_factory=lambda: [
            str(
                Path(__file__).resolve().parents[1]
                / "server"
                / "layout_wo_robot.py"
            )
        ],
        metadata={"help": "MCP server script paths."},
    )
