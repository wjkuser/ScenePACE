"""Start the selected repository's Isaac MCP extension on GPU 7."""

import os
from pathlib import Path

from isaacsim import SimulationApp

gpu = int(os.environ.get("SAGE_ISAAC_GPU", "7"))
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": gpu,
        "physics_gpu": gpu,
        "multi_gpu": False,
        "width": 1024,
        "height": 768,
    }
)
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
project = Path(os.environ["SAGE_PROJECT"])
manager.add_path(str(project / "server/isaacsim"))
manager.set_extension_enabled_immediate("isaac.sim.mcp_extension", True)
print("ISAAC_SERVICE_READY", flush=True)
try:
    while app.is_running():
        app.update()
finally:
    app.close()
