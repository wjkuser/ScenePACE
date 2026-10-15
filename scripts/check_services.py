"""Check model, asset, and Isaac readiness before starting an episode."""

import argparse, json, os, socket, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_pace.runtime import load

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/runtime.json")
a = p.parse_args()
c = load(a.config)
for name in ["actor", "vlm"]:
    if name == "actor" and c[name]["mode"] == "api":
        continue
    o = c[name]
    r = requests.get(
        o["endpoint"].rstrip("/") + "/models",
        headers={
            "Authorization": "Bearer " + os.environ.get(o["api_key_env"], "EMPTY")
        },
        timeout=10,
    )
    r.raise_for_status()
    if o["served_name"] not in [x["id"] for x in r.json()["data"]]:
        raise RuntimeError(name + " served model does not match config")
r = requests.get(c["trellis"]["endpoint"].rstrip("/") + "/health", timeout=10)
r.raise_for_status()
if r.json().get("service") != "sage-trellis":
    raise RuntimeError("TRELLIS protocol mismatch")
with socket.create_connection(("127.0.0.1", c["scene"]["isaac_port"]), timeout=10) as s:
    s.sendall(json.dumps({"type": "get_scene_info", "params": {}}).encode())
    data = json.loads(s.recv(1024 * 1024))
    if data.get("status") == "error":
        raise RuntimeError(data)
print("ALL_SERVICES_READY")
