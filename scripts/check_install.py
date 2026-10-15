"""Verify source, dataset schema, models, and runtime dependencies without generation."""

import argparse, importlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_pace.runtime import load, path, environment

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/runtime.json")
p.add_argument("--training", action="store_true", help="Also check the optional RL environment and AReaL patch")
a = p.parse_args()
c = load(a.config)
os.environ.update(environment(c))
sys.path.insert(0, str(ROOT / "server"))
from sage_rl.task_schema import load_tasks

counts = {
    name: len(load_tasks(ROOT / "data" / f"{name}.jsonl"))
    for name in ["train", "test"]
}
for name in [
    "torch",
    "trimesh",
    "trimesh.ray.ray_pyembree",
    "mcp",
    "sentence_transformers",
    "pytorch3d",
    "nvdiffrast.torch",
    "clip",
    "taming",
    "sage_rl.agent.sage_efficiency_room_agent",
]:
    importlib.import_module(name)
import trimesh
ray = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.creation.box())
assert ray.intersects_any([[0, 0, 2]], [[0, 0, -1]]).item()
for name in ["actor", "vlm"]:
    if name == "actor" and c[name]["mode"] == "api":
        continue
    directory = path(c["paths"]["models"]) / c[name]["model"]
    if not (directory / "config.json").exists():
        raise FileNotFoundError(directory / "config.json")
if c["actor"]["mode"] == "local":
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(
        path(c["paths"]["models"]) / c["actor"]["model"], local_files_only=True
    )
for name in ["scene", "trellis", "isaac", "vllm"] + (["rl"] if a.training else []):
    if not path(c["python"][name]).is_file():
        raise FileNotFoundError(c["python"][name])
if a.training:
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT / "third_party/AReaL"),
            "apply",
            "--reverse",
            "--check",
            str(ROOT / "patches/areal/runtime.patch"),
        ],
        check=True,
    )

print(
    json.dumps(
        {
            "status": "passed",
            "datasets": counts,
            "cuda": __import__("torch").cuda.is_available(),
        },
        indent=2,
    )
)
