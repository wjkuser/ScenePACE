"""Generate and evaluate a task using the same policy as RL."""

import argparse, asyncio, json, os, random, time, shutil
from pathlib import Path
from scene_pace.runtime import ROOT, load, path, environment


async def run(args):
    c = load(args.config)
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("SAGE_EXTERNAL_ACTOR_CONFIG", "SAGE_EXTERNAL_ACTOR_AUDIT"):
        os.environ.pop(name, None)
    os.environ.update(environment(c, out))
    from sage_rl.task_schema import load_tasks

    tasks = load_tasks(path(args.tasks))
    task = (
        next((x for x in tasks if x["task_id"] == args.task_id), None)
        if args.task_id
        else tasks[0]
    )
    if task is None:
        raise ValueError("Requested task ID not found")
    seed = args.seed if args.seed is not None else c["scene"]["seed"]
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    task = {**task, "seed": seed, "model_name": c["actor"]["served_name"]}
    task["environment"] = {
        **task["environment"],
        "max_tool_calls": c["scene"]["max_tool_calls"],
    }
    (out / "task.json").write_text(json.dumps(task, indent=2))
    key = os.environ.get(c["actor"]["api_key_env"], "EMPTY")
    if c["actor"]["mode"] == "api":
        f = out / "external-actor.json"
        f.write_text(
            json.dumps(
                {
                    "base_url": c["actor"]["endpoint"],
                    "api_key": key,
                    "wire_api": c["actor"]["wire_api"],
                    "output_tokens": c["actor"].get("api_output_tokens", 16384),
                    "reasoning_effort": c["actor"].get(
                        "api_reasoning_effort", "medium"
                    ),
                }
            )
        )
        f.chmod(0o600)
        os.environ["SAGE_EXTERNAL_ACTOR_CONFIG"] = str(f)
        os.environ["SAGE_EXTERNAL_ACTOR_AUDIT"] = str(out / "api-timing.jsonl")
    from sage_rl.agent.sage_efficiency_room_agent import SageEfficiencyRoomAgent

    agent = SageEfficiencyRoomAgent(
        max_tool_calls=c["scene"]["max_tool_calls"],
        sampling_temperature=c["actor"]["temperature"],
        use_final_appearance_critic=c["scene"]["evaluate"],
    )
    start = time.monotonic()
    try:
        await agent.run(task, base_url=c["actor"]["endpoint"], api_key=key)
        result = {
            **agent.last_rollout_meta,
            "task_id": task["task_id"],
            "seed": seed,
            "pipeline_seconds": time.monotonic() - start,
        }
        checks = result.get("verification", {}).get("checks", [])
        required = [
            x
            for x in checks
            if x["id"].startswith("object:")
            or x["section"] == "relations"
            or x["id"] == "door_access_clear"
        ]
        result["task_success"] = bool(required) and all(
            x["status"] == "pass" for x in required
        )
        layout = (
            path(c["paths"]["output"]) / "results" / str(result.get("layout_id", ""))
        )
        if result.get("layout_id") and layout.is_dir():
            shutil.copytree(layout, out / "room", dirs_exist_ok=True)
        result["room_directory"] = "room"
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in [
                        "task_id",
                        "seed",
                        "layout_id",
                        "object_count",
                        "terminal_reward",
                        "pipeline_seconds",
                        "outcome",
                    ]
                }
            ),
            flush=True,
        )
        if (
            not result.get("layout_id")
            or result.get("infrastructure_failure")
            or not result.get("object_count")
            or (
                c["scene"]["evaluate"]
                and (
                    not result.get("verification", {}).get("evaluable")
                    or not result.get("verification", {}).get("visual_available")
                )
            )
        ):
            raise RuntimeError(
                "Pipeline incomplete; inspect pipeline.log and result.json"
            )
    finally:
        if c["actor"]["mode"] == "api":
            f.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/runtime.json")
    p.add_argument("--tasks", default="data/test.jsonl")
    p.add_argument("--task-id")
    p.add_argument("--seed", type=int)
    p.add_argument("--output", default="outputs/scene")
    a = p.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
