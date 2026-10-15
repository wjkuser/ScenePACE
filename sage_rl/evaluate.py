"""Independent, append-only evaluation of a served policy on fixed task IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from importlib.metadata import version
from pathlib import Path
import time

from openai import AsyncOpenAI

# AReaL installs its event-loop policy during import, before asyncio.run creates
# the loop. Importing it inside evaluate breaks MCP subprocesses on Python 3.12.
from sage_rl.agent.sage_efficiency_room_agent import SageEfficiencyRoomAgent
from sage_rl.policy import TEMPERATURE, sampling_contract
from sage_rl.task_schema import load_tasks

TRAINABLE_OUTCOMES = {"success", "policy_failure"}


async def evaluate(args):
    runtime = {name: version(name) for name in ("torch", "transformers", "vllm")}
    tasks = load_tasks(args.tasks, args.split)
    if args.task_ids:
        requested = set(args.task_ids.split(","))
        tasks = [task for task in tasks if task["task_id"] in requested]
        if {task["task_id"] for task in tasks} != requested:
            raise ValueError(
                "Some requested task IDs are absent from the selected split"
            )
    if not tasks:
        raise ValueError("Evaluation selection is empty")
    async with AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY") as client:
        models = await client.models.list()
    policy = next(model for model in models.data if model.id == "sage-policy")
    served_root = policy.model_dump()["root"]
    if Path(served_root).resolve() != Path(args.checkpoint).resolve():
        raise ValueError(f"Endpoint serves {served_root}, expected {args.checkpoint}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    attempts = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row["checkpoint"] != args.checkpoint:
                raise ValueError("Output file belongs to a different checkpoint")
            if row.get("runtime") != runtime:
                raise ValueError(
                    "Output file belongs to a different evaluation runtime"
                )
            if row.get("sampling_contract") != sampling_contract(args.temperature):
                raise ValueError("Output file belongs to a different policy sampling contract")
            key = (row["task_id"], row["seed"])
            attempts[key] = max(attempts.get(key, 0), row["attempt"] + 1)
            if row["outcome"]["status"] in TRAINABLE_OUTCOMES:
                completed.add(key)
    unresolved = []
    for task in tasks:
        for seed in args.seeds:
            key = (task["task_id"], seed)
            if key in completed:
                continue
            first_attempt = attempts.get(key, 0)
            for attempt in range(first_attempt, 3):
                agent = SageEfficiencyRoomAgent(
                    max_tool_calls=task["environment"]["max_tool_calls"],
                    sampling_temperature=args.temperature,
                )
                started = time.monotonic()
                await agent.run(
                    {**task, "seed": seed, "model_name": "sage-policy"},
                    base_url=args.endpoint,
                    api_key="EMPTY",
                )
                result = {
                    **agent.last_rollout_meta,
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "seed": seed,
                    "attempt": attempt,
                    "checkpoint": args.checkpoint,
                    "served_root": served_root,
                    "temperature": args.temperature,
                    "runtime": runtime,
                    "sampling_contract": sampling_contract(args.temperature),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                with output.open("a") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if os.environ.get('SAGE_SWANLAB_MODE', 'disabled') != 'disabled':
                    from sage_rl.monitoring import log_evaluation_episode
                    log_evaluation_episode(result, sum(attempts.values()))
                    attempts[key] = attempt + 1
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "task_id",
                                "seed",
                                "attempt",
                                "outcome",
                                "elapsed_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
                if result["outcome"]["status"] in TRAINABLE_OUTCOMES:
                    completed.add(key)
                    break
                if result["outcome"]["status"] != "environment_failure":
                    break
            if key not in completed:
                unresolved.append(key)
    if unresolved:
        raise RuntimeError(f"Evaluation incomplete after retries: {unresolved}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument(
        "--checkpoint", required=True, help="Path of the model served at endpoint"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:1082/v1")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.environ.get('SAGE_SWANLAB_MODE', 'disabled') == 'disabled':
        asyncio.run(evaluate(args))
    else:
        import swanlab
        swanlab.init(project="sage-room-grpo",
                     experiment_name=os.environ.get('SAGE_SWANLAB_RUN_NAME', Path(args.output).stem),
                     config={"split": args.split, "seeds": args.seeds,
                             "checkpoint": args.checkpoint, "sampling": sampling_contract(args.temperature)},
                     logdir=str(Path(args.output).parent / 'swanlog'),
                     mode=os.environ['SAGE_SWANLAB_MODE'])
        try:
            asyncio.run(evaluate(args))
        finally:
            swanlab.finish()


if __name__ == "__main__":
    main()
