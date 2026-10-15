"""One isolated scene service per Isaac instance; accepts one episode at a time."""

import argparse, asyncio, json, os, secrets, signal, sys, time, uuid
from pathlib import Path
from scene_pace.runtime import ROOT, load, path, environment


async def episode(config, request, result):
    c = load(config)
    payload = json.loads(Path(request).read_text())
    out = Path(result).parent
    Path(request).unlink()
    os.environ.update(environment(c, out))
    from sage_rl.agent.sage_efficiency_room_agent import SageEfficiencyRoomAgent

    options = payload["kwargs"]
    options["server_paths"] = [str(ROOT / "server/layout_wo_robot.py")]
    agent = SageEfficiencyRoomAgent(**options)
    started = time.monotonic()
    reward = await agent.run(payload["data"], **payload["endpoint"])
    Path(result).write_text(
        json.dumps(
            {
                "reward": reward,
                "metadata": {
                    **agent.last_rollout_meta,
                    "pipeline_seconds": time.monotonic() - started,
                },
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--request")
    p.add_argument("--result")
    a = p.parse_args()
    if a.request:
        asyncio.run(episode(a.config, a.request, a.result))
        return
    from fastapi import FastAPI, HTTPException, Request
    import uvicorn

    app = FastAPI()
    c = load(a.config)
    state = {"busy": False}
    key = os.environ.get("SCENEPACE_SCENE_KEY")
    if not key:
        raise ValueError("Export SCENEPACE_SCENE_KEY for scene-worker authentication")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "busy": state["busy"]}

    @app.post("/episode")
    async def run(request: Request):
        if not secrets.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + key
        ):
            raise HTTPException(401)
        if state["busy"]:
            raise HTTPException(409, "Scene worker is busy")
        state["busy"] = True
        proc = None
        out = path(c["paths"]["output"]) / "training-scenes" / uuid.uuid4().hex
        out.mkdir(parents=True)
        req = out / "request.private.json"
        result = out / "result.json"
        try:
            req.write_text(json.dumps(await request.json()))
            req.chmod(0o600)
            with (out / "pipeline.log").open("wb") as log:
                proc = await asyncio.create_subprocess_exec(
                    str(path(c["python"]["scene"])),
                    "-m",
                    "scene_pace.worker",
                    "--config",
                    str(path(a.config)),
                    "--request",
                    str(req),
                    "--result",
                    str(result),
                    cwd=ROOT,
                    env=environment(c, out),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                await asyncio.wait_for(
                    proc.wait(), timeout=c["scene"]["timeout_seconds"] + 180
                )
            if proc.returncode or not result.exists():
                raise RuntimeError("Scene subprocess failed; inspect its pipeline.log")
            return json.loads(result.read_text())
        finally:
            if proc and proc.returncode is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), 30)
                except asyncio.TimeoutError:
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
            req.unlink(missing_ok=True)
            state["busy"] = False

    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
