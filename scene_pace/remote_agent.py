"""Run an RL scene on a dedicated service while preserving proxy trajectories."""

import asyncio, copy, json, os
from urllib.parse import urlsplit, urlunsplit
import httpx
from sage_rl.agent.sage_proxy_workflow import (
    SageOpenAIProxyWorkflow,
    apply_sage_proxy_workflow_patch,
)


class RemoteSceneAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_rollout_meta = {}

    async def run(self, data, **extra_kwargs):
        endpoints = json.loads(os.environ["SCENEPACE_SCENE_ENDPOINTS"])
        endpoint = {key: extra_kwargs[key] for key in ("base_url", "api_key")}
        url = urlsplit(endpoint["base_url"])
        if url.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            host = os.environ.get("SCENEPACE_PROXY_HOST")
            if not host:
                raise RuntimeError(
                    "Configure training.proxy_host as the routable proxy host"
                )
            endpoint["base_url"] = urlunsplit(url._replace(netloc=f"{host}:{url.port}"))
        # Services reject busy requests before accepting ownership. Never retry an
        # accepted episode after a network timeout (it may still be executing).
        timeout = float(os.environ.get("SCENEPACE_EPISODE_TIMEOUT", "10800"))
        deadline = asyncio.get_running_loop().time() + timeout
        headers = {"Authorization": "Bearer " + os.environ["SCENEPACE_SCENE_KEY"]}
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            while asyncio.get_running_loop().time() < deadline:
                for address in endpoints:
                    response = await client.post(
                        address.rstrip("/") + "/episode",
                        json={
                            "data": data,
                            "kwargs": self.kwargs,
                            "endpoint": endpoint,
                        },
                        headers=headers,
                    )
                    if response.status_code == 409:
                        continue
                    response.raise_for_status()
                    result = response.json()
                    self.last_rollout_meta = result["metadata"]
                    return result["reward"]
                await asyncio.sleep(1)
        raise TimeoutError("No scene worker available before the episode deadline")


class IsolatedProxyWorkflow(SageOpenAIProxyWorkflow):
    async def arun_episode(self, engine, data):
        episode = copy.copy(self)
        episode.agent = RemoteSceneAgent(**self.agent.kwargs)
        return await SageOpenAIProxyWorkflow.arun_episode(episode, engine, data)


def _install_wrapper():
    apply_sage_proxy_workflow_patch()
    from areal.infra.remote_inf_engine import RemoteInfEngine

    previous = RemoteInfEngine._wrap_openai_agent
    if getattr(previous, "_scenepace_remote", False):
        return

    def wrap(engine, agent, proxy_addr):
        workflow = previous(engine, agent, proxy_addr)
        if isinstance(agent, RemoteSceneAgent):
            workflow.__class__ = IsolatedProxyWorkflow
        return workflow

    wrap._scenepace_remote = True
    RemoteInfEngine._wrap_openai_agent = wrap


_install_wrapper()
