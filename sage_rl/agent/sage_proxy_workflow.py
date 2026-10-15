"""AReaL proxy workflow with Sage rollout metadata on trajectories."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.utils.data import concat_padded_tensors
from sage_rl.task_schema import task_key
from sage_rl.turn_export import mask_current_turn

_SAGE_PROXY_PATCH_APPLIED = False


def apply_sage_proxy_workflow_patch() -> None:
    """Ensure rollout workers wrap SageRoomAgent with SageOpenAIProxyWorkflow."""
    global _SAGE_PROXY_PATCH_APPLIED
    if _SAGE_PROXY_PATCH_APPLIED:
        return

    from areal.infra.remote_inf_engine import RemoteInfEngine

    original = RemoteInfEngine._wrap_openai_agent

    def _wrap_openai_agent(self, agent, proxy_addr: str):
        agent_cfg = self.config.agent
        return SageOpenAIProxyWorkflow(
            mode=agent_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=agent_cfg.admin_api_key,
            discount=agent_cfg.turn_discount,
            export_style=agent_cfg.export_style,
            subproc_max_workers=agent_cfg.subproc_max_workers,
            proxy_gateway_addr=self._proxy_gateway_addr,
        )

    RemoteInfEngine._wrap_openai_agent = _wrap_openai_agent  # type: ignore[method-assign]
    _SAGE_PROXY_PATCH_APPLIED = True


class SageOpenAIProxyWorkflow(OpenAIProxyWorkflow):
    """Attach Sage agent metadata to exported trajectories for rollout filtering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sage_layout_id = ""
        self._sage_agent_reward = 0.0
        self._sage_terminal_reward = 0.0
        self._sage_tool_call_count = 0
        self._sage_max_tool_calls = 0
        self._sage_completion_count = 0
        self._sage_tool_names: list[str] = []
        self._sage_object_count = -1
        self._sage_empty_layout = 0
        self._sage_infrastructure_failure = 0
        self._sage_failure_reason = ""
        self._sage_blocked_layout_regenerations = 0

    def _capture_agent_metadata(self, result: Any) -> Any:
        self._sage_layout_id = ""
        self._sage_agent_reward = 0.0
        self._sage_terminal_reward = 0.0
        self._sage_tool_call_count = 0
        self._sage_max_tool_calls = 0
        self._sage_completion_count = 0
        self._sage_tool_names = []
        self._sage_object_count = -1
        self._sage_empty_layout = 0
        self._sage_infrastructure_failure = 0
        self._sage_failure_reason = ""
        self._sage_blocked_layout_regenerations = 0
        meta = getattr(self.agent, "last_rollout_meta", None) or {}
        if meta:
            self._sage_layout_id = str(meta.get("layout_id") or "")
            if meta.get("agent_reward") is not None:
                self._sage_agent_reward = float(meta["agent_reward"])
            if meta.get("terminal_reward") is not None:
                self._sage_terminal_reward = float(meta["terminal_reward"])
            self._sage_tool_call_count = int(meta.get("tool_call_count") or 0)
            self._sage_max_tool_calls = int(meta.get("max_tool_calls") or 0)
            self._sage_completion_count = int(meta.get("completion_count") or 0)
            self._sage_tool_names = [str(value) for value in meta.get("tool_names", [])]
            self._sage_object_count = int(meta.get("object_count", -1))
            self._sage_empty_layout = int(meta.get("empty_layout") or 0)
            self._sage_infrastructure_failure = int(
                meta.get("infrastructure_failure") or 0
            )
            self._sage_failure_reason = str(meta.get("failure_reason") or "")
            self._sage_blocked_layout_regenerations = int(
                meta.get("blocked_layout_regenerations") or 0
            )
        if isinstance(result, (int, float)):
            if self._sage_agent_reward <= 0:
                self._sage_agent_reward = float(result)
            return float(result)
        return result

    async def _run_agent(self, session_api_key: str, data: dict):
        result = await super()._run_agent(session_api_key, data)
        return self._capture_agent_metadata(result)

    async def arun_episode(self, engine, data: dict[str, Any]):
        started = time.monotonic()
        policy_version = engine.get_version()
        self._sage_layout_id = ""
        self._sage_agent_reward = 0.0
        self._sage_terminal_reward = 0.0
        self._sage_tool_call_count = 0
        self._sage_max_tool_calls = 0
        self._sage_completion_count = 0
        self._sage_tool_names = []
        self._sage_object_count = -1
        self._sage_empty_layout = 0
        self._sage_infrastructure_failure = 0
        self._sage_failure_reason = ""
        self._sage_blocked_layout_regenerations = 0
        episode_timeout = float(
            os.environ.get("SAGE_RL_ROLLOUT_EPISODE_TIMEOUT_SECONDS", "7200")
        )
        export_timeout = float(
            os.environ.get("SAGE_RL_PROXY_EXPORT_TIMEOUT_SECONDS", "600")
        )
        try:
            interactions = await asyncio.wait_for(
                self._run_episode_with_export_timeout(
                    engine, data, export_timeout=export_timeout
                ),
                timeout=episode_timeout,
            )
        except asyncio.TimeoutError:
            print(
                "[SageOpenAIProxyWorkflow] rollout episode timed out after "
                f"{episode_timeout:.0f}s",
                flush=True,
            )
            return None
        if interactions is None:
            return None
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if not (
            isinstance(interactions, dict)
            and interactions
            and all(
                isinstance(v, InteractionWithTokenLogpReward)
                for v in interactions.values()
            )
        ):
            return interactions

        # A single external/incomplete interaction must not downgrade the
        # complete rollout to string-only data. Keep tensor-backed turns so
        # the actor can still train on the useful part of the trajectory.
        tensor_interactions = [
            interaction
            for interaction in interactions.values()
            if interaction.has_tensor_data
        ]
        if not tensor_interactions:
            print(
                "[SageOpenAIProxyWorkflow] no tensor-backed interactions; "
                "rejecting non-trainable rollout",
                flush=True,
            )
            return None
        if len(tensor_interactions) != len(interactions):
            print(
                "[SageOpenAIProxyWorkflow] dropping "
                f"{len(interactions) - len(tensor_interactions)} non-tensor "
                "interaction(s) from rollout export",
                flush=True,
            )

        tensors = [interaction.to_tensor_dict() for interaction in tensor_interactions]
        if self.export_style == "individual":
            tensors = [mask_current_turn(turn) for turn in tensors]
        traj = concat_padded_tensors(tensors)
        traj["sage_layout_id"] = self._sage_layout_id
        traj["sage_agent_reward"] = self._sage_agent_reward
        traj["sage_terminal_reward"] = self._sage_terminal_reward
        traj["sage_tool_call_count"] = self._sage_tool_call_count
        traj["sage_max_tool_calls"] = self._sage_max_tool_calls
        traj["sage_completion_count"] = self._sage_completion_count
        traj["sage_tool_names"] = list(self._sage_tool_names)
        traj["sage_object_count"] = self._sage_object_count
        traj["sage_empty_layout"] = self._sage_empty_layout
        traj["sage_infrastructure_failure"] = self._sage_infrastructure_failure
        traj["sage_failure_reason"] = self._sage_failure_reason
        traj["sage_blocked_layout_regenerations"] = (
            self._sage_blocked_layout_regenerations
        )
        traj["sage_prompt_key"] = task_key(data) if "task_id" in data else data["room_desc"]
        meta = getattr(self.agent, "last_rollout_meta", {})
        traj["sage_outcome"] = meta.get("outcome", {})
        task = json.loads(data['task_json']) if 'task_json' in data else data
        traj['sage_episode'] = {
            'task_id': task.get('task_id'), 'family_id': task.get('family_id'), 'seed': data.get('seed'),
            'policy_version': policy_version,
            'visit': data.get('visit'),
            'context_budget_exhausted': meta.get('context_budget_exhausted', False),
            'context_compactions': meta.get('context_compactions', []),
            'tool_call_limit_reached': meta.get('tool_call_limit_reached', False),
            'elapsed_seconds': round(time.monotonic() - started, 2),
            'verification': meta.get('verification'),
            'sampling_contract': meta.get('sampling_contract'),
            'task_feedback': meta.get('task_feedback', []),
        }

        traj["sage_scene_type"] = task["scene_type"]
        return traj

    async def _run_episode_with_export_timeout(
        self,
        engine,
        data: dict[str, Any],
        *,
        export_timeout: float,
    ):
        if self.mode != "inline":
            return await super().arun_episode(engine, data)

        from areal.infra import workflow_context
        from areal.utils import stats_tracker

        from areal.experimental.openai.proxy.client_session import OpenAIProxyClient

        task_id = workflow_context.get().task_id
        http_session = await workflow_context.get_aiohttp_session()
        await self._grant_capacity(http_session)

        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url=self.proxy_addr,
            task_id=str(task_id),
            admin_api_key=self._admin_api_key,
        )
        async with proxy_client:
            try:
                rewards = await self._run_agent(proxy_client.session_api_key, data)
            except Exception as exc:
                print(
                    "[SageOpenAIProxyWorkflow] agent task failed "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
                raise
            if isinstance(rewards, dict):
                for completion_id, reward in rewards.items():
                    await proxy_client.set_reward(completion_id, reward)
            elif isinstance(rewards, float):
                await proxy_client.set_last_reward(rewards)
            else:
                raise ValueError(f"Invalid reward type: {type(rewards)}")

        try:
            interactions = await asyncio.wait_for(
                proxy_client.export_interactions(
                    discount=self.discount,
                    style=self.export_style,
                ),
                timeout=export_timeout,
            )
        except asyncio.TimeoutError:
            print(
                "[SageOpenAIProxyWorkflow] export_interactions timed out after "
                f"{export_timeout:.0f}s",
                flush=True,
            )
            return None

        if interactions:
            last_id = list(interactions.keys())[-1]
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                reward=last_reward
            )
        return interactions
