"""GRPO trainer hooks for GPU pinning and optional external rollout vLLM."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from areal import PPOTrainer
from areal.api.cli_args import InferenceEngineConfig, vLLMConfig
from areal.api.io_struct import LocalInfServerInfo, StepInfo
from areal.engine import RemotevLLMEngine
from areal.infra import LocalScheduler
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.utils.environ import is_single_controller

from sage_rl.collection import collect_task_batch
from sage_rl.rollout_batch import build_variable_turn_grpo_batch, dump_rollout_trajectories, read_engine_int_attr, validate_expanded_rollout_sequences

_SAGE_EXPAND_PATCH_APPLIED = False
_DISPATCHER_CANCEL_PATCH_APPLIED = False


def _partition_complete_prompt_groups(
    trajectories: list[dict[str, Any]],
    *,
    group_size: int,
    group_count: int,
) -> list[list[dict[str, Any]]]:
    """Select complete same-prompt GRPO groups from out-of-order results."""
    if group_size <= 0 or group_count <= 0:
        raise ValueError("group_size and group_count must be positive")

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        prompt_key = str(trajectory.get("sage_prompt_key", "") or "__default__")
        by_prompt.setdefault(prompt_key, []).append(trajectory)

    groups: list[list[dict[str, Any]]] = []
    for prompt_trajectories in by_prompt.values():
        # A single prompt may fill multiple GRPO groups (e.g. a one-prompt
        # dataset with batch_size > n_samples); keep slicing complete groups
        # until this prompt's rollouts are exhausted or enough groups exist.
        while len(prompt_trajectories) >= group_size and len(groups) < group_count:
            groups.append(prompt_trajectories[:group_size])
            prompt_trajectories = prompt_trajectories[group_size:]
        if len(groups) >= group_count:
            break
    return groups


def _return_accepted_rollout_capacity(staleness_manager: Any, count: int) -> int:
    """Return capacity for accepted rollouts that will never be trained."""
    if count <= 0:
        return 0

    with staleness_manager.lock:
        consumer_batch_size = max(1, int(staleness_manager.consumer_batch_size))
        version = max(0, int(staleness_manager.version_provider.get_version()))
        consumed_floor = version * consumer_batch_size
        returnable = max(
            0,
            int(staleness_manager.rollout_stat.accepted) - consumed_floor,
        )
        returned = min(count, returnable)
        staleness_manager.rollout_stat.accepted -= returned
    return returned


def _ensure_dispatcher_cancel_hooks() -> None:
    global _DISPATCHER_CANCEL_PATCH_APPLIED
    if _DISPATCHER_CANCEL_PATCH_APPLIED:
        return

    from areal.infra.workflow_executor import BatchTaskDispatcher

    def cancel_pending_inputs(self) -> int:
        with self._input_cv:
            cancelled = len(self._pending_inputs)
            if cancelled <= 0:
                return 0
            cancelled_ids = {task_input.task_id for task_input in self._pending_inputs}
            self._pending_inputs.clear()
            with self.staleness_manager.lock:
                self.staleness_manager.rollout_stat.enqueued = max(
                    0,
                    self.staleness_manager.rollout_stat.enqueued - cancelled,
                )
            for task_id in cancelled_ids:
                self._active_task_ids.discard(task_id)
            self._input_cv.notify()
        return cancelled

    def discard_pending_results(self) -> tuple[int, int]:
        with self._result_cv:
            discarded = len(self._pending_results)
            if discarded <= 0:
                return 0, 0
            accepted = sum(
                result.data is not None for result in self._pending_results.values()
            )
            orphan_ids = list(self._pending_results.keys())
            self._pending_results.clear()
            for task_id in orphan_ids:
                self._active_task_ids.discard(task_id)
            self._result_cv.notify_all()
        returned = _return_accepted_rollout_capacity(
            self.staleness_manager,
            accepted,
        )
        return discarded, returned

    BatchTaskDispatcher.cancel_pending_inputs = cancel_pending_inputs  # type: ignore[attr-defined]
    BatchTaskDispatcher.discard_pending_results = discard_pending_results  # type: ignore[attr-defined]
    _DISPATCHER_CANCEL_PATCH_APPLIED = True


def parse_rollout_vllm_hostport(url: str) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    if not parsed.port:
        raise ValueError(f"rollout vLLM url must include port: {url}")
    return host, int(parsed.port)


class SageGRPOTrainer(PPOTrainer):
    """Pin actor/rollout GPUs; optionally attach to a pre-started client vLLM."""

    def __init__(
        self,
        *args,
        rollout_vllm_url: str | None = None,
        actor_gpu_devices: list[int] | None = None,
        rollout_gpu_devices: list[int] | None = None,
        **kwargs,
    ):
        self._rollout_vllm_url = rollout_vllm_url
        self._actor_gpu_devices = actor_gpu_devices
        self._rollout_gpu_devices = rollout_gpu_devices
        self._sage_group_stats_step = 0
        self._sage_update_stats_step = 0

        super().__init__(*args, **kwargs)
        _ensure_dispatcher_cancel_hooks()
        self._patch_ungrouped_rollout_prepare_batch()
        self._patch_rollout_prepare_batch_retry()
        self._patch_expand_trajectories_before_redistribute()

    def _patch_expand_trajectories_before_redistribute(self) -> None:
        """Keep precomputed variable-turn completions split for DP redistribution."""
        global _SAGE_EXPAND_PATCH_APPLIED
        if _SAGE_EXPAND_PATCH_APPLIED:
            return

        original = DistRolloutCoordinator._broadcast_and_redistribute_trajectories

        def patched(coordinator_self, trajectories):
            if trajectories is not None:
                dp_size = 1
                ppo_n_minibatches = 1
                engine = coordinator_self.train_engine
                if engine is not None:
                    dp_size = read_engine_int_attr(
                        engine, "data_parallel_world_size", dp_size
                    )
                    config = getattr(engine, "config", None)
                    if config is not None:
                        ppo_n_minibatches = max(
                            1, int(getattr(config, "ppo_n_minibatches", 1))
                        )
                trajectories = validate_expanded_rollout_sequences(
                    trajectories,
                    dp_size=dp_size,
                    ppo_n_minibatches=ppo_n_minibatches,
                    turn_group_size=1,
                )
            return original(coordinator_self, trajectories)

        DistRolloutCoordinator._broadcast_and_redistribute_trajectories = patched
        _SAGE_EXPAND_PATCH_APPLIED = True


    def _patch_rollout_prepare_batch_retry(self) -> None:
        """Collect complete groups; checklist mode retains ordinary policy failures."""
        rollout = getattr(self.actor, "rollout", None)
        if rollout is None:
            return
        prepare_timeout_s = float(os.environ.get("SAGE_RL_ROLLOUT_PREPARE_TIMEOUT_SECONDS", "14400"))

        def _trajectory_dump_dir() -> Path:
            from pathlib import Path

            default = (
                Path(self.config.cluster.fileroot)
                / "trajectories"
                / self.config.experiment_name
                / self.config.trial_name
            )
            raw = os.environ.get("SAGE_RL_TRAJECTORY_DUMP_DIR", "").strip()
            return Path(raw) if raw else default

        def _actor_dp_requirements() -> tuple[int, int]:
            dp_size = 1
            ppo_n_minibatches = max(
                1, int(getattr(self.config.actor, "ppo_n_minibatches", 1))
            )
            backend = str(getattr(self.config.actor, "backend", ""))
            match = re.search(r":d(\d+)p", backend)
            if match:
                dp_size = max(1, int(match.group(1)))
            engine = getattr(self.actor, "train_engine", None)
            if engine is None:
                engine = getattr(self.actor, "_train_engine", None)
            if engine is not None:
                dp_size = read_engine_int_attr(
                    engine, "data_parallel_world_size", dp_size
                )
                config = getattr(engine, "config", None)
                if config is not None:
                    ppo_n_minibatches = max(
                        1, int(getattr(config, "ppo_n_minibatches", ppo_n_minibatches))
                    )
            return dp_size, ppo_n_minibatches

        def _finalize_batch(
            groups: list[list[dict[str, Any]]], required: int
        ) -> list[dict[str, Any]]:
            batch = [trajectory for group in groups for trajectory in group]
            if len(batch) != required:
                raise RuntimeError(
                    "Cannot finalize incomplete GRPO prompt groups: "
                    f"got {len(batch)} rollouts, expected {required}"
                )
            dump_rollout_trajectories(batch, _trajectory_dump_dir(), tag="valid_batch")
            expanded: list[dict[str, Any]] = []
            batch_stats = []
            for group in groups:
                self._sage_group_stats_step += 1
                group_expanded, stats = build_variable_turn_grpo_batch(
                    group,
                    group_size=int(self.config.gconfig.n_samples),
                    stats_path=_trajectory_dump_dir() / "group_reward_stats.jsonl",
                    step=self._sage_group_stats_step,
                )
                batch_stats.append(stats)
                if self._sage_group_stats_step == 1:
                    visual_complete = all(
                        trajectory.get('sage_episode', {}).get('verification', {}).get('visual_available')
                        for trajectory in group
                    )
                    if not visual_complete or stats['terminal_std_sample'] <= 1e-5:
                        raise RuntimeError(
                            'First-group quality gate failed: inspect saved rollouts before updating. '
                            f"visual_complete={visual_complete}, reward_std={stats['terminal_std_sample']}"
                        )
                expanded.extend(group_expanded)
            from sage_rl.monitoring import log_training_batch
            log_training_batch(groups, batch_stats, self._sage_update_stats_step)
            self._sage_update_stats_step += 1
            dp_size, ppo_n_minibatches = _actor_dp_requirements()
            trainable = validate_expanded_rollout_sequences(
                expanded,
                dp_size=dp_size,
                ppo_n_minibatches=ppo_n_minibatches,
                turn_group_size=1,
            )
            if is_single_controller():
                try:
                    # The completion tensors are now local. Release the original
                    # rollout RTensor shards before the next step starts.
                    self.actor.clear_batches(batch)
                except Exception as exc:
                    print(
                        "[SageGRPOTrainer] WARNING: failed to clear localized "
                        f"rollout shards ({exc})",
                        flush=True,
                    )
            return trainable

        def prepare_batch_with_retry(*args, **kwargs):
            dataloader = kwargs.get("dataloader")
            if dataloader is not None and getattr(dataloader, "batch_size", None):
                batch_size = int(dataloader.batch_size)
                required = batch_size
            else:
                required = int(getattr(self.config.gconfig, "n_samples", 4))
            group_size = int(getattr(self.config.gconfig, "n_samples", 1))
            if required % group_size:
                raise RuntimeError(
                    "train batch size must contain complete GRPO groups: "
                    f"batch_size={required}, group_size={group_size}"
                )
            group_count = required // group_size
            kwargs.pop("should_accept_fn", None)
            self._ensure_rollout_data_generator(rollout, *args, **kwargs)
            dispatcher = self._get_rollout_dispatcher(rollout)
            batch = collect_task_batch(
                rollout, dispatcher, required, group_size,
                timeout=prepare_timeout_s,
                release_capacity=_return_accepted_rollout_capacity,
                audit_path=_trajectory_dump_dir() / 'collection_attempts.jsonl',
            )
            groups = _partition_complete_prompt_groups(
                batch, group_size=group_size, group_count=group_count,
            )
            return _finalize_batch(groups, required)

        rollout.prepare_batch = prepare_batch_with_retry  # type: ignore[method-assign]

    @staticmethod
    def _get_rollout_dispatcher(rollout: Any) -> Any | None:
        dispatcher = getattr(rollout, "dispatcher", None)
        if dispatcher is not None:
            return dispatcher
        workflow_executor = getattr(rollout, "workflow_executor", None)
        return getattr(workflow_executor, "dispatcher", None)


    @staticmethod
    def _ensure_rollout_data_generator(
        rollout: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # AReaL may create and cache a generator during controller setup.  That
        # generator captures the original should_accept_fn and group_size, so
        # reusing it here can silently reject every result before this collector
        # sees it.  Only reuse a generator created by this patch.
        if getattr(rollout, "_sage_data_generator_initialized", False):
            return

        resolve = getattr(rollout, "_resolve_workflow_str", None)
        if resolve is None:
            raise RuntimeError(
                "Cannot initialize rollout data_generator: unsupported rollout type"
            )

        from areal.infra.controller.rollout_controller import _RemoteRolloutTaskInput
        from areal.utils.data import cycle_dataloader

        dataloader = kwargs.get("dataloader")
        if dataloader is None and args:
            dataloader = args[0]
        workflow = kwargs.get("workflow")
        if workflow is None and len(args) > 1:
            workflow = args[1]
        workflow_kwargs = kwargs.get("workflow_kwargs")
        if workflow_kwargs is None and len(args) > 2:
            workflow_kwargs = args[2]
        workflow_kwargs = workflow_kwargs or {}
        should_accept_fn = kwargs.get("should_accept_fn")
        # ``RLTrainer.train`` passes group_size positionally through
        # ``prepare_batch``; preserve it when the wrapper has captured
        # positional arguments instead of keyword arguments.
        positional_group_size = args[4] if len(args) > 4 else 1
        group_size = int(kwargs.get("group_size", positional_group_size))
        workflow_str = resolve(workflow)
        task_id_gen = rollout._task_id_generator

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=task_id_gen.next(),
                        group_size=group_size,
                    )

        rollout.data_generator = task_input_generator()
        rollout._sage_data_generator_initialized = True


    def _patch_ungrouped_rollout_prepare_batch(self) -> None:
        """OpenAI proxy workflows merge grouped rollouts into one trajectory.

        Use batch_size=n_samples with group_size=1 so each sample is a separate
        trajectory (required for actor DP partitioning).
        """
        actor = self.actor
        original = actor.prepare_batch

        def prepare_batch(*args, **kwargs):
            kwargs["group_size"] = 1
            return original(*args, **kwargs)

        actor.prepare_batch = prepare_batch  # type: ignore[method-assign]

    def _init_scheduler(self):
        if self.config.scheduler.type != "local":
            return super()._init_scheduler()

        gpu_devices = None
        if self._actor_gpu_devices and self._rollout_gpu_devices:
            gpu_devices = list(self._actor_gpu_devices) + list(self._rollout_gpu_devices)
        elif self._actor_gpu_devices:
            gpu_devices = list(self._actor_gpu_devices)

        if gpu_devices is not None:
            return LocalScheduler(
                exp_config=self.config,
                gpu_devices=gpu_devices,
            )
        return super()._init_scheduler()

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ):
        if (
            not self._rollout_vllm_url
            or not is_single_controller()
            or is_eval
        ):
            return super()._init_rollout(
                rollout_config,
                is_eval=is_eval,
                lora_path=lora_path,
            )

        if self.rollout_alloc.backend != "vllm":
            raise ValueError(
                "External rollout vLLM requires rollout.backend vllm:* "
                f"(got {self.config.rollout.backend})"
            )

        config = deepcopy(rollout_config)
        if config.agent is not None:
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
            server_args["enable_auto_tool_choice"] = True
            server_args["tool_call_parser"] = config.agent.tool_call_parser
            if config.agent.reasoning_parser:
                server_args["reasoning_parser"] = config.agent.reasoning_parser
        else:
            server_args = None

        controller = RemotevLLMEngine.as_controller(config, self.scheduler)
        host, port = parse_rollout_vllm_hostport(self._rollout_vllm_url)
        controller.initialize(
            role="rollout",
            server_args=None,
            server_infos=[LocalInfServerInfo(process=None, host=host, port=port)],
        )
        return controller


def trainer_kwargs_from_env() -> dict[str, Any]:
    rollout_url = os.environ.get("SAGE_RL_ROLLOUT_VLLM_URL", "").strip() or None
    actor_gpus_raw = os.environ.get("SAGE_RL_ACTOR_GPU_DEVICES", "").strip()
    rollout_gpus_raw = os.environ.get("SAGE_RL_ROLLOUT_GPU_DEVICES", "").strip()

    actor_gpu_devices = None
    rollout_gpu_devices = None
    if actor_gpus_raw:
        actor_gpu_devices = [
            int(item.strip()) for item in actor_gpus_raw.split(",") if item.strip()
        ]
    if rollout_gpus_raw:
        rollout_gpu_devices = [
            int(item.strip()) for item in rollout_gpus_raw.split(",") if item.strip()
        ]

    if not (rollout_url or actor_gpu_devices or rollout_gpu_devices):
        return {}

    kwargs: dict[str, Any] = {}
    if rollout_url:
        kwargs["rollout_vllm_url"] = rollout_url
    if actor_gpu_devices:
        kwargs["actor_gpu_devices"] = actor_gpu_devices
    if rollout_gpu_devices:
        kwargs["rollout_gpu_devices"] = rollout_gpu_devices
    return kwargs
