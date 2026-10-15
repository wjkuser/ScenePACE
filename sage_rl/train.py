"""GRPO training entry for SAGE room-generation RL."""

from __future__ import annotations

import os
import pathlib
import sys
import json

from datasets import Dataset

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
AREAL_ROOT = pathlib.Path(os.environ.get("AREAL_ROOT", str(PROJECT_ROOT / "third_party/AReaL")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AREAL_ROOT) not in sys.path:
    sys.path.insert(0, str(AREAL_ROOT))

from areal.api.cli_args import load_expr_config  # noqa: E402
from areal.utils.hf_utils import load_hf_tokenizer  # noqa: E402

from sage_rl.policy import TOP_P, TOP_K, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, sampling_contract
from sage_rl.task_schema import load_tasks
from sage_rl.configs import SageRLConfig  # noqa: E402
from sage_rl.schedule import training_rows
from sage_rl.agent.sage_proxy_workflow import apply_sage_proxy_workflow_patch  # noqa: E402
from sage_rl.grpo_trainer import SageGRPOTrainer, trainer_kwargs_from_env  # noqa: E402


def main(args, trainer_type=SageGRPOTrainer):
    apply_sage_proxy_workflow_patch()
    config, _ = load_expr_config(args, SageRLConfig)
    if config.actor.backend.startswith("fsdp:") and config.actor.attn_impl != "flash_attention_2":
        raise ValueError(
            "Sage's packed FSDP batches require flash_attention_2. SDPA/eager "
            "do not isolate sequences in the pinned AReaL forward path."
        )
    config.gconfig.top_p = TOP_P
    config.gconfig.top_k = TOP_K
    config.gconfig.max_tokens = MAX_CONTEXT_TOKENS
    config.gconfig.max_new_tokens = MAX_OUTPUT_TOKENS
    config.actor.temperature = config.gconfig.temperature
    config.rollout.agent.chat_template_type = "hf"
    config.rollout.agent.engine_max_tokens = MAX_CONTEXT_TOKENS
    print("SAGE_POLICY_CONTRACT " + json.dumps(sampling_contract(config.gconfig.temperature)))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    prompt_file = os.environ.get("SAGE_RL_PROMPT_FILE", "").strip()
    tasks = load_tasks(prompt_file)
    if config.train_dataset.batch_size != config.tasks_per_step * config.gconfig.n_samples:
        raise ValueError('batch_size must equal tasks_per_step * n_samples')
    train_dataset = Dataset.from_list(training_rows(
        tasks, visits=config.task_visits, group_size=config.gconfig.n_samples,
        seed=config.seed, tasks_per_step=config.tasks_per_step,
    ))

    workflow_kwargs = {
        "server_paths": config.server_paths,
        "sampling_temperature": config.gconfig.temperature,
        "max_tool_calls": config.max_tool_calls,
        "use_final_appearance_critic": config.use_final_appearance_critic,
    }

    extra_trainer_kwargs = trainer_kwargs_from_env()

    with trainer_type(
        config,
        train_dataset=train_dataset,
        **extra_trainer_kwargs,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            workflow_kwargs=workflow_kwargs,
            dynamic_filter_fn="sage_rl.rollout_filter.accept_valid_layout_rollout",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
