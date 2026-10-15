"""Shared policy request settings for online rollouts and held-out evaluation."""

import os

TEMPERATURE = float(os.environ.get("SCENEPACE_TEMPERATURE", "0.6"))
TOP_P = 0.95
TOP_K = 20
STOP_TOKEN_IDS = [151645, 151643]
MAX_CONTEXT_TOKENS = int(os.environ.get("SCENEPACE_CONTEXT_TOKENS", "16384"))
MAX_OUTPUT_TOKENS = int(os.environ.get("SCENEPACE_OUTPUT_TOKENS", "2048"))


def sampling_contract(temperature=TEMPERATURE):
    return {
        'temperature': temperature, 'top_p': TOP_P, 'top_k': TOP_K,
        'stop_token_ids': STOP_TOKEN_IDS, 'enable_thinking': False,
        'frequency_penalty': 0.0, 'presence_penalty': 0.0,
        'repetition_penalty': 1.0, 'max_context_tokens': MAX_CONTEXT_TOKENS,
        'max_output_tokens': MAX_OUTPUT_TOKENS, 'chat_template_type': 'hf',
        'tool_call_parser': 'qwen3_coder', 'seed_rule': 'episode_seed + completion_index',
    }
