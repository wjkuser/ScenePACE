"""Resolve all user-editable paths relative to the checkout."""

import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(filename="configs/runtime.json"):
    return json.loads(path(filename).read_text())


def environment(config, output=None):
    c = config
    model = path(c["paths"]["models"])
    out = path(output or c["paths"]["output"])
    cache = path(c["paths"]["cache"])
    for d in [out, cache, out / "materials", out / "results"]:
        d.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    for private_name in ("SAGE_EXTERNAL_ACTOR_CONFIG", "SAGE_EXTERNAL_ACTOR_AUDIT"):
        e.pop(private_name, None)
    e.update(
        SCENEPACE_OUTPUT=str(out),
        SCENEPACE_MAX_TOOL_CALLS=str(c["scene"]["max_tool_calls"]),
        SCENEPACE_ACTOR_MODEL=str(model / c["actor"]["model"]),
        SCENEPACE_CONTEXT_TOKENS=str(c["actor"]["context_tokens"]),
        SCENEPACE_OUTPUT_TOKENS=str(c["actor"]["output_tokens"]),
        SCENEPACE_TEMPERATURE=str(c["actor"]["temperature"]),
    )
    e.update(
        SAGE_PROJECT=str(ROOT),
        SAGE_EFFICIENCY_ROOT=str(ROOT),
        SAGE_PYTHON_BIN=str(path(c["python"]["scene"])),
        SAGE_RENDER_PYTHON=str(path(c["python"]["isaac"])),
        SAGE_SCENE_CUDA_DEVICES=str(c["scene"]["gpu"]),
        SAGE_ISAAC_GPU=str(c["scene"]["isaac_gpu"]),
        ISAAC_MCP_PORT=str(c["scene"]["isaac_port"]),
        SAGE_LOG_DIR=str(out / "logs"),
        SAGE_RUN_DIR=str(out / "run"),
        SCENEPACE_STAGE_TIMING=str(int(c["scene"]["stage_timing"])),
        SAGE_RESULTS_DIR=str(path(c["paths"]["output"]) / "results"),
        SAGE_GENERATED_ASSETS_ROOT=str(out / "assets"),
        SAGE_RL_CRITIC_LOG_DIR=str(out / "critics"),
        SAGE_RL_TRAJECTORY_DUMP_DIR=str(out / "trajectories"),
        SAGE_RL_SEMANTIC_CACHE=str(cache / "semantic.sqlite"),
        SAGE_GENERATION_TASK_FILE=str(ROOT / "configs/generation_prompt.txt"),
        SAGE_RL_TOKENIZER_PATH=str(model / c["actor"]["model"]),
        SAGE_RL_MAX_TOOL_CALLS=str(c["scene"]["max_tool_calls"]),
        SAGE_RL_SESSION_TIMEOUT_SECONDS=str(c["scene"]["timeout_seconds"]),
        SAGE_CLIENT_MAX_TOKENS=str(c["actor"]["output_tokens"]),
        SAGE_VLM_BASE_URL=c["vlm"]["endpoint"],
        SAGE_VLM_MODEL=c["vlm"]["served_name"],
        SAGE_VLM_API_KEY=e.get(c["vlm"]["api_key_env"], "EMPTY"),
        SAGE_VLM_LOCAL="1",
        SAGE_VLM_DUAL_INSTANCE="0",
        SAGE_APPEARANCE_EVAL_LOCAL="1",
        SAGE_APPEARANCE_EVAL_URL=c["vlm"]["endpoint"],
        SAGE_APPEARANCE_EVAL_MODEL=c["vlm"]["served_name"],
        TEXT_TO_3D_BACKEND="trellis",
        TEXT_TO_3D_SERVER_URL=c["trellis"]["endpoint"],
        TRELLIS_MODEL_PATH=str(model / c["trellis"]["model"]),
        MATFUSE_CKPT_PATH=str(model / "matfuse-full.ckpt"),
        MATFUSE_TEXT_ENCODER_PATH=str(model / "sentence-transformers-clip-ViT-B-16"),
        MATFUSE_IMAGE_ENCODER_PATH=str(model / "ViT-B-16.pt"),
        HF_HOME=str(cache / "huggingface"),
        XDG_CACHE_HOME=str(cache),
        SAGE_SWANLAB_MODE="disabled",
        SAGE_ASSET_MODE="generation",
        SAGE_RETRIEVAL_ONLY="0",
        SAGE_RENDER_BACKEND="isaac",
        SAGE_EVAL_ITERATIONS="1",
        SAGE_ASSET_WORKERS=str(c["scene"]["asset_workers"]),
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        PYTHONPATH=os.pathsep.join(
            map(str, [ROOT, ROOT / "client"])
        ),
    )
    key = out / "runtime-key.json"
    key.write_text("{}")
    key.chmod(0o600)
    e["SAGE_KEY_FILE"] = e["SAGE_MCP_SERVER_KEY_FILE"] = str(key)
    return e
