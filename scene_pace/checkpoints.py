"""Portable, integrity-checked training recovery bundles (only load trusted checkpoints)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import tempfile

MANIFEST = "full-checkpoint.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contract(config):
    """Training semantics only; exclude run paths, credentials and telemetry."""
    from omegaconf import OmegaConf

    if not isinstance(config, dict):
        config = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    actor = config["actor"]
    result = {
        key: config.get(key)
        for key in (
            "seed",
            "total_train_steps",
            "total_train_epochs",
            "task_visits",
            "tasks_per_step",
            "gconfig",
            "max_tool_calls",
            "train_dataset",
            "workflow",
            "use_final_appearance_critic",
            "reward",
            "critic",
        )
    }
    # Paths and node addresses can change, the numerical configuration cannot.
    result["actor"] = {
        k: v
        for k, v in actor.items()
        if k not in {"scheduling_spec", "path", "fileroot"}
    }
    result["rollout_backend"] = config["rollout"]["backend"]
    source = Path(os.environ.get("SAGE_RL_PROMPT_FILE", "data/train.jsonl"))
    result["task_file_sha256"] = digest(source)
    result["base_model"] = Path(config["actor"]["path"]).name
    repo = Path(__file__).resolve().parents[1]
    sources = sorted((repo / "sage_rl").rglob("*.py"))
    sources += sorted((repo / "scene_pace").rglob("*.py"))
    sources += [
        repo / "configs/generation_prompt.txt",
        repo / "client/client_generation_room_desc.py",
        repo / "server/layout_wo_robot.py",
        repo / "sage_rl/evaluation/visual_prompt.txt",
        repo / "third_party/AReaL/areal/engine/fsdp_engine.py",
        repo / "third_party/AReaL/areal/utils/recover.py",
    ]
    result["training_source_sha256"] = {
        str(p.relative_to(repo)): digest(p) for p in sources
    }

    return result


def recovery_root(config):
    from areal.utils.saver import Saver

    return Path(
        Saver.get_save_root(
            config.experiment_name, config.trial_name, config.recover.fileroot
        )
    )


def controller_state(trainer):
    import numpy as np
    import torch

    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        group_step=trainer._sage_group_stats_step,
        update_step=trainer._sage_update_stats_step,
        task_counter=trainer.rollout._task_id_generator._task_cnt,
    )


def restore_controller(trainer, root):
    import numpy as np
    import torch

    state = pickle.loads((root / "controller.pkl").read_bytes())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    trainer._sage_group_stats_step = state["group_step"]
    trainer._sage_update_stats_step = state["update_step"]
    with trainer.rollout._task_id_generator._lock:
        trainer.rollout._task_id_generator._task_cnt = state["task_counter"]


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def snapshot(root, hf, output, state):
    """Pin immutable saved generations with hardlinks before training continues."""
    pointer = json.loads((root / "LATEST").read_text())
    generation = pointer["generation"]
    if generation != f"generation_step{pointer['global_step']:08d}":
        raise ValueError("Invalid recovery generation")
    shutil.copytree(hf, output, copy_function=_link)
    recovery = output / "recovery"
    recovery.mkdir()
    shutil.copy2(root / "LATEST", recovery / "LATEST")
    shutil.copytree(
        root / "checkpoint_generations" / generation,
        recovery / "checkpoint_generations" / generation,
        copy_function=_link,
    )
    (recovery / "controller.pkl").write_bytes(pickle.dumps(state))


def seal(output, compatibility):
    output = Path(output)
    pointer = json.loads((output / "recovery/LATEST").read_text())
    files = {
        str(p.relative_to(output)): dict(size=p.stat().st_size, sha256=digest(p))
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        format_version=1,
        completed_steps=pointer["global_step"] + 1,
        compatibility=compatibility,
        files=files,
    )
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    validate(output, compatibility)
    return manifest


def package(root, hf, output, state, compatibility):
    snapshot(root, hf, output, state)
    return seal(output, compatibility)


def validate(bundle, compatibility=None):
    bundle = Path(bundle)
    manifest = json.loads((bundle / MANIFEST).read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported full checkpoint format")
    if compatibility is not None and not compatible(
        manifest["compatibility"], compatibility, manifest["completed_steps"]
    ):
        raise ValueError(
            "Checkpoint training configuration/task data differs; refusing partial resume"
        )
    files = manifest["files"]
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    if (bundle / "COMMITTED").exists():
        if (bundle / "COMMITTED").read_text() != "complete\n":
            raise ValueError("Invalid remote publication marker")
        actual.discard("COMMITTED")
    if actual != set(files) | {MANIFEST}:
        raise ValueError("Checkpoint contains untracked or missing files")
    for name, info in files.items():
        path = bundle / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("Unsafe checkpoint path")
        if (
            not path.is_file()
            or path.stat().st_size != info["size"]
            or digest(path) != info["sha256"]
        ):
            raise ValueError(f"Checkpoint integrity failed: {name}")
    recovery = bundle / "recovery"
    pointer = json.loads((recovery / "LATEST").read_text())
    if pointer["generation"] != f"generation_step{pointer['global_step']:08d}":
        raise ValueError("Invalid generation pointer")
    if pointer["global_step"] + 1 != manifest["completed_steps"]:
        raise ValueError("Checkpoint step mismatch")
    generation = recovery / "checkpoint_generations" / pointer["generation"]
    required = [recovery / "LATEST", recovery / "controller.pkl"]
    required += [
        generation / "manifest" / name
        for name in (
            "step_info.json",
            "saver_info.json",
            "evaluator_info.json",
            "stats_logger_info.json",
            "checkpoint_info.json",
            "dataloader_info.pkl",
        )
    ]
    step = json.loads((generation / "manifest/step_info.json").read_text())
    if step["global_step"] != pointer["global_step"]:
        raise ValueError("Recovery manifest step mismatch")
    for engine in pointer["engines"]:
        if engine not in {"default", "critic"}:
            raise ValueError("Unexpected checkpoint engine")
        payload = generation / "payloads" / engine
        marker = payload / "engine_state.json"
        metadata = json.loads(marker.read_text())
        required += [marker, payload / ".metadata"]
        required += [
            payload / f"engine_state_rank_{rank}.pt"
            for rank in range(metadata["world_size"])
        ]
        if not list(payload.glob("*.distcp")):
            raise ValueError("Missing distributed checkpoint shards")
    if "default" not in pointer["engines"]:
        raise ValueError("Missing actor checkpoint")
    for path in required:
        if str(path.relative_to(bundle)) not in files:
            raise ValueError(
                f"Required state is not covered by integrity manifest: {path.name}"
            )
    return manifest


def install(bundle, root, compatibility):
    validate(bundle, compatibility)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=root.parent) as tmp:
        staged = Path(tmp) / "recovery"
        shutil.copytree(Path(bundle) / "recovery", staged)
        pointer = json.loads((staged / "LATEST").read_text())
        generation = pointer["generation"]
        target = root / "checkpoint_generations" / generation
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staged / "checkpoint_generations" / generation, target)
        os.replace(staged / "controller.pkl", root / "controller.pkl")
        os.replace(staged / "LATEST", root / "LATEST")


def compatible(saved, current, completed_steps):
    return saved == current
