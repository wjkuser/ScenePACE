"""Launch the configured AReaL GRPO run."""

import argparse, json, os, sys
from scene_pace.runtime import ROOT, load, path, environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-config", default="configs/runtime.json")
    p.add_argument("--config")
    p.add_argument("--check", action="store_true")
    args, extra = p.parse_known_args()
    c = load(args.runtime_config)
    os.environ.update(environment(c))
    sys.path.insert(0, str(ROOT / "third_party/AReaL"))
    os.environ["SAGE_RL_PROMPT_FILE"] = str(path(c["training"]["tasks"]))
    os.environ["SAGE_RL_RECOVER_MODE"] = c["training"]["resume"]
    config = str(path(args.config or c["training"]["config"]))
    extra = ["recover.mode=" + c["training"]["resume"], *extra]
    if args.check:
        from areal.api.cli_args import load_expr_config
        from sage_rl.configs import SageRLConfig

        cfg, _ = load_expr_config(["--config", config, *extra], SageRLConfig)
        from sage_rl.task_schema import load_tasks
        from sage_rl.schedule import training_rows

        tasks = load_tasks(os.environ["SAGE_RL_PROMPT_FILE"])
        rows = training_rows(
            tasks,
            visits=cfg.task_visits,
            group_size=cfg.gconfig.n_samples,
            seed=cfg.seed,
            tasks_per_step=cfg.tasks_per_step,
        )
        assert (
            cfg.train_dataset.batch_size == cfg.tasks_per_step * cfg.gconfig.n_samples
        )
        assert cfg.saver.freq_steps == cfg.recover.freq_steps
        print(
            json.dumps(
                {
                    "train_tasks": len(tasks),
                    "rollouts": len(rows),
                    "planned_updates": len(rows) // cfg.train_dataset.batch_size,
                    "total_train_steps": cfg.total_train_steps,
                    "checkpoint_interval": cfg.saver.freq_steps,
                }
            )
        )
        return
    if not c["training"].get("proxy_host"):
        raise ValueError(
            "Set training.proxy_host to the routable rollout/proxy host address"
        )
    if not os.environ.get("SCENEPACE_SCENE_KEY"):
        raise ValueError("Export SCENEPACE_SCENE_KEY, matching the scene worker")
    from omegaconf import OmegaConf

    runtime = OmegaConf.load(config)
    forward = {
        "SCENEPACE_SCENE_ENDPOINTS": json.dumps(c["training"]["scene_endpoints"]),
        "SCENEPACE_PROXY_HOST": c["training"]["proxy_host"],
        "SCENEPACE_SCENE_KEY": os.environ["SCENEPACE_SCENE_KEY"],
        "SCENEPACE_EPISODE_TIMEOUT": str(c["scene"]["timeout_seconds"]),
        "SCENEPACE_ACTOR_MODEL": os.environ["SCENEPACE_ACTOR_MODEL"],
        "SCENEPACE_OUTPUT": os.environ["SCENEPACE_OUTPUT"],
        "SAGE_EFFICIENCY_ROOT": str(ROOT),
        "PYTHONPATH": os.environ["PYTHONPATH"],
    }
    for spec in runtime.rollout.scheduling_spec:
        spec.env_vars.update(forward)
    for spec in runtime.actor.scheduling_spec:
        spec.env_vars.update(
            {
                k: v
                for k, v in forward.items()
                if k
                in [
                    "SCENEPACE_ACTOR_MODEL",
                    "SCENEPACE_OUTPUT",
                    "SAGE_EFFICIENCY_ROOT",
                    "PYTHONPATH",
                ]
            }
        )
    import tempfile

    private = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    os.chmod(private.name, 0o600)
    private.write(OmegaConf.to_yaml(runtime, resolve=True))
    private.close()
    config = private.name
    try:
        from sage_rl.train import main as train
        from scene_pace.trainer import SceneTrainer

        import ray

        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"), ignore_reinit_error=True
        )
        train(["--config", config, *extra], trainer_type=SceneTrainer)
    finally:
        os.unlink(private.name)


if __name__ == "__main__":
    main()
