"""Complete local recovery snapshots and optional asynchronous publication."""

import json, os, shutil, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from areal.utils.saver import Saver
from sage_rl.grpo_trainer import SageGRPOTrainer
from scene_pace import checkpoints as full


class SceneTrainer(SageGRPOTrainer):
    def __init__(self, config, *args, **kwargs):
        self._publisher = ThreadPoolExecutor(max_workers=1)
        self._pending = []
        bundle = os.environ.get("SCENEPACE_RESUME_BUNDLE")
        root = full.recovery_root(config)
        if not bundle and config.recover.mode == "auto":
            completed = sorted(
                root.parent.glob("complete-step-*"),
                key=lambda p: int(p.name.rsplit("-", 1)[1]),
                reverse=True,
            )
            if completed:
                bundle = str(completed[0])
            elif (root / "LATEST").exists():
                raise RuntimeError(
                    "No sealed recovery bundle; refusing an incomplete controller/DCP checkpoint"
                )
        if bundle:
            full.install(Path(bundle), root, full.contract(config))
            if os.environ.get("SCENEPACE_CHECKPOINT_URI"):
                self._pending.append(self._publisher.submit(self._upload, Path(bundle)))
        super().__init__(config, *args, **kwargs)
        root = full.recovery_root(config)
        if self.recover_info is not None:
            if not (root / "controller.pkl").is_file():
                raise RuntimeError(
                    "Recovery requires the matching controller RNG state"
                )
            full.restore_controller(self, root)
        for owner, settings in [
            (self.saver, config.saver),
            (self.recover_handler, config.recover),
        ]:
            owner.freq_ctl.step_ctl.frequency_steps = settings.freq_steps

    def _hf_path(self, epoch, epoch_step, global_step):
        return (
            Path(
                Saver.get_model_save_root(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.config.saver.fileroot,
                    "default",
                )
            )
            / f"epoch{epoch}epochstep{epoch_step}globalstep{global_step}"
        )

    def _save_recover_checkpoint(self, epoch, epoch_step, global_step):
        super()._save_recover_checkpoint(epoch, epoch_step, global_step)
        root = full.recovery_root(self.config)
        pointer = root / "LATEST"
        if (
            not pointer.exists()
            or json.loads(pointer.read_text())["global_step"] != global_step
        ):
            if global_step + 1 >= self.config.total_train_steps:
                raise RuntimeError("Final complete recovery snapshot is missing")
            return
        self._check()
        snapshot = root.parent / f"complete-step-{global_step + 1}"
        tmp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root.parent)) / "bundle"
        state = full.controller_state(self)
        full.snapshot(root, self._hf_path(epoch, epoch_step, global_step), tmp, state)
        # Match the controller RNG/counters to the atomically published DCP pointer.
        shutil.copy2(tmp / "recovery/controller.pkl", root / "controller.pkl")
        self._pending.append(
            self._publisher.submit(
                self._publish, tmp, snapshot, full.contract(self.config)
            )
        )
        if global_step + 1 >= self.config.total_train_steps:
            self._check(wait=True)

    @staticmethod
    def _publish(tmp, snapshot, contract):
        full.seal(tmp, contract)
        if snapshot.exists():
            raise FileExistsError(snapshot)
        tmp.rename(snapshot)
        tmp.parent.rmdir()
        SceneTrainer._upload(snapshot)

    @staticmethod
    def _upload(snapshot):
        destination = os.environ.get("SCENEPACE_CHECKPOINT_URI")
        if destination:
            import fsspec

            fs, remote = fsspec.core.url_to_fs(
                destination.rstrip("/") + "/" + snapshot.name
            )
            if fs.exists(remote + "/COMMITTED"):
                with fs.open(remote + "/" + full.MANIFEST, "r") as handle:
                    published = json.load(handle)
                if published != json.loads((snapshot / full.MANIFEST).read_text()):
                    raise RuntimeError("Remote checkpoint name already contains a different run")
                return
            for p in snapshot.rglob("*"):
                if p.is_file() and p.name != "COMMITTED":
                    target = remote + "/" + str(p.relative_to(snapshot))
                    fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
                    with p.open("rb") as src, fs.open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            with fs.open(remote + "/COMMITTED", "w") as f:
                f.write("complete\n")

    def _check(self, wait=False):
        for f in self._pending:
            if wait or f.done():
                f.result()

    def _save_training_state(self, **kwargs):
        self._check()
        final = kwargs["global_step"] + 1 >= self.config.total_train_steps
        if final:
            for owner in (self.saver, self.recover_handler):
                owner.freq_ctl.step_ctl.frequency_steps = 1
        return super()._save_training_state(**kwargs)

    def close(self):
        try:
            self._check(wait=True)
        finally:
            self._publisher.shutdown(wait=True)
            super().close()
