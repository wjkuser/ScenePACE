# ScenePACE

**Policy Adaptation and Concurrent Execution for Agentic 3D Scene Generation**

ScenePACE builds furnished, physically validated 3D rooms from natural-language requests. Building on SAGE, it improves two parts of the scene-generation process:

- **Concurrent execution:** validate placement candidates in isolated copies of a shared scene, combine compatible candidates, and verify the resulting scene. This reduces repeated scene construction and serial simulation.
- **Policy adaptation:** train the language actor with trajectory-level GRPO, using task fulfillment and furnishing quality assessed from the final scene. The generation tools and critics remain fixed.

This repository includes scene generation, the training integration, task datasets, and evaluation code. Assets are generated with TRELLIS and materials with MatFuse; scene simulation and final room rendering use Isaac Sim.

## Requirements

Use Linux x86-64, Conda/Miniforge, Git LFS, NVIDIA GPUs, a compatible NVIDIA driver, and a CUDA development toolkit for native extensions. The reference configuration assigns **eight GPUs with approximately 48 GB VRAM each** to generation services:

| Component | Default GPU IDs |
|---|---|
| Visual model | `0,1,2,3` |
| Language actor | `4` |
| Scene and material generation | `5` |
| TRELLIS | `6` |
| Isaac Sim | `7` |

These are configurable assignments, not a measured minimum hardware requirement. Training additionally uses a Ray cluster; see [Policy training](#policy-training). Allow at least 200 GB of storage for environments, models, builds, and initial outputs; training checkpoints require additional space.

## Installation

```bash
git clone https://github.com/wjkuser/ScenePACE.git
cd ScenePACE

# Ubuntu 22.04 or newer:
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git git-lfs libgl1 libegl1 libvulkan1 vulkan-tools libglib2.0-0 libxrender1 libxi6 libsm6

bash setup.sh
```

Initialize Conda in the shell and configure `CUDA_HOME` before running the installer. Read and accept the applicable NVIDIA Isaac Sim license before starting Isaac; the launcher sets its EULA acceptance variable.

By default, the installer creates four environments under `.runtime/envs/`: `scene` (generation controller and scene tools), `trellis`, `isaac`, and `vllm`. The `rl` environment and AReaL are optional and are installed only for training. Each environment uses its matching `requirements/<environment>.txt`; their Python/PyTorch requirements differ. Run from this checkout and keep the environments separate; a standalone `pip install .` does not install the simulation stack, source dependencies, or models.

`bash setup.sh generation` is equivalent to the default installation. Use `bash setup.sh all` to include training, or add it later with `bash setup.sh rl`. Individual stages are `sources`, `scene`, `trellis`, `isaac`, `vllm`, `rl`, `models`, and `check`; run `sources` before installing generation environments individually. Dependencies and model revisions are specified in `setup.sh`, `requirements/`, and `configs/models.json`.

## Scene generation

### Configure models and services

Copy the default configuration to an ignored local file and edit GPU assignments, interpreter paths, model paths, and service endpoints:

```bash
cp configs/runtime.json configs/local.json
export SCENEPACE_CONFIG=configs/local.json
```

Relative paths resolve against the repository root. The defaults use **Qwen3-14B** as the actor and **Qwen3.6-35B-A3B** as the visual model. The download manifest contains public pretrained models; it does **not** provide the RL-adapted ScenePACE actor. To use an adapted actor, set `actor.model` to your exported Hugging Face checkpoint directory (relative to `paths.models`, or absolute).

Start each service in a separate terminal, with the same configuration:

```bash
bash scripts/serve.sh actor
bash scripts/serve.sh vlm
bash scripts/serve.sh trellis
bash scripts/serve.sh isaac
```

Wait for model loading and `ISAAC_SERVICE_READY`. Service readiness is checked automatically before generation.

### Generate a room

```bash
bash scripts/generate.sh --task-id sage_test30_bedroom_02_evening_tea --seed 11
```

Requests default to `data/test.jsonl`. Use `--tasks path/to/tasks.jsonl` for another task file, following the supplied schema: a request includes `room_desc`, object requirements, support relations, furniture preferences, and evaluation settings. The shared actor instruction is in `configs/generation_prompt.txt`.

Each run writes `pipeline.log`, `task.json`, `result.json`, conversation logs, and a `room/` directory containing the room layout, assets, materials, and critic outputs. Choose a fresh output directory with `SCENEPACE_OUTPUT=outputs/my-run`. Shared assets also remain under the configured `paths.output` so the scene process and Isaac service can access them.

### Use an API actor

Set `actor.mode` to `api`, configure `actor.endpoint`, `actor.served_name`, and `actor.wire_api` (`chat` or `responses`), and export the credential under the environment-variable name in `actor.api_key_env`. Skip `serve.sh actor`; the visual, TRELLIS, and Isaac services are still required. Keep credentials in the environment and local configuration outside Git.

## Policy training

Install the optional training environment with `bash setup.sh rl` (or use `bash setup.sh all` for a complete installation). This initializes the pinned AReaL submodule and applies `patches/areal/runtime.patch`. Training uses AReaL with FSDP and vLLM rollouts. `configs/training.yaml` controls GRPO and `configs/runtime.json` controls generation services and task paths. The supplied profile uses Qwen3-14B, learning rate `1e-6`, four requests with four sampled trajectories each per update, and 90 updates.

The training profile allocates an eight-GPU FSDP actor and a separate one-GPU rollout engine across **two eight-GPU Ray nodes**, in addition to generation-service resources. Nodes need a shared checkout, environment paths, model paths, and output filesystem. Do not assign the same GPU to Ray and an independently launched generation service.

1. Start the visual, TRELLIS, and Isaac services on the generation host. Export a private `SCENEPACE_SCENE_KEY` there and on the training controller, then run `bash scripts/scene_worker.sh --host 0.0.0.0` on that host.
2. Set `training.scene_endpoints` to the reachable worker URL and `training.proxy_host` to the reachable rollout proxy host in the runtime configuration.
3. Start Ray and launch training from the checkout:

```bash
# Training head:
.runtime/envs/rl/bin/ray start --head --port=6379 --num-gpus=8
# Second training node (replace TRAINING_HEAD with the head address):
.runtime/envs/rl/bin/ray start --address=TRAINING_HEAD:6379 --num-gpus=8
# Controller:
export RAY_ADDRESS=TRAINING_HEAD:6379
bash scripts/train.sh --check
bash scripts/train.sh
```

A scene worker runs one episode at a time. The default rollout concurrency is one; scale it by adding workers with independent Isaac instances, GPU assignments, ports, and output roots. Generation services and training workers should run on a trusted network.

Use a separate checkpoint destination per run. Complete recovery bundles include model and optimizer state, RNG state, and training progress; resume selects the latest complete local bundle. Set `SCENEPACE_RESUME_BUNDLE` to select a particular complete bundle. Exported model weights alone are suitable for inference, not exact training recovery. Optional checkpoint publication uses `SCENEPACE_CHECKPOINT_URI` and requires the corresponding fsspec backend. Only load trusted training checkpoints.

## Tasks and evaluation

`data/train.jsonl` contains **120 training requests**; `data/test.jsonl` contains **30 held-out test requests**. Task validation is implemented in `sage_rl/task_schema.py`.

Generation evaluates each final scene and records component scores and check evidence in `result.json`. Task success requires the requested object counts, requested support relations, and entrance access. Evaluation also reports visual composition, furnishing richness, and reward. Auxiliary checks remain available for diagnostics; `verification.strict_success` includes additional checks and differs from task success.

The task files specify evaluation options. Training retains its tool-limit reward penalty; test evaluation disables that penalty. Visual scores and their transformed reward contribution are separate values. See `sage_rl/evaluation/scoring.py`, `sage_rl/evaluation/visual.py`, and `sage_rl/reward.py` for the definitions.

## Code structure

| Path | Purpose |
|---|---|
| `scene_pace/` | Generation, service, training, and checkpoint entry points |
| `client/` | Tool-using language actor client |
| `server/objects/` | Asset generation and placement planning |
| `server/isaacsim/` | Simulation, concurrent candidate validation, and rendering |
| `sage_rl/agent/` | Scene-editing policy and trajectory collection |
| `sage_rl/evaluation/` | Task verification and scene-quality scoring |
| `configs/`, `data/` | Runtime/training configuration, prompts, and task datasets |
| `requirements/`, `scripts/` | Environment specifications and launch/install utilities |
| `third_party/AReaL`, `patches/areal/` | Pinned training framework and integration patch |
| `matfuse-sd/` | Bundled MatFuse material-generation code |

The internal `sage_rl` package name is retained for import and checkpoint compatibility. Models, environments, generated scenes, logs, and local configuration are excluded from Git.

## Acknowledgments and license

ScenePACE builds on [SAGE](https://github.com/NVlabs/sage), [AReaL](https://github.com/areal-project/AReaL), [TRELLIS](https://github.com/microsoft/TRELLIS), and [MatFuse](https://github.com/giuvecchio/matfuse).

Project code is distributed under [Apache-2.0](LICENSE), with upstream notices retained. See [NOTICE](NOTICE) and the bundled third-party licenses. Downloaded models, external dependencies, and Isaac Sim remain subject to their respective licenses.
