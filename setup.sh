#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
STAGE="${1:-generation}"
case "$STAGE" in generation|all|sources|scene|trellis|vllm|isaac|rl|models|check) ;; *) echo 'Usage: ./setup.sh [generation|all|sources|scene|trellis|vllm|isaac|rl|models|check]'; exit 2;; esac
RUNTIME="${SCENEPACE_RUNTIME:-$ROOT/.runtime}"
mkdir -p "$RUNTIME/deps" "$RUNTIME/envs"
source_git() {
 if [ -n "${GITHUB_MIRROR:-}" ]; then
  git -c "url.${GITHUB_MIRROR%/}/https://github.com/.insteadOf=https://github.com/" "$@"
 else
  git "$@"
 fi
}
clone() {
 local name="$1" repo="$2" ref="${3:-}"
 if [ ! -d "$RUNTIME/deps/$name/.git" ]; then source_git clone --recursive "https://github.com/$repo.git" "$RUNTIME/deps/$name"; fi
 if [ -n "$ref" ]; then git -C "$RUNTIME/deps/$name" checkout "$ref"; fi
 source_git -C "$RUNTIME/deps/$name" submodule update --init --recursive
}
source_deps() {
 clone TRELLIS microsoft/TRELLIS 442aa1e1afb9014e80681d3bf604e8d728a86ee7
 clone CLIP openai/CLIP d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
 clone taming-transformers CompVis/taming-transformers 3ba01b241669f5ade541ce990f7650a3b8f65318
 clone nvdiffrast NVlabs/nvdiffrast v0.4.0
 clone pytorch3d facebookresearch/pytorch3d 2d4d345b6fd2720580bff5f63dcbd3b230b43996
 clone utils3d EasternJournalist/utils3d 9a4eb15e4021b67b12c460c7057d642626897ec8
 clone mip-splatting autonomousvision/mip-splatting dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
}
areal_sources() {
 source_git submodule update --init --recursive
 local patch="$ROOT/patches/areal/runtime.patch"
 if git -C third_party/AReaL apply --check "$patch" 2>/dev/null; then git -C third_party/AReaL apply "$patch";
 elif ! git -C third_party/AReaL apply --reverse --check "$patch" 2>/dev/null; then echo 'AReaL checkout does not match the supplied patch.' >&2; exit 1; fi
}
make_env() {
 local name="$1" version="$2"
 command -v conda >/dev/null || { echo 'Install Miniforge/Conda and initialize it in this shell first.' >&2; exit 1; }
 if [ ! -x "$RUNTIME/envs/$name/bin/python" ]; then conda create -y -p "$RUNTIME/envs/$name" "python=$version" pip; fi
 PY="$RUNTIME/envs/$name/bin/python"
 "$PY" -m pip install --upgrade pip wheel 'setuptools<81'
 "$PY" -m pip install -r "requirements/$name.txt"
}
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = sources ]; then source_deps; fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = scene ]; then
 make_env scene 3.10
 "$PY" -m pip install --no-build-isolation "$RUNTIME/deps/CLIP" "$RUNTIME/deps/taming-transformers" "$RUNTIME/deps/nvdiffrast" "$RUNTIME/deps/pytorch3d"
 "$PY" -c 'import site,sys,pathlib; pathlib.Path(site.getsitepackages()[0],"taming_source.pth").write_text(sys.argv[1]+"\n")' "$RUNTIME/deps/taming-transformers"
 "$PY" -m pip install --no-deps -e .
fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = trellis ]; then
 make_env trellis 3.10
 "$PY" -m pip install --no-build-isolation "$RUNTIME/deps/utils3d" "$RUNTIME/deps/nvdiffrast" "$RUNTIME/deps/mip-splatting/submodules/diff-gaussian-rasterization"
 "$PY" -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = vllm ]; then make_env vllm 3.12; fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = isaac ]; then make_env isaac 3.10; fi
if [ "$STAGE" = all ] || [ "$STAGE" = rl ]; then
 areal_sources
 make_env rl 3.12
 "$PY" scripts/download_areal_dependencies.py > "$RUNTIME/areal-public-requirements.txt"
 "$PY" -m pip install -r "$RUNTIME/areal-public-requirements.txt"
 "$PY" -m pip install flash-attn==2.8.3 --no-build-isolation
 "$PY" -m pip install -e third_party/AReaL --no-deps
 "$PY" -m pip install --no-deps -e .
fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = models ]; then
 "$RUNTIME/envs/scene/bin/python" scripts/download_models.py
fi
if [ "$STAGE" = generation ] || [ "$STAGE" = all ] || [ "$STAGE" = check ]; then
 CHECK_ARGS=()
 if [ "$STAGE" = all ]; then CHECK_ARGS+=(--training); fi
 "$RUNTIME/envs/scene/bin/python" scripts/check_install.py "${CHECK_ARGS[@]}"
fi
