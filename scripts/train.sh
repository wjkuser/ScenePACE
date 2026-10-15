#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${SCENEPACE_CONFIG:-configs/runtime.json}"
PYTHON="${SCENEPACE_RL_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"]["rl"])' "$CONFIG")}"
exec "$PYTHON" -m scene_pace.train --runtime-config "$CONFIG" "$@"
