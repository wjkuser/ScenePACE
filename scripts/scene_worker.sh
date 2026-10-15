#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${SCENEPACE_CONFIG:-configs/runtime.json}"
PYTHON="${SCENEPACE_SCENE_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"]["scene"])' "$CONFIG")}"
exec "$PYTHON" -m scene_pace.worker --config "$CONFIG" "$@"
