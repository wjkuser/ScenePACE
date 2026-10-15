#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${SCENEPACE_CONFIG:-configs/runtime.json}"
PYTHON="${SCENEPACE_SCENE_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"]["scene"])' "$CONFIG")}"
OUTPUT="${SCENEPACE_OUTPUT:-outputs/scene-$(date -u +%Y%m%d-%H%M%S)}"
"$PYTHON" scripts/check_services.py --config "$CONFIG"
mkdir -p "$OUTPUT"
"$PYTHON" -m scene_pace.generate --config "$CONFIG" --output "$OUTPUT" "$@" 2>&1 | tee "$OUTPUT/pipeline.log"
