"""Persist diagnostic artifacts without mixing them into timing log records."""
from __future__ import annotations
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from constants import RESULTS_DIR

def _artifact_root(explicit_run_dir: Optional[Path]=None) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir
    configured = os.environ.get('SAGE_RUN_DIR', '').strip()
    if configured:
        return Path(configured)
    return Path(RESULTS_DIR) / 'run_artifacts'

def persist_raw_vlm_response(stage: str, response_text: str, *, metadata: Optional[Dict[str, Any]]=None, run_dir: Optional[Path]=None) -> Path:
    """Atomically persist a complete pre-parse VLM response for diagnosis."""
    safe_stage = re.sub('[^A-Za-z0-9_.-]+', '_', stage).strip('_.') or 'vlm'
    request_id = uuid.uuid4().hex
    payload = {'stage': stage, 'request_id': request_id, 'captured_at_ns': time.time_ns(), 'response_text': response_text, 'metadata': dict(metadata or {})}
    artifact_dir = _artifact_root(run_dir) / 'raw_vlm_responses'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / f'{safe_stage}_{request_id}.json'
    temporary = artifact_dir / f'.{target.name}.tmp'
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, target)
    return target
