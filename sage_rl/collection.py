"""Collect an explicit batch with bounded, same-slot infrastructure retries."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

from sage_rl.rollout_batch import layout_rollout_rejection_reason
from sage_rl.task_schema import task_key


def collect_task_batch(
    rollout,
    dispatcher,
    required,
    group_size,
    *,
    timeout=14400,
    max_retries=2,
    release_capacity,
    audit_path=None,
):
    inputs = [next(rollout.data_generator) for _ in range(required)]
    counts = Counter(task_key(item.data) for item in inputs)
    if any(count != group_size for count in counts.values()):
        raise ValueError(
            f"Batch must contain exactly {group_size} copies per task: {dict(counts)}"
        )
    # The full task specification must agree even when an ID was reused wrongly.
    specifications = {}
    for item in inputs:
        key = task_key(item.data)
        spec = item.data["task_json"]
        if key in specifications and specifications[key] != spec:
            raise ValueError(f"Conflicting specifications for {key}")
        specifications[key] = spec
    if dispatcher.is_paused():
        dispatcher.resume()
    for item in inputs:
        dispatcher.submit_task_input(item)
    deadline = time.monotonic() + timeout
    results = []
    for slot, initial in enumerate(inputs):
        item = initial
        for attempt in range(max_retries + 1):
            result = dispatcher.wait_for_task(
                item.task_id, timeout=max(0.1, deadline - time.monotonic())
            )
            trajectory = result.trajectory if result is not None else None
            reason = (
                layout_rollout_rejection_reason(trajectory)
                if trajectory is not None
                else "no exported trajectory"
            )
            if audit_path is not None:
                path = Path(audit_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "task_key": task_key(initial.data),
                    "seed": initial.data.get("seed"),
                    "slot": slot,
                    "visit": initial.data.get("visit"),
                    "attempt": attempt,
                    "task_id": item.task_id,
                    "excluded_reason": reason,
                    "outcome": trajectory.get("sage_outcome") if trajectory else None,
                    "episode": trajectory.get("sage_episode") if trajectory else None,
                    "layout_id": trajectory.get("sage_layout_id")
                    if trajectory
                    else None,
                }
                with path.open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if trajectory is not None and not trajectory.get("sage_outcome"):
                raise ValueError("Exported trajectory lost its outcome classification")
            if not reason:
                if trajectory.get("sage_prompt_key") != task_key(initial.data):
                    raise ValueError(
                        "Exported trajectory lost or changed its task identity"
                    )
                results.append(trajectory)
                break
            if trajectory is None or trajectory.get("sage_outcome", {}).get("status") != "environment_failure":
                raise RuntimeError(f"Non-retryable rollout failure: {reason}")
            if trajectory is not None:
                release_capacity(dispatcher.staleness_manager, 1)
            print(
                f"[SageCollection] slot={slot} attempt={attempt} excluded={reason}",
                flush=True,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Task {task_key(initial.data)} exhausted environment retries: {reason}"
                )
            item = replace(initial, task_id=rollout._task_id_generator.next())
            dispatcher.submit_task_input(item)
    return results
