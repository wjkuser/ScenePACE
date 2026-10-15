"""Fresh prompt visits; complete same-prompt GRPO groups in balanced updates."""
import json
import random
from collections import Counter


def prompt_batches(tasks, tasks_per_step, rng):
    if tasks_per_step == 1:
        return [[task] for task in tasks]
    if tasks_per_step != 4:
        raise ValueError("Supported prompt batch sizes are one and four")
    modes = Counter(t['preferences']['object_mode'] for t in tasks)
    if len(tasks) % 12 or modes['open'] != 2 * modes['strict']:
        raise ValueError("Four-prompt batches require a multiple of 12 tasks and 1:2 strict/open")
    buckets = {mode: [t for t in tasks if t['preferences']['object_mode'] == mode]
               for mode in ('strict', 'open')}
    for bucket in buckets.values():
        rng.shuffle(bucket)
    batches = []
    for index in range(len(tasks) // 4):
        strict_count = (1, 1, 2)[index % 3]
        batch = []
        for mode, count in (('strict', strict_count), ('open', 4 - strict_count)):
            for _ in range(count):
                used = Counter(t['scene_type'] for t in batch)
                # Prefer another room type; if the tail lacks one, still consume
                # every task exactly once instead of dropping or oversampling it.
                task = min(buckets[mode], key=lambda t: used[t['scene_type']])
                buckets[mode].remove(task)
                batch.append(task)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def training_rows(tasks, *, visits, group_size, seed, tasks_per_step=1):
    if visits < 1 or group_size < 2:
        raise ValueError("Training requires positive visits and group_size >= 2")
    tasks = [task for task in tasks if task['split'] == 'train']
    rng = random.Random(seed)
    rows = []
    for visit in range(visits):
        order = list(tasks)
        if visit:
            rng.shuffle(order)
        for batch in prompt_batches(order, tasks_per_step, rng):
            for task in batch:
                for _ in range(group_size):
                    rows.append({
                        'room_desc': task['room_desc'], 'task_id': task['task_id'],
                        'condition_id': task.get('condition_id', 'default'),
                        'task_json': json.dumps(task), 'visit': visit + 1,
                        'seed': seed + len(rows),
                    })
    return rows
