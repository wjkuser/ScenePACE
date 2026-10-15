"""Sage-specific metrics alongside the framework's optimizer telemetry."""
from collections import Counter
from pathlib import Path
import os
import statistics


def episode_metrics(trajectories, stats):
    metrics = {
        'sage/reward_mean': stats['terminal_mean'],
        'sage/reward_std': stats['terminal_std_sample'],
        'sage/equal_reward_group': float(stats['terminal_std_sample'] <= 1e-5),
        'sage/tool_calls': statistics.mean(t['sage_tool_call_count'] for t in trajectories),
        'sage/object_count': statistics.mean(t['sage_object_count'] for t in trajectories),
    }
    sections = {}
    failures = Counter()
    for t in trajectories:
        episode = t['sage_episode']
        verification = episode.get('verification') or {}
        for key, value in verification.get('section_scores', {}).items():
            sections.setdefault(key, []).append(value)
        failures.update(c['id'] for c in verification.get('checks', []) if c['status'] == 'fail')
    metrics.update({f'sage/score/{k}': statistics.mean(v) for k, v in sections.items()})
    metrics.update({f'sage/check_failure/{k}': n / len(trajectories) for k, n in failures.items()})
    metrics['sage/tool_budget_cutoff'] = statistics.mean(
        bool(t['sage_episode'].get('tool_call_limit_reached')) for t in trajectories)
    metrics['sage/pipeline_seconds'] = statistics.mean(
        t['sage_episode']['elapsed_seconds'] for t in trajectories)
    task_id = trajectories[0]['sage_episode']['task_id']
    metrics[f'sage/task/{task_id}/reward'] = stats['terminal_mean']
    return metrics


def scene_preview(layout_id):
    """One fixed five-view preview; never re-render for monitoring."""
    from PIL import Image
    root = Path(os.environ['SAGE_RESULTS_DIR']) / layout_id / 'rl_final_critic'
    paths = sorted(root.glob('*.png'))[:5]
    if not paths:
        return None
    canvas = Image.new('RGB', (320 * len(paths), 320), 'white')
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            frame = source.convert('RGB')
            frame.thumbnail((320, 320))
            canvas.paste(frame, (index * 320, 0))
    return canvas


def log_training_group(trajectories, stats):
    import swanlab
    if os.environ.get('SAGE_SWANLAB_MODE', 'disabled') == 'disabled':
        return
    metrics = episode_metrics(trajectories, stats)
    preview = scene_preview(trajectories[0]['sage_layout_id'])
    if preview is not None:
        metrics['sage/first_rollout_views'] = swanlab.Image(preview)
    swanlab.log(metrics, step=stats['step'] - 1)


def log_evaluation_episode(result, step):
    import swanlab
    verification = result.get('verification') or {}
    metrics = {f'evaluation/score/{k}': v for k, v in verification.get('section_scores', {}).items()}
    metrics.update({
        'evaluation/reward': result.get('agent_reward', verification.get('reward', 0.)),
        'evaluation/pipeline_seconds': result['elapsed_seconds'],
        'evaluation/tool_calls': result.get('tool_call_count', 0),
        'evaluation/success': float(result['outcome']['status'] == 'success'),
    })
    metrics[f"evaluation/task/{result['task_id']}/reward"] = metrics['evaluation/reward']
    preview = scene_preview(result.get('layout_id', ''))
    if preview is not None:
        metrics['evaluation/views'] = swanlab.Image(preview)
    swanlab.log(metrics, step=step)


def log_training_batch(groups, stats, step):
    """Log once per optimizer update, retaining each prompt's reward."""
    if os.environ.get('SAGE_SWANLAB_MODE', 'disabled') == 'disabled':
        return
    import swanlab
    grouped = [episode_metrics(group, stat) for group, stat in zip(groups, stats)]
    keys = set().union(*(m.keys() for m in grouped))
    metrics = {key: statistics.mean(m[key] for m in grouped if key in m) for key in keys}
    preview = scene_preview(groups[0][0]['sage_layout_id'])
    if preview is not None:
        metrics['sage/first_rollout_views'] = swanlab.Image(preview)
    swanlab.log(metrics, step=step)
