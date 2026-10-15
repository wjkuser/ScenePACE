"""Rollout batch helpers for Sage GRPO actor DP safety."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def read_engine_int_attr(engine: Any, attr: str, default: int) -> int:
    """Read an int engine attribute; RPC/property wrappers may return a property object."""
    if engine is None:
        return default
    try:
        value = getattr(engine, attr, default)
        if isinstance(value, property):
            value = value.fget(engine)
        return max(1, int(value))
    except (TypeError, ValueError, AttributeError):
        return default


def trajectory_seq_logprobs(traj: dict[str, Any]) -> list[float]:
    """Per-sequence logprob = sum of token logprobs over loss_mask (response tokens).

    Returns one float per packed sequence in the trajectory's batch dim.
    Returns [] when logprobs/loss_mask are unavailable (e.g. eval-only rollout).
    Handles AReaL RTensor (remote tensor refs) by localizing first.
    """
    logprobs = traj.get("logprobs")
    loss_mask = traj.get("loss_mask")
    # AReaL may store RTensor (remote ref) at dump time — localize to a plain tensor.
    if logprobs is not None and not isinstance(logprobs, torch.Tensor) and hasattr(logprobs, "local"):
        try:
            logprobs = logprobs.local()
        except Exception:
            logprobs = None
    if loss_mask is not None and not isinstance(loss_mask, torch.Tensor) and hasattr(loss_mask, "local"):
        try:
            loss_mask = loss_mask.local()
        except Exception:
            loss_mask = None
    if not isinstance(logprobs, torch.Tensor) or not isinstance(loss_mask, torch.Tensor):
        return []
    lp = logprobs.float()
    mask = loss_mask.float()
    if lp.shape != mask.shape:
        # broadcast a [1, T] mask against [B, T] logprobs if needed
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if lp.ndim == 1:
            lp = lp.unsqueeze(0)
    if lp.dim() == 0 or mask.dim() == 0 or lp.shape[-1] != mask.shape[-1]:
        return []
    per_seq = (lp * mask).sum(dim=-1)
    return [float(x) for x in per_seq.tolist()]


def dump_rollout_trajectories(
    trajectories: list[dict[str, Any]],
    dump_dir: str | Path,
    *,
    tag: str = "batch",
) -> Path:
    """Persist rollout trajectories to disk before training consumes them."""
    from areal.infra.rpc.rtensor import RTensor

    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_pt = out_dir / f"{tag}_{stamp}.pt"
    localized = [RTensor.localize(traj) for traj in trajectories]
    meta = {
        "tag": tag,
        "count": len(localized),
        "saved_at": stamp,
    }
    torch.save({"trajectories": localized, "meta": meta}, out_pt)
    summary = {
        "count": len(localized),
        "saved_pt": str(out_pt),
        "episodes": [traj.get("sage_episode", {}) for traj in localized],
        "outcomes": [traj.get("sage_outcome", {}) for traj in localized],
        "layout_ids": [str(traj.get("sage_layout_id", "")) for traj in localized],
        "tool_call_counts": [
            int(traj.get("sage_tool_call_count", 0) or 0) for traj in localized
        ],
        "max_tool_calls": [
            int(traj.get("sage_max_tool_calls", 0) or 0) for traj in localized
        ],
        "completion_counts": [
            int(traj.get("sage_completion_count", 0) or 0) for traj in localized
        ],
        "object_counts": [
            int(traj.get("sage_object_count", -1)) for traj in localized
        ],
        "infrastructure_failures": [
            int(traj.get("sage_infrastructure_failure", 0) or 0)
            for traj in localized
        ],
        "failure_reasons": [
            str(traj.get("sage_failure_reason", "") or "") for traj in localized
        ],
        "rewards": [
            float(traj.get("sage_agent_reward", 0.0) or 0.0) for traj in localized
        ],
        "terminal_rewards": [
            trajectory_terminal_reward(traj) for traj in localized
        ],
        "seq_logprobs": [trajectory_seq_logprobs(traj) for traj in localized],
    }
    out_json = out_pt.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SageGRPOTrainer] dumped {len(localized)} trajectory(ies) -> {out_pt}", flush=True)
    # Log per-trajectory reward + sequence logprob so each rollout's logp is visible.
    # Use localized trajectories (plain tensors) — the originals hold RTensor refs.
    for i, traj in enumerate(localized):
        lid = str(traj.get("sage_layout_id", ""))
        r = float(traj.get("sage_agent_reward", 0.0) or 0.0)
        tool_calls = int(traj.get("sage_tool_call_count", 0) or 0)
        max_tool_calls = int(traj.get("sage_max_tool_calls", 0) or 0)
        completion_count = int(traj.get("sage_completion_count", 0) or 0)
        object_count = int(traj.get("sage_object_count", -1))
        infrastructure_failure = int(
            traj.get("sage_infrastructure_failure", 0) or 0
        )
        seq_lp = trajectory_seq_logprobs(traj)
        seq_lp_str = ", ".join(f"{v:.4f}" for v in seq_lp) if seq_lp else "n/a"
        print(
            f"[SageGRPOTrainer] traj[{i}] layout_id={lid} reward={r:.4f} "
            f"tool_calls={tool_calls}/{max_tool_calls} "
            f"completions={completion_count}/{max_tool_calls} "
            f"objects={object_count} infra_failure={infrastructure_failure} "
            f"seq_logprob=[{seq_lp_str}]",
            flush=True,
        )
    return out_pt


def _tensor_batch_size(traj: dict[str, Any]) -> int | None:
    attn = traj.get("attention_mask")
    if isinstance(attn, torch.Tensor) and attn.ndim >= 1:
        return int(attn.shape[0])
    for value in traj.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    return None


def expand_trajectory_to_sequences(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one trajectory dict with batch dim B into B single-sequence trajectories."""
    batch_size = _tensor_batch_size(traj)
    if batch_size is None or batch_size <= 1:
        return [traj]

    expanded: list[dict[str, Any]] = []
    for idx in range(batch_size):
        single: dict[str, Any] = {}
        for key, value in traj.items():
            if key == "rollout_group":
                # The adapter normalizes whole-episode rewards before splitting.
                # Each returned actor item is now one row, not the original
                # multi-row rollout described by upstream's metadata.
                continue
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
                single[key] = value[idx : idx + 1]
            elif isinstance(value, list) and len(value) == batch_size:
                single[key] = [value[idx]]
            else:
                single[key] = value
        expanded.append(single)
    return expanded


def trajectory_reward_total(traj: dict[str, Any]) -> float:
    rewards = traj.get("rewards")
    if rewards is None:
        return 0.0
    if isinstance(rewards, torch.Tensor):
        if rewards.numel() == 0:
            return 0.0
        return float(rewards.max().item())
    try:
        return float(rewards)
    except (TypeError, ValueError):
        return 0.0


def sage_agent_reward_total(traj: dict[str, Any]) -> float:
    value = traj.get("sage_agent_reward")
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.max().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def trajectory_terminal_reward(traj: dict[str, Any]) -> float:
    value = traj.get("sage_terminal_reward")
    if value is None:
        return sage_agent_reward_total(traj)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return sage_agent_reward_total(traj)
        return float(value.reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return sage_agent_reward_total(traj)


def layout_rollout_rejection_reason(traj: dict[str, Any]) -> str:
    """Return why a rollout cannot enter a variable-turn trajectory group."""
    completion_count = int(traj.get("sage_completion_count", 0) or 0)
    sequence_count = len(expand_trajectory_to_sequences(traj))
    if completion_count <= 0 or sequence_count <= 0:
        return "no assistant completions"
    outcome = traj.get("sage_outcome")
    if outcome:
        if outcome["status"] in {"success", "policy_failure"}:
            return ""
        return f"{outcome['status']}: {outcome['reason_code']}"
    return "missing explicit rollout outcome; diagnose before training"


def is_failed_layout_rollout(traj: dict[str, Any]) -> bool:
    """Detect non-trainable results; only confirmed infrastructure failures may retry."""
    return bool(layout_rollout_rejection_reason(traj))


def filter_valid_layout_rollouts(
    trajectories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    valid = [traj for traj in trajectories if not is_failed_layout_rollout(traj)]
    return valid, len(trajectories) - len(valid)


def expand_rollout_trajectories(
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for traj in trajectories:
        expanded.extend(expand_trajectory_to_sequences(traj))
    return expanded


def build_variable_turn_grpo_batch(
    trajectories: list[dict[str, Any]],
    *,
    group_size: int,
    stats_path: str | Path | None = None,
    step: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one trajectory-level GRPO group using terminal rewards.

    Terminal rewards are normalized once across the complete rollout group.
    Every assistant completion inherits its rollout's normalized terminal
    advantage.
    Rollouts may contain different numbers of completions.
    """
    if group_size <= 1:
        raise ValueError(f"group_size must be greater than one, got {group_size}")
    if len(trajectories) != group_size:
        raise RuntimeError(
            "Cannot build trajectory-level GRPO group: "
            f"got {len(trajectories)} rollouts, expected {group_size}"
        )
    from areal.infra.rpc.rtensor import RTensor

    localized = [RTensor.localize(traj) for traj in trajectories]
    verifications = [t.get("sage_episode", {}).get("verification", {}) for t in localized]
    if any(v.get("method") == "checklist" and not v.get("visual_available")
           for v in verifications):
        # Compare like with like when a completed scene has no visual score.
        # Early policy failures have no verification and retain their failure reward.
        localized = [dict(t, sage_terminal_reward=v["reward_without_visual"],
                          sage_agent_reward=v["reward_without_visual"])
                     if v.get("method") == "checklist" else t
                     for t, v in zip(localized, verifications)]
    terminal_rewards = torch.tensor(
        [trajectory_terminal_reward(traj) for traj in localized],
        dtype=torch.float64,
    )
    shaped_rewards = torch.tensor(
        [sage_agent_reward_total(traj) for traj in localized],
        dtype=torch.float64,
    )
    terminal_mean = terminal_rewards.mean()
    terminal_var_population = terminal_rewards.var(unbiased=False)
    terminal_var_sample = terminal_rewards.var(unbiased=True)
    terminal_std_sample = terminal_rewards.std(unbiased=True)
    if terminal_std_sample <= 1e-5:
        trajectory_advantages = torch.zeros_like(terminal_rewards)
    else:
        trajectory_advantages = (
            terminal_rewards - terminal_mean
        ) / (terminal_std_sample + 1e-5)

    expanded_rollouts = [
        expand_trajectory_to_sequences(traj) for traj in localized
    ]
    completion_counts = [len(turns) for turns in expanded_rollouts]
    if any(count <= 0 for count in completion_counts):
        raise RuntimeError(
            "Cannot train an empty variable-turn rollout: "
            f"completion_counts={completion_counts}"
        )

    action_token_counts = [
        sum(
            int(turn["loss_mask"].count_nonzero().item())
            for turn in turns
            if isinstance(turn.get("loss_mask"), torch.Tensor)
        )
        for turns in expanded_rollouts
    ]
    if any(count <= 0 for count in action_token_counts):
        raise RuntimeError(
            "Cannot balance variable-turn rollouts without assistant tokens: "
            f"action_token_counts={action_token_counts}"
        )
    total_action_tokens = sum(action_token_counts)
    rollout_balance = [
        total_action_tokens / (group_size * count)
        for count in action_token_counts
    ]

    expanded: list[dict[str, Any]] = []
    synthetic_rewards: list[list[float]] = []
    for rollout_index, turns in enumerate(expanded_rollouts):
        rollout_synthetic: list[float] = []
        for turn_index, turn in enumerate(turns):
            reward = rollout_balance[rollout_index] * float(trajectory_advantages[rollout_index])
            reward_tensor = turn.get("rewards")
            if not isinstance(reward_tensor, torch.Tensor):
                reward_tensor = torch.zeros(1, dtype=torch.float32)
            turn["rewards"] = torch.full_like(
                reward_tensor,
                float(reward),
                dtype=torch.float32,
            )
            expanded.append(turn)
            rollout_synthetic.append(float(reward))
        synthetic_rewards.append(rollout_synthetic)

    shaped_var_population = shaped_rewards.var(unbiased=False)
    shaped_var_sample = shaped_rewards.var(unbiased=True)
    stats: dict[str, Any] = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "group_size": group_size,
        "task_ids": [t.get("sage_episode", {}).get("task_id") for t in localized],
        "visits": [t.get("sage_episode", {}).get("visit") for t in localized],
        "policy_versions": [t.get("sage_episode", {}).get("policy_version") for t in localized],
        "terminal_rewards": terminal_rewards.tolist(),
        "terminal_mean": float(terminal_mean),
        "terminal_variance_population": float(terminal_var_population),
        "terminal_variance_sample": float(terminal_var_sample),
        "terminal_std_sample": float(terminal_std_sample),
        "trajectory_advantages": trajectory_advantages.tolist(),
        "shaped_rewards": shaped_rewards.tolist(),
        "shaped_mean": float(shaped_rewards.mean()),
        "shaped_variance_population": float(shaped_var_population),
        "shaped_variance_sample": float(shaped_var_sample),
        "completion_counts": completion_counts,
        "tool_call_counts": [
            int(traj.get("sage_tool_call_count", 0) or 0) for traj in localized
        ],
        "action_token_counts": action_token_counts,
        "rollout_balance": rollout_balance,
        "synthetic_completion_rewards": synthetic_rewards,
        "expanded_completion_count": len(expanded),
    }
    print(
        "[SageGRPOTrainer] GROUP_REWARD_STATS "
        + json.dumps(stats, ensure_ascii=True, separators=(",", ":")),
        flush=True,
    )
    if stats_path is not None:
        output_path = Path(stats_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(stats, ensure_ascii=True) + "\n")
    return expanded, stats


def group_rollout_trajectories_by_turn(
    trajectories: list[dict[str, Any]],
    *,
    group_size: int,
) -> list[dict[str, Any]]:
    """Transpose rollout-major trajectories into same-turn GRPO groups.

    A proxy rollout stores all assistant completions in one trajectory batch. For
    GRPO, completion ``t`` must be compared only with completion ``t`` from the
    other samples for the same prompt. Each returned trajectory therefore has a
    batch dimension of ``group_size`` and represents one turn-relative group.

    If rollouts have different turn counts, only their common prefix is trained.
    Later unmatched completions cannot form a complete relative-reward group;
    their terminal signal is already present in the common turns' return-to-go.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not trajectories:
        return []
    if len(trajectories) % group_size:
        raise RuntimeError(
            "Cannot build same-turn GRPO groups: "
            f"got {len(trajectories)} rollouts, group_size={group_size}"
        )

    from areal.infra.rpc.rtensor import RTensor
    from areal.utils.data import concat_padded_tensors

    # The single-controller rollout path returns remote tensor references.
    # Grouping needs real tensors for slicing and concatenation; localize in one
    # batched fetch per rollout before inspecting the completion dimension.
    trajectories = [RTensor.localize(traj) for traj in trajectories]

    grouped: list[dict[str, Any]] = []
    dropped_turns = 0
    for start in range(0, len(trajectories), group_size):
        rollout_group = trajectories[start : start + group_size]
        expanded_rollouts = [
            expand_trajectory_to_sequences(traj) for traj in rollout_group
        ]
        turn_counts = [len(turns) for turns in expanded_rollouts]
        common_turns = min(turn_counts)
        if common_turns <= 0:
            raise RuntimeError(
                "Cannot build same-turn GRPO groups from empty trajectories: "
                f"turn_counts={turn_counts}"
            )

        dropped_turns += sum(count - common_turns for count in turn_counts)
        for turn_index in range(common_turns):
            same_turn = [turns[turn_index] for turns in expanded_rollouts]
            turn_group = concat_padded_tensors(same_turn)
            turn_group["sage_turn_index"] = turn_index
            turn_group["sage_grpo_group_size"] = group_size
            grouped.append(turn_group)

    if dropped_turns:
        print(
            "[SageGRPOTrainer] dropped "
            f"{dropped_turns} unmatched trailing completion(s) while forming "
            "same-turn groups",
            flush=True,
        )
    return grouped


def min_sequences_for_actor_dp(dp_size: int, ppo_n_minibatches: int) -> int:
    return max(1, dp_size) * max(1, ppo_n_minibatches)


def _zero_loss_padding_sequence(template: dict[str, Any]) -> dict[str, Any]:
    """Clone one sequence as a dispatch-only item with no loss or reward."""
    padded: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, torch.Tensor):
            padded[key] = value.clone()
        elif isinstance(value, list):
            padded[key] = list(value)
        else:
            padded[key] = value

    loss_mask = padded.get("loss_mask")
    rewards = padded.get("rewards")
    if not isinstance(loss_mask, torch.Tensor) or not isinstance(rewards, torch.Tensor):
        raise RuntimeError(
            "Cannot construct actor DP padding without tensor loss_mask and rewards"
        )
    padded["loss_mask"] = torch.zeros_like(loss_mask)
    padded["rewards"] = torch.zeros_like(rewards)
    padded["sage_layout_id"] = "__actor_dp_padding__"
    return padded


def validate_expanded_rollout_sequences(
    trajectories: list[dict[str, Any]],
    *,
    dp_size: int,
    ppo_n_minibatches: int,
    turn_group_size: int = 1,
) -> list[dict[str, Any]]:
    if turn_group_size > 1:
        already_grouped = bool(trajectories) and all(
            traj.get("sage_grpo_group_size") == turn_group_size
            and _tensor_batch_size(traj) == turn_group_size
            for traj in trajectories
        )
        if already_grouped:
            expanded = trajectories
        else:
            expanded = group_rollout_trajectories_by_turn(
                trajectories,
                group_size=turn_group_size,
            )
        sequence_count = sum(_tensor_batch_size(group) or 0 for group in expanded)
        description = (
            f"{len(expanded)} same-turn groups x {turn_group_size} "
            f"rollouts = {sequence_count} sequences"
        )
    else:
        expanded = expand_rollout_trajectories(trajectories)
        description = f"{len(expanded)} sequences"
    required = min_sequences_for_actor_dp(dp_size, ppo_n_minibatches)
    if len(expanded) < required:
        raise RuntimeError(
            "[SageGRPOTrainer] rollout produced too few trainable groups for actor DP: "
            f"got {len(expanded)} groups from {len(trajectories)} trajectories, "
            f"need at least {required} groups (dp_size={dp_size}, "
            f"ppo_n_minibatches={ppo_n_minibatches}). "
            "Retry rollout or reduce actor DP / ppo_n_minibatches."
        )
    padding_count = (-len(expanded)) % required if turn_group_size == 1 else 0
    if padding_count:
        template = min(
            expanded,
            key=lambda item: int(item["attention_mask"].sum().item())
            if isinstance(item.get("attention_mask"), torch.Tensor)
            else 1,
        )
        expanded = [
            *expanded,
            *(_zero_loss_padding_sequence(template) for _ in range(padding_count)),
        ]
        print(
            "[SageGRPOTrainer] padded actor dispatch with "
            f"{padding_count} zero-loss sequence(s): "
            f"total={len(expanded)}, divisor={required} "
            f"(dp_size={dp_size}, ppo_n_minibatches={ppo_n_minibatches})",
            flush=True,
        )
    print(
        f"[SageGRPOTrainer] transformed {len(trajectories)} trajectories -> "
        f"{description} (required_groups>={required})",
        flush=True,
    )
    return expanded
