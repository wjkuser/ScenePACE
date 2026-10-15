"""Dynamic rollout filters for GRPO training."""

from __future__ import annotations

from typing import Any

from sage_rl.rollout_batch import (
    is_failed_layout_rollout,
    layout_rollout_rejection_reason,
)


def accept_valid_layout_rollout(traj: dict[str, Any]) -> bool:
    """Reject failed layout rollouts so the dispatcher runs supplemental rollouts."""
    if is_failed_layout_rollout(traj):
        print(
            "[rollout_filter] rejected layout rollout: "
            f"{layout_rollout_rejection_reason(traj)}; requesting supplemental rollout",
            flush=True,
        )
        return False
    return True
