"""Loss masks for individually exported, full-context agent turns."""

from typing import Any

import torch


def mask_current_turn(tensors: dict[str, Any]) -> dict[str, Any]:
    """Keep context tokens, but train each assistant action exactly once.

    AReaL's concat template carries ancestor action masks into child rows.
    That is appropriate for a final concatenated trajectory, but individual
    exports already contain those ancestors as separate training rows.
    """
    turn_ids = tensors["turn_ids"]
    current = turn_ids.max(dim=-1, keepdim=True).values
    mask = tensors["loss_mask"] * (turn_ids == current) * (current >= 0)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("Individual turn export has no current assistant tokens")
    return {**tensors, "loss_mask": mask}
