"""Shared design dimensions and visual reward calibration."""
from pathlib import Path

VISUAL_DIMENSIONS = (
    "spatial_organization", "furnishing_composition",
    "color_style_composition", "brief_realization",
)
VISUAL_PROMPT = Path(__file__).with_name("visual_prompt.txt").read_text()


def visual_reward(raw_score: float) -> float:
    """Map the normalized aggregate score to the visual reward component."""
    return max(raw_score - 0.5, 0.0) / 0.5
