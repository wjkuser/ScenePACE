"""Failure provenance, separate from task reward and episode quality."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EpisodeOutcome:
    status: str
    reason_code: str
    detail: str = ""

    @property
    def trainable(self) -> bool:
        return self.status in {"success", "policy_failure"}

    def to_dict(self) -> dict:
        return asdict(self)


def classify_outcome(
    *,
    strict_success: bool,
    process_error: str = "",
    environment_error: str = "",
    evaluation_error: str = "",
    policy_error: str = "",
    layout_exists: bool = True,
) -> EpisodeOutcome:
    """Callers supply provenance, not a low reward threshold.

    Timeouts alone are ambiguous. Confirmed service transport failures are
    separate inputs; a policy producing valid but poor layouts remains trainable.
    """
    if environment_error:
        return EpisodeOutcome(
            "environment_failure", "service_failure", environment_error
        )
    if process_error:
        return EpisodeOutcome("unknown_failure", "episode_interrupted", process_error)
    if policy_error:
        return EpisodeOutcome("policy_failure", "invalid_action", policy_error)
    if not layout_exists:
        return EpisodeOutcome("policy_failure", "no_layout")
    if evaluation_error:
        return EpisodeOutcome(
            "unknown_failure", "evaluation_unavailable", evaluation_error
        )
    return EpisodeOutcome(
        "success" if strict_success else "policy_failure",
        "constraints_satisfied" if strict_success else "constraints_unmet",
    )
