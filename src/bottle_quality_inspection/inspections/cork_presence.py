"""Multi-frame cork-presence evaluation for tracked bottles."""

from dataclasses import dataclass
from enum import StrEnum
from statistics import median


class CorkPresenceStatus(StrEnum):
    """Possible outcomes from tracked multi-frame cork evidence."""

    PRESENT = "present"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class CorkPresenceFrameScore:
    """Cork and negative-control template scores for one frame."""

    frame_index: int
    cork_score: float
    body_control_score: float

    def __post_init__(self) -> None:
        """Validate one frame-level cork-presence measurement."""
        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative."
            )

        score_values = {
            "cork_score": self.cork_score,
            "body_control_score": self.body_control_score,
        }

        for name, value in score_values.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between -1.0 and 1.0."
                )

    @property
    def margin(self) -> float:
        """Return cork score minus body-control score."""
        return (
            self.cork_score
            - self.body_control_score
        )


@dataclass(frozen=True, slots=True)
class CorkPresenceConfig:
    """
    Calibrated tracked-cork presence settings.

    These defaults are based on three visibly corked tracked bottles
    from the original video.

    They identify evidence consistent with cork presence. They do not
    classify physical cork defects.
    """

    top_k_frames: int = 3

    minimum_top_k_cork_score: float = 0.75

    minimum_top_k_margin: float = 0.20

    def __post_init__(self) -> None:
        """Validate cork-presence calibration parameters."""
        if self.top_k_frames <= 0:
            raise ValueError(
                "top_k_frames must be greater than zero."
            )

        if not -1.0 <= self.minimum_top_k_cork_score <= 1.0:
            raise ValueError(
                "minimum_top_k_cork_score must be "
                "between -1.0 and 1.0."
            )

        if not -2.0 <= self.minimum_top_k_margin <= 2.0:
            raise ValueError(
                "minimum_top_k_margin must be "
                "between -2.0 and 2.0."
            )


@dataclass(frozen=True, slots=True)
class CorkPresenceResult:
    """Aggregated cork-presence evidence for one tracked bottle."""

    status: CorkPresenceStatus

    evaluated_frame_count: int

    selected_frame_indices: tuple[
        int,
        ...
    ]

    top_k_cork_median: float
    top_k_body_control_median: float
    top_k_margin: float

    minimum_required_cork_score: float
    minimum_required_margin: float

    @property
    def cork_present(self) -> bool:
        """Return whether calibrated presence evidence passed."""
        return (
            self.status
            is CorkPresenceStatus.PRESENT
        )


def evaluate_cork_presence(
    frame_scores: list[CorkPresenceFrameScore],
    config: CorkPresenceConfig | None = None,
) -> CorkPresenceResult:
    """
    Aggregate multi-frame cork evidence for one tracked bottle.

    The frames with the strongest cork-template scores are selected.
    Cork and negative-control medians are then calculated using those
    same frames so the comparison remains paired.
    """
    if not frame_scores:
        raise ValueError(
            "At least one cork-presence frame score is required."
        )

    active_config = (
        config
        if config is not None
        else CorkPresenceConfig()
    )

    ordered_scores = sorted(
        frame_scores,
        key=lambda score: (
            score.cork_score
        ),
        reverse=True,
    )

    selected = ordered_scores[
        : min(
            active_config.top_k_frames,
            len(ordered_scores),
        )
    ]

    cork_values = [
        score.cork_score
        for score in selected
    ]

    body_values = [
        score.body_control_score
        for score in selected
    ]

    top_k_cork_median = float(
        median(
            cork_values
        )
    )

    top_k_body_control_median = float(
        median(
            body_values
        )
    )

    top_k_margin = (
        top_k_cork_median
        - top_k_body_control_median
    )

    status = (
        CorkPresenceStatus.PRESENT
        if (
            top_k_cork_median
            >= active_config.minimum_top_k_cork_score
            and top_k_margin
            >= active_config.minimum_top_k_margin
        )
        else CorkPresenceStatus.UNVERIFIED
    )

    return CorkPresenceResult(
        status=status,
        evaluated_frame_count=len(
            frame_scores
        ),
        selected_frame_indices=tuple(
            score.frame_index
            for score in selected
        ),
        top_k_cork_median=(
            top_k_cork_median
        ),
        top_k_body_control_median=(
            top_k_body_control_median
        ),
        top_k_margin=(
            top_k_margin
        ),
        minimum_required_cork_score=(
            active_config.minimum_top_k_cork_score
        ),
        minimum_required_margin=(
            active_config.minimum_top_k_margin
        ),
    )