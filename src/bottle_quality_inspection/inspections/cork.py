"""Template-matching inspection for bottle cork quality."""

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import InspectionError


class CorkStatus(StrEnum):
    """Possible outcomes from one cork inspection."""

    CLEAR = "clear"
    DEFECTIVE = "defective"


@dataclass(frozen=True, slots=True)
class CorkConfig:
    """Parameters controlling the legacy cork template match."""

    method: str = "TM_CCOEFF_NORMED"
    minimum_match_score: float = 0.80

    def __post_init__(self) -> None:
        """Validate cork-inspection configuration."""
        if self.method != "TM_CCOEFF_NORMED":
            raise ValueError(
                "The legacy cork baseline currently supports only "
                "TM_CCOEFF_NORMED."
            )

        if not 0.0 <= self.minimum_match_score <= 1.0:
            raise ValueError(
                "minimum_match_score must be between 0.0 and 1.0."
            )


@dataclass(frozen=True, slots=True)
class CorkResult:
    """Detailed result from one cork template match."""

    status: CorkStatus

    minimum_score: float
    maximum_score: float

    minimum_location: tuple[int, int]
    maximum_location: tuple[int, int]

    template_width: int
    template_height: int

    matches_at_or_above_threshold: int

    @property
    def is_defective(self) -> bool:
        """Return whether the cork failed the configured threshold."""
        return self.status is CorkStatus.DEFECTIVE

    @property
    def match_found(self) -> bool:
        """Return whether at least one acceptable match was found."""
        return (
            self.matches_at_or_above_threshold
            > 0
        )

    @property
    def bounding_box(
        self,
    ) -> tuple[int, int, int, int]:
        """
        Return the best-match rectangle.

        The result is:

            x, y, width, height
        """
        x, y = self.maximum_location

        return (
            x,
            y,
            self.template_width,
            self.template_height,
        )


def _validate_bgr_image(
    image: np.ndarray,
    *,
    name: str,
) -> None:
    """Validate an image used by the cork inspector."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{name} must be a NumPy array."
        )

    if image.size == 0:
        raise InspectionError(
            f"{name} cannot be empty."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise InspectionError(
            f"{name} must be a three-channel BGR image."
        )

    if image.dtype != np.uint8:
        raise InspectionError(
            f"{name} must contain uint8 pixel values."
        )


def inspect_cork(
    image: np.ndarray,
    template: np.ndarray,
    config: CorkConfig | None = None,
) -> CorkResult:
    """
    Inspect an image using the legacy 2022 cork template.

    The legacy standalone cork experiment:

    1. Converted the source image to grayscale.
    2. Converted the correct-cork template to grayscale.
    3. Used cv2.TM_CCOEFF_NORMED.
    4. Used 0.80 as the acceptable template-match threshold.

    The best match is used for the primary result while the number
    of locations meeting the threshold is retained for diagnostics.
    """
    _validate_bgr_image(
        image,
        name="Inspection image",
    )

    _validate_bgr_image(
        template,
        name="Cork template",
    )

    active_config = (
        config
        if config is not None
        else CorkConfig()
    )

    image_height, image_width = (
        image.shape[:2]
    )

    template_height, template_width = (
        template.shape[:2]
    )

    if (
        template_width > image_width
        or template_height > image_height
    ):
        raise InspectionError(
            "Cork template cannot be larger than "
            "the inspection image."
        )

    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    grayscale_template = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY,
    )

    result_matrix = cv2.matchTemplate(
        grayscale_image,
        grayscale_template,
        cv2.TM_CCOEFF_NORMED,
    )

    (
        minimum_score,
        maximum_score,
        minimum_location,
        maximum_location,
    ) = cv2.minMaxLoc(
        result_matrix
    )

    matches_at_or_above_threshold = int(
        np.count_nonzero(
            result_matrix
            >= active_config.minimum_match_score
        )
    )

    status = (
        CorkStatus.CLEAR
        if (
            maximum_score
            >= active_config.minimum_match_score
        )
        else CorkStatus.DEFECTIVE
    )

    return CorkResult(
        status=status,
        minimum_score=float(
            minimum_score
        ),
        maximum_score=float(
            maximum_score
        ),
        minimum_location=(
            int(minimum_location[0]),
            int(minimum_location[1]),
        ),
        maximum_location=(
            int(maximum_location[0]),
            int(maximum_location[1]),
        ),
        template_width=(
            template_width
        ),
        template_height=(
            template_height
        ),
        matches_at_or_above_threshold=(
            matches_at_or_above_threshold
        ),
    )


def render_cork_result(
    image: np.ndarray,
    result: CorkResult,
) -> np.ndarray:
    """Render the best cork-template match on an image copy."""
    _validate_bgr_image(
        image,
        name="Inspection image",
    )

    annotated = image.copy()

    status_colors = {
        CorkStatus.CLEAR: (
            0,
            255,
            0,
        ),
        CorkStatus.DEFECTIVE: (
            0,
            0,
            255,
        ),
    }

    color = status_colors[
        result.status
    ]

    x, y, width, height = (
        result.bounding_box
    )

    cv2.rectangle(
        annotated,
        (x, y),
        (
            x + width,
            y + height,
        ),
        color,
        3,
    )

    label = (
        f"CORK {result.status.value.upper()} | "
        f"score: {result.maximum_score:.3f}"
    )

    text_y = (
        y - 10
        if y >= 40
        else y + height + 30
    )

    cv2.putText(
        annotated,
        label,
        (
            max(x, 0),
            text_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated