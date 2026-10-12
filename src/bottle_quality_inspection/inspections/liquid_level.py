"""Classical computer-vision inspection for bottle liquid level."""

from dataclasses import dataclass, field
from enum import StrEnum
from math import pi

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.roi import RegionOfInterest


class LiquidLevelStatus(StrEnum):
    """Possible outcomes from one liquid-level inspection."""

    NO_LEVEL = "no_level"
    CLEAR = "clear"
    DEFECTIVE = "defective"


@dataclass(frozen=True, slots=True)
class LiquidLevelConfig:
    """Parameters controlling the legacy liquid-level algorithm."""

    neck_region: RegionOfInterest = field(
        default_factory=lambda: RegionOfInterest(
            name="liquid-level-neck",
            x=0,
            y=192,
            width=200,
            height=58,
        )
    )

    canny_threshold1: int = 252
    canny_threshold2: int = 119

    hough_rho: float = 1.0
    hough_theta_degrees: float = 1.0
    hough_threshold: int = 15
    minimum_line_length: int = 20
    maximum_line_gap: int = 10

    acceptable_minimum: float = 225.0
    acceptable_maximum: float = 247.0

    def __post_init__(self) -> None:
        """Validate configurable liquid-level parameters."""
        threshold_values = {
            "canny_threshold1": self.canny_threshold1,
            "canny_threshold2": self.canny_threshold2,
        }

        for name, value in threshold_values.items():
            if not 0 <= value <= 255:
                raise ValueError(
                    f"{name} must be between 0 and 255, received {value}."
                )

        if self.hough_rho <= 0:
            raise ValueError(
                "hough_rho must be greater than zero."
            )

        if self.hough_theta_degrees <= 0:
            raise ValueError(
                "hough_theta_degrees must be greater than zero."
            )

        if self.hough_threshold <= 0:
            raise ValueError(
                "hough_threshold must be greater than zero."
            )

        if self.minimum_line_length <= 0:
            raise ValueError(
                "minimum_line_length must be greater than zero."
            )

        if self.maximum_line_gap < 0:
            raise ValueError(
                "maximum_line_gap cannot be negative."
            )

        if self.acceptable_minimum >= self.acceptable_maximum:
            raise ValueError(
                "acceptable_minimum must be lower than acceptable_maximum."
            )


@dataclass(frozen=True, slots=True)
class LiquidLevelResult:
    """Detailed result from one liquid-level inspection."""

    status: LiquidLevelStatus

    relative_level: float | None
    absolute_level: float | None

    detected_line_count: int

    edges: np.ndarray

    detected_lines: tuple[
        tuple[int, int, int, int],
        ...
    ]

    selected_line: tuple[int, int, int, int] | None

    @property
    def level_detected(self) -> bool:
        """Return whether a probable liquid level was detected."""
        return self.status is not LiquidLevelStatus.NO_LEVEL

    @property
    def is_defective(self) -> bool:
        """Return whether the liquid level is outside allowed limits."""
        return self.status is LiquidLevelStatus.DEFECTIVE


def _validate_bgr_image(
    image: np.ndarray,
) -> None:
    """Validate an image before liquid-level inspection."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Inspection image must be a NumPy array."
        )

    if image.size == 0:
        raise InspectionError(
            "Inspection image cannot be empty."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise InspectionError(
            "Liquid-level inspection requires a "
            "three-channel BGR image."
        )

    if image.dtype != np.uint8:
        raise InspectionError(
            "Liquid-level inspection requires uint8 pixel values."
        )


def _empty_result(
    edges: np.ndarray,
) -> LiquidLevelResult:
    """Create a result for a frame where no liquid level was found."""
    return LiquidLevelResult(
        status=LiquidLevelStatus.NO_LEVEL,
        relative_level=None,
        absolute_level=None,
        detected_line_count=0,
        edges=edges,
        detected_lines=(),
        selected_line=None,
    )


def _normalise_lines(
    lines: np.ndarray,
) -> tuple[
    tuple[int, int, int, int],
    ...
]:
    """Convert OpenCV Hough lines into immutable coordinate tuples."""
    normalised: list[
        tuple[int, int, int, int]
    ] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        normalised.append(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )
        )

    return tuple(normalised)


def _line_average_y(
    line: tuple[int, int, int, int],
) -> float:
    """Return the average vertical coordinate of a detected line."""
    _, y1, _, y2 = line

    return (
        float(y1)
        + float(y2)
    ) / 2.0


def inspect_liquid_level(
    image: np.ndarray,
    config: LiquidLevelConfig | None = None,
) -> LiquidLevelResult:
    """
    Inspect a bottle ROI for its liquid level.

    This intentionally reproduces the 2022 processing sequence.

    The original source used:

        cv2.cvtColor(roi, 0)

    OpenCV conversion code 0 converts BGR to BGRA. Although unusual
    for this task, it is preserved here because this module represents
    the legacy-2022 baseline rather than an improved algorithm.
    """
    _validate_bgr_image(image)

    active_config = (
        config
        if config is not None
        else LiquidLevelConfig()
    )

    # Preserve the original 2022 operation exactly.
    #
    # Original:
    #     roi = cv2.cvtColor(roi, 0)
    #
    # OpenCV code 0 corresponds to BGR -> BGRA.
    legacy_image = cv2.cvtColor(
        image,
        0,
    )

    # Original:
    #     cropped_image = roi[192:250, 0:300]
    #
    # The parent ROI is only 200 pixels wide, so the effective
    # crop width is 200 pixels.
    neck_image = active_config.neck_region.extract(
        legacy_image
    )

    # Preserve the original threshold order exactly.
    #
    # Original:
    #     cv2.Canny(cropped_image, 252, 119)
    edges = cv2.Canny(
        neck_image,
        active_config.canny_threshold1,
        active_config.canny_threshold2,
    )

    theta_radians = (
        active_config.hough_theta_degrees
        * pi
        / 180.0
    )

    lines = cv2.HoughLinesP(
        edges,
        active_config.hough_rho,
        theta_radians,
        active_config.hough_threshold,
        np.array([]),
        active_config.minimum_line_length,
        active_config.maximum_line_gap,
    )

    if lines is None or len(lines) == 0:
        return _empty_result(
            edges
        )

    detected_lines = _normalise_lines(
        lines
    )

    # Original:
    #
    # y_avgs = [
    #     (line[0, 1] + line[0, 3]) / 2
    #     for line in lines
    # ]
    #
    # frame_liquid_level = max(y_avgs)
    selected_line = max(
        detected_lines,
        key=_line_average_y,
    )

    relative_level = _line_average_y(
        selected_line
    )

    # The original caller performed:
    #
    #     liquid_lvl = liquidlevel(roi) + 192
    #
    # Therefore the final measurement is relative level + neck Y.
    absolute_level = (
        float(active_config.neck_region.y)
        + relative_level
    )

    status = (
        LiquidLevelStatus.CLEAR
        if (
            active_config.acceptable_minimum
            < absolute_level
            < active_config.acceptable_maximum
        )
        else LiquidLevelStatus.DEFECTIVE
    )

    return LiquidLevelResult(
        status=status,
        relative_level=relative_level,
        absolute_level=absolute_level,
        detected_line_count=len(
            detected_lines
        ),
        edges=edges,
        detected_lines=detected_lines,
        selected_line=selected_line,
    )


def render_liquid_level_result(
    image: np.ndarray,
    result: LiquidLevelResult,
    config: LiquidLevelConfig | None = None,
) -> np.ndarray:
    """Render liquid-level detection information on an image copy."""
    _validate_bgr_image(image)

    active_config = (
        config
        if config is not None
        else LiquidLevelConfig()
    )

    annotated = image.copy()

    neck_region = active_config.neck_region

    annotated = neck_region.draw(
        annotated,
        color=(255, 255, 0),
        thickness=2,
        show_label=True,
    )

    minimum_y = int(
        round(
            active_config.acceptable_minimum
        )
    )

    maximum_y = int(
        round(
            active_config.acceptable_maximum
        )
    )

    cv2.line(
        annotated,
        (
            neck_region.x,
            minimum_y,
        ),
        (
            neck_region.x_end - 1,
            minimum_y,
        ),
        (255, 0, 0),
        2,
    )

    cv2.line(
        annotated,
        (
            neck_region.x,
            maximum_y,
        ),
        (
            neck_region.x_end - 1,
            maximum_y,
        ),
        (255, 0, 0),
        2,
    )

    status_colors = {
        LiquidLevelStatus.NO_LEVEL: (
            0,
            255,
            255,
        ),
        LiquidLevelStatus.CLEAR: (
            0,
            255,
            0,
        ),
        LiquidLevelStatus.DEFECTIVE: (
            0,
            0,
            255,
        ),
    }

    status_color = status_colors[
        result.status
    ]

    if (
        result.absolute_level is not None
        and result.selected_line is not None
    ):
        x1, _, x2, _ = result.selected_line

        detected_y = int(
            round(
                result.absolute_level
            )
        )

        cv2.line(
            annotated,
            (
                neck_region.x + x1,
                detected_y,
            ),
            (
                neck_region.x + x2,
                detected_y,
            ),
            status_color,
            3,
        )

    if result.absolute_level is None:
        level_text = "N/A"
    else:
        level_text = (
            f"{result.absolute_level:.1f}"
        )

    label = (
        f"{result.status.value.upper()} | "
        f"level: {level_text}"
    )

    cv2.putText(
        annotated,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return annotated