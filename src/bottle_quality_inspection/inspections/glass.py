"""Glass-integrity preprocessing, feature extraction and classification."""

from dataclasses import dataclass, field
from enum import StrEnum
from math import atan2, degrees, hypot, pi

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.roi import RegionOfInterest


class GlassStatus(StrEnum):
    """Possible outcomes from the calibrated glass inspection."""

    CLEAR = "clear"
    DEFECTIVE = "defective"


@dataclass(frozen=True, slots=True)
class GlassPreprocessConfig:
    """Parameters reproduced from the 2022 glass experiments."""

    average_blur_kernel_size: int = 3
    gaussian_blur_kernel_size: int = 3

    canny_low_threshold: int = 17
    canny_high_threshold: int = 60

    binary_threshold: int = 100

    median_blur_kernel_size: int = 5
    global_binary_threshold: int = 127

    adaptive_block_size: int = 11
    adaptive_constant: float = 2.0

    def __post_init__(self) -> None:
        """Validate glass-preprocessing parameters."""
        if self.average_blur_kernel_size <= 0:
            raise ValueError(
                "average_blur_kernel_size must be greater than zero."
            )

        if (
            self.gaussian_blur_kernel_size <= 0
            or self.gaussian_blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "gaussian_blur_kernel_size must be a positive odd integer."
            )

        if (
            self.median_blur_kernel_size <= 1
            or self.median_blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "median_blur_kernel_size must be an odd integer greater than 1."
            )

        if (
            self.adaptive_block_size <= 1
            or self.adaptive_block_size % 2 == 0
        ):
            raise ValueError(
                "adaptive_block_size must be an odd integer greater than 1."
            )

        threshold_values = {
            "canny_low_threshold": self.canny_low_threshold,
            "canny_high_threshold": self.canny_high_threshold,
            "binary_threshold": self.binary_threshold,
            "global_binary_threshold": self.global_binary_threshold,
        }

        for name, value in threshold_values.items():
            if not 0 <= value <= 255:
                raise ValueError(
                    f"{name} must be between 0 and 255, received {value}."
                )

        if self.canny_low_threshold >= self.canny_high_threshold:
            raise ValueError(
                "canny_low_threshold must be lower than "
                "canny_high_threshold."
            )


@dataclass(frozen=True, slots=True)
class GlassPreprocessResult:
    """Outputs reproduced from the 2022 glass preprocessing experiments."""

    grayscale: np.ndarray

    average_blurred: np.ndarray
    average_edges: np.ndarray

    gaussian_blurred: np.ndarray
    gaussian_edges: np.ndarray

    binary_thresholded: np.ndarray
    binary_inverted: np.ndarray

    otsu_threshold_value: float
    otsu_inverted: np.ndarray
    otsu_masked: np.ndarray

    median_blurred: np.ndarray
    global_binary: np.ndarray
    adaptive_mean: np.ndarray
    adaptive_gaussian: np.ndarray


@dataclass(frozen=True, slots=True)
class GlassFeatureConfig:
    """
    Parameters for 2026 glass-integrity feature extraction.

    The analyzed region focuses on the cylindrical bottle body.

    The upper neck and shoulder area is intentionally excluded because
    normal bottle geometry creates strong horizontal edges there.

    These parameters define features only. They do not define a glass
    defect threshold.
    """

    preprocess: GlassPreprocessConfig = field(
        default_factory=GlassPreprocessConfig
    )

    interior_left_ratio: float = 0.18
    interior_right_ratio: float = 0.82

    interior_top_ratio: float = 0.42
    interior_bottom_ratio: float = 0.72

    hough_rho: float = 1.0
    hough_theta_degrees: float = 1.0
    hough_threshold: int = 15

    minimum_line_length: int = 20
    maximum_line_gap: int = 4

    maximum_horizontal_angle_degrees: float = 10.0

    def __post_init__(self) -> None:
        """Validate glass-feature extraction parameters."""
        if not (
            0.0
            <= self.interior_left_ratio
            < self.interior_right_ratio
            <= 1.0
        ):
            raise ValueError(
                "Glass horizontal interior ratios must satisfy "
                "0 <= left < right <= 1."
            )

        if not (
            0.0
            <= self.interior_top_ratio
            < self.interior_bottom_ratio
            <= 1.0
        ):
            raise ValueError(
                "Glass vertical interior ratios must satisfy "
                "0 <= top < bottom <= 1."
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

        if not (
            0.0
            <= self.maximum_horizontal_angle_degrees
            <= 90.0
        ):
            raise ValueError(
                "maximum_horizontal_angle_degrees must be "
                "between 0 and 90."
            )


@dataclass(frozen=True, slots=True)
class GlassDecisionConfig:
    """
    Calibrated 2026 glass decision parameters.

    These defaults were derived from the confirmed broken reference,
    the labeled normal reference and four visually normal bottles from
    the original video.

    They are calibration defaults, not universal manufacturing limits.
    """

    minimum_edge_density: float = 0.015

    minimum_horizontal_line_count: int = 1

    minimum_longest_horizontal_line_ratio: float = 0.20

    minimum_maximum_row_edge_ratio: float = 0.20

    minimum_triggered_signals: int = 3

    def __post_init__(self) -> None:
        """Validate calibrated glass-decision parameters."""
        ratio_values = {
            "minimum_edge_density": self.minimum_edge_density,
            "minimum_longest_horizontal_line_ratio": (
                self.minimum_longest_horizontal_line_ratio
            ),
            "minimum_maximum_row_edge_ratio": (
                self.minimum_maximum_row_edge_ratio
            ),
        }

        for name, value in ratio_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        if self.minimum_horizontal_line_count < 0:
            raise ValueError(
                "minimum_horizontal_line_count cannot be negative."
            )

        if not 1 <= self.minimum_triggered_signals <= 4:
            raise ValueError(
                "minimum_triggered_signals must be between 1 and 4."
            )


@dataclass(frozen=True, slots=True)
class GlassFeatureResult:
    """
    Measured glass-integrity features for one bottle crop.

    This object contains measurements only.
    """

    interior_region: RegionOfInterest

    edge_pixel_count: int
    edge_density: float

    detected_line_count: int

    horizontal_line_count: int
    longest_horizontal_line_length: float
    total_horizontal_line_length: float

    maximum_row_edge_count: int
    maximum_row_edge_ratio: float

    interior_edges: np.ndarray

    horizontal_lines: tuple[
        tuple[int, int, int, int],
        ...
    ]

    @property
    def interior_width(self) -> int:
        """Return the width of the analyzed bottle interior."""
        return int(
            self.interior_edges.shape[1]
        )

    @property
    def longest_horizontal_line_ratio(self) -> float:
        """Normalize the longest horizontal line by interior width."""
        if self.interior_width <= 0:
            return 0.0

        return (
            self.longest_horizontal_line_length
            / self.interior_width
        )

    @property
    def total_horizontal_line_ratio(self) -> float:
        """Normalize total horizontal-line length by interior width."""
        if self.interior_width <= 0:
            return 0.0

        return (
            self.total_horizontal_line_length
            / self.interior_width
        )


@dataclass(frozen=True, slots=True)
class GlassResult:
    """Final calibrated glass inspection result."""

    status: GlassStatus

    features: GlassFeatureResult

    triggered_signals: tuple[
        str,
        ...
    ]

    required_signal_count: int

    @property
    def is_defective(self) -> bool:
        """Return whether the glass was classified as defective."""
        return self.status is GlassStatus.DEFECTIVE

    @property
    def triggered_signal_count(self) -> int:
        """Return the number of defect signals that fired."""
        return len(
            self.triggered_signals
        )


def _validate_bgr_image(
    image: np.ndarray,
) -> None:
    """Validate an image before glass processing."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Glass inspection image must be a NumPy array."
        )

    if image.size == 0:
        raise InspectionError(
            "Glass inspection image cannot be empty."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise InspectionError(
            "Glass preprocessing requires a "
            "three-channel BGR image."
        )

    if image.dtype != np.uint8:
        raise InspectionError(
            "Glass preprocessing requires uint8 pixel values."
        )


def preprocess_glass(
    image: np.ndarray,
    config: GlassPreprocessConfig | None = None,
) -> GlassPreprocessResult:
    """
    Reproduce the preprocessing experiments from Glass preprocess.py.

    This function intentionally does not classify the glass as CLEAR
    or DEFECTIVE because the 2022 source did not contain a completed
    glass-integrity decision rule.
    """
    _validate_bgr_image(
        image
    )

    active_config = (
        config
        if config is not None
        else GlassPreprocessConfig()
    )

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    average_blurred = cv2.blur(
        grayscale,
        (
            active_config.average_blur_kernel_size,
            active_config.average_blur_kernel_size,
        ),
    )

    average_edges = cv2.Canny(
        average_blurred,
        active_config.canny_low_threshold,
        active_config.canny_high_threshold,
        L2gradient=True,
    )

    gaussian_blurred = cv2.GaussianBlur(
        grayscale,
        (
            active_config.gaussian_blur_kernel_size,
            active_config.gaussian_blur_kernel_size,
        ),
        0,
    )

    gaussian_edges = cv2.Canny(
        gaussian_blurred,
        active_config.canny_low_threshold,
        active_config.canny_high_threshold,
        L2gradient=True,
    )

    _, binary_thresholded = cv2.threshold(
        gaussian_blurred,
        active_config.binary_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    _, binary_inverted = cv2.threshold(
        gaussian_blurred,
        active_config.binary_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    (
        otsu_threshold_value,
        otsu_inverted,
    ) = cv2.threshold(
        gaussian_blurred,
        active_config.binary_threshold,
        255,
        cv2.THRESH_BINARY_INV
        | cv2.THRESH_OTSU,
    )

    otsu_masked = cv2.bitwise_and(
        image,
        image,
        mask=otsu_inverted,
    )

    median_blurred = cv2.medianBlur(
        grayscale,
        active_config.median_blur_kernel_size,
    )

    _, global_binary = cv2.threshold(
        median_blurred,
        active_config.global_binary_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    adaptive_mean = cv2.adaptiveThreshold(
        median_blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        active_config.adaptive_block_size,
        active_config.adaptive_constant,
    )

    adaptive_gaussian = cv2.adaptiveThreshold(
        median_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        active_config.adaptive_block_size,
        active_config.adaptive_constant,
    )

    return GlassPreprocessResult(
        grayscale=grayscale,
        average_blurred=average_blurred,
        average_edges=average_edges,
        gaussian_blurred=gaussian_blurred,
        gaussian_edges=gaussian_edges,
        binary_thresholded=binary_thresholded,
        binary_inverted=binary_inverted,
        otsu_threshold_value=float(
            otsu_threshold_value
        ),
        otsu_inverted=otsu_inverted,
        otsu_masked=otsu_masked,
        median_blurred=median_blurred,
        global_binary=global_binary,
        adaptive_mean=adaptive_mean,
        adaptive_gaussian=adaptive_gaussian,
    )


def _create_interior_region(
    image: np.ndarray,
    config: GlassFeatureConfig,
) -> RegionOfInterest:
    """Create a central bottle-body region using relative coordinates."""
    height, width = image.shape[:2]

    x_start = int(
        round(
            width
            * config.interior_left_ratio
        )
    )

    x_end = int(
        round(
            width
            * config.interior_right_ratio
        )
    )

    y_start = int(
        round(
            height
            * config.interior_top_ratio
        )
    )

    y_end = int(
        round(
            height
            * config.interior_bottom_ratio
        )
    )

    region_width = (
        x_end
        - x_start
    )

    region_height = (
        y_end
        - y_start
    )

    if (
        region_width <= 0
        or region_height <= 0
    ):
        raise InspectionError(
            "Glass interior region became empty "
            "for the supplied bottle image."
        )

    return RegionOfInterest(
        name="glass-interior",
        x=x_start,
        y=y_start,
        width=region_width,
        height=region_height,
    )


def _line_angle_degrees(
    line: tuple[int, int, int, int],
) -> float:
    """Return the smallest angle between a line and the horizontal axis."""
    x1, y1, x2, y2 = line

    angle = abs(
        degrees(
            atan2(
                y2 - y1,
                x2 - x1,
            )
        )
    )

    if angle > 90.0:
        angle = (
            180.0
            - angle
        )

    return angle


def _line_length(
    line: tuple[int, int, int, int],
) -> float:
    """Return Euclidean line length."""
    x1, y1, x2, y2 = line

    return hypot(
        x2 - x1,
        y2 - y1,
    )


def extract_glass_features(
    image: np.ndarray,
    config: GlassFeatureConfig | None = None,
) -> GlassFeatureResult:
    """Measure candidate glass-integrity features from one bottle crop."""
    _validate_bgr_image(
        image
    )

    active_config = (
        config
        if config is not None
        else GlassFeatureConfig()
    )

    preprocessing = preprocess_glass(
        image,
        active_config.preprocess,
    )

    interior_region = _create_interior_region(
        image,
        active_config,
    )

    interior_edges = interior_region.extract(
        preprocessing.gaussian_edges
    )

    edge_pixel_count = int(
        np.count_nonzero(
            interior_edges
        )
    )

    total_pixels = int(
        interior_edges.size
    )

    edge_density = (
        edge_pixel_count
        / total_pixels
        if total_pixels > 0
        else 0.0
    )

    theta_radians = (
        active_config.hough_theta_degrees
        * pi
        / 180.0
    )

    detected_lines_raw = cv2.HoughLinesP(
        interior_edges,
        active_config.hough_rho,
        theta_radians,
        active_config.hough_threshold,
        minLineLength=(
            active_config.minimum_line_length
        ),
        maxLineGap=(
            active_config.maximum_line_gap
        ),
    )

    detected_lines: list[
        tuple[int, int, int, int]
    ] = []

    if detected_lines_raw is not None:
        for raw_line in detected_lines_raw:
            x1, y1, x2, y2 = raw_line[0]

            detected_lines.append(
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            )

    horizontal_lines = tuple(
        line
        for line in detected_lines
        if (
            _line_angle_degrees(
                line
            )
            <= active_config.maximum_horizontal_angle_degrees
        )
    )

    horizontal_lengths = [
        _line_length(
            line
        )
        for line in horizontal_lines
    ]

    longest_horizontal_line_length = (
        max(
            horizontal_lengths
        )
        if horizontal_lengths
        else 0.0
    )

    total_horizontal_line_length = float(
        sum(
            horizontal_lengths
        )
    )

    row_edge_counts = np.count_nonzero(
        interior_edges,
        axis=1,
    )

    maximum_row_edge_count = int(
        row_edge_counts.max()
        if row_edge_counts.size
        else 0
    )

    maximum_row_edge_ratio = (
        maximum_row_edge_count
        / interior_edges.shape[1]
        if interior_edges.shape[1] > 0
        else 0.0
    )

    return GlassFeatureResult(
        interior_region=interior_region,
        edge_pixel_count=edge_pixel_count,
        edge_density=float(
            edge_density
        ),
        detected_line_count=len(
            detected_lines
        ),
        horizontal_line_count=len(
            horizontal_lines
        ),
        longest_horizontal_line_length=float(
            longest_horizontal_line_length
        ),
        total_horizontal_line_length=(
            total_horizontal_line_length
        ),
        maximum_row_edge_count=(
            maximum_row_edge_count
        ),
        maximum_row_edge_ratio=float(
            maximum_row_edge_ratio
        ),
        interior_edges=interior_edges.copy(),
        horizontal_lines=horizontal_lines,
    )


def classify_glass_features(
    features: GlassFeatureResult,
    config: GlassDecisionConfig | None = None,
) -> GlassResult:
    """
    Classify measured glass features using the calibrated 2026 rule.

    At least the configured number of independent defect signals must
    trigger before the bottle is classified as DEFECTIVE.
    """
    active_config = (
        config
        if config is not None
        else GlassDecisionConfig()
    )

    triggered_signals: list[
        str
    ] = []

    if (
        features.edge_density
        >= active_config.minimum_edge_density
    ):
        triggered_signals.append(
            "edge_density"
        )

    if (
        features.horizontal_line_count
        >= active_config.minimum_horizontal_line_count
    ):
        triggered_signals.append(
            "horizontal_line_count"
        )

    if (
        features.longest_horizontal_line_ratio
        >= active_config.minimum_longest_horizontal_line_ratio
    ):
        triggered_signals.append(
            "longest_horizontal_line_ratio"
        )

    if (
        features.maximum_row_edge_ratio
        >= active_config.minimum_maximum_row_edge_ratio
    ):
        triggered_signals.append(
            "maximum_row_edge_ratio"
        )

    status = (
        GlassStatus.DEFECTIVE
        if (
            len(triggered_signals)
            >= active_config.minimum_triggered_signals
        )
        else GlassStatus.CLEAR
    )

    return GlassResult(
        status=status,
        features=features,
        triggered_signals=tuple(
            triggered_signals
        ),
        required_signal_count=(
            active_config.minimum_triggered_signals
        ),
    )


def inspect_glass(
    image: np.ndarray,
    feature_config: GlassFeatureConfig | None = None,
    decision_config: GlassDecisionConfig | None = None,
) -> GlassResult:
    """Run calibrated glass feature extraction and classification."""
    features = extract_glass_features(
        image,
        feature_config,
    )

    return classify_glass_features(
        features,
        decision_config,
    )


def render_glass_features(
    image: np.ndarray,
    result: GlassFeatureResult,
) -> np.ndarray:
    """Visualize the measured glass features without classifying them."""
    _validate_bgr_image(
        image
    )

    annotated = image.copy()

    region = result.interior_region

    annotated = region.draw(
        annotated,
        color=(255, 255, 0),
        thickness=2,
        show_label=True,
    )

    for line in result.horizontal_lines:
        x1, y1, x2, y2 = line

        cv2.line(
            annotated,
            (
                region.x + x1,
                region.y + y1,
            ),
            (
                region.x + x2,
                region.y + y2,
            ),
            (0, 0, 255),
            2,
        )

    label = (
        "GLASS FEATURES | "
        f"edges={result.edge_density:.3f} "
        f"horizontal={result.horizontal_line_count}"
    )

    cv2.putText(
        annotated,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def render_glass_result(
    image: np.ndarray,
    result: GlassResult,
) -> np.ndarray:
    """Render the calibrated glass decision and its measured features."""
    annotated = render_glass_features(
        image,
        result.features,
    )

    status_colors = {
        GlassStatus.CLEAR: (
            0,
            255,
            0,
        ),
        GlassStatus.DEFECTIVE: (
            0,
            0,
            255,
        ),
    }

    color = status_colors[
        result.status
    ]

    label = (
        f"GLASS {result.status.value.upper()} | "
        f"signals="
        f"{result.triggered_signal_count}/"
        f"{result.required_signal_count}"
    )

    cv2.putText(
        annotated,
        label,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated