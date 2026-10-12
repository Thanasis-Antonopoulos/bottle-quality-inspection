"""Classical computer-vision inspection for foreign objects."""

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.roi import RegionOfInterest


class ForeignObjectStatus(StrEnum):
    """Possible outcomes from one foreign-object inspection."""

    NO_BOTTLE = "no_bottle"
    CLEAR = "clear"
    DEFECTIVE = "defective"


@dataclass(frozen=True, slots=True)
class ForeignObjectConfig:
    """Parameters controlling the classical inspection algorithm."""

    gaussian_kernel_size: int = 5
    closing_kernel_size: int = 13
    content_blur_kernel_size: int = 11

    canny_low_threshold: int = 17
    canny_high_threshold: int = 60

    content_threshold: int = 127
    reflection_threshold: int = 156

    component_count: int = 2
    minimum_contour_area: float = 1.0

    defect_pixel_threshold: int = 900

    analysis_region: RegionOfInterest | None = None

    def __post_init__(self) -> None:
        """Validate configurable algorithm parameters."""
        kernel_values = {
            "gaussian_kernel_size": self.gaussian_kernel_size,
            "closing_kernel_size": self.closing_kernel_size,
            "content_blur_kernel_size": self.content_blur_kernel_size,
        }

        for name, value in kernel_values.items():
            if value <= 0 or value % 2 == 0:
                raise ValueError(
                    f"{name} must be a positive odd integer, received {value}."
                )

        threshold_values = {
            "canny_low_threshold": self.canny_low_threshold,
            "canny_high_threshold": self.canny_high_threshold,
            "content_threshold": self.content_threshold,
            "reflection_threshold": self.reflection_threshold,
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

        if self.component_count <= 0:
            raise ValueError(
                "component_count must be greater than zero."
            )

        if self.minimum_contour_area < 0:
            raise ValueError(
                "minimum_contour_area cannot be negative."
            )

        if self.defect_pixel_threshold < 0:
            raise ValueError(
                "defect_pixel_threshold cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class ForeignObjectResult:
    """Detailed result from one foreign-object inspection."""

    status: ForeignObjectStatus

    raw_anomaly_pixel_count: int
    decision_black_pixel_count: int

    largest_contour_area: float
    selected_component_count: int

    bottle_mask: np.ndarray
    content_mask: np.ndarray
    anomaly_mask: np.ndarray
    legacy_output: np.ndarray

    @property
    def anomaly_pixel_count(self) -> int:
        """
        Return the legacy decision count.

        This property is retained for compatibility with earlier code.
        """
        return self.decision_black_pixel_count

    @property
    def bottle_detected(self) -> bool:
        """Return whether a probable bottle region was found."""
        return self.status is not ForeignObjectStatus.NO_BOTTLE

    @property
    def is_defective(self) -> bool:
        """Return whether the configured legacy threshold was exceeded."""
        return self.status is ForeignObjectStatus.DEFECTIVE


def _validate_bgr_image(image: np.ndarray) -> None:
    """Validate an image before running the inspection."""
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
            "Foreign-object inspection requires a "
            "three-channel BGR image."
        )

    if image.dtype != np.uint8:
        raise InspectionError(
            "Foreign-object inspection requires uint8 pixel values."
        )


def _no_bottle_result(
    image: np.ndarray,
) -> ForeignObjectResult:
    """Create a result representing an undetected bottle."""
    height, width = image.shape[:2]
    empty_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    return ForeignObjectResult(
        status=ForeignObjectStatus.NO_BOTTLE,
        raw_anomaly_pixel_count=0,
        decision_black_pixel_count=0,
        largest_contour_area=0.0,
        selected_component_count=0,
        bottle_mask=empty_mask.copy(),
        content_mask=empty_mask.copy(),
        anomaly_mask=empty_mask.copy(),
        legacy_output=empty_mask.copy(),
    )


def _count_black_pixels(
    image: np.ndarray,
) -> int:
    """
    Count pixels that are exactly black.

    This reproduces the intent of the final black-pixel count used
    by the 2022 implementation.
    """
    return int(
        np.count_nonzero(image == 0)
    )


def inspect_foreign_objects(
    image: np.ndarray,
    config: ForeignObjectConfig | None = None,
) -> ForeignObjectResult:
    """
    Inspect a bottle ROI for possible foreign objects.

    The processing pipeline preserves the original 2022 classical
    computer-vision approach. The final PASS/FAIL decision is based
    on the number of exactly black pixels in the configured decision
    region of the legacy-style output image.
    """
    _validate_bgr_image(image)

    active_config = (
        config
        if config is not None
        else ForeignObjectConfig()
    )

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Legacy step 1:
    # smooth the grayscale image.
    blurred = cv2.GaussianBlur(
        grayscale,
        (
            active_config.gaussian_kernel_size,
            active_config.gaussian_kernel_size,
        ),
        0,
    )

    # Legacy step 2:
    # detect discontinuities.
    edges = cv2.Canny(
        blurred,
        active_config.canny_low_threshold,
        active_config.canny_high_threshold,
        L2gradient=True,
    )

    # Legacy step 3:
    # close small gaps in detected edges.
    closing_kernel = np.ones(
        (
            active_config.closing_kernel_size,
            active_config.closing_kernel_size,
        ),
        dtype=np.uint8,
    )

    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        closing_kernel,
    )

    # Legacy step 4:
    # find contours and retain the largest.
    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return _no_bottle_result(image)

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    largest_contour_area = float(
        cv2.contourArea(largest_contour)
    )

    if (
        largest_contour_area
        < active_config.minimum_contour_area
    ):
        return _no_bottle_result(image)

    # Legacy step 5:
    # create a mask from the largest contour.
    bottle_mask = np.zeros_like(
        grayscale
    )

    cv2.drawContours(
        bottle_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    masked_grayscale = cv2.bitwise_and(
        grayscale,
        grayscale,
        mask=bottle_mask,
    )

    # Legacy step 6:
    # blur and threshold the probable bottle content.
    content_blur = cv2.GaussianBlur(
        masked_grayscale,
        (
            active_config.content_blur_kernel_size,
            active_config.content_blur_kernel_size,
        ),
        0,
    )

    _, binary_content = cv2.threshold(
        content_blur,
        active_config.content_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # Legacy step 7:
    # connected-component analysis.
    label_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_content
        )
    )

    foreground_labels = list(
        range(1, label_count)
    )

    foreground_labels.sort(
        key=lambda label: int(
            stats[label, cv2.CC_STAT_AREA]
        ),
        reverse=True,
    )

    selected_labels = foreground_labels[
        : active_config.component_count
    ]

    if not selected_labels:
        return _no_bottle_result(image)

    # Legacy step 8:
    # retain the two largest foreground components.
    content_mask = np.zeros_like(
        grayscale
    )

    for label in selected_labels:
        content_mask[
            labels == label
        ] = 255

    masked_content = cv2.bitwise_and(
        grayscale,
        grayscale,
        mask=content_mask,
    )

    # Legacy step 9:
    # extract internal edges.
    anomaly_mask = cv2.Canny(
        masked_content,
        active_config.canny_low_threshold,
        active_config.canny_high_threshold,
    )

    # Legacy steps 10-12:
    # remove bright reflection edges.
    edge_intensities = cv2.bitwise_and(
        grayscale,
        grayscale,
        mask=anomaly_mask,
    )

    anomaly_mask[
        edge_intensities
        > active_config.reflection_threshold
    ] = 0

    # Diagnostic value:
    # all retained anomaly-edge pixels across the processing ROI.
    raw_anomaly_pixel_count = int(
        cv2.countNonZero(anomaly_mask)
    )

    # Legacy steps 13-15:
    # reproduce the final processed image used by the 2022 script.
    final_product = cv2.bitwise_and(
        grayscale,
        grayscale,
        mask=anomaly_mask,
    )

    legacy_output = cv2.absdiff(
        grayscale,
        final_product,
    )

    # The old script performed its final pixel count only inside
    # frame[570:850, 600:780].
    #
    # In the new system that crop is represented by analysis_region
    # relative to the foreign-object inspection ROI.
    if active_config.analysis_region is not None:
        decision_image = (
            active_config.analysis_region.extract(
                legacy_output
            )
        )
    else:
        decision_image = legacy_output

    decision_black_pixel_count = (
        _count_black_pixels(
            decision_image
        )
    )

    status = (
        ForeignObjectStatus.DEFECTIVE
        if (
            decision_black_pixel_count
            > active_config.defect_pixel_threshold
        )
        else ForeignObjectStatus.CLEAR
    )

    return ForeignObjectResult(
        status=status,
        raw_anomaly_pixel_count=(
            raw_anomaly_pixel_count
        ),
        decision_black_pixel_count=(
            decision_black_pixel_count
        ),
        largest_contour_area=(
            largest_contour_area
        ),
        selected_component_count=len(
            selected_labels
        ),
        bottle_mask=bottle_mask,
        content_mask=content_mask,
        anomaly_mask=anomaly_mask,
        legacy_output=legacy_output,
    )


def render_foreign_object_result(
    image: np.ndarray,
    result: ForeignObjectResult,
) -> np.ndarray:
    """Render anomaly pixels and inspection status on an image copy."""
    _validate_bgr_image(image)

    if result.anomaly_mask.shape != image.shape[:2]:
        raise InspectionError(
            "Anomaly-mask dimensions do not match "
            "the inspection image."
        )

    if result.legacy_output.shape != image.shape[:2]:
        raise InspectionError(
            "Legacy-output dimensions do not match "
            "the inspection image."
        )

    annotated = image.copy()

    # Red pixels show diagnostic anomaly edges.
    annotated[
        result.anomaly_mask > 0
    ] = (0, 0, 255)

    status_colors = {
        ForeignObjectStatus.NO_BOTTLE: (
            0,
            255,
            255,
        ),
        ForeignObjectStatus.CLEAR: (
            0,
            255,
            0,
        ),
        ForeignObjectStatus.DEFECTIVE: (
            0,
            0,
            255,
        ),
    }

    color = status_colors[
        result.status
    ]

    label = (
        f"{result.status.value.upper()} | "
        f"black pixels: "
        f"{result.decision_black_pixel_count}"
    )

    cv2.putText(
        annotated,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated