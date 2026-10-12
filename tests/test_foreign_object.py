"""Tests for the foreign-object inspection algorithm."""

import cv2
import numpy as np
import pytest

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.inspections.foreign_object import (
    ForeignObjectConfig,
    ForeignObjectStatus,
    inspect_foreign_objects,
    render_foreign_object_result,
)


def create_synthetic_bottle() -> np.ndarray:
    """Create a simple high-contrast bottle-like test image."""
    image = np.full((700, 200, 3), 220, dtype=np.uint8)

    cv2.rectangle(
        image,
        (40, 40),
        (160, 650),
        (180, 180, 180),
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        image,
        (40, 40),
        (160, 650),
        (60, 60, 60),
        thickness=3,
    )

    cv2.rectangle(
        image,
        (75, 10),
        (125, 50),
        (180, 180, 180),
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        image,
        (75, 10),
        (125, 50),
        (60, 60, 60),
        thickness=3,
    )

    cv2.line(
        image,
        (70, 350),
        (130, 350),
        (20, 20, 20),
        thickness=5,
    )

    return image


def test_blank_image_returns_no_bottle() -> None:
    """A featureless image should not be treated as a bottle."""
    image = np.zeros((700, 200, 3), dtype=np.uint8)

    result = inspect_foreign_objects(image)

    assert result.status is ForeignObjectStatus.NO_BOTTLE
    assert result.bottle_detected is False
    assert result.is_defective is False
    assert result.anomaly_pixel_count == 0


def test_inspector_detects_probable_bottle_content() -> None:
    """A bottle-like image should produce masks and anomaly pixels."""
    image = create_synthetic_bottle()

    result = inspect_foreign_objects(
        image,
        ForeignObjectConfig(
            defect_pixel_threshold=100_000,
        ),
    )

    assert result.status is ForeignObjectStatus.CLEAR
    assert result.bottle_detected is True
    assert result.is_defective is False
    assert result.largest_contour_area > 0
    assert result.selected_component_count > 0
    assert result.anomaly_pixel_count > 0
    assert result.bottle_mask.shape == image.shape[:2]
    assert result.anomaly_mask.shape == image.shape[:2]


def test_inspector_marks_result_defective_above_threshold() -> None:
    """The result should be defective when its threshold is exceeded."""
    image = create_synthetic_bottle()

    result = inspect_foreign_objects(
        image,
        ForeignObjectConfig(
            defect_pixel_threshold=0,
        ),
    )

    assert result.status is ForeignObjectStatus.DEFECTIVE
    assert result.is_defective is True
    assert result.anomaly_pixel_count > 0


def test_invalid_kernel_size_is_rejected() -> None:
    """OpenCV Gaussian kernels must use positive odd dimensions."""
    with pytest.raises(ValueError, match="positive odd"):
        ForeignObjectConfig(
            gaussian_kernel_size=4,
        )


def test_invalid_image_shape_is_rejected() -> None:
    """The inspector requires a three-channel BGR image."""
    grayscale = np.zeros((700, 200), dtype=np.uint8)

    with pytest.raises(InspectionError, match="three-channel"):
        inspect_foreign_objects(grayscale)


def test_render_returns_annotated_copy() -> None:
    """Rendering should not modify the original inspection image."""
    image = create_synthetic_bottle()

    result = inspect_foreign_objects(
        image,
        ForeignObjectConfig(
            defect_pixel_threshold=0,
        ),
    )

    annotated = render_foreign_object_result(image, result)

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert not np.shares_memory(annotated, image)
    assert np.any(annotated != image)