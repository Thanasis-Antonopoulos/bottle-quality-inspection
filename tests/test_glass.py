"""Tests for glass preprocessing, feature extraction and classification."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.inspections.glass import (
    GlassDecisionConfig,
    GlassFeatureConfig,
    GlassPreprocessConfig,
    GlassStatus,
    classify_glass_features,
    extract_glass_features,
    inspect_glass,
    preprocess_glass,
    render_glass_features,
    render_glass_result,
)


def create_image() -> np.ndarray:
    """Create a deterministic synthetic BGR image."""
    horizontal = np.tile(
        np.arange(
            0,
            120,
            dtype=np.uint8,
        ),
        (
            100,
            1,
        ),
    )

    vertical = np.tile(
        np.arange(
            0,
            100,
            dtype=np.uint8,
        )[:, None],
        (
            1,
            120,
        ),
    )

    third_channel = (
        (
            horizontal.astype(np.uint16)
            + vertical.astype(np.uint16)
        )
        % 256
    ).astype(
        np.uint8
    )

    return np.dstack(
        (
            horizontal,
            vertical,
            third_channel,
        )
    )


def create_bottle_image() -> np.ndarray:
    """Create a synthetic bottle-sized image."""
    return np.zeros(
        (
            400,
            200,
            3,
        ),
        dtype=np.uint8,
    )


def create_blank_features():
    """Create a valid feature result containing no defect evidence."""
    return extract_glass_features(
        create_bottle_image()
    )


def test_preprocessing_outputs_expected_shapes() -> None:
    """Legacy preprocessing should preserve source dimensions."""
    image = create_image()

    result = preprocess_glass(
        image
    )

    expected_grayscale_shape = (
        image.shape[0],
        image.shape[1],
    )

    grayscale_outputs = [
        result.grayscale,
        result.average_blurred,
        result.average_edges,
        result.gaussian_blurred,
        result.gaussian_edges,
        result.binary_thresholded,
        result.binary_inverted,
        result.otsu_inverted,
        result.median_blurred,
        result.global_binary,
        result.adaptive_mean,
        result.adaptive_gaussian,
    ]

    for output in grayscale_outputs:
        assert (
            output.shape
            == expected_grayscale_shape
        )

    assert (
        result.otsu_masked.shape
        == image.shape
    )


def test_preprocessing_outputs_uint8_images() -> None:
    """Generated image products should remain uint8."""
    image = create_image()

    result = preprocess_glass(
        image
    )

    image_outputs = [
        result.grayscale,
        result.average_blurred,
        result.average_edges,
        result.gaussian_blurred,
        result.gaussian_edges,
        result.binary_thresholded,
        result.binary_inverted,
        result.otsu_inverted,
        result.otsu_masked,
        result.median_blurred,
        result.global_binary,
        result.adaptive_mean,
        result.adaptive_gaussian,
    ]

    for output in image_outputs:
        assert output.dtype == np.uint8


def test_otsu_threshold_is_reported() -> None:
    """The calculated Otsu threshold should be retained."""
    image = create_image()

    result = preprocess_glass(
        image
    )

    assert isinstance(
        result.otsu_threshold_value,
        float,
    )

    assert (
        0.0
        <= result.otsu_threshold_value
        <= 255.0
    )


def test_grayscale_input_is_rejected() -> None:
    """Glass preprocessing requires a BGR source image."""
    grayscale = np.zeros(
        (
            100,
            120,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        InspectionError,
        match="three-channel",
    ):
        preprocess_glass(
            grayscale
        )


def test_non_uint8_image_is_rejected() -> None:
    """Glass preprocessing should reject unsupported dtypes."""
    image = np.zeros(
        (
            100,
            120,
            3,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        InspectionError,
        match="uint8",
    ):
        preprocess_glass(
            image
        )


def test_even_gaussian_kernel_is_rejected() -> None:
    """Gaussian blur requires an odd kernel size."""
    with pytest.raises(
        ValueError,
        match="gaussian_blur_kernel_size",
    ):
        GlassPreprocessConfig(
            gaussian_blur_kernel_size=4,
        )


def test_even_median_kernel_is_rejected() -> None:
    """Median blur requires an odd kernel size."""
    with pytest.raises(
        ValueError,
        match="median_blur_kernel_size",
    ):
        GlassPreprocessConfig(
            median_blur_kernel_size=4,
        )


def test_even_adaptive_block_size_is_rejected() -> None:
    """Adaptive thresholding requires an odd block size."""
    with pytest.raises(
        ValueError,
        match="adaptive_block_size",
    ):
        GlassPreprocessConfig(
            adaptive_block_size=10,
        )


def test_blank_image_has_no_glass_features() -> None:
    """A blank bottle crop should contain no edge features."""
    result = create_blank_features()

    assert result.edge_pixel_count == 0
    assert result.edge_density == 0.0

    assert result.detected_line_count == 0
    assert result.horizontal_line_count == 0

    assert (
        result.longest_horizontal_line_length
        == 0.0
    )

    assert (
        result.total_horizontal_line_length
        == 0.0
    )

    assert result.maximum_row_edge_count == 0
    assert result.maximum_row_edge_ratio == 0.0


def test_horizontal_feature_is_detected() -> None:
    """A synthetic horizontal defect inside the body should be detected."""
    image = create_bottle_image()

    cv2.line(
        image,
        (50, 180),
        (150, 180),
        (255, 255, 255),
        2,
    )

    result = extract_glass_features(
        image
    )

    assert result.edge_pixel_count > 0
    assert result.detected_line_count > 0
    assert result.horizontal_line_count > 0

    assert (
        result.longest_horizontal_line_length
        >= 20.0
    )


def test_upper_shoulder_feature_is_excluded() -> None:
    """A horizontal edge above the body region should not be measured."""
    image = create_bottle_image()

    cv2.line(
        image,
        (50, 150),
        (150, 150),
        (255, 255, 255),
        2,
    )

    result = extract_glass_features(
        image
    )

    assert result.horizontal_line_count == 0
    assert (
        result.longest_horizontal_line_length
        == 0.0
    )
    assert (
        result.total_horizontal_line_length
        == 0.0
    )


def test_feature_density_is_normalized() -> None:
    """Feature ratios should remain between zero and one."""
    image = create_bottle_image()

    cv2.line(
        image,
        (50, 180),
        (150, 180),
        (255, 255, 255),
        2,
    )

    result = extract_glass_features(
        image
    )

    assert (
        0.0
        <= result.edge_density
        <= 1.0
    )

    assert (
        0.0
        <= result.maximum_row_edge_ratio
        <= 1.0
    )

    assert (
        result.longest_horizontal_line_ratio
        >= 0.0
    )


def test_invalid_horizontal_ratios_are_rejected() -> None:
    """The interior left boundary must precede the right boundary."""
    with pytest.raises(
        ValueError,
        match="horizontal interior ratios",
    ):
        GlassFeatureConfig(
            interior_left_ratio=0.80,
            interior_right_ratio=0.20,
        )


def test_invalid_vertical_ratios_are_rejected() -> None:
    """The interior top boundary must precede the bottom boundary."""
    with pytest.raises(
        ValueError,
        match="vertical interior ratios",
    ):
        GlassFeatureConfig(
            interior_top_ratio=0.80,
            interior_bottom_ratio=0.20,
        )


def test_invalid_horizontal_angle_is_rejected() -> None:
    """Horizontal-angle tolerance must remain meaningful."""
    with pytest.raises(
        ValueError,
        match="maximum_horizontal_angle_degrees",
    ):
        GlassFeatureConfig(
            maximum_horizontal_angle_degrees=100.0,
        )


def test_render_glass_features_returns_copy() -> None:
    """Feature rendering should preserve the source image."""
    image = create_bottle_image()

    cv2.line(
        image,
        (50, 180),
        (150, 180),
        (255, 255, 255),
        2,
    )

    original = image.copy()

    result = extract_glass_features(
        image
    )

    annotated = render_glass_features(
        image,
        result,
    )

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8

    assert not np.shares_memory(
        annotated,
        image,
    )

    assert np.array_equal(
        image,
        original,
    )

    assert not np.array_equal(
        annotated,
        image,
    )


def test_invalid_decision_signal_requirement_is_rejected() -> None:
    """The classifier has four available independent signals."""
    with pytest.raises(
        ValueError,
        match="minimum_triggered_signals",
    ):
        GlassDecisionConfig(
            minimum_triggered_signals=5,
        )


def test_clear_features_remain_clear() -> None:
    """Features below all calibrated limits should pass."""
    features = create_blank_features()

    result = classify_glass_features(
        features
    )

    assert result.status is GlassStatus.CLEAR
    assert result.is_defective is False
    assert result.triggered_signal_count == 0


def test_three_strong_signals_are_defective() -> None:
    """Three independent defect signals should reject the bottle."""
    features = create_blank_features()

    suspicious = replace(
        features,
        edge_density=0.020,
        horizontal_line_count=1,
        maximum_row_edge_ratio=0.25,
    )

    result = classify_glass_features(
        suspicious
    )

    assert result.status is GlassStatus.DEFECTIVE
    assert result.is_defective is True
    assert result.triggered_signal_count == 3


def test_two_signals_are_not_enough() -> None:
    """Two isolated indicators should not reject a bottle by default."""
    features = create_blank_features()

    suspicious = replace(
        features,
        edge_density=0.020,
        horizontal_line_count=1,
        maximum_row_edge_ratio=0.10,
    )

    result = classify_glass_features(
        suspicious
    )

    assert result.status is GlassStatus.CLEAR
    assert result.triggered_signal_count == 2


def test_decision_boundary_values_count_as_signals() -> None:
    """Decision thresholds are inclusive."""
    features = create_blank_features()

    exact_longest_length = (
        features.interior_width
        * 0.20
    )

    boundary = replace(
        features,
        edge_density=0.015,
        horizontal_line_count=1,
        longest_horizontal_line_length=(
            exact_longest_length
        ),
        maximum_row_edge_ratio=0.20,
    )

    result = classify_glass_features(
        boundary
    )

    assert result.status is GlassStatus.DEFECTIVE
    assert result.triggered_signal_count == 4


def test_inspect_glass_returns_clear_for_blank_image() -> None:
    """The full inspection path should accept a clean blank image."""
    image = create_bottle_image()

    result = inspect_glass(
        image
    )

    assert result.status is GlassStatus.CLEAR
    assert result.is_defective is False


def test_render_glass_result_returns_copy() -> None:
    """Final-decision rendering should not mutate the source image."""
    image = create_bottle_image()

    original = image.copy()

    result = inspect_glass(
        image
    )

    annotated = render_glass_result(
        image,
        result,
    )

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8

    assert not np.shares_memory(
        annotated,
        image,
    )

    assert np.array_equal(
        image,
        original,
    )

    assert not np.array_equal(
        annotated,
        image,
    )