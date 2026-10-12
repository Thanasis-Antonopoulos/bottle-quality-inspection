"""Tests for the liquid-level inspection algorithm."""

import numpy as np
import pytest

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.inspections import liquid_level
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    LiquidLevelStatus,
    inspect_liquid_level,
    render_liquid_level_result,
)


def create_bottle_roi() -> np.ndarray:
    """Create a valid synthetic bottle ROI."""
    return np.zeros(
        (700, 200, 3),
        dtype=np.uint8,
    )


def make_lines(
    *coordinates: tuple[int, int, int, int],
) -> np.ndarray:
    """Create an OpenCV-style Hough line array."""
    return np.array(
        [
            [[x1, y1, x2, y2]]
            for x1, y1, x2, y2
            in coordinates
        ],
        dtype=np.int32,
    )


def test_no_hough_lines_returns_no_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No detected Hough lines should produce NO_LEVEL."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: None,
    )

    result = inspect_liquid_level(
        image
    )

    assert result.status is LiquidLevelStatus.NO_LEVEL
    assert result.level_detected is False
    assert result.is_defective is False
    assert result.relative_level is None
    assert result.absolute_level is None
    assert result.detected_line_count == 0


def test_legacy_opencv_conversion_code_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2022 implementation used cvtColor with conversion code 0."""
    image = create_bottle_roi()

    original_cvt_color = (
        liquid_level.cv2.cvtColor
    )

    captured: dict[str, int] = {}

    def capture_conversion_code(
        source: np.ndarray,
        code: int,
    ) -> np.ndarray:
        captured["code"] = code

        return original_cvt_color(
            source,
            code,
        )

    monkeypatch.setattr(
        liquid_level.cv2,
        "cvtColor",
        capture_conversion_code,
    )

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: None,
    )

    inspect_liquid_level(
        image
    )

    assert captured["code"] == 0


def test_level_inside_legacy_range_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected level between 225 and 247 should pass."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: make_lines(
            (10, 40, 180, 40),
        ),
    )

    result = inspect_liquid_level(
        image
    )

    assert result.status is LiquidLevelStatus.CLEAR
    assert result.level_detected is True
    assert result.is_defective is False

    assert result.relative_level == pytest.approx(
        40.0
    )

    assert result.absolute_level == pytest.approx(
        232.0
    )


def test_level_below_legacy_range_is_defective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A liquid level below the accepted range should fail."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: make_lines(
            (10, 20, 180, 20),
        ),
    )

    result = inspect_liquid_level(
        image
    )

    assert (
        result.status
        is LiquidLevelStatus.DEFECTIVE
    )

    assert result.is_defective is True

    assert result.absolute_level == pytest.approx(
        212.0
    )


def test_level_above_legacy_range_is_defective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A liquid level above the accepted range should fail."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: make_lines(
            (10, 60, 180, 60),
        ),
    )

    result = inspect_liquid_level(
        image
    )

    assert (
        result.status
        is LiquidLevelStatus.DEFECTIVE
    )

    assert result.absolute_level == pytest.approx(
        252.0
    )


def test_lowest_detected_line_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy algorithm should select the greatest average Y."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: make_lines(
            (10, 15, 180, 15),
            (10, 35, 180, 35),
            (10, 43, 180, 43),
        ),
    )

    result = inspect_liquid_level(
        image
    )

    assert result.detected_line_count == 3

    assert result.relative_level == pytest.approx(
        43.0
    )

    assert result.absolute_level == pytest.approx(
        235.0
    )

    assert result.status is LiquidLevelStatus.CLEAR


def test_invalid_image_shape_is_rejected() -> None:
    """Liquid-level inspection requires a BGR image."""
    grayscale = np.zeros(
        (700, 200),
        dtype=np.uint8,
    )

    with pytest.raises(
        InspectionError,
        match="three-channel",
    ):
        inspect_liquid_level(
            grayscale
        )


def test_invalid_acceptable_range_is_rejected() -> None:
    """The minimum level must be below the maximum."""
    with pytest.raises(
        ValueError,
        match="acceptable_minimum",
    ):
        LiquidLevelConfig(
            acceptable_minimum=247.0,
            acceptable_maximum=225.0,
        )


def test_render_returns_annotated_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering should preserve the original ROI."""
    image = create_bottle_roi()

    monkeypatch.setattr(
        liquid_level.cv2,
        "HoughLinesP",
        lambda *args, **kwargs: make_lines(
            (10, 40, 180, 40),
        ),
    )

    result = inspect_liquid_level(
        image
    )

    annotated = render_liquid_level_result(
        image,
        result,
    )

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8

    assert not np.shares_memory(
        annotated,
        image,
    )

    assert np.all(
        image == 0
    )

    assert np.any(
        annotated != 0
    )