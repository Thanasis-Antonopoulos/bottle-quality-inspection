"""Tests for the cork template-matching inspection."""

import cv2
import numpy as np
import pytest

from bottle_quality_inspection.exceptions import InspectionError
from bottle_quality_inspection.inspections import cork
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    CorkStatus,
    inspect_cork,
    render_cork_result,
)


def create_image(
    height: int = 100,
    width: int = 120,
) -> np.ndarray:
    """Create a valid synthetic BGR image."""
    return np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )


def create_template() -> np.ndarray:
    """Create a valid synthetic cork template."""
    return np.zeros(
        (
            20,
            30,
            3,
        ),
        dtype=np.uint8,
    )


def patch_match_result(
    monkeypatch: pytest.MonkeyPatch,
    result_matrix: np.ndarray,
) -> None:
    """Replace OpenCV template matching with a known matrix."""
    monkeypatch.setattr(
        cork.cv2,
        "matchTemplate",
        lambda *args, **kwargs: (
            result_matrix.copy()
        ),
    )


def test_score_above_threshold_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A score above 0.80 should pass."""
    image = create_image()
    template = create_template()

    result_matrix = np.array(
        [
            [0.20, 0.30],
            [0.40, 0.91],
        ],
        dtype=np.float32,
    )

    patch_match_result(
        monkeypatch,
        result_matrix,
    )

    result = inspect_cork(
        image,
        template,
    )

    assert result.status is CorkStatus.CLEAR
    assert result.is_defective is False
    assert result.match_found is True

    assert result.maximum_score == pytest.approx(
        0.91
    )


def test_score_below_threshold_is_defective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A score below 0.80 should fail."""
    image = create_image()
    template = create_template()

    result_matrix = np.array(
        [
            [0.20, 0.30],
            [0.40, 0.79],
        ],
        dtype=np.float32,
    )

    patch_match_result(
        monkeypatch,
        result_matrix,
    )

    result = inspect_cork(
        image,
        template,
    )

    assert (
        result.status
        is CorkStatus.DEFECTIVE
    )

    assert result.is_defective is True
    assert result.match_found is False


def test_score_equal_to_threshold_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy >= 0.80 threshold should include exactly 0.80."""
    image = create_image()
    template = create_template()

    result_matrix = np.array(
        [
            [0.30, 0.80],
        ],
        dtype=np.float32,
    )

    patch_match_result(
        monkeypatch,
        result_matrix,
    )

    result = inspect_cork(
        image,
        template,
    )

    assert result.status is CorkStatus.CLEAR
    assert result.match_found is True


def test_best_match_location_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The maximum template-match location should be retained."""
    image = create_image()
    template = create_template()

    result_matrix = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.95, 0.50],
        ],
        dtype=np.float32,
    )

    patch_match_result(
        monkeypatch,
        result_matrix,
    )

    result = inspect_cork(
        image,
        template,
    )

    assert result.maximum_location == (
        1,
        1,
    )

    assert result.minimum_location == (
        0,
        0,
    )

    assert result.bounding_box == (
        1,
        1,
        30,
        20,
    )


def test_multiple_threshold_matches_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All result locations meeting the threshold should be counted."""
    image = create_image()
    template = create_template()

    result_matrix = np.array(
        [
            [0.81, 0.50],
            [0.85, 0.92],
        ],
        dtype=np.float32,
    )

    patch_match_result(
        monkeypatch,
        result_matrix,
    )

    result = inspect_cork(
        image,
        template,
    )

    assert (
        result.matches_at_or_above_threshold
        == 3
    )


def test_template_larger_than_image_is_rejected() -> None:
    """OpenCV cannot match a template larger than its source."""
    image = create_image(
        height=50,
        width=50,
    )

    template = create_image(
        height=60,
        width=60,
    )

    with pytest.raises(
        InspectionError,
        match="larger",
    ):
        inspect_cork(
            image,
            template,
        )


def test_invalid_image_shape_is_rejected() -> None:
    """The cork inspector requires BGR source images."""
    grayscale = np.zeros(
        (
            100,
            120,
        ),
        dtype=np.uint8,
    )

    template = create_template()

    with pytest.raises(
        InspectionError,
        match="three-channel",
    ):
        inspect_cork(
            grayscale,
            template,
        )


def test_invalid_match_threshold_is_rejected() -> None:
    """Template-match thresholds must be normalized scores."""
    with pytest.raises(
        ValueError,
        match="minimum_match_score",
    ):
        CorkConfig(
            minimum_match_score=1.20,
        )


def test_unsupported_method_is_rejected() -> None:
    """The legacy baseline should not silently change methods."""
    with pytest.raises(
        ValueError,
        match="TM_CCOEFF_NORMED",
    ):
        CorkConfig(
            method="TM_SQDIFF",
        )


def test_legacy_opencv_method_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implementation should use TM_CCOEFF_NORMED."""
    image = create_image()
    template = create_template()

    captured: dict[str, int] = {}

    def fake_match_template(
        source: np.ndarray,
        template_image: np.ndarray,
        method: int,
    ) -> np.ndarray:
        captured["method"] = method

        return np.array(
            [[0.90]],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        cork.cv2,
        "matchTemplate",
        fake_match_template,
    )

    inspect_cork(
        image,
        template,
    )

    assert (
        captured["method"]
        == cv2.TM_CCOEFF_NORMED
    )


def test_render_returns_annotated_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering should not modify the source image."""
    image = create_image()
    template = create_template()

    patch_match_result(
        monkeypatch,
        np.array(
            [[0.90]],
            dtype=np.float32,
        ),
    )

    result = inspect_cork(
        image,
        template,
    )

    annotated = render_cork_result(
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