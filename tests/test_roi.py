"""Tests for region-of-interest validation and extraction."""

import numpy as np
import pytest

from bottle_quality_inspection.exceptions import RegionOfInterestError
from bottle_quality_inspection.roi import RegionOfInterest


def test_roi_extracts_expected_image_area() -> None:
    """The extracted image should match the configured region dimensions."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:60, 50:130] = 120

    region = RegionOfInterest(
        name="inspection-area",
        x=50,
        y=20,
        width=80,
        height=40,
    )

    extracted = region.extract(frame)

    assert extracted.shape == (40, 80, 3)
    assert np.all(extracted == 120)


def test_roi_extract_returns_independent_copy() -> None:
    """Changing an extracted region should not modify the source frame."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    region = RegionOfInterest(
        name="inspection-area",
        x=50,
        y=20,
        width=80,
        height=40,
    )

    extracted = region.extract(frame)
    extracted[:] = 255

    assert np.all(frame == 0)


def test_roi_draw_returns_annotated_copy() -> None:
    """Drawing should annotate a copy without changing the source frame."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    region = RegionOfInterest(
        name="inspection-area",
        x=50,
        y=20,
        width=80,
        height=40,
    )

    annotated = region.draw(
        frame,
        show_label=False,
    )

    assert np.all(frame == 0)
    assert np.any(annotated != 0)
    assert annotated.shape == frame.shape


def test_roi_rejects_negative_coordinates() -> None:
    """Negative image coordinates should be rejected."""
    with pytest.raises(RegionOfInterestError, match="cannot be negative"):
        RegionOfInterest(
            name="invalid",
            x=-1,
            y=20,
            width=80,
            height=40,
        )


def test_roi_rejects_zero_dimensions() -> None:
    """A region must have positive width and height."""
    with pytest.raises(RegionOfInterestError, match="greater than zero"):
        RegionOfInterest(
            name="invalid",
            x=10,
            y=20,
            width=0,
            height=40,
        )


def test_roi_rejects_empty_name() -> None:
    """Every region should have a meaningful name."""
    with pytest.raises(RegionOfInterestError, match="cannot be empty"):
        RegionOfInterest(
            name=" ",
            x=10,
            y=20,
            width=80,
            height=40,
        )


def test_roi_rejects_out_of_bounds_region() -> None:
    """A region extending past the frame boundary should be rejected."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    region = RegionOfInterest(
        name="outside-frame",
        x=150,
        y=20,
        width=80,
        height=40,
    )

    with pytest.raises(RegionOfInterestError, match="exceeds frame boundaries"):
        region.extract(frame)


def test_roi_rejects_empty_frame() -> None:
    """An empty NumPy image should be rejected."""
    region = RegionOfInterest(
        name="inspection-area",
        x=0,
        y=0,
        width=10,
        height=10,
    )

    with pytest.raises(RegionOfInterestError, match="cannot be empty"):
        region.extract(np.array([], dtype=np.uint8))