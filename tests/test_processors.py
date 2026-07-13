"""Tests for reusable frame processors."""

import numpy as np

from bottle_quality_inspection.processors import RoiOverlayProcessor
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import VideoFrame


def test_roi_overlay_processor_draws_region_on_frame_copy() -> None:
    """The ROI processor should annotate without changing the input frame."""
    source_image = np.zeros((100, 200, 3), dtype=np.uint8)

    video_frame = VideoFrame(
        index=0,
        timestamp_seconds=0.0,
        image=source_image,
    )

    region = RegionOfInterest(
        name="inspection-area",
        x=50,
        y=20,
        width=80,
        height=40,
    )

    processor = RoiOverlayProcessor(
        region=region,
        show_label=False,
    )

    processed = processor(video_frame)

    assert processed.shape == source_image.shape
    assert processed.dtype == np.uint8
    assert np.all(source_image == 0)
    assert np.any(processed != 0)