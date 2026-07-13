"""Tests for ROI calibration preview generation."""

from pathlib import Path

import numpy as np
import pytest

from bottle_quality_inspection import calibration
from bottle_quality_inspection.exceptions import ImageWriteError, VideoReadError
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import VideoFrame


def create_region() -> RegionOfInterest:
    """Create a valid test inspection region."""
    return RegionOfInterest(
        name="inspection-area",
        x=50,
        y=20,
        width=80,
        height=40,
    )


def create_source_video(tmp_path: Path) -> Path:
    """Create a placeholder video path."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    return source


def test_save_roi_preview_writes_annotated_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A selected frame should be annotated and passed to OpenCV."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "preview.jpg"

    frame = VideoFrame(
        index=3,
        timestamp_seconds=0.12,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )

    written_images: list[np.ndarray] = []

    monkeypatch.setattr(
        calibration,
        "iter_video_frames",
        lambda _: iter([frame]),
    )
    monkeypatch.setattr(
        calibration.cv2,
        "imwrite",
        lambda _, image: written_images.append(image.copy()) or True,
    )

    result = calibration.save_roi_preview(
        source,
        destination,
        create_region(),
        frame_index=3,
    )

    assert result.input_path == source.resolve()
    assert result.output_path == destination.resolve()
    assert result.frame_index == 3
    assert result.timestamp_seconds == pytest.approx(0.12)
    assert len(written_images) == 1
    assert np.any(written_images[0] != 0)


def test_save_roi_preview_rejects_negative_frame_index(
    tmp_path: Path,
) -> None:
    """Negative frame indexes should be rejected."""
    source = create_source_video(tmp_path)

    with pytest.raises(ValueError, match="cannot be negative"):
        calibration.save_roi_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
            frame_index=-1,
        )


def test_save_roi_preview_rejects_missing_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unavailable frame index should raise a clear error."""
    source = create_source_video(tmp_path)

    monkeypatch.setattr(
        calibration,
        "iter_video_frames",
        lambda _: iter([]),
    )

    with pytest.raises(VideoReadError, match="could not be decoded"):
        calibration.save_roi_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
            frame_index=10,
        )


def test_save_roi_preview_rejects_existing_output(
    tmp_path: Path,
) -> None:
    """Existing previews should not be replaced without permission."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "preview.jpg"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        calibration.save_roi_preview(
            source,
            destination,
            create_region(),
        )


def test_save_roi_preview_rejects_image_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An OpenCV image-write failure should raise ImageWriteError."""
    source = create_source_video(tmp_path)

    frame = VideoFrame(
        index=0,
        timestamp_seconds=0.0,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(
        calibration,
        "iter_video_frames",
        lambda _: iter([frame]),
    )
    monkeypatch.setattr(
        calibration.cv2,
        "imwrite",
        lambda *_: False,
    )

    with pytest.raises(ImageWriteError, match="could not write"):
        calibration.save_roi_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
        )