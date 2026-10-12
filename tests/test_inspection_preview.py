"""Tests for inspection preview generation."""

from pathlib import Path

import numpy as np
import pytest

from bottle_quality_inspection import inspection_preview
from bottle_quality_inspection.exceptions import ImageWriteError, VideoReadError
from bottle_quality_inspection.inspections.foreign_object import (
    ForeignObjectStatus,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import VideoFrame


def create_source_video(tmp_path: Path) -> Path:
    """Create a placeholder video path."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    return source


def create_region() -> RegionOfInterest:
    """Create a valid region inside the synthetic frame."""
    return RegionOfInterest(
        name="foreign-object",
        x=50,
        y=20,
        width=80,
        height=60,
    )


def create_video_frame() -> VideoFrame:
    """Create one synthetic video frame."""
    return VideoFrame(
        index=3,
        timestamp_seconds=0.12,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )


def test_save_foreign_object_preview_writes_annotated_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The selected ROI should be inspected and saved on the full frame."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "preview.jpg"

    written_images: list[np.ndarray] = []

    monkeypatch.setattr(
        inspection_preview,
        "iter_video_frames",
        lambda _: iter([create_video_frame()]),
    )

    monkeypatch.setattr(
        inspection_preview.cv2,
        "imwrite",
        lambda _, image: written_images.append(image.copy()) or True,
    )

    result = inspection_preview.save_foreign_object_preview(
        source,
        destination,
        create_region(),
        frame_index=3,
    )

    assert result.input_path == source.resolve()
    assert result.output_path == destination.resolve()
    assert result.frame_index == 3
    assert result.timestamp_seconds == pytest.approx(0.12)
    assert result.inspection.status is ForeignObjectStatus.NO_BOTTLE
    assert len(written_images) == 1
    assert written_images[0].shape == (100, 200, 3)
    assert np.any(written_images[0] != 0)


def test_foreign_object_preview_rejects_negative_frame_index(
    tmp_path: Path,
) -> None:
    """Negative video-frame indexes should be rejected."""
    source = create_source_video(tmp_path)

    with pytest.raises(ValueError, match="cannot be negative"):
        inspection_preview.save_foreign_object_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
            frame_index=-1,
        )


def test_foreign_object_preview_rejects_missing_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unavailable video-frame index should raise a clear error."""
    source = create_source_video(tmp_path)

    monkeypatch.setattr(
        inspection_preview,
        "iter_video_frames",
        lambda _: iter([]),
    )

    with pytest.raises(VideoReadError, match="could not be decoded"):
        inspection_preview.save_foreign_object_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
            frame_index=50,
        )


def test_foreign_object_preview_protects_existing_output(
    tmp_path: Path,
) -> None:
    """An existing image should not be replaced without permission."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "preview.jpg"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        inspection_preview.save_foreign_object_preview(
            source,
            destination,
            create_region(),
        )


def test_foreign_object_preview_rejects_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An OpenCV image-write failure should raise ImageWriteError."""
    source = create_source_video(tmp_path)

    monkeypatch.setattr(
        inspection_preview,
        "iter_video_frames",
        lambda _: iter([create_video_frame()]),
    )

    monkeypatch.setattr(
        inspection_preview.cv2,
        "imwrite",
        lambda *_: False,
    )

    with pytest.raises(ImageWriteError, match="could not write"):
        inspection_preview.save_foreign_object_preview(
            source,
            tmp_path / "preview.jpg",
            create_region(),
            frame_index=3,
        )