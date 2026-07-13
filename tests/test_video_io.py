"""Tests for video input and metadata handling."""

from pathlib import Path

import cv2
import pytest

from bottle_quality_inspection import video_io
from bottle_quality_inspection.exceptions import VideoOpenError


class FakeVideoCapture:
    """Small OpenCV VideoCapture replacement used by unit tests."""

    def __init__(
        self,
        *,
        opened: bool,
        width: float = 0,
        height: float = 0,
        fps: float = 0,
        frame_count: float = 0,
    ) -> None:
        self.opened = opened
        self.released = False
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
        }

    def isOpened(self) -> bool:
        """Return the configured open state."""
        return self.opened

    def get(self, property_id: int) -> float:
        """Return a configured video property."""
        return self.values.get(property_id, 0)

    def release(self) -> None:
        """Record that the resource was released."""
        self.released = True


def test_read_video_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata should be read and the capture resource released."""
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")

    fake_capture = FakeVideoCapture(
        opened=True,
        width=1920,
        height=1080,
        fps=25,
        frame_count=250,
    )

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    metadata = video_io.read_video_metadata(video_path)

    assert metadata.path == video_path.resolve()
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == pytest.approx(25.0)
    assert metadata.frame_count == 250
    assert metadata.duration_seconds == pytest.approx(10.0)
    assert fake_capture.released is True


def test_missing_video_raises_file_not_found(tmp_path: Path) -> None:
    """A missing input video should raise a clear exception."""
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        video_io.read_video_metadata(missing_path)


def test_video_that_cannot_open_raises_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unreadable video should raise VideoOpenError."""
    video_path = tmp_path / "invalid.mp4"
    video_path.write_bytes(b"not a valid video")

    fake_capture = FakeVideoCapture(opened=False)

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    with pytest.raises(VideoOpenError, match="could not open"):
        video_io.read_video_metadata(video_path)

    assert fake_capture.released is True


def test_invalid_video_dimensions_raise_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A video with invalid dimensions should be rejected."""
    video_path = tmp_path / "invalid-dimensions.mp4"
    video_path.write_bytes(b"placeholder")

    fake_capture = FakeVideoCapture(
        opened=True,
        width=0,
        height=1080,
        fps=25,
        frame_count=250,
    )

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    with pytest.raises(VideoOpenError, match="invalid dimensions"):
        video_io.read_video_metadata(video_path)

    assert fake_capture.released is True