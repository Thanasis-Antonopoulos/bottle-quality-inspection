"""Tests for validated output-video writing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_quality_inspection import video_io
from bottle_quality_inspection.exceptions import VideoWriteError


class FakeCapture:
    """Configurable source-video capture used by writer tests."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        """Report that the fake source opened."""
        return True

    def get(self, property_id: int) -> float:
        """Return source-video metadata."""
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: float(len(self.frames)),
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return source frames sequentially."""
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        """Record that the source was released."""
        self.released = True


class FakeWriter:
    """Configurable OpenCV writer replacement."""

    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        """Return the configured writer state."""
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        """Record one written frame."""
        self.frames.append(frame.copy())

    def release(self) -> None:
        """Record that the writer was released."""
        self.released = True


def create_source_video(tmp_path: Path) -> Path:
    """Create a placeholder input path."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    return source


def test_copy_video_writes_all_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All decoded frames should be passed to the output writer."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "output.mp4"

    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640, 3), dtype=np.uint8),
        np.full((480, 640, 3), 2, dtype=np.uint8),
    ]

    fake_capture = FakeCapture(frames)
    fake_writer = FakeWriter()

    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(video_io.cv2, "VideoWriter", lambda *args: fake_writer)
    monkeypatch.setattr(video_io.cv2, "VideoWriter_fourcc", lambda *args: 1234)

    result = video_io.copy_video(source, destination)

    assert result.input_path == source.resolve()
    assert result.output_path == destination.resolve()
    assert result.frames_written == 3
    assert result.codec == "mp4v"
    assert result.processing_fps is not None
    assert len(fake_writer.frames) == 3
    assert fake_capture.released is True
    assert fake_writer.released is True


def test_copy_video_rejects_existing_output(tmp_path: Path) -> None:
    """An existing output should not be replaced without permission."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "output.mp4"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        video_io.copy_video(source, destination)


def test_copy_video_rejects_same_input_and_output(tmp_path: Path) -> None:
    """The source video must never be overwritten in place."""
    source = create_source_video(tmp_path)

    with pytest.raises(VideoWriteError, match="must be different"):
        video_io.copy_video(source, source, overwrite=True)


def test_copy_video_rejects_writer_that_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed OpenCV writer should raise a clear exception."""
    source = create_source_video(tmp_path)
    destination = tmp_path / "output.mp4"

    fake_capture = FakeCapture(
        [np.zeros((480, 640, 3), dtype=np.uint8)]
    )
    fake_writer = FakeWriter(opened=False)

    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(video_io.cv2, "VideoWriter", lambda *args: fake_writer)
    monkeypatch.setattr(video_io.cv2, "VideoWriter_fourcc", lambda *args: 1234)

    with pytest.raises(VideoWriteError, match="could not create"):
        video_io.copy_video(source, destination)

    assert fake_capture.released is True
    assert fake_writer.released is True


def test_copy_video_rejects_invalid_codec(
    tmp_path: Path,
) -> None:
    """The OpenCV codec must contain exactly four characters."""
    source = create_source_video(tmp_path)

    with pytest.raises(ValueError, match="exactly four"):
        video_io.copy_video(
            source,
            tmp_path / "output.mp4",
            codec="invalid",
        )