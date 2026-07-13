"""Tests for frame-by-frame video decoding."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_quality_inspection import video_io
from bottle_quality_inspection.exceptions import VideoReadError


class FakeFrameCapture:
    """Video capture replacement with configurable decoded frames."""

    def __init__(
        self,
        frames: list[np.ndarray],
        timestamps_ms: list[float] | None = None,
    ) -> None:
        self.frames = frames
        self.timestamps_ms = timestamps_ms or [
            index * 40.0 for index in range(len(frames))
        ]
        self.position = 0
        self.current_timestamp_ms = 0.0
        self.released = False

    def isOpened(self) -> bool:
        """Report that the fake capture opened successfully."""
        return True

    def get(self, property_id: int) -> float:
        """Return configured metadata and frame position values."""
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: float(len(self.frames)),
            cv2.CAP_PROP_POS_MSEC: self.current_timestamp_ms,
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return frames sequentially until the fake video ends."""
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]
        self.current_timestamp_ms = self.timestamps_ms[self.position]
        self.position += 1

        return True, frame.copy()

    def release(self) -> None:
        """Record that the fake capture was released."""
        self.released = True


def create_placeholder_video(tmp_path: Path) -> Path:
    """Create a placeholder path accepted by path validation."""
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"placeholder")
    return video_path


def test_iter_video_frames_decodes_frames_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decoded frames should contain indexes, timestamps, and images."""
    video_path = create_placeholder_video(tmp_path)

    source_frames = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640, 3), dtype=np.uint8),
        np.full((480, 640, 3), 2, dtype=np.uint8),
    ]

    fake_capture = FakeFrameCapture(
        source_frames,
        timestamps_ms=[0.0, 40.0, 80.0],
    )

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    decoded_frames = list(video_io.iter_video_frames(video_path))

    assert [frame.index for frame in decoded_frames] == [0, 1, 2]
    assert decoded_frames[0].timestamp_seconds == pytest.approx(0.0)
    assert decoded_frames[1].timestamp_seconds == pytest.approx(0.04)
    assert decoded_frames[2].timestamp_seconds == pytest.approx(0.08)
    assert decoded_frames[0].image.shape == (480, 640, 3)
    assert fake_capture.released is True


def test_scan_video_returns_processing_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scanning should count all frames and report processing performance."""
    video_path = create_placeholder_video(tmp_path)

    fake_capture = FakeFrameCapture(
        [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
    )

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    result = video_io.scan_video(video_path)

    assert result.path == video_path.resolve()
    assert result.frames_decoded == 3
    assert result.elapsed_seconds > 0
    assert result.processing_fps is not None
    assert result.processing_fps > 0
    assert fake_capture.released is True


def test_scan_video_rejects_video_without_decodable_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A video without any decodable frame should be rejected."""
    video_path = create_placeholder_video(tmp_path)
    fake_capture = FakeFrameCapture([])

    monkeypatch.setattr(
        video_io.cv2,
        "VideoCapture",
        lambda _: fake_capture,
    )

    with pytest.raises(VideoReadError, match="No video frames"):
        video_io.scan_video(video_path)

    assert fake_capture.released is True