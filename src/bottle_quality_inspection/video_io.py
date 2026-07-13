"""Video input, frame decoding, and metadata utilities."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import VideoOpenError, VideoReadError


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Technical metadata describing a local video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        """Return the estimated video duration when FPS is available."""
        if self.fps <= 0:
            return None

        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded video frame and its position in the video."""

    index: int
    timestamp_seconds: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class VideoScanResult:
    """Summary produced after decoding a complete video."""

    path: Path
    frames_decoded: int
    elapsed_seconds: float

    @property
    def processing_fps(self) -> float | None:
        """Return the achieved frame-decoding throughput."""
        if self.elapsed_seconds <= 0:
            return None

        return self.frames_decoded / self.elapsed_seconds


def _resolve_video_path(video_path: str | Path) -> Path:
    """Resolve and validate a local video path."""
    path = Path(video_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Video path is not a file: {path}")

    return path


def _metadata_from_capture(
    path: Path,
    capture: cv2.VideoCapture,
) -> VideoMetadata:
    """Extract validated metadata from an open video capture."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise VideoOpenError(
            f"Video has invalid dimensions: width={width}, height={height}"
        )

    return VideoMetadata(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
    )


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read and validate metadata from a local video."""
    path = _resolve_video_path(video_path)
    capture = cv2.VideoCapture(str(path))

    try:
        if not capture.isOpened():
            raise VideoOpenError(f"OpenCV could not open the video: {path}")

        return _metadata_from_capture(path, capture)
    finally:
        capture.release()


def iter_video_frames(video_path: str | Path) -> Iterator[VideoFrame]:
    """Decode a local video and yield validated frames in order."""
    path = _resolve_video_path(video_path)
    capture = cv2.VideoCapture(str(path))

    try:
        if not capture.isOpened():
            raise VideoOpenError(f"OpenCV could not open the video: {path}")

        metadata = _metadata_from_capture(path, capture)
        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame is None or frame.size == 0:
                raise VideoReadError(
                    f"OpenCV returned an empty frame at index {frame_index}: {path}"
                )

            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))

            if timestamp_ms > 0:
                timestamp_seconds = timestamp_ms / 1000
            elif metadata.fps > 0:
                timestamp_seconds = frame_index / metadata.fps
            else:
                timestamp_seconds = 0.0

            yield VideoFrame(
                index=frame_index,
                timestamp_seconds=timestamp_seconds,
                image=frame,
            )

            frame_index += 1
    finally:
        capture.release()


def scan_video(video_path: str | Path) -> VideoScanResult:
    """Decode a complete video and return a processing summary."""
    path = _resolve_video_path(video_path)
    started_at = perf_counter()
    frames_decoded = 0

    for _ in iter_video_frames(path):
        frames_decoded += 1

    elapsed_seconds = perf_counter() - started_at

    if frames_decoded == 0:
        raise VideoReadError(f"No video frames could be decoded: {path}")

    return VideoScanResult(
        path=path,
        frames_decoded=frames_decoded,
        elapsed_seconds=elapsed_seconds,
    )