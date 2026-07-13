"""Video input and metadata utilities."""

from dataclasses import dataclass
from pathlib import Path

import cv2

from bottle_quality_inspection.exceptions import VideoOpenError


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


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read and validate metadata from a local video.

    Args:
        video_path: Path to the input video.

    Returns:
        Validated video metadata.

    Raises:
        FileNotFoundError: If the input path does not exist.
        IsADirectoryError: If the input path points to a directory.
        VideoOpenError: If OpenCV cannot open the video or read its dimensions.
    """
    path = Path(video_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Video path is not a file: {path}")

    capture = cv2.VideoCapture(str(path))

    try:
        if not capture.isOpened():
            raise VideoOpenError(f"OpenCV could not open the video: {path}")

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
    finally:
        capture.release()