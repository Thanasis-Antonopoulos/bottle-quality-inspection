"""Reusable frame-by-frame video-processing pipeline."""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import (
    FrameProcessingError,
    VideoReadError,
    VideoWriteError,
)
from bottle_quality_inspection.video_io import (
    VideoFrame,
    iter_video_frames,
    read_video_metadata,
)

FrameProcessor = Callable[[VideoFrame], np.ndarray]


@dataclass(frozen=True, slots=True)
class VideoProcessResult:
    """Summary produced after processing and writing a complete video."""

    input_path: Path
    output_path: Path
    frames_processed: int
    elapsed_seconds: float
    codec: str

    @property
    def processing_fps(self) -> float | None:
        """Return the achieved processing throughput."""
        if self.elapsed_seconds <= 0:
            return None

        return self.frames_processed / self.elapsed_seconds


def _prepare_output_path(
    input_path: Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Resolve and validate the destination video path."""
    destination = Path(output_path).expanduser().resolve()

    if destination == input_path:
        raise VideoWriteError("Input and output video paths must be different.")

    if destination.exists():
        if not destination.is_file():
            raise IsADirectoryError(
                f"Output path is not a file: {destination}"
            )

        if not overwrite:
            raise FileExistsError(
                "Output video already exists. "
                f"Use --overwrite to replace it: {destination}"
            )

        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)

    return destination


def _validate_processed_frame(
    frame: np.ndarray,
    *,
    frame_index: int,
    expected_width: int,
    expected_height: int,
) -> None:
    """Validate a frame returned by a processing component."""
    if not isinstance(frame, np.ndarray):
        raise FrameProcessingError(
            f"Frame processor returned a non-array value at index {frame_index}."
        )

    if frame.size == 0:
        raise FrameProcessingError(
            f"Frame processor returned an empty frame at index {frame_index}."
        )

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FrameProcessingError(
            "Processed frames must be three-channel BGR images: "
            f"index={frame_index}, shape={frame.shape}"
        )

    height, width = frame.shape[:2]

    if width != expected_width or height != expected_height:
        raise FrameProcessingError(
            "Processed frame dimensions differ from the input video: "
            f"index={frame_index}, "
            f"frame={width}x{height}, "
            f"expected={expected_width}x{expected_height}"
        )

    if frame.dtype != np.uint8:
        raise FrameProcessingError(
            "Processed frames must use uint8 pixel values: "
            f"index={frame_index}, dtype={frame.dtype}"
        )


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    processor: FrameProcessor,
    *,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> VideoProcessResult:
    """Apply a processor to every frame and write the resulting video."""
    source = Path(input_path).expanduser().resolve()
    metadata = read_video_metadata(source)

    if len(codec) != 4:
        raise ValueError("Video codec must contain exactly four characters.")

    if metadata.fps <= 0:
        raise VideoWriteError(
            f"Input video has an invalid FPS value: {metadata.fps}"
        )

    destination = _prepare_output_path(
        source,
        output_path,
        overwrite=overwrite,
    )

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(destination),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )

    frames_processed = 0
    completed = False
    started_at = perf_counter()

    try:
        if not writer.isOpened():
            raise VideoWriteError(
                f"OpenCV could not create the output video: {destination}"
            )

        for video_frame in iter_video_frames(source):
            try:
                processed_frame = processor(video_frame)
            except FrameProcessingError:
                raise
            except Exception as exc:
                raise FrameProcessingError(
                    "Frame processor failed: "
                    f"index={video_frame.index}, "
                    f"timestamp={video_frame.timestamp_seconds:.3f}"
                ) from exc

            _validate_processed_frame(
                processed_frame,
                frame_index=video_frame.index,
                expected_width=metadata.width,
                expected_height=metadata.height,
            )

            writer.write(processed_frame)
            frames_processed += 1

        if frames_processed == 0:
            raise VideoReadError(
                f"No video frames could be processed: {source}"
            )

        completed = True
    finally:
        writer.release()

        if not completed and destination.exists():
            with suppress(OSError):
                destination.unlink()

    elapsed_seconds = perf_counter() - started_at

    return VideoProcessResult(
        input_path=source,
        output_path=destination,
        frames_processed=frames_processed,
        elapsed_seconds=elapsed_seconds,
        codec=codec,
    )