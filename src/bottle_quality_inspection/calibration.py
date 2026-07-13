"""Calibration preview utilities."""

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2

from bottle_quality_inspection.exceptions import ImageWriteError, VideoReadError
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class RoiPreviewResult:
    """Summary produced after saving an ROI calibration preview."""

    input_path: Path
    output_path: Path
    frame_index: int
    timestamp_seconds: float
    region: RegionOfInterest


def _resolve_preview_output_path(
    input_path: Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Resolve and validate an ROI preview output path."""
    output = Path(output_path).expanduser().resolve()

    if output == input_path:
        raise ImageWriteError("Input video and output image paths must be different.")

    if output.exists():
        if not output.is_file():
            raise IsADirectoryError(f"Output path is not a file: {output}")

        if not overwrite:
            raise FileExistsError(
                f"Output image already exists. Use --overwrite to replace it: {output}"
            )

        output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)

    return output


def save_roi_preview(
    video_path: str | Path,
    output_path: str | Path,
    region: RegionOfInterest,
    *,
    frame_index: int = 0,
    thickness: int = 2,
    show_label: bool = True,
    overwrite: bool = False,
) -> RoiPreviewResult:
    """Draw an ROI on a selected video frame and save it as an image."""
    if frame_index < 0:
        raise ValueError("Frame index cannot be negative.")

    source = Path(video_path).expanduser().resolve()
    destination = _resolve_preview_output_path(
        source,
        output_path,
        overwrite=overwrite,
    )

    selected_frame = None
    frames = iter_video_frames(source)

    try:
        for video_frame in frames:
            if video_frame.index == frame_index:
                selected_frame = video_frame
                break
    finally:
        close = getattr(frames, "close", None)

        if close is not None:
            close()

    if selected_frame is None:
        raise VideoReadError(
            f"Frame index {frame_index} could not be decoded from video: {source}"
        )

    annotated = region.draw(
        selected_frame.image,
        thickness=thickness,
        show_label=show_label,
    )

    try:
        written = cv2.imwrite(str(destination), annotated)
    except cv2.error as exc:
        raise ImageWriteError(
            f"OpenCV could not write the ROI preview image: {destination}"
        ) from exc

    if not written:
        if destination.exists():
            with suppress(OSError):
                destination.unlink()

        raise ImageWriteError(
            f"OpenCV could not write the ROI preview image: {destination}"
        )

    return RoiPreviewResult(
        input_path=source,
        output_path=destination,
        frame_index=selected_frame.index,
        timestamp_seconds=selected_frame.timestamp_seconds,
        region=region,
    )