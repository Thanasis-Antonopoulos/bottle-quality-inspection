"""Preview utilities for bottle inspection algorithms."""

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2

from bottle_quality_inspection.exceptions import (
    ImageWriteError,
    InspectionError,
    VideoReadError,
)
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    CorkResult,
    inspect_cork,
    render_cork_result,
)
from bottle_quality_inspection.inspections.foreign_object import (
    ForeignObjectConfig,
    ForeignObjectResult,
    ForeignObjectStatus,
    inspect_foreign_objects,
    render_foreign_object_result,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    LiquidLevelResult,
    LiquidLevelStatus,
    inspect_liquid_level,
    render_liquid_level_result,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import VideoFrame, iter_video_frames


@dataclass(frozen=True, slots=True)
class ForeignObjectPreviewResult:
    """Summary produced after saving a foreign-object preview."""

    input_path: Path
    output_path: Path
    frame_index: int
    timestamp_seconds: float
    region: RegionOfInterest
    inspection: ForeignObjectResult


@dataclass(frozen=True, slots=True)
class LiquidLevelPreviewResult:
    """Summary produced after saving a liquid-level preview."""

    input_path: Path
    output_path: Path
    frame_index: int
    timestamp_seconds: float
    region: RegionOfInterest
    inspection: LiquidLevelResult


@dataclass(frozen=True, slots=True)
class CorkPreviewResult:
    """Summary produced after saving a cork preview."""

    input_path: Path
    output_path: Path
    template_path: Path
    frame_index: int
    timestamp_seconds: float
    inspection: CorkResult


def _resolve_output_image_path(
    input_path: Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Resolve and validate an inspection-preview destination."""
    destination = Path(output_path).expanduser().resolve()

    if destination == input_path:
        raise ImageWriteError(
            "Input video and output image paths must be different."
        )

    if destination.exists():
        if not destination.is_file():
            raise IsADirectoryError(
                f"Output path is not a file: {destination}"
            )

        if not overwrite:
            raise FileExistsError(
                "Output image already exists. "
                f"Use --overwrite to replace it: {destination}"
            )

        destination.unlink()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return destination


def _select_video_frame(
    source: Path,
    frame_index: int,
) -> VideoFrame:
    """Read one frame from a video by zero-based frame index."""
    if frame_index < 0:
        raise ValueError(
            "Frame index cannot be negative."
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
            f"Frame index {frame_index} could not be decoded "
            f"from video: {source}"
        )

    return selected_frame


def _read_bgr_image(
    image_path: str | Path,
    *,
    description: str,
) -> tuple[Path, object]:
    """Load one BGR image from disk."""
    source = Path(image_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {source}"
        )

    if not source.is_file():
        raise IsADirectoryError(
            f"{description} is not a file: {source}"
        )

    image = cv2.imread(
        str(source),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise InspectionError(
            f"OpenCV could not read {description.lower()}: {source}"
        )

    return source, image


def _foreign_object_status_color(
    status: ForeignObjectStatus,
) -> tuple[int, int, int]:
    """Return BGR annotation color for foreign-object status."""
    colors = {
        ForeignObjectStatus.NO_BOTTLE: (0, 255, 255),
        ForeignObjectStatus.CLEAR: (0, 255, 0),
        ForeignObjectStatus.DEFECTIVE: (0, 0, 255),
    }

    return colors[status]


def _liquid_level_status_color(
    status: LiquidLevelStatus,
) -> tuple[int, int, int]:
    """Return BGR annotation color for liquid-level status."""
    colors = {
        LiquidLevelStatus.NO_LEVEL: (0, 255, 255),
        LiquidLevelStatus.CLEAR: (0, 255, 0),
        LiquidLevelStatus.DEFECTIVE: (0, 0, 255),
    }

    return colors[status]


def _write_preview_image(
    destination: Path,
    image,
) -> None:
    """Write a generated preview image safely."""
    try:
        written = cv2.imwrite(
            str(destination),
            image,
        )
    except cv2.error as exc:
        raise ImageWriteError(
            "OpenCV could not write the inspection preview: "
            f"{destination}"
        ) from exc

    if written:
        return

    if destination.exists():
        with suppress(OSError):
            destination.unlink()

    raise ImageWriteError(
        "OpenCV could not write the inspection preview: "
        f"{destination}"
    )


def save_foreign_object_preview(
    video_path: str | Path,
    output_path: str | Path,
    region: RegionOfInterest,
    *,
    config: ForeignObjectConfig | None = None,
    frame_index: int = 0,
    overwrite: bool = False,
) -> ForeignObjectPreviewResult:
    """Inspect one video-frame ROI and save a foreign-object preview."""
    source = Path(video_path).expanduser().resolve()

    destination = _resolve_output_image_path(
        source,
        output_path,
        overwrite=overwrite,
    )

    selected_frame = _select_video_frame(
        source,
        frame_index,
    )

    roi_image = region.extract(
        selected_frame.image
    )

    active_config = (
        config
        if config is not None
        else ForeignObjectConfig()
    )

    inspection = inspect_foreign_objects(
        roi_image,
        active_config,
    )

    annotated_roi = render_foreign_object_result(
        roi_image,
        inspection,
    )

    if active_config.analysis_region is not None:
        annotated_roi = active_config.analysis_region.draw(
            annotated_roi,
            color=(255, 255, 0),
            thickness=2,
            show_label=True,
        )

    annotated_frame = selected_frame.image.copy()

    annotated_frame[
        region.y : region.y_end,
        region.x : region.x_end,
    ] = annotated_roi

    status_region = RegionOfInterest(
        name=(
            f"{region.name}: "
            f"{inspection.status.value}"
        ),
        x=region.x,
        y=region.y,
        width=region.width,
        height=region.height,
    )

    annotated_frame = status_region.draw(
        annotated_frame,
        color=_foreign_object_status_color(
            inspection.status
        ),
        thickness=3,
        show_label=True,
    )

    _write_preview_image(
        destination,
        annotated_frame,
    )

    return ForeignObjectPreviewResult(
        input_path=source,
        output_path=destination,
        frame_index=selected_frame.index,
        timestamp_seconds=selected_frame.timestamp_seconds,
        region=region,
        inspection=inspection,
    )


def save_liquid_level_preview(
    video_path: str | Path,
    output_path: str | Path,
    region: RegionOfInterest,
    *,
    config: LiquidLevelConfig | None = None,
    frame_index: int = 0,
    overwrite: bool = False,
) -> LiquidLevelPreviewResult:
    """Inspect one video-frame ROI and save a liquid-level preview."""
    source = Path(video_path).expanduser().resolve()

    destination = _resolve_output_image_path(
        source,
        output_path,
        overwrite=overwrite,
    )

    selected_frame = _select_video_frame(
        source,
        frame_index,
    )

    roi_image = region.extract(
        selected_frame.image
    )

    active_config = (
        config
        if config is not None
        else LiquidLevelConfig()
    )

    inspection = inspect_liquid_level(
        roi_image,
        active_config,
    )

    annotated_roi = render_liquid_level_result(
        roi_image,
        inspection,
        active_config,
    )

    annotated_frame = selected_frame.image.copy()

    annotated_frame[
        region.y : region.y_end,
        region.x : region.x_end,
    ] = annotated_roi

    status_region = RegionOfInterest(
        name=(
            f"{region.name}: "
            f"{inspection.status.value}"
        ),
        x=region.x,
        y=region.y,
        width=region.width,
        height=region.height,
    )

    annotated_frame = status_region.draw(
        annotated_frame,
        color=_liquid_level_status_color(
            inspection.status
        ),
        thickness=3,
        show_label=True,
    )

    _write_preview_image(
        destination,
        annotated_frame,
    )

    return LiquidLevelPreviewResult(
        input_path=source,
        output_path=destination,
        frame_index=selected_frame.index,
        timestamp_seconds=selected_frame.timestamp_seconds,
        region=region,
        inspection=inspection,
    )


def save_cork_preview(
    video_path: str | Path,
    output_path: str | Path,
    template_path: str | Path,
    *,
    config: CorkConfig | None = None,
    frame_index: int = 0,
    overwrite: bool = False,
) -> CorkPreviewResult:
    """Inspect one complete video frame using the cork template."""
    source = Path(video_path).expanduser().resolve()

    destination = _resolve_output_image_path(
        source,
        output_path,
        overwrite=overwrite,
    )

    selected_frame = _select_video_frame(
        source,
        frame_index,
    )

    resolved_template_path, template = _read_bgr_image(
        template_path,
        description="Cork template",
    )

    active_config = (
        config
        if config is not None
        else CorkConfig()
    )

    inspection = inspect_cork(
        selected_frame.image,
        template,
        active_config,
    )

    annotated_frame = render_cork_result(
        selected_frame.image,
        inspection,
    )

    _write_preview_image(
        destination,
        annotated_frame,
    )

    return CorkPreviewResult(
        input_path=source,
        output_path=destination,
        template_path=resolved_template_path,
        frame_index=selected_frame.index,
        timestamp_seconds=selected_frame.timestamp_seconds,
        inspection=inspection,
    )