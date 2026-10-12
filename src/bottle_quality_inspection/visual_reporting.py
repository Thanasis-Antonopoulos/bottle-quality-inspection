"""Visual reporting for tracked bottle inspection runs."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.bottle_pipeline import (
    BottleInspectionRun,
    TrackedBottleInspection,
)
from bottle_quality_inspection.bottle_result import (
    BottleStatus,
)
from bottle_quality_inspection.exceptions import (
    ImageWriteError,
)
from bottle_quality_inspection.pipeline import (
    VideoProcessResult,
    process_video,
)
from bottle_quality_inspection.roi import (
    RegionOfInterest,
)
from bottle_quality_inspection.video_io import (
    VideoFrame,
    iter_video_frames,
)


@dataclass(frozen=True, slots=True)
class FlaggedBottleImage:
    """Saved review or rejected image for one physical bottle."""

    bottle_id: str
    status: BottleStatus
    frame_index: int
    path: Path


@dataclass(frozen=True, slots=True)
class VisualInspectionArtifacts:
    """Visual artifacts created for one inspection run."""

    video_result: VideoProcessResult

    review_images: tuple[
        FlaggedBottleImage,
        ...
    ]

    rejected_images: tuple[
        FlaggedBottleImage,
        ...
    ]


def _status_color(
    status: BottleStatus,
) -> tuple[int, int, int]:
    """Return BGR annotation color for one final bottle status."""
    if status is BottleStatus.GOOD:
        return (
            0,
            180,
            0,
        )

    if status is BottleStatus.REVIEW:
        return (
            0,
            165,
            255,
        )

    return (
        0,
        0,
        255,
    )


def _draw_label(
    frame: np.ndarray,
    region: RegionOfInterest,
    *,
    text: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw an inspection-region label on a frame."""
    annotated = frame.copy()

    text_y = max(
        region.y - 12,
        30,
    )

    cv2.putText(
        annotated,
        text,
        (
            region.x,
            text_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return annotated


def _draw_active_bottle(
    frame: np.ndarray,
    tracked: TrackedBottleInspection,
    region: RegionOfInterest,
    *,
    station_name: str,
) -> np.ndarray:
    """Draw one active tracked bottle station."""
    color = _status_color(
        tracked.result.status
    )

    annotated = region.draw(
        frame,
        color=color,
        thickness=3,
        show_label=False,
    )

    label = (
        f"{tracked.bottle_id} "
        f"| {station_name} "
        f"| {tracked.result.status.value.upper()}"
    )

    return _draw_label(
        annotated,
        region,
        text=label,
        color=color,
    )


def _draw_unmatched_passage(
    frame: np.ndarray,
    region: RegionOfInterest,
    *,
    station_name: str,
) -> np.ndarray:
    """Draw an unmatched station passage."""
    color = (
        0,
        255,
        255,
    )

    annotated = region.draw(
        frame,
        color=color,
        thickness=3,
        show_label=False,
    )

    return _draw_label(
        annotated,
        region,
        text=(
            f"UNMATCHED | {station_name}"
        ),
        color=color,
    )


def _draw_summary(
    frame: np.ndarray,
    run: BottleInspectionRun,
) -> np.ndarray:
    """Draw run-level inspection counts."""
    annotated = frame.copy()

    good = sum(
        tracked.result.status
        is BottleStatus.GOOD
        for tracked in run.bottles
    )

    defective = sum(
        tracked.result.status
        is BottleStatus.DEFECTIVE
        for tracked in run.bottles
    )

    review = sum(
        tracked.result.status
        is BottleStatus.REVIEW
        for tracked in run.bottles
    )

    text = (
        f"GOOD {good} | "
        f"REVIEW {review} | "
        f"DEFECTIVE {defective}"
    )

    cv2.rectangle(
        annotated,
        (20, 20),
        (570, 65),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        text,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def annotate_inspection_frame(
    frame: np.ndarray,
    frame_index: int,
    run: BottleInspectionRun,
    foreign_region: RegionOfInterest,
    downstream_region: RegionOfInterest,
) -> np.ndarray:
    """Annotate one frame with active physical-bottle identities."""
    if frame_index < 0:
        raise ValueError(
            "frame_index cannot be negative."
        )

    annotated = (
        frame.copy()
    )

    for tracked in run.bottles:
        if (
            tracked.foreign_start_frame
            <= frame_index
            <= tracked.foreign_end_frame
        ):
            annotated = (
                _draw_active_bottle(
                    annotated,
                    tracked,
                    foreign_region,
                    station_name=(
                        "FOREIGN"
                    ),
                )
            )

        if (
            tracked.downstream_start_frame
            <= frame_index
            <= tracked.downstream_end_frame
        ):
            annotated = (
                _draw_active_bottle(
                    annotated,
                    tracked,
                    downstream_region,
                    station_name=(
                        "LIQUID/GLASS/CORK"
                    ),
                )
            )

    for passage in (
        run.unmatched_foreign_passages
    ):
        if (
            passage.start_frame
            <= frame_index
            <= passage.end_frame
        ):
            annotated = (
                _draw_unmatched_passage(
                    annotated,
                    foreign_region,
                    station_name="FOREIGN",
                )
            )

    for passage in (
        run.unmatched_downstream_passages
    ):
        if (
            passage.start_frame
            <= frame_index
            <= passage.end_frame
        ):
            annotated = (
                _draw_unmatched_passage(
                    annotated,
                    downstream_region,
                    station_name=(
                        "LIQUID/GLASS"
                    ),
                )
            )

    return _draw_summary(
        annotated,
        run,
    )


def create_inspection_frame_processor(
    run: BottleInspectionRun,
    foreign_region: RegionOfInterest,
    downstream_region: RegionOfInterest,
):
    """Create a processor compatible with the reusable video pipeline."""

    def processor(
        video_frame: VideoFrame,
    ) -> np.ndarray:
        return annotate_inspection_frame(
            video_frame.image,
            video_frame.index,
            run,
            foreign_region,
            downstream_region,
        )

    return processor


def render_annotated_inspection_video(
    run: BottleInspectionRun,
    output_video: str | Path,
    foreign_region: RegionOfInterest,
    downstream_region: RegionOfInterest,
    *,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> VideoProcessResult:
    """Write an annotated video for one complete inspection run."""
    processor = (
        create_inspection_frame_processor(
            run,
            foreign_region,
            downstream_region,
        )
    )

    return process_video(
        run.video_path,
        output_video,
        processor,
        codec=codec,
        overwrite=overwrite,
    )


def flagged_bottle_frame(
    tracked: TrackedBottleInspection,
) -> int:
    """Return a representative downstream frame for flagged evidence."""
    return (
        tracked.downstream_start_frame
        + tracked.downstream_end_frame
    ) // 2


def flagged_bottles(
    run: BottleInspectionRun,
) -> tuple[
    TrackedBottleInspection,
    ...,
]:
    """Return bottles requiring review or confirmed rejection."""
    return tuple(
        tracked
        for tracked in run.bottles
        if (
            tracked.result.status
            is not BottleStatus.GOOD
        )
    )


def _prepare_flagged_image_paths(
    run: BottleInspectionRun,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[
    str,
    tuple[
        TrackedBottleInspection,
        int,
        Path,
    ],
]:
    """Prepare output paths before writing flagged bottle images."""
    prepared: dict[
        str,
        tuple[
            TrackedBottleInspection,
            int,
            Path,
        ],
    ] = {}

    for tracked in flagged_bottles(
        run
    ):
        frame_index = (
            flagged_bottle_frame(
                tracked
            )
        )

        if (
            tracked.result.status
            is BottleStatus.DEFECTIVE
        ):
            category_dir = (
                output_dir
                / "rejected"
            )
        else:
            category_dir = (
                output_dir
                / "review"
            )

        destination = (
            category_dir
            / (
                f"{tracked.bottle_id}-"
                f"frame-{frame_index:03d}.jpg"
            )
        )

        if destination.exists():
            if not destination.is_file():
                raise IsADirectoryError(
                    "Flagged image output path "
                    "is not a file: "
                    f"{destination}"
                )

            if not overwrite:
                raise FileExistsError(
                    "Flagged bottle image already exists. "
                    "Use overwrite=True to replace it: "
                    f"{destination}"
                )

        prepared[
            tracked.bottle_id
        ] = (
            tracked,
            frame_index,
            destination,
        )

    return prepared


def save_flagged_bottle_images(
    run: BottleInspectionRun,
    output_dir: str | Path,
    foreign_region: RegionOfInterest,
    downstream_region: RegionOfInterest,
    *,
    overwrite: bool = False,
) -> tuple[
    tuple[
        FlaggedBottleImage,
        ...,
    ],
    tuple[
        FlaggedBottleImage,
        ...,
    ],
]:
    """Save representative images for REVIEW and DEFECTIVE bottles."""
    destination_root = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    prepared = (
        _prepare_flagged_image_paths(
            run,
            destination_root,
            overwrite=overwrite,
        )
    )

    if not prepared:
        return (
            (),
            (),
        )

    required_frames = {
        frame_index
        for (
            _tracked,
            frame_index,
            _destination,
        )
        in prepared.values()
    }

    saved: dict[
        str,
        FlaggedBottleImage,
    ] = {}

    found_frames: set[int] = set()

    for video_frame in iter_video_frames(
        run.video_path
    ):
        if (
            video_frame.index
            not in required_frames
        ):
            continue

        found_frames.add(
            video_frame.index
        )

        for (
            bottle_id,
            (
                tracked,
                frame_index,
                destination,
            ),
        ) in prepared.items():
            if (
                frame_index
                != video_frame.index
            ):
                continue

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            annotated = (
                annotate_inspection_frame(
                    video_frame.image,
                    video_frame.index,
                    run,
                    foreign_region,
                    downstream_region,
                )
            )

            written = cv2.imwrite(
                str(
                    destination
                ),
                annotated,
            )

            if not written:
                raise ImageWriteError(
                    "Could not write flagged bottle image: "
                    f"{destination}"
                )

            saved[
                bottle_id
            ] = (
                FlaggedBottleImage(
                    bottle_id=(
                        bottle_id
                    ),
                    status=(
                        tracked.result.status
                    ),
                    frame_index=(
                        frame_index
                    ),
                    path=(
                        destination
                    ),
                )
            )

    missing_frames = (
        required_frames
        - found_frames
    )

    if missing_frames:
        missing_text = ", ".join(
            str(frame)
            for frame in sorted(
                missing_frames
            )
        )

        raise RuntimeError(
            "Could not decode flagged bottle frames: "
            f"{missing_text}"
        )

    review_images = tuple(
        saved[
            tracked.bottle_id
        ]
        for tracked in run.bottles
        if (
            tracked.result.status
            is BottleStatus.REVIEW
            and tracked.bottle_id
            in saved
        )
    )

    rejected_images = tuple(
        saved[
            tracked.bottle_id
        ]
        for tracked in run.bottles
        if (
            tracked.result.status
            is BottleStatus.DEFECTIVE
            and tracked.bottle_id
            in saved
        )
    )

    return (
        review_images,
        rejected_images,
    )


def write_visual_inspection_artifacts(
    run: BottleInspectionRun,
    output_dir: str | Path,
    foreign_region: RegionOfInterest,
    downstream_region: RegionOfInterest,
    *,
    video_filename: str = "annotated-inspection.mp4",
    codec: str = "mp4v",
    overwrite: bool = False,
) -> VisualInspectionArtifacts:
    """Write annotated video plus review/rejected evidence images."""
    destination = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    if (
        Path(
            video_filename
        ).name
        != video_filename
    ):
        raise ValueError(
            "video_filename must be a filename, not a path."
        )

    if not video_filename.strip():
        raise ValueError(
            "video_filename cannot be empty."
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_result = (
        render_annotated_inspection_video(
            run,
            destination
            / video_filename,
            foreign_region,
            downstream_region,
            codec=codec,
            overwrite=overwrite,
        )
    )

    (
        review_images,
        rejected_images,
    ) = save_flagged_bottle_images(
        run,
        destination,
        foreign_region,
        downstream_region,
        overwrite=overwrite,
    )

    return VisualInspectionArtifacts(
        video_result=(
            video_result
        ),
        review_images=(
            review_images
        ),
        rejected_images=(
            rejected_images
        ),
    )