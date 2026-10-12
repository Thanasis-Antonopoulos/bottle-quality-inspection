"""Visualize candidate bottle associations between inspection stations."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.foreign_object import (
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.tracking import (
    PassageAssociationResult,
    StationObservation,
    associate_passages_by_time,
    segment_station_passages,
)
from bottle_quality_inspection.video_io import iter_video_frames


def build_parser() -> argparse.ArgumentParser:
    """Create the visualization command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side images for candidate bottle "
            "associations between the foreign-object and "
            "liquid/glass stations."
        )
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/legacy-2022.toml"
        ),
        help=(
            "Inspection configuration profile. "
            "Default: configs/legacy-2022.toml."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/bottle-associations"
        ),
        help=(
            "Directory for association images. "
            "Default: outputs/bottle-associations."
        ),
    )

    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--minimum-detected-frames",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--minimum-frame-offset",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--maximum-frame-offset",
        type=int,
        default=30,
    )

    return parser


def create_liquid_config(
    profile,
) -> LiquidLevelConfig:
    """Convert loaded profile values into liquid-level configuration."""
    liquid = profile.liquid_level

    return LiquidLevelConfig(
        neck_region=(
            liquid.neck_region.to_roi()
        ),
        canny_threshold1=(
            liquid.canny_threshold1
        ),
        canny_threshold2=(
            liquid.canny_threshold2
        ),
        hough_rho=(
            liquid.hough_rho
        ),
        hough_theta_degrees=(
            liquid.hough_theta_degrees
        ),
        hough_threshold=(
            liquid.hough_threshold
        ),
        minimum_line_length=(
            liquid.minimum_line_length
        ),
        maximum_line_gap=(
            liquid.maximum_line_gap
        ),
        acceptable_minimum=(
            liquid.acceptable_minimum
        ),
        acceptable_maximum=(
            liquid.acceptable_maximum
        ),
    )


def detect_associations(
    video_path: Path,
    profile,
    *,
    max_missed_frames: int,
    minimum_detected_frames: int,
    minimum_frame_offset: int,
    maximum_frame_offset: int,
) -> PassageAssociationResult:
    """Detect station passages and associate them chronologically."""
    foreign_profile = (
        profile.foreign_object
    )

    liquid_profile = (
        profile.liquid_level
    )

    foreign_region = (
        foreign_profile.roi.to_roi()
    )

    liquid_region = (
        liquid_profile.roi.to_roi()
    )

    foreign_config = (
        foreign_profile.to_algorithm_config()
    )

    liquid_config = create_liquid_config(
        profile
    )

    foreign_observations: list[
        StationObservation
    ] = []

    liquid_observations: list[
        StationObservation
    ] = []

    for video_frame in iter_video_frames(
        video_path
    ):
        foreign_crop = (
            foreign_region.extract(
                video_frame.image
            )
        )

        liquid_crop = (
            liquid_region.extract(
                video_frame.image
            )
        )

        foreign_result = (
            inspect_foreign_objects(
                foreign_crop,
                foreign_config,
            )
        )

        liquid_result = (
            inspect_liquid_level(
                liquid_crop,
                liquid_config,
            )
        )

        foreign_observations.append(
            StationObservation(
                station="foreign-object",
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                detected=(
                    foreign_result.bottle_detected
                ),
            )
        )

        liquid_observations.append(
            StationObservation(
                station="liquid-glass",
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                detected=(
                    liquid_result.level_detected
                ),
            )
        )

    foreign_passages = segment_station_passages(
        foreign_observations,
        max_missed_frames=(
            max_missed_frames
        ),
        minimum_detected_frames=(
            minimum_detected_frames
        ),
    )

    liquid_passages = segment_station_passages(
        liquid_observations,
        max_missed_frames=(
            max_missed_frames
        ),
        minimum_detected_frames=(
            minimum_detected_frames
        ),
    )

    return associate_passages_by_time(
        foreign_passages,
        liquid_passages,
        minimum_frame_offset=(
            minimum_frame_offset
        ),
        maximum_frame_offset=(
            maximum_frame_offset
        ),
    )


def collect_frames(
    video_path: Path,
    frame_indices: set[int],
) -> dict[int, np.ndarray]:
    """Read selected video frames into memory."""
    collected: dict[
        int,
        np.ndarray
    ] = {}

    for video_frame in iter_video_frames(
        video_path
    ):
        if video_frame.index not in frame_indices:
            continue

        collected[
            video_frame.index
        ] = video_frame.image.copy()

    missing = (
        frame_indices
        - set(
            collected
        )
    )

    if missing:
        missing_text = ", ".join(
            str(frame)
            for frame in sorted(
                missing
            )
        )

        raise RuntimeError(
            "Could not decode required video frames: "
            f"{missing_text}"
        )

    return collected


def add_header(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str,
) -> np.ndarray:
    """Add a text header above one station crop."""
    header_height = 90

    canvas = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    cv2.putText(
        canvas,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        subtitle,
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return canvas


def create_station_pair_image(
    foreign_frame: np.ndarray,
    liquid_frame: np.ndarray,
    foreign_region: RegionOfInterest,
    liquid_region: RegionOfInterest,
    *,
    bottle_id: str,
    foreign_frame_index: int,
    liquid_frame_index: int,
    frame_offset: int,
) -> np.ndarray:
    """Create a side-by-side comparison of one candidate bottle."""
    foreign_crop = (
        foreign_region.extract(
            foreign_frame
        )
        .copy()
    )

    liquid_crop = (
        liquid_region.extract(
            liquid_frame
        )
        .copy()
    )

    foreign_panel = add_header(
        foreign_crop,
        title=(
            f"{bottle_id} | FOREIGN"
        ),
        subtitle=(
            f"frame {foreign_frame_index}"
        ),
    )

    liquid_panel = add_header(
        liquid_crop,
        title=(
            f"{bottle_id} | LIQUID / GLASS"
        ),
        subtitle=(
            f"frame {liquid_frame_index} "
            f"| offset +{frame_offset}"
        ),
    )

    separator = np.zeros(
        (
            foreign_panel.shape[0],
            20,
            3,
        ),
        dtype=np.uint8,
    )

    return np.hstack(
        (
            foreign_panel,
            separator,
            liquid_panel,
        )
    )


def create_full_frame_panel(
    frame: np.ndarray,
    region: RegionOfInterest,
    *,
    label: str,
) -> np.ndarray:
    """Annotate and resize one full-frame station view."""
    annotated = frame.copy()

    annotated = region.draw(
        annotated,
        color=(0, 255, 0),
        thickness=4,
        show_label=False,
    )

    cv2.putText(
        annotated,
        label,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    target_width = 900

    scale = (
        target_width
        / annotated.shape[1]
    )

    target_height = int(
        round(
            annotated.shape[0]
            * scale
        )
    )

    return cv2.resize(
        annotated,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def create_full_pair_image(
    foreign_frame: np.ndarray,
    liquid_frame: np.ndarray,
    foreign_region: RegionOfInterest,
    liquid_region: RegionOfInterest,
    *,
    bottle_id: str,
    foreign_frame_index: int,
    liquid_frame_index: int,
) -> np.ndarray:
    """Create a contextual side-by-side full-frame comparison."""
    left = create_full_frame_panel(
        foreign_frame,
        foreign_region,
        label=(
            f"{bottle_id} FOREIGN "
            f"frame {foreign_frame_index}"
        ),
    )

    right = create_full_frame_panel(
        liquid_frame,
        liquid_region,
        label=(
            f"{bottle_id} LIQUID/GLASS "
            f"frame {liquid_frame_index}"
        ),
    )

    return np.hstack(
        (
            left,
            right,
        )
    )


def save_image(
    destination: Path,
    image: np.ndarray,
) -> None:
    """Write one visualization image."""
    written = cv2.imwrite(
        str(destination),
        image,
    )

    if not written:
        raise RuntimeError(
            "Could not write association image: "
            f"{destination}"
        )


def main() -> int:
    """Generate visual association evidence."""
    parser = build_parser()
    args = parser.parse_args()

    video_path = (
        args.input_video
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = load_inspection_config(
        args.config
    )

    association = detect_associations(
        video_path,
        profile,
        max_missed_frames=(
            args.max_missed_frames
        ),
        minimum_detected_frames=(
            args.minimum_detected_frames
        ),
        minimum_frame_offset=(
            args.minimum_frame_offset
        ),
        maximum_frame_offset=(
            args.maximum_frame_offset
        ),
    )

    foreign_region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    liquid_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    required_frames: set[int] = set()

    for pair in association.pairs:
        required_frames.add(
            pair.first.representative_frame
        )

        required_frames.add(
            pair.second.representative_frame
        )

    for passage in association.unmatched_second:
        required_frames.add(
            passage.representative_frame
        )

    frames = collect_frames(
        video_path,
        required_frames,
    )

    print(
        f"Profile: {profile.profile}"
    )

    print(
        f"Video: {video_path}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print()

    print(
        "ASSOCIATED BOTTLE CANDIDATES"
    )

    print(
        "============================"
    )

    print()

    for index, pair in enumerate(
        association.pairs,
        start=1,
    ):
        bottle_id = (
            f"bottle-{index:03d}"
        )

        foreign_frame_index = (
            pair.first.representative_frame
        )

        liquid_frame_index = (
            pair.second.representative_frame
        )

        crop_pair = (
            create_station_pair_image(
                frames[
                    foreign_frame_index
                ],
                frames[
                    liquid_frame_index
                ],
                foreign_region,
                liquid_region,
                bottle_id=bottle_id,
                foreign_frame_index=(
                    foreign_frame_index
                ),
                liquid_frame_index=(
                    liquid_frame_index
                ),
                frame_offset=(
                    pair.frame_offset
                ),
            )
        )

        full_pair = (
            create_full_pair_image(
                frames[
                    foreign_frame_index
                ],
                frames[
                    liquid_frame_index
                ],
                foreign_region,
                liquid_region,
                bottle_id=bottle_id,
                foreign_frame_index=(
                    foreign_frame_index
                ),
                liquid_frame_index=(
                    liquid_frame_index
                ),
            )
        )

        crop_path = (
            output_dir
            / (
                f"{bottle_id}-stations.jpg"
            )
        )

        full_path = (
            output_dir
            / (
                f"{bottle_id}-full-frames.jpg"
            )
        )

        save_image(
            crop_path,
            crop_pair,
        )

        save_image(
            full_path,
            full_pair,
        )

        print(
            f"{bottle_id}: "
            f"foreign frame "
            f"{foreign_frame_index} "
            f"→ liquid/glass frame "
            f"{liquid_frame_index} "
            f"| +{pair.frame_offset} frames"
        )

        print(
            f"  station view: "
            f"{crop_path}"
        )

        print(
            f"  full frames: "
            f"{full_path}"
        )

    print()

    print(
        "UNMATCHED DOWNSTREAM PASSAGES"
    )

    print(
        "============================="
    )

    print()

    for passage in association.unmatched_second:
        frame_index = (
            passage.representative_frame
        )

        crop = (
            liquid_region.extract(
                frames[
                    frame_index
                ]
            )
            .copy()
        )

        panel = add_header(
            crop,
            title=(
                "UNMATCHED LIQUID / GLASS"
            ),
            subtitle=(
                f"frame {frame_index} "
                "| likely present before recording"
            ),
        )

        destination = (
            output_dir
            / (
                "unmatched-liquid-"
                f"frame-{frame_index:03d}.jpg"
            )
        )

        save_image(
            destination,
            panel,
        )

        print(
            f"frame {frame_index}: "
            f"{destination}"
        )

    print()

    print(
        "Visual association export complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )