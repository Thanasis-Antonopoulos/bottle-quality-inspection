"""Compare bottle passages between fixed inspection stations."""

import argparse
from pathlib import Path
from statistics import median

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.foreign_object import (
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.tracking import (
    StationObservation,
    associate_passages_by_time,
    segment_station_passages,
)
from bottle_quality_inspection.video_io import iter_video_frames


def build_parser() -> argparse.ArgumentParser:
    """Create the passage diagnostic parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect and associate bottle passages through "
            "the foreign-object and liquid/glass stations."
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
    )

    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=1,
        help=(
            "Maximum missed detections tolerated inside "
            "one station passage. Default: 1."
        ),
    )

    parser.add_argument(
        "--minimum-detected-frames",
        type=int,
        default=2,
        help=(
            "Minimum positive detections required for a "
            "station passage. Default: 2."
        ),
    )

    parser.add_argument(
        "--minimum-frame-offset",
        type=int,
        default=1,
        help=(
            "Minimum allowed representative-frame travel "
            "offset between stations. Default: 1."
        ),
    )

    parser.add_argument(
        "--maximum-frame-offset",
        type=int,
        default=30,
        help=(
            "Maximum allowed representative-frame travel "
            "offset between stations. Default: 30."
        ),
    )

    return parser


def create_liquid_config(
    profile,
) -> LiquidLevelConfig:
    """Convert profile configuration into algorithm configuration."""
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


def print_passages(
    name,
    passages,
) -> None:
    """Print detected passages for one station."""
    print(
        f"{name}: {len(passages)} passages"
    )

    for passage in passages:
        print(
            "  "
            f"#{passage.sequence_number}: "
            f"frames {passage.start_frame}-"
            f"{passage.end_frame}, "
            f"detections="
            f"{passage.detected_frame_count}, "
            f"representative="
            f"{passage.representative_frame}"
        )

    print()


def print_unmatched(
    title,
    passages,
) -> None:
    """Print passages which could not be associated."""
    print(
        f"{title}: {len(passages)}"
    )

    for passage in passages:
        print(
            "  "
            f"#{passage.sequence_number}: "
            f"frames "
            f"{passage.start_frame}-"
            f"{passage.end_frame}, "
            f"representative="
            f"{passage.representative_frame}"
        )

    print()


def main() -> int:
    """Run the station-passage diagnostic."""
    parser = build_parser()
    args = parser.parse_args()

    if args.max_missed_frames < 0:
        parser.error(
            "--max-missed-frames cannot be negative."
        )

    if args.minimum_detected_frames <= 0:
        parser.error(
            "--minimum-detected-frames must be greater than zero."
        )

    if args.minimum_frame_offset < 0:
        parser.error(
            "--minimum-frame-offset cannot be negative."
        )

    if (
        args.maximum_frame_offset
        < args.minimum_frame_offset
    ):
        parser.error(
            "--maximum-frame-offset must be greater than "
            "or equal to --minimum-frame-offset."
        )

    video_path = (
        args.input_video
        .expanduser()
        .resolve()
    )

    profile = load_inspection_config(
        args.config
    )

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

    foreign_observations = []
    liquid_observations = []

    print(
        f"Profile: {profile.profile}"
    )

    print(
        f"Video: {video_path}"
    )

    print()

    print(
        "Scanning both inspection stations..."
    )

    print()

    for video_frame in iter_video_frames(
        video_path
    ):
        foreign_image = (
            foreign_region.extract(
                video_frame.image
            )
        )

        liquid_image = (
            liquid_region.extract(
                video_frame.image
            )
        )

        foreign_result = inspect_foreign_objects(
            foreign_image,
            foreign_config,
        )

        liquid_result = inspect_liquid_level(
            liquid_image,
            liquid_config,
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
                station="liquid-level",
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
            args.max_missed_frames
        ),
        minimum_detected_frames=(
            args.minimum_detected_frames
        ),
    )

    liquid_passages = segment_station_passages(
        liquid_observations,
        max_missed_frames=(
            args.max_missed_frames
        ),
        minimum_detected_frames=(
            args.minimum_detected_frames
        ),
    )

    print_passages(
        "FOREIGN-OBJECT STATION",
        foreign_passages,
    )

    print_passages(
        "LIQUID/GLASS STATION",
        liquid_passages,
    )

    association = associate_passages_by_time(
        foreign_passages,
        liquid_passages,
        minimum_frame_offset=(
            args.minimum_frame_offset
        ),
        maximum_frame_offset=(
            args.maximum_frame_offset
        ),
    )

    print(
        "TIME-BASED ASSOCIATION CANDIDATES"
    )

    print(
        "---------------------------------"
    )

    if not association.pairs:
        print(
            "No passage associations were found."
        )

        print()

    offsets = []

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        offsets.append(
            pair.frame_offset
        )

        print(
            f"Bottle candidate {bottle_number}: "
            f"foreign frame "
            f"{pair.first.representative_frame} "
            f"→ liquid frame "
            f"{pair.second.representative_frame} "
            f"| offset="
            f"{pair.frame_offset} frames "
            f"("
            f"{pair.timestamp_offset_seconds:.3f}s"
            f")"
        )

    print()

    if offsets:
        print(
            "Observed travel offset range: "
            f"{min(offsets)}-"
            f"{max(offsets)} frames"
        )

        print(
            "Median travel offset: "
            f"{median(offsets):.1f} frames"
        )

        print()

    print_unmatched(
        "UNMATCHED FOREIGN-OBJECT PASSAGES",
        association.unmatched_first,
    )

    print_unmatched(
        "UNMATCHED LIQUID/GLASS PASSAGES",
        association.unmatched_second,
    )

    print(
        "Association summary"
    )

    print(
        "-------------------"
    )

    print(
        f"Associated bottles: "
        f"{association.pair_count}"
    )

    print(
        "Unmatched foreign passages: "
        f"{len(association.unmatched_first)}"
    )

    print(
        "Unmatched liquid/glass passages: "
        f"{len(association.unmatched_second)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )