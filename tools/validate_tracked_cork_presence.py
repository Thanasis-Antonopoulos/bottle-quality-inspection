"""Validate calibrated multi-frame cork presence on tracked bottles."""

import argparse
from pathlib import Path

import cv2

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    inspect_cork,
)
from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceFrameScore,
    evaluate_cork_presence,
)
from bottle_quality_inspection.inspections.foreign_object import (
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.tracking import (
    StationObservation,
    associate_passages_by_time,
    segment_station_passages,
)
from bottle_quality_inspection.video_io import iter_video_frames


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate tracked multi-frame cork presence "
            "for associated bottles."
        )
    )

    parser.add_argument(
        "input_video",
        type=Path,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/legacy-2022.toml"
        ),
    )

    return parser


def create_liquid_config(
    profile,
) -> LiquidLevelConfig:
    """Convert profile values to liquid-level configuration."""
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


def main() -> int:
    """Run tracked cork-presence validation."""
    parser = build_parser()
    args = parser.parse_args()

    video_path = (
        args.input_video
        .expanduser()
        .resolve()
    )

    profile = load_inspection_config(
        args.config
    )

    foreign_region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    bottle_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    cork_region = RegionOfInterest(
        name="cork-search",
        x=0,
        y=40,
        width=200,
        height=200,
    )

    body_region = RegionOfInterest(
        name="body-control",
        x=0,
        y=260,
        width=200,
        height=300,
    )

    template = cv2.imread(
        str(
            profile.cork.template
            .expanduser()
            .resolve()
        ),
        cv2.IMREAD_COLOR,
    )

    if template is None:
        raise RuntimeError(
            "Could not read cork template."
        )

    cork_config = CorkConfig(
        method=profile.cork.method,
        minimum_match_score=(
            profile.cork.minimum_match_score
            if (
                profile.cork.minimum_match_score
                is not None
            )
            else 0.80
        ),
    )

    foreign_config = (
        profile.foreign_object
        .to_algorithm_config()
    )

    liquid_config = create_liquid_config(
        profile
    )

    frames = {}

    foreign_observations = []
    liquid_observations = []

    for video_frame in iter_video_frames(
        video_path
    ):
        frames[
            video_frame.index
        ] = (
            video_frame.image.copy()
        )

        foreign_result = (
            inspect_foreign_objects(
                foreign_region.extract(
                    video_frame.image
                ),
                foreign_config,
            )
        )

        liquid_result = (
            inspect_liquid_level(
                bottle_region.extract(
                    video_frame.image
                ),
                liquid_config,
            )
        )

        foreign_observations.append(
            StationObservation(
                station="foreign",
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
                station="liquid",
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
        max_missed_frames=1,
        minimum_detected_frames=2,
    )

    liquid_passages = segment_station_passages(
        liquid_observations,
        max_missed_frames=1,
        minimum_detected_frames=2,
    )

    association = associate_passages_by_time(
        foreign_passages,
        liquid_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=30,
    )

    print(
        f"Associated bottles: "
        f"{association.pair_count}"
    )

    print()

    passed = 0

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        frame_scores = []

        for observation in (
            pair.second.observations
        ):
            bottle_crop = (
                bottle_region.extract(
                    frames[
                        observation.frame_index
                    ]
                )
            )

            cork_result = inspect_cork(
                cork_region.extract(
                    bottle_crop
                ),
                template,
                cork_config,
            )

            body_result = inspect_cork(
                body_region.extract(
                    bottle_crop
                ),
                template,
                cork_config,
            )

            frame_scores.append(
                CorkPresenceFrameScore(
                    frame_index=(
                        observation.frame_index
                    ),
                    cork_score=(
                        cork_result.maximum_score
                    ),
                    body_control_score=(
                        body_result.maximum_score
                    ),
                )
            )

        result = evaluate_cork_presence(
            frame_scores
        )

        bottle_id = (
            f"bottle-{bottle_number:03d}"
        )

        print(
            bottle_id
        )

        print(
            "-" * len(
                bottle_id
            )
        )

        print(
            f"Status: "
            f"{result.status.value}"
        )

        print(
            "Selected frames: "
            + ", ".join(
                str(frame)
                for frame
                in result.selected_frame_indices
            )
        )

        print(
            "Top-3 cork median: "
            f"{result.top_k_cork_median:.4f}"
        )

        print(
            "Top-3 paired body median: "
            f"{result.top_k_body_control_median:.4f}"
        )

        print(
            "Top-3 margin: "
            f"{result.top_k_margin:.4f}"
        )

        print()

        if result.cork_present:
            passed += 1

    print(
        "SUMMARY"
    )

    print(
        "-------"
    )

    print(
        f"Tracked bottles: "
        f"{association.pair_count}"
    )

    print(
        f"Cork presence confirmed: "
        f"{passed}"
    )

    print(
        "Cork presence unverified: "
        f"{association.pair_count - passed}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )