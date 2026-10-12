"""Diagnose all inspection signals across each tracked physical bottle."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

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
    ForeignObjectStatus,
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.glass import (
    GlassStatus,
    inspect_glass,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    LiquidLevelStatus,
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


@dataclass(frozen=True, slots=True)
class ForeignFrameResult:
    """Foreign-object evidence for one upstream frame."""

    frame_index: int
    status: ForeignObjectStatus
    decision_black_pixels: int


@dataclass(frozen=True, slots=True)
class DownstreamFrameResult:
    """Liquid, glass and cork evidence for one downstream frame."""

    frame_index: int

    liquid_status: LiquidLevelStatus
    liquid_level: float | None

    glass_status: GlassStatus
    glass_triggered_signals: int

    cork_score: float
    body_control_score: float


def build_parser() -> argparse.ArgumentParser:
    """Create the multi-frame diagnostic parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect every frame belonging to each associated physical bottle."
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

    return parser


def create_liquid_config(
    profile,
) -> LiquidLevelConfig:
    """Convert loaded profile values to liquid algorithm configuration."""
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


def read_cork_template(
    template_path: Path,
) -> np.ndarray:
    """Read the configured normal-cork template."""
    resolved = (
        template_path
        .expanduser()
        .resolve()
    )

    template = cv2.imread(
        str(resolved),
        cv2.IMREAD_COLOR,
    )

    if template is None:
        raise RuntimeError(
            f"Could not read cork template: {resolved}"
        )

    return template


def detect_associations(
    video_path: Path,
    profile,
) -> PassageAssociationResult:
    """Detect upstream/downstream passages and associate them."""
    foreign_region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    downstream_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    foreign_config = (
        profile.foreign_object
        .to_algorithm_config()
    )

    liquid_config = create_liquid_config(
        profile
    )

    foreign_observations: list[
        StationObservation
    ] = []

    downstream_observations: list[
        StationObservation
    ] = []

    for video_frame in iter_video_frames(
        video_path
    ):
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
                downstream_region.extract(
                    video_frame.image
                ),
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

        downstream_observations.append(
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
        max_missed_frames=1,
        minimum_detected_frames=2,
    )

    downstream_passages = segment_station_passages(
        downstream_observations,
        max_missed_frames=1,
        minimum_detected_frames=2,
    )

    return associate_passages_by_time(
        foreign_passages,
        downstream_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=30,
    )


def collect_frames(
    video_path: Path,
    required_frames: set[int],
) -> dict[int, np.ndarray]:
    """Decode all frames needed for the associated bottle passages."""
    frames: dict[
        int,
        np.ndarray
    ] = {}

    for video_frame in iter_video_frames(
        video_path
    ):
        if (
            video_frame.index
            not in required_frames
        ):
            continue

        frames[
            video_frame.index
        ] = (
            video_frame.image.copy()
        )

    missing = (
        required_frames
        - set(
            frames
        )
    )

    if missing:
        raise RuntimeError(
            "Could not decode frames: "
            + ", ".join(
                str(frame)
                for frame in sorted(
                    missing
                )
            )
        )

    return frames


def format_level(
    level: float | None,
) -> str:
    """Format an optional liquid-level measurement."""
    if level is None:
        return "N/A"

    return f"{level:.2f}"


def print_foreign_summary(
    results: list[ForeignFrameResult],
) -> None:
    """Print frame-level foreign-object evidence."""
    counts = [
        result.decision_black_pixels
        for result in results
    ]

    clear_count = sum(
        result.status
        is ForeignObjectStatus.CLEAR
        for result in results
    )

    defective_count = sum(
        result.status
        is ForeignObjectStatus.DEFECTIVE
        for result in results
    )

    print(
        "FOREIGN OBJECT"
    )

    print(
        "--------------"
    )

    print(
        f"Frames: {len(results)}"
    )

    print(
        f"CLEAR frames: {clear_count}"
    )

    print(
        f"DEFECTIVE frames: {defective_count}"
    )

    print(
        "Decision-black range: "
        f"{min(counts)}-"
        f"{max(counts)}"
    )

    print(
        "Decision-black median: "
        f"{median(counts):.1f}"
    )

    print()

    for result in results:
        print(
            "  "
            f"frame={result.frame_index:03d} "
            f"black="
            f"{result.decision_black_pixels:4d} "
            f"status="
            f"{result.status.value}"
        )

    print()


def print_downstream_summary(
    results: list[DownstreamFrameResult],
) -> None:
    """Print liquid, glass and cork evidence."""
    liquid_values = [
        result.liquid_level
        for result in results
        if result.liquid_level is not None
    ]

    liquid_clear = sum(
        result.liquid_status
        is LiquidLevelStatus.CLEAR
        for result in results
    )

    liquid_defective = sum(
        result.liquid_status
        is LiquidLevelStatus.DEFECTIVE
        for result in results
    )

    glass_clear = sum(
        result.glass_status
        is GlassStatus.CLEAR
        for result in results
    )

    glass_defective = sum(
        result.glass_status
        is GlassStatus.DEFECTIVE
        for result in results
    )

    cork_frame_scores = [
        CorkPresenceFrameScore(
            frame_index=(
                result.frame_index
            ),
            cork_score=(
                result.cork_score
            ),
            body_control_score=(
                result.body_control_score
            ),
        )
        for result in results
    ]

    cork_presence = evaluate_cork_presence(
        cork_frame_scores
    )

    print(
        "LIQUID LEVEL"
    )

    print(
        "------------"
    )

    print(
        f"CLEAR frames: {liquid_clear}"
    )

    print(
        f"DEFECTIVE frames: {liquid_defective}"
    )

    if liquid_values:
        print(
            "Measured level range: "
            f"{min(liquid_values):.2f}-"
            f"{max(liquid_values):.2f}"
        )

        print(
            "Measured level median: "
            f"{median(liquid_values):.2f}"
        )

    print()

    print(
        "GLASS"
    )

    print(
        "-----"
    )

    print(
        f"CLEAR frames: {glass_clear}"
    )

    print(
        f"DEFECTIVE frames: {glass_defective}"
    )

    print()

    print(
        "CORK PRESENCE"
    )

    print(
        "-------------"
    )

    print(
        f"Status: "
        f"{cork_presence.status.value}"
    )

    print(
        "Selected frames: "
        + ", ".join(
            str(frame)
            for frame
            in cork_presence.selected_frame_indices
        )
    )

    print(
        "Top-3 cork median: "
        f"{cork_presence.top_k_cork_median:.4f}"
    )

    print(
        "Top-3 body median: "
        f"{cork_presence.top_k_body_control_median:.4f}"
    )

    print(
        "Top-3 margin: "
        f"{cork_presence.top_k_margin:.4f}"
    )

    print()

    print(
        "DOWNSTREAM PER-FRAME EVIDENCE"
    )

    print(
        "-----------------------------"
    )

    for result in results:
        print(
            "  "
            f"frame={result.frame_index:03d} "
            f"liquid="
            f"{format_level(result.liquid_level):>6} "
            f"{result.liquid_status.value:<9} "
            f"glass="
            f"{result.glass_status.value:<9} "
            f"glass_signals="
            f"{result.glass_triggered_signals} "
            f"cork="
            f"{result.cork_score:.4f} "
            f"body="
            f"{result.body_control_score:.4f}"
        )

    print()


def main() -> int:
    """Run complete multi-frame bottle diagnostics."""
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

    association = detect_associations(
        video_path,
        profile,
    )

    foreign_region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    downstream_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    foreign_config = (
        profile.foreign_object
        .to_algorithm_config()
    )

    liquid_config = create_liquid_config(
        profile
    )

    cork_template = (
        read_cork_template(
            profile.cork.template
        )
    )

    cork_config = CorkConfig(
        method=(
            profile.cork.method
        ),
        minimum_match_score=(
            profile.cork.minimum_match_score
            if (
                profile.cork.minimum_match_score
                is not None
            )
            else 0.80
        ),
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

    required_frames: set[int] = set()

    for pair in association.pairs:
        for observation in (
            pair.first.observations
        ):
            required_frames.add(
                observation.frame_index
            )

        for observation in (
            pair.second.observations
        ):
            required_frames.add(
                observation.frame_index
            )

    frames = collect_frames(
        video_path,
        required_frames,
    )

    print(
        f"Profile: "
        f"{profile.profile}"
    )

    print(
        f"Video: "
        f"{video_path}"
    )

    print(
        f"Associated bottles: "
        f"{association.pair_count}"
    )

    print()

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        bottle_id = (
            f"bottle-{bottle_number:03d}"
        )

        print(
            "=" * 72
        )

        print(
            bottle_id
        )

        print(
            "=" * 72
        )

        print(
            "Foreign passage: "
            f"{pair.first.start_frame}-"
            f"{pair.first.end_frame}"
        )

        print(
            "Downstream passage: "
            f"{pair.second.start_frame}-"
            f"{pair.second.end_frame}"
        )

        print(
            "Travel offset: "
            f"{pair.frame_offset} frames"
        )

        print()

        foreign_results: list[
            ForeignFrameResult
        ] = []

        for observation in (
            pair.first.observations
        ):
            frame_index = (
                observation.frame_index
            )

            result = (
                inspect_foreign_objects(
                    foreign_region.extract(
                        frames[
                            frame_index
                        ]
                    ),
                    foreign_config,
                )
            )

            foreign_results.append(
                ForeignFrameResult(
                    frame_index=(
                        frame_index
                    ),
                    status=(
                        result.status
                    ),
                    decision_black_pixels=(
                        result.decision_black_pixel_count
                    ),
                )
            )

        downstream_results: list[
            DownstreamFrameResult
        ] = []

        for observation in (
            pair.second.observations
        ):
            frame_index = (
                observation.frame_index
            )

            bottle_crop = (
                downstream_region.extract(
                    frames[
                        frame_index
                    ]
                )
            )

            liquid_result = (
                inspect_liquid_level(
                    bottle_crop,
                    liquid_config,
                )
            )

            glass_result = (
                inspect_glass(
                    bottle_crop
                )
            )

            cork_result = inspect_cork(
                cork_region.extract(
                    bottle_crop
                ),
                cork_template,
                cork_config,
            )

            body_result = inspect_cork(
                body_region.extract(
                    bottle_crop
                ),
                cork_template,
                cork_config,
            )

            downstream_results.append(
                DownstreamFrameResult(
                    frame_index=(
                        frame_index
                    ),
                    liquid_status=(
                        liquid_result.status
                    ),
                    liquid_level=(
                        liquid_result.absolute_level
                    ),
                    glass_status=(
                        glass_result.status
                    ),
                    glass_triggered_signals=(
                        glass_result.triggered_signal_count
                    ),
                    cork_score=(
                        cork_result.maximum_score
                    ),
                    body_control_score=(
                        body_result.maximum_score
                    ),
                )
            )

        print_foreign_summary(
            foreign_results
        )

        print_downstream_summary(
            downstream_results
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )