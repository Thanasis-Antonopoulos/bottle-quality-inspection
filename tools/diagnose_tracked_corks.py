"""Measure cork-template quality for individually tracked bottles."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    CorkResult,
    inspect_cork,
    render_cork_result,
)
from bottle_quality_inspection.inspections.foreign_object import (
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.tracking import (
    PassageAssociationResult,
    StationObservation,
    associate_passages_by_time,
    segment_station_passages,
)
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class CorkMeasurement:
    """Cork-template result for one tracked-bottle frame."""

    frame_index: int
    result: CorkResult


def build_parser() -> argparse.ArgumentParser:
    """Create the tracked-cork diagnostic parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure cork-template scores for each tracked bottle "
            "inside its downstream bottle ROI."
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
            "outputs/tracked-cork-diagnostics"
        ),
        help=(
            "Directory for annotated cork images. "
            "Default: outputs/tracked-cork-diagnostics."
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
    """Convert profile values into liquid-level configuration."""
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
) -> tuple[Path, np.ndarray]:
    """Read the configured cork template."""
    resolved = (
        template_path
        .expanduser()
        .resolve()
    )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Cork template does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise IsADirectoryError(
            f"Cork template is not a file: {resolved}"
        )

    template = cv2.imread(
        str(resolved),
        cv2.IMREAD_COLOR,
    )

    if template is None:
        raise RuntimeError(
            "OpenCV could not read cork template: "
            f"{resolved}"
        )

    return resolved, template


def detect_associations(
    video_path: Path,
    profile,
    *,
    max_missed_frames: int,
    minimum_detected_frames: int,
    minimum_frame_offset: int,
    maximum_frame_offset: int,
) -> PassageAssociationResult:
    """Detect and associate upstream and downstream bottle passages."""
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
    """Read selected source frames into memory."""
    frames: dict[
        int,
        np.ndarray
    ] = {}

    for video_frame in iter_video_frames(
        video_path
    ):
        if (
            video_frame.index
            not in frame_indices
        ):
            continue

        frames[
            video_frame.index
        ] = (
            video_frame.image.copy()
        )

    missing = (
        frame_indices
        - set(
            frames
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
            "Could not decode required frames: "
            f"{missing_text}"
        )

    return frames


def annotate_identity(
    image: np.ndarray,
    *,
    bottle_id: str,
    frame_index: int,
) -> np.ndarray:
    """Add tracked-bottle identity to an annotated crop."""
    annotated = image.copy()

    cv2.putText(
        annotated,
        (
            f"{bottle_id} | "
            f"frame {frame_index}"
        ),
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated


def save_image(
    destination: Path,
    image: np.ndarray,
) -> None:
    """Save one diagnostic image."""
    written = cv2.imwrite(
        str(destination),
        image,
    )

    if not written:
        raise RuntimeError(
            "Could not write cork diagnostic image: "
            f"{destination}"
        )


def print_measurements(
    bottle_id: str,
    measurements: list[CorkMeasurement],
    *,
    threshold: float,
    representative_frame: int,
) -> None:
    """Print multi-frame cork evidence for one tracked bottle."""
    scores = [
        measurement.result.maximum_score
        for measurement in measurements
    ]

    passing = [
        measurement
        for measurement in measurements
        if (
            measurement.result.maximum_score
            >= threshold
        )
    ]

    best = max(
        measurements,
        key=lambda measurement: (
            measurement.result.maximum_score
        ),
    )

    representative = next(
        measurement
        for measurement in measurements
        if (
            measurement.frame_index
            == representative_frame
        )
    )

    print(
        bottle_id
    )

    print(
        "-" * len(bottle_id)
    )

    print(
        "Frames evaluated: "
        f"{len(measurements)}"
    )

    print(
        "Frame range: "
        f"{measurements[0].frame_index}-"
        f"{measurements[-1].frame_index}"
    )

    print(
        "Minimum score: "
        f"{min(scores):.4f}"
    )

    print(
        "Median score: "
        f"{median(scores):.4f}"
    )

    print(
        "Maximum score: "
        f"{max(scores):.4f}"
    )

    print(
        "Frames meeting 0.80: "
        f"{len(passing)}/"
        f"{len(measurements)}"
    )

    print(
        "Representative frame: "
        f"{representative.frame_index}"
    )

    print(
        "Representative score: "
        f"{representative.result.maximum_score:.4f}"
    )

    print(
        "Representative status: "
        f"{representative.result.status.value}"
    )

    print(
        "Best frame: "
        f"{best.frame_index}"
    )

    print(
        "Best score: "
        f"{best.result.maximum_score:.4f}"
    )

    print(
        "Best local match location: "
        f"x={best.result.maximum_location[0]}, "
        f"y={best.result.maximum_location[1]}"
    )

    print()

    print(
        "Per-frame scores:"
    )

    for measurement in measurements:
        marker = (
            "PASS"
            if (
                measurement.result.maximum_score
                >= threshold
            )
            else "FAIL"
        )

        print(
            "  "
            f"frame="
            f"{measurement.frame_index:03d} "
            f"score="
            f"{measurement.result.maximum_score:.4f} "
            f"{marker}"
        )

    print()


def main() -> int:
    """Run tracked-bottle cork diagnostics."""
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

    cork_profile = (
        profile.cork
    )

    if not cork_profile.enabled:
        parser.error(
            "Cork inspection is disabled."
        )

    if (
        cork_profile.minimum_match_score
        is None
    ):
        parser.error(
            "Cork minimum_match_score "
            "is not configured."
        )

    template_path, template = (
        read_cork_template(
            cork_profile.template
        )
    )

    cork_config = CorkConfig(
        method=(
            cork_profile.method
        ),
        minimum_match_score=(
            cork_profile.minimum_match_score
        ),
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

    liquid_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    required_frames: set[int] = set()

    for pair in association.pairs:
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
        f"Cork template: "
        f"{template_path}"
    )

    print(
        "Bottle-specific cork ROI: "
        f"x={liquid_region.x}, "
        f"y={liquid_region.y}, "
        f"width={liquid_region.width}, "
        f"height={liquid_region.height}"
    )

    print(
        "Threshold: "
        f"{cork_profile.minimum_match_score:.4f}"
    )

    print()

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        bottle_id = (
            f"bottle-{bottle_number:03d}"
        )

        measurements: list[
            CorkMeasurement
        ] = []

        for observation in (
            pair.second.observations
        ):
            frame_index = (
                observation.frame_index
            )

            bottle_crop = (
                liquid_region.extract(
                    frames[
                        frame_index
                    ]
                )
                .copy()
            )

            result = inspect_cork(
                bottle_crop,
                template,
                cork_config,
            )

            measurements.append(
                CorkMeasurement(
                    frame_index=(
                        frame_index
                    ),
                    result=result,
                )
            )

        print_measurements(
            bottle_id,
            measurements,
            threshold=(
                cork_profile.minimum_match_score
            ),
            representative_frame=(
                pair.second.representative_frame
            ),
        )

        best = max(
            measurements,
            key=lambda measurement: (
                measurement.result.maximum_score
            ),
        )

        representative = next(
            measurement
            for measurement in measurements
            if (
                measurement.frame_index
                == pair.second.representative_frame
            )
        )

        for label, measurement in (
            (
                "representative",
                representative,
            ),
            (
                "best",
                best,
            ),
        ):
            bottle_crop = (
                liquid_region.extract(
                    frames[
                        measurement.frame_index
                    ]
                )
                .copy()
            )

            annotated = (
                render_cork_result(
                    bottle_crop,
                    measurement.result,
                )
            )

            annotated = annotate_identity(
                annotated,
                bottle_id=bottle_id,
                frame_index=(
                    measurement.frame_index
                ),
            )

            destination = (
                output_dir
                / (
                    f"{bottle_id}-"
                    f"{label}-"
                    f"frame-"
                    f"{measurement.frame_index:03d}.jpg"
                )
            )

            save_image(
                destination,
                annotated,
            )

    print(
        "Tracked cork diagnostics complete."
    )

    print(
        f"Images saved to: "
        f"{output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )