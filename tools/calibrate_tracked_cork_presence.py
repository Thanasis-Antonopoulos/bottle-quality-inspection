"""Calibrate bottle-specific cork presence using tracked bottle passages."""

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


@dataclass(frozen=True, slots=True)
class CorkPresenceMeasurement:
    """Positive and negative-control cork scores for one frame."""

    frame_index: int

    cork_score: float
    body_score: float

    cork_match_location: tuple[int, int]
    body_match_location: tuple[int, int]

    @property
    def score_margin(self) -> float:
        """Return cork-region score minus body-control score."""
        return (
            self.cork_score
            - self.body_score
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the cork-presence calibration parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare cork-template scores in the expected cork "
            "area against a negative-control bottle-body area."
        )
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the original source video.",
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
            "outputs/tracked-cork-presence"
        ),
        help=(
            "Directory for calibration images. "
            "Default: outputs/tracked-cork-presence."
        ),
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


def read_template(
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
) -> PassageAssociationResult:
    """Detect and associate foreign and liquid/glass bottle passages."""
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

        foreign_result = inspect_foreign_objects(
            foreign_crop,
            foreign_config,
        )

        liquid_result = inspect_liquid_level(
            liquid_crop,
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

    liquid_passages = segment_station_passages(
        liquid_observations,
        max_missed_frames=1,
        minimum_detected_frames=2,
    )

    return associate_passages_by_time(
        foreign_passages,
        liquid_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=30,
    )


def collect_frames(
    video_path: Path,
    required_frames: set[int],
) -> dict[int, np.ndarray]:
    """Decode only the frames required for calibration."""
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
            "Could not decode calibration frames: "
            + ", ".join(
                str(frame)
                for frame in sorted(
                    missing
                )
            )
        )

    return frames


def top_k_median(
    values: list[float],
    *,
    k: int,
) -> float:
    """Return the median of the strongest k measurements."""
    if not values:
        raise ValueError(
            "Cannot calculate top-k median from no values."
        )

    strongest = sorted(
        values,
        reverse=True,
    )[
        : min(
            k,
            len(values),
        )
    ]

    return float(
        median(
            strongest
        )
    )


def annotate_regions(
    bottle_crop: np.ndarray,
    cork_region: RegionOfInterest,
    body_region: RegionOfInterest,
    *,
    bottle_id: str,
    frame_index: int,
) -> np.ndarray:
    """Draw positive and negative-control search regions."""
    annotated = bottle_crop.copy()

    annotated = cork_region.draw(
        annotated,
        color=(0, 255, 0),
        thickness=2,
        show_label=True,
    )

    annotated = body_region.draw(
        annotated,
        color=(0, 0, 255),
        thickness=2,
        show_label=True,
    )

    cv2.putText(
        annotated,
        (
            f"{bottle_id} | "
            f"frame {frame_index}"
        ),
        (10, 30),
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
    """Save one calibration image."""
    written = cv2.imwrite(
        str(destination),
        image,
    )

    if not written:
        raise RuntimeError(
            "Could not write calibration image: "
            f"{destination}"
        )


def main() -> int:
    """Run cork-presence calibration."""
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

    template_path, template = (
        read_template(
            cork_profile.template
        )
    )

    legacy_config = CorkConfig(
        method=(
            cork_profile.method
        ),
        minimum_match_score=(
            cork_profile.minimum_match_score
            if (
                cork_profile.minimum_match_score
                is not None
            )
            else 0.80
        ),
    )

    association = detect_associations(
        video_path,
        profile,
    )

    bottle_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    # The tracked bottle ROI is 200 x 700.
    #
    # Existing tracked-cork matches appeared around local
    # y=88..108. We therefore isolate the upper portion of
    # the bottle while still allowing movement between frames.
    cork_region = RegionOfInterest(
        name="cork-search",
        x=0,
        y=40,
        width=200,
        height=200,
    )

    # This region contains bottle body rather than cork and acts
    # as a real-image negative control for template correlation.
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
        "Tracked bottle ROI: "
        f"x={bottle_region.x}, "
        f"y={bottle_region.y}, "
        f"width={bottle_region.width}, "
        f"height={bottle_region.height}"
    )

    print(
        "Cork search region: "
        f"x={cork_region.x}, "
        f"y={cork_region.y}, "
        f"width={cork_region.width}, "
        f"height={cork_region.height}"
    )

    print(
        "Body negative-control region: "
        f"x={body_region.x}, "
        f"y={body_region.y}, "
        f"width={body_region.width}, "
        f"height={body_region.height}"
    )

    print()

    all_cork_scores: list[
        float
    ] = []

    all_body_scores: list[
        float
    ] = []

    bottle_top3_scores: list[
        float
    ] = []

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        bottle_id = (
            f"bottle-{bottle_number:03d}"
        )

        measurements: list[
            CorkPresenceMeasurement
        ] = []

        for observation in (
            pair.second.observations
        ):
            frame_index = (
                observation.frame_index
            )

            bottle_crop = (
                bottle_region.extract(
                    frames[
                        frame_index
                    ]
                )
                .copy()
            )

            cork_crop = (
                cork_region.extract(
                    bottle_crop
                )
            )

            body_crop = (
                body_region.extract(
                    bottle_crop
                )
            )

            cork_result = inspect_cork(
                cork_crop,
                template,
                legacy_config,
            )

            body_result = inspect_cork(
                body_crop,
                template,
                legacy_config,
            )

            measurement = (
                CorkPresenceMeasurement(
                    frame_index=(
                        frame_index
                    ),
                    cork_score=(
                        cork_result.maximum_score
                    ),
                    body_score=(
                        body_result.maximum_score
                    ),
                    cork_match_location=(
                        cork_result.maximum_location
                    ),
                    body_match_location=(
                        body_result.maximum_location
                    ),
                )
            )

            measurements.append(
                measurement
            )

            all_cork_scores.append(
                measurement.cork_score
            )

            all_body_scores.append(
                measurement.body_score
            )

        cork_scores = [
            item.cork_score
            for item in measurements
        ]

        body_scores = [
            item.body_score
            for item in measurements
        ]

        margins = [
            item.score_margin
            for item in measurements
        ]

        top3_cork = top_k_median(
            cork_scores,
            k=3,
        )

        top3_body = top_k_median(
            body_scores,
            k=3,
        )

        bottle_top3_scores.append(
            top3_cork
        )

        best = max(
            measurements,
            key=lambda item: (
                item.cork_score
            ),
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
            "Frames evaluated: "
            f"{len(measurements)}"
        )

        print(
            "Cork score range: "
            f"{min(cork_scores):.4f} "
            "to "
            f"{max(cork_scores):.4f}"
        )

        print(
            "Body-control score range: "
            f"{min(body_scores):.4f} "
            "to "
            f"{max(body_scores):.4f}"
        )

        print(
            "Best-frame margin: "
            f"{best.score_margin:.4f}"
        )

        print(
            "Median frame margin: "
            f"{median(margins):.4f}"
        )

        print(
            "Top-3 cork median: "
            f"{top3_cork:.4f}"
        )

        print(
            "Top-3 body median: "
            f"{top3_body:.4f}"
        )

        print(
            "Top-3 separation: "
            f"{top3_cork - top3_body:.4f}"
        )

        print(
            "Best cork frame: "
            f"{best.frame_index}"
        )

        print(
            "Best cork score: "
            f"{best.cork_score:.4f}"
        )

        print()

        print(
            "Per-frame calibration:"
        )

        for item in measurements:
            print(
                "  "
                f"frame="
                f"{item.frame_index:03d} "
                f"cork="
                f"{item.cork_score:.4f} "
                f"body="
                f"{item.body_score:.4f} "
                f"margin="
                f"{item.score_margin:.4f}"
            )

        print()

        bottle_crop = (
            bottle_region.extract(
                frames[
                    best.frame_index
                ]
            )
            .copy()
        )

        annotated = annotate_regions(
            bottle_crop,
            cork_region,
            body_region,
            bottle_id=bottle_id,
            frame_index=(
                best.frame_index
            ),
        )

        save_image(
            output_dir
            / (
                f"{bottle_id}-"
                f"best-frame-"
                f"{best.frame_index:03d}.jpg"
            ),
            annotated,
        )

    print(
        "GLOBAL CALIBRATION SUMMARY"
    )

    print(
        "--------------------------"
    )

    print(
        "All cork-region scores: "
        f"{min(all_cork_scores):.4f} "
        "to "
        f"{max(all_cork_scores):.4f}"
    )

    print(
        "All body-control scores: "
        f"{min(all_body_scores):.4f} "
        "to "
        f"{max(all_body_scores):.4f}"
    )

    print(
        "Per-bottle top-3 cork median range: "
        f"{min(bottle_top3_scores):.4f} "
        "to "
        f"{max(bottle_top3_scores):.4f}"
    )

    print()

    print(
        "No new cork threshold was created."
    )

    print(
        "This diagnostic measures whether normal cork evidence "
        "separates from real bottle-body negative controls."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )