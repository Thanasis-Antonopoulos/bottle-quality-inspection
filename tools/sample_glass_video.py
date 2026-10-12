"""Sample centered bottle passages for glass-integrity calibration."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.glass import (
    GlassFeatureConfig,
    GlassFeatureResult,
    extract_glass_features,
    render_glass_features,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class DetectedFrame:
    """One frame where the liquid station contains a measurable bottle."""

    frame_index: int
    timestamp_seconds: float
    liquid_level: float
    detected_line_count: int


@dataclass(frozen=True, slots=True)
class BottlePassage:
    """One contiguous sequence of frames containing a bottle."""

    start_frame: int
    end_frame: int
    representative_frame: DetectedFrame

    @property
    def frame_count(self) -> int:
        """Return the number of detected frames in the passage."""
        return (
            self.end_frame
            - self.start_frame
            + 1
        )


@dataclass(frozen=True, slots=True)
class GlassCandidate:
    """Measured glass features for one representative bottle frame."""

    candidate_number: int
    passage: BottlePassage
    result: GlassFeatureResult


def build_parser() -> argparse.ArgumentParser:
    """Create the calibration command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Find bottle passages using the liquid-level station "
            "and measure glass features on one centered frame "
            "from each passage."
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
            "outputs/glass-video-candidates"
        ),
        help=(
            "Directory for candidate images. "
            "Default: outputs/glass-video-candidates."
        ),
    )

    return parser


def create_liquid_config(
    inspection_config,
) -> LiquidLevelConfig:
    """Convert profile values into liquid-level algorithm configuration."""
    liquid = inspection_config.liquid_level

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


def detect_bottle_frames(
    video_path: Path,
    liquid_region: RegionOfInterest,
    liquid_config: LiquidLevelConfig,
) -> list[DetectedFrame]:
    """Find video frames where the liquid station detects a bottle level."""
    detections: list[
        DetectedFrame
    ] = []

    for video_frame in iter_video_frames(
        video_path
    ):
        roi_image = liquid_region.extract(
            video_frame.image
        )

        result = inspect_liquid_level(
            roi_image,
            liquid_config,
        )

        if (
            result.level_detected
            and result.absolute_level is not None
        ):
            detections.append(
                DetectedFrame(
                    frame_index=(
                        video_frame.index
                    ),
                    timestamp_seconds=(
                        video_frame.timestamp_seconds
                    ),
                    liquid_level=(
                        result.absolute_level
                    ),
                    detected_line_count=(
                        result.detected_line_count
                    ),
                )
            )

    return detections


def segment_passages(
    detections: list[DetectedFrame],
) -> list[BottlePassage]:
    """Group consecutive detected frames into bottle passages."""
    if not detections:
        return []

    grouped: list[
        list[DetectedFrame]
    ] = []

    current_group = [
        detections[0]
    ]

    for detection in detections[1:]:
        previous = current_group[-1]

        if (
            detection.frame_index
            == previous.frame_index + 1
        ):
            current_group.append(
                detection
            )
        else:
            grouped.append(
                current_group
            )

            current_group = [
                detection
            ]

    grouped.append(
        current_group
    )

    passages: list[
        BottlePassage
    ] = []

    for group in grouped:
        representative = group[
            len(group) // 2
        ]

        passages.append(
            BottlePassage(
                start_frame=(
                    group[0].frame_index
                ),
                end_frame=(
                    group[-1].frame_index
                ),
                representative_frame=(
                    representative
                ),
            )
        )

    return passages


def save_image(
    output_dir: Path,
    filename: str,
    image: np.ndarray,
) -> None:
    """Save one calibration image."""
    destination = (
        output_dir
        / filename
    )

    written = cv2.imwrite(
        str(destination),
        image,
    )

    if not written:
        raise RuntimeError(
            "Could not write glass calibration image: "
            f"{destination}"
        )


def line_ratio(
    length: float,
    result: GlassFeatureResult,
) -> float:
    """Normalize a horizontal-line length by interior width."""
    width = (
        result.interior_edges.shape[1]
    )

    if width <= 0:
        return 0.0

    return (
        length
        / width
    )


def print_candidate(
    candidate: GlassCandidate,
) -> None:
    """Print glass measurements for one bottle candidate."""
    passage = candidate.passage
    selected = (
        passage.representative_frame
    )

    result = candidate.result

    longest_ratio = line_ratio(
        result.longest_horizontal_line_length,
        result,
    )

    total_ratio = line_ratio(
        result.total_horizontal_line_length,
        result,
    )

    print(
        f"CANDIDATE {candidate.candidate_number}"
    )

    print(
        "-" * 40
    )

    print(
        "Bottle passage: "
        f"frames {passage.start_frame}-"
        f"{passage.end_frame} "
        f"({passage.frame_count} frames)"
    )

    print(
        "Representative frame: "
        f"{selected.frame_index}"
    )

    print(
        "Timestamp: "
        f"{selected.timestamp_seconds:.3f} s"
    )

    print(
        "Liquid level: "
        f"{selected.liquid_level:.2f}"
    )

    print(
        "Liquid Hough lines: "
        f"{selected.detected_line_count}"
    )

    print(
        "Glass edge pixels: "
        f"{result.edge_pixel_count}"
    )

    print(
        "Glass edge density: "
        f"{result.edge_density:.6f}"
    )

    print(
        "Glass Hough lines: "
        f"{result.detected_line_count}"
    )

    print(
        "Near-horizontal lines: "
        f"{result.horizontal_line_count}"
    )

    print(
        "Longest horizontal line: "
        f"{result.longest_horizontal_line_length:.2f} px"
    )

    print(
        "Longest horizontal / interior width: "
        f"{longest_ratio:.6f}"
    )

    print(
        "Total horizontal-line length: "
        f"{result.total_horizontal_line_length:.2f} px"
    )

    print(
        "Total horizontal / interior width: "
        f"{total_ratio:.6f}"
    )

    print(
        "Maximum row edge count: "
        f"{result.maximum_row_edge_count}"
    )

    print(
        "Maximum row edge ratio: "
        f"{result.maximum_row_edge_ratio:.6f}"
    )

    print()


def save_candidate_images(
    video_path: Path,
    passages: list[BottlePassage],
    liquid_region: RegionOfInterest,
    glass_config: GlassFeatureConfig,
    output_dir: Path,
) -> list[GlassCandidate]:
    """Read representative frames and save their glass diagnostics."""
    selected_indices = {
        passage.representative_frame.frame_index: (
            index,
            passage,
        )
        for index, passage in enumerate(
            passages,
            start=1,
        )
    }

    candidates: list[
        GlassCandidate
    ] = []

    for video_frame in iter_video_frames(
        video_path
    ):
        selected = selected_indices.get(
            video_frame.index
        )

        if selected is None:
            continue

        candidate_number, passage = (
            selected
        )

        bottle_crop = (
            liquid_region.extract(
                video_frame.image
            )
            .copy()
        )

        result = extract_glass_features(
            bottle_crop,
            glass_config,
        )

        feature_image = render_glass_features(
            bottle_crop,
            result,
        )

        full_frame = (
            video_frame.image.copy()
        )

        full_frame = liquid_region.draw(
            full_frame,
            color=(0, 255, 0),
            thickness=3,
            show_label=True,
        )

        prefix = (
            f"candidate-"
            f"{candidate_number:02d}-"
            f"frame-"
            f"{video_frame.index:03d}"
        )

        save_image(
            output_dir,
            f"{prefix}-full.jpg",
            full_frame,
        )

        save_image(
            output_dir,
            f"{prefix}-crop.jpg",
            bottle_crop,
        )

        save_image(
            output_dir,
            f"{prefix}-features.jpg",
            feature_image,
        )

        save_image(
            output_dir,
            f"{prefix}-edges.jpg",
            result.interior_edges,
        )

        candidates.append(
            GlassCandidate(
                candidate_number=(
                    candidate_number
                ),
                passage=passage,
                result=result,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.candidate_number
        )
    )

    return candidates


def print_summary(
    candidates: list[GlassCandidate],
) -> None:
    """Print aggregate calibration ranges without classifying them."""
    if not candidates:
        return

    edge_densities = [
        candidate.result.edge_density
        for candidate in candidates
    ]

    row_ratios = [
        candidate.result.maximum_row_edge_ratio
        for candidate in candidates
    ]

    horizontal_counts = [
        candidate.result.horizontal_line_count
        for candidate in candidates
    ]

    longest_ratios = [
        line_ratio(
            candidate.result.longest_horizontal_line_length,
            candidate.result,
        )
        for candidate in candidates
    ]

    print(
        "VIDEO CANDIDATE RANGE"
    )

    print(
        "---------------------"
    )

    print(
        "Edge density: "
        f"{min(edge_densities):.6f} "
        "to "
        f"{max(edge_densities):.6f}"
    )

    print(
        "Near-horizontal lines: "
        f"{min(horizontal_counts)} "
        "to "
        f"{max(horizontal_counts)}"
    )

    print(
        "Longest-line ratio: "
        f"{min(longest_ratios):.6f} "
        "to "
        f"{max(longest_ratios):.6f}"
    )

    print(
        "Maximum-row-edge ratio: "
        f"{min(row_ratios):.6f} "
        "to "
        f"{max(row_ratios):.6f}"
    )

    print()


def main() -> int:
    """Run video glass-candidate sampling."""
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

    inspection_config = load_inspection_config(
        args.config
    )

    liquid_profile = (
        inspection_config.liquid_level
    )

    if not liquid_profile.enabled:
        parser.error(
            "Liquid-level inspection must be enabled "
            "to identify bottle passages."
        )

    liquid_region = (
        liquid_profile.roi.to_roi()
    )

    liquid_config = create_liquid_config(
        inspection_config
    )

    glass_config = (
        GlassFeatureConfig()
    )

    print(
        f"Profile: "
        f"{inspection_config.profile}"
    )

    print(
        f"Video: "
        f"{video_path}"
    )

    print(
        "Bottle sampling ROI: "
        f"x={liquid_region.x}, "
        f"y={liquid_region.y}, "
        f"width={liquid_region.width}, "
        f"height={liquid_region.height}"
    )

    print()
    print(
        "Detecting bottle passages..."
    )
    print()

    detections = detect_bottle_frames(
        video_path,
        liquid_region,
        liquid_config,
    )

    passages = segment_passages(
        detections
    )

    print(
        "Frames with detected liquid level: "
        f"{len(detections)}"
    )

    print(
        "Bottle passages detected: "
        f"{len(passages)}"
    )

    print()

    if not passages:
        print(
            "No bottle candidates were found."
        )

        return 0

    for index, passage in enumerate(
        passages,
        start=1,
    ):
        representative = (
            passage.representative_frame
        )

        print(
            f"Passage {index}: "
            f"frames "
            f"{passage.start_frame}-"
            f"{passage.end_frame} "
            f"→ representative frame "
            f"{representative.frame_index}"
        )

    print()
    print(
        "Measuring glass features..."
    )
    print()

    candidates = save_candidate_images(
        video_path,
        passages,
        liquid_region,
        glass_config,
        output_dir,
    )

    for candidate in candidates:
        print_candidate(
            candidate
        )

    print_summary(
        candidates
    )

    print(
        f"Saved candidate images to: "
        f"{output_dir}"
    )

    print()

    print(
        "No CLEAR/DEFECTIVE glass decision was made."
    )

    print(
        "The exported bottle candidates must be "
        "reviewed before calibration."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )