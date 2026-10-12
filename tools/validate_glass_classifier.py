"""Validate the calibrated 2026 glass classifier against known examples."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.glass import (
    GlassDecisionConfig,
    GlassFeatureConfig,
    GlassResult,
    GlassStatus,
    inspect_glass,
    render_glass_result,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import iter_video_frames

REFERENCE_WIDTH = 2557
REFERENCE_HEIGHT = 1287


BROKEN_BOTTLE_REGION = RegionOfInterest(
    name="confirmed-broken-bottle",
    x=390,
    y=330,
    width=270,
    height=620,
)


NORMAL_BOTTLE_REGION = RegionOfInterest(
    name="normal-reference-bottle",
    x=1260,
    y=330,
    width=270,
    height=620,
)


VIDEO_CANDIDATE_FRAMES = (
    2,
    44,
    59,
    85,
)


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """Expected and actual result for one glass calibration case."""

    name: str
    expected_status: GlassStatus
    result: GlassResult

    @property
    def passed(self) -> bool:
        """Return whether the classifier matched the expected label."""
        return (
            self.result.status
            is self.expected_status
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the validation command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the calibrated 2026 glass classifier "
            "against labeled reference and video examples."
        )
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the original 2022 source video.",
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
            "outputs/glass-classifier-validation"
        ),
        help=(
            "Directory for annotated validation images. "
            "Default: outputs/glass-classifier-validation."
        ),
    )

    return parser


def read_reference_image(
    image_path: Path,
) -> np.ndarray:
    """Read and validate the original broken-glass reference."""
    resolved_path = (
        image_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Glass reference does not exist: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise IsADirectoryError(
            "Glass reference is not a file: "
            f"{resolved_path}"
        )

    image = cv2.imread(
        str(resolved_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "OpenCV could not read glass reference: "
            f"{resolved_path}"
        )

    height, width = image.shape[:2]

    if (
        width != REFERENCE_WIDTH
        or height != REFERENCE_HEIGHT
    ):
        raise RuntimeError(
            "The glass calibration expects the original "
            "2557 x 1287 broken_glass.jpg reference. "
            f"Received {width} x {height}."
        )

    return image


def save_image(
    output_dir: Path,
    filename: str,
    image: np.ndarray,
) -> None:
    """Save one annotated glass-validation image."""
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
            "Could not write validation image: "
            f"{destination}"
        )


def print_result(
    case: ValidationCase,
) -> None:
    """Print one glass-classifier validation result."""
    result = case.result
    features = result.features

    verdict = (
        "PASS"
        if case.passed
        else "FAIL"
    )

    signals = (
        ", ".join(
            result.triggered_signals
        )
        if result.triggered_signals
        else "none"
    )

    print(
        f"{case.name}"
    )

    print(
        "-" * len(case.name)
    )

    print(
        f"Expected: "
        f"{case.expected_status.value}"
    )

    print(
        f"Actual: "
        f"{result.status.value}"
    )

    print(
        f"Validation: "
        f"{verdict}"
    )

    print(
        "Triggered signals: "
        f"{result.triggered_signal_count}/"
        f"{result.required_signal_count}"
    )

    print(
        f"Signals: "
        f"{signals}"
    )

    print(
        "Edge density: "
        f"{features.edge_density:.6f}"
    )

    print(
        "Horizontal lines: "
        f"{features.horizontal_line_count}"
    )

    print(
        "Longest horizontal line: "
        f"{features.longest_horizontal_line_length:.2f} px"
    )

    print(
        "Longest horizontal ratio: "
        f"{features.longest_horizontal_line_ratio:.6f}"
    )

    print(
        "Maximum row-edge ratio: "
        f"{features.maximum_row_edge_ratio:.6f}"
    )

    print()


def validate_reference_cases(
    reference_image: np.ndarray,
    feature_config: GlassFeatureConfig,
    decision_config: GlassDecisionConfig,
    output_dir: Path,
) -> list[ValidationCase]:
    """Validate the confirmed broken and normal reference bottles."""
    broken_crop = (
        BROKEN_BOTTLE_REGION.extract(
            reference_image
        )
        .copy()
    )

    normal_crop = (
        NORMAL_BOTTLE_REGION.extract(
            reference_image
        )
        .copy()
    )

    broken_result = inspect_glass(
        broken_crop,
        feature_config,
        decision_config,
    )

    normal_result = inspect_glass(
        normal_crop,
        feature_config,
        decision_config,
    )

    save_image(
        output_dir,
        "reference-broken-result.jpg",
        render_glass_result(
            broken_crop,
            broken_result,
        ),
    )

    save_image(
        output_dir,
        "reference-normal-result.jpg",
        render_glass_result(
            normal_crop,
            normal_result,
        ),
    )

    return [
        ValidationCase(
            name="CONFIRMED BROKEN REFERENCE",
            expected_status=(
                GlassStatus.DEFECTIVE
            ),
            result=broken_result,
        ),
        ValidationCase(
            name="NORMAL REFERENCE",
            expected_status=(
                GlassStatus.CLEAR
            ),
            result=normal_result,
        ),
    ]


def validate_video_cases(
    video_path: Path,
    bottle_region: RegionOfInterest,
    feature_config: GlassFeatureConfig,
    decision_config: GlassDecisionConfig,
    output_dir: Path,
) -> list[ValidationCase]:
    """Validate the four visually normal video candidates."""
    requested_frames = set(
        VIDEO_CANDIDATE_FRAMES
    )

    results: dict[
        int,
        GlassResult,
    ] = {}

    for video_frame in iter_video_frames(
        video_path
    ):
        if (
            video_frame.index
            not in requested_frames
        ):
            continue

        bottle_crop = (
            bottle_region.extract(
                video_frame.image
            )
            .copy()
        )

        result = inspect_glass(
            bottle_crop,
            feature_config,
            decision_config,
        )

        results[
            video_frame.index
        ] = result

        save_image(
            output_dir,
            (
                "video-frame-"
                f"{video_frame.index:03d}-result.jpg"
            ),
            render_glass_result(
                bottle_crop,
                result,
            ),
        )

    missing_frames = (
        requested_frames
        - set(
            results
        )
    )

    if missing_frames:
        missing_text = ", ".join(
            str(frame)
            for frame in sorted(
                missing_frames
            )
        )

        raise RuntimeError(
            "Could not decode expected calibration frames: "
            f"{missing_text}"
        )

    cases: list[
        ValidationCase
    ] = []

    for candidate_number, frame_index in enumerate(
        VIDEO_CANDIDATE_FRAMES,
        start=1,
    ):
        cases.append(
            ValidationCase(
                name=(
                    f"NORMAL VIDEO CANDIDATE "
                    f"{candidate_number} "
                    f"(FRAME {frame_index})"
                ),
                expected_status=(
                    GlassStatus.CLEAR
                ),
                result=results[
                    frame_index
                ],
            )
        )

    return cases


def print_configuration(
    decision_config: GlassDecisionConfig,
) -> None:
    """Print the provisional calibrated classifier settings."""
    print(
        "2026 GLASS DECISION CONFIGURATION"
    )

    print(
        "---------------------------------"
    )

    print(
        "Minimum edge density: "
        f"{decision_config.minimum_edge_density:.6f}"
    )

    print(
        "Minimum horizontal lines: "
        f"{decision_config.minimum_horizontal_line_count}"
    )

    print(
        "Minimum longest-line ratio: "
        f"{decision_config.minimum_longest_horizontal_line_ratio:.6f}"
    )

    print(
        "Minimum maximum-row ratio: "
        f"{decision_config.minimum_maximum_row_edge_ratio:.6f}"
    )

    print(
        "Required triggered signals: "
        f"{decision_config.minimum_triggered_signals}/4"
    )

    print()


def main() -> int:
    """Run the complete glass-classifier calibration validation."""
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

    glass_profile = (
        inspection_config.glass
    )

    liquid_profile = (
        inspection_config.liquid_level
    )

    if glass_profile.defect_reference is None:
        parser.error(
            "No glass defect reference is configured."
        )

    reference_image = read_reference_image(
        glass_profile.defect_reference
    )

    bottle_region = (
        liquid_profile.roi.to_roi()
    )

    feature_config = (
        GlassFeatureConfig()
    )

    decision_config = (
        GlassDecisionConfig()
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
        "Glass reference: "
        f"{glass_profile.defect_reference.expanduser().resolve()}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print()

    print_configuration(
        decision_config
    )

    cases = []

    cases.extend(
        validate_reference_cases(
            reference_image,
            feature_config,
            decision_config,
            output_dir,
        )
    )

    cases.extend(
        validate_video_cases(
            video_path,
            bottle_region,
            feature_config,
            decision_config,
            output_dir,
        )
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=================="
    )

    print()

    for case in cases:
        print_result(
            case
        )

    passed = sum(
        case.passed
        for case in cases
    )

    total = len(
        cases
    )

    failed = (
        total
        - passed
    )

    print(
        "SUMMARY"
    )

    print(
        "-------"
    )

    print(
        f"Cases: "
        f"{total}"
    )

    print(
        f"Passed: "
        f"{passed}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print()

    if failed:
        print(
            "GLASS CLASSIFIER VALIDATION: FAILED"
        )

        return 1

    print(
        "GLASS CLASSIFIER VALIDATION: PASSED"
    )

    print()

    print(
        "The calibrated classifier correctly handled "
        "all currently labeled calibration examples."
    )

    print(
        "This remains a project-specific 2026 baseline, "
        "not a universal manufacturing threshold."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )