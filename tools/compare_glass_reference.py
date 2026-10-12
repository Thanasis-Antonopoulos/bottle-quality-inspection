"""Compare confirmed broken and normal bottles in the 2022 glass reference."""

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
from bottle_quality_inspection.roi import RegionOfInterest

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
    name="normal-bottle",
    x=1260,
    y=330,
    width=270,
    height=620,
)


@dataclass(frozen=True, slots=True)
class BottleComparison:
    """Glass-feature results for the labeled reference bottles."""

    broken: GlassFeatureResult
    normal: GlassFeatureResult


def build_parser() -> argparse.ArgumentParser:
    """Create the comparison command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare the confirmed broken left bottle "
            "with the normal center bottle in broken_glass.jpg."
        )
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
            "outputs/glass-reference-comparison"
        ),
        help=(
            "Directory for comparison images. "
            "Default: outputs/glass-reference-comparison."
        ),
    )

    return parser


def read_reference_image(
    image_path: Path,
) -> np.ndarray:
    """Read and validate the known 2022 glass reference."""
    resolved_path = (
        image_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Glass defect reference does not exist: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise IsADirectoryError(
            "Glass defect reference is not a file: "
            f"{resolved_path}"
        )

    image = cv2.imread(
        str(resolved_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "OpenCV could not read glass defect reference: "
            f"{resolved_path}"
        )

    height, width = image.shape[:2]

    if (
        width != REFERENCE_WIDTH
        or height != REFERENCE_HEIGHT
    ):
        raise RuntimeError(
            "This calibration tool expects the original "
            "2557 x 1287 broken_glass.jpg reference. "
            f"Received {width} x {height}."
        )

    return image


def save_image(
    output_dir: Path,
    filename: str,
    image: np.ndarray,
) -> None:
    """Save one comparison image."""
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
            "Could not write comparison image: "
            f"{destination}"
        )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    """Return a ratio unless the denominator is zero."""
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def format_ratio(
    ratio: float | None,
) -> str:
    """Format an optional comparison ratio."""
    if ratio is None:
        return "N/A"

    return f"{ratio:.3f}x"


def print_feature_result(
    label: str,
    result: GlassFeatureResult,
) -> None:
    """Print one bottle's measured glass features."""
    print(label)
    print(
        "-" * len(label)
    )

    print(
        "Interior edge pixels: "
        f"{result.edge_pixel_count}"
    )

    print(
        "Interior edge density: "
        f"{result.edge_density:.6f}"
    )

    print(
        "Detected Hough lines: "
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
        "Total horizontal-line length: "
        f"{result.total_horizontal_line_length:.2f} px"
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


def print_comparison(
    comparison: BottleComparison,
) -> None:
    """Print numerical differences between broken and normal bottles."""
    broken = comparison.broken
    normal = comparison.normal

    print(
        "BROKEN / NORMAL COMPARISON"
    )
    print(
        "--------------------------"
    )

    print(
        "Edge-density ratio: "
        f"{format_ratio(safe_ratio(
            broken.edge_density,
            normal.edge_density,
        ))}"
    )

    print(
        "Horizontal-line-count difference: "
        f"{broken.horizontal_line_count - normal.horizontal_line_count}"
    )

    print(
        "Longest-horizontal-line difference: "
        f"{(
            broken.longest_horizontal_line_length
            - normal.longest_horizontal_line_length
        ):.2f} px"
    )

    print(
        "Total-horizontal-length difference: "
        f"{(
            broken.total_horizontal_line_length
            - normal.total_horizontal_line_length
        ):.2f} px"
    )

    print(
        "Maximum-row-edge-ratio: "
        f"{format_ratio(safe_ratio(
            broken.maximum_row_edge_ratio,
            normal.maximum_row_edge_ratio,
        ))}"
    )


def draw_reference_regions(
    image: np.ndarray,
) -> np.ndarray:
    """Draw the two labeled calibration crops on the source reference."""
    annotated = image.copy()

    annotated = BROKEN_BOTTLE_REGION.draw(
        annotated,
        color=(0, 0, 255),
        thickness=3,
        show_label=True,
    )

    annotated = NORMAL_BOTTLE_REGION.draw(
        annotated,
        color=(0, 255, 0),
        thickness=3,
        show_label=True,
    )

    return annotated


def main() -> int:
    """Run the labeled broken-vs-normal glass comparison."""
    parser = build_parser()
    args = parser.parse_args()

    inspection_config = load_inspection_config(
        args.config
    )

    glass_profile = (
        inspection_config.glass
    )

    if not glass_profile.enabled:
        parser.error(
            "Glass inspection is disabled "
            "in the selected configuration."
        )

    if glass_profile.defect_reference is None:
        parser.error(
            "No glass defect reference is configured."
        )

    reference_path = (
        glass_profile.defect_reference
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

    reference_image = read_reference_image(
        reference_path
    )

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

    feature_config = GlassFeatureConfig()

    broken_result = extract_glass_features(
        broken_crop,
        feature_config,
    )

    normal_result = extract_glass_features(
        normal_crop,
        feature_config,
    )

    comparison = BottleComparison(
        broken=broken_result,
        normal=normal_result,
    )

    broken_annotated = render_glass_features(
        broken_crop,
        broken_result,
    )

    normal_annotated = render_glass_features(
        normal_crop,
        normal_result,
    )

    reference_annotated = draw_reference_regions(
        reference_image
    )

    print(
        f"Profile: "
        f"{inspection_config.profile}"
    )

    print(
        f"Reference image: "
        f"{reference_path}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print()

    print_feature_result(
        "CONFIRMED BROKEN LEFT BOTTLE",
        broken_result,
    )

    print_feature_result(
        "NORMAL CENTER BOTTLE",
        normal_result,
    )

    print_comparison(
        comparison
    )

    save_image(
        output_dir,
        "00-reference-regions.jpg",
        reference_annotated,
    )

    save_image(
        output_dir,
        "01-broken-bottle.jpg",
        broken_crop,
    )

    save_image(
        output_dir,
        "02-normal-bottle.jpg",
        normal_crop,
    )

    save_image(
        output_dir,
        "03-broken-features.jpg",
        broken_annotated,
    )

    save_image(
        output_dir,
        "04-normal-features.jpg",
        normal_annotated,
    )

    save_image(
        output_dir,
        "05-broken-interior-edges.jpg",
        broken_result.interior_edges,
    )

    save_image(
        output_dir,
        "06-normal-interior-edges.jpg",
        normal_result.interior_edges,
    )

    print()
    print(
        "Saved 7 comparison images."
    )

    print()
    print(
        "No glass defect threshold was created."
    )

    print(
        "These measurements are calibration evidence only."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )