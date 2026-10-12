"""Run legacy glass preprocessing on the configured defect reference."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspections.glass import (
    GlassPreprocessConfig,
    preprocess_glass,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the diagnostic command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the reconstructed 2022 glass preprocessing "
            "pipeline and save all intermediate results."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/legacy-2022.toml"),
        help=(
            "Inspection configuration profile. "
            "Default: configs/legacy-2022.toml."
        ),
    )

    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help=(
            "Optional image to inspect. "
            "When omitted, the configured glass defect reference is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/glass-diagnostics"),
        help=(
            "Directory for generated preprocessing images. "
            "Default: outputs/glass-diagnostics."
        ),
    )

    return parser


def read_image(
    image_path: Path,
) -> np.ndarray:
    """Read and validate one BGR image."""
    resolved_path = (
        image_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Glass diagnostic image does not exist: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise IsADirectoryError(
            f"Glass diagnostic path is not a file: "
            f"{resolved_path}"
        )

    image = cv2.imread(
        str(resolved_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "OpenCV could not read glass diagnostic image: "
            f"{resolved_path}"
        )

    return image


def count_nonzero(
    image: np.ndarray,
) -> int:
    """Count non-zero pixels in a single-channel image."""
    return int(
        cv2.countNonZero(
            image
        )
    )


def pixel_percentage(
    count: int,
    image: np.ndarray,
) -> float:
    """Return a pixel count as a percentage of the image."""
    total = int(
        image.shape[0]
        * image.shape[1]
    )

    if total == 0:
        return 0.0

    return (
        count
        / total
        * 100.0
    )


def save_image(
    output_dir: Path,
    filename: str,
    image: np.ndarray,
) -> None:
    """Save one diagnostic image."""
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
            f"Could not write diagnostic image: "
            f"{destination}"
        )


def main() -> int:
    """Run glass preprocessing diagnostics."""
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

    if args.input_image is not None:
        image_path = (
            args.input_image
            .expanduser()
            .resolve()
        )
    else:
        if glass_profile.defect_reference is None:
            parser.error(
                "No glass defect reference is configured."
            )

        image_path = (
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

    image = read_image(
        image_path
    )

    config = GlassPreprocessConfig()

    result = preprocess_glass(
        image,
        config,
    )

    average_edge_pixels = count_nonzero(
        result.average_edges
    )

    gaussian_edge_pixels = count_nonzero(
        result.gaussian_edges
    )

    binary_white_pixels = count_nonzero(
        result.binary_thresholded
    )

    binary_inverted_white_pixels = count_nonzero(
        result.binary_inverted
    )

    otsu_white_pixels = count_nonzero(
        result.otsu_inverted
    )

    global_binary_white_pixels = count_nonzero(
        result.global_binary
    )

    adaptive_mean_white_pixels = count_nonzero(
        result.adaptive_mean
    )

    adaptive_gaussian_white_pixels = count_nonzero(
        result.adaptive_gaussian
    )

    print(
        f"Profile: "
        f"{inspection_config.profile}"
    )

    print(
        f"Input image: "
        f"{image_path}"
    )

    print(
        f"Resolution: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print()
    print(
        "Legacy glass preprocessing metrics"
    )
    print(
        "----------------------------------"
    )

    print(
        "Average-blur Canny edges: "
        f"{average_edge_pixels} "
        f"("
        f"{pixel_percentage(average_edge_pixels, result.average_edges):.4f}%"
        f")"
    )

    print(
        "Gaussian-blur Canny edges: "
        f"{gaussian_edge_pixels} "
        f"("
        f"{pixel_percentage(gaussian_edge_pixels, result.gaussian_edges):.4f}%"
        f")"
    )

    print(
        "Binary white pixels: "
        f"{binary_white_pixels} "
        f"("
        f"{pixel_percentage(binary_white_pixels, result.binary_thresholded):.4f}%"
        f")"
    )

    print(
        "Binary-inverted white pixels: "
        f"{binary_inverted_white_pixels} "
        f"("
        f"{pixel_percentage(binary_inverted_white_pixels, result.binary_inverted):.4f}%"
        f")"
    )

    print(
        "Otsu threshold value: "
        f"{result.otsu_threshold_value:.2f}"
    )

    print(
        "Otsu-inverted white pixels: "
        f"{otsu_white_pixels} "
        f"("
        f"{pixel_percentage(otsu_white_pixels, result.otsu_inverted):.4f}%"
        f")"
    )

    print(
        "Global-binary white pixels: "
        f"{global_binary_white_pixels} "
        f"("
        f"{pixel_percentage(global_binary_white_pixels, result.global_binary):.4f}%"
        f")"
    )

    print(
        "Adaptive-mean white pixels: "
        f"{adaptive_mean_white_pixels} "
        f"("
        f"{pixel_percentage(adaptive_mean_white_pixels, result.adaptive_mean):.4f}%"
        f")"
    )

    print(
        "Adaptive-Gaussian white pixels: "
        f"{adaptive_gaussian_white_pixels} "
        f"("
        f"{pixel_percentage(adaptive_gaussian_white_pixels, result.adaptive_gaussian):.4f}%"
        f")"
    )

    save_image(
        output_dir,
        "00-original.jpg",
        image,
    )

    save_image(
        output_dir,
        "01-grayscale.jpg",
        result.grayscale,
    )

    save_image(
        output_dir,
        "02-average-blurred.jpg",
        result.average_blurred,
    )

    save_image(
        output_dir,
        "03-average-edges.jpg",
        result.average_edges,
    )

    save_image(
        output_dir,
        "04-gaussian-blurred.jpg",
        result.gaussian_blurred,
    )

    save_image(
        output_dir,
        "05-gaussian-edges.jpg",
        result.gaussian_edges,
    )

    save_image(
        output_dir,
        "06-binary.jpg",
        result.binary_thresholded,
    )

    save_image(
        output_dir,
        "07-binary-inverted.jpg",
        result.binary_inverted,
    )

    save_image(
        output_dir,
        "08-otsu-inverted.jpg",
        result.otsu_inverted,
    )

    save_image(
        output_dir,
        "09-otsu-masked.jpg",
        result.otsu_masked,
    )

    save_image(
        output_dir,
        "10-median-blurred.jpg",
        result.median_blurred,
    )

    save_image(
        output_dir,
        "11-global-binary.jpg",
        result.global_binary,
    )

    save_image(
        output_dir,
        "12-adaptive-mean.jpg",
        result.adaptive_mean,
    )

    save_image(
        output_dir,
        "13-adaptive-gaussian.jpg",
        result.adaptive_gaussian,
    )

    print()
    print(
        "Saved 14 glass diagnostic images."
    )

    print()
    print(
        "No CLEAR/DEFECTIVE classification was performed."
    )

    print(
        "The 2022 glass code did not define "
        "a final glass decision rule."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )