"""Diagnose liquid-level detection across an entire video."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspection_preview import (
    save_liquid_level_preview,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class FrameDiagnostic:
    """Liquid-level diagnostic information for one video frame."""

    frame_index: int
    timestamp_seconds: float
    edge_pixel_count: int
    detected_line_count: int
    status: str
    absolute_level: float | None


def build_parser() -> argparse.ArgumentParser:
    """Create the diagnostic command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan every video frame for liquid-level edges "
            "and Hough-line detections."
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
        default=Path("configs/legacy-2022.toml"),
        help=(
            "Inspection configuration profile. "
            "Default: configs/legacy-2022.toml."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/liquid-diagnostics"),
        help=(
            "Directory where candidate preview images are saved. "
            "Default: outputs/liquid-diagnostics."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "Number of frames with the most Canny edge pixels "
            "to save as previews. Default: 10."
        ),
    )

    return parser


def create_algorithm_config(
    inspection_config,
) -> LiquidLevelConfig:
    """Convert the loaded profile into algorithm configuration."""
    liquid_config = inspection_config.liquid_level

    return LiquidLevelConfig(
        neck_region=liquid_config.neck_region.to_roi(),
        canny_threshold1=(
            liquid_config.canny_threshold1
        ),
        canny_threshold2=(
            liquid_config.canny_threshold2
        ),
        hough_rho=(
            liquid_config.hough_rho
        ),
        hough_theta_degrees=(
            liquid_config.hough_theta_degrees
        ),
        hough_threshold=(
            liquid_config.hough_threshold
        ),
        minimum_line_length=(
            liquid_config.minimum_line_length
        ),
        maximum_line_gap=(
            liquid_config.maximum_line_gap
        ),
        acceptable_minimum=(
            liquid_config.acceptable_minimum
        ),
        acceptable_maximum=(
            liquid_config.acceptable_maximum
        ),
    )


def format_level(
    level: float | None,
) -> str:
    """Format an optional liquid-level measurement."""
    if level is None:
        return "N/A"

    return f"{level:.2f}"


def main() -> int:
    """Run the full-video liquid-level diagnostic."""
    parser = build_parser()
    args = parser.parse_args()

    if args.top <= 0:
        parser.error(
            "--top must be greater than zero."
        )

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

    inspection_config = load_inspection_config(
        args.config
    )

    liquid_config = (
        inspection_config.liquid_level
    )

    if not liquid_config.enabled:
        parser.error(
            "Liquid-level inspection is disabled "
            "in the selected configuration."
        )

    region = liquid_config.roi.to_roi()

    algorithm_config = create_algorithm_config(
        inspection_config
    )

    diagnostics: list[
        FrameDiagnostic
    ] = []

    print(
        f"Profile: {inspection_config.profile}"
    )

    print(
        f"Video: {video_path}"
    )

    print(
        "Liquid ROI: "
        f"x={liquid_config.roi.x}, "
        f"y={liquid_config.roi.y}, "
        f"width={liquid_config.roi.width}, "
        f"height={liquid_config.roi.height}"
    )

    print(
        "Neck region: "
        f"x={liquid_config.neck_region.x}, "
        f"y={liquid_config.neck_region.y}, "
        f"width={liquid_config.neck_region.width}, "
        f"height={liquid_config.neck_region.height}"
    )

    print()
    print("Scanning video...")
    print()

    for video_frame in iter_video_frames(
        video_path
    ):
        roi_image = region.extract(
            video_frame.image
        )

        result = inspect_liquid_level(
            roi_image,
            algorithm_config,
        )

        edge_pixel_count = int(
            cv2.countNonZero(
                result.edges
            )
        )

        diagnostics.append(
            FrameDiagnostic(
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                edge_pixel_count=(
                    edge_pixel_count
                ),
                detected_line_count=(
                    result.detected_line_count
                ),
                status=(
                    result.status.value
                ),
                absolute_level=(
                    result.absolute_level
                ),
            )
        )

    print(
        f"Frames scanned: {len(diagnostics)}"
    )

    frames_with_edges = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.edge_pixel_count > 0
    ]

    frames_with_lines = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.detected_line_count > 0
    ]

    print(
        "Frames with Canny edges: "
        f"{len(frames_with_edges)}"
    )

    print(
        "Frames with Hough lines: "
        f"{len(frames_with_lines)}"
    )

    print()

    if frames_with_lines:
        print(
            "Frames where a liquid-level line was detected:"
        )

        for diagnostic in frames_with_lines:
            print(
                f"  frame={diagnostic.frame_index:03d} "
                f"time={diagnostic.timestamp_seconds:.3f}s "
                f"edges={diagnostic.edge_pixel_count} "
                f"lines={diagnostic.detected_line_count} "
                f"level={format_level(diagnostic.absolute_level)} "
                f"status={diagnostic.status}"
            )
    else:
        print(
            "No Hough lines were detected in any frame."
        )

    print()

    ranked_frames = sorted(
        diagnostics,
        key=lambda diagnostic: (
            diagnostic.edge_pixel_count
        ),
        reverse=True,
    )

    top_frames = ranked_frames[
        : args.top
    ]

    print(
        f"Top {len(top_frames)} frames "
        "by Canny edge-pixel count:"
    )

    for diagnostic in top_frames:
        print(
            f"  frame={diagnostic.frame_index:03d} "
            f"time={diagnostic.timestamp_seconds:.3f}s "
            f"edges={diagnostic.edge_pixel_count} "
            f"lines={diagnostic.detected_line_count} "
            f"level={format_level(diagnostic.absolute_level)} "
            f"status={diagnostic.status}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Saving previews to: {output_dir}"
    )

    for diagnostic in top_frames:
        output_path = (
            output_dir
            / (
                "liquid-frame-"
                f"{diagnostic.frame_index:03d}.jpg"
            )
        )

        save_liquid_level_preview(
            video_path,
            output_path,
            region,
            config=algorithm_config,
            frame_index=(
                diagnostic.frame_index
            ),
            overwrite=True,
        )

    print()
    print("Diagnostic scan complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )