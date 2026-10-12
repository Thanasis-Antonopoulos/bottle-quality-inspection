"""Command-line entry point for Bottle Quality Inspection."""

import argparse
from pathlib import Path

from bottle_quality_inspection import __version__
from bottle_quality_inspection.bottle_pipeline import (
    inspect_tracked_bottles,
)
from bottle_quality_inspection.bottle_result import (
    BottleStatus,
)
from bottle_quality_inspection.calibration import save_roi_preview
from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.exceptions import (
    BottleInspectionError,
    ConfigurationError,
)
from bottle_quality_inspection.inspection_preview import (
    save_cork_preview,
    save_foreign_object_preview,
    save_liquid_level_preview,
)
from bottle_quality_inspection.inspections.cork import CorkConfig
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
)
from bottle_quality_inspection.pipeline import process_video
from bottle_quality_inspection.processors import RoiOverlayProcessor
from bottle_quality_inspection.reporting import (
    write_inspection_reports,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import (
    copy_video,
    read_video_metadata,
    scan_video,
)
from bottle_quality_inspection.visual_reporting import (
    write_visual_inspection_artifacts,
)


def add_roi_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add common region-of-interest arguments to a parser."""
    parser.add_argument(
        "--name",
        default="inspection-area",
        help="Name displayed above the region.",
    )

    parser.add_argument(
        "--x",
        type=int,
        required=True,
        help="Horizontal start coordinate.",
    )

    parser.add_argument(
        "--y",
        type=int,
        required=True,
        help="Vertical start coordinate.",
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Region width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Region height in pixels.",
    )

    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Rectangle and text thickness. Default: 2.",
    )

    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Draw the rectangle without its name.",
    )


def add_config_preview_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add common configuration-driven preview arguments."""
    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    parser.add_argument(
        "output_image",
        type=Path,
        help="Path for the generated inspection preview.",
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
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based video-frame index. Default: 0.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output image.",
    )


def add_inspect_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add complete tracked-bottle inspection arguments."""
    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source inspection video.",
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
            "outputs/inspection"
        ),
        help=(
            "Directory for all generated inspection artifacts. "
            "Default: outputs/inspection."
        ),
    )

    parser.add_argument(
        "--stem",
        default="inspection-results",
        help=(
            "Base filename for CSV and JSON reports. "
            "Default: inspection-results."
        ),
    )

    parser.add_argument(
        "--video-filename",
        default="annotated-inspection.mp4",
        help=(
            "Filename for the annotated output video. "
            "Default: annotated-inspection.mp4."
        ),
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "Four-character OpenCV output codec. "
            "Default: mp4v."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing inspection artifacts.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the application command-line parser."""
    parser = argparse.ArgumentParser(
        prog="bottle-quality-inspection",
        description=(
            "Automated bottle quality inspection tools."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Read and validate video metadata.",
    )

    metadata_parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video.",
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help=(
            "Decode every frame and report "
            "processing performance."
        ),
    )

    scan_parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video.",
    )

    copy_parser = subparsers.add_parser(
        "copy",
        help="Decode and write every frame to a new video.",
    )

    copy_parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    copy_parser.add_argument(
        "output_video",
        type=Path,
        help="Path for the output video.",
    )

    copy_parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "Four-character OpenCV video codec. "
            "Default: mp4v."
        ),
    )

    copy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )

    roi_preview_parser = subparsers.add_parser(
        "roi-preview",
        help=(
            "Draw a region of interest on a "
            "selected video frame."
        ),
    )

    roi_preview_parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    roi_preview_parser.add_argument(
        "output_image",
        type=Path,
        help="Path for the generated preview image.",
    )

    roi_preview_parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based video-frame index. Default: 0.",
    )

    add_roi_arguments(
        roi_preview_parser
    )

    roi_preview_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output image.",
    )

    roi_video_parser = subparsers.add_parser(
        "roi-video",
        help=(
            "Draw a fixed region of interest "
            "on every video frame."
        ),
    )

    roi_video_parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    roi_video_parser.add_argument(
        "output_video",
        type=Path,
        help="Path for the processed output video.",
    )

    add_roi_arguments(
        roi_video_parser
    )

    roi_video_parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "Four-character OpenCV video codec. "
            "Default: mp4v."
        ),
    )

    roi_video_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video.",
    )

    foreign_preview_parser = subparsers.add_parser(
        "foreign-object-preview",
        help=(
            "Inspect one bottle ROI for "
            "possible foreign objects."
        ),
    )

    add_config_preview_arguments(
        foreign_preview_parser
    )

    liquid_preview_parser = subparsers.add_parser(
        "liquid-level-preview",
        help=(
            "Inspect the liquid level in one "
            "configured bottle ROI."
        ),
    )

    add_config_preview_arguments(
        liquid_preview_parser
    )

    cork_preview_parser = subparsers.add_parser(
        "cork-preview",
        help=(
            "Inspect one complete video frame "
            "using the cork template."
        ),
    )

    add_config_preview_arguments(
        cork_preview_parser
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help=(
            "Run complete tracked-bottle inspection "
            "and export all artifacts."
        ),
    )

    add_inspect_arguments(
        inspect_parser
    )

    return parser


def create_region(
    args: argparse.Namespace,
) -> RegionOfInterest:
    """Create a validated region from CLI arguments."""
    return RegionOfInterest(
        name=args.name,
        x=args.x,
        y=args.y,
        width=args.width,
        height=args.height,
    )


def print_video_metadata(
    video_path: Path,
) -> None:
    """Read and print metadata for a video."""
    metadata = read_video_metadata(
        video_path
    )

    duration = (
        f"{metadata.duration_seconds:.2f} seconds"
        if metadata.duration_seconds is not None
        else "Unavailable"
    )

    print(
        f"Video: {metadata.path}"
    )

    print(
        "Resolution: "
        f"{metadata.width} x "
        f"{metadata.height}"
    )

    print(
        f"FPS: {metadata.fps:.2f}"
    )

    print(
        f"Frames: {metadata.frame_count}"
    )

    print(
        f"Estimated duration: {duration}"
    )


def print_video_scan(
    video_path: Path,
) -> None:
    """Decode a complete video and print the scan summary."""
    result = scan_video(
        video_path
    )

    processing_fps = (
        f"{result.processing_fps:.2f}"
        if result.processing_fps is not None
        else "Unavailable"
    )

    print(
        f"Video: {result.path}"
    )

    print(
        "Frames decoded: "
        f"{result.frames_decoded}"
    )

    print(
        "Elapsed time: "
        f"{result.elapsed_seconds:.3f} seconds"
    )

    print(
        f"Processing FPS: {processing_fps}"
    )


def print_video_copy(
    input_path: Path,
    output_path: Path,
    *,
    codec: str,
    overwrite: bool,
) -> None:
    """Copy a video through the validated OpenCV writer."""
    result = copy_video(
        input_path,
        output_path,
        codec=codec,
        overwrite=overwrite,
    )

    processing_fps = (
        f"{result.processing_fps:.2f}"
        if result.processing_fps is not None
        else "Unavailable"
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output video: {result.output_path}"
    )

    print(
        f"Codec: {result.codec}"
    )

    print(
        "Frames written: "
        f"{result.frames_written}"
    )

    print(
        "Elapsed time: "
        f"{result.elapsed_seconds:.3f} seconds"
    )

    print(
        f"Processing FPS: {processing_fps}"
    )


def print_roi_preview(
    args: argparse.Namespace,
) -> None:
    """Create and report an ROI calibration preview."""
    region = create_region(
        args
    )

    result = save_roi_preview(
        args.input_video,
        args.output_image,
        region,
        frame_index=args.frame_index,
        thickness=args.thickness,
        show_label=not args.no_label,
        overwrite=args.overwrite,
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output image: {result.output_path}"
    )

    print(
        f"Frame index: {result.frame_index}"
    )

    print(
        "Timestamp: "
        f"{result.timestamp_seconds:.3f} seconds"
    )

    print(
        "Region: "
        f"{result.region.name} "
        f"(x={result.region.x}, "
        f"y={result.region.y}, "
        f"width={result.region.width}, "
        f"height={result.region.height})"
    )


def print_roi_video(
    args: argparse.Namespace,
) -> None:
    """Draw a configured ROI on every frame of a video."""
    region = create_region(
        args
    )

    processor = RoiOverlayProcessor(
        region=region,
        thickness=args.thickness,
        show_label=not args.no_label,
    )

    result = process_video(
        args.input_video,
        args.output_video,
        processor,
        codec=args.codec,
        overwrite=args.overwrite,
    )

    processing_fps = (
        f"{result.processing_fps:.2f}"
        if result.processing_fps is not None
        else "Unavailable"
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output video: {result.output_path}"
    )

    print(
        "Frames processed: "
        f"{result.frames_processed}"
    )

    print(
        f"Codec: {result.codec}"
    )

    print(
        "Elapsed time: "
        f"{result.elapsed_seconds:.3f} seconds"
    )

    print(
        f"Processing FPS: {processing_fps}"
    )

    print(
        "Region: "
        f"{region.name} "
        f"(x={region.x}, "
        f"y={region.y}, "
        f"width={region.width}, "
        f"height={region.height})"
    )


def print_foreign_object_preview(
    args: argparse.Namespace,
) -> None:
    """Inspect and report one foreign-object preview."""
    inspection_config = load_inspection_config(
        args.config
    )

    foreign_config = (
        inspection_config.foreign_object
    )

    if not foreign_config.enabled:
        raise ConfigurationError(
            "Foreign-object inspection is disabled "
            "in the selected profile."
        )

    region = foreign_config.roi.to_roi()

    algorithm_config = (
        foreign_config.to_algorithm_config()
    )

    result = save_foreign_object_preview(
        args.input_video,
        args.output_image,
        region,
        config=algorithm_config,
        frame_index=args.frame_index,
        overwrite=args.overwrite,
    )

    inspection = result.inspection

    decision_region = (
        foreign_config.decision_region
    )

    print(
        f"Profile: {inspection_config.profile}"
    )

    print(
        "Configuration: "
        f"{args.config.expanduser().resolve()}"
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output image: {result.output_path}"
    )

    print(
        f"Frame index: {result.frame_index}"
    )

    print(
        "Timestamp: "
        f"{result.timestamp_seconds:.3f} seconds"
    )

    print(
        "Inspection ROI: "
        f"x={foreign_config.roi.x}, "
        f"y={foreign_config.roi.y}, "
        f"width={foreign_config.roi.width}, "
        f"height={foreign_config.roi.height}"
    )

    print(
        "Decision region: "
        f"x={decision_region.x}, "
        f"y={decision_region.y}, "
        f"width={decision_region.width}, "
        f"height={decision_region.height}"
    )

    print(
        f"Status: {inspection.status.value}"
    )

    print(
        "Bottle detected: "
        f"{inspection.bottle_detected}"
    )

    print(
        "Raw anomaly pixels: "
        f"{inspection.raw_anomaly_pixel_count}"
    )

    print(
        "Decision black pixels: "
        f"{inspection.decision_black_pixel_count}"
    )

    print(
        "Largest contour area: "
        f"{inspection.largest_contour_area:.2f}"
    )

    print(
        "Selected components: "
        f"{inspection.selected_component_count}"
    )

    print(
        "Defect threshold: "
        f"{foreign_config.defect_pixel_threshold}"
    )


def print_liquid_level_preview(
    args: argparse.Namespace,
) -> None:
    """Inspect and report one liquid-level preview."""
    inspection_config = load_inspection_config(
        args.config
    )

    liquid_config = (
        inspection_config.liquid_level
    )

    if not liquid_config.enabled:
        raise ConfigurationError(
            "Liquid-level inspection is disabled "
            "in the selected profile."
        )

    region = liquid_config.roi.to_roi()

    algorithm_config = LiquidLevelConfig(
        neck_region=(
            liquid_config
            .neck_region
            .to_roi()
        ),
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

    result = save_liquid_level_preview(
        args.input_video,
        args.output_image,
        region,
        config=algorithm_config,
        frame_index=args.frame_index,
        overwrite=args.overwrite,
    )

    inspection = result.inspection

    neck_region = (
        liquid_config.neck_region
    )

    print(
        f"Profile: {inspection_config.profile}"
    )

    print(
        "Configuration: "
        f"{args.config.expanduser().resolve()}"
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output image: {result.output_path}"
    )

    print(
        f"Frame index: {result.frame_index}"
    )

    print(
        "Timestamp: "
        f"{result.timestamp_seconds:.3f} seconds"
    )

    print(
        "Inspection ROI: "
        f"x={liquid_config.roi.x}, "
        f"y={liquid_config.roi.y}, "
        f"width={liquid_config.roi.width}, "
        f"height={liquid_config.roi.height}"
    )

    print(
        "Neck region: "
        f"x={neck_region.x}, "
        f"y={neck_region.y}, "
        f"width={neck_region.width}, "
        f"height={neck_region.height}"
    )

    print(
        f"Status: {inspection.status.value}"
    )

    print(
        "Level detected: "
        f"{inspection.level_detected}"
    )

    print(
        "Detected lines: "
        f"{inspection.detected_line_count}"
    )

    if (
        inspection.relative_level
        is None
    ):
        print(
            "Relative liquid level: unavailable"
        )
    else:
        print(
            "Relative liquid level: "
            f"{inspection.relative_level:.2f}"
        )

    if (
        inspection.absolute_level
        is None
    ):
        print(
            "Liquid level: unavailable"
        )
    else:
        print(
            "Liquid level: "
            f"{inspection.absolute_level:.2f}"
        )

    print(
        "Acceptable range: "
        f"{liquid_config.acceptable_minimum:.2f} "
        "< level < "
        f"{liquid_config.acceptable_maximum:.2f}"
    )


def print_cork_preview(
    args: argparse.Namespace,
) -> None:
    """Inspect and report one cork-template preview."""
    inspection_config = load_inspection_config(
        args.config
    )

    cork_config = (
        inspection_config.cork
    )

    if not cork_config.enabled:
        raise ConfigurationError(
            "Cork inspection is disabled "
            "in the selected profile."
        )

    if (
        cork_config.minimum_match_score
        is None
    ):
        raise ConfigurationError(
            "Cork inspection requires "
            "minimum_match_score."
        )

    algorithm_config = CorkConfig(
        method=(
            cork_config.method
        ),
        minimum_match_score=(
            cork_config.minimum_match_score
        ),
    )

    result = save_cork_preview(
        args.input_video,
        args.output_image,
        cork_config.template,
        config=algorithm_config,
        frame_index=args.frame_index,
        overwrite=args.overwrite,
    )

    inspection = result.inspection

    print(
        f"Profile: {inspection_config.profile}"
    )

    print(
        "Configuration: "
        f"{args.config.expanduser().resolve()}"
    )

    print(
        f"Input video: {result.input_path}"
    )

    print(
        f"Output image: {result.output_path}"
    )

    print(
        f"Cork template: {result.template_path}"
    )

    print(
        f"Frame index: {result.frame_index}"
    )

    print(
        "Timestamp: "
        f"{result.timestamp_seconds:.3f} seconds"
    )

    print(
        f"Method: {cork_config.method}"
    )

    print(
        f"Status: {inspection.status.value}"
    )

    print(
        "Best match score: "
        f"{inspection.maximum_score:.4f}"
    )

    print(
        "Minimum required score: "
        f"{cork_config.minimum_match_score:.4f}"
    )

    print(
        "Matches at or above threshold: "
        f"{inspection.matches_at_or_above_threshold}"
    )

    print(
        "Best match location: "
        f"x={inspection.maximum_location[0]}, "
        f"y={inspection.maximum_location[1]}"
    )

    print(
        "Template size: "
        f"{inspection.template_width} x "
        f"{inspection.template_height}"
    )


def print_complete_inspection(
    args: argparse.Namespace,
) -> None:
    """Run complete inspection and write all production artifacts."""
    inspection_config = load_inspection_config(
        args.config
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    print(
        f"Profile: "
        f"{inspection_config.profile}"
    )

    print(
        "Configuration: "
        f"{args.config.expanduser().resolve()}"
    )

    print(
        "Input video: "
        f"{args.input_video.expanduser().resolve()}"
    )

    print(
        "Output directory: "
        f"{output_dir}"
    )

    print()

    print(
        "Running tracked bottle inspection..."
    )

    run = inspect_tracked_bottles(
        args.input_video,
        inspection_config,
    )

    print(
        "Writing CSV and JSON reports..."
    )

    reports = write_inspection_reports(
        run,
        output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
    )

    print(
        "Writing annotated video and "
        "flagged bottle images..."
    )

    foreign_region = (
        inspection_config.foreign_object
        .roi
        .to_roi()
    )

    downstream_region = (
        inspection_config.liquid_level
        .roi
        .to_roi()
    )

    visuals = (
        write_visual_inspection_artifacts(
            run,
            output_dir,
            foreign_region,
            downstream_region,
            video_filename=(
                args.video_filename
            ),
            codec=args.codec,
            overwrite=args.overwrite,
        )
    )

    review_dir = (
        output_dir
        / "review"
    )

    rejected_dir = (
        output_dir
        / "rejected"
    )

    review_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rejected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    good_count = sum(
        tracked.result.status
        is BottleStatus.GOOD
        for tracked in run.bottles
    )

    defective_count = sum(
        tracked.result.status
        is BottleStatus.DEFECTIVE
        for tracked in run.bottles
    )

    review_count = sum(
        tracked.result.status
        is BottleStatus.REVIEW
        for tracked in run.bottles
    )

    print()

    print(
        "INSPECTION SUMMARY"
    )

    print(
        "=================="
    )

    print(
        "Associated bottles: "
        f"{run.bottle_count}"
    )

    print(
        f"GOOD: {good_count}"
    )

    print(
        f"REVIEW: {review_count}"
    )

    print(
        f"DEFECTIVE: {defective_count}"
    )

    print(
        "Unmatched foreign passages: "
        f"{len(run.unmatched_foreign_passages)}"
    )

    print(
        "Unmatched downstream passages: "
        f"{len(run.unmatched_downstream_passages)}"
    )

    print()

    print(
        "PER-BOTTLE RESULTS"
    )

    print(
        "------------------"
    )

    for tracked in run.bottles:
        result = tracked.result

        print(
            f"{tracked.bottle_id}: "
            f"{result.status.value.upper()}"
        )

        print(
            "  foreign object: "
            f"{result.foreign_object.status.value}"
        )

        print(
            "  liquid level: "
            f"{result.liquid_level.status.value}"
        )

        print(
            "  glass: "
            f"{result.glass.status.value}"
        )

        print(
            "  cork: "
            f"{result.cork.status.value}"
        )

        if result.defective_checks:
            print(
                "  defective checks: "
                + ", ".join(
                    result.defective_checks
                )
            )

        if result.review_checks:
            print(
                "  review checks: "
                + ", ".join(
                    result.review_checks
                )
            )

    print()

    print(
        "OUTPUT ARTIFACTS"
    )

    print(
        "----------------"
    )

    print(
        f"CSV: "
        f"{reports.csv_path}"
    )

    print(
        f"JSON: "
        f"{reports.json_path}"
    )

    print(
        "Annotated video: "
        f"{visuals.video_result.output_path}"
    )

    print(
        f"Review directory: "
        f"{review_dir}"
    )

    print(
        f"Rejected directory: "
        f"{rejected_dir}"
    )

    print(
        "Review images: "
        f"{len(visuals.review_images)}"
    )

    print(
        "Rejected images: "
        f"{len(visuals.rejected_images)}"
    )


def main() -> int:
    """Run the command-line application."""
    parser = build_parser()

    args = parser.parse_args()

    try:
        if (
            args.command
            == "metadata"
        ):
            print_video_metadata(
                args.video
            )

            return 0

        if (
            args.command
            == "scan"
        ):
            print_video_scan(
                args.video
            )

            return 0

        if (
            args.command
            == "copy"
        ):
            print_video_copy(
                args.input_video,
                args.output_video,
                codec=args.codec,
                overwrite=args.overwrite,
            )

            return 0

        if (
            args.command
            == "roi-preview"
        ):
            print_roi_preview(
                args
            )

            return 0

        if (
            args.command
            == "roi-video"
        ):
            print_roi_video(
                args
            )

            return 0

        if (
            args.command
            == "foreign-object-preview"
        ):
            print_foreign_object_preview(
                args
            )

            return 0

        if (
            args.command
            == "liquid-level-preview"
        ):
            print_liquid_level_preview(
                args
            )

            return 0

        if (
            args.command
            == "cork-preview"
        ):
            print_cork_preview(
                args
            )

            return 0

        if (
            args.command
            == "inspect"
        ):
            print_complete_inspection(
                args
            )

            return 0

    except (
        OSError,
        ValueError,
        BottleInspectionError,
    ) as exc:
        parser.error(
            str(exc)
        )

    parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )