"""Command-line entry point for Bottle Quality Inspection."""

import argparse
from pathlib import Path

from bottle_quality_inspection import __version__
from bottle_quality_inspection.exceptions import BottleInspectionError
from bottle_quality_inspection.video_io import (
    copy_video,
    read_video_metadata,
    scan_video,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the application command-line parser."""
    parser = argparse.ArgumentParser(
        prog="bottle-quality-inspection",
        description="Automated bottle quality inspection tools.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

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
        help="Decode every frame and report processing performance.",
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
        help="Four-character OpenCV video codec. Default: mp4v.",
    )
    copy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )

    return parser


def print_video_metadata(video_path: Path) -> None:
    """Read and print metadata for a video."""
    metadata = read_video_metadata(video_path)

    duration = (
        f"{metadata.duration_seconds:.2f} seconds"
        if metadata.duration_seconds is not None
        else "Unavailable"
    )

    print(f"Video: {metadata.path}")
    print(f"Resolution: {metadata.width} x {metadata.height}")
    print(f"FPS: {metadata.fps:.2f}")
    print(f"Frames: {metadata.frame_count}")
    print(f"Estimated duration: {duration}")


def print_video_scan(video_path: Path) -> None:
    """Decode a complete video and print the scan summary."""
    result = scan_video(video_path)

    processing_fps = (
        f"{result.processing_fps:.2f}"
        if result.processing_fps is not None
        else "Unavailable"
    )

    print(f"Video: {result.path}")
    print(f"Frames decoded: {result.frames_decoded}")
    print(f"Elapsed time: {result.elapsed_seconds:.3f} seconds")
    print(f"Processing FPS: {processing_fps}")


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

    print(f"Input video: {result.input_path}")
    print(f"Output video: {result.output_path}")
    print(f"Codec: {result.codec}")
    print(f"Frames written: {result.frames_written}")
    print(f"Elapsed time: {result.elapsed_seconds:.3f} seconds")
    print(f"Processing FPS: {processing_fps}")


def main() -> int:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "metadata":
            print_video_metadata(args.video)
            return 0

        if args.command == "scan":
            print_video_scan(args.video)
            return 0

        if args.command == "copy":
            print_video_copy(
                args.input_video,
                args.output_video,
                codec=args.codec,
                overwrite=args.overwrite,
            )
            return 0

    except (OSError, ValueError, BottleInspectionError) as exc:
        parser.error(str(exc))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())