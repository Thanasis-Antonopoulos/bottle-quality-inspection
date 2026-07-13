"""Command-line entry point for Bottle Quality Inspection."""

import argparse
from pathlib import Path

from bottle_quality_inspection import __version__
from bottle_quality_inspection.exceptions import BottleInspectionError
from bottle_quality_inspection.video_io import read_video_metadata


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


def main() -> int:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "metadata":
        try:
            print_video_metadata(args.video)
        except (OSError, BottleInspectionError) as exc:
            parser.error(str(exc))

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())