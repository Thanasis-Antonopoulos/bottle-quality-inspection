"""Export annotated video and flagged bottle evidence images."""

import argparse
from pathlib import Path

from bottle_quality_inspection.bottle_pipeline import (
    inspect_tracked_bottles,
)
from bottle_quality_inspection.config import (
    load_inspection_config,
)
from bottle_quality_inspection.visual_reporting import (
    write_visual_inspection_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the visual inspection exporter parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run tracked bottle inspection and export "
            "annotated video plus review/rejected images."
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
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/inspection-visual"
        ),
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> int:
    """Run inspection and export visual evidence."""
    parser = build_parser()
    args = parser.parse_args()

    profile = load_inspection_config(
        args.config
    )

    print(
        f"Profile: {profile.profile}"
    )

    print(
        f"Video: "
        f"{args.input_video.expanduser().resolve()}"
    )

    print()

    print(
        "Running tracked bottle inspection..."
    )

    run = inspect_tracked_bottles(
        args.input_video,
        profile,
    )

    foreign_region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    downstream_region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    artifacts = (
        write_visual_inspection_artifacts(
            run,
            args.output_dir,
            foreign_region,
            downstream_region,
            codec=args.codec,
            overwrite=args.overwrite,
        )
    )

    print()

    print(
        "VISUAL INSPECTION ARTIFACTS"
    )

    print(
        "---------------------------"
    )

    print(
        "Annotated video: "
        f"{artifacts.video_result.output_path}"
    )

    print(
        "Frames written: "
        f"{artifacts.video_result.frames_processed}"
    )

    print(
        "Review images: "
        f"{len(artifacts.review_images)}"
    )

    for image in (
        artifacts.review_images
    ):
        print(
            f"  {image.bottle_id}: "
            f"{image.path}"
        )

    print(
        "Rejected images: "
        f"{len(artifacts.rejected_images)}"
    )

    for image in (
        artifacts.rejected_images
    ):
        print(
            f"  {image.bottle_id}: "
            f"{image.path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )