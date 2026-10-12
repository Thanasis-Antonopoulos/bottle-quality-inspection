"""Run tracked bottle inspection and export CSV and JSON reports."""

import argparse
from pathlib import Path

from bottle_quality_inspection.bottle_pipeline import (
    inspect_tracked_bottles,
)
from bottle_quality_inspection.bottle_result import (
    BottleStatus,
)
from bottle_quality_inspection.config import (
    load_inspection_config,
)
from bottle_quality_inspection.reporting import (
    write_inspection_reports,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the reporting command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run tracked bottle inspection and export "
            "per-bottle CSV and JSON reports."
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
            "outputs/inspection-report"
        ),
        help=(
            "Directory for CSV and JSON reports. "
            "Default: outputs/inspection-report."
        ),
    )

    parser.add_argument(
        "--stem",
        default="inspection-results",
        help=(
            "Base filename for report files. "
            "Default: inspection-results."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report files.",
    )

    return parser


def main() -> int:
    """Run inspection and export report files."""
    parser = build_parser()
    args = parser.parse_args()

    profile = load_inspection_config(
        args.config
    )

    print(
        f"Profile: "
        f"{profile.profile}"
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

    report_paths = (
        write_inspection_reports(
            run,
            args.output_dir,
            stem=args.stem,
            overwrite=args.overwrite,
        )
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
        "------------------"
    )

    print(
        f"Associated bottles: "
        f"{run.bottle_count}"
    )

    print(
        f"GOOD: "
        f"{good_count}"
    )

    print(
        f"DEFECTIVE: "
        f"{defective_count}"
    )

    print(
        f"REVIEW: "
        f"{review_count}"
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
        "REPORT FILES"
    )

    print(
        "------------"
    )

    print(
        f"CSV: "
        f"{report_paths.csv_path}"
    )

    print(
        f"JSON: "
        f"{report_paths.json_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )