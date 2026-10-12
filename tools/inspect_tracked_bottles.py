"""Run complete multi-frame quality inspection on tracked bottles."""

import argparse
from pathlib import Path

from bottle_quality_inspection.bottle_pipeline import (
    inspect_tracked_bottles,
)
from bottle_quality_inspection.config import (
    load_inspection_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the end-to-end inspection parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect physical bottles across fixed stations "
            "and produce one final result per associated bottle."
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

    return parser


def format_optional_level(
    value: float | None,
) -> str:
    """Format an optional liquid-level measurement."""
    if value is None:
        return "unavailable"

    return f"{value:.2f}"


def main() -> int:
    """Run complete tracked-bottle inspection."""
    parser = build_parser()
    args = parser.parse_args()

    profile = load_inspection_config(
        args.config
    )

    run = inspect_tracked_bottles(
        args.input_video,
        profile,
    )

    print(
        f"Profile: {run.profile_name}"
    )

    print(
        f"Video: {run.video_path}"
    )

    print(
        f"Associated bottles: "
        f"{run.bottle_count}"
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
        "FINAL BOTTLE INSPECTION RESULTS"
    )

    print(
        "==============================="
    )

    print()

    for tracked in run.bottles:
        result = tracked.result

        print(
            tracked.bottle_id
        )

        print(
            "-" * len(
                tracked.bottle_id
            )
        )

        print(
            "Tracking: "
            f"foreign "
            f"{tracked.foreign_start_frame}-"
            f"{tracked.foreign_end_frame} "
            "→ downstream "
            f"{tracked.downstream_start_frame}-"
            f"{tracked.downstream_end_frame}"
        )

        print(
            "Travel offset: "
            f"{tracked.travel_frame_offset} frames "
            f"("
            f"{tracked.travel_time_seconds:.3f}s"
            f")"
        )

        print()

        print(
            "Foreign object: "
            f"{result.foreign_object.status.value.upper()}"
        )

        print(
            "  clear frames: "
            f"{result.foreign_object.clear_frames}"
        )

        print(
            "  defective frames: "
            f"{result.foreign_object.defective_frames}"
        )

        print(
            "  max black pixels: "
            f"{result.foreign_object.maximum_black_pixels}"
        )

        print(
            "  median black pixels: "
            f"{result.foreign_object.median_black_pixels:.1f}"
        )

        print()

        print(
            "Liquid level: "
            f"{result.liquid_level.status.value.upper()}"
        )

        print(
            "  valid measurements: "
            f"{result.liquid_level.valid_measurements}"
        )

        print(
            "  clear frames: "
            f"{result.liquid_level.clear_frames}"
        )

        print(
            "  defective frames: "
            f"{result.liquid_level.defective_frames}"
        )

        print(
            "  median level: "
            f"{format_optional_level(
                result.liquid_level.median_level
            )}"
        )

        print()

        print(
            "Glass: "
            f"{result.glass.status.value.upper()}"
        )

        print(
            "  clear frames: "
            f"{result.glass.clear_frames}"
        )

        print(
            "  defective frames: "
            f"{result.glass.defective_frames}"
        )

        print(
            "  maximum triggered signals: "
            f"{result.glass.maximum_triggered_signals}"
        )

        print()

        print(
            "Cork: "
            f"{result.cork.status.value.upper()}"
        )

        print(
            "  presence: "
            f"{result.cork.presence_status.value}"
        )

        print(
            "  selected frames: "
            + ", ".join(
                str(frame)
                for frame
                in result.cork.selected_frame_indices
            )
        )

        print(
            "  top-3 cork median: "
            f"{result.cork.top_k_cork_median:.4f}"
        )

        print(
            "  top-3 body median: "
            f"{result.cork.top_k_body_control_median:.4f}"
        )

        print(
            "  top-3 margin: "
            f"{result.cork.top_k_margin:.4f}"
        )

        print()

        print(
            "FINAL: "
            f"{result.status.value.upper()}"
        )

        if result.defective_checks:
            print(
                "Confirmed defect checks: "
                + ", ".join(
                    result.defective_checks
                )
            )

        if result.review_checks:
            print(
                "Review checks: "
                + ", ".join(
                    result.review_checks
                )
            )

        print()

    print(
        "SUMMARY"
    )

    print(
        "-------"
    )

    good = sum(
        tracked.result.status.value
        == "good"
        for tracked in run.bottles
    )

    defective = sum(
        tracked.result.status.value
        == "defective"
        for tracked in run.bottles
    )

    review = sum(
        tracked.result.status.value
        == "review"
        for tracked in run.bottles
    )

    print(
        f"GOOD: {good}"
    )

    print(
        f"DEFECTIVE: {defective}"
    )

    print(
        f"REVIEW: {review}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )