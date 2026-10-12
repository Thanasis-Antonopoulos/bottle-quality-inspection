"""CSV and JSON reporting for tracked bottle inspection runs."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bottle_quality_inspection.bottle_pipeline import (
    BottleInspectionRun,
    TrackedBottleInspection,
)
from bottle_quality_inspection.bottle_result import (
    BottleStatus,
)
from bottle_quality_inspection.tracking import (
    BottlePassage,
)

CSV_FIELDNAMES = (
    "bottle_id",
    "final_status",
    "foreign_object_status",
    "foreign_clear_frames",
    "foreign_defective_frames",
    "foreign_max_black_pixels",
    "foreign_median_black_pixels",
    "liquid_level_status",
    "liquid_valid_measurements",
    "liquid_clear_frames",
    "liquid_defective_frames",
    "liquid_median_level",
    "liquid_acceptable_minimum",
    "liquid_acceptable_maximum",
    "glass_status",
    "glass_clear_frames",
    "glass_defective_frames",
    "glass_maximum_triggered_signals",
    "cork_status",
    "cork_presence_status",
    "cork_selected_frames",
    "cork_top_k_median",
    "cork_body_control_median",
    "cork_margin",
    "foreign_start_frame",
    "foreign_end_frame",
    "downstream_start_frame",
    "downstream_end_frame",
    "travel_frame_offset",
    "travel_time_seconds",
    "defective_checks",
    "review_checks",
)


@dataclass(frozen=True, slots=True)
class InspectionReportPaths:
    """Paths written for one inspection reporting operation."""

    csv_path: Path
    json_path: Path
    bottle_count: int


def _passage_to_dict(
    passage: BottlePassage,
) -> dict[str, Any]:
    """Convert one unmatched bottle passage into serializable data."""
    return {
        "station": passage.station,
        "sequence_number": (
            passage.sequence_number
        ),
        "start_frame": (
            passage.start_frame
        ),
        "end_frame": (
            passage.end_frame
        ),
        "detected_frame_count": (
            passage.detected_frame_count
        ),
        "representative_frame": (
            passage.representative_frame
        ),
        "representative_timestamp_seconds": (
            passage.representative_timestamp_seconds
        ),
    }


def tracked_bottle_to_dict(
    tracked: TrackedBottleInspection,
) -> dict[str, Any]:
    """Convert one tracked bottle result into nested report data."""
    result = tracked.result

    return {
        "bottle_id": (
            tracked.bottle_id
        ),
        "final_status": (
            result.status.value
        ),
        "tracking": {
            "foreign_start_frame": (
                tracked.foreign_start_frame
            ),
            "foreign_end_frame": (
                tracked.foreign_end_frame
            ),
            "downstream_start_frame": (
                tracked.downstream_start_frame
            ),
            "downstream_end_frame": (
                tracked.downstream_end_frame
            ),
            "travel_frame_offset": (
                tracked.travel_frame_offset
            ),
            "travel_time_seconds": (
                tracked.travel_time_seconds
            ),
        },
        "checks": {
            "foreign_object": {
                "status": (
                    result.foreign_object
                    .status
                    .value
                ),
                "observed_frames": (
                    result.foreign_object
                    .observed_frames
                ),
                "clear_frames": (
                    result.foreign_object
                    .clear_frames
                ),
                "defective_frames": (
                    result.foreign_object
                    .defective_frames
                ),
                "maximum_black_pixels": (
                    result.foreign_object
                    .maximum_black_pixels
                ),
                "median_black_pixels": (
                    result.foreign_object
                    .median_black_pixels
                ),
            },
            "liquid_level": {
                "status": (
                    result.liquid_level
                    .status
                    .value
                ),
                "observed_frames": (
                    result.liquid_level
                    .observed_frames
                ),
                "valid_measurements": (
                    result.liquid_level
                    .valid_measurements
                ),
                "clear_frames": (
                    result.liquid_level
                    .clear_frames
                ),
                "defective_frames": (
                    result.liquid_level
                    .defective_frames
                ),
                "median_level": (
                    result.liquid_level
                    .median_level
                ),
                "acceptable_minimum": (
                    result.liquid_level
                    .acceptable_minimum
                ),
                "acceptable_maximum": (
                    result.liquid_level
                    .acceptable_maximum
                ),
            },
            "glass": {
                "status": (
                    result.glass
                    .status
                    .value
                ),
                "observed_frames": (
                    result.glass
                    .observed_frames
                ),
                "clear_frames": (
                    result.glass
                    .clear_frames
                ),
                "defective_frames": (
                    result.glass
                    .defective_frames
                ),
                "maximum_triggered_signals": (
                    result.glass
                    .maximum_triggered_signals
                ),
            },
            "cork": {
                "status": (
                    result.cork
                    .status
                    .value
                ),
                "presence_status": (
                    result.cork
                    .presence_status
                    .value
                ),
                "selected_frame_indices": list(
                    result.cork
                    .selected_frame_indices
                ),
                "top_k_cork_median": (
                    result.cork
                    .top_k_cork_median
                ),
                "top_k_body_control_median": (
                    result.cork
                    .top_k_body_control_median
                ),
                "top_k_margin": (
                    result.cork
                    .top_k_margin
                ),
            },
        },
        "defective_checks": list(
            result.defective_checks
        ),
        "review_checks": list(
            result.review_checks
        ),
    }


def tracked_bottle_to_csv_row(
    tracked: TrackedBottleInspection,
) -> dict[str, object]:
    """Flatten one tracked bottle into a single CSV row."""
    result = tracked.result

    median_level = (
        ""
        if result.liquid_level.median_level is None
        else result.liquid_level.median_level
    )

    return {
        "bottle_id": (
            tracked.bottle_id
        ),
        "final_status": (
            result.status.value
        ),
        "foreign_object_status": (
            result.foreign_object
            .status
            .value
        ),
        "foreign_clear_frames": (
            result.foreign_object
            .clear_frames
        ),
        "foreign_defective_frames": (
            result.foreign_object
            .defective_frames
        ),
        "foreign_max_black_pixels": (
            result.foreign_object
            .maximum_black_pixels
        ),
        "foreign_median_black_pixels": (
            result.foreign_object
            .median_black_pixels
        ),
        "liquid_level_status": (
            result.liquid_level
            .status
            .value
        ),
        "liquid_valid_measurements": (
            result.liquid_level
            .valid_measurements
        ),
        "liquid_clear_frames": (
            result.liquid_level
            .clear_frames
        ),
        "liquid_defective_frames": (
            result.liquid_level
            .defective_frames
        ),
        "liquid_median_level": (
            median_level
        ),
        "liquid_acceptable_minimum": (
            result.liquid_level
            .acceptable_minimum
        ),
        "liquid_acceptable_maximum": (
            result.liquid_level
            .acceptable_maximum
        ),
        "glass_status": (
            result.glass
            .status
            .value
        ),
        "glass_clear_frames": (
            result.glass
            .clear_frames
        ),
        "glass_defective_frames": (
            result.glass
            .defective_frames
        ),
        "glass_maximum_triggered_signals": (
            result.glass
            .maximum_triggered_signals
        ),
        "cork_status": (
            result.cork
            .status
            .value
        ),
        "cork_presence_status": (
            result.cork
            .presence_status
            .value
        ),
        "cork_selected_frames": ";".join(
            str(frame)
            for frame
            in result.cork.selected_frame_indices
        ),
        "cork_top_k_median": (
            result.cork
            .top_k_cork_median
        ),
        "cork_body_control_median": (
            result.cork
            .top_k_body_control_median
        ),
        "cork_margin": (
            result.cork
            .top_k_margin
        ),
        "foreign_start_frame": (
            tracked.foreign_start_frame
        ),
        "foreign_end_frame": (
            tracked.foreign_end_frame
        ),
        "downstream_start_frame": (
            tracked.downstream_start_frame
        ),
        "downstream_end_frame": (
            tracked.downstream_end_frame
        ),
        "travel_frame_offset": (
            tracked.travel_frame_offset
        ),
        "travel_time_seconds": (
            tracked.travel_time_seconds
        ),
        "defective_checks": ";".join(
            result.defective_checks
        ),
        "review_checks": ";".join(
            result.review_checks
        ),
    }


def inspection_run_to_dict(
    run: BottleInspectionRun,
) -> dict[str, Any]:
    """Convert one complete inspection run into JSON-ready data."""
    good_count = sum(
        bottle.result.status
        is BottleStatus.GOOD
        for bottle in run.bottles
    )

    defective_count = sum(
        bottle.result.status
        is BottleStatus.DEFECTIVE
        for bottle in run.bottles
    )

    review_count = sum(
        bottle.result.status
        is BottleStatus.REVIEW
        for bottle in run.bottles
    )

    return {
        "schema_version": 1,
        "profile": (
            run.profile_name
        ),
        "video": str(
            run.video_path
        ),
        "summary": {
            "associated_bottles": (
                run.bottle_count
            ),
            "good": (
                good_count
            ),
            "defective": (
                defective_count
            ),
            "review": (
                review_count
            ),
            "unmatched_foreign_passages": len(
                run.unmatched_foreign_passages
            ),
            "unmatched_downstream_passages": len(
                run.unmatched_downstream_passages
            ),
        },
        "bottles": [
            tracked_bottle_to_dict(
                bottle
            )
            for bottle in run.bottles
        ],
        "unmatched_foreign_passages": [
            _passage_to_dict(
                passage
            )
            for passage
            in run.unmatched_foreign_passages
        ],
        "unmatched_downstream_passages": [
            _passage_to_dict(
                passage
            )
            for passage
            in run.unmatched_downstream_passages
        ],
    }


def _validate_report_stem(
    stem: str,
) -> str:
    """Validate a report filename stem."""
    normalized = stem.strip()

    if not normalized:
        raise ValueError(
            "Report stem cannot be empty."
        )

    if Path(normalized).name != normalized:
        raise ValueError(
            "Report stem must be a filename stem, "
            "not a path."
        )

    return normalized


def _check_output_paths(
    paths: tuple[Path, ...],
    *,
    overwrite: bool,
) -> None:
    """Reject existing report files unless overwrite is enabled."""
    if overwrite:
        return

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if not existing:
        return

    existing_text = ", ".join(
        str(path)
        for path in existing
    )

    raise FileExistsError(
        "Report output already exists: "
        f"{existing_text}. "
        "Use overwrite=True to replace it."
    )


def write_inspection_reports(
    run: BottleInspectionRun,
    output_dir: str | Path,
    *,
    stem: str = "inspection-results",
    overwrite: bool = False,
) -> InspectionReportPaths:
    """Write CSV and JSON reports for one complete inspection run."""
    normalized_stem = (
        _validate_report_stem(
            stem
        )
    )

    destination = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    csv_path = (
        destination
        / f"{normalized_stem}.csv"
    )

    json_path = (
        destination
        / f"{normalized_stem}.json"
    )

    _check_output_paths(
        (
            csv_path,
            json_path,
        ),
        overwrite=overwrite,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDNAMES,
        )

        writer.writeheader()

        for tracked in run.bottles:
            writer.writerow(
                tracked_bottle_to_csv_row(
                    tracked
                )
            )

    report_data = (
        inspection_run_to_dict(
            run
        )
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            report_data,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

        json_file.write(
            "\n"
        )

    return InspectionReportPaths(
        csv_path=csv_path,
        json_path=json_path,
        bottle_count=(
            run.bottle_count
        ),
    )