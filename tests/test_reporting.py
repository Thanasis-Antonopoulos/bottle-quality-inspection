"""Tests for inspection CSV and JSON reporting."""

import csv
import json
from pathlib import Path

import pytest

from bottle_quality_inspection.bottle_pipeline import (
    BottleInspectionRun,
    TrackedBottleInspection,
)
from bottle_quality_inspection.bottle_result import (
    BottleInspectionResult,
    BottleStatus,
    CheckStatus,
    CorkBottleDecision,
    ForeignObjectBottleDecision,
    GlassBottleDecision,
    LiquidLevelBottleDecision,
)
from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceStatus,
)
from bottle_quality_inspection.reporting import (
    inspection_run_to_dict,
    tracked_bottle_to_csv_row,
    write_inspection_reports,
)


def make_good_bottle() -> TrackedBottleInspection:
    """Create one deterministic GOOD bottle."""
    result = BottleInspectionResult(
        bottle_id="bottle-001",
        status=BottleStatus.GOOD,
        foreign_object=(
            ForeignObjectBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=11,
                clear_frames=11,
                defective_frames=0,
                maximum_black_pixels=8,
                median_black_pixels=0.0,
            )
        ),
        liquid_level=(
            LiquidLevelBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=8,
                valid_measurements=8,
                clear_frames=6,
                defective_frames=2,
                median_level=227.25,
                acceptable_minimum=225.0,
                acceptable_maximum=247.0,
            )
        ),
        glass=(
            GlassBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=8,
                clear_frames=8,
                defective_frames=0,
                maximum_triggered_signals=1,
            )
        ),
        cork=(
            CorkBottleDecision(
                status=CheckStatus.CLEAR,
                presence_status=(
                    CorkPresenceStatus.PRESENT
                ),
                selected_frame_indices=(
                    43,
                    42,
                    45,
                ),
                top_k_cork_median=0.7679,
                top_k_body_control_median=0.3223,
                top_k_margin=0.4456,
            )
        ),
    )

    return TrackedBottleInspection(
        bottle_id="bottle-001",
        foreign_start_frame=28,
        foreign_end_frame=38,
        downstream_start_frame=40,
        downstream_end_frame=47,
        travel_frame_offset=11,
        travel_time_seconds=0.367,
        result=result,
    )


def make_review_bottle() -> TrackedBottleInspection:
    """Create one deterministic REVIEW bottle."""
    result = BottleInspectionResult(
        bottle_id="bottle-002",
        status=BottleStatus.REVIEW,
        foreign_object=(
            ForeignObjectBottleDecision(
                status=CheckStatus.REVIEW,
                observed_frames=11,
                clear_frames=10,
                defective_frames=1,
                maximum_black_pixels=990,
                median_black_pixels=0.0,
            )
        ),
        liquid_level=(
            LiquidLevelBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=7,
                valid_measurements=7,
                clear_frames=7,
                defective_frames=0,
                median_level=244.0,
                acceptable_minimum=225.0,
                acceptable_maximum=247.0,
            )
        ),
        glass=(
            GlassBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=7,
                clear_frames=7,
                defective_frames=0,
                maximum_triggered_signals=1,
            )
        ),
        cork=(
            CorkBottleDecision(
                status=CheckStatus.CLEAR,
                presence_status=(
                    CorkPresenceStatus.PRESENT
                ),
                selected_frame_indices=(
                    58,
                    60,
                    57,
                ),
                top_k_cork_median=0.7965,
                top_k_body_control_median=0.3401,
                top_k_margin=0.4564,
            )
        ),
    )

    return TrackedBottleInspection(
        bottle_id="bottle-002",
        foreign_start_frame=43,
        foreign_end_frame=53,
        downstream_start_frame=56,
        downstream_end_frame=62,
        travel_frame_offset=11,
        travel_time_seconds=0.367,
        result=result,
    )


def make_run() -> BottleInspectionRun:
    """Create a deterministic inspection run."""
    return BottleInspectionRun(
        video_path=Path(
            "example-video.mp4"
        ),
        profile_name="test-profile",
        bottles=(
            make_good_bottle(),
            make_review_bottle(),
        ),
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )


def test_flattened_csv_row_contains_final_result() -> None:
    """CSV conversion should preserve the important bottle result."""
    row = tracked_bottle_to_csv_row(
        make_review_bottle()
    )

    assert (
        row["bottle_id"]
        == "bottle-002"
    )

    assert (
        row["final_status"]
        == "review"
    )

    assert (
        row["foreign_object_status"]
        == "review"
    )

    assert (
        row["foreign_max_black_pixels"]
        == 990
    )

    assert (
        row["review_checks"]
        == "foreign_object"
    )


def test_json_structure_contains_summary() -> None:
    """Nested report data should summarize run outcomes."""
    data = inspection_run_to_dict(
        make_run()
    )

    assert (
        data["schema_version"]
        == 1
    )

    assert (
        data["summary"]["associated_bottles"]
        == 2
    )

    assert (
        data["summary"]["good"]
        == 1
    )

    assert (
        data["summary"]["review"]
        == 1
    )

    assert (
        data["summary"]["defective"]
        == 0
    )


def test_write_reports_creates_csv_and_json(
    tmp_path: Path,
) -> None:
    """Reporting should create both required files."""
    result = write_inspection_reports(
        make_run(),
        tmp_path,
    )

    assert result.csv_path.exists()
    assert result.json_path.exists()

    assert result.bottle_count == 2


def test_written_csv_contains_one_row_per_bottle(
    tmp_path: Path,
) -> None:
    """CSV output should contain exactly one row per physical bottle."""
    result = write_inspection_reports(
        make_run(),
        tmp_path,
    )

    with result.csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(
                csv_file
            )
        )

    assert len(rows) == 2

    assert (
        rows[0]["bottle_id"]
        == "bottle-001"
    )

    assert (
        rows[0]["final_status"]
        == "good"
    )

    assert (
        rows[1]["bottle_id"]
        == "bottle-002"
    )

    assert (
        rows[1]["final_status"]
        == "review"
    )


def test_written_json_preserves_nested_checks(
    tmp_path: Path,
) -> None:
    """JSON output should retain richer nested inspection evidence."""
    result = write_inspection_reports(
        make_run(),
        tmp_path,
    )

    with result.json_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        data = json.load(
            json_file
        )

    first_bottle = (
        data["bottles"][0]
    )

    assert (
        first_bottle["final_status"]
        == "good"
    )

    assert (
        first_bottle[
            "checks"
        ][
            "liquid_level"
        ][
            "median_level"
        ]
        == 227.25
    )

    assert (
        first_bottle[
            "checks"
        ][
            "cork"
        ][
            "presence_status"
        ]
        == "present"
    )


def test_existing_reports_require_overwrite(
    tmp_path: Path,
) -> None:
    """Existing reports should not be replaced silently."""
    write_inspection_reports(
        make_run(),
        tmp_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        write_inspection_reports(
            make_run(),
            tmp_path,
        )

    result = write_inspection_reports(
        make_run(),
        tmp_path,
        overwrite=True,
    )

    assert result.csv_path.exists()
    assert result.json_path.exists()


def test_invalid_report_stem_is_rejected(
    tmp_path: Path,
) -> None:
    """A report stem must remain a filename rather than a path."""
    with pytest.raises(
        ValueError,
        match="filename stem",
    ):
        write_inspection_reports(
            make_run(),
            tmp_path,
            stem="nested/report",
        )