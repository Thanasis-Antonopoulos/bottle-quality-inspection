"""Tests for inspection runtime and performance metrics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bottle_quality_inspection.runtime_metrics import (
    InspectionEvaluationReport,
    InspectionTimingMetrics,
    build_timing_metrics,
    write_inspection_metrics,
)
from bottle_quality_inspection.video_io import (
    VideoMetadata,
)


def make_metadata() -> VideoMetadata:
    """Create deterministic source-video metadata."""
    return VideoMetadata(
        path=Path(
            "example.mp4"
        ),
        width=2400,
        height=1080,
        fps=30.0,
        frame_count=90,
    )


def make_quality_metrics() -> SimpleNamespace:
    """Create minimal quality metrics for serialization tests."""
    return SimpleNamespace(
        to_dict=lambda: {
            "profile": "test-profile",
            "associated_bottles": 3,
        }
    )


def make_evaluation_report() -> InspectionEvaluationReport:
    """Create one deterministic evaluation report."""
    timing = InspectionTimingMetrics(
        source_frame_count=90,
        source_fps=30.0,
        source_duration_seconds=3.0,
        inspection_seconds=1.0,
        report_seconds=0.2,
        visual_seconds=0.8,
        total_seconds=2.0,
    )

    return InspectionEvaluationReport(
        quality=make_quality_metrics(),
        timing=timing,
    )


def test_timing_metrics_calculate_effective_fps() -> None:
    """Complete-pipeline throughput should use source frame count."""
    timing = InspectionTimingMetrics(
        source_frame_count=90,
        source_fps=30.0,
        source_duration_seconds=3.0,
        inspection_seconds=1.0,
        report_seconds=0.2,
        visual_seconds=0.8,
        total_seconds=2.0,
    )

    assert (
        timing.effective_processing_fps
        == 45.0
    )


def test_timing_metrics_calculate_realtime_factor() -> None:
    """Realtime factor should compare source duration with runtime."""
    timing = InspectionTimingMetrics(
        source_frame_count=90,
        source_fps=30.0,
        source_duration_seconds=3.0,
        inspection_seconds=1.0,
        report_seconds=0.2,
        visual_seconds=0.8,
        total_seconds=2.0,
    )

    assert (
        timing.realtime_factor
        == 1.5
    )


def test_build_timing_metrics_uses_video_metadata() -> None:
    """Timing construction should retain source-video properties."""
    timing = build_timing_metrics(
        make_metadata(),
        inspection_seconds=1.1,
        report_seconds=0.2,
        visual_seconds=0.7,
        total_seconds=2.0,
    )

    assert (
        timing.source_frame_count
        == 90
    )

    assert timing.source_fps == 30.0

    assert (
        timing.source_duration_seconds
        == 3.0
    )


def test_negative_timing_is_rejected() -> None:
    """Elapsed-time measurements cannot be negative."""
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        build_timing_metrics(
            make_metadata(),
            inspection_seconds=-1.0,
            report_seconds=0.2,
            visual_seconds=0.7,
            total_seconds=2.0,
        )


def test_evaluation_report_combines_quality_and_timing() -> None:
    """Evaluation output should contain both metric categories."""
    report = (
        make_evaluation_report()
    )

    data = report.to_dict()

    assert (
        data["schema_version"]
        == 1
    )

    assert (
        data["quality"][
            "associated_bottles"
        ]
        == 3
    )

    assert (
        data["timing"][
            "effective_processing_fps"
        ]
        == 45.0
    )


def test_write_inspection_metrics_creates_json(
    tmp_path: Path,
) -> None:
    """Metrics should be written as structured JSON."""
    result = write_inspection_metrics(
        make_evaluation_report(),
        tmp_path,
    )

    assert result.json_path.exists()

    with result.json_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        data = json.load(
            json_file
        )

    assert (
        data["quality"]["profile"]
        == "test-profile"
    )

    assert (
        data["timing"]["total_seconds"]
        == 2.0
    )


def test_existing_metrics_require_overwrite(
    tmp_path: Path,
) -> None:
    """Existing metric reports should not be replaced silently."""
    report = (
        make_evaluation_report()
    )

    write_inspection_metrics(
        report,
        tmp_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exist",
    ):
        write_inspection_metrics(
            report,
            tmp_path,
        )

    result = write_inspection_metrics(
        report,
        tmp_path,
        overwrite=True,
    )

    assert result.json_path.exists()


def test_invalid_metrics_filename_is_rejected(
    tmp_path: Path,
) -> None:
    """Metrics filename must remain a simple JSON filename."""
    with pytest.raises(
        ValueError,
        match="JSON filename",
    ):
        write_inspection_metrics(
            make_evaluation_report(),
            tmp_path,
            filename="nested/metrics.json",
        )