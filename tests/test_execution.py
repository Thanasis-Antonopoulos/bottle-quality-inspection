"""Tests for complete inspection execution orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from bottle_quality_inspection import (
    execution,
)
from bottle_quality_inspection.runtime_metrics import (
    InspectionMetricsPath,
)
from bottle_quality_inspection.video_io import (
    VideoMetadata,
)


class FakeRectangle:
    """Minimal configurable rectangle used by execution tests."""

    def __init__(
        self,
        value: str,
    ) -> None:
        self.value = value

    def to_roi(
        self,
    ) -> str:
        """Return a deterministic ROI marker."""
        return self.value


def make_profile() -> SimpleNamespace:
    """Create the minimum profile required by execution orchestration."""
    return SimpleNamespace(
        foreign_object=(
            SimpleNamespace(
                roi=FakeRectangle(
                    "foreign-region"
                )
            )
        ),
        liquid_level=(
            SimpleNamespace(
                roi=FakeRectangle(
                    "downstream-region"
                )
            )
        ),
    )


def make_metadata() -> VideoMetadata:
    """Create deterministic test video metadata."""
    return VideoMetadata(
        path=Path(
            "example.mp4"
        ),
        width=2400,
        height=1080,
        fps=30.0,
        frame_count=90,
    )


def test_execute_inspection_runs_all_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution should call every production stage once."""
    calls: list[str] = []

    run = SimpleNamespace(
        bottles=(),
        bottle_count=0,
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )

    reports = SimpleNamespace(
        csv_path=(
            tmp_path
            / "inspection-results.csv"
        ),
        json_path=(
            tmp_path
            / "inspection-results.json"
        ),
    )

    visuals = SimpleNamespace(
        video_result=SimpleNamespace(
            output_path=(
                tmp_path
                / "annotated-inspection.mp4"
            )
        ),
        review_images=(),
        rejected_images=(),
    )

    quality = SimpleNamespace(
        to_dict=lambda: {
            "associated_bottles": 0,
        }
    )

    monkeypatch.setattr(
        execution,
        "read_video_metadata",
        lambda _path: make_metadata(),
    )

    def fake_inspect(
        *_args,
        **_kwargs,
    ):
        calls.append(
            "inspection"
        )

        return run

    def fake_reports(
        *_args,
        **_kwargs,
    ):
        calls.append(
            "reports"
        )

        return reports

    def fake_visuals(
        *_args,
        **_kwargs,
    ):
        calls.append(
            "visuals"
        )

        return visuals

    def fake_quality(
        _run,
    ):
        calls.append(
            "quality"
        )

        return quality

    def fake_metrics_write(
        _report,
        output_dir,
        *,
        filename,
        overwrite,
    ):
        calls.append(
            "metrics"
        )

        assert (
            Path(output_dir)
            == tmp_path.resolve()
        )

        assert (
            filename
            == "inspection-metrics.json"
        )

        assert overwrite is True

        return InspectionMetricsPath(
            json_path=(
                tmp_path
                / filename
            )
        )

    monkeypatch.setattr(
        execution,
        "inspect_tracked_bottles",
        fake_inspect,
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_reports",
        fake_reports,
    )

    monkeypatch.setattr(
        execution,
        "write_visual_inspection_artifacts",
        fake_visuals,
    )

    monkeypatch.setattr(
        execution,
        "build_inspection_metrics",
        fake_quality,
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_metrics",
        fake_metrics_write,
    )

    times = iter(
        [
            10.0,
            12.0,
            20.0,
            20.25,
            30.0,
            31.5,
        ]
    )

    monkeypatch.setattr(
        execution,
        "perf_counter",
        lambda: next(
            times
        ),
    )

    result = execution.execute_inspection(
        "example.mp4",
        make_profile(),
        tmp_path,
        overwrite=True,
    )

    assert calls == [
        "inspection",
        "reports",
        "visuals",
        "quality",
        "metrics",
    ]

    assert result.run is run
    assert result.reports is reports
    assert result.visuals is visuals

    assert (
        result.quality_metrics
        is quality
    )


def test_execute_inspection_records_stage_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution should measure each major production stage."""
    run = SimpleNamespace(
        bottles=(),
        bottle_count=0,
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )

    monkeypatch.setattr(
        execution,
        "read_video_metadata",
        lambda _path: make_metadata(),
    )

    monkeypatch.setattr(
        execution,
        "inspect_tracked_bottles",
        lambda *_args, **_kwargs: run,
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_reports",
        lambda *_args, **_kwargs: (
            SimpleNamespace()
        ),
    )

    monkeypatch.setattr(
        execution,
        "write_visual_inspection_artifacts",
        lambda *_args, **_kwargs: (
            SimpleNamespace()
        ),
    )

    monkeypatch.setattr(
        execution,
        "build_inspection_metrics",
        lambda _run: (
            SimpleNamespace(
                to_dict=lambda: {}
            )
        ),
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_metrics",
        lambda *_args, **_kwargs: (
            InspectionMetricsPath(
                json_path=(
                    tmp_path
                    / "inspection-metrics.json"
                )
            )
        ),
    )

    times = iter(
        [
            10.0,
            12.0,
            20.0,
            20.25,
            30.0,
            31.5,
        ]
    )

    monkeypatch.setattr(
        execution,
        "perf_counter",
        lambda: next(
            times
        ),
    )

    result = execution.execute_inspection(
        "example.mp4",
        make_profile(),
        tmp_path,
    )

    timing = (
        result.timing_metrics
    )

    assert (
        timing.inspection_seconds
        == 2.0
    )

    assert (
        timing.report_seconds
        == 0.25
    )

    assert (
        timing.visual_seconds
        == 1.5
    )

    assert (
        timing.total_seconds
        == 3.75
    )

    assert (
        timing.effective_processing_fps
        == 24.0
    )

    assert (
        timing.realtime_factor
        == pytest.approx(
            0.8
        )
    )


def test_execute_inspection_passes_regions_to_visual_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured station regions should reach visual reporting."""
    run = SimpleNamespace(
        bottles=(),
        bottle_count=0,
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )

    received: dict[
        str,
        object,
    ] = {}

    monkeypatch.setattr(
        execution,
        "read_video_metadata",
        lambda _path: make_metadata(),
    )

    monkeypatch.setattr(
        execution,
        "inspect_tracked_bottles",
        lambda *_args, **_kwargs: run,
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_reports",
        lambda *_args, **_kwargs: (
            SimpleNamespace()
        ),
    )

    def fake_visuals(
        _run,
        _output_dir,
        foreign_region,
        downstream_region,
        **_kwargs,
    ):
        received[
            "foreign"
        ] = foreign_region

        received[
            "downstream"
        ] = downstream_region

        return SimpleNamespace()

    monkeypatch.setattr(
        execution,
        "write_visual_inspection_artifacts",
        fake_visuals,
    )

    monkeypatch.setattr(
        execution,
        "build_inspection_metrics",
        lambda _run: (
            SimpleNamespace(
                to_dict=lambda: {}
            )
        ),
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_metrics",
        lambda *_args, **_kwargs: (
            InspectionMetricsPath(
                json_path=(
                    tmp_path
                    / "inspection-metrics.json"
                )
            )
        ),
    )

    times = iter(
        [
            0.0,
            1.0,
            1.0,
            1.1,
            1.1,
            2.0,
        ]
    )

    monkeypatch.setattr(
        execution,
        "perf_counter",
        lambda: next(
            times
        ),
    )

    execution.execute_inspection(
        "example.mp4",
        make_profile(),
        tmp_path,
    )

    assert (
        received["foreign"]
        == "foreign-region"
    )

    assert (
        received["downstream"]
        == "downstream-region"
    )


def test_execute_inspection_creates_flagged_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review and rejected directories should always exist after execution."""
    run = SimpleNamespace(
        bottles=(),
        bottle_count=0,
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )

    monkeypatch.setattr(
        execution,
        "read_video_metadata",
        lambda _path: make_metadata(),
    )

    monkeypatch.setattr(
        execution,
        "inspect_tracked_bottles",
        lambda *_args, **_kwargs: run,
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_reports",
        lambda *_args, **_kwargs: (
            SimpleNamespace()
        ),
    )

    monkeypatch.setattr(
        execution,
        "write_visual_inspection_artifacts",
        lambda *_args, **_kwargs: (
            SimpleNamespace()
        ),
    )

    monkeypatch.setattr(
        execution,
        "build_inspection_metrics",
        lambda _run: (
            SimpleNamespace(
                to_dict=lambda: {}
            )
        ),
    )

    monkeypatch.setattr(
        execution,
        "write_inspection_metrics",
        lambda *_args, **_kwargs: (
            InspectionMetricsPath(
                json_path=(
                    tmp_path
                    / "inspection-metrics.json"
                )
            )
        ),
    )

    times = iter(
        [
            0.0,
            1.0,
            1.0,
            1.1,
            1.1,
            2.0,
        ]
    )

    monkeypatch.setattr(
        execution,
        "perf_counter",
        lambda: next(
            times
        ),
    )

    result = execution.execute_inspection(
        "example.mp4",
        make_profile(),
        tmp_path,
    )

    assert (
        result.review_dir.is_dir()
    )

    assert (
        result.rejected_dir.is_dir()
    )