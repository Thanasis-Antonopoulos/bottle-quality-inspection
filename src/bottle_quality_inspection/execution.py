"""Orchestration for complete bottle-inspection executions."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from bottle_quality_inspection.bottle_pipeline import (
    BottleInspectionRun,
    inspect_tracked_bottles,
)
from bottle_quality_inspection.config import (
    InspectionConfig,
)
from bottle_quality_inspection.metrics import (
    InspectionRunMetrics,
    build_inspection_metrics,
)
from bottle_quality_inspection.reporting import (
    InspectionReportPaths,
    write_inspection_reports,
)
from bottle_quality_inspection.runtime_metrics import (
    InspectionEvaluationReport,
    InspectionMetricsPath,
    InspectionTimingMetrics,
    build_timing_metrics,
    write_inspection_metrics,
)
from bottle_quality_inspection.video_io import (
    VideoMetadata,
    read_video_metadata,
)
from bottle_quality_inspection.visual_reporting import (
    VisualInspectionArtifacts,
    write_visual_inspection_artifacts,
)


@dataclass(frozen=True, slots=True)
class InspectionExecutionResult:
    """Artifacts and measurements produced by one complete execution."""

    metadata: VideoMetadata
    run: BottleInspectionRun

    reports: InspectionReportPaths
    visuals: VisualInspectionArtifacts

    quality_metrics: InspectionRunMetrics
    timing_metrics: InspectionTimingMetrics

    metrics_path: InspectionMetricsPath

    output_dir: Path
    review_dir: Path
    rejected_dir: Path


def execute_inspection(
    video_path: str | Path,
    profile: InspectionConfig,
    output_dir: str | Path,
    *,
    stem: str = "inspection-results",
    metrics_filename: str = "inspection-metrics.json",
    video_filename: str = "annotated-inspection.mp4",
    codec: str = "mp4v",
    overwrite: bool = False,
    max_missed_frames: int = 1,
    minimum_detected_frames: int = 2,
    minimum_frame_offset: int = 1,
    maximum_frame_offset: int = 30,
) -> InspectionExecutionResult:
    """
    Run inspection, reporting, visual output, and evaluation metrics.

    Runtime measurements cover the three main production stages:

    - tracked bottle inspection
    - CSV/JSON report generation
    - visual artifact generation

    The final metrics-file write itself is intentionally excluded from
    total_seconds so the reported value remains the sum of those measured
    production stages.
    """
    destination = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    metadata = read_video_metadata(
        video_path
    )

    inspection_started = (
        perf_counter()
    )

    run = inspect_tracked_bottles(
        video_path,
        profile,
        max_missed_frames=(
            max_missed_frames
        ),
        minimum_detected_frames=(
            minimum_detected_frames
        ),
        minimum_frame_offset=(
            minimum_frame_offset
        ),
        maximum_frame_offset=(
            maximum_frame_offset
        ),
    )

    inspection_seconds = (
        perf_counter()
        - inspection_started
    )

    reporting_started = (
        perf_counter()
    )

    reports = write_inspection_reports(
        run,
        destination,
        stem=stem,
        overwrite=overwrite,
    )

    report_seconds = (
        perf_counter()
        - reporting_started
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

    visual_started = (
        perf_counter()
    )

    visuals = (
        write_visual_inspection_artifacts(
            run,
            destination,
            foreign_region,
            downstream_region,
            video_filename=(
                video_filename
            ),
            codec=codec,
            overwrite=overwrite,
        )
    )

    visual_seconds = (
        perf_counter()
        - visual_started
    )

    review_dir = (
        destination
        / "review"
    )

    rejected_dir = (
        destination
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

    quality_metrics = (
        build_inspection_metrics(
            run
        )
    )

    total_seconds = (
        inspection_seconds
        + report_seconds
        + visual_seconds
    )

    timing_metrics = (
        build_timing_metrics(
            metadata,
            inspection_seconds=(
                inspection_seconds
            ),
            report_seconds=(
                report_seconds
            ),
            visual_seconds=(
                visual_seconds
            ),
            total_seconds=(
                total_seconds
            ),
        )
    )

    evaluation_report = (
        InspectionEvaluationReport(
            quality=(
                quality_metrics
            ),
            timing=(
                timing_metrics
            ),
        )
    )

    metrics_path = (
        write_inspection_metrics(
            evaluation_report,
            destination,
            filename=(
                metrics_filename
            ),
            overwrite=overwrite,
        )
    )

    return InspectionExecutionResult(
        metadata=metadata,
        run=run,
        reports=reports,
        visuals=visuals,
        quality_metrics=(
            quality_metrics
        ),
        timing_metrics=(
            timing_metrics
        ),
        metrics_path=(
            metrics_path
        ),
        output_dir=destination,
        review_dir=review_dir,
        rejected_dir=rejected_dir,
    )