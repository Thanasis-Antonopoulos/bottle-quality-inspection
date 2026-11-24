"""Runtime and performance metrics for complete inspection runs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bottle_quality_inspection.metrics import (
    InspectionRunMetrics,
)
from bottle_quality_inspection.video_io import (
    VideoMetadata,
)


@dataclass(frozen=True, slots=True)
class InspectionTimingMetrics:
    """Timing and throughput measurements for one inspection execution."""

    source_frame_count: int
    source_fps: float
    source_duration_seconds: float | None

    inspection_seconds: float
    report_seconds: float
    visual_seconds: float
    total_seconds: float

    @property
    def effective_processing_fps(
        self,
    ) -> float | None:
        """Return source-frame throughput across the complete pipeline."""
        if self.total_seconds <= 0:
            return None

        return (
            self.source_frame_count
            / self.total_seconds
        )

    @property
    def realtime_factor(
        self,
    ) -> float | None:
        """
        Return source-video duration divided by processing duration.

        Values greater than 1.0 mean the complete pipeline processed the
        recording faster than real time.
        """
        if (
            self.source_duration_seconds
            is None
            or self.total_seconds <= 0
        ):
            return None

        return (
            self.source_duration_seconds
            / self.total_seconds
        )

    def to_dict(
        self,
    ) -> dict[str, int | float | None]:
        """Return JSON-compatible timing metrics."""
        return {
            "source_frame_count": (
                self.source_frame_count
            ),
            "source_fps": (
                self.source_fps
            ),
            "source_duration_seconds": (
                self.source_duration_seconds
            ),
            "inspection_seconds": (
                self.inspection_seconds
            ),
            "report_seconds": (
                self.report_seconds
            ),
            "visual_seconds": (
                self.visual_seconds
            ),
            "total_seconds": (
                self.total_seconds
            ),
            "effective_processing_fps": (
                self.effective_processing_fps
            ),
            "realtime_factor": (
                self.realtime_factor
            ),
        }


@dataclass(frozen=True, slots=True)
class InspectionEvaluationReport:
    """Combined quality and runtime metrics for one inspection run."""

    quality: InspectionRunMetrics
    timing: InspectionTimingMetrics

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return the complete evaluation payload."""
        return {
            "schema_version": 1,
            "quality": (
                self.quality.to_dict()
            ),
            "timing": (
                self.timing.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class InspectionMetricsPath:
    """Path written for one metrics export."""

    json_path: Path


def build_timing_metrics(
    metadata: VideoMetadata,
    *,
    inspection_seconds: float,
    report_seconds: float,
    visual_seconds: float,
    total_seconds: float,
) -> InspectionTimingMetrics:
    """Build validated timing metrics from one completed execution."""
    timings = {
        "inspection_seconds": inspection_seconds,
        "report_seconds": report_seconds,
        "visual_seconds": visual_seconds,
        "total_seconds": total_seconds,
    }

    for name, value in timings.items():
        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    return InspectionTimingMetrics(
        source_frame_count=(
            metadata.frame_count
        ),
        source_fps=(
            metadata.fps
        ),
        source_duration_seconds=(
            metadata.duration_seconds
        ),
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


def write_inspection_metrics(
    report: InspectionEvaluationReport,
    output_dir: str | Path,
    *,
    filename: str = "inspection-metrics.json",
    overwrite: bool = False,
) -> InspectionMetricsPath:
    """Write inspection quality and runtime metrics as JSON."""
    output_directory = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    output_filename = Path(
        filename
    )

    if (
        output_filename.name
        != filename
        or output_filename.suffix.lower()
        != ".json"
    ):
        raise ValueError(
            "Metrics filename must be a JSON filename, not a path."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / output_filename
    )

    if json_path.exists():
        if not overwrite:
            raise FileExistsError(
                "Inspection metrics already exist. "
                "Use overwrite=True to replace them: "
                f"{json_path}"
            )

        if not json_path.is_file():
            raise IsADirectoryError(
                "Metrics output path is not a file: "
                f"{json_path}"
            )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            report.to_dict(),
            json_file,
            indent=2,
        )

        json_file.write(
            "\n"
        )

    return InspectionMetricsPath(
        json_path=json_path
    )