"""Tests for inspection video and flagged-bottle visualization."""

from pathlib import Path

import numpy as np

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
from bottle_quality_inspection.roi import (
    RegionOfInterest,
)
from bottle_quality_inspection.visual_reporting import (
    annotate_inspection_frame,
    flagged_bottle_frame,
    flagged_bottles,
)


def make_result(
    bottle_id: str,
    status: BottleStatus,
) -> BottleInspectionResult:
    """Create a minimal deterministic bottle result."""
    check_status = (
        CheckStatus.CLEAR
        if status is BottleStatus.GOOD
        else CheckStatus.REVIEW
    )

    return BottleInspectionResult(
        bottle_id=bottle_id,
        status=status,
        foreign_object=(
            ForeignObjectBottleDecision(
                status=(
                    check_status
                ),
                observed_frames=5,
                clear_frames=5,
                defective_frames=0,
                maximum_black_pixels=0,
                median_black_pixels=0.0,
            )
        ),
        liquid_level=(
            LiquidLevelBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=5,
                valid_measurements=5,
                clear_frames=5,
                defective_frames=0,
                median_level=230.0,
                acceptable_minimum=225.0,
                acceptable_maximum=247.0,
            )
        ),
        glass=(
            GlassBottleDecision(
                status=CheckStatus.CLEAR,
                observed_frames=5,
                clear_frames=5,
                defective_frames=0,
                maximum_triggered_signals=0,
            )
        ),
        cork=(
            CorkBottleDecision(
                status=CheckStatus.CLEAR,
                presence_status=(
                    CorkPresenceStatus.PRESENT
                ),
                selected_frame_indices=(
                    1,
                    2,
                    3,
                ),
                top_k_cork_median=0.80,
                top_k_body_control_median=0.30,
                top_k_margin=0.50,
            )
        ),
    )


def make_tracked(
    bottle_id: str,
    status: BottleStatus,
) -> TrackedBottleInspection:
    """Create deterministic tracking metadata."""
    return TrackedBottleInspection(
        bottle_id=bottle_id,
        foreign_start_frame=10,
        foreign_end_frame=14,
        downstream_start_frame=20,
        downstream_end_frame=24,
        travel_frame_offset=10,
        travel_time_seconds=0.333,
        result=make_result(
            bottle_id,
            status,
        ),
    )


def make_run(
    *bottles: TrackedBottleInspection,
) -> BottleInspectionRun:
    """Create a deterministic visual-reporting run."""
    return BottleInspectionRun(
        video_path=Path(
            "example.mp4"
        ),
        profile_name="test",
        bottles=tuple(
            bottles
        ),
        unmatched_foreign_passages=(),
        unmatched_downstream_passages=(),
    )


def test_foreign_passage_is_annotated() -> None:
    """An active upstream bottle should draw the foreign ROI."""
    tracked = make_tracked(
        "bottle-001",
        BottleStatus.GOOD,
    )

    run = make_run(
        tracked
    )

    frame = np.zeros(
        (
            300,
            500,
            3,
        ),
        dtype=np.uint8,
    )

    foreign = RegionOfInterest(
        name="foreign",
        x=50,
        y=100,
        width=100,
        height=100,
    )

    downstream = RegionOfInterest(
        name="downstream",
        x=250,
        y=100,
        width=100,
        height=100,
    )

    annotated = annotate_inspection_frame(
        frame,
        12,
        run,
        foreign,
        downstream,
    )

    assert np.any(
        annotated[
            100,
            50,
        ]
        != 0
    )

    assert np.all(
        annotated[
            100,
            250,
        ]
        == 0
    )


def test_downstream_passage_is_annotated() -> None:
    """An active downstream bottle should draw the downstream ROI."""
    tracked = make_tracked(
        "bottle-001",
        BottleStatus.REVIEW,
    )

    run = make_run(
        tracked
    )

    frame = np.zeros(
        (
            300,
            500,
            3,
        ),
        dtype=np.uint8,
    )

    foreign = RegionOfInterest(
        name="foreign",
        x=50,
        y=100,
        width=100,
        height=100,
    )

    downstream = RegionOfInterest(
        name="downstream",
        x=250,
        y=100,
        width=100,
        height=100,
    )

    annotated = annotate_inspection_frame(
        frame,
        22,
        run,
        foreign,
        downstream,
    )

    assert np.all(
        annotated[
            100,
            50,
        ]
        == 0
    )

    assert np.any(
        annotated[
            100,
            250,
        ]
        != 0
    )


def test_only_non_good_bottles_are_flagged() -> None:
    """GOOD bottles should not create review or rejection images."""
    run = make_run(
        make_tracked(
            "bottle-001",
            BottleStatus.GOOD,
        ),
        make_tracked(
            "bottle-002",
            BottleStatus.REVIEW,
        ),
        make_tracked(
            "bottle-003",
            BottleStatus.DEFECTIVE,
        ),
    )

    selected = flagged_bottles(
        run
    )

    assert [
        tracked.bottle_id
        for tracked in selected
    ] == [
        "bottle-002",
        "bottle-003",
    ]


def test_flagged_frame_uses_downstream_midpoint() -> None:
    """Flagged evidence should use the center of downstream passage."""
    tracked = make_tracked(
        "bottle-002",
        BottleStatus.REVIEW,
    )

    assert (
        flagged_bottle_frame(
            tracked
        )
        == 22
    )