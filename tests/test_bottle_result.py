"""Tests for physical-bottle multi-frame aggregation."""

from bottle_quality_inspection.bottle_result import (
    BottleStatus,
    CheckStatus,
    ForeignObjectFrameEvidence,
    GlassFrameEvidence,
    LiquidLevelFrameEvidence,
    aggregate_cork_presence,
    aggregate_foreign_object,
    aggregate_glass,
    aggregate_liquid_level,
    build_bottle_inspection_result,
)
from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceFrameScore,
    evaluate_cork_presence,
)
from bottle_quality_inspection.inspections.foreign_object import (
    ForeignObjectStatus,
)
from bottle_quality_inspection.inspections.glass import (
    GlassStatus,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelStatus,
)


def present_cork():
    """Create calibrated cork-presence evidence."""
    return evaluate_cork_presence(
        [
            CorkPresenceFrameScore(
                frame_index=1,
                cork_score=0.78,
                body_control_score=0.30,
            ),
            CorkPresenceFrameScore(
                frame_index=2,
                cork_score=0.79,
                body_control_score=0.31,
            ),
            CorkPresenceFrameScore(
                frame_index=3,
                cork_score=0.77,
                body_control_score=0.32,
            ),
        ]
    )


def unverified_cork():
    """Create intentionally weak cork-presence evidence."""
    return evaluate_cork_presence(
        [
            CorkPresenceFrameScore(
                frame_index=1,
                cork_score=0.60,
                body_control_score=0.45,
            ),
            CorkPresenceFrameScore(
                frame_index=2,
                cork_score=0.61,
                body_control_score=0.46,
            ),
            CorkPresenceFrameScore(
                frame_index=3,
                cork_score=0.62,
                body_control_score=0.47,
            ),
        ]
    )


def clear_foreign():
    """Create a clear foreign-object bottle decision."""
    return aggregate_foreign_object(
        [
            ForeignObjectFrameEvidence(
                frame_index=1,
                status=(
                    ForeignObjectStatus.CLEAR
                ),
                decision_black_pixels=0,
            ),
            ForeignObjectFrameEvidence(
                frame_index=2,
                status=(
                    ForeignObjectStatus.CLEAR
                ),
                decision_black_pixels=5,
            ),
        ]
    )


def clear_liquid():
    """Create a clear liquid-level bottle decision."""
    return aggregate_liquid_level(
        [
            LiquidLevelFrameEvidence(
                frame_index=1,
                status=(
                    LiquidLevelStatus.CLEAR
                ),
                level=230.0,
            ),
            LiquidLevelFrameEvidence(
                frame_index=2,
                status=(
                    LiquidLevelStatus.CLEAR
                ),
                level=231.0,
            ),
            LiquidLevelFrameEvidence(
                frame_index=3,
                status=(
                    LiquidLevelStatus.CLEAR
                ),
                level=232.0,
            ),
        ]
    )


def clear_glass():
    """Create a clear glass bottle decision."""
    return aggregate_glass(
        [
            GlassFrameEvidence(
                frame_index=1,
                status=GlassStatus.CLEAR,
                triggered_signal_count=0,
            ),
            GlassFrameEvidence(
                frame_index=2,
                status=GlassStatus.CLEAR,
                triggered_signal_count=1,
            ),
        ]
    )


def test_foreign_all_clear_is_clear() -> None:
    """No foreign-object defect frames should pass."""
    result = clear_foreign()

    assert result.status is CheckStatus.CLEAR
    assert result.defective_frames == 0


def test_single_foreign_defect_requires_review() -> None:
    """One isolated foreign-object spike should not confirm a defect."""
    result = aggregate_foreign_object(
        [
            ForeignObjectFrameEvidence(
                frame_index=1,
                status=(
                    ForeignObjectStatus.CLEAR
                ),
                decision_black_pixels=0,
            ),
            ForeignObjectFrameEvidence(
                frame_index=2,
                status=(
                    ForeignObjectStatus.DEFECTIVE
                ),
                decision_black_pixels=990,
            ),
            ForeignObjectFrameEvidence(
                frame_index=3,
                status=(
                    ForeignObjectStatus.CLEAR
                ),
                decision_black_pixels=768,
            ),
        ]
    )

    assert result.status is CheckStatus.REVIEW
    assert result.defective_frames == 1


def test_repeated_foreign_defect_is_defective() -> None:
    """Repeated threshold crossings should confirm a foreign defect."""
    result = aggregate_foreign_object(
        [
            ForeignObjectFrameEvidence(
                frame_index=1,
                status=(
                    ForeignObjectStatus.DEFECTIVE
                ),
                decision_black_pixels=950,
            ),
            ForeignObjectFrameEvidence(
                frame_index=2,
                status=(
                    ForeignObjectStatus.DEFECTIVE
                ),
                decision_black_pixels=1100,
            ),
        ]
    )

    assert (
        result.status
        is CheckStatus.DEFECTIVE
    )


def test_liquid_transition_outliers_use_median() -> None:
    """Entry/exit outliers should not override stable bottle level."""
    result = aggregate_liquid_level(
        [
            LiquidLevelFrameEvidence(
                40,
                LiquidLevelStatus.DEFECTIVE,
                223.5,
            ),
            LiquidLevelFrameEvidence(
                41,
                LiquidLevelStatus.CLEAR,
                227.0,
            ),
            LiquidLevelFrameEvidence(
                42,
                LiquidLevelStatus.CLEAR,
                228.5,
            ),
            LiquidLevelFrameEvidence(
                43,
                LiquidLevelStatus.CLEAR,
                227.0,
            ),
            LiquidLevelFrameEvidence(
                44,
                LiquidLevelStatus.CLEAR,
                228.5,
            ),
            LiquidLevelFrameEvidence(
                45,
                LiquidLevelStatus.CLEAR,
                229.0,
            ),
            LiquidLevelFrameEvidence(
                46,
                LiquidLevelStatus.CLEAR,
                227.5,
            ),
            LiquidLevelFrameEvidence(
                47,
                LiquidLevelStatus.DEFECTIVE,
                223.5,
            ),
        ]
    )

    assert result.status is CheckStatus.CLEAR

    assert (
        result.median_level
        == 227.25
    )


def test_liquid_median_outside_range_is_defective() -> None:
    """A stable out-of-range bottle level should fail."""
    result = aggregate_liquid_level(
        [
            LiquidLevelFrameEvidence(
                1,
                LiquidLevelStatus.DEFECTIVE,
                220.0,
            ),
            LiquidLevelFrameEvidence(
                2,
                LiquidLevelStatus.DEFECTIVE,
                221.0,
            ),
            LiquidLevelFrameEvidence(
                3,
                LiquidLevelStatus.DEFECTIVE,
                222.0,
            ),
        ]
    )

    assert (
        result.status
        is CheckStatus.DEFECTIVE
    )


def test_insufficient_liquid_measurements_require_review() -> None:
    """Too little liquid evidence should not become a hard decision."""
    result = aggregate_liquid_level(
        [
            LiquidLevelFrameEvidence(
                1,
                LiquidLevelStatus.CLEAR,
                230.0,
            ),
            LiquidLevelFrameEvidence(
                2,
                LiquidLevelStatus.CLEAR,
                231.0,
            ),
        ]
    )

    assert result.status is CheckStatus.REVIEW


def test_glass_all_clear_is_clear() -> None:
    """No glass defect frames should pass."""
    result = clear_glass()

    assert result.status is CheckStatus.CLEAR


def test_single_glass_defect_requires_review() -> None:
    """One isolated glass defect frame should require review."""
    result = aggregate_glass(
        [
            GlassFrameEvidence(
                frame_index=1,
                status=GlassStatus.CLEAR,
                triggered_signal_count=0,
            ),
            GlassFrameEvidence(
                frame_index=2,
                status=GlassStatus.DEFECTIVE,
                triggered_signal_count=3,
            ),
        ]
    )

    assert result.status is CheckStatus.REVIEW


def test_repeated_glass_defect_is_defective() -> None:
    """Repeated glass defect frames should confirm damage."""
    result = aggregate_glass(
        [
            GlassFrameEvidence(
                frame_index=1,
                status=GlassStatus.DEFECTIVE,
                triggered_signal_count=4,
            ),
            GlassFrameEvidence(
                frame_index=2,
                status=GlassStatus.DEFECTIVE,
                triggered_signal_count=3,
            ),
        ]
    )

    assert (
        result.status
        is CheckStatus.DEFECTIVE
    )


def test_present_cork_maps_to_clear() -> None:
    """Confirmed cork presence should satisfy the cork check."""
    result = aggregate_cork_presence(
        present_cork()
    )

    assert result.status is CheckStatus.CLEAR


def test_unverified_cork_maps_to_review() -> None:
    """Weak cork evidence should require review, not defect."""
    result = aggregate_cork_presence(
        unverified_cork()
    )

    assert result.status is CheckStatus.REVIEW


def test_all_clear_checks_make_good_bottle() -> None:
    """A bottle passing every required check should be GOOD."""
    result = build_bottle_inspection_result(
        "bottle-001",
        foreign_object=clear_foreign(),
        liquid_level=clear_liquid(),
        glass=clear_glass(),
        cork=aggregate_cork_presence(
            present_cork()
        ),
    )

    assert result.status is BottleStatus.GOOD
    assert result.defective_checks == ()
    assert result.review_checks == ()


def test_review_check_makes_bottle_review() -> None:
    """Uncertain evidence should propagate to final REVIEW."""
    result = build_bottle_inspection_result(
        "bottle-001",
        foreign_object=clear_foreign(),
        liquid_level=clear_liquid(),
        glass=clear_glass(),
        cork=aggregate_cork_presence(
            unverified_cork()
        ),
    )

    assert result.status is BottleStatus.REVIEW

    assert result.review_checks == (
        "cork",
    )


def test_confirmed_defect_has_highest_priority() -> None:
    """A confirmed defect should override another review condition."""
    foreign = aggregate_foreign_object(
        [
            ForeignObjectFrameEvidence(
                1,
                ForeignObjectStatus.DEFECTIVE,
                1000,
            ),
            ForeignObjectFrameEvidence(
                2,
                ForeignObjectStatus.DEFECTIVE,
                1050,
            ),
        ]
    )

    result = build_bottle_inspection_result(
        "bottle-001",
        foreign_object=foreign,
        liquid_level=clear_liquid(),
        glass=clear_glass(),
        cork=aggregate_cork_presence(
            unverified_cork()
        ),
    )

    assert (
        result.status
        is BottleStatus.DEFECTIVE
    )

    assert result.defective_checks == (
        "foreign_object",
    )

    assert result.review_checks == (
        "cork",
    )