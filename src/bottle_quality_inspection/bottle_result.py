"""Per-bottle aggregation of multi-frame inspection evidence."""

from dataclasses import dataclass
from enum import StrEnum
from statistics import median

from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceResult,
    CorkPresenceStatus,
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


class CheckStatus(StrEnum):
    """Bottle-level outcome for one quality check."""

    CLEAR = "clear"
    DEFECTIVE = "defective"
    REVIEW = "review"


class BottleStatus(StrEnum):
    """Final quality outcome for one physical bottle."""

    GOOD = "good"
    DEFECTIVE = "defective"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class BottleAggregationConfig:
    """Rules controlling multi-frame bottle aggregation."""

    foreign_confirmed_defect_frames: int = 2
    glass_confirmed_defect_frames: int = 2

    minimum_liquid_measurements: int = 3

    liquid_acceptable_minimum: float = 225.0
    liquid_acceptable_maximum: float = 247.0

    def __post_init__(self) -> None:
        """Validate bottle-level aggregation settings."""
        if self.foreign_confirmed_defect_frames <= 0:
            raise ValueError(
                "foreign_confirmed_defect_frames "
                "must be greater than zero."
            )

        if self.glass_confirmed_defect_frames <= 0:
            raise ValueError(
                "glass_confirmed_defect_frames "
                "must be greater than zero."
            )

        if self.minimum_liquid_measurements <= 0:
            raise ValueError(
                "minimum_liquid_measurements "
                "must be greater than zero."
            )

        if (
            self.liquid_acceptable_minimum
            >= self.liquid_acceptable_maximum
        ):
            raise ValueError(
                "liquid_acceptable_minimum must be "
                "lower than liquid_acceptable_maximum."
            )


@dataclass(frozen=True, slots=True)
class ForeignObjectFrameEvidence:
    """Foreign-object evidence from one tracked upstream frame."""

    frame_index: int
    status: ForeignObjectStatus
    decision_black_pixels: int

    def __post_init__(self) -> None:
        """Validate frame-level foreign-object evidence."""
        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative."
            )

        if self.decision_black_pixels < 0:
            raise ValueError(
                "decision_black_pixels cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class LiquidLevelFrameEvidence:
    """Liquid-level evidence from one tracked downstream frame."""

    frame_index: int
    status: LiquidLevelStatus
    level: float | None

    def __post_init__(self) -> None:
        """Validate frame-level liquid evidence."""
        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class GlassFrameEvidence:
    """Glass-integrity evidence from one tracked downstream frame."""

    frame_index: int
    status: GlassStatus
    triggered_signal_count: int

    def __post_init__(self) -> None:
        """Validate frame-level glass evidence."""
        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative."
            )

        if self.triggered_signal_count < 0:
            raise ValueError(
                "triggered_signal_count cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class ForeignObjectBottleDecision:
    """Aggregated foreign-object decision for one bottle."""

    status: CheckStatus

    observed_frames: int
    clear_frames: int
    defective_frames: int

    maximum_black_pixels: int
    median_black_pixels: float


@dataclass(frozen=True, slots=True)
class LiquidLevelBottleDecision:
    """Aggregated liquid-level decision for one bottle."""

    status: CheckStatus

    observed_frames: int
    valid_measurements: int

    clear_frames: int
    defective_frames: int

    median_level: float | None

    acceptable_minimum: float
    acceptable_maximum: float


@dataclass(frozen=True, slots=True)
class GlassBottleDecision:
    """Aggregated glass-integrity decision for one bottle."""

    status: CheckStatus

    observed_frames: int
    clear_frames: int
    defective_frames: int

    maximum_triggered_signals: int


@dataclass(frozen=True, slots=True)
class CorkBottleDecision:
    """Bottle-level interpretation of cork-presence evidence."""

    status: CheckStatus

    presence_status: CorkPresenceStatus

    selected_frame_indices: tuple[
        int,
        ...
    ]

    top_k_cork_median: float
    top_k_body_control_median: float
    top_k_margin: float


@dataclass(frozen=True, slots=True)
class BottleInspectionResult:
    """Unified inspection result for one physical tracked bottle."""

    bottle_id: str

    status: BottleStatus

    foreign_object: ForeignObjectBottleDecision
    liquid_level: LiquidLevelBottleDecision
    glass: GlassBottleDecision
    cork: CorkBottleDecision

    @property
    def defective_checks(self) -> tuple[str, ...]:
        """Return check names containing confirmed defects."""
        checks = {
            "foreign_object": self.foreign_object.status,
            "liquid_level": self.liquid_level.status,
            "glass": self.glass.status,
            "cork": self.cork.status,
        }

        return tuple(
            name
            for name, status in checks.items()
            if status is CheckStatus.DEFECTIVE
        )

    @property
    def review_checks(self) -> tuple[str, ...]:
        """Return check names requiring human or later review."""
        checks = {
            "foreign_object": self.foreign_object.status,
            "liquid_level": self.liquid_level.status,
            "glass": self.glass.status,
            "cork": self.cork.status,
        }

        return tuple(
            name
            for name, status in checks.items()
            if status is CheckStatus.REVIEW
        )


def aggregate_foreign_object(
    evidence: list[ForeignObjectFrameEvidence],
    config: BottleAggregationConfig | None = None,
) -> ForeignObjectBottleDecision:
    """Aggregate foreign-object evidence across one bottle passage."""
    active_config = (
        config
        if config is not None
        else BottleAggregationConfig()
    )

    valid = [
        item
        for item in evidence
        if item.status
        in {
            ForeignObjectStatus.CLEAR,
            ForeignObjectStatus.DEFECTIVE,
        }
    ]

    if not valid:
        return ForeignObjectBottleDecision(
            status=CheckStatus.REVIEW,
            observed_frames=len(
                evidence
            ),
            clear_frames=0,
            defective_frames=0,
            maximum_black_pixels=0,
            median_black_pixels=0.0,
        )

    clear_frames = sum(
        item.status
        is ForeignObjectStatus.CLEAR
        for item in valid
    )

    defective_frames = sum(
        item.status
        is ForeignObjectStatus.DEFECTIVE
        for item in valid
    )

    black_counts = [
        item.decision_black_pixels
        for item in valid
    ]

    if (
        defective_frames
        >= active_config.foreign_confirmed_defect_frames
    ):
        status = CheckStatus.DEFECTIVE

    elif defective_frames == 0:
        status = CheckStatus.CLEAR

    else:
        status = CheckStatus.REVIEW

    return ForeignObjectBottleDecision(
        status=status,
        observed_frames=len(
            valid
        ),
        clear_frames=(
            clear_frames
        ),
        defective_frames=(
            defective_frames
        ),
        maximum_black_pixels=max(
            black_counts
        ),
        median_black_pixels=float(
            median(
                black_counts
            )
        ),
    )


def aggregate_liquid_level(
    evidence: list[LiquidLevelFrameEvidence],
    config: BottleAggregationConfig | None = None,
) -> LiquidLevelBottleDecision:
    """Aggregate liquid-level measurements using passage median."""
    active_config = (
        config
        if config is not None
        else BottleAggregationConfig()
    )

    valid = [
        item
        for item in evidence
        if (
            item.status
            is not LiquidLevelStatus.NO_LEVEL
            and item.level is not None
        )
    ]

    clear_frames = sum(
        item.status
        is LiquidLevelStatus.CLEAR
        for item in valid
    )

    defective_frames = sum(
        item.status
        is LiquidLevelStatus.DEFECTIVE
        for item in valid
    )

    if not valid:
        median_level = None

    else:
        median_level = float(
            median(
                [
                    item.level
                    for item in valid
                    if item.level is not None
                ]
            )
        )

    if (
        len(valid)
        < active_config.minimum_liquid_measurements
    ):
        status = CheckStatus.REVIEW

    elif median_level is None:
        status = CheckStatus.REVIEW

    elif (
        active_config.liquid_acceptable_minimum
        < median_level
        < active_config.liquid_acceptable_maximum
    ):
        status = CheckStatus.CLEAR

    else:
        status = CheckStatus.DEFECTIVE

    return LiquidLevelBottleDecision(
        status=status,
        observed_frames=len(
            evidence
        ),
        valid_measurements=len(
            valid
        ),
        clear_frames=(
            clear_frames
        ),
        defective_frames=(
            defective_frames
        ),
        median_level=(
            median_level
        ),
        acceptable_minimum=(
            active_config.liquid_acceptable_minimum
        ),
        acceptable_maximum=(
            active_config.liquid_acceptable_maximum
        ),
    )


def aggregate_glass(
    evidence: list[GlassFrameEvidence],
    config: BottleAggregationConfig | None = None,
) -> GlassBottleDecision:
    """Aggregate calibrated glass classifications across a bottle."""
    active_config = (
        config
        if config is not None
        else BottleAggregationConfig()
    )

    if not evidence:
        return GlassBottleDecision(
            status=CheckStatus.REVIEW,
            observed_frames=0,
            clear_frames=0,
            defective_frames=0,
            maximum_triggered_signals=0,
        )

    clear_frames = sum(
        item.status
        is GlassStatus.CLEAR
        for item in evidence
    )

    defective_frames = sum(
        item.status
        is GlassStatus.DEFECTIVE
        for item in evidence
    )

    if (
        defective_frames
        >= active_config.glass_confirmed_defect_frames
    ):
        status = CheckStatus.DEFECTIVE

    elif defective_frames == 0:
        status = CheckStatus.CLEAR

    else:
        status = CheckStatus.REVIEW

    return GlassBottleDecision(
        status=status,
        observed_frames=len(
            evidence
        ),
        clear_frames=(
            clear_frames
        ),
        defective_frames=(
            defective_frames
        ),
        maximum_triggered_signals=max(
            item.triggered_signal_count
            for item in evidence
        ),
    )


def aggregate_cork_presence(
    result: CorkPresenceResult,
) -> CorkBottleDecision:
    """Map calibrated cork-presence evidence to bottle-level status."""
    status = (
        CheckStatus.CLEAR
        if (
            result.status
            is CorkPresenceStatus.PRESENT
        )
        else CheckStatus.REVIEW
    )

    return CorkBottleDecision(
        status=status,
        presence_status=(
            result.status
        ),
        selected_frame_indices=(
            result.selected_frame_indices
        ),
        top_k_cork_median=(
            result.top_k_cork_median
        ),
        top_k_body_control_median=(
            result.top_k_body_control_median
        ),
        top_k_margin=(
            result.top_k_margin
        ),
    )


def determine_bottle_status(
    *,
    foreign_object: ForeignObjectBottleDecision,
    liquid_level: LiquidLevelBottleDecision,
    glass: GlassBottleDecision,
    cork: CorkBottleDecision,
) -> BottleStatus:
    """Calculate final bottle status from all required checks."""
    statuses = (
        foreign_object.status,
        liquid_level.status,
        glass.status,
        cork.status,
    )

    if CheckStatus.DEFECTIVE in statuses:
        return BottleStatus.DEFECTIVE

    if CheckStatus.REVIEW in statuses:
        return BottleStatus.REVIEW

    return BottleStatus.GOOD


def build_bottle_inspection_result(
    bottle_id: str,
    *,
    foreign_object: ForeignObjectBottleDecision,
    liquid_level: LiquidLevelBottleDecision,
    glass: GlassBottleDecision,
    cork: CorkBottleDecision,
) -> BottleInspectionResult:
    """Create one unified physical-bottle result."""
    if not bottle_id.strip():
        raise ValueError(
            "bottle_id cannot be empty."
        )

    status = determine_bottle_status(
        foreign_object=(
            foreign_object
        ),
        liquid_level=(
            liquid_level
        ),
        glass=glass,
        cork=cork,
    )

    return BottleInspectionResult(
        bottle_id=bottle_id,
        status=status,
        foreign_object=(
            foreign_object
        ),
        liquid_level=(
            liquid_level
        ),
        glass=glass,
        cork=cork,
    )