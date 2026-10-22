"""Bottle-level inspection evaluation metrics."""

from dataclasses import dataclass

from bottle_quality_inspection.bottle_pipeline import (
    BottleInspectionRun,
)
from bottle_quality_inspection.bottle_result import (
    BottleStatus,
    CheckStatus,
)


@dataclass(frozen=True, slots=True)
class CheckMetrics:
    """Aggregated outcomes for one inspection check."""

    clear: int
    review: int
    defective: int

    @property
    def total(self) -> int:
        """Return the number of evaluated bottles."""
        return (
            self.clear
            + self.review
            + self.defective
        )

    @property
    def review_rate(self) -> float:
        """Return the fraction of bottles requiring review."""
        if self.total == 0:
            return 0.0

        return (
            self.review
            / self.total
        )

    @property
    def defective_rate(self) -> float:
        """Return the fraction of bottles classified defective."""
        if self.total == 0:
            return 0.0

        return (
            self.defective
            / self.total
        )

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-compatible check metrics."""
        return {
            "total": self.total,
            "clear": self.clear,
            "review": self.review,
            "defective": self.defective,
            "review_rate": self.review_rate,
            "defective_rate": (
                self.defective_rate
            ),
        }


@dataclass(frozen=True, slots=True)
class InspectionRunMetrics:
    """Aggregated quality metrics for one inspection run."""

    profile_name: str

    associated_bottles: int

    good: int
    review: int
    defective: int

    unmatched_foreign_passages: int
    unmatched_downstream_passages: int

    foreign_object: CheckMetrics
    liquid_level: CheckMetrics
    glass: CheckMetrics
    cork: CheckMetrics

    @property
    def review_rate(self) -> float:
        """Return the final bottle review rate."""
        if self.associated_bottles == 0:
            return 0.0

        return (
            self.review
            / self.associated_bottles
        )

    @property
    def defective_rate(self) -> float:
        """Return the final bottle defect rate."""
        if self.associated_bottles == 0:
            return 0.0

        return (
            self.defective
            / self.associated_bottles
        )

    def to_dict(self) -> dict:
        """Return JSON-compatible run metrics."""
        return {
            "profile": self.profile_name,
            "associated_bottles": (
                self.associated_bottles
            ),
            "final_status": {
                "good": self.good,
                "review": self.review,
                "defective": (
                    self.defective
                ),
                "review_rate": (
                    self.review_rate
                ),
                "defective_rate": (
                    self.defective_rate
                ),
            },
            "unmatched_passages": {
                "foreign_object": (
                    self.unmatched_foreign_passages
                ),
                "downstream": (
                    self.unmatched_downstream_passages
                ),
            },
            "checks": {
                "foreign_object": (
                    self.foreign_object.to_dict()
                ),
                "liquid_level": (
                    self.liquid_level.to_dict()
                ),
                "glass": (
                    self.glass.to_dict()
                ),
                "cork": (
                    self.cork.to_dict()
                ),
            },
        }


def _count_check_statuses(
    run: BottleInspectionRun,
    check_name: str,
) -> CheckMetrics:
    """Count CLEAR, REVIEW and DEFECTIVE outcomes for one check."""
    clear = 0
    review = 0
    defective = 0

    for tracked in run.bottles:
        check = getattr(
            tracked.result,
            check_name,
        )

        if check.status is CheckStatus.CLEAR:
            clear += 1

        elif (
            check.status
            is CheckStatus.REVIEW
        ):
            review += 1

        elif (
            check.status
            is CheckStatus.DEFECTIVE
        ):
            defective += 1

    return CheckMetrics(
        clear=clear,
        review=review,
        defective=defective,
    )


def build_inspection_metrics(
    run: BottleInspectionRun,
) -> InspectionRunMetrics:
    """Aggregate bottle and check outcomes for one inspection run."""
    good = sum(
        tracked.result.status
        is BottleStatus.GOOD
        for tracked in run.bottles
    )

    review = sum(
        tracked.result.status
        is BottleStatus.REVIEW
        for tracked in run.bottles
    )

    defective = sum(
        tracked.result.status
        is BottleStatus.DEFECTIVE
        for tracked in run.bottles
    )

    return InspectionRunMetrics(
        profile_name=run.profile_name,
        associated_bottles=(
            run.bottle_count
        ),
        good=good,
        review=review,
        defective=defective,
        unmatched_foreign_passages=len(
            run.unmatched_foreign_passages
        ),
        unmatched_downstream_passages=len(
            run.unmatched_downstream_passages
        ),
        foreign_object=(
            _count_check_statuses(
                run,
                "foreign_object",
            )
        ),
        liquid_level=(
            _count_check_statuses(
                run,
                "liquid_level",
            )
        ),
        glass=(
            _count_check_statuses(
                run,
                "glass",
            )
        ),
        cork=(
            _count_check_statuses(
                run,
                "cork",
            )
        ),
    )