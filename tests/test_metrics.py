"""Tests for bottle inspection evaluation metrics."""

from types import SimpleNamespace

from bottle_quality_inspection.bottle_result import (
    BottleStatus,
    CheckStatus,
)
from bottle_quality_inspection.metrics import (
    CheckMetrics,
    build_inspection_metrics,
)


def make_check(
    status: CheckStatus,
) -> SimpleNamespace:
    """Create a minimal inspection-check result."""
    return SimpleNamespace(
        status=status
    )


def make_bottle(
    final_status: BottleStatus,
    *,
    foreign_object: CheckStatus,
    liquid_level: CheckStatus,
    glass: CheckStatus,
    cork: CheckStatus,
) -> SimpleNamespace:
    """Create a minimal tracked bottle for metric tests."""
    result = SimpleNamespace(
        status=final_status,
        foreign_object=make_check(
            foreign_object
        ),
        liquid_level=make_check(
            liquid_level
        ),
        glass=make_check(
            glass
        ),
        cork=make_check(
            cork
        ),
    )

    return SimpleNamespace(
        result=result
    )


def make_run() -> SimpleNamespace:
    """Create one deterministic inspection run."""
    bottles = (
        make_bottle(
            BottleStatus.GOOD,
            foreign_object=(
                CheckStatus.CLEAR
            ),
            liquid_level=(
                CheckStatus.CLEAR
            ),
            glass=(
                CheckStatus.CLEAR
            ),
            cork=(
                CheckStatus.CLEAR
            ),
        ),
        make_bottle(
            BottleStatus.REVIEW,
            foreign_object=(
                CheckStatus.REVIEW
            ),
            liquid_level=(
                CheckStatus.CLEAR
            ),
            glass=(
                CheckStatus.CLEAR
            ),
            cork=(
                CheckStatus.CLEAR
            ),
        ),
        make_bottle(
            BottleStatus.DEFECTIVE,
            foreign_object=(
                CheckStatus.CLEAR
            ),
            liquid_level=(
                CheckStatus.DEFECTIVE
            ),
            glass=(
                CheckStatus.CLEAR
            ),
            cork=(
                CheckStatus.REVIEW
            ),
        ),
    )

    return SimpleNamespace(
        profile_name="test-profile",
        bottles=bottles,
        bottle_count=len(
            bottles
        ),
        unmatched_foreign_passages=(
            object(),
        ),
        unmatched_downstream_passages=(
            object(),
            object(),
        ),
    )


def test_check_metrics_rates() -> None:
    """Check-level rates should use the total evaluated bottles."""
    metrics = CheckMetrics(
        clear=7,
        review=2,
        defective=1,
    )

    assert metrics.total == 10
    assert metrics.review_rate == 0.2
    assert metrics.defective_rate == 0.1


def test_build_inspection_metrics_counts_final_statuses() -> None:
    """Run metrics should count bottle-level final outcomes."""
    metrics = build_inspection_metrics(
        make_run()
    )

    assert (
        metrics.associated_bottles
        == 3
    )

    assert metrics.good == 1
    assert metrics.review == 1
    assert metrics.defective == 1

    assert (
        metrics.review_rate
        == 1 / 3
    )

    assert (
        metrics.defective_rate
        == 1 / 3
    )


def test_build_inspection_metrics_counts_checks() -> None:
    """Each inspection check should be summarized independently."""
    metrics = build_inspection_metrics(
        make_run()
    )

    assert (
        metrics.foreign_object.clear
        == 2
    )

    assert (
        metrics.foreign_object.review
        == 1
    )

    assert (
        metrics.liquid_level.defective
        == 1
    )

    assert metrics.glass.clear == 3

    assert metrics.cork.review == 1


def test_build_inspection_metrics_counts_unmatched_passages() -> None:
    """Unmatched station passages should remain visible in metrics."""
    metrics = build_inspection_metrics(
        make_run()
    )

    assert (
        metrics.unmatched_foreign_passages
        == 1
    )

    assert (
        metrics.unmatched_downstream_passages
        == 2
    )


def test_metrics_dictionary_is_serializable_structure() -> None:
    """Metric conversion should produce a stable nested structure."""
    data = (
        build_inspection_metrics(
            make_run()
        )
        .to_dict()
    )

    assert (
        data["profile"]
        == "test-profile"
    )

    assert (
        data["final_status"]["good"]
        == 1
    )

    assert (
        data[
            "checks"
        ][
            "foreign_object"
        ][
            "review"
        ]
        == 1
    )

    assert (
        data[
            "unmatched_passages"
        ][
            "downstream"
        ]
        == 2
    )