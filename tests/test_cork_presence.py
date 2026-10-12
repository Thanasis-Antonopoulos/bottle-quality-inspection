"""Tests for tracked multi-frame cork-presence evaluation."""

import pytest

from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceConfig,
    CorkPresenceFrameScore,
    CorkPresenceStatus,
    evaluate_cork_presence,
)


def score(
    frame: int,
    cork: float,
    body: float,
) -> CorkPresenceFrameScore:
    """Create one deterministic frame-level score."""
    return CorkPresenceFrameScore(
        frame_index=frame,
        cork_score=cork,
        body_control_score=body,
    )


def test_calibrated_normal_bottle_is_present() -> None:
    """Strong cork evidence with clear body separation should pass."""
    frame_scores = [
        score(40, 0.50, 0.38),
        score(41, 0.41, 0.36),
        score(42, 0.768, 0.309),
        score(43, 0.773, 0.479),
        score(44, 0.761, 0.465),
        score(45, 0.766, 0.322),
    ]

    result = evaluate_cork_presence(
        frame_scores
    )

    assert (
        result.status
        is CorkPresenceStatus.PRESENT
    )

    assert result.cork_present is True

    assert (
        result.top_k_cork_median
        >= 0.75
    )

    assert (
        result.top_k_margin
        >= 0.20
    )


def test_strong_raw_score_without_separation_is_unverified() -> None:
    """A high score should not pass if body control is similarly high."""
    frame_scores = [
        score(1, 0.80, 0.72),
        score(2, 0.79, 0.71),
        score(3, 0.78, 0.70),
    ]

    result = evaluate_cork_presence(
        frame_scores
    )

    assert (
        result.status
        is CorkPresenceStatus.UNVERIFIED
    )

    assert result.cork_present is False


def test_good_separation_but_weak_cork_score_is_unverified() -> None:
    """Separation alone should not be enough without cork similarity."""
    frame_scores = [
        score(1, 0.65, 0.20),
        score(2, 0.66, 0.20),
        score(3, 0.67, 0.20),
    ]

    result = evaluate_cork_presence(
        frame_scores
    )

    assert (
        result.status
        is CorkPresenceStatus.UNVERIFIED
    )


def test_top_three_frames_are_used() -> None:
    """Entry and exit frames should not dominate stable cork evidence."""
    frame_scores = [
        score(1, 0.20, 0.30),
        score(2, 0.77, 0.31),
        score(3, 0.80, 0.34),
        score(4, 0.79, 0.33),
        score(5, 0.25, 0.35),
    ]

    result = evaluate_cork_presence(
        frame_scores
    )

    assert set(
        result.selected_frame_indices
    ) == {
        2,
        3,
        4,
    }

    assert result.evaluated_frame_count == 5

    assert (
        result.status
        is CorkPresenceStatus.PRESENT
    )


def test_fewer_than_three_frames_can_still_be_evaluated() -> None:
    """Short tracked passages should use all available observations."""
    frame_scores = [
        score(10, 0.80, 0.40),
        score(11, 0.78, 0.39),
    ]

    result = evaluate_cork_presence(
        frame_scores
    )

    assert result.evaluated_frame_count == 2

    assert len(
        result.selected_frame_indices
    ) == 2

    assert (
        result.status
        is CorkPresenceStatus.PRESENT
    )


def test_empty_input_is_rejected() -> None:
    """Presence evaluation requires actual frame evidence."""
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        evaluate_cork_presence(
            []
        )


def test_invalid_presence_threshold_is_rejected() -> None:
    """Presence similarity threshold must remain normalized."""
    with pytest.raises(
        ValueError,
        match="minimum_top_k_cork_score",
    ):
        CorkPresenceConfig(
            minimum_top_k_cork_score=1.5,
        )