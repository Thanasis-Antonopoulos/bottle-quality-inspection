"""Tests for fixed-station bottle passage tracking."""

import pytest

from bottle_quality_inspection.tracking import (
    BottlePassage,
    StationObservation,
    associate_passages_by_time,
    pair_passages_by_order,
    segment_station_passages,
)


def observation(
    frame: int,
    detected: bool,
    *,
    station: str = "station-a",
) -> StationObservation:
    """Create a deterministic test observation."""
    return StationObservation(
        station=station,
        frame_index=frame,
        timestamp_seconds=(
            frame / 30.0
        ),
        detected=detected,
    )


def passage(
    frame: int,
    *,
    station: str,
    sequence_number: int,
) -> BottlePassage:
    """Create a one-observation passage for association tests."""
    return BottlePassage(
        station=station,
        sequence_number=sequence_number,
        observations=(
            observation(
                frame,
                True,
                station=station,
            ),
        ),
    )


def test_contiguous_detections_form_one_passage() -> None:
    """Consecutive detections should belong to one bottle passage."""
    observations = [
        observation(0, False),
        observation(1, True),
        observation(2, True),
        observation(3, True),
        observation(4, False),
    ]

    passages = segment_station_passages(
        observations
    )

    assert len(passages) == 1

    result = passages[0]

    assert result.start_frame == 1
    assert result.end_frame == 3
    assert result.detected_frame_count == 3
    assert result.representative_frame == 2


def test_separated_detection_groups_form_multiple_passages() -> None:
    """Separated groups should produce separate bottle passages."""
    observations = [
        observation(0, True),
        observation(1, True),
        observation(2, False),
        observation(3, False),
        observation(4, True),
        observation(5, True),
    ]

    passages = segment_station_passages(
        observations
    )

    assert len(passages) == 2

    assert passages[0].start_frame == 0
    assert passages[0].end_frame == 1

    assert passages[1].start_frame == 4
    assert passages[1].end_frame == 5


def test_short_detection_gap_can_be_tolerated() -> None:
    """One missed frame may be retained within the same passage."""
    observations = [
        observation(0, True),
        observation(1, True),
        observation(2, False),
        observation(3, True),
        observation(4, True),
    ]

    passages = segment_station_passages(
        observations,
        max_missed_frames=1,
    )

    assert len(passages) == 1

    assert passages[0].start_frame == 0
    assert passages[0].end_frame == 4

    assert (
        passages[0].detected_frame_count
        == 4
    )


def test_minimum_detection_count_removes_noise() -> None:
    """Single-frame detections can be filtered as station noise."""
    observations = [
        observation(0, False),
        observation(1, True),
        observation(2, False),
        observation(3, False),
        observation(4, True),
        observation(5, True),
        observation(6, True),
    ]

    passages = segment_station_passages(
        observations,
        minimum_detected_frames=2,
    )

    assert len(passages) == 1

    assert passages[0].start_frame == 4
    assert passages[0].end_frame == 6


def test_no_detections_returns_no_passages() -> None:
    """A station with no detections should return no bottle passages."""
    observations = [
        observation(0, False),
        observation(1, False),
        observation(2, False),
    ]

    assert (
        segment_station_passages(
            observations
        )
        == []
    )


def test_mixed_stations_are_rejected() -> None:
    """One segmentation call must represent one physical station."""
    observations = [
        observation(
            0,
            True,
            station="foreign",
        ),
        observation(
            1,
            True,
            station="liquid",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="same station",
    ):
        segment_station_passages(
            observations
        )


def test_order_pairing_reports_station_offset() -> None:
    """Strict order pairing should expose travel-time offset."""
    first = BottlePassage(
        station="foreign",
        sequence_number=1,
        observations=(
            observation(
                10,
                True,
                station="foreign",
            ),
            observation(
                11,
                True,
                station="foreign",
            ),
            observation(
                12,
                True,
                station="foreign",
            ),
        ),
    )

    second = BottlePassage(
        station="liquid",
        sequence_number=1,
        observations=(
            observation(
                20,
                True,
                station="liquid",
            ),
            observation(
                21,
                True,
                station="liquid",
            ),
            observation(
                22,
                True,
                station="liquid",
            ),
        ),
    )

    pairs = pair_passages_by_order(
        [first],
        [second],
    )

    assert len(pairs) == 1

    assert pairs[0].frame_offset == 10

    assert (
        pairs[0].timestamp_offset_seconds
        == pytest.approx(
            10 / 30.0
        )
    )


def test_order_pairing_requires_equal_counts() -> None:
    """Strict pairing should reject unequal station passage counts."""
    first = passage(
        1,
        station="foreign",
        sequence_number=1,
    )

    with pytest.raises(
        ValueError,
        match="Passage counts must match",
    ):
        pair_passages_by_order(
            [first],
            [],
        )


def test_time_association_skips_leading_downstream_passage() -> None:
    """A bottle already downstream at recording start stays unmatched."""
    first_passages = [
        passage(
            33,
            station="foreign",
            sequence_number=1,
        ),
        passage(
            48,
            station="foreign",
            sequence_number=2,
        ),
        passage(
            77,
            station="foreign",
            sequence_number=3,
        ),
    ]

    second_passages = [
        passage(
            2,
            station="liquid",
            sequence_number=1,
        ),
        passage(
            44,
            station="liquid",
            sequence_number=2,
        ),
        passage(
            59,
            station="liquid",
            sequence_number=3,
        ),
        passage(
            85,
            station="liquid",
            sequence_number=4,
        ),
    ]

    result = associate_passages_by_time(
        first_passages,
        second_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=30,
    )

    assert result.pair_count == 3

    assert [
        pair.frame_offset
        for pair in result.pairs
    ] == [
        11,
        11,
        8,
    ]

    assert (
        result.unmatched_first
        == ()
    )

    assert len(
        result.unmatched_second
    ) == 1

    assert (
        result.unmatched_second[
            0
        ].representative_frame
        == 2
    )


def test_time_association_can_leave_upstream_passage_unmatched() -> None:
    """An upstream passage should remain unmatched if travel time is too large."""
    first_passages = [
        passage(
            10,
            station="foreign",
            sequence_number=1,
        ),
        passage(
            90,
            station="foreign",
            sequence_number=2,
        ),
    ]

    second_passages = [
        passage(
            100,
            station="liquid",
            sequence_number=1,
        ),
    ]

    result = associate_passages_by_time(
        first_passages,
        second_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=20,
    )

    assert result.pair_count == 1

    assert (
        result.pairs[
            0
        ].first.representative_frame
        == 90
    )

    assert (
        result.pairs[
            0
        ].second.representative_frame
        == 100
    )

    assert len(
        result.unmatched_first
    ) == 1

    assert (
        result.unmatched_first[
            0
        ].representative_frame
        == 10
    )


def test_time_association_reports_trailing_downstream_passage() -> None:
    """Unused downstream passages should be retained explicitly."""
    first_passages = [
        passage(
            10,
            station="foreign",
            sequence_number=1,
        ),
    ]

    second_passages = [
        passage(
            20,
            station="liquid",
            sequence_number=1,
        ),
        passage(
            50,
            station="liquid",
            sequence_number=2,
        ),
    ]

    result = associate_passages_by_time(
        first_passages,
        second_passages,
        minimum_frame_offset=1,
        maximum_frame_offset=20,
    )

    assert result.pair_count == 1

    assert len(
        result.unmatched_second
    ) == 1

    assert (
        result.unmatched_second[
            0
        ].representative_frame
        == 50
    )


def test_invalid_time_association_offset_range_is_rejected() -> None:
    """Maximum travel time cannot be below minimum travel time."""
    with pytest.raises(
        ValueError,
        match="maximum_frame_offset",
    ):
        associate_passages_by_time(
            [],
            [],
            minimum_frame_offset=10,
            maximum_frame_offset=5,
        )