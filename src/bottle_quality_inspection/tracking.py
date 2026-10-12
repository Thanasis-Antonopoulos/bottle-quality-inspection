"""Bottle-passage tracking utilities for fixed inspection stations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StationObservation:
    """Detection state for one inspection station at one video frame."""

    station: str
    frame_index: int
    timestamp_seconds: float
    detected: bool

    def __post_init__(self) -> None:
        """Validate one station observation."""
        if not self.station.strip():
            raise ValueError(
                "station cannot be empty."
            )

        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative."
            )

        if self.timestamp_seconds < 0:
            raise ValueError(
                "timestamp_seconds cannot be negative."
            )


@dataclass(frozen=True, slots=True)
class BottlePassage:
    """One contiguous sequence where a bottle occupies a station."""

    station: str
    sequence_number: int
    observations: tuple[
        StationObservation,
        ...
    ]

    def __post_init__(self) -> None:
        """Validate a segmented bottle passage."""
        if not self.station.strip():
            raise ValueError(
                "station cannot be empty."
            )

        if self.sequence_number <= 0:
            raise ValueError(
                "sequence_number must be greater than zero."
            )

        if not self.observations:
            raise ValueError(
                "BottlePassage requires at least one observation."
            )

        previous_frame = -1

        for observation in self.observations:
            if observation.station != self.station:
                raise ValueError(
                    "All passage observations must belong "
                    "to the same station."
                )

            if not observation.detected:
                raise ValueError(
                    "BottlePassage can contain only "
                    "detected observations."
                )

            if observation.frame_index <= previous_frame:
                raise ValueError(
                    "Passage observations must be ordered "
                    "by increasing frame index."
                )

            previous_frame = (
                observation.frame_index
            )

    @property
    def start_frame(self) -> int:
        """Return the first detected frame."""
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        """Return the final detected frame."""
        return self.observations[-1].frame_index

    @property
    def detected_frame_count(self) -> int:
        """Return the number of positive observations."""
        return len(
            self.observations
        )

    @property
    def representative_observation(
        self,
    ) -> StationObservation:
        """Return the middle positive observation."""
        return self.observations[
            len(self.observations) // 2
        ]

    @property
    def representative_frame(self) -> int:
        """Return the representative frame index."""
        return (
            self.representative_observation
            .frame_index
        )

    @property
    def representative_timestamp_seconds(
        self,
    ) -> float:
        """Return the representative timestamp."""
        return (
            self.representative_observation
            .timestamp_seconds
        )


@dataclass(frozen=True, slots=True)
class PassagePair:
    """
    Candidate association between two station passages.

    The first passage is expected to occur at the upstream station and
    the second passage at the downstream station.
    """

    first: BottlePassage
    second: BottlePassage

    @property
    def frame_offset(self) -> int:
        """Return representative-frame difference."""
        return (
            self.second.representative_frame
            - self.first.representative_frame
        )

    @property
    def timestamp_offset_seconds(self) -> float:
        """Return representative-time difference."""
        return (
            self.second.representative_timestamp_seconds
            - self.first.representative_timestamp_seconds
        )


@dataclass(frozen=True, slots=True)
class PassageAssociationResult:
    """Result from chronological passage association."""

    pairs: tuple[
        PassagePair,
        ...
    ]

    unmatched_first: tuple[
        BottlePassage,
        ...
    ]

    unmatched_second: tuple[
        BottlePassage,
        ...
    ]

    @property
    def pair_count(self) -> int:
        """Return the number of associated passage pairs."""
        return len(
            self.pairs
        )


def _validate_observation_order(
    observations: list[StationObservation],
) -> None:
    """Validate station consistency and chronological ordering."""
    if not observations:
        return

    station = observations[0].station

    previous_frame = -1

    for observation in observations:
        if observation.station != station:
            raise ValueError(
                "All observations supplied for segmentation "
                "must belong to the same station."
            )

        if observation.frame_index <= previous_frame:
            raise ValueError(
                "Observations must be ordered by strictly "
                "increasing frame_index."
            )

        previous_frame = (
            observation.frame_index
        )


def _validate_passage_order(
    passages: list[BottlePassage],
    *,
    description: str,
) -> None:
    """Validate chronological ordering of passage collections."""
    previous_frame = -1

    for passage in passages:
        if (
            passage.representative_frame
            <= previous_frame
        ):
            raise ValueError(
                f"{description} passages must be ordered "
                "by representative frame."
            )

        previous_frame = (
            passage.representative_frame
        )


def segment_station_passages(
    observations: list[StationObservation],
    *,
    max_missed_frames: int = 0,
    minimum_detected_frames: int = 1,
) -> list[BottlePassage]:
    """
    Group positive station observations into bottle passages.

    Short detection gaps can be tolerated through max_missed_frames.
    Only positive observations are retained inside each passage.
    """
    if max_missed_frames < 0:
        raise ValueError(
            "max_missed_frames cannot be negative."
        )

    if minimum_detected_frames <= 0:
        raise ValueError(
            "minimum_detected_frames must be greater than zero."
        )

    if not observations:
        return []

    _validate_observation_order(
        observations
    )

    station = observations[0].station

    detected_observations = [
        observation
        for observation in observations
        if observation.detected
    ]

    if not detected_observations:
        return []

    groups: list[
        list[StationObservation]
    ] = []

    current_group = [
        detected_observations[0]
    ]

    for observation in detected_observations[1:]:
        previous = current_group[-1]

        frame_gap = (
            observation.frame_index
            - previous.frame_index
            - 1
        )

        if frame_gap <= max_missed_frames:
            current_group.append(
                observation
            )
        else:
            groups.append(
                current_group
            )

            current_group = [
                observation
            ]

    groups.append(
        current_group
    )

    passages: list[
        BottlePassage
    ] = []

    for group in groups:
        if (
            len(group)
            < minimum_detected_frames
        ):
            continue

        passages.append(
            BottlePassage(
                station=station,
                sequence_number=(
                    len(passages) + 1
                ),
                observations=tuple(
                    group
                ),
            )
        )

    return passages


def pair_passages_by_order(
    first_passages: list[BottlePassage],
    second_passages: list[BottlePassage],
) -> list[PassagePair]:
    """
    Pair station passages according to strict conveyor sequence.

    This strict helper requires equal passage counts. It is useful when
    a recording is known to contain complete passages at both stations.
    """
    if len(first_passages) != len(second_passages):
        raise ValueError(
            "Passage counts must match before order-based pairing. "
            f"Received {len(first_passages)} and "
            f"{len(second_passages)}."
        )

    return [
        PassagePair(
            first=first,
            second=second,
        )
        for first, second in zip(
            first_passages,
            second_passages,
            strict=True,
        )
    ]


def associate_passages_by_time(
    first_passages: list[BottlePassage],
    second_passages: list[BottlePassage],
    *,
    minimum_frame_offset: int = 1,
    maximum_frame_offset: int | None = None,
) -> PassageAssociationResult:
    """
    Associate upstream and downstream passages chronologically.

    Unlike strict order pairing, this function allows unmatched passages
    at either end of a recording. That is important when recording starts
    after a bottle has already passed the upstream station or stops before
    a bottle reaches the downstream station.

    Pairing remains monotonic: passage order cannot cross.
    """
    if minimum_frame_offset < 0:
        raise ValueError(
            "minimum_frame_offset cannot be negative."
        )

    if (
        maximum_frame_offset is not None
        and maximum_frame_offset
        < minimum_frame_offset
    ):
        raise ValueError(
            "maximum_frame_offset must be greater than "
            "or equal to minimum_frame_offset."
        )

    _validate_passage_order(
        first_passages,
        description="First-station",
    )

    _validate_passage_order(
        second_passages,
        description="Second-station",
    )

    pairs: list[
        PassagePair
    ] = []

    unmatched_first: list[
        BottlePassage
    ] = []

    unmatched_second: list[
        BottlePassage
    ] = []

    second_index = 0

    for first_passage in first_passages:
        minimum_second_frame = (
            first_passage.representative_frame
            + minimum_frame_offset
        )

        while (
            second_index
            < len(second_passages)
            and (
                second_passages[
                    second_index
                ].representative_frame
                < minimum_second_frame
            )
        ):
            unmatched_second.append(
                second_passages[
                    second_index
                ]
            )

            second_index += 1

        if second_index >= len(
            second_passages
        ):
            unmatched_first.append(
                first_passage
            )

            continue

        second_passage = (
            second_passages[
                second_index
            ]
        )

        frame_offset = (
            second_passage.representative_frame
            - first_passage.representative_frame
        )

        if (
            maximum_frame_offset is not None
            and frame_offset
            > maximum_frame_offset
        ):
            unmatched_first.append(
                first_passage
            )

            continue

        pairs.append(
            PassagePair(
                first=first_passage,
                second=second_passage,
            )
        )

        second_index += 1

    unmatched_second.extend(
        second_passages[
            second_index:
        ]
    )

    return PassageAssociationResult(
        pairs=tuple(
            pairs
        ),
        unmatched_first=tuple(
            unmatched_first
        ),
        unmatched_second=tuple(
            unmatched_second
        ),
    )