"""End-to-end inspection pipeline for tracked physical bottles."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bottle_quality_inspection.bottle_result import (
    BottleAggregationConfig,
    BottleInspectionResult,
    ForeignObjectFrameEvidence,
    GlassFrameEvidence,
    LiquidLevelFrameEvidence,
    aggregate_cork_presence,
    aggregate_foreign_object,
    aggregate_glass,
    aggregate_liquid_level,
    build_bottle_inspection_result,
)
from bottle_quality_inspection.config import InspectionConfig
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    inspect_cork,
)
from bottle_quality_inspection.inspections.cork_presence import (
    CorkPresenceConfig,
    CorkPresenceFrameScore,
    evaluate_cork_presence,
)
from bottle_quality_inspection.inspections.foreign_object import (
    inspect_foreign_objects,
)
from bottle_quality_inspection.inspections.glass import (
    GlassDecisionConfig,
    GlassFeatureConfig,
    inspect_glass,
)
from bottle_quality_inspection.inspections.liquid_level import (
    LiquidLevelConfig,
    inspect_liquid_level,
)
from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.tracking import (
    BottlePassage,
    PassageAssociationResult,
    PassagePair,
    StationObservation,
    associate_passages_by_time,
    segment_station_passages,
)
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class TrackedBottleInspection:
    """Inspection result and tracking metadata for one physical bottle."""

    bottle_id: str

    foreign_start_frame: int
    foreign_end_frame: int

    downstream_start_frame: int
    downstream_end_frame: int

    travel_frame_offset: int
    travel_time_seconds: float

    result: BottleInspectionResult


@dataclass(frozen=True, slots=True)
class BottleInspectionRun:
    """Complete result from inspecting one video."""

    video_path: Path
    profile_name: str

    bottles: tuple[
        TrackedBottleInspection,
        ...
    ]

    unmatched_foreign_passages: tuple[
        BottlePassage,
        ...
    ]

    unmatched_downstream_passages: tuple[
        BottlePassage,
        ...
    ]

    @property
    def bottle_count(self) -> int:
        """Return the number of fully associated bottles."""
        return len(
            self.bottles
        )


def _create_liquid_config(
    profile: InspectionConfig,
) -> LiquidLevelConfig:
    """Convert profile values into liquid-level algorithm settings."""
    liquid = profile.liquid_level

    return LiquidLevelConfig(
        neck_region=(
            liquid.neck_region.to_roi()
        ),
        canny_threshold1=(
            liquid.canny_threshold1
        ),
        canny_threshold2=(
            liquid.canny_threshold2
        ),
        hough_rho=(
            liquid.hough_rho
        ),
        hough_theta_degrees=(
            liquid.hough_theta_degrees
        ),
        hough_threshold=(
            liquid.hough_threshold
        ),
        minimum_line_length=(
            liquid.minimum_line_length
        ),
        maximum_line_gap=(
            liquid.maximum_line_gap
        ),
        acceptable_minimum=(
            liquid.acceptable_minimum
        ),
        acceptable_maximum=(
            liquid.acceptable_maximum
        ),
    )


def _read_cork_template(
    template_path: Path,
) -> np.ndarray:
    """Read the configured normal-cork template."""
    resolved = (
        template_path
        .expanduser()
        .resolve()
    )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Cork template does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise IsADirectoryError(
            f"Cork template is not a file: {resolved}"
        )

    template = cv2.imread(
        str(resolved),
        cv2.IMREAD_COLOR,
    )

    if template is None:
        raise RuntimeError(
            f"OpenCV could not read cork template: {resolved}"
        )

    return template


def _detect_passage_associations(
    video_path: Path,
    profile: InspectionConfig,
    *,
    max_missed_frames: int,
    minimum_detected_frames: int,
    minimum_frame_offset: int,
    maximum_frame_offset: int,
) -> PassageAssociationResult:
    """Detect and associate bottle passages between inspection stations."""
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

    foreign_config = (
        profile.foreign_object
        .to_algorithm_config()
    )

    liquid_config = _create_liquid_config(
        profile
    )

    foreign_observations: list[
        StationObservation
    ] = []

    downstream_observations: list[
        StationObservation
    ] = []

    for video_frame in iter_video_frames(
        video_path
    ):
        foreign_crop = (
            foreign_region.extract(
                video_frame.image
            )
        )

        downstream_crop = (
            downstream_region.extract(
                video_frame.image
            )
        )

        foreign_result = (
            inspect_foreign_objects(
                foreign_crop,
                foreign_config,
            )
        )

        liquid_result = (
            inspect_liquid_level(
                downstream_crop,
                liquid_config,
            )
        )

        foreign_observations.append(
            StationObservation(
                station="foreign-object",
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                detected=(
                    foreign_result.bottle_detected
                ),
            )
        )

        downstream_observations.append(
            StationObservation(
                station="liquid-glass",
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                detected=(
                    liquid_result.level_detected
                ),
            )
        )

    foreign_passages = segment_station_passages(
        foreign_observations,
        max_missed_frames=(
            max_missed_frames
        ),
        minimum_detected_frames=(
            minimum_detected_frames
        ),
    )

    downstream_passages = segment_station_passages(
        downstream_observations,
        max_missed_frames=(
            max_missed_frames
        ),
        minimum_detected_frames=(
            minimum_detected_frames
        ),
    )

    return associate_passages_by_time(
        foreign_passages,
        downstream_passages,
        minimum_frame_offset=(
            minimum_frame_offset
        ),
        maximum_frame_offset=(
            maximum_frame_offset
        ),
    )


def _required_frame_indices(
    association: PassageAssociationResult,
) -> set[int]:
    """Return all source frames required for associated bottles."""
    required: set[int] = set()

    for pair in association.pairs:
        for observation in (
            pair.first.observations
        ):
            required.add(
                observation.frame_index
            )

        for observation in (
            pair.second.observations
        ):
            required.add(
                observation.frame_index
            )

    return required


def _collect_frames(
    video_path: Path,
    frame_indices: set[int],
) -> dict[int, np.ndarray]:
    """Decode only frames required by associated passages."""
    frames: dict[
        int,
        np.ndarray
    ] = {}

    for video_frame in iter_video_frames(
        video_path
    ):
        if (
            video_frame.index
            not in frame_indices
        ):
            continue

        frames[
            video_frame.index
        ] = (
            video_frame.image.copy()
        )

    missing = (
        frame_indices
        - set(
            frames
        )
    )

    if missing:
        missing_text = ", ".join(
            str(frame)
            for frame in sorted(
                missing
            )
        )

        raise RuntimeError(
            "Could not decode required bottle frames: "
            f"{missing_text}"
        )

    return frames


def _inspect_foreign_passage(
    pair: PassagePair,
    frames: dict[int, np.ndarray],
    profile: InspectionConfig,
) -> list[ForeignObjectFrameEvidence]:
    """Collect foreign-object evidence across one upstream passage."""
    region = (
        profile.foreign_object
        .roi
        .to_roi()
    )

    config = (
        profile.foreign_object
        .to_algorithm_config()
    )

    evidence: list[
        ForeignObjectFrameEvidence
    ] = []

    for observation in (
        pair.first.observations
    ):
        frame_index = (
            observation.frame_index
        )

        result = inspect_foreign_objects(
            region.extract(
                frames[
                    frame_index
                ]
            ),
            config,
        )

        evidence.append(
            ForeignObjectFrameEvidence(
                frame_index=(
                    frame_index
                ),
                status=(
                    result.status
                ),
                decision_black_pixels=(
                    result.decision_black_pixel_count
                ),
            )
        )

    return evidence


def _inspect_downstream_passage(
    pair: PassagePair,
    frames: dict[int, np.ndarray],
    profile: InspectionConfig,
    cork_template: np.ndarray,
) -> tuple[
    list[LiquidLevelFrameEvidence],
    list[GlassFrameEvidence],
    list[CorkPresenceFrameScore],
]:
    """Collect liquid, glass and cork evidence for one downstream passage."""
    region = (
        profile.liquid_level
        .roi
        .to_roi()
    )

    liquid_config = (
        _create_liquid_config(
            profile
        )
    )

    glass_feature_config = (
        GlassFeatureConfig()
    )

    glass_decision_config = (
        GlassDecisionConfig()
    )

    cork_config = CorkConfig(
        method=(
            profile.cork.method
        ),
        minimum_match_score=(
            profile.cork.minimum_match_score
            if (
                profile.cork.minimum_match_score
                is not None
            )
            else 0.80
        ),
    )

    cork_region = RegionOfInterest(
        name="cork-search",
        x=0,
        y=40,
        width=200,
        height=200,
    )

    body_region = RegionOfInterest(
        name="body-control",
        x=0,
        y=260,
        width=200,
        height=300,
    )

    liquid_evidence: list[
        LiquidLevelFrameEvidence
    ] = []

    glass_evidence: list[
        GlassFrameEvidence
    ] = []

    cork_evidence: list[
        CorkPresenceFrameScore
    ] = []

    for observation in (
        pair.second.observations
    ):
        frame_index = (
            observation.frame_index
        )

        bottle_crop = (
            region.extract(
                frames[
                    frame_index
                ]
            )
        )

        liquid_result = (
            inspect_liquid_level(
                bottle_crop,
                liquid_config,
            )
        )

        glass_result = (
            inspect_glass(
                bottle_crop,
                glass_feature_config,
                glass_decision_config,
            )
        )

        cork_result = inspect_cork(
            cork_region.extract(
                bottle_crop
            ),
            cork_template,
            cork_config,
        )

        body_result = inspect_cork(
            body_region.extract(
                bottle_crop
            ),
            cork_template,
            cork_config,
        )

        liquid_evidence.append(
            LiquidLevelFrameEvidence(
                frame_index=(
                    frame_index
                ),
                status=(
                    liquid_result.status
                ),
                level=(
                    liquid_result.absolute_level
                ),
            )
        )

        glass_evidence.append(
            GlassFrameEvidence(
                frame_index=(
                    frame_index
                ),
                status=(
                    glass_result.status
                ),
                triggered_signal_count=(
                    glass_result.triggered_signal_count
                ),
            )
        )

        cork_evidence.append(
            CorkPresenceFrameScore(
                frame_index=(
                    frame_index
                ),
                cork_score=(
                    cork_result.maximum_score
                ),
                body_control_score=(
                    body_result.maximum_score
                ),
            )
        )

    return (
        liquid_evidence,
        glass_evidence,
        cork_evidence,
    )


def _inspect_pair(
    bottle_id: str,
    pair: PassagePair,
    frames: dict[int, np.ndarray],
    profile: InspectionConfig,
    cork_template: np.ndarray,
) -> TrackedBottleInspection:
    """Create the complete result for one associated physical bottle."""
    aggregation_config = (
        BottleAggregationConfig(
            liquid_acceptable_minimum=(
                profile.liquid_level
                .acceptable_minimum
            ),
            liquid_acceptable_maximum=(
                profile.liquid_level
                .acceptable_maximum
            ),
        )
    )

    foreign_evidence = (
        _inspect_foreign_passage(
            pair,
            frames,
            profile,
        )
    )

    (
        liquid_evidence,
        glass_evidence,
        cork_frame_scores,
    ) = _inspect_downstream_passage(
        pair,
        frames,
        profile,
        cork_template,
    )

    foreign_decision = (
        aggregate_foreign_object(
            foreign_evidence,
            aggregation_config,
        )
    )

    liquid_decision = (
        aggregate_liquid_level(
            liquid_evidence,
            aggregation_config,
        )
    )

    glass_decision = (
        aggregate_glass(
            glass_evidence,
            aggregation_config,
        )
    )

    cork_presence = (
        evaluate_cork_presence(
            cork_frame_scores,
            CorkPresenceConfig(),
        )
    )

    cork_decision = (
        aggregate_cork_presence(
            cork_presence
        )
    )

    result = (
        build_bottle_inspection_result(
            bottle_id,
            foreign_object=(
                foreign_decision
            ),
            liquid_level=(
                liquid_decision
            ),
            glass=(
                glass_decision
            ),
            cork=(
                cork_decision
            ),
        )
    )

    return TrackedBottleInspection(
        bottle_id=bottle_id,
        foreign_start_frame=(
            pair.first.start_frame
        ),
        foreign_end_frame=(
            pair.first.end_frame
        ),
        downstream_start_frame=(
            pair.second.start_frame
        ),
        downstream_end_frame=(
            pair.second.end_frame
        ),
        travel_frame_offset=(
            pair.frame_offset
        ),
        travel_time_seconds=(
            pair.timestamp_offset_seconds
        ),
        result=result,
    )


def inspect_tracked_bottles(
    video_path: str | Path,
    profile: InspectionConfig,
    *,
    max_missed_frames: int = 1,
    minimum_detected_frames: int = 2,
    minimum_frame_offset: int = 1,
    maximum_frame_offset: int = 30,
) -> BottleInspectionRun:
    """
    Inspect all physical bottles that can be associated across stations.

    Bottles that appear only at one station are retained as unmatched
    passages rather than being given incomplete final quality results.
    """
    source = (
        Path(
            video_path
        )
        .expanduser()
        .resolve()
    )

    association = (
        _detect_passage_associations(
            source,
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
    )

    required_frames = (
        _required_frame_indices(
            association
        )
    )

    frames = _collect_frames(
        source,
        required_frames,
    )

    cork_template = (
        _read_cork_template(
            profile.cork.template
        )
    )

    bottles: list[
        TrackedBottleInspection
    ] = []

    for bottle_number, pair in enumerate(
        association.pairs,
        start=1,
    ):
        bottle_id = (
            f"bottle-{bottle_number:03d}"
        )

        bottles.append(
            _inspect_pair(
                bottle_id,
                pair,
                frames,
                profile,
                cork_template,
            )
        )

    return BottleInspectionRun(
        video_path=source,
        profile_name=(
            profile.profile
        ),
        bottles=tuple(
            bottles
        ),
        unmatched_foreign_passages=(
            association.unmatched_first
        ),
        unmatched_downstream_passages=(
            association.unmatched_second
        ),
    )