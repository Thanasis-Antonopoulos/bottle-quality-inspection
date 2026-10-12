"""Diagnose legacy cork-template matching across an entire video."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from bottle_quality_inspection.config import load_inspection_config
from bottle_quality_inspection.inspection_preview import save_cork_preview
from bottle_quality_inspection.inspections.cork import (
    CorkConfig,
    inspect_cork,
)
from bottle_quality_inspection.video_io import iter_video_frames


@dataclass(frozen=True, slots=True)
class CorkFrameDiagnostic:
    """Cork-template diagnostic information for one frame."""

    frame_index: int
    timestamp_seconds: float
    maximum_score: float
    status: str
    threshold_matches: int
    match_x: int
    match_y: int


def build_parser() -> argparse.ArgumentParser:
    """Create the diagnostic command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan every video frame using the legacy "
            "correct-cork template."
        )
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the source video.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/legacy-2022.toml"
        ),
        help=(
            "Inspection configuration profile. "
            "Default: configs/legacy-2022.toml."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/cork-diagnostics"
        ),
        help=(
            "Directory for candidate preview images. "
            "Default: outputs/cork-diagnostics."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "Number of highest-scoring frames "
            "to save as previews. Default: 10."
        ),
    )

    return parser


def read_template(
    template_path: Path,
):
    """Read the configured cork template."""
    resolved_path = (
        template_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Cork template does not exist: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise IsADirectoryError(
            f"Cork template is not a file: "
            f"{resolved_path}"
        )

    template = cv2.imread(
        str(resolved_path),
        cv2.IMREAD_COLOR,
    )

    if template is None:
        raise RuntimeError(
            "OpenCV could not read cork template: "
            f"{resolved_path}"
        )

    return resolved_path, template


def main() -> int:
    """Run the full-video cork diagnostic."""
    parser = build_parser()
    args = parser.parse_args()

    if args.top <= 0:
        parser.error(
            "--top must be greater than zero."
        )

    video_path = (
        args.input_video
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    inspection_config = load_inspection_config(
        args.config
    )

    cork_profile = (
        inspection_config.cork
    )

    if not cork_profile.enabled:
        parser.error(
            "Cork inspection is disabled "
            "in the selected configuration."
        )

    if cork_profile.minimum_match_score is None:
        parser.error(
            "Cork minimum_match_score "
            "is not configured."
        )

    template_path, template = read_template(
        cork_profile.template
    )

    algorithm_config = CorkConfig(
        method=cork_profile.method,
        minimum_match_score=(
            cork_profile.minimum_match_score
        ),
    )

    diagnostics: list[
        CorkFrameDiagnostic
    ] = []

    print(
        f"Profile: "
        f"{inspection_config.profile}"
    )

    print(
        f"Video: {video_path}"
    )

    print(
        f"Cork template: "
        f"{template_path}"
    )

    print(
        f"Method: "
        f"{cork_profile.method}"
    )

    print(
        "Minimum match score: "
        f"{cork_profile.minimum_match_score:.4f}"
    )

    print()
    print("Scanning video...")
    print()

    for video_frame in iter_video_frames(
        video_path
    ):
        result = inspect_cork(
            video_frame.image,
            template,
            algorithm_config,
        )

        diagnostics.append(
            CorkFrameDiagnostic(
                frame_index=(
                    video_frame.index
                ),
                timestamp_seconds=(
                    video_frame.timestamp_seconds
                ),
                maximum_score=(
                    result.maximum_score
                ),
                status=(
                    result.status.value
                ),
                threshold_matches=(
                    result.matches_at_or_above_threshold
                ),
                match_x=(
                    result.maximum_location[0]
                ),
                match_y=(
                    result.maximum_location[1]
                ),
            )
        )

    clear_frames = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.status == "clear"
    ]

    defective_frames = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.status == "defective"
    ]

    threshold = (
        cork_profile.minimum_match_score
    )

    near_threshold_frames = [
        diagnostic
        for diagnostic in diagnostics
        if abs(
            diagnostic.maximum_score
            - threshold
        ) <= 0.02
    ]

    print(
        f"Frames scanned: "
        f"{len(diagnostics)}"
    )

    print(
        f"CLEAR frames: "
        f"{len(clear_frames)}"
    )

    print(
        f"DEFECTIVE frames: "
        f"{len(defective_frames)}"
    )

    print(
        "Frames within ±0.02 "
        "of threshold: "
        f"{len(near_threshold_frames)}"
    )

    print()

    print(
        "Frames meeting the legacy "
        "0.80 cork threshold:"
    )

    if clear_frames:
        for diagnostic in clear_frames:
            print(
                f"  frame="
                f"{diagnostic.frame_index:03d} "
                f"time="
                f"{diagnostic.timestamp_seconds:.3f}s "
                f"score="
                f"{diagnostic.maximum_score:.4f} "
                f"matches="
                f"{diagnostic.threshold_matches} "
                f"location=("
                f"{diagnostic.match_x}, "
                f"{diagnostic.match_y})"
            )
    else:
        print(
            "  No frames reached the threshold."
        )

    print()
    print(
        "Frames close to the threshold "
        "(±0.02):"
    )

    if near_threshold_frames:
        for diagnostic in near_threshold_frames:
            print(
                f"  frame="
                f"{diagnostic.frame_index:03d} "
                f"time="
                f"{diagnostic.timestamp_seconds:.3f}s "
                f"score="
                f"{diagnostic.maximum_score:.4f} "
                f"status="
                f"{diagnostic.status} "
                f"location=("
                f"{diagnostic.match_x}, "
                f"{diagnostic.match_y})"
            )
    else:
        print(
            "  No frames were close "
            "to the threshold."
        )

    ranked_frames = sorted(
        diagnostics,
        key=lambda diagnostic: (
            diagnostic.maximum_score
        ),
        reverse=True,
    )

    top_frames = ranked_frames[
        : args.top
    ]

    print()
    print(
        f"Top {len(top_frames)} frames "
        "by cork-template score:"
    )

    for diagnostic in top_frames:
        print(
            f"  frame="
            f"{diagnostic.frame_index:03d} "
            f"time="
            f"{diagnostic.timestamp_seconds:.3f}s "
            f"score="
            f"{diagnostic.maximum_score:.4f} "
            f"matches="
            f"{diagnostic.threshold_matches} "
            f"location=("
            f"{diagnostic.match_x}, "
            f"{diagnostic.match_y}) "
            f"status="
            f"{diagnostic.status}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Saving previews to: "
        f"{output_dir}"
    )

    for diagnostic in top_frames:
        output_path = (
            output_dir
            / (
                "cork-frame-"
                f"{diagnostic.frame_index:03d}.jpg"
            )
        )

        save_cork_preview(
            video_path,
            output_path,
            template_path,
            config=algorithm_config,
            frame_index=(
                diagnostic.frame_index
            ),
            overwrite=True,
        )

    print()
    print(
        "Cork diagnostic scan complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )