"""Tests for the complete inspection CLI command."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bottle_quality_inspection import __main__ as cli


def test_parser_accepts_inspect_command() -> None:
    """The main CLI should expose the complete inspect command."""
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "inspect",
            "video.mp4",
        ]
    )

    assert args.command == "inspect"
    assert args.input_video == Path(
        "video.mp4"
    )

    assert args.config == Path(
        "configs/legacy-2022.toml"
    )

    assert args.output_dir == Path(
        "outputs/inspection"
    )

    assert (
        args.stem
        == "inspection-results"
    )

    assert (
        args.metrics_filename
        == "inspection-metrics.json"
    )

    assert (
        args.video_filename
        == "annotated-inspection.mp4"
    )

    assert args.codec == "mp4v"
    assert args.overwrite is False


def test_main_dispatches_inspect_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package entry point should dispatch to complete inspection."""
    received = {}

    def fake_print_complete_inspection(
        args,
    ) -> None:
        received[
            "input_video"
        ] = args.input_video

        received[
            "output_dir"
        ] = args.output_dir

    monkeypatch.setattr(
        cli,
        "print_complete_inspection",
        fake_print_complete_inspection,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bottle-quality-inspection",
            "inspect",
            "video.mp4",
            "--output-dir",
            "outputs/test",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0

    assert (
        received["input_video"]
        == Path("video.mp4")
    )

    assert (
        received["output_dir"]
        == Path("outputs/test")
    )


def test_complete_inspection_uses_execution_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inspect CLI should delegate the production workflow."""
    profile = SimpleNamespace(
        profile="test-profile"
    )

    received: dict[
        str,
        object,
    ] = {}

    quality = SimpleNamespace(
        associated_bottles=0,
        good=0,
        review=0,
        defective=0,
        unmatched_foreign_passages=0,
        unmatched_downstream_passages=0,
        foreign_object=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        liquid_level=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        glass=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        cork=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
    )

    timing = SimpleNamespace(
        source_frame_count=90,
        source_fps=30.0,
        source_duration_seconds=3.0,
        inspection_seconds=1.2,
        report_seconds=0.1,
        visual_seconds=0.7,
        total_seconds=2.0,
        effective_processing_fps=45.0,
        realtime_factor=1.5,
    )

    run = SimpleNamespace(
        bottles=(),
    )

    reports = SimpleNamespace(
        csv_path=(
            tmp_path
            / "inspection-results.csv"
        ),
        json_path=(
            tmp_path
            / "inspection-results.json"
        ),
    )

    visuals = SimpleNamespace(
        video_result=SimpleNamespace(
            output_path=(
                tmp_path
                / "annotated-inspection.mp4"
            )
        ),
        review_images=(),
        rejected_images=(),
    )

    execution_result = (
        SimpleNamespace(
            run=run,
            reports=reports,
            visuals=visuals,
            quality_metrics=quality,
            timing_metrics=timing,
            metrics_path=SimpleNamespace(
                json_path=(
                    tmp_path
                    / "inspection-metrics.json"
                )
            ),
            review_dir=(
                tmp_path
                / "review"
            ),
            rejected_dir=(
                tmp_path
                / "rejected"
            ),
        )
    )

    monkeypatch.setattr(
        cli,
        "load_inspection_config",
        lambda _path: profile,
    )

    def fake_execute_inspection(
        video_path,
        received_profile,
        output_dir,
        **kwargs,
    ):
        received[
            "video_path"
        ] = video_path

        received[
            "profile"
        ] = received_profile

        received[
            "output_dir"
        ] = output_dir

        received[
            "kwargs"
        ] = kwargs

        return execution_result

    monkeypatch.setattr(
        cli,
        "execute_inspection",
        fake_execute_inspection,
    )

    args = SimpleNamespace(
        input_video=Path(
            "video.mp4"
        ),
        config=Path(
            "config.toml"
        ),
        output_dir=tmp_path,
        stem="inspection-results",
        metrics_filename=(
            "inspection-metrics.json"
        ),
        video_filename=(
            "annotated-inspection.mp4"
        ),
        codec="mp4v",
        overwrite=True,
    )

    cli.print_complete_inspection(
        args
    )

    assert (
        received["video_path"]
        == Path("video.mp4")
    )

    assert (
        received["profile"]
        is profile
    )

    assert (
        received["output_dir"]
        == tmp_path.resolve()
    )

    kwargs = received["kwargs"]

    assert (
        kwargs["metrics_filename"]
        == "inspection-metrics.json"
    )

    assert kwargs["overwrite"] is True


def test_complete_inspection_prints_performance_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The inspect CLI should display runtime evaluation results."""
    profile = SimpleNamespace(
        profile="test-profile"
    )

    quality = SimpleNamespace(
        associated_bottles=0,
        good=0,
        review=0,
        defective=0,
        unmatched_foreign_passages=0,
        unmatched_downstream_passages=0,
        foreign_object=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        liquid_level=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        glass=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
        cork=SimpleNamespace(
            clear=0,
            review=0,
            defective=0,
        ),
    )

    timing = SimpleNamespace(
        source_frame_count=90,
        source_fps=30.0,
        source_duration_seconds=3.0,
        inspection_seconds=1.2,
        report_seconds=0.1,
        visual_seconds=0.7,
        total_seconds=2.0,
        effective_processing_fps=45.0,
        realtime_factor=1.5,
    )

    result = SimpleNamespace(
        run=SimpleNamespace(
            bottles=()
        ),
        reports=SimpleNamespace(
            csv_path=(
                tmp_path
                / "inspection-results.csv"
            ),
            json_path=(
                tmp_path
                / "inspection-results.json"
            ),
        ),
        visuals=SimpleNamespace(
            video_result=SimpleNamespace(
                output_path=(
                    tmp_path
                    / "annotated-inspection.mp4"
                )
            ),
            review_images=(),
            rejected_images=(),
        ),
        quality_metrics=quality,
        timing_metrics=timing,
        metrics_path=SimpleNamespace(
            json_path=(
                tmp_path
                / "inspection-metrics.json"
            )
        ),
        review_dir=(
            tmp_path
            / "review"
        ),
        rejected_dir=(
            tmp_path
            / "rejected"
        ),
    )

    monkeypatch.setattr(
        cli,
        "load_inspection_config",
        lambda _path: profile,
    )

    monkeypatch.setattr(
        cli,
        "execute_inspection",
        lambda *_args, **_kwargs: result,
    )

    args = SimpleNamespace(
        input_video=Path(
            "video.mp4"
        ),
        config=Path(
            "config.toml"
        ),
        output_dir=tmp_path,
        stem="inspection-results",
        metrics_filename=(
            "inspection-metrics.json"
        ),
        video_filename=(
            "annotated-inspection.mp4"
        ),
        codec="mp4v",
        overwrite=True,
    )

    cli.print_complete_inspection(
        args
    )

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "PERFORMANCE"
        in output
    )

    assert (
        "Effective processing FPS: 45.00"
        in output
    )

    assert (
        "Real-time factor: 1.50x"
        in output
    )

    assert (
        "inspection-metrics.json"
        in output
    )