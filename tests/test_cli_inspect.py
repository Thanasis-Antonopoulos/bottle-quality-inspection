"""Tests for the complete inspection CLI command."""

import sys
from pathlib import Path

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
    assert args.input_video == Path("video.mp4")

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
        args.video_filename
        == "annotated-inspection.mp4"
    )

    assert args.codec == "mp4v"
    assert args.overwrite is False


def test_main_dispatches_inspect_command(
    monkeypatch,
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