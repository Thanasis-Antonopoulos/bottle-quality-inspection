"""Subprocess smoke tests for the installed application interfaces."""

import shutil
import subprocess
import sys


def run_command(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run a CLI command and capture its output."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_module_cli_starts() -> None:
    """The package should start through python -m."""
    result = run_command(
        [
            sys.executable,
            "-m",
            "bottle_quality_inspection",
            "--help",
        ]
    )

    assert result.returncode == 0

    assert (
        "Automated bottle quality inspection tools."
        in result.stdout
    )

    assert (
        "inspect"
        in result.stdout
    )


def test_module_inspect_help_starts() -> None:
    """The complete inspect command should construct successfully."""
    result = run_command(
        [
            sys.executable,
            "-m",
            "bottle_quality_inspection",
            "inspect",
            "--help",
        ]
    )

    assert result.returncode == 0

    assert (
        "--output-dir"
        in result.stdout
    )

    assert (
        "--config"
        in result.stdout
    )

    assert (
        "--overwrite"
        in result.stdout
    )


def test_installed_console_cli_starts() -> None:
    """The installed bottle-inspection command should be executable."""
    executable = shutil.which(
        "bottle-inspection"
    )

    assert executable is not None

    result = run_command(
        [
            executable,
            "--help",
        ]
    )

    assert result.returncode == 0

    assert (
        "Automated bottle quality inspection tools."
        in result.stdout
    )

    assert (
        "inspect"
        in result.stdout
    )