"""Tests for project packaging and command-line metadata."""

import tomllib
from pathlib import Path

from bottle_quality_inspection import __version__

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def load_pyproject() -> dict:
    """Load the repository pyproject configuration."""
    path = (
        PROJECT_ROOT
        / "pyproject.toml"
    )

    with path.open(
        "rb"
    ) as pyproject_file:
        return tomllib.load(
            pyproject_file
        )


def test_package_version_matches_pyproject() -> None:
    """Runtime and packaging versions should remain synchronized."""
    config = load_pyproject()

    assert (
        config["project"]["version"]
        == __version__
    )


def test_console_script_is_registered() -> None:
    """The package should expose the product CLI after installation."""
    config = load_pyproject()

    scripts = (
        config["project"]["scripts"]
    )

    assert (
        scripts["bottle-inspection"]
        == "bottle_quality_inspection.__main__:main"
    )