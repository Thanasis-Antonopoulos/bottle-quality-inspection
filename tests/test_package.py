"""Basic tests for the Bottle Quality Inspection package."""

from bottle_quality_inspection import __version__


def test_package_version() -> None:
    """Verify that the package exposes its version."""
    assert __version__ == "0.1.0"