"""Tests for inspection configuration loading."""

from pathlib import Path

import pytest

from bottle_quality_inspection.config import (
    RectangleConfig,
    load_inspection_config,
)
from bottle_quality_inspection.exceptions import ConfigurationError


def test_loads_legacy_2022_profile() -> None:
    """The repository baseline configuration should load successfully."""
    config = load_inspection_config(
        Path("configs/legacy-2022.toml")
    )

    assert config.profile == "legacy-2022"
    assert config.policy.require_all_checks is True

    assert config.bottle.enabled is True
    assert config.bottle.minimum_match_score is None

    assert config.cork.enabled is True
    assert config.cork.minimum_match_score == pytest.approx(0.80)

    assert config.foreign_object.roi.x == 600
    assert config.foreign_object.roi.y == 300
    assert config.foreign_object.roi.width == 200
    assert config.foreign_object.roi.height == 700

    assert config.foreign_object.decision_region.x == 0
    assert config.foreign_object.decision_region.y == 270
    assert config.foreign_object.decision_region.width == 180
    assert config.foreign_object.decision_region.height == 280

    assert config.foreign_object.defect_pixel_threshold == 900

    assert config.liquid_level.roi.x == 910
    assert config.liquid_level.roi.y == 300

    assert config.liquid_level.canny_threshold1 == 252
    assert config.liquid_level.canny_threshold2 == 119

    assert config.liquid_level.acceptable_minimum == pytest.approx(225.0)
    assert config.liquid_level.acceptable_maximum == pytest.approx(247.0)

    assert config.glass.enabled is True
    assert config.glass.decision_defined is False


def test_foreign_object_config_converts_to_algorithm_config() -> None:
    """Application settings should create the existing algorithm config."""
    config = load_inspection_config(
        Path("configs/legacy-2022.toml")
    )

    algorithm = config.foreign_object.to_algorithm_config()

    assert algorithm.gaussian_kernel_size == 5
    assert algorithm.closing_kernel_size == 13
    assert algorithm.canny_low_threshold == 17
    assert algorithm.canny_high_threshold == 60
    assert algorithm.defect_pixel_threshold == 900

    assert algorithm.analysis_region is not None
    assert algorithm.analysis_region.x == 0
    assert algorithm.analysis_region.y == 270
    assert algorithm.analysis_region.width == 180
    assert algorithm.analysis_region.height == 280


def test_missing_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing profile should produce a clear filesystem error."""
    missing = tmp_path / "missing.toml"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        load_inspection_config(missing)


def test_invalid_rectangle_is_rejected() -> None:
    """Invalid ROI geometry should fail during configuration validation."""
    with pytest.raises(
        ConfigurationError,
        match="dimensions must be greater than zero",
    ):
        RectangleConfig(
            name="invalid",
            x=0,
            y=0,
            width=0,
            height=100,
        )