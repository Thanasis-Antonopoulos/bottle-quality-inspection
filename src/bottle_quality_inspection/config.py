"""Configuration models and loaders for bottle inspection profiles."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bottle_quality_inspection.exceptions import ConfigurationError
from bottle_quality_inspection.inspections.foreign_object import (
    ForeignObjectConfig,
)
from bottle_quality_inspection.roi import RegionOfInterest


@dataclass(frozen=True, slots=True)
class RectangleConfig:
    """Serializable rectangular image region."""

    name: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate rectangle geometry."""
        if not self.name.strip():
            raise ConfigurationError("Rectangle name cannot be empty.")

        if self.x < 0 or self.y < 0:
            raise ConfigurationError(
                f"Rectangle '{self.name}' coordinates cannot be negative."
            )

        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"Rectangle '{self.name}' dimensions must be greater than zero."
            )

    def to_roi(self) -> RegionOfInterest:
        """Convert configuration coordinates into an ROI object."""
        return RegionOfInterest(
            name=self.name,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class TemplateInspectionConfig:
    """Configuration for template-based inspection."""

    enabled: bool
    template: Path
    method: str
    minimum_match_score: float | None = None

    def __post_init__(self) -> None:
        """Validate template inspection settings."""
        if not self.method.strip():
            raise ConfigurationError(
                "Template matching method cannot be empty."
            )

        if (
            self.minimum_match_score is not None
            and not 0.0 <= self.minimum_match_score <= 1.0
        ):
            raise ConfigurationError(
                "minimum_match_score must be between 0.0 and 1.0."
            )


@dataclass(frozen=True, slots=True)
class ForeignObjectInspectionConfig:
    """Complete setup for foreign-object inspection."""

    enabled: bool
    roi: RectangleConfig
    decision_region: RectangleConfig

    gaussian_kernel_size: int
    closing_kernel_size: int
    content_blur_kernel_size: int

    canny_low_threshold: int
    canny_high_threshold: int

    content_threshold: int
    reflection_threshold: int

    component_count: int
    minimum_contour_area: float
    defect_pixel_threshold: int

    def to_algorithm_config(self) -> ForeignObjectConfig:
        """Convert application settings into the algorithm configuration."""
        return ForeignObjectConfig(
            gaussian_kernel_size=self.gaussian_kernel_size,
            closing_kernel_size=self.closing_kernel_size,
            content_blur_kernel_size=self.content_blur_kernel_size,
            canny_low_threshold=self.canny_low_threshold,
            canny_high_threshold=self.canny_high_threshold,
            content_threshold=self.content_threshold,
            reflection_threshold=self.reflection_threshold,
            component_count=self.component_count,
            minimum_contour_area=self.minimum_contour_area,
            defect_pixel_threshold=self.defect_pixel_threshold,
            analysis_region=self.decision_region.to_roi(),
        )


@dataclass(frozen=True, slots=True)
class LiquidLevelInspectionConfig:
    """Configuration for liquid-level inspection."""

    enabled: bool
    roi: RectangleConfig
    neck_region: RectangleConfig

    canny_threshold1: int
    canny_threshold2: int

    hough_rho: float
    hough_theta_degrees: float
    hough_threshold: int
    minimum_line_length: int
    maximum_line_gap: int

    acceptable_minimum: float
    acceptable_maximum: float

    def __post_init__(self) -> None:
        """Validate liquid-level parameters."""
        for name, value in {
            "canny_threshold1": self.canny_threshold1,
            "canny_threshold2": self.canny_threshold2,
        }.items():
            if not 0 <= value <= 255:
                raise ConfigurationError(
                    f"{name} must be between 0 and 255."
                )

        if self.hough_rho <= 0:
            raise ConfigurationError(
                "hough_rho must be greater than zero."
            )

        if self.hough_theta_degrees <= 0:
            raise ConfigurationError(
                "hough_theta_degrees must be greater than zero."
            )

        if self.hough_threshold <= 0:
            raise ConfigurationError(
                "hough_threshold must be greater than zero."
            )

        if self.minimum_line_length <= 0:
            raise ConfigurationError(
                "minimum_line_length must be greater than zero."
            )

        if self.maximum_line_gap < 0:
            raise ConfigurationError(
                "maximum_line_gap cannot be negative."
            )

        if self.acceptable_minimum >= self.acceptable_maximum:
            raise ConfigurationError(
                "acceptable_minimum must be lower than acceptable_maximum."
            )


@dataclass(frozen=True, slots=True)
class GlassInspectionConfig:
    """Configuration for glass-integrity inspection."""

    enabled: bool
    defect_reference: Path | None
    decision_method: str | None

    @property
    def decision_defined(self) -> bool:
        """Return whether the profile defines a glass decision method."""
        return bool(
            self.decision_method
            and self.decision_method.strip()
        )


@dataclass(frozen=True, slots=True)
class InspectionPolicyConfig:
    """Rules controlling the overall bottle decision."""

    require_all_checks: bool


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Complete configuration for one inspection profile."""

    profile: str
    policy: InspectionPolicyConfig
    bottle: TemplateInspectionConfig
    cork: TemplateInspectionConfig
    foreign_object: ForeignObjectInspectionConfig
    liquid_level: LiquidLevelInspectionConfig
    glass: GlassInspectionConfig


def _require_table(
    data: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Return a required TOML table."""
    value = data.get(name)

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Missing or invalid configuration table: [{name}]"
        )

    return value


def _rectangle_from_table(
    name: str,
    data: dict[str, Any],
) -> RectangleConfig:
    """Create rectangle configuration from a TOML table."""
    try:
        return RectangleConfig(
            name=name,
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
    except KeyError as exc:
        raise ConfigurationError(
            f"Rectangle '{name}' is missing field: {exc.args[0]}"
        ) from exc


def _template_config(
    data: dict[str, Any],
) -> TemplateInspectionConfig:
    """Build template-inspection configuration."""
    try:
        score = data.get("minimum_match_score")

        return TemplateInspectionConfig(
            enabled=bool(data["enabled"]),
            template=Path(str(data["template"])),
            method=str(data["method"]),
            minimum_match_score=(
                float(score)
                if score is not None
                else None
            ),
        )
    except KeyError as exc:
        raise ConfigurationError(
            f"Template configuration is missing field: {exc.args[0]}"
        ) from exc


def load_inspection_config(
    path: str | Path,
) -> InspectionConfig:
    """Load and validate an inspection profile from TOML."""
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Inspection configuration does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise IsADirectoryError(
            f"Inspection configuration is not a file: {config_path}"
        )

    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"Invalid TOML configuration: {config_path}"
        ) from exc

    try:
        profile = str(data["profile"])

        inspection_data = _require_table(
            data,
            "inspection",
        )
        bottle_data = _require_table(
            data,
            "bottle",
        )
        cork_data = _require_table(
            data,
            "cork",
        )
        foreign_data = _require_table(
            data,
            "foreign_object",
        )
        liquid_data = _require_table(
            data,
            "liquid_level",
        )
        glass_data = _require_table(
            data,
            "glass",
        )

        foreign_roi = _rectangle_from_table(
            "foreign-object",
            _require_table(
                foreign_data,
                "roi",
            ),
        )

        decision_region = _rectangle_from_table(
            "foreign-object-decision",
            _require_table(
                foreign_data,
                "decision_region",
            ),
        )

        liquid_roi = _rectangle_from_table(
            "liquid-level",
            _require_table(
                liquid_data,
                "roi",
            ),
        )

        neck_region = _rectangle_from_table(
            "liquid-level-neck",
            _require_table(
                liquid_data,
                "neck_region",
            ),
        )

        foreign_object = ForeignObjectInspectionConfig(
            enabled=bool(foreign_data["enabled"]),
            roi=foreign_roi,
            decision_region=decision_region,
            gaussian_kernel_size=int(
                foreign_data["gaussian_kernel_size"]
            ),
            closing_kernel_size=int(
                foreign_data["closing_kernel_size"]
            ),
            content_blur_kernel_size=int(
                foreign_data["content_blur_kernel_size"]
            ),
            canny_low_threshold=int(
                foreign_data["canny_low_threshold"]
            ),
            canny_high_threshold=int(
                foreign_data["canny_high_threshold"]
            ),
            content_threshold=int(
                foreign_data["content_threshold"]
            ),
            reflection_threshold=int(
                foreign_data["reflection_threshold"]
            ),
            component_count=int(
                foreign_data["component_count"]
            ),
            minimum_contour_area=float(
                foreign_data["minimum_contour_area"]
            ),
            defect_pixel_threshold=int(
                foreign_data["defect_pixel_threshold"]
            ),
        )

        liquid_level = LiquidLevelInspectionConfig(
            enabled=bool(liquid_data["enabled"]),
            roi=liquid_roi,
            neck_region=neck_region,
            canny_threshold1=int(
                liquid_data["canny_threshold1"]
            ),
            canny_threshold2=int(
                liquid_data["canny_threshold2"]
            ),
            hough_rho=float(
                liquid_data["hough_rho"]
            ),
            hough_theta_degrees=float(
                liquid_data["hough_theta_degrees"]
            ),
            hough_threshold=int(
                liquid_data["hough_threshold"]
            ),
            minimum_line_length=int(
                liquid_data["minimum_line_length"]
            ),
            maximum_line_gap=int(
                liquid_data["maximum_line_gap"]
            ),
            acceptable_minimum=float(
                liquid_data["acceptable_minimum"]
            ),
            acceptable_maximum=float(
                liquid_data["acceptable_maximum"]
            ),
        )

        defect_reference_value = glass_data.get(
            "defect_reference"
        )
        decision_method_value = glass_data.get(
            "decision_method"
        )

        glass = GlassInspectionConfig(
            enabled=bool(glass_data["enabled"]),
            defect_reference=(
                Path(str(defect_reference_value))
                if defect_reference_value is not None
                else None
            ),
            decision_method=(
                str(decision_method_value)
                if decision_method_value is not None
                else None
            ),
        )

        return InspectionConfig(
            profile=profile,
            policy=InspectionPolicyConfig(
                require_all_checks=bool(
                    inspection_data["require_all_checks"]
                )
            ),
            bottle=_template_config(
                bottle_data
            ),
            cork=_template_config(
                cork_data
            ),
            foreign_object=foreign_object,
            liquid_level=liquid_level,
            glass=glass,
        )

    except KeyError as exc:
        raise ConfigurationError(
            f"Configuration is missing field: {exc.args[0]}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Configuration contains an invalid value: {exc}"
        ) from exc