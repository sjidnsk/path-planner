from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PLATFORM_KEY = "yutu2"


@dataclass(frozen=True)
class PlannerPlatformProfile:
    platform_key: str
    platform_name: str
    config_path: Path | None
    safety_margin_m: float
    body_length_m: float | None
    body_width_m: float | None
    footprint_radius_m: float | None
    max_slope_deg: float | None
    max_obstacle_height_m: float | None
    ground_clearance_m: float | None
    raw_min_turning_radius_m: float | None
    effective_min_turning_radius_m: float | None
    energy_model: Any | None
    parameter_sources: dict[str, dict[str, Any]]
    constraint_sources: dict[str, str]
    constraint_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_key": self.platform_key,
            "platform_name": self.platform_name,
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "dimensions": {
                "body_length_m": self.body_length_m,
                "body_width_m": self.body_width_m,
            },
            "limits": {
                "max_slope_deg": self.max_slope_deg,
                "max_obstacle_height_m": self.max_obstacle_height_m,
                "ground_clearance_m": self.ground_clearance_m,
                "raw_min_turning_radius_m": self.raw_min_turning_radius_m,
                "effective_min_turning_radius_m": self.effective_min_turning_radius_m,
            },
            "derived_constraints": {
                "safety_margin_m": self.safety_margin_m,
                "footprint_radius_m": self.footprint_radius_m,
                "energy_model": self.energy_model,
            },
            "parameter_sources": self.parameter_sources,
            "constraint_sources": self.constraint_sources,
            "constraint_warnings": list(self.constraint_warnings),
        }


def load_planner_platform_profile(
    *,
    platform: str = DEFAULT_PLATFORM_KEY,
    config_path: str | Path | None = None,
    safety_margin_m: float = 0.0,
    min_turning_radius_override_m: float | None = None,
) -> PlannerPlatformProfile:
    if safety_margin_m < 0.0:
        raise ValueError("safety_margin_m must be nonnegative")
    if min_turning_radius_override_m is not None and min_turning_radius_override_m < 0.0:
        raise ValueError("min_turning_radius_override_m must be nonnegative")

    default_platform_config_path, load_platform_parameters = _load_dev_platform_functions()

    resolved_path = Path(config_path) if config_path is not None else default_platform_config_path(platform)
    platform_parameters = load_platform_parameters(resolved_path)
    return planner_profile_from_platform_parameters(
        platform_parameters,
        platform_key=platform,
        config_path=resolved_path,
        safety_margin_m=safety_margin_m,
        min_turning_radius_override_m=min_turning_radius_override_m,
    )


def planner_profile_from_platform_parameters(
    platform_parameters: Any,
    *,
    platform_key: str,
    config_path: str | Path | None = None,
    safety_margin_m: float = 0.0,
    min_turning_radius_override_m: float | None = None,
) -> PlannerPlatformProfile:
    if safety_margin_m < 0.0:
        raise ValueError("safety_margin_m must be nonnegative")
    if min_turning_radius_override_m is not None and min_turning_radius_override_m < 0.0:
        raise ValueError("min_turning_radius_override_m must be nonnegative")

    parameter_sources = {
        key: _parameter_to_source(value)
        for key, value in platform_parameters.parameters.items()
    }
    warnings = list(platform_parameters.validate())

    body_length_m = _optional_float(platform_parameters, "body_length")
    body_width_m = _optional_float(platform_parameters, "body_width")
    footprint_radius_m: float | None = None
    constraint_sources: dict[str, str] = {}
    if body_length_m is not None and body_width_m is not None:
        footprint_radius_m = math.hypot(body_length_m, body_width_m) / 2.0 + safety_margin_m
        constraint_sources["footprint_radius_m"] = (
            "derived from dev-platform-constraints body_length/body_width plus safety_margin_m"
        )
    else:
        warnings.append("body_length/body_width missing; vehicle footprint is not enforced")
        constraint_sources["footprint_radius_m"] = "missing"

    raw_min_turning_radius_m = _optional_float(platform_parameters, "min_turning_radius")
    min_turning_source = parameter_sources.get("min_turning_radius", {}).get("source_kind")
    effective_min_turning_radius_m: float | None
    if min_turning_radius_override_m is not None:
        effective_min_turning_radius_m = min_turning_radius_override_m if min_turning_radius_override_m > 0.0 else None
        constraint_sources["effective_min_turning_radius_m"] = "override"
    elif (
        raw_min_turning_radius_m is not None
        and raw_min_turning_radius_m > 0.0
        and min_turning_source != "assumed"
    ):
        effective_min_turning_radius_m = raw_min_turning_radius_m
        constraint_sources["effective_min_turning_radius_m"] = "platform:min_turning_radius"
    else:
        effective_min_turning_radius_m = None
        constraint_sources["effective_min_turning_radius_m"] = "disabled"
        warnings.append("min_turning_radius is not enforced because it is missing, zero, or assumed")

    return PlannerPlatformProfile(
        platform_key=platform_key,
        platform_name=str(platform_parameters.name),
        config_path=Path(config_path) if config_path is not None else None,
        safety_margin_m=float(safety_margin_m),
        body_length_m=body_length_m,
        body_width_m=body_width_m,
        footprint_radius_m=footprint_radius_m,
        max_slope_deg=_optional_float(platform_parameters, "max_slope_deg"),
        max_obstacle_height_m=_optional_float(platform_parameters, "max_obstacle_height"),
        ground_clearance_m=_optional_float(platform_parameters, "ground_clearance"),
        raw_min_turning_radius_m=raw_min_turning_radius_m,
        effective_min_turning_radius_m=effective_min_turning_radius_m,
        energy_model=_optional_raw_value(platform_parameters, "energy_model"),
        parameter_sources=parameter_sources,
        constraint_sources=constraint_sources,
        constraint_warnings=tuple(warnings),
    )


def _optional_float(platform_parameters: Any, key: str) -> float | None:
    if key not in platform_parameters.parameters:
        return None
    return float(platform_parameters.float_value(key))


def _optional_raw_value(platform_parameters: Any, key: str) -> Any | None:
    if key not in platform_parameters.parameters:
        return None
    return platform_parameters.parameters[key].value


def _parameter_to_source(parameter: Any) -> dict[str, Any]:
    return {
        "value": parameter.value,
        "unit": parameter.unit,
        "source_kind": parameter.source_kind,
        "note": parameter.note,
    }


def _load_dev_platform_functions() -> tuple[Any, Any]:
    try:
        from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters

        return default_platform_config_path, load_platform_parameters
    except ImportError as first_error:
        workspace_src = Path(__file__).resolve().parents[3] / "dev-platform-constraints" / "src"
        if workspace_src.exists():
            src_text = str(workspace_src)
            if src_text not in sys.path:
                sys.path.insert(0, src_text)
            try:
                from dev_platform_constraints.platforms import default_platform_config_path, load_platform_parameters

                return default_platform_config_path, load_platform_parameters
            except ImportError:
                pass
        raise RuntimeError(
            "dev-platform-constraints must be installed, available on PYTHONPATH, "
            "or present as ../dev-platform-constraints/src"
        ) from first_error
