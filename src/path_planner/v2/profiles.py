from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

from path_planner.v2.contracts import PlatformKindV2


PLATFORM_PROFILE_SCHEMA_VERSION_V2 = "path-planner-v2-platform-profile/v1"
WHEEL_RELATIVE_ENERGY_PROXY_ID_V2 = "wheel_relative_motion_energy/v1"


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _slope_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("max_traversable_slope_deg must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError("max_traversable_slope_deg must be finite")
    if not 0.0 <= normalized <= 90.0:
        raise ValueError("max_traversable_slope_deg must be in [0, 90]")
    return normalized


def _goal_tolerance(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most pi")
    return normalized


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class PlatformProfileV2:
    profile_id: str
    platform_kind: PlatformKindV2
    capability_revision: str
    simulation_proxy: bool
    max_traversable_slope_deg: float
    goal_position_tolerance_m: float = 0.0
    goal_heading_tolerance_rad: float = 0.0
    schema_version: str = PLATFORM_PROFILE_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        _nonempty_string(self.profile_id, "profile_id")
        if not isinstance(self.platform_kind, PlatformKindV2):
            raise TypeError("platform_kind must be PlatformKindV2")
        _nonempty_string(self.capability_revision, "capability_revision")
        if not isinstance(self.simulation_proxy, bool):
            raise TypeError("simulation_proxy must be bool")
        object.__setattr__(
            self,
            "max_traversable_slope_deg",
            _slope_threshold(self.max_traversable_slope_deg),
        )
        object.__setattr__(
            self,
            "goal_position_tolerance_m",
            _goal_tolerance(
                self.goal_position_tolerance_m,
                "goal_position_tolerance_m",
            ),
        )
        object.__setattr__(
            self,
            "goal_heading_tolerance_rad",
            _goal_tolerance(
                self.goal_heading_tolerance_rad,
                "goal_heading_tolerance_rad",
                maximum=pi,
            ),
        )
        if self.schema_version != PLATFORM_PROFILE_SCHEMA_VERSION_V2:
            raise ValueError(
                f"schema_version must be {PLATFORM_PROFILE_SCHEMA_VERSION_V2}"
            )


@dataclass(frozen=True, slots=True)
class WheelProfileV2:
    profile: PlatformProfileV2
    steering_model: str = "differential_skid_steer"
    body_length_m: float = 0.612
    body_width_m: float = 0.580
    footprint_safety_margin_m: float = 0.0
    reverse_enabled: bool = True
    turn_in_place_enabled: bool = True
    min_turning_radius_m: float = 0.0
    theta_bin_count: int = 72
    primitive_duration_s: float = 1.0
    integration_dt_s: float = 0.25
    max_speed_mps: float = 1.0
    max_angular_speed_radps: float = pi / 4.0
    relative_energy_proxy_id: str = WHEEL_RELATIVE_ENERGY_PROXY_ID_V2
    translation_energy_per_m: float = 1.0
    rotation_energy_per_rad: float = 0.2
    idle_energy_per_s: float = 0.05
    reverse_energy_multiplier: float = 1.25
    energy_normalization: float = 1.0
    time_normalization_s: float = 1.0

    def __post_init__(self) -> None:
        if type(self.profile) is not PlatformProfileV2:
            raise TypeError("profile must be exact PlatformProfileV2")
        if self.profile.platform_kind is not PlatformKindV2.WHEEL:
            raise ValueError("wheel profile requires PlatformKindV2.WHEEL")
        if self.profile.max_traversable_slope_deg != 30.0:
            raise ValueError("wheel profile slope boundary must be exactly 30.0")
        if self.steering_model != "differential_skid_steer":
            raise ValueError("steering_model must be differential_skid_steer")

        body_length = _positive_float(self.body_length_m, "body_length_m")
        body_width = _positive_float(self.body_width_m, "body_width_m")
        if body_length != 0.612:
            raise ValueError("body_length_m must be exactly 0.612")
        if body_width != 0.580:
            raise ValueError("body_width_m must be exactly 0.580")
        object.__setattr__(self, "body_length_m", body_length)
        object.__setattr__(self, "body_width_m", body_width)
        object.__setattr__(
            self,
            "footprint_safety_margin_m",
            _goal_tolerance(
                self.footprint_safety_margin_m,
                "footprint_safety_margin_m",
            ),
        )

        for name in ("reverse_enabled", "turn_in_place_enabled"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        object.__setattr__(
            self,
            "min_turning_radius_m",
            _goal_tolerance(self.min_turning_radius_m, "min_turning_radius_m"),
        )
        if isinstance(self.theta_bin_count, bool) or not isinstance(
            self.theta_bin_count,
            int,
        ):
            raise TypeError("theta_bin_count must be an integer and must not be bool")
        if self.theta_bin_count <= 0:
            raise ValueError("theta_bin_count must be positive")

        for name in (
            "primitive_duration_s",
            "integration_dt_s",
            "max_speed_mps",
            "max_angular_speed_radps",
            "energy_normalization",
            "time_normalization_s",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if not isinstance(self.relative_energy_proxy_id, str) or not re.search(
            r"/v[1-9][0-9]*$",
            self.relative_energy_proxy_id,
        ):
            raise ValueError("relative_energy_proxy_id must be a versioned nonempty id")
        if self.relative_energy_proxy_id != WHEEL_RELATIVE_ENERGY_PROXY_ID_V2:
            raise ValueError(
                "relative_energy_proxy_id must be the fixed wheel relative-energy proxy"
            )
        for name in (
            "translation_energy_per_m",
            "rotation_energy_per_rad",
            "idle_energy_per_s",
            "reverse_energy_multiplier",
        ):
            object.__setattr__(self, name, _goal_tolerance(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class PlatformProfileRegistryV2:
    profiles: tuple[PlatformProfileV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profiles, tuple):
            raise TypeError("profiles must be a tuple")
        if any(type(profile) is not PlatformProfileV2 for profile in self.profiles):
            raise TypeError("profiles must contain exact PlatformProfileV2 values")
        profile_ids = tuple(profile.profile_id for profile in self.profiles)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("profile_id values must be unique")
        if profile_ids != tuple(sorted(profile_ids)):
            raise ValueError("profiles must be sorted by profile_id")

    def resolve(self, profile_id: str) -> PlatformProfileV2 | None:
        if not isinstance(profile_id, str):
            raise TypeError("profile_id must be a string")
        _nonempty_string(profile_id, "profile_id")
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        return None
