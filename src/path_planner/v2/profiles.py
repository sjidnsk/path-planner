from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

from path_planner.v2.contracts import PlatformKindV2


PLATFORM_PROFILE_SCHEMA_VERSION_V2 = "path-planner-v2-platform-profile/v1"


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
