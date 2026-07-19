from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

from path_planner.v2.contracts import PlatformKindV2


PLATFORM_PROFILE_SCHEMA_VERSION_V2 = "path-planner-v2-platform-profile/v1"
WHEEL_RELATIVE_ENERGY_PROXY_ID_V2 = "wheel_relative_motion_energy/v1"
LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2 = "simulation_proxy_static_crawl/v1"
HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 = (
    "simulation_proxy_lunar_ballistic/v1"
)
HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2 = "hopper_proxy_profile_incomplete"


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, RuntimeError):
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _slope_threshold(value: object) -> float:
    normalized = _finite_float(value, "max_traversable_slope_deg")
    if not 0.0 <= normalized <= 90.0:
        raise ValueError("max_traversable_slope_deg must be in [0, 90]")
    return normalized


def _goal_tolerance(value: object, name: str, *, maximum: float | None = None) -> float:
    normalized = _finite_float(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most pi")
    return normalized


def _positive_float(value: object, name: str) -> float:
    normalized = _finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _fixed_float(value: object, name: str, expected: float) -> float:
    normalized = _finite_float(value, name)
    if normalized != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return expected


def _exact_profile_string(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact built-in str")
    return value


def _exact_profile_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _fixed_exact_float(value: object, name: str, expected: float) -> float:
    normalized = _exact_profile_float(value, name)
    if normalized != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return expected


def _fixed_exact_float_tuple(
    value: object,
    name: str,
    expected: tuple[float, ...],
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be exact tuple")
    if any(type(item) is not float for item in value):
        raise TypeError(f"{name} elements must be exact float")
    if value != expected:
        raise ValueError(f"{name} must match the frozen proxy values")
    return expected


def _optional_positive_exact_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = _exact_profile_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive when provided")
    return normalized


def _optional_versioned_proxy_id(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str when provided")
    if not value or value.strip() != value or not re.search(r"/v[1-9][0-9]*$", value):
        raise ValueError(f"{name} must be a nonempty versioned proxy id")
    return value


def _required_hopper_profile_field(value: PlatformProfileV2, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError:
        raise TypeError(f"profile is missing required field {name}") from None


def _reaudit_hopper_platform_profile(value: object) -> PlatformProfileV2:
    if type(value) is not PlatformProfileV2:
        raise TypeError("profile must be exact PlatformProfileV2")
    profile_id = _exact_profile_string(
        _required_hopper_profile_field(value, "profile_id"), "profile_id"
    )
    capability_revision = _exact_profile_string(
        _required_hopper_profile_field(value, "capability_revision"),
        "capability_revision",
    )
    schema_version = _exact_profile_string(
        _required_hopper_profile_field(value, "schema_version"), "schema_version"
    )
    platform_kind = _required_hopper_profile_field(value, "platform_kind")
    if type(platform_kind) is not PlatformKindV2:
        raise TypeError("platform_kind must be exact PlatformKindV2")
    simulation_proxy = _required_hopper_profile_field(value, "simulation_proxy")
    if type(simulation_proxy) is not bool:
        raise TypeError("simulation_proxy must be exact bool")
    audited = PlatformProfileV2(
        profile_id=profile_id,
        platform_kind=platform_kind,
        capability_revision=capability_revision,
        simulation_proxy=simulation_proxy,
        max_traversable_slope_deg=_exact_profile_float(
            _required_hopper_profile_field(value, "max_traversable_slope_deg"),
            "max_traversable_slope_deg",
        ),
        goal_position_tolerance_m=_exact_profile_float(
            _required_hopper_profile_field(value, "goal_position_tolerance_m"),
            "goal_position_tolerance_m",
        ),
        goal_heading_tolerance_rad=_exact_profile_float(
            _required_hopper_profile_field(value, "goal_heading_tolerance_rad"),
            "goal_heading_tolerance_rad",
        ),
        schema_version=schema_version,
    )
    if audited.platform_kind is not PlatformKindV2.HOPPER:
        raise ValueError("hopper profile requires PlatformKindV2.HOPPER")
    if audited.simulation_proxy is not True:
        raise ValueError("hopper profile requires simulation_proxy=True")
    if audited.capability_revision != HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2:
        raise ValueError("capability_revision must be the fixed lunar ballistic proxy")
    if audited.max_traversable_slope_deg != 30.0:
        raise ValueError("hopper profile slope boundary must be exactly 30.0")
    if audited.goal_position_tolerance_m != 0.0:
        raise ValueError("goal_position_tolerance_m must be exactly 0.0")
    if audited.goal_heading_tolerance_rad != 0.0:
        raise ValueError("goal_heading_tolerance_rad must be exactly 0.0")
    return audited


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
class LeggedProfileV2:
    profile: PlatformProfileV2
    body_length_m: float = 0.60
    body_width_m: float = 0.40
    nominal_foot_rectangle_length_m: float = 0.70
    nominal_foot_rectangle_width_m: float = 0.50
    max_step_length_m: float = 0.50
    max_step_height_m: float = 0.25
    max_foothold_slope_deg: float = 25.0
    min_support_margin_m: float = 0.05
    local_foothold_grid_spacing_m: float = 0.25

    def __post_init__(self) -> None:
        if type(self.profile) is not PlatformProfileV2:
            raise TypeError("profile must be exact PlatformProfileV2")

        profile_id = _exact_profile_string(self.profile.profile_id, "profile_id")
        capability_revision = _exact_profile_string(
            self.profile.capability_revision,
            "capability_revision",
        )
        schema_version = _exact_profile_string(
            self.profile.schema_version,
            "schema_version",
        )
        platform_kind = self.profile.platform_kind
        if type(platform_kind) is not PlatformKindV2:
            raise TypeError("platform_kind must be exact PlatformKindV2")
        simulation_proxy = self.profile.simulation_proxy
        if type(simulation_proxy) is not bool:
            raise TypeError("simulation_proxy must be exact bool")
        max_slope = _exact_profile_float(
            self.profile.max_traversable_slope_deg,
            "max_traversable_slope_deg",
        )
        position_tolerance = _exact_profile_float(
            self.profile.goal_position_tolerance_m,
            "goal_position_tolerance_m",
        )
        heading_tolerance = _exact_profile_float(
            self.profile.goal_heading_tolerance_rad,
            "goal_heading_tolerance_rad",
        )
        audited_profile = PlatformProfileV2(
            profile_id=profile_id,
            platform_kind=platform_kind,
            capability_revision=capability_revision,
            simulation_proxy=simulation_proxy,
            max_traversable_slope_deg=max_slope,
            goal_position_tolerance_m=position_tolerance,
            goal_heading_tolerance_rad=heading_tolerance,
            schema_version=schema_version,
        )

        if audited_profile.platform_kind is not PlatformKindV2.LEGGED:
            raise ValueError("legged profile requires PlatformKindV2.LEGGED")
        if audited_profile.simulation_proxy is not True:
            raise ValueError("legged profile requires simulation_proxy=True")
        if (
            audited_profile.capability_revision
            != LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2
        ):
            raise ValueError(
                "capability_revision must be the fixed static-crawl simulation proxy"
            )
        if audited_profile.max_traversable_slope_deg != 30.0:
            raise ValueError("legged profile slope boundary must be exactly 30.0")
        if audited_profile.goal_position_tolerance_m != 0.0:
            raise ValueError("goal_position_tolerance_m must be exactly 0.0")
        if audited_profile.goal_heading_tolerance_rad != 0.0:
            raise ValueError("goal_heading_tolerance_rad must be exactly 0.0")

        frozen_values = (
            ("body_length_m", 0.60),
            ("body_width_m", 0.40),
            ("nominal_foot_rectangle_length_m", 0.70),
            ("nominal_foot_rectangle_width_m", 0.50),
            ("max_step_length_m", 0.50),
            ("max_step_height_m", 0.25),
            ("max_foothold_slope_deg", 25.0),
            ("min_support_margin_m", 0.05),
            ("local_foothold_grid_spacing_m", 0.25),
        )
        for name, expected in frozen_values:
            object.__setattr__(
                self,
                name,
                _fixed_float(getattr(self, name), name, expected),
            )


@dataclass(frozen=True, slots=True)
class HopperProfileV2:
    profile: PlatformProfileV2
    gravity_mps2: float = 1.62
    launch_speeds_mps: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    launch_elevations_rad: tuple[float, ...] = (pi / 6.0, pi / 4.0, pi / 3.0)
    azimuth_direction_count: int = 16
    landing_sigma_range_scale: float = 0.05
    landing_sigma_offset_m: float = 0.05
    max_landing_slope_deg: float = 15.0
    landing_probability_threshold: float = 0.99
    midcourse_correction_enabled: bool = False
    inflight_observation_enabled: bool = False
    body_envelope_radius_m: float | None = None
    launch_reference_height_m: float | None = None
    arc_clearance_margin_m: float | None = None
    landing_footprint_radius_m: float | None = None
    stop_condition: str | None = None
    energy_model: str | None = None

    def __post_init__(self) -> None:
        _reaudit_hopper_platform_profile(self.profile)
        frozen_floats = (
            ("gravity_mps2", 1.62),
            ("landing_sigma_range_scale", 0.05),
            ("landing_sigma_offset_m", 0.05),
            ("max_landing_slope_deg", 15.0),
            ("landing_probability_threshold", 0.99),
        )
        for name, expected in frozen_floats:
            object.__setattr__(
                self,
                name,
                _fixed_exact_float(getattr(self, name), name, expected),
            )
        object.__setattr__(
            self,
            "launch_speeds_mps",
            _fixed_exact_float_tuple(
                self.launch_speeds_mps,
                "launch_speeds_mps",
                (1.5, 2.0, 2.5, 3.0),
            ),
        )
        object.__setattr__(
            self,
            "launch_elevations_rad",
            _fixed_exact_float_tuple(
                self.launch_elevations_rad,
                "launch_elevations_rad",
                (pi / 6.0, pi / 4.0, pi / 3.0),
            ),
        )
        if type(self.azimuth_direction_count) is not int:
            raise TypeError("azimuth_direction_count must be exact int")
        if self.azimuth_direction_count != 16:
            raise ValueError("azimuth_direction_count must be exactly 16")
        for name in (
            "midcourse_correction_enabled",
            "inflight_observation_enabled",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be exact bool")
            if value is not False:
                raise ValueError(f"{name} must be exactly False")
        for name in (
            "body_envelope_radius_m",
            "launch_reference_height_m",
            "arc_clearance_margin_m",
            "landing_footprint_radius_m",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_exact_float(getattr(self, name), name),
            )
        for name in ("stop_condition", "energy_model"):
            object.__setattr__(
                self,
                name,
                _optional_versioned_proxy_id(getattr(self, name), name),
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
