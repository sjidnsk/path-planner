from __future__ import annotations

import struct
from dataclasses import dataclass
from math import copysign, isfinite, pi

from path_planner.v2.contracts import (
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    ValidationLevelV2,
)


HOPPER_SEARCH_STATE_SCHEMA_V2 = "hopper-nominal-mean-search-state/v1"
HOPPER_JUMP_PRIMITIVE_SCHEMA_V2 = "hopper-jump-primitive/v1"
HOPPER_STATE_KEY_TAG_V1 = 0x484F505045525631

_HOPPER_SPEED_COUNT_V2 = 4
_HOPPER_ELEVATION_COUNT_V2 = 3
_HOPPER_AZIMUTH_COUNT_V2 = 16
_MIN_SELECTED_LANDING_MASS_V2 = 0.99


def _exact_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _canonical_pose(value: object, name: str) -> PoseStateV2:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    x_m = _canonical_zero(_exact_float(value.x_m, f"{name}.x_m"))
    y_m = _canonical_zero(_exact_float(value.y_m, f"{name}.y_m"))
    heading = _canonical_zero(
        _exact_float(value.heading_rad, f"{name}.heading_rad")
    )
    if not -pi <= heading <= pi:
        raise ValueError(f"{name}.heading_rad must be canonical")
    return PoseStateV2(x_m, y_m, heading)


def _exact_schema(value: object, expected: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact built-in str")
    if value != expected:
        raise ValueError(f"{name} must be {expected}")
    return value


def _exact_index(value: object, name: str, upper_bound: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not 0 <= value < upper_bound:
        raise ValueError(f"{name} must be in [0, {upper_bound - 1}]")
    return value


def _float_bits_v2(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big", signed=False)


@dataclass(frozen=True, slots=True)
class HopperSearchStateV2:
    nominal_state: PoseStateV2
    support_height_m: float
    schema_version: str = HOPPER_SEARCH_STATE_SCHEMA_V2

    def __post_init__(self) -> None:
        nominal_state = _canonical_pose(self.nominal_state, "nominal_state")
        support_height_m = _canonical_zero(
            _exact_float(self.support_height_m, "support_height_m")
        )
        schema_version = _exact_schema(
            self.schema_version,
            HOPPER_SEARCH_STATE_SCHEMA_V2,
            "schema_version",
        )
        object.__setattr__(self, "nominal_state", nominal_state)
        object.__setattr__(self, "support_height_m", support_height_m)
        object.__setattr__(self, "schema_version", schema_version)


def nominal_hopper_search_state_v2(
    nominal_state: PoseStateV2,
    support_height_m: float,
) -> HopperSearchStateV2:
    return HopperSearchStateV2(nominal_state, support_height_m)


def hopper_state_key_v2(state: HopperSearchStateV2) -> tuple[int, int, int, int, int]:
    if type(state) is not HopperSearchStateV2:
        raise TypeError("state must be exact HopperSearchStateV2")
    pose = _canonical_pose(state.nominal_state, "state.nominal_state")
    support_height_m = _canonical_zero(
        _exact_float(state.support_height_m, "state.support_height_m")
    )
    _exact_schema(
        state.schema_version,
        HOPPER_SEARCH_STATE_SCHEMA_V2,
        "state.schema_version",
    )
    return (
        HOPPER_STATE_KEY_TAG_V1,
        _float_bits_v2(pose.x_m),
        _float_bits_v2(pose.y_m),
        _float_bits_v2(pose.heading_rad),
        _float_bits_v2(support_height_m),
    )


@dataclass(frozen=True, slots=True)
class HopperJumpPrimitiveV2(RoutePrimitiveV2):
    start_hopper_state: HopperSearchStateV2
    end_hopper_state: HopperSearchStateV2
    speed_index: int
    elevation_index: int
    azimuth_index: int
    parameter_set_id: str
    selected_landing_mass: float
    primitive_schema_version: str = HOPPER_JUMP_PRIMITIVE_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.kind is not PrimitiveKindV2.BALLISTIC_JUMP:
            raise ValueError("kind must be BALLISTIC_JUMP")
        if self.validation_level is not ValidationLevelV2.L2:
            raise ValueError("validation_level must be L2")

        start_pose = _canonical_pose(self.start_state, "start_state")
        end_pose = _canonical_pose(self.end_state, "end_state")
        if type(self.start_hopper_state) is not HopperSearchStateV2:
            raise TypeError("start_hopper_state must be exact HopperSearchStateV2")
        if type(self.end_hopper_state) is not HopperSearchStateV2:
            raise TypeError("end_hopper_state must be exact HopperSearchStateV2")
        start_hopper = HopperSearchStateV2(
            self.start_hopper_state.nominal_state,
            self.start_hopper_state.support_height_m,
            self.start_hopper_state.schema_version,
        )
        end_hopper = HopperSearchStateV2(
            self.end_hopper_state.nominal_state,
            self.end_hopper_state.support_height_m,
            self.end_hopper_state.schema_version,
        )
        if start_pose != start_hopper.nominal_state:
            raise ValueError("start_state must match start_hopper_state.nominal_state")
        if end_pose != end_hopper.nominal_state:
            raise ValueError("end_state must match end_hopper_state.nominal_state")
        if start_hopper.nominal_state.heading_rad != end_hopper.nominal_state.heading_rad:
            raise ValueError("hopper jump heading must remain invariant")
        if start_hopper.support_height_m != end_hopper.support_height_m:
            raise ValueError("hopper jump support height must remain invariant")

        for name in (
            "duration_s",
            "distance_m",
            "energy_cost",
            "observation_contribution",
        ):
            value = _exact_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, _canonical_zero(value))
        if self.observation_contribution != 0.0 or copysign(
            1.0, self.observation_contribution
        ) < 0.0:
            raise ValueError("observation_contribution must be canonical 0.0")

        speed_index = _exact_index(
            self.speed_index, "speed_index", _HOPPER_SPEED_COUNT_V2
        )
        elevation_index = _exact_index(
            self.elevation_index,
            "elevation_index",
            _HOPPER_ELEVATION_COUNT_V2,
        )
        azimuth_index = _exact_index(
            self.azimuth_index, "azimuth_index", _HOPPER_AZIMUTH_COUNT_V2
        )
        if type(self.parameter_set_id) is not str:
            raise TypeError("parameter_set_id must be an exact built-in str")
        if not self.parameter_set_id.strip():
            raise ValueError("parameter_set_id must be nonempty")
        selected_landing_mass = _exact_float(
            self.selected_landing_mass,
            "selected_landing_mass",
        )
        if not _MIN_SELECTED_LANDING_MASS_V2 <= selected_landing_mass <= 1.0:
            raise ValueError("selected_landing_mass must be in [0.99, 1.0]")
        primitive_schema_version = _exact_schema(
            self.primitive_schema_version,
            HOPPER_JUMP_PRIMITIVE_SCHEMA_V2,
            "primitive_schema_version",
        )

        object.__setattr__(self, "start_state", start_pose)
        object.__setattr__(self, "end_state", end_pose)
        object.__setattr__(self, "start_hopper_state", start_hopper)
        object.__setattr__(self, "end_hopper_state", end_hopper)
        object.__setattr__(self, "speed_index", speed_index)
        object.__setattr__(self, "elevation_index", elevation_index)
        object.__setattr__(self, "azimuth_index", azimuth_index)
        object.__setattr__(self, "selected_landing_mass", selected_landing_mass)
        object.__setattr__(self, "primitive_schema_version", primitive_schema_version)
        RoutePrimitiveV2.__post_init__(self)


__all__ = (
    "HopperJumpPrimitiveV2",
    "HopperSearchStateV2",
    "hopper_state_key_v2",
    "nominal_hopper_search_state_v2",
)
