from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, nextafter
from numbers import Real

from path_planner.search import MotionPrimitive, Pose2D, replay_motion_primitive
from path_planner.v2.contracts import (
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    ValidationLevelV2,
)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _to_pose_state(pose: Pose2D) -> PoseStateV2:
    return PoseStateV2(pose.x_m, pose.y_m, pose.theta_rad)


@dataclass(frozen=True, slots=True)
class WheelMotionPrimitiveV2(RoutePrimitiveV2):
    control_name: str
    samples: tuple[PoseStateV2, ...]
    v_mps: float
    omega_radps: float
    reverse: bool
    turn_in_place: bool

    def __post_init__(self) -> None:
        RoutePrimitiveV2.__post_init__(self)
        if self.kind is not PrimitiveKindV2.WHEEL_MOTION:
            raise ValueError("kind must be PrimitiveKindV2.WHEEL_MOTION")
        if not isinstance(self.control_name, str) or not self.control_name.strip():
            raise ValueError("control_name must be a nonempty string")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple")
        if not self.samples or any(
            not isinstance(sample, PoseStateV2) for sample in self.samples
        ):
            raise ValueError("samples must contain PoseStateV2 values")
        if self.samples[0] != self.start_state:
            raise ValueError("samples must begin with start_state")
        if self.samples[-1] != self.end_state:
            raise ValueError("samples must end with end_state")
        if type(self.reverse) is not bool or type(self.turn_in_place) is not bool:
            raise TypeError("reverse and turn_in_place flags must be bool")

        v_mps = _finite_real(self.v_mps, "v_mps")
        omega_radps = _finite_real(self.omega_radps, "omega_radps")
        object.__setattr__(self, "v_mps", v_mps)
        object.__setattr__(self, "omega_radps", omega_radps)

        if len(self.samples) == 1:
            self._validate_hold()
            return
        if self.control_name == "hold":
            raise ValueError("hold must be the single-sample zero motion contract")
        if self.duration_s <= 0.0:
            raise ValueError("non-hold duration_s must be positive")
        if v_mps == 0.0 and omega_radps == 0.0:
            raise ValueError("non-hold control must move or rotate")

        expected_reverse = v_mps < 0.0
        expected_turn_in_place = v_mps == 0.0 and omega_radps != 0.0
        if self.reverse is not expected_reverse:
            raise ValueError("reverse flag must match v_mps")
        if self.turn_in_place is not expected_turn_in_place:
            raise ValueError("turn_in_place flag must match v_mps and omega_radps")

        control = MotionPrimitive(
            self.control_name,
            v_mps,
            omega_radps,
            self.duration_s,
            reverse=self.reverse,
            turn_in_place=self.turn_in_place,
        )
        start = Pose2D(
            self.start_state.x_m,
            self.start_state.y_m,
            self.start_state.heading_rad,
        )
        raw_integration_dt_s = _finite_real(
            self.duration_s / float(len(self.samples) - 1),
            "inferred integration_dt_s",
        )
        replay_dt = nextafter(raw_integration_dt_s, float("inf"))
        if not isfinite(replay_dt):
            replay_dt = raw_integration_dt_s
        replay = replay_motion_primitive(start, control, replay_dt)
        expected_samples = tuple(_to_pose_state(sample) for sample in replay.samples)
        if self.samples != expected_samples:
            raise ValueError("samples must match exact public replay")
        if self.end_state != _to_pose_state(replay.end):
            raise ValueError("end_state must match exact public replay")
        if not isclose(
            self.distance_m,
            replay.distance_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("distance_m must match public replay chord distance")

    def _validate_hold(self) -> None:
        if self.control_name != "hold":
            raise ValueError("single-sample zero motion must use hold control")
        if self.start_state != self.end_state:
            raise ValueError("hold start_state and end_state must match")
        if self.duration_s != 0.0 or self.distance_m != 0.0 or self.energy_cost != 0.0:
            raise ValueError("hold duration, distance, and energy must be zero")
        if self.v_mps != 0.0 or self.omega_radps != 0.0:
            raise ValueError("hold velocities must be zero")
        if self.reverse or self.turn_in_place:
            raise ValueError("hold flags must be false")
        if self.validation_level is not ValidationLevelV2.L2:
            raise ValueError("hold validation_level must be L2")
