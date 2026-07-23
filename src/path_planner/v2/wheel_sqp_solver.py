from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, cos, hypot, inf, isfinite, pi, remainder, sin
from numbers import Real
from struct import pack as struct_pack
from typing import Callable, NamedTuple

import numpy as np

from path_planner.v2.contracts import PlanningRequestV2, PoseStateV2
from path_planner.v2.profiles import WheelKinematicSQPProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import snapshot_hash
from path_planner.v2.wheel_corridors import (
    WheelSQPTerrainGuideV1,
    wheel_corridor_path_hash_v1,
    wheel_sqp_corridor_broadphase_cell_bound_v1,
)
from path_planner.v2.wheel_kinematics import (
    integrate_wheel_segment_v2,
    wheel_relative_energy_jacobian_v1,
    wheel_relative_energy_v1,
    wheel_segment_center_control_slew_v1,
    wheel_segment_jacobian_v2,
)
from path_planner.v2.wheel_sqp_contracts import (
    WheelCorridorV2,
    WheelSQPCandidateV2,
    WheelSQPInitialGuessV2,
    WheelSQPModeV2,
    WheelSQPOptimizationResultV2,
    WheelSQPResourceEstimateV1,
    WheelSQPResourceLedgerV1,
    WheelSQPStatusV2,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    L2ReserveModelV1,
    _make_wheel_sqp_resource_estimate_v1,
    _WheelSQPExactSegmentV1,
)
from path_planner.v2.wheel_sqp_initialization import (
    wheel_sqp_initial_guess_hash_v1,
)


WHEEL_SQP_SOLVER_AUDIT_CANDIDATE_V1 = "wheel_sqp_solver_audit_candidate/v1"

_POSITION_TOLERANCE_M = 0.25
_HEADING_TOLERANCE_RAD = 0.08726646259971647
_STATE_SCALES = np.array((1.0, 1.0, pi), dtype=np.float64)
_VARIABLE_SCALES = np.array((1.0, 1.0, pi, 1.0, pi / 4.0, 1.0), dtype=np.float64)
_TERRAIN_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_U63_MAX = (1 << 63) - 1


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if nonnegative and normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be exact int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _absolute_wrapped_heading_error(value: float, target: float) -> float:
    raw_difference = value - target
    if -pi <= raw_difference <= pi:
        return abs(raw_difference)
    return abs(remainder(raw_difference, 2.0 * pi))


@dataclass(frozen=True, slots=True)
class _WheelSQPLayoutValuesV1:
    states: tuple[PoseStateV2, ...]
    controls: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class WheelSQPLayoutV1:
    segment_count: int
    start_state: PoseStateV2

    def __post_init__(self) -> None:
        count = _exact_nonnegative_int(self.segment_count, "segment_count")
        if count == 0:
            raise ValueError("segment_count must be positive")
        if count > 48:
            raise ValueError("segment_count must not exceed the v1 limit of 48")
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")

    @property
    def variable_count(self) -> int:
        return 6 * self.segment_count

    @property
    def variable_scales(self) -> np.ndarray:
        return np.tile(_VARIABLE_SCALES, self.segment_count)

    @property
    def residual_scales(self) -> np.ndarray:
        return np.tile(_STATE_SCALES, self.segment_count)

    def pack(
        self,
        states: tuple[PoseStateV2, ...],
        controls: tuple[tuple[float, float, float], ...],
    ) -> np.ndarray:
        if type(states) is not tuple or len(states) != self.segment_count:
            raise TypeError("states must be an exact tuple with segment_count values")
        if any(type(state) is not PoseStateV2 for state in states):
            raise TypeError("states must contain exact PoseStateV2 values")
        if type(controls) is not tuple or len(controls) != self.segment_count:
            raise TypeError("controls must be an exact tuple with segment_count values")
        vector = np.empty(self.variable_count, dtype=np.float64)
        for index, (state, control) in enumerate(zip(states, controls, strict=True)):
            if type(control) is not tuple or len(control) != 3:
                raise TypeError("controls must contain exact (v, omega, duration) tuples")
            v_mps = _finite_float(control[0], "v_mps")
            omega_radps = _finite_float(control[1], "omega_radps")
            duration_s = _finite_float(control[2], "duration_s")
            offset = 6 * index
            vector[offset : offset + 6] = (
                state.x_m,
                state.y_m,
                state.heading_rad,
                v_mps,
                omega_radps,
                duration_s,
            )
        return vector

    def _view(self, vector: object) -> np.ndarray:
        if type(vector) is not np.ndarray:
            raise TypeError("vector must be an exact numpy.ndarray")
        if vector.dtype != np.dtype(np.float64):
            raise TypeError("vector dtype must be exactly float64")
        if vector.ndim != 1 or vector.shape != (self.variable_count,):
            raise ValueError("vector shape must exactly match the fixed SQP layout")
        if not vector.flags.c_contiguous:
            raise ValueError("vector must be C-contiguous")
        if not bool(np.isfinite(vector).all()):
            raise ValueError("vector must contain only finite values")
        return vector

    def unpack(self, vector: object) -> _WheelSQPLayoutValuesV1:
        values = self._view(vector)
        states: list[PoseStateV2] = []
        controls: list[tuple[float, float, float]] = []
        for index in range(self.segment_count):
            offset = 6 * index
            states.append(
                PoseStateV2(
                    float(values[offset]),
                    float(values[offset + 1]),
                    float(values[offset + 2]),
                )
            )
            controls.append(
                (
                    float(values[offset + 3]),
                    float(values[offset + 4]),
                    float(values[offset + 5]),
                )
            )
        return _WheelSQPLayoutValuesV1(tuple(states), tuple(controls))


def _translation_sign(mode: WheelSQPModeV2) -> int:
    if mode is WheelSQPModeV2.FORWARD:
        return 1
    if mode is WheelSQPModeV2.REVERSE:
        return -1
    return 0


def _require_dedicated_stop(modes: tuple[WheelSQPModeV2, ...]) -> None:
    previous_translation_sign = 0
    stop_since_translation = False
    for mode in modes:
        if mode is WheelSQPModeV2.STOP:
            stop_since_translation = True
            continue
        current_sign = _translation_sign(mode)
        if current_sign == 0:
            continue
        if (
            previous_translation_sign != 0
            and current_sign != previous_translation_sign
            and not stop_since_translation
        ):
            raise ValueError("translation direction change requires a dedicated STOP")
        previous_translation_sign = current_sign
        stop_since_translation = False


def _controls_contract_reason(
    profile: WheelKinematicSQPProfileV2,
    modes: tuple[WheelSQPModeV2, ...],
    controls: tuple[tuple[float, float, float], ...],
) -> str | None:
    for mode, (v_mps, omega_radps, duration_s) in zip(
        modes,
        controls,
        strict=True,
    ):
        if duration_s < profile.min_segment_duration_s or duration_s > profile.max_segment_duration_s:
            return "wheel_sqp_mode_contract_failed"
        if mode is WheelSQPModeV2.FORWARD:
            valid = (
                profile.min_nonzero_control <= v_mps <= profile.max_speed_mps
                and -profile.max_angular_speed_radps
                <= omega_radps
                <= profile.max_angular_speed_radps
            )
        elif mode is WheelSQPModeV2.REVERSE:
            valid = (
                -profile.max_speed_mps <= v_mps <= -profile.min_nonzero_control
                and -profile.max_angular_speed_radps
                <= omega_radps
                <= profile.max_angular_speed_radps
            )
        elif mode is WheelSQPModeV2.TURN_LEFT:
            valid = (
                v_mps == 0.0
                and profile.min_nonzero_control
                <= omega_radps
                <= profile.max_angular_speed_radps
            )
        elif mode is WheelSQPModeV2.TURN_RIGHT:
            valid = (
                v_mps == 0.0
                and -profile.max_angular_speed_radps
                <= omega_radps
                <= -profile.min_nonzero_control
            )
        else:
            valid = v_mps == 0.0 and omega_radps == 0.0
        if not valid:
            return "wheel_sqp_mode_contract_failed"
        if abs(omega_radps) * duration_s > profile.max_segment_heading_change_rad:
            return "wheel_sqp_segment_heading_change_exceeded"
    for left, right in zip(controls, controls[1:]):
        if left[0] * right[0] < 0.0:
            return "wheel_sqp_mode_contract_failed"
        speed_up, slow_down, angular = wheel_segment_center_control_slew_v1(
            *left,
            *right,
        )
        if speed_up > profile.max_linear_accel_mps2:
            return "wheel_sqp_linear_accel_exceeded"
        if slow_down > profile.max_linear_decel_mps2:
            return "wheel_sqp_linear_decel_exceeded"
        if angular > profile.max_angular_accel_radps2:
            return "wheel_sqp_angular_slew_exceeded"
    return None


@dataclass(frozen=True, slots=True)
class WheelSQPTerrainConstraintBlockV1:
    values: np.ndarray
    jacobian: np.ndarray
    tie_count: int

    def __post_init__(self) -> None:
        for value, name in ((self.values, "values"), (self.jacobian, "jacobian")):
            if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
                raise TypeError(f"{name} must be an exact float64 ndarray")
            if not value.flags.c_contiguous or value.flags.writeable:
                raise ValueError(f"{name} must be contiguous and read-only")
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if self.values.ndim != 1 or self.jacobian.ndim != 2:
            raise ValueError("terrain constraint block ranks are fixed")
        if self.jacobian.shape[0] != self.values.shape[0]:
            raise ValueError("terrain value/Jacobian row mismatch")
        _exact_nonnegative_int(self.tie_count, "tie_count")


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelSQPProblemV1:
    corridor: WheelCorridorV2
    initial_guess: WheelSQPInitialGuessV2
    request: PlanningRequestV2
    profile: WheelKinematicSQPProfileV2
    post_solver_reserve: WheelSQPResourceLedgerV1
    terrain_guide: WheelSQPTerrainGuideV1

    def __post_init__(self) -> None:
        if type(self.corridor) is not WheelCorridorV2:
            raise TypeError("corridor must be exact WheelCorridorV2")
        if type(self.initial_guess) is not WheelSQPInitialGuessV2:
            raise TypeError("initial_guess must be exact WheelSQPInitialGuessV2")
        if type(self.request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(self.profile) is not WheelKinematicSQPProfileV2:
            raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
        if type(self.post_solver_reserve) is not WheelSQPResourceLedgerV1:
            raise TypeError("post_solver_reserve must be exact WheelSQPResourceLedgerV1")
        if type(self.terrain_guide) is not WheelSQPTerrainGuideV1:
            raise TypeError("terrain_guide must be exact WheelSQPTerrainGuideV1")
        if not self.post_solver_reserve.accepted or self.post_solver_reserve.reason_code is not None:
            raise ValueError("post_solver_reserve must be an accepted typed receipt")
        if self.request.platform_profile_id != self.profile.profile.profile_id:
            raise ValueError("request and wheel SQP profile identity mismatch")
        if self.terrain_guide.snapshot is not self.request.terrain_snapshot:
            raise ValueError("terrain guide must bind the exact request snapshot")
        if self.terrain_guide.terrain_snapshot_hash != snapshot_hash(
            self.request.terrain_snapshot
        ):
            raise ValueError("terrain guide snapshot identity mismatch")
        if self.terrain_guide.geometry != self.request.terrain_snapshot.geometry:
            raise ValueError("terrain guide geometry identity mismatch")
        if self.terrain_guide.max_slope_deg != self.profile.profile.max_traversable_slope_deg:
            raise ValueError("terrain guide slope authority mismatch")
        if self.corridor.corridor_hash != self.initial_guess.corridor_hash:
            raise ValueError("corridor and initial guess hash mismatch")
        expected_corridor_hash = wheel_corridor_path_hash_v1(
            snapshot_hash(self.request.terrain_snapshot),
            self.corridor.cells,
        )
        if self.corridor.corridor_hash != expected_corridor_hash:
            raise ValueError("corridor hash does not match request terrain snapshot")
        if self.initial_guess.start_state != self.request.start_state:
            raise ValueError("initial guess must use the exact request start value")
        if self.initial_guess.requested_goal != self.request.goal_state:
            raise ValueError("initial guess must preserve the request goal")
        expected_initial_hash = wheel_sqp_initial_guess_hash_v1(
            self.corridor,
            self.request,
            self.profile,
            self.initial_guess.segments,
        )
        if self.initial_guess.initial_guess_hash != expected_initial_hash:
            raise ValueError("initial guess hash does not match its semantic payload")
        count = len(self.initial_guess.segments)
        if count > self.profile.max_segments:
            raise ValueError("initial guess exceeds max_segments")
        _require_dedicated_stop(self.initial_guess.modes)
        initial_controls = tuple(
            (segment.v_mps, segment.omega_radps, segment.duration_s)
            for segment in self.initial_guess.segments
        )
        if _controls_contract_reason(
            self.profile,
            self.initial_guess.modes,
            initial_controls,
        ) is not None:
            raise ValueError("initial guess violates the wheel SQP control contract")

    @property
    def layout(self) -> WheelSQPLayoutV1:
        return WheelSQPLayoutV1(len(self.initial_guess.segments), self.request.start_state)

    @property
    def initial_vector(self) -> np.ndarray:
        return self.layout.pack(
            tuple(segment.end_state for segment in self.initial_guess.segments),
            tuple(
                (segment.v_mps, segment.omega_radps, segment.duration_s)
                for segment in self.initial_guess.segments
            ),
        )

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.layout.variable_count, -inf, dtype=np.float64)
        upper = np.full(self.layout.variable_count, inf, dtype=np.float64)
        for index, mode in enumerate(self.initial_guess.modes):
            offset = 6 * index
            if mode is WheelSQPModeV2.FORWARD:
                lower[offset + 3] = self.profile.min_nonzero_control
                upper[offset + 3] = self.profile.max_speed_mps
                lower[offset + 4] = -self.profile.max_angular_speed_radps
                upper[offset + 4] = self.profile.max_angular_speed_radps
            elif mode is WheelSQPModeV2.REVERSE:
                lower[offset + 3] = -self.profile.max_speed_mps
                upper[offset + 3] = -self.profile.min_nonzero_control
                lower[offset + 4] = -self.profile.max_angular_speed_radps
                upper[offset + 4] = self.profile.max_angular_speed_radps
            elif mode is WheelSQPModeV2.TURN_LEFT:
                lower[offset + 3] = upper[offset + 3] = 0.0
                lower[offset + 4] = self.profile.min_nonzero_control
                upper[offset + 4] = self.profile.max_angular_speed_radps
            elif mode is WheelSQPModeV2.TURN_RIGHT:
                lower[offset + 3] = upper[offset + 3] = 0.0
                lower[offset + 4] = -self.profile.max_angular_speed_radps
                upper[offset + 4] = -self.profile.min_nonzero_control
            elif mode is WheelSQPModeV2.STOP:
                lower[offset + 3] = upper[offset + 3] = 0.0
                lower[offset + 4] = upper[offset + 4] = 0.0
            else:  # pragma: no cover - exact enum is already sealed by the guess
                raise ValueError("unsupported wheel SQP mode")
            lower[offset + 5] = self.profile.min_segment_duration_s
            upper[offset + 5] = self.profile.max_segment_duration_s
        return lower, upper

    def dynamics_residual(self, vector: object) -> np.ndarray:
        unpacked = self.layout.unpack(vector)
        residual = np.empty(3 * self.layout.segment_count, dtype=np.float64)
        start = self.request.start_state
        for index, (declared, control) in enumerate(
            zip(unpacked.states, unpacked.controls, strict=True)
        ):
            replay = integrate_wheel_segment_v2(start, *control)
            row = 3 * index
            residual[row : row + 3] = (
                (declared.x_m - replay.x_m) / _STATE_SCALES[0],
                (declared.y_m - replay.y_m) / _STATE_SCALES[1],
                (declared.heading_rad - replay.heading_rad) / _STATE_SCALES[2],
            )
            start = declared
        return residual

    def dynamics_jacobian(self, vector: object) -> np.ndarray:
        unpacked = self.layout.unpack(vector)
        result = np.zeros(
            (3 * self.layout.segment_count, self.layout.variable_count),
            dtype=np.float64,
        )
        start = self.request.start_state
        for index, (declared, control) in enumerate(
            zip(unpacked.states, unpacked.controls, strict=True)
        ):
            replay_jacobian = wheel_segment_jacobian_v2(start, *control)
            row = 3 * index
            current = 6 * index
            for component in range(3):
                scale = _STATE_SCALES[component]
                result[row + component, current + component] = 1.0 / scale
                if index > 0:
                    previous = 6 * (index - 1)
                    for column in range(3):
                        result[row + component, previous + column] = (
                            -replay_jacobian[component][column] / scale
                        )
                for control_column in range(3):
                    result[row + component, current + 3 + control_column] = (
                        -replay_jacobian[component][3 + control_column] / scale
                    )
            start = declared
        return result

    def terminal_inequalities(self, vector: object) -> np.ndarray:
        final = self.layout.unpack(vector).states[-1]
        goal = self.request.goal_state
        dx = final.x_m - goal.x_m
        dy = final.y_m - goal.y_m
        return np.array(
            (
                _POSITION_TOLERANCE_M**2 - dx * dx - dy * dy,
                cos(final.heading_rad - goal.heading_rad)
                - cos(_HEADING_TOLERANCE_RAD),
            ),
            dtype=np.float64,
        )

    def terrain_inequalities(self, vector: object) -> np.ndarray:
        values = np.empty(5 * self.layout.segment_count, dtype=np.float64)
        _fill_wheel_sqp_terrain_constraints_v2(
            self,
            vector,
            values=values,
            jacobian=None,
        )
        return values

    def terrain_jacobian(self, vector: object) -> np.ndarray:
        jacobian = np.zeros(
            (5 * self.layout.segment_count, self.layout.variable_count),
            dtype=np.float64,
        )
        _fill_wheel_sqp_terrain_constraints_v2(
            self,
            vector,
            values=None,
            jacobian=jacobian,
        )
        return jacobian

    def inequalities(self, vector: object) -> np.ndarray:
        unpacked = self.layout.unpack(vector)
        values = list(self.terminal_inequalities(vector))
        for _, omega_radps, duration_s in unpacked.controls:
            values.append(
                self.profile.max_segment_heading_change_rad
                - abs(omega_radps) * duration_s
            )
        for left, right in zip(unpacked.controls, unpacked.controls[1:]):
            speed_up, slow_down, angular = wheel_segment_center_control_slew_v1(
                *left,
                *right,
            )
            values.extend(
                (
                    self.profile.max_linear_accel_mps2 - speed_up,
                    self.profile.max_linear_decel_mps2 - slow_down,
                    self.profile.max_angular_accel_radps2 - angular,
                )
            )
        values.extend(self.terrain_inequalities(vector))
        return np.asarray(values, dtype=np.float64)

    def inequality_jacobian(self, vector: object) -> np.ndarray:
        unpacked = self.layout.unpack(vector)
        row_count = 9 * self.layout.segment_count - 1
        result = np.zeros((row_count, self.layout.variable_count), dtype=np.float64)
        goal = self.request.goal_state
        final = unpacked.states[-1]
        final_offset = 6 * (self.layout.segment_count - 1)
        result[0, final_offset] = -2.0 * (final.x_m - goal.x_m)
        result[0, final_offset + 1] = -2.0 * (final.y_m - goal.y_m)
        result[1, final_offset + 2] = -sin(final.heading_rad - goal.heading_rad)
        row = 2
        for index, (_, omega_radps, duration_s) in enumerate(unpacked.controls):
            offset = 6 * index
            result[row, offset + 4] = -_sign(omega_radps) * duration_s
            result[row, offset + 5] = -abs(omega_radps)
            row += 1
        for index, (left, right) in enumerate(
            zip(unpacked.controls, unpacked.controls[1:])
        ):
            left_v, left_omega, left_dt = left
            right_v, right_omega, right_dt = right
            tau = 0.5 * (left_dt + right_dt)
            speed_delta = abs(right_v) - abs(left_v)
            angular_delta = right_omega - left_omega
            left_offset = 6 * index
            right_offset = 6 * (index + 1)

            if speed_delta > 0.0:
                numerator = speed_delta
                result[row, left_offset + 3] = _sign(left_v) / tau
                result[row, right_offset + 3] = -_sign(right_v) / tau
                duration_derivative = numerator / (2.0 * tau * tau)
                result[row, left_offset + 5] = duration_derivative
                result[row, right_offset + 5] = duration_derivative
            row += 1
            if speed_delta < 0.0:
                numerator = -speed_delta
                result[row, left_offset + 3] = -_sign(left_v) / tau
                result[row, right_offset + 3] = _sign(right_v) / tau
                duration_derivative = numerator / (2.0 * tau * tau)
                result[row, left_offset + 5] = duration_derivative
                result[row, right_offset + 5] = duration_derivative
            row += 1
            if angular_delta != 0.0:
                angular_sign = _sign(angular_delta)
                result[row, left_offset + 4] = angular_sign / tau
                result[row, right_offset + 4] = -angular_sign / tau
                duration_derivative = abs(angular_delta) / (2.0 * tau * tau)
                result[row, left_offset + 5] = duration_derivative
                result[row, right_offset + 5] = duration_derivative
            row += 1
        _fill_wheel_sqp_terrain_constraints_v2(
            self,
            vector,
            values=None,
            jacobian=result,
            row_offset=row,
        )
        return result

    def objective(self, vector: object) -> float:
        unpacked = self.layout.unpack(vector)
        objective = self.request.objective_profile
        request_cost = 0.0
        for v_mps, omega_radps, duration_s in unpacked.controls:
            request_cost += objective.distance_weight * abs(v_mps) * duration_s
            request_cost += objective.energy_weight * wheel_relative_energy_v1(
                v_mps,
                omega_radps,
                duration_s,
                self.profile,
            )
            request_cost += (
                objective.time_weight
                * duration_s
                / self.profile.time_normalization_s
            )
        slew_regularization = 0.0
        for left, right in zip(unpacked.controls, unpacked.controls[1:]):
            slew = wheel_segment_center_control_slew_v1(*left, *right)
            slew_regularization += sum(component * component for component in slew)
        corridor_deviation = sum(
            (state.x_m - reference.end_state.x_m) ** 2
            + (state.y_m - reference.end_state.y_m) ** 2
            for state, reference in zip(
                unpacked.states,
                self.initial_guess.segments,
                strict=True,
            )
        )
        return float(
            request_cost
            + self.profile.control_slew_regularization_weight * slew_regularization
            + self.profile.corridor_deviation_regularization_weight * corridor_deviation
        )

    def objective_jacobian(self, vector: object) -> np.ndarray:
        unpacked = self.layout.unpack(vector)
        objective = self.request.objective_profile
        gradient = np.zeros(self.layout.variable_count, dtype=np.float64)
        for index, (v_mps, omega_radps, duration_s) in enumerate(unpacked.controls):
            offset = 6 * index
            energy_dv, energy_domega, energy_ddt = wheel_relative_energy_jacobian_v1(
                v_mps,
                omega_radps,
                duration_s,
                self.profile,
            )
            gradient[offset + 3] += (
                objective.distance_weight * _sign(v_mps) * duration_s
                + objective.energy_weight * energy_dv
            )
            gradient[offset + 4] += objective.energy_weight * energy_domega
            gradient[offset + 5] += (
                objective.distance_weight * abs(v_mps)
                + objective.energy_weight * energy_ddt
                + objective.time_weight / self.profile.time_normalization_s
            )
        slew_weight = self.profile.control_slew_regularization_weight
        for index, (left, right) in enumerate(
            zip(unpacked.controls, unpacked.controls[1:])
        ):
            left_v, left_omega, left_dt = left
            right_v, right_omega, right_dt = right
            tau = 0.5 * (left_dt + right_dt)
            speed_delta = abs(right_v) - abs(left_v)
            angular_delta = right_omega - left_omega
            numerator = speed_delta * speed_delta + angular_delta * angular_delta
            scale = slew_weight / (tau * tau)
            left_offset = 6 * index
            right_offset = 6 * (index + 1)
            gradient[left_offset + 3] += -2.0 * speed_delta * _sign(left_v) * scale
            gradient[right_offset + 3] += 2.0 * speed_delta * _sign(right_v) * scale
            gradient[left_offset + 4] += -2.0 * angular_delta * scale
            gradient[right_offset + 4] += 2.0 * angular_delta * scale
            duration_derivative = -slew_weight * numerator / (tau * tau * tau)
            gradient[left_offset + 5] += duration_derivative
            gradient[right_offset + 5] += duration_derivative
        deviation_weight = self.profile.corridor_deviation_regularization_weight
        for index, (state, reference) in enumerate(
            zip(unpacked.states, self.initial_guess.segments, strict=True)
        ):
            offset = 6 * index
            gradient[offset] += 2.0 * deviation_weight * (
                state.x_m - reference.end_state.x_m
            )
            gradient[offset + 1] += 2.0 * deviation_weight * (
                state.y_m - reference.end_state.y_m
            )
        return gradient


def _fill_wheel_sqp_terrain_constraints_v2(
    problem: WheelSQPProblemV1,
    vector: object,
    *,
    values: np.ndarray | None,
    jacobian: np.ndarray | None,
    row_offset: int = 0,
) -> int:
    if type(problem) is not WheelSQPProblemV1:
        raise TypeError("problem must be exact WheelSQPProblemV1")
    unpacked = problem.layout.unpack(vector)
    row_count = 5 * problem.layout.segment_count
    _exact_nonnegative_int(row_offset, "row_offset")
    if values is not None:
        if (
            type(values) is not np.ndarray
            or values.dtype != np.dtype(np.float64)
            or values.ndim != 1
            or values.shape[0] < row_offset + row_count
        ):
            raise ValueError("terrain values output has the wrong fixed shape")
    if jacobian is not None:
        if (
            type(jacobian) is not np.ndarray
            or jacobian.dtype != np.dtype(np.float64)
            or jacobian.ndim != 2
            or jacobian.shape[0] < row_offset + row_count
            or jacobian.shape[1] != problem.layout.variable_count
        ):
            raise ValueError("terrain Jacobian output has the wrong fixed shape")
    radius = hypot(
        problem.profile.body_length_m / 2.0
        + problem.profile.footprint_safety_margin_m,
        problem.profile.body_width_m / 2.0
        + problem.profile.footprint_safety_margin_m,
    )
    tie_count = 0
    row = row_offset
    for segment_index, control in enumerate(unpacked.controls):
        start = (
            problem.request.start_state
            if segment_index == 0
            else unpacked.states[segment_index - 1]
        )
        v_mps, omega_radps, duration_s = control
        for rho in _TERRAIN_FRACTIONS:
            if rho == 0.0:
                sample = start
                replay_jacobian = None
            else:
                sampled_duration = rho * duration_s
                sample = integrate_wheel_segment_v2(
                    start,
                    v_mps,
                    omega_radps,
                    sampled_duration,
                )
                replay_jacobian = (
                    wheel_segment_jacobian_v2(
                        start,
                        v_mps,
                        omega_radps,
                        sampled_duration,
                    )
                    if jacobian is not None
                    else None
                )
            clearance = problem.terrain_guide.nearest_signed_clearance(
                sample.x_m,
                sample.y_m,
            )
            if values is not None:
                values[row] = clearance.signed_distance_m - radius
            tie_count += clearance.tie_count
            gx = clearance.gradient_x
            gy = clearance.gradient_y
            if jacobian is None:
                row += 1
                continue
            if rho == 0.0:
                if segment_index > 0:
                    previous = 6 * (segment_index - 1)
                    jacobian[row, previous] = gx
                    jacobian[row, previous + 1] = gy
            else:
                assert replay_jacobian is not None
                if segment_index > 0:
                    previous = 6 * (segment_index - 1)
                    for column in range(3):
                        jacobian[row, previous + column] = (
                            gx * replay_jacobian[0][column]
                            + gy * replay_jacobian[1][column]
                        )
                current = 6 * segment_index
                jacobian[row, current + 3] = (
                    gx * replay_jacobian[0][3]
                    + gy * replay_jacobian[1][3]
                )
                jacobian[row, current + 4] = (
                    gx * replay_jacobian[0][4]
                    + gy * replay_jacobian[1][4]
                )
                jacobian[row, current + 5] = rho * (
                    gx * replay_jacobian[0][5]
                    + gy * replay_jacobian[1][5]
                )
            row += 1
    return tie_count


def build_wheel_sqp_terrain_constraints_v2(
    problem: WheelSQPProblemV1,
    vector: object,
) -> WheelSQPTerrainConstraintBlockV1:
    if type(problem) is not WheelSQPProblemV1:
        raise TypeError("problem must be exact WheelSQPProblemV1")
    values = np.empty(5 * problem.layout.segment_count, dtype=np.float64)
    jacobian = np.zeros(
        (5 * problem.layout.segment_count, problem.layout.variable_count),
        dtype=np.float64,
    )
    tie_count = _fill_wheel_sqp_terrain_constraints_v2(
        problem,
        vector,
        values=values,
        jacobian=jacobian,
    )
    values.setflags(write=False)
    jacobian.setflags(write=False)
    return WheelSQPTerrainConstraintBlockV1(values, jacobian, tie_count)


def _checked_u63(value: object, name: str) -> int:
    exact = _exact_nonnegative_int(value, name)
    if exact > _U63_MAX:
        raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
    return exact


def _checked_u63_add(left: int, right: int) -> int:
    left = _checked_u63(left, "left")
    right = _checked_u63(right, "right")
    if left > _U63_MAX - right:
        raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
    return left + right


def _checked_u63_mul(left: int, right: int) -> int:
    left = _checked_u63(left, "left")
    right = _checked_u63(right, "right")
    if left != 0 and right > _U63_MAX // left:
        raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
    return left * right


def _ceil_sampling_intervals_v1(maximum: float, spacing: float) -> int:
    maximum = _finite_float(maximum, "maximum", nonnegative=True)
    spacing = _finite_float(spacing, "spacing")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return _checked_u63(ceil(maximum / spacing), "sampling intervals")


def _interval_record_bound_v1(broadphase_cell_bound: int, cap: int) -> int:
    broadphase = _checked_u63(broadphase_cell_bound, "broadphase_cell_bound")
    cap = _checked_u63(cap, "interval cap")
    factor = (1 << 25) - 1
    if broadphase > cap // factor:
        return cap
    return _checked_u63_mul(broadphase, factor)


def _reserve_receipt_v1(
    problem: WheelSQPProblemV1,
    deadline: PlanningDeadlineV2,
    *,
    segment_count: int,
    broadphase_cell_bound: int,
    interval_record_bound: int,
    encoded_state_bound: int,
    encoded_scalar_bound: int,
) -> WheelSQPResourceLedgerV1:
    receipt = L2ReserveModelV1().assess(
        deadline,
        segment_count=segment_count,
        broadphase_cell_bound=broadphase_cell_bound,
        interval_record_bound=interval_record_bound,
        encoded_state_bound=encoded_state_bound,
        encoded_scalar_bound=encoded_scalar_bound,
        max_segments=problem.profile.max_segments,
        max_l2_candidate_cells=problem.profile.max_l2_candidate_cells,
        max_l2_interval_records=problem.profile.max_l2_interval_records,
        max_encoded_state_bound=encoded_state_bound,
        max_encoded_scalar_bound=encoded_scalar_bound,
    )
    if not receipt.accepted:
        assert receipt.reason_code is not None
        raise WheelSQPWorkLimitError(receipt.reason_code)
    return receipt


def estimate_wheel_sqp_attempt_resources_v2(
    problem: WheelSQPProblemV1,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
) -> WheelSQPResourceEstimateV1:
    if type(problem) is not WheelSQPProblemV1:
        raise TypeError("problem must be exact WheelSQPProblemV1")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1")
    if ledger.deadline is not deadline or ledger.resource_budget != problem.request.resource_budget:
        raise ValueError("wheel SQP resource authority identity mismatch")
    ledger.check_deadline()

    segment_count = _checked_u63(len(problem.initial_guess.segments), "segment_count")
    if not 1 <= segment_count <= problem.profile.max_segments:
        raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
    variable_count = _checked_u63_mul(6, segment_count)
    equality_count = _checked_u63_mul(3, segment_count)
    base_inequality_count = _checked_u63(
        4 * segment_count - 1,
        "base_inequality_count",
    )
    terrain_inequality_count = _checked_u63_mul(5, segment_count)
    total_constraint_count = _checked_u63(12 * segment_count - 1, "total_constraint_count")
    ledger.check_deadline()

    radius = hypot(
        problem.profile.body_length_m / 2.0
        + problem.profile.footprint_safety_margin_m,
        problem.profile.body_width_m / 2.0
        + problem.profile.footprint_safety_margin_m,
    )
    broadphase_cell_bound = wheel_sqp_corridor_broadphase_cell_bound_v1(
        problem.corridor,
        problem.terrain_guide.geometry,
        radius,
        ledger=ledger,
        cap=problem.profile.max_l2_candidate_cells,
    )
    interval_record_bound = _interval_record_bound_v1(
        broadphase_cell_bound,
        problem.profile.max_l2_interval_records,
    )
    ledger.check_deadline()

    rotation_intervals = _ceil_sampling_intervals_v1(
        problem.profile.max_segment_heading_change_rad,
        problem.profile.observation_sample_heading_rad,
    )
    sample_count = 0
    for mode in problem.initial_guess.modes:
        ledger.check_deadline()
        maximum_speed = (
            problem.profile.max_speed_mps
            if mode in (WheelSQPModeV2.FORWARD, WheelSQPModeV2.REVERSE)
            else 0.0
        )
        translation_intervals = _ceil_sampling_intervals_v1(
            maximum_speed * problem.profile.max_segment_duration_s,
            problem.profile.observation_sample_translation_m,
        )
        intervals = max(1, translation_intervals, rotation_intervals)
        sample_count = _checked_u63_add(sample_count, intervals + 1)
    encoded_state_bound = _checked_u63_add(
        _checked_u63_add(3, _checked_u63_mul(4, segment_count)),
        _checked_u63_mul(2, sample_count),
    )
    encoded_scalar_bound = _checked_u63_add(
        _checked_u63_add(14, _checked_u63_mul(24, segment_count)),
        _checked_u63_mul(6, sample_count),
    )
    if ledger.route_states + encoded_state_bound > problem.request.resource_budget.max_route_states:
        raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
    ledger.check_deadline()

    decision_bytes = _checked_u63_mul(8, variable_count)
    jacobian_bytes = _checked_u63_mul(
        _checked_u63_mul(8, total_constraint_count),
        variable_count,
    )
    solver_bytes = _checked_u63(
        problem.profile.solver_memory_reservation_bytes,
        "solver_bytes",
    )
    l2_queue_bytes = _checked_u63_mul(96, interval_record_bound)
    codec_bytes = _checked_u63_add(
        _checked_u63_mul(64, encoded_state_bound),
        _checked_u63_mul(16, encoded_scalar_bound),
    )
    required_bytes = 0
    for component in (
        decision_bytes,
        jacobian_bytes,
        solver_bytes,
        l2_queue_bytes,
        codec_bytes,
    ):
        required_bytes = _checked_u63_add(required_bytes, component)
    ledger.check_deadline()

    _reserve_receipt_v1(
        problem,
        deadline,
        segment_count=segment_count,
        broadphase_cell_bound=broadphase_cell_bound,
        interval_record_bound=interval_record_bound,
        encoded_state_bound=encoded_state_bound,
        encoded_scalar_bound=encoded_scalar_bound,
    )
    ledger.check_deadline()
    ledger.reserve_attempt(required_bytes, encoded_state_bound)
    ledger.check_deadline()
    receipt = _reserve_receipt_v1(
        problem,
        deadline,
        segment_count=segment_count,
        broadphase_cell_bound=broadphase_cell_bound,
        interval_record_bound=interval_record_bound,
        encoded_state_bound=encoded_state_bound,
        encoded_scalar_bound=encoded_scalar_bound,
    )
    ledger.check_deadline()
    estimate = _make_wheel_sqp_resource_estimate_v1(
        segment_count=segment_count,
        variable_count=variable_count,
        equality_count=equality_count,
        base_inequality_count=base_inequality_count,
        terrain_inequality_count=terrain_inequality_count,
        total_constraint_count=total_constraint_count,
        broadphase_cell_bound=broadphase_cell_bound,
        interval_record_bound=interval_record_bound,
        encoded_state_bound=encoded_state_bound,
        encoded_scalar_bound=encoded_scalar_bound,
        decision_bytes=decision_bytes,
        jacobian_bytes=jacobian_bytes,
        solver_bytes=solver_bytes,
        l2_queue_bytes=l2_queue_bytes,
        codec_bytes=codec_bytes,
        required_bytes=required_bytes,
        post_solver_reserve=receipt,
    )
    ledger.check_deadline()
    ledger._bind_latest_attempt_estimate(estimate, problem.request, problem.profile)
    return estimate


@dataclass(frozen=True, slots=True)
class WheelSQPConstraintAuditV1:
    passed: bool
    reason_code: str | None
    max_abs_scaled_dynamics_residual: float | None
    min_inequality_margin: float | None
    objective_value: float | None
    mode_and_slew_passed: bool
    terminal_only_failure: bool

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be exact bool")
        if self.reason_code is not None and type(self.reason_code) is not str:
            raise TypeError("reason_code must be exact str or None")
        for name in (
            "max_abs_scaled_dynamics_residual",
            "min_inequality_margin",
            "objective_value",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = _finite_float(value, name)
                if name != "min_inequality_margin" and normalized < 0.0:
                    raise ValueError(f"{name} must be nonnegative")
        if type(self.mode_and_slew_passed) is not bool:
            raise TypeError("mode_and_slew_passed must be exact bool")
        if type(self.terminal_only_failure) is not bool:
            raise TypeError("terminal_only_failure must be exact bool")
        if self.passed and self.reason_code is not None:
            raise ValueError("passing audit must not have a reason_code")
        if not self.passed and self.reason_code is None:
            raise ValueError("failed audit requires a reason_code")
        if self.passed:
            if any(
                getattr(self, name) is None
                for name in (
                    "max_abs_scaled_dynamics_residual",
                    "min_inequality_margin",
                    "objective_value",
                )
            ):
                raise ValueError("passing audit requires all numeric metrics")
            if not self.mode_and_slew_passed:
                raise ValueError("passing audit requires mode and slew approval")
            if self.terminal_only_failure:
                raise ValueError("passing audit cannot be a terminal-only failure")
        elif self.terminal_only_failure and self.reason_code != "wheel_sqp_infeasible":
            raise ValueError("terminal-only failure must use wheel_sqp_infeasible")


def _failed_audit(
    reason_code: str,
    *,
    dynamics: float | None = None,
    margin: float | None = None,
    objective: float | None = None,
    mode_and_slew_passed: bool = False,
    terminal_only_failure: bool = False,
) -> WheelSQPConstraintAuditV1:
    return WheelSQPConstraintAuditV1(
        passed=False,
        reason_code=reason_code,
        max_abs_scaled_dynamics_residual=dynamics,
        min_inequality_margin=margin,
        objective_value=objective,
        mode_and_slew_passed=mode_and_slew_passed,
        terminal_only_failure=terminal_only_failure,
    )


def _mode_and_slew_reason(
    problem: WheelSQPProblemV1,
    controls: tuple[tuple[float, float, float], ...],
) -> str | None:
    return _controls_contract_reason(
        problem.profile,
        problem.initial_guess.modes,
        controls,
    )


def audit_wheel_sqp_candidate_v2(
    problem: WheelSQPProblemV1,
    vector: object,
    *,
    function_evaluation_count: int = 0,
) -> WheelSQPConstraintAuditV1:
    if type(problem) is not WheelSQPProblemV1:
        return _failed_audit("wheel_sqp_numeric_contract_failed")
    try:
        evaluation_count = _exact_nonnegative_int(
            function_evaluation_count,
            "function_evaluation_count",
        )
        if evaluation_count > problem.profile.max_sqp_function_evaluations:
            return _failed_audit("wheel_sqp_resource_budget_exceeded")
        values = problem.layout._view(vector)
        copied = values.copy(order="C")
        dynamics_values = problem.dynamics_residual(copied)
        if not bool(np.isfinite(dynamics_values).all()):
            return _failed_audit("wheel_sqp_numeric_contract_failed")
        maximum_dynamics = float(np.max(np.abs(dynamics_values)))
        inequality_values = problem.inequalities(copied)
        if not bool(np.isfinite(inequality_values).all()):
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
            )
        minimum_margin = float(np.min(inequality_values))
        objective_value = problem.objective(copied)
        if not isfinite(objective_value) or objective_value < 0.0:
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
            )
        if maximum_dynamics > problem.profile.hard_constraint_tolerance:
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
            )
        unpacked = problem.layout.unpack(copied)
        mode_reason = _mode_and_slew_reason(problem, unpacked.controls)
        if mode_reason is not None:
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
            )
        terrain = problem.terrain_inequalities(copied)
        if not bool(np.isfinite(terrain).all()):
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
                mode_and_slew_passed=True,
            )
        if bool(np.any(terrain < 0.0)):
            return _failed_audit(
                "wheel_sqp_infeasible",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
                mode_and_slew_passed=True,
            )
        terminal = problem.terminal_inequalities(copied)
        exact_heading_error = _absolute_wrapped_heading_error(
            unpacked.states[-1].heading_rad,
            problem.request.goal_state.heading_rad,
        )
        if (
            terminal[0] < 0.0
            or exact_heading_error > _HEADING_TOLERANCE_RAD
        ):
            return _failed_audit(
                "wheel_sqp_infeasible",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
                mode_and_slew_passed=True,
                terminal_only_failure=True,
            )
        # All remaining Task 5 inequality rows are exact mode/slew rows already
        # checked above.  Task 6 may add explicitly tolerance-authorized rows.
        if minimum_margin < 0.0:
            return _failed_audit(
                "wheel_sqp_numeric_contract_failed",
                dynamics=maximum_dynamics,
                margin=minimum_margin,
                objective=objective_value,
                mode_and_slew_passed=True,
            )
        return WheelSQPConstraintAuditV1(
            passed=True,
            reason_code=None,
            max_abs_scaled_dynamics_residual=maximum_dynamics,
            min_inequality_margin=minimum_margin,
            objective_value=objective_value,
            mode_and_slew_passed=True,
            terminal_only_failure=False,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return _failed_audit("wheel_sqp_numeric_contract_failed")


class _WheelSQPAbort(RuntimeError):
    __slots__ = ("reason_code",)

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _ScipyBackendV1(NamedTuple):
    minimize: Callable[..., object]
    Bounds: type
    NonlinearConstraint: type


def _load_scipy_optimize_v1() -> _ScipyBackendV1 | None:
    try:
        import scipy

        if scipy.__version__ != "1.18.0":
            return None
        from scipy.optimize import Bounds, NonlinearConstraint, minimize
    except (ImportError, AttributeError):
        return None
    return _ScipyBackendV1(minimize, Bounds, NonlinearConstraint)


@dataclass(slots=True)
class _AttemptGuardV1:
    problem: WheelSQPProblemV1
    deadline: PlanningDeadlineV2
    ledger: WheelSQPWorkLedgerV1
    post_solver_reserve: WheelSQPResourceLedgerV1 | None = None
    function_evaluation_count: int = 0

    def check(self, vector: object, *, charge_function_evaluation: bool) -> np.ndarray:
        if self.deadline.expired:
            raise _WheelSQPAbort("planning_deadline_expired")
        try:
            checked = self.problem.layout._view(vector)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed") from exc
        if not self.ledger.within_limits:
            raise _WheelSQPAbort("wheel_sqp_resource_budget_exceeded")
        if charge_function_evaluation:
            if (
                self.function_evaluation_count
                >= self.problem.profile.max_sqp_function_evaluations
            ):
                raise _WheelSQPAbort("wheel_sqp_resource_budget_exceeded")
        remaining_s = self.deadline.remaining_s
        reserve = self.post_solver_reserve or self.problem.post_solver_reserve
        if remaining_s <= reserve.reserve_s:
            if remaining_s == 0.0 or self.deadline.expired:
                raise _WheelSQPAbort("planning_deadline_expired")
            raise _WheelSQPAbort("wheel_sqp_resource_budget_exceeded")
        if charge_function_evaluation:
            self.function_evaluation_count += 1
        return checked


def _run_slsqp_v1(
    *,
    backend: _ScipyBackendV1,
    objective_fn: Callable[[np.ndarray], float],
    objective_jacobian: Callable[[np.ndarray], np.ndarray],
    equality_fn: Callable[[np.ndarray], np.ndarray],
    equality_jacobian: Callable[[np.ndarray], np.ndarray],
    inequality_fn: Callable[[np.ndarray], np.ndarray],
    inequality_jacobian: Callable[[np.ndarray], np.ndarray],
    callback: Callable[[np.ndarray], None],
    initial_vector: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    profile: WheelKinematicSQPProfileV2,
) -> object:
    bounds = backend.Bounds(lower_bounds, upper_bounds)
    equality_constraint = backend.NonlinearConstraint(
        equality_fn,
        0.0,
        0.0,
        jac=equality_jacobian,
    )
    inequality_constraint = backend.NonlinearConstraint(
        inequality_fn,
        0.0,
        inf,
        jac=inequality_jacobian,
    )
    # The compiled SLSQP inner QP is not Python-preemptible.  The fixed
    # iteration/evaluation bounds and the late-return deadline check below are
    # promotion requirements, not a claim that callback checks preempt it.
    return backend.minimize(
        objective_fn,
        initial_vector,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=bounds,
        constraints=(equality_constraint, inequality_constraint),
        callback=callback,
        options={
            "ftol": profile.sqp_ftol,
            "maxiter": profile.max_sqp_iterations,
            "disp": False,
        },
    )


def _status_for_reason(reason_code: str) -> WheelSQPStatusV2:
    if reason_code == "planning_deadline_expired":
        return WheelSQPStatusV2.DEADLINE_EXPIRED
    if reason_code == "wheel_sqp_resource_budget_exceeded":
        return WheelSQPStatusV2.RESOURCE_BUDGET_EXCEEDED
    if reason_code == "wheel_sqp_infeasible":
        return WheelSQPStatusV2.INFEASIBLE
    return WheelSQPStatusV2.NUMERIC_CONTRACT_FAILED


def _failure_result(
    reason_code: str,
    *,
    iteration_count: int = 0,
    function_evaluation_count: int = 0,
) -> WheelSQPOptimizationResultV2:
    return WheelSQPOptimizationResultV2(
        status=_status_for_reason(reason_code),
        candidate=None,
        iteration_count=max(0, iteration_count),
        function_evaluation_count=max(0, function_evaluation_count),
        reason_code=reason_code,
    )


def _binary64(value: float) -> bytes:
    normalized = 0.0 if value == 0.0 else value
    return struct_pack(">d", normalized)


def _candidate_hash(
    problem: WheelSQPProblemV1,
    vector: np.ndarray,
    audit: WheelSQPConstraintAuditV1,
) -> str:
    assert audit.objective_value is not None
    assert audit.max_abs_scaled_dynamics_residual is not None
    assert audit.min_inequality_margin is not None
    digest = sha256()
    digest.update(WHEEL_SQP_SOLVER_AUDIT_CANDIDATE_V1.encode("ascii"))
    digest.update(b"\0")
    digest.update(problem.corridor.corridor_hash.encode("ascii"))
    digest.update(problem.initial_guess.initial_guess_hash.encode("ascii"))
    digest.update(struct_pack(">I", problem.layout.segment_count))
    for value in vector:
        digest.update(_binary64(float(value)))
    digest.update(_binary64(audit.objective_value))
    digest.update(_binary64(audit.max_abs_scaled_dynamics_residual))
    digest.update(_binary64(audit.min_inequality_margin))
    digest.update(b"wheel_sqp_feasible")
    return digest.hexdigest()


def _candidate_from_audit(
    problem: WheelSQPProblemV1,
    vector: np.ndarray,
    audit: WheelSQPConstraintAuditV1,
) -> WheelSQPCandidateV2:
    assert audit.passed
    assert audit.objective_value is not None
    unpacked = problem.layout.unpack(vector)
    segments: list[_WheelSQPExactSegmentV1] = []
    start = problem.request.start_state
    for end, control, mode in zip(
        unpacked.states,
        unpacked.controls,
        problem.initial_guess.modes,
        strict=True,
    ):
        v_mps, omega_radps, duration_s = control
        segments.append(
            _WheelSQPExactSegmentV1(
                start_state=start,
                end_state=end,
                v_mps=v_mps,
                omega_radps=omega_radps,
                duration_s=duration_s,
                mode=mode,
                distance_m=abs(v_mps) * duration_s,
                relative_energy=wheel_relative_energy_v1(
                    v_mps,
                    omega_radps,
                    duration_s,
                    problem.profile,
                ),
            )
        )
        start = end
    return WheelSQPCandidateV2(
        candidate_hash=_candidate_hash(problem, vector, audit),
        corridor_hash=problem.corridor.corridor_hash,
        initial_guess_hash=problem.initial_guess.initial_guess_hash,
        segments=tuple(segments),
        objective_value=audit.objective_value,
        status=WheelSQPStatusV2.FEASIBLE,
    )


def solve_wheel_sqp_v2(
    problem: WheelSQPProblemV1,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
) -> WheelSQPOptimizationResultV2:
    if type(problem) is not WheelSQPProblemV1:
        return _failure_result("wheel_sqp_numeric_contract_failed")
    if type(deadline) is not PlanningDeadlineV2:
        return _failure_result("wheel_sqp_numeric_contract_failed")
    if type(ledger) is not WheelSQPWorkLedgerV1:
        return _failure_result("wheel_sqp_numeric_contract_failed")
    if ledger.deadline is not deadline or ledger.resource_budget != problem.request.resource_budget:
        return _failure_result("wheel_sqp_identity_mismatch")

    guard = _AttemptGuardV1(problem, deadline, ledger)
    iteration_count = 0
    try:
        ledger.check_deadline()
        if problem.request.objective_profile.risk_weight != 0.0:
            return _failure_result("wheel_sqp_objective_unsupported")
        resource_estimate = estimate_wheel_sqp_attempt_resources_v2(
            problem,
            deadline,
            ledger,
        )
        guard.post_solver_reserve = resource_estimate.post_solver_reserve
        initial_vector = problem.initial_vector
        guard.check(initial_vector, charge_function_evaluation=False)
        initial_audit = audit_wheel_sqp_candidate_v2(problem, initial_vector)
        initial_incumbent = (
            initial_vector.copy(order="C") if initial_audit.passed else None
        )
        guard.check(initial_vector, charge_function_evaluation=False)
        backend = _load_scipy_optimize_v1()
        guard.check(initial_vector, charge_function_evaluation=False)
        if backend is None:
            return _failure_result("wheel_sqp_backend_unavailable")
        lower_bounds, upper_bounds = problem.bounds_arrays()

        def objective_fn(vector: np.ndarray) -> float:
            checked = guard.check(vector, charge_function_evaluation=True)
            value = problem.objective(checked)
            if not isfinite(value):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def objective_jacobian(vector: np.ndarray) -> np.ndarray:
            checked = guard.check(vector, charge_function_evaluation=False)
            value = problem.objective_jacobian(checked)
            if not bool(np.isfinite(value).all()):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def equality_fn(vector: np.ndarray) -> np.ndarray:
            checked = guard.check(vector, charge_function_evaluation=True)
            value = problem.dynamics_residual(checked)
            if not bool(np.isfinite(value).all()):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def equality_jacobian(vector: np.ndarray) -> np.ndarray:
            checked = guard.check(vector, charge_function_evaluation=False)
            value = problem.dynamics_jacobian(checked)
            if not bool(np.isfinite(value).all()):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def inequality_fn(vector: np.ndarray) -> np.ndarray:
            checked = guard.check(vector, charge_function_evaluation=True)
            value = problem.inequalities(checked)
            if not bool(np.isfinite(value).all()):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def inequality_jacobian(vector: np.ndarray) -> np.ndarray:
            checked = guard.check(vector, charge_function_evaluation=False)
            value = problem.inequality_jacobian(checked)
            if not bool(np.isfinite(value).all()):
                raise _WheelSQPAbort("wheel_sqp_numeric_contract_failed")
            return value

        def iteration_callback(vector: np.ndarray) -> None:
            nonlocal iteration_count
            guard.check(vector, charge_function_evaluation=False)
            if iteration_count >= problem.profile.max_sqp_iterations:
                raise _WheelSQPAbort("wheel_sqp_resource_budget_exceeded")
            iteration_count += 1

        raw = _run_slsqp_v1(
            backend=backend,
            objective_fn=objective_fn,
            objective_jacobian=objective_jacobian,
            equality_fn=equality_fn,
            equality_jacobian=equality_jacobian,
            inequality_fn=inequality_fn,
            inequality_jacobian=inequality_jacobian,
            callback=iteration_callback,
            initial_vector=initial_vector,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            profile=problem.profile,
        )
        raw_vector = getattr(raw, "x", None)
        checked_raw = guard.check(raw_vector, charge_function_evaluation=False)
        audit = audit_wheel_sqp_candidate_v2(
            problem,
            checked_raw,
            function_evaluation_count=guard.function_evaluation_count,
        )
        if not audit.passed:
            assert audit.reason_code is not None
            if (
                audit.reason_code == "wheel_sqp_infeasible"
                and audit.terminal_only_failure
                and initial_incumbent is not None
            ):
                guard.check(initial_incumbent, charge_function_evaluation=False)
                incumbent_audit = audit_wheel_sqp_candidate_v2(
                    problem,
                    initial_incumbent,
                    function_evaluation_count=guard.function_evaluation_count,
                )
                if incumbent_audit.passed:
                    incumbent = _candidate_from_audit(
                        problem,
                        initial_incumbent,
                        incumbent_audit,
                    )
                    guard.check(
                        initial_incumbent,
                        charge_function_evaluation=False,
                    )
                    return WheelSQPOptimizationResultV2(
                        status=WheelSQPStatusV2.FEASIBLE,
                        candidate=incumbent,
                        iteration_count=iteration_count,
                        function_evaluation_count=guard.function_evaluation_count,
                        reason_code=None,
                    )
            return _failure_result(
                audit.reason_code,
                iteration_count=iteration_count,
                function_evaluation_count=guard.function_evaluation_count,
            )
        guard.check(checked_raw, charge_function_evaluation=False)
        copied_raw = checked_raw.copy(order="C")
        candidate = _candidate_from_audit(problem, copied_raw, audit)
        guard.check(copied_raw, charge_function_evaluation=False)
        return WheelSQPOptimizationResultV2(
            status=WheelSQPStatusV2.FEASIBLE,
            candidate=candidate,
            iteration_count=iteration_count,
            function_evaluation_count=guard.function_evaluation_count,
            reason_code=None,
        )
    except _WheelSQPAbort as exc:
        return _failure_result(
            exc.reason_code,
            iteration_count=iteration_count,
            function_evaluation_count=guard.function_evaluation_count,
        )
    except WheelSQPWorkLimitError as exc:
        return _failure_result(
            exc.reason_code,
            iteration_count=iteration_count,
            function_evaluation_count=guard.function_evaluation_count,
        )
    except MemoryError:
        raise
    except Exception:
        try:
            expired_after_exception = deadline.expired
        except MemoryError:
            raise
        except Exception:
            expired_after_exception = False
        if expired_after_exception:
            return _failure_result(
                "planning_deadline_expired",
                iteration_count=iteration_count,
                function_evaluation_count=guard.function_evaluation_count,
            )
        return _failure_result(
            "wheel_sqp_internal_error",
            iteration_count=iteration_count,
            function_evaluation_count=guard.function_evaluation_count,
        )
