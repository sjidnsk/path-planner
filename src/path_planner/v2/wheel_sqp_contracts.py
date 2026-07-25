from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from path_planner.core import Cell
from path_planner.v2.contracts import (
    PlatformKindV2,
    PlanningRequestV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.profiles import WheelKinematicSQPProfileV2


WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2 = "wheel_kinematic_segment/v1"
WHEEL_KINEMATIC_SOLVER_CONTRACT_V2 = "wheel_kinematic_direct_multiple_shooting_sqp/v1"
WHEEL_KINEMATIC_CANONICALIZATION_V2 = "wheel_kinematic_decimal12_half_even/v1"
WHEEL_KINEMATIC_L2_VALIDATOR_V2 = "wheel_kinematic_continuous_rectangle_sweep_l2/v1"
WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2 = "wheel_deterministic_yen_h_signature_corridors/v1"
WHEEL_KINEMATIC_INITIALIZER_V2 = "wheel_forward_reverse_turn_dp_initializer/v1"
WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2 = "wheel_l2_reserve_model/v1"
WHEEL_KINEMATIC_CONTROL_SLEW_V2 = "wheel_segment_center_control_slew/v1"
WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2 = "wheel_kinematic_derived_samples/v1"
WHEEL_SQP_WORK_LEDGER_V1 = "wheel_sqp_shared_work_ledger/v1"
WHEEL_SQP_ATTEMPT_RESOURCE_ESTIMATE_V1 = "wheel_sqp_attempt_resource_estimate/v1"

_WHEEL_SQP_HARD_MEMORY_LIMIT_BYTES_V1 = 64 * 1024 * 1024
_WHEEL_SQP_U63_MAX_V1 = (1 << 63) - 1
_WHEEL_SQP_RESOURCE_ESTIMATE_AUTHORITY = object()
_WHEEL_SQP_L2_RESULT_AUTHORITY = object()


class WheelSQPModeV2(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"


class WheelSQPStatusV2(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INITIALIZATION_FAILED = "initialization_failed"
    NUMERIC_CONTRACT_FAILED = "numeric_contract_failed"
    RESOURCE_BUDGET_EXCEEDED = "resource_budget_exceeded"
    DEADLINE_EXPIRED = "deadline_expired"
    L2_REJECTED = "l2_rejected"


class WheelSQPWorkLimitError(RuntimeError):
    __slots__ = ("reason_code",)

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str:
            raise TypeError("reason_code must be exact str")
        if reason_code not in (
            "planning_deadline_expired",
            "wheel_sqp_corridor_budget_exceeded",
            "wheel_sqp_resource_budget_exceeded",
        ):
            raise ValueError("reason_code must be a shared work-limit reason")
        super().__init__(reason_code)
        self.reason_code = reason_code


class WheelSQPRepairNotAllowed(ValueError):
    """Raised when an L2 counterexample cannot authorize geometric repair."""


class WheelSQPWorkLedgerV1:
    __slots__ = (
        "_accounted_bytes",
        "_deadline",
        "_expanded_states",
        "_latest_attempt_estimate",
        "_latest_attempt_profile",
        "_latest_attempt_request",
        "_resource_budget",
        "_route_states",
        "_validation_candidate_hash",
        "_validation_delta_memory_bytes",
        "_validation_delta_route_states",
    )

    authority_id = WHEEL_SQP_WORK_LEDGER_V1

    def __init__(
        self,
        resource_budget: ResourceBudgetV2,
        deadline: PlanningDeadlineV2,
    ) -> None:
        if type(resource_budget) is not ResourceBudgetV2:
            raise TypeError("resource_budget must be exact ResourceBudgetV2")
        if type(deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")
        self._resource_budget = resource_budget
        self._deadline = deadline
        self._expanded_states = 0
        self._accounted_bytes = 0
        self._route_states = 0
        self._latest_attempt_estimate = None
        self._latest_attempt_request = None
        self._latest_attempt_profile = None
        self._validation_candidate_hash = None
        self._validation_delta_memory_bytes = 0
        self._validation_delta_route_states = 0

    @property
    def resource_budget(self) -> ResourceBudgetV2:
        return self._resource_budget

    @property
    def deadline(self) -> PlanningDeadlineV2:
        return self._deadline

    @property
    def expanded_states(self) -> int:
        return self._expanded_states

    @property
    def accounted_bytes(self) -> int:
        return self._accounted_bytes

    @property
    def route_states(self) -> int:
        return self._route_states

    @property
    def effective_memory_limit_bytes(self) -> int:
        requested = self.resource_budget.max_memory_bytes
        if requested == 0:
            return _WHEEL_SQP_HARD_MEMORY_LIMIT_BYTES_V1
        return min(requested, _WHEEL_SQP_HARD_MEMORY_LIMIT_BYTES_V1)

    @property
    def within_limits(self) -> bool:
        """Report only shared count/memory limits; deadline is checked separately."""
        return (
            self.expanded_states <= self.resource_budget.max_expanded_states
            and self.accounted_bytes <= self.effective_memory_limit_bytes
            and self.route_states <= self.resource_budget.max_route_states
        )

    def check_deadline(self) -> None:
        if self.deadline.expired:
            raise WheelSQPWorkLimitError("planning_deadline_expired")

    def charge_expansion(self) -> None:
        self.check_deadline()
        if self.expanded_states >= self.resource_budget.max_expanded_states:
            raise WheelSQPWorkLimitError("wheel_sqp_corridor_budget_exceeded")
        self._expanded_states += 1

    def charge_memory(self, amount: int) -> None:
        self.check_deadline()
        _exact_nonnegative_int(amount, "amount")
        attempted = self.accounted_bytes + amount
        if attempted > self.effective_memory_limit_bytes:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        self._accounted_bytes = attempted

    def charge_route_states(self, amount: int) -> None:
        self.check_deadline()
        _exact_nonnegative_int(amount, "amount")
        attempted = self.route_states + amount
        if attempted > self.resource_budget.max_route_states:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        self._route_states = attempted

    def reserve_attempt(self, memory_bytes: int, route_states: int) -> None:
        """Atomically reserve one SQP attempt's memory and route-state tail."""

        self.check_deadline()
        _exact_nonnegative_int(memory_bytes, "memory_bytes")
        _exact_nonnegative_int(route_states, "route_states")
        if memory_bytes > _WHEEL_SQP_U63_MAX_V1 or route_states > _WHEEL_SQP_U63_MAX_V1:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        if (
            self.accounted_bytes > _WHEEL_SQP_U63_MAX_V1 - memory_bytes
            or self.route_states > _WHEEL_SQP_U63_MAX_V1 - route_states
        ):
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        prospective_memory = self.accounted_bytes + memory_bytes
        prospective_states = self.route_states + route_states
        if (
            prospective_memory > self.effective_memory_limit_bytes
            or prospective_states > self.resource_budget.max_route_states
        ):
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        self.check_deadline()
        self._accounted_bytes = prospective_memory
        self._route_states = prospective_states

    def _bind_latest_attempt_estimate(
        self,
        estimate: WheelSQPResourceEstimateV1,
        request: object,
        profile: object,
    ) -> None:
        from path_planner.v2.contracts import PlanningRequestV2
        from path_planner.v2.profiles import WheelKinematicSQPProfileV2

        self.check_deadline()
        if type(estimate) is not WheelSQPResourceEstimateV1:
            raise TypeError("estimate must be exact WheelSQPResourceEstimateV1")
        if type(request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(profile) is not WheelKinematicSQPProfileV2:
            raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
        self._latest_attempt_estimate = estimate
        self._latest_attempt_request = request
        self._latest_attempt_profile = profile
        self._validation_candidate_hash = None
        self._validation_delta_memory_bytes = 0
        self._validation_delta_route_states = 0

    def latest_attempt_estimate(
        self,
        request: object,
        profile: object,
    ) -> WheelSQPResourceEstimateV1:
        from path_planner.v2.contracts import PlanningRequestV2
        from path_planner.v2.profiles import WheelKinematicSQPProfileV2

        self.check_deadline()
        if type(request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(profile) is not WheelKinematicSQPProfileV2:
            raise TypeError("profile must be exact WheelKinematicSQPProfileV2")
        if (
            self._latest_attempt_estimate is None
            or self._latest_attempt_request is not request
            or self._latest_attempt_profile is not profile
        ):
            raise ValueError("wheel SQP latest attempt identity mismatch")
        return self._latest_attempt_estimate

    def admit_validation_delta(
        self,
        estimate: WheelSQPResourceEstimateV1,
        candidate_hash: str,
        *,
        memory_bytes: int,
        route_states: int,
    ) -> None:
        self.check_deadline()
        if type(estimate) is not WheelSQPResourceEstimateV1:
            raise TypeError("estimate must be exact WheelSQPResourceEstimateV1")
        _exact_hash(candidate_hash, "candidate_hash")
        desired_memory = _exact_nonnegative_int(memory_bytes, "memory_bytes")
        desired_states = _exact_nonnegative_int(route_states, "route_states")
        if estimate is not self._latest_attempt_estimate:
            raise ValueError("wheel SQP validation attempt identity mismatch")
        if (
            self._validation_candidate_hash is not None
            and self._validation_candidate_hash != candidate_hash
        ):
            raise ValueError("wheel SQP validation candidate identity mismatch")
        memory_delta = max(0, desired_memory - self._validation_delta_memory_bytes)
        state_delta = max(0, desired_states - self._validation_delta_route_states)
        if memory_delta > _WHEEL_SQP_U63_MAX_V1 or state_delta > _WHEEL_SQP_U63_MAX_V1:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        if (
            self.accounted_bytes > _WHEEL_SQP_U63_MAX_V1 - memory_delta
            or self.route_states > _WHEEL_SQP_U63_MAX_V1 - state_delta
        ):
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        prospective_memory = self.accounted_bytes + memory_delta
        prospective_states = self.route_states + state_delta
        if (
            prospective_memory > self.effective_memory_limit_bytes
            or prospective_states > self.resource_budget.max_route_states
        ):
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        self.check_deadline()
        self._accounted_bytes = prospective_memory
        self._route_states = prospective_states
        self._validation_candidate_hash = candidate_hash
        self._validation_delta_memory_bytes = max(
            self._validation_delta_memory_bytes,
            desired_memory,
        )
        self._validation_delta_route_states = max(
            self._validation_delta_route_states,
            desired_states,
        )


def _exact_hash(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _exact_id(value: object, name: str, expected: str | None = None) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    if expected is not None and value != expected:
        raise ValueError(f"{name} must be {expected}")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exact bool")
    return value


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be exact int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _exact_cell(value: object, name: str) -> Cell:
    if type(value) is not Cell:
        raise TypeError(f"{name} must be exact Cell")
    if type(value.x) is not int or type(value.y) is not int:
        raise TypeError(f"{name} coordinates must be exact ints")
    return value


def _exact_finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be exact float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class WheelKinematicSegmentV2(RoutePrimitiveV2):
    v_mps: float
    omega_radps: float
    reverse: bool
    turn_in_place: bool
    relative_energy: float
    samples: tuple[PoseStateV2, ...]
    segment_hash: str
    segment_schema_id: str = WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2

    def __post_init__(self) -> None:
        RoutePrimitiveV2.__post_init__(self)
        if self.kind is not PrimitiveKindV2.WHEEL_MOTION:
            raise ValueError("wheel SQP segment kind must be WHEEL_MOTION")
        _exact_finite_float(self.v_mps, "v_mps")
        _exact_finite_float(self.omega_radps, "omega_radps")
        _exact_bool(self.reverse, "reverse")
        _exact_bool(self.turn_in_place, "turn_in_place")
        _exact_finite_float(self.relative_energy, "relative_energy", nonnegative=True)
        if self.energy_cost != self.relative_energy:
            raise ValueError("energy_cost must equal relative_energy")
        if type(self.samples) is not tuple or not self.samples:
            raise TypeError("samples must be a nonempty exact tuple")
        if any(type(sample) is not PoseStateV2 for sample in self.samples):
            raise TypeError("samples must contain exact PoseStateV2 values")
        if self.samples[0] != self.start_state or self.samples[-1] != self.end_state:
            raise ValueError("samples must begin and end at segment states")
        _exact_hash(self.segment_hash, "segment_hash")
        _exact_id(
            self.segment_schema_id,
            "segment_schema_id",
            WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2,
        )
        if self.reverse is not (self.v_mps < 0.0):
            raise ValueError("reverse flag must match v_mps")
        expected_turn = self.v_mps == 0.0 and self.omega_radps != 0.0
        if self.turn_in_place is not expected_turn:
            raise ValueError("turn_in_place flag must match controls")
        if self.validation_level is not ValidationLevelV2.L2:
            raise ValueError("wheel SQP public segment must be L2")


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelKinematicRouteV2(TypedRouteV2):
    route_hash: str
    source_candidate_hash: str
    request_hash: str
    profile_hash: str
    terrain_snapshot_hash: str
    capability_revision: str
    solver_contract_id: str
    canonicalization_id: str
    validator_contract_id: str

    def __post_init__(self) -> None:
        TypedRouteV2.__post_init__(self)
        if self.platform_kind is not PlatformKindV2.WHEEL:
            raise ValueError("wheel SQP route requires PlatformKindV2.WHEEL")
        if any(type(primitive) is not WheelKinematicSegmentV2 for primitive in self.primitives):
            raise TypeError("wheel SQP route requires exact WheelKinematicSegmentV2 primitives")
        for name in (
            "route_hash",
            "source_candidate_hash",
            "request_hash",
            "profile_hash",
            "terrain_snapshot_hash",
        ):
            _exact_hash(getattr(self, name), name)
        _exact_id(self.capability_revision, "capability_revision", "wheel_kinematic_corridor_sqp/v1")
        _exact_id(
            self.solver_contract_id,
            "solver_contract_id",
            WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
        )
        _exact_id(
            self.canonicalization_id,
            "canonicalization_id",
            WHEEL_KINEMATIC_CANONICALIZATION_V2,
        )
        _exact_id(
            self.validator_contract_id,
            "validator_contract_id",
            WHEEL_KINEMATIC_L2_VALIDATOR_V2,
        )


@dataclass(frozen=True, slots=True, order=True)
class WheelTopologySignatureV1:
    entries: tuple[tuple[str, int], ...]
    signature_id: str = "wheel_component_cut_crossings/v1"

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("entries must be an exact tuple")
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("entries must contain exact (component_hash, count) tuples")
            component_hash, crossing_count = entry
            _exact_hash(component_hash, "component_hash")
            if type(crossing_count) is not int:
                raise TypeError("crossing_count must be exact int")
        _exact_id(
            self.signature_id,
            "signature_id",
            "wheel_component_cut_crossings/v1",
        )


@dataclass(frozen=True, slots=True)
class WheelCorridorV2:
    corridor_index: int
    cells: tuple[Cell, ...]
    corridor_hash: str
    guide_cost: float
    path_length_m: float
    topology_signature: WheelTopologySignatureV1
    source_id: str = WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.corridor_index, "corridor_index")
        if type(self.cells) is not tuple or not self.cells:
            raise TypeError("cells must be a nonempty exact tuple")
        for cell in self.cells:
            _exact_cell(cell, "cells")
        _exact_hash(self.corridor_hash, "corridor_hash")
        _exact_finite_float(self.guide_cost, "guide_cost", nonnegative=True)
        _exact_finite_float(self.path_length_m, "path_length_m", nonnegative=True)
        if type(self.topology_signature) is not WheelTopologySignatureV1:
            raise TypeError("topology_signature must be exact WheelTopologySignatureV1")
        _exact_id(self.source_id, "source_id", WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2)

    @property
    def path_hash(self) -> str:
        return self.corridor_hash


@dataclass(frozen=True, slots=True)
class _WheelSQPInitialSegmentV1:
    start_state: PoseStateV2
    end_state: PoseStateV2
    v_mps: float
    omega_radps: float
    duration_s: float
    mode: WheelSQPModeV2
    distance_m: float
    relative_energy: float

    def __post_init__(self) -> None:
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")
        if type(self.end_state) is not PoseStateV2:
            raise TypeError("end_state must be exact PoseStateV2")
        v_mps = _exact_finite_float(self.v_mps, "v_mps")
        omega_radps = _exact_finite_float(self.omega_radps, "omega_radps")
        duration_s = _exact_finite_float(
            self.duration_s,
            "duration_s",
            nonnegative=True,
        )
        if duration_s == 0.0:
            raise ValueError("duration_s must be positive")
        if type(self.mode) is not WheelSQPModeV2:
            raise TypeError("mode must be exact WheelSQPModeV2")
        _exact_finite_float(self.distance_m, "distance_m", nonnegative=True)
        _exact_finite_float(
            self.relative_energy,
            "relative_energy",
            nonnegative=True,
        )
        expected_mode = (
            WheelSQPModeV2.FORWARD
            if v_mps > 0.0
            else WheelSQPModeV2.REVERSE
            if v_mps < 0.0
            else WheelSQPModeV2.TURN_LEFT
            if omega_radps > 0.0
            else WheelSQPModeV2.TURN_RIGHT
            if omega_radps < 0.0
            else WheelSQPModeV2.STOP
        )
        if self.mode is not expected_mode:
            raise ValueError("mode must exactly match initial controls")
        if self.distance_m != abs(v_mps) * duration_s:
            raise ValueError("distance_m must match initial translation controls")


@dataclass(frozen=True, slots=True)
class WheelSQPInitialGuessV2:
    corridor_hash: str
    start_state: PoseStateV2
    requested_goal: PoseStateV2
    segments: tuple[_WheelSQPInitialSegmentV1, ...]
    actual_endpoint: PoseStateV2
    initial_guess_hash: str
    initializer_id: str = WHEEL_KINEMATIC_INITIALIZER_V2

    def __post_init__(self) -> None:
        _exact_hash(self.corridor_hash, "corridor_hash")
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")
        if type(self.requested_goal) is not PoseStateV2:
            raise TypeError("requested_goal must be exact PoseStateV2")
        if type(self.segments) is not tuple or not self.segments:
            raise TypeError("segments must be a nonempty exact tuple")
        if any(type(segment) is not _WheelSQPInitialSegmentV1 for segment in self.segments):
            raise TypeError("segments must contain exact internal initial segments")
        if self.segments[0].start_state != self.start_state:
            raise ValueError("first segment must start at exact request start")
        for left, right in zip(self.segments, self.segments[1:]):
            if left.end_state != right.start_state:
                raise ValueError("initial segment endpoints must be connected")
        if type(self.actual_endpoint) is not PoseStateV2:
            raise TypeError("actual_endpoint must be exact PoseStateV2")
        if self.segments[-1].end_state != self.actual_endpoint:
            raise ValueError("actual_endpoint must match the final replay endpoint")
        _exact_hash(self.initial_guess_hash, "initial_guess_hash")
        _exact_id(self.initializer_id, "initializer_id", WHEEL_KINEMATIC_INITIALIZER_V2)

    @property
    def modes(self) -> tuple[WheelSQPModeV2, ...]:
        return tuple(segment.mode for segment in self.segments)

    @property
    def v_mps(self) -> tuple[float, ...]:
        return tuple(segment.v_mps for segment in self.segments)

    @property
    def omega_radps(self) -> tuple[float, ...]:
        return tuple(segment.omega_radps for segment in self.segments)

    @property
    def duration_s(self) -> tuple[float, ...]:
        return tuple(segment.duration_s for segment in self.segments)


@dataclass(frozen=True, slots=True)
class _WheelSQPExactSegmentV1:
    """Private, independently audited solver output; this is not an L-level primitive."""

    start_state: PoseStateV2
    end_state: PoseStateV2
    v_mps: float
    omega_radps: float
    duration_s: float
    mode: WheelSQPModeV2
    distance_m: float
    relative_energy: float

    def __post_init__(self) -> None:
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")
        if type(self.end_state) is not PoseStateV2:
            raise TypeError("end_state must be exact PoseStateV2")
        v_mps = _exact_finite_float(self.v_mps, "v_mps")
        omega_radps = _exact_finite_float(self.omega_radps, "omega_radps")
        duration_s = _exact_finite_float(
            self.duration_s,
            "duration_s",
            nonnegative=True,
        )
        if duration_s == 0.0:
            raise ValueError("duration_s must be positive")
        if type(self.mode) is not WheelSQPModeV2:
            raise TypeError("mode must be exact WheelSQPModeV2")
        distance_m = _exact_finite_float(
            self.distance_m,
            "distance_m",
            nonnegative=True,
        )
        _exact_finite_float(
            self.relative_energy,
            "relative_energy",
            nonnegative=True,
        )
        if distance_m != abs(v_mps) * duration_s:
            raise ValueError("distance_m must match exact solver controls")
        expected_mode = (
            WheelSQPModeV2.FORWARD
            if v_mps > 0.0
            else WheelSQPModeV2.REVERSE
            if v_mps < 0.0
            else WheelSQPModeV2.TURN_LEFT
            if omega_radps > 0.0
            else WheelSQPModeV2.TURN_RIGHT
            if omega_radps < 0.0
            else WheelSQPModeV2.STOP
        )
        if self.mode is not expected_mode:
            raise ValueError("mode must exactly match solver controls")


@dataclass(frozen=True, slots=True)
class WheelSQPCandidateV2:
    candidate_hash: str
    corridor_hash: str
    initial_guess_hash: str
    segments: tuple[_WheelSQPExactSegmentV1, ...]
    objective_value: float
    status: WheelSQPStatusV2

    def __post_init__(self) -> None:
        for name in ("candidate_hash", "corridor_hash", "initial_guess_hash"):
            _exact_hash(getattr(self, name), name)
        if type(self.segments) is not tuple or not self.segments:
            raise TypeError("segments must be a nonempty exact tuple")
        if any(type(segment) is not _WheelSQPExactSegmentV1 for segment in self.segments):
            raise TypeError("segments must contain exact private solver segments")
        for left, right in zip(self.segments, self.segments[1:]):
            if left.end_state != right.start_state:
                raise ValueError("solver candidate segments must be connected")
        _exact_finite_float(self.objective_value, "objective_value", nonnegative=True)
        if type(self.status) is not WheelSQPStatusV2:
            raise TypeError("status must be exact WheelSQPStatusV2")
        if self.status is not WheelSQPStatusV2.FEASIBLE:
            raise ValueError("solver candidate status must be FEASIBLE")


@dataclass(frozen=True, slots=True)
class WheelSQPOptimizationResultV2:
    status: WheelSQPStatusV2
    candidate: WheelSQPCandidateV2 | None
    iteration_count: int
    function_evaluation_count: int
    reason_code: str | None

    def __post_init__(self) -> None:
        if type(self.status) is not WheelSQPStatusV2:
            raise TypeError("status must be exact WheelSQPStatusV2")
        if self.candidate is not None and type(self.candidate) is not WheelSQPCandidateV2:
            raise TypeError("candidate must be exact WheelSQPCandidateV2 or None")
        _exact_nonnegative_int(self.iteration_count, "iteration_count")
        _exact_nonnegative_int(self.function_evaluation_count, "function_evaluation_count")
        if self.reason_code is not None:
            _exact_id(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class WheelL2CounterexampleV2:
    segment_index: int
    candidate_segment_hash: str
    time_fraction_lo: float
    time_fraction_mid: float
    time_fraction_hi: float
    cell: Cell
    terrain_reason: str
    snapshot_hash: str
    candidate_hash: str
    repairable: bool

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.segment_index, "segment_index")
        _exact_hash(self.candidate_segment_hash, "candidate_segment_hash")
        fractions = (
            _exact_finite_float(self.time_fraction_lo, "time_fraction_lo", nonnegative=True),
            _exact_finite_float(self.time_fraction_mid, "time_fraction_mid", nonnegative=True),
            _exact_finite_float(self.time_fraction_hi, "time_fraction_hi", nonnegative=True),
        )
        if not fractions[0] <= fractions[1] <= fractions[2] <= 1.0:
            raise ValueError("time fractions must be ordered within [0, 1]")
        scale = 1 << 24
        if any(not (fraction * scale).is_integer() for fraction in fractions):
            raise ValueError("time fractions must use the depth-24 dyadic grid")
        _exact_cell(self.cell, "cell")
        if self.terrain_reason not in (
            "terrain_out_of_bounds",
            "terrain_unknown",
            "terrain_hard_obstacle",
            "terrain_not_traversable",
            "terrain_slope_exceeded",
        ):
            raise ValueError("terrain_reason must be a stable terrain safety reason")
        _exact_hash(self.snapshot_hash, "snapshot_hash")
        _exact_hash(self.candidate_hash, "candidate_hash")
        _exact_bool(self.repairable, "repairable")
        if self.terrain_reason == "terrain_out_of_bounds" and self.repairable:
            raise ValueError("out-of-bounds counterexamples are not repairable")


@dataclass(frozen=True, slots=True)
class WheelSQPRepairConstraintV1:
    source_candidate_hash: str
    source_segment_hash: str
    source_snapshot_hash: str
    segment_index: int
    cell: Cell
    face: str
    rho_lo: float
    rho_mid: float
    rho_hi: float
    cell_left_x_m: float
    cell_right_x_m: float
    cell_bottom_y_m: float
    cell_top_y_m: float
    clearance_m: float

    def __post_init__(self) -> None:
        _exact_hash(self.source_candidate_hash, "source_candidate_hash")
        _exact_hash(self.source_segment_hash, "source_segment_hash")
        _exact_hash(self.source_snapshot_hash, "source_snapshot_hash")
        _exact_nonnegative_int(self.segment_index, "segment_index")
        _exact_cell(self.cell, "cell")
        if type(self.face) is not str:
            raise TypeError("face must be exact str")
        if self.face not in ("left", "right", "bottom", "top"):
            raise ValueError("face must be left, right, bottom, or top")
        fractions = (
            _exact_finite_float(self.rho_lo, "rho_lo", nonnegative=True),
            _exact_finite_float(self.rho_mid, "rho_mid", nonnegative=True),
            _exact_finite_float(self.rho_hi, "rho_hi", nonnegative=True),
        )
        if not fractions[0] <= fractions[1] <= fractions[2] <= 1.0:
            raise ValueError("repair rho values must be ordered within [0, 1]")
        scale = 1 << 24
        if any(not (fraction * scale).is_integer() for fraction in fractions):
            raise ValueError("repair rho values must use the depth-24 dyadic grid")
        bounds = (
            _exact_finite_float(self.cell_left_x_m, "cell_left_x_m"),
            _exact_finite_float(self.cell_right_x_m, "cell_right_x_m"),
            _exact_finite_float(self.cell_bottom_y_m, "cell_bottom_y_m"),
            _exact_finite_float(self.cell_top_y_m, "cell_top_y_m"),
        )
        if not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
            raise ValueError("repair cell world bounds must be strictly ordered")
        if bounds[1] != bounds[0] + 0.5 or bounds[3] != bounds[2] + 0.5:
            raise ValueError("repair cell bounds must use the fine-grid resolution")
        clearance = _exact_finite_float(
            self.clearance_m,
            "clearance_m",
            nonnegative=True,
        )
        if clearance != 1.0e-4:
            raise ValueError("clearance_m must be exactly 1e-4")

    @property
    def time_fractions(self) -> tuple[float, float, float]:
        return (self.rho_lo, self.rho_mid, self.rho_hi)

    @property
    def cell_bounds_m(self) -> tuple[float, float, float, float]:
        return (
            self.cell_left_x_m,
            self.cell_right_x_m,
            self.cell_bottom_y_m,
            self.cell_top_y_m,
        )


@dataclass(frozen=True, slots=True, init=False)
class WheelTrajectoryL2ResultV2:
    passed: bool
    reason_code: str | None
    route: WheelKinematicRouteV2 | None
    evidence: WheelSQPValidationEvidenceV2 | None
    counterexample: WheelL2CounterexampleV2 | None
    actual_start: PoseStateV2 | None
    actual_goal: PoseStateV2 | None
    goal_position_error_m: float | None
    goal_heading_error_rad: float | None
    checked_cell_count: int
    checked_interval_count: int
    candidate_hash: str | None
    request_hash: str | None
    profile_hash: str | None
    terrain_snapshot_hash: str | None
    capability_revision: str | None
    solver_contract_id: str | None
    canonicalization_id: str | None
    control_slew_id: str | None
    observation_source_id: str | None
    validator_contract_id: str | None
    _request: PlanningRequestV2
    _profile: WheelKinematicSQPProfileV2
    _deadline: PlanningDeadlineV2
    _input_bytes_hash: str
    _promotion_authority: object

    def __init__(self, *, _authority: object, **values: object) -> None:
        if _authority is not _WHEEL_SQP_L2_RESULT_AUTHORITY:
            raise TypeError("WheelTrajectoryL2ResultV2 is validator-only")
        public_fields = (
            "passed",
            "reason_code",
            "route",
            "evidence",
            "counterexample",
            "actual_start",
            "actual_goal",
            "goal_position_error_m",
            "goal_heading_error_rad",
            "checked_cell_count",
            "checked_interval_count",
            "candidate_hash",
            "request_hash",
            "profile_hash",
            "terrain_snapshot_hash",
            "capability_revision",
            "solver_contract_id",
            "canonicalization_id",
            "control_slew_id",
            "observation_source_id",
            "validator_contract_id",
        )
        private_fields = ("request", "profile", "deadline", "input_bytes_hash")
        if set(values) != set(public_fields) | set(private_fields):
            raise TypeError("L2 result fields must match the frozen receipt schema")
        for name in public_fields:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_request", values["request"])
        object.__setattr__(self, "_profile", values["profile"])
        object.__setattr__(self, "_deadline", values["deadline"])
        object.__setattr__(self, "_input_bytes_hash", values["input_bytes_hash"])
        object.__setattr__(self, "_promotion_authority", _WHEEL_SQP_L2_RESULT_AUTHORITY)
        self.__post_init__()

    def __post_init__(self) -> None:
        _exact_bool(self.passed, "passed")
        if self.reason_code is not None:
            _exact_id(self.reason_code, "reason_code")
        if self.route is not None or self.evidence is not None:
            raise ValueError("pre-promotion L2 receipt cannot carry route or evidence")
        if self.counterexample is not None and type(self.counterexample) is not WheelL2CounterexampleV2:
            raise TypeError("counterexample must be exact WheelL2CounterexampleV2 or None")
        for name in ("actual_start", "actual_goal"):
            value = getattr(self, name)
            if value is not None and type(value) is not PoseStateV2:
                raise TypeError(f"{name} must be exact PoseStateV2 or None")
        for name in ("goal_position_error_m", "goal_heading_error_rad"):
            value = getattr(self, name)
            if value is not None:
                _exact_finite_float(value, name, nonnegative=True)
        _exact_nonnegative_int(self.checked_cell_count, "checked_cell_count")
        _exact_nonnegative_int(self.checked_interval_count, "checked_interval_count")
        for name in (
            "candidate_hash",
            "request_hash",
            "profile_hash",
            "terrain_snapshot_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                _exact_hash(value, name)
        if type(self._request) is not PlanningRequestV2:
            raise TypeError("promotion request must be exact PlanningRequestV2")
        if type(self._profile) is not WheelKinematicSQPProfileV2:
            raise TypeError("promotion profile must be exact WheelKinematicSQPProfileV2")
        if type(self._deadline) is not PlanningDeadlineV2:
            raise TypeError("promotion deadline must be exact PlanningDeadlineV2")
        _exact_hash(self._input_bytes_hash, "input_bytes_hash")
        if self._promotion_authority is not _WHEEL_SQP_L2_RESULT_AUTHORITY:
            raise TypeError("invalid L2 promotion authority")
        if self.passed:
            if self.reason_code is not None or self.counterexample is not None:
                raise ValueError("passed receipt cannot carry failure state")
            required = (
                self.actual_start,
                self.actual_goal,
                self.goal_position_error_m,
                self.goal_heading_error_rad,
                self.candidate_hash,
                self.request_hash,
                self.profile_hash,
                self.terrain_snapshot_hash,
                self.capability_revision,
                self.solver_contract_id,
                self.canonicalization_id,
                self.control_slew_id,
                self.observation_source_id,
                self.validator_contract_id,
            )
            if any(value is None for value in required):
                raise ValueError("passed receipt requires all terminal and identity fields")
            _exact_id(self.capability_revision, "capability_revision", "wheel_kinematic_corridor_sqp/v1")
            _exact_id(self.solver_contract_id, "solver_contract_id", WHEEL_KINEMATIC_SOLVER_CONTRACT_V2)
            _exact_id(self.canonicalization_id, "canonicalization_id", WHEEL_KINEMATIC_CANONICALIZATION_V2)
            _exact_id(self.control_slew_id, "control_slew_id", WHEEL_KINEMATIC_CONTROL_SLEW_V2)
            _exact_id(self.observation_source_id, "observation_source_id", WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2)
            _exact_id(self.validator_contract_id, "validator_contract_id", WHEEL_KINEMATIC_L2_VALIDATOR_V2)
        elif self.reason_code is None:
            raise ValueError("failed receipt requires a reason_code")


def _make_wheel_trajectory_l2_result_v2(**values: object) -> WheelTrajectoryL2ResultV2:
    return WheelTrajectoryL2ResultV2(
        _authority=_WHEEL_SQP_L2_RESULT_AUTHORITY,
        **values,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelSQPValidationEvidenceV2(ValidationEvidenceV2):
    route_hash: str
    candidate_hash: str
    request_hash: str
    profile_hash: str
    terrain_snapshot_hash: str
    solver_contract_id: str
    checked_interval_count: int
    checked_cell_count: int
    repair_applied: bool

    def __post_init__(self) -> None:
        ValidationEvidenceV2.__post_init__(self)
        _exact_id(
            self.validator_id,
            "validator_id",
            WHEEL_KINEMATIC_L2_VALIDATOR_V2,
        )
        if self.level is not ValidationLevelV2.L2:
            raise ValueError("wheel SQP validation evidence must be L2")
        for name in (
            "route_hash",
            "candidate_hash",
            "request_hash",
            "profile_hash",
            "terrain_snapshot_hash",
        ):
            _exact_hash(getattr(self, name), name)
        _exact_id(
            self.solver_contract_id,
            "solver_contract_id",
            WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
        )
        _exact_nonnegative_int(self.checked_interval_count, "checked_interval_count")
        _exact_nonnegative_int(self.checked_cell_count, "checked_cell_count")
        _exact_bool(self.repair_applied, "repair_applied")


@dataclass(frozen=True, slots=True)
class WheelSQPPromotionV2:
    route: WheelKinematicRouteV2
    evidence: WheelSQPValidationEvidenceV2

    def __post_init__(self) -> None:
        if type(self.route) is not WheelKinematicRouteV2:
            raise TypeError("route must be exact WheelKinematicRouteV2")
        if type(self.evidence) is not WheelSQPValidationEvidenceV2:
            raise TypeError("evidence must be exact WheelSQPValidationEvidenceV2")
        if self.evidence.passed is not True:
            raise ValueError("promotion evidence must pass")
        if self.route.route_hash != self.evidence.route_hash:
            raise ValueError("promotion route and evidence hashes must match")
        if self.route.source_candidate_hash != self.evidence.candidate_hash:
            raise ValueError("promotion candidate hashes must match")


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelSQPSearchTelemetryV2(SearchTelemetryV2):
    corridor_count: int
    selected_corridor_index: int | None
    corridor_expansions: int
    sqp_iterations: int
    sqp_function_evaluations: int
    l2_interval_count: int
    l2_cell_count: int
    repair_attempted: bool
    decision_hash: str | None

    def __post_init__(self) -> None:
        SearchTelemetryV2.__post_init__(self)
        _exact_nonnegative_int(self.corridor_count, "corridor_count")
        if self.selected_corridor_index is not None:
            _exact_nonnegative_int(self.selected_corridor_index, "selected_corridor_index")
            if self.selected_corridor_index >= self.corridor_count:
                raise ValueError("selected_corridor_index must be within corridor_count")
        for name in (
            "corridor_expansions",
            "sqp_iterations",
            "sqp_function_evaluations",
            "l2_interval_count",
            "l2_cell_count",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        _exact_bool(self.repair_attempted, "repair_attempted")
        if self.decision_hash is not None:
            _exact_hash(self.decision_hash, "decision_hash")


@dataclass(frozen=True, slots=True)
class WheelSQPResourceLedgerV1:
    reserve_s: float
    remaining_s: float
    accepted: bool
    reason_code: str | None
    model_id: str = WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2

    def __post_init__(self) -> None:
        reserve = _exact_finite_float(self.reserve_s, "reserve_s", nonnegative=True)
        remaining = _exact_finite_float(self.remaining_s, "remaining_s", nonnegative=True)
        _exact_bool(self.accepted, "accepted")
        _exact_id(self.model_id, "model_id", WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2)
        if self.reason_code is not None:
            _exact_id(self.reason_code, "reason_code")
        expected_reason = None if self.accepted else (
            "planning_deadline_expired" if remaining == 0.0 else "wheel_sqp_resource_budget_exceeded"
        )
        if self.reason_code != expected_reason:
            raise ValueError("resource ledger reason_code must match accepted and remaining_s")
        if self.accepted is not (reserve < remaining):
            raise ValueError("resource ledger accepted must match strict reserve comparison")


@dataclass(frozen=True, slots=True, init=False)
class WheelSQPResourceEstimateV1:
    segment_count: int
    variable_count: int
    equality_count: int
    base_inequality_count: int
    terrain_inequality_count: int
    total_constraint_count: int
    broadphase_cell_bound: int
    interval_record_bound: int
    encoded_state_bound: int
    encoded_scalar_bound: int
    decision_bytes: int
    jacobian_bytes: int
    solver_bytes: int
    l2_queue_bytes: int
    codec_bytes: int
    required_bytes: int
    post_solver_reserve: WheelSQPResourceLedgerV1
    estimate_id: str = WHEEL_SQP_ATTEMPT_RESOURCE_ESTIMATE_V1

    def __init__(self, *, _authority: object, **values: object) -> None:
        if _authority is not _WHEEL_SQP_RESOURCE_ESTIMATE_AUTHORITY:
            raise TypeError("WheelSQPResourceEstimateV1 is estimator-only")
        field_names = (
            "segment_count",
            "variable_count",
            "equality_count",
            "base_inequality_count",
            "terrain_inequality_count",
            "total_constraint_count",
            "broadphase_cell_bound",
            "interval_record_bound",
            "encoded_state_bound",
            "encoded_scalar_bound",
            "decision_bytes",
            "jacobian_bytes",
            "solver_bytes",
            "l2_queue_bytes",
            "codec_bytes",
            "required_bytes",
            "post_solver_reserve",
        )
        unexpected = set(values) - set(field_names) - {"estimate_id"}
        missing = set(field_names) - set(values)
        if unexpected or missing:
            raise TypeError("resource estimate fields must match the frozen schema")
        for name in field_names:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "estimate_id",
            values.get("estimate_id", WHEEL_SQP_ATTEMPT_RESOURCE_ESTIMATE_V1),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        integer_fields = (
            "segment_count",
            "variable_count",
            "equality_count",
            "base_inequality_count",
            "terrain_inequality_count",
            "total_constraint_count",
            "broadphase_cell_bound",
            "interval_record_bound",
            "encoded_state_bound",
            "encoded_scalar_bound",
            "decision_bytes",
            "jacobian_bytes",
            "solver_bytes",
            "l2_queue_bytes",
            "codec_bytes",
            "required_bytes",
        )
        for name in integer_fields:
            _exact_nonnegative_int(getattr(self, name), name)
        if not 1 <= self.segment_count <= 48:
            raise ValueError("segment_count must be within the v1 SQP bound")
        if self.variable_count != 6 * self.segment_count:
            raise ValueError("variable_count must equal 6N")
        if self.equality_count != 3 * self.segment_count:
            raise ValueError("equality_count must equal 3N")
        if self.base_inequality_count != 4 * self.segment_count - 1:
            raise ValueError("base_inequality_count must equal 4N-1")
        if self.terrain_inequality_count != 5 * self.segment_count:
            raise ValueError("terrain_inequality_count must equal 5N")
        legacy_constraint_count = 12 * self.segment_count - 1
        if self.total_constraint_count not in (
            legacy_constraint_count,
            legacy_constraint_count + 3,
        ):
            raise ValueError("total_constraint_count must equal 12N-1 or 12N+2")
        if self.required_bytes != (
            self.decision_bytes
            + self.jacobian_bytes
            + self.solver_bytes
            + self.l2_queue_bytes
            + self.codec_bytes
        ):
            raise ValueError("required_bytes must equal the frozen component sum")
        if self.broadphase_cell_bound > 1_000_000:
            raise ValueError("broadphase_cell_bound exceeds the frozen cap")
        if self.interval_record_bound > 262_144:
            raise ValueError("interval_record_bound exceeds the frozen cap")
        interval_factor = (1 << 25) - 1
        expected_interval_bound = (
            262_144
            if self.broadphase_cell_bound > 262_144 // interval_factor
            else self.broadphase_cell_bound * interval_factor
        )
        if self.interval_record_bound != expected_interval_bound:
            raise ValueError("interval_record_bound does not match broadphase saturation")
        encoded_state_base = 3 + 4 * self.segment_count
        encoded_state_delta = self.encoded_state_bound - encoded_state_base
        if encoded_state_delta < 0 or encoded_state_delta % 2 != 0:
            raise ValueError("encoded_state_bound does not encode an exact sample count")
        sample_count = encoded_state_delta // 2
        if self.encoded_scalar_bound != (
            14 + 24 * self.segment_count + 6 * sample_count
        ):
            raise ValueError("encoded_scalar_bound does not match encoded_state_bound")
        if self.decision_bytes != 8 * self.variable_count:
            raise ValueError("decision_bytes does not match the frozen formula")
        if self.jacobian_bytes != (
            8 * self.total_constraint_count * self.variable_count
        ):
            raise ValueError("jacobian_bytes does not match the frozen formula")
        if self.solver_bytes != 16_777_216:
            raise ValueError("solver_bytes does not match the frozen reservation")
        if self.l2_queue_bytes != 96 * self.interval_record_bound:
            raise ValueError("l2_queue_bytes does not match the frozen formula")
        if self.codec_bytes != (
            64 * self.encoded_state_bound + 16 * self.encoded_scalar_bound
        ):
            raise ValueError("codec_bytes does not match the frozen formula")
        if type(self.post_solver_reserve) is not WheelSQPResourceLedgerV1:
            raise TypeError("post_solver_reserve must be exact WheelSQPResourceLedgerV1")
        if (
            not self.post_solver_reserve.accepted
            or self.post_solver_reserve.reason_code is not None
        ):
            raise ValueError("post_solver_reserve must be accepted")
        expected_reserve = L2ReserveModelV1().reserve_s(
            segment_count=self.segment_count,
            broadphase_cell_bound=self.broadphase_cell_bound,
            interval_record_bound=self.interval_record_bound,
            encoded_state_bound=self.encoded_state_bound,
            encoded_scalar_bound=self.encoded_scalar_bound,
            max_segments=48,
            max_l2_candidate_cells=1_000_000,
            max_l2_interval_records=262_144,
            max_encoded_state_bound=self.encoded_state_bound,
            max_encoded_scalar_bound=self.encoded_scalar_bound,
        )
        if self.post_solver_reserve.reserve_s != expected_reserve:
            raise ValueError("post_solver_reserve does not bind the estimate bounds")
        _exact_id(
            self.estimate_id,
            "estimate_id",
            WHEEL_SQP_ATTEMPT_RESOURCE_ESTIMATE_V1,
        )


def _make_wheel_sqp_resource_estimate_v1(
    **values: object,
) -> WheelSQPResourceEstimateV1:
    return WheelSQPResourceEstimateV1(
        _authority=_WHEEL_SQP_RESOURCE_ESTIMATE_AUTHORITY,
        **values,
    )


@dataclass(frozen=True, slots=True)
class L2ReserveModelV1:
    model_id: str = WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2

    def __post_init__(self) -> None:
        _exact_id(self.model_id, "model_id", WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2)

    def reserve_s(
        self,
        *,
        segment_count: int,
        broadphase_cell_bound: int,
        interval_record_bound: int,
        encoded_state_bound: int,
        encoded_scalar_bound: int,
        max_segments: int,
        max_l2_candidate_cells: int,
        max_l2_interval_records: int,
        max_encoded_state_bound: int,
        max_encoded_scalar_bound: int,
    ) -> float:
        values = {
            "segment_count": segment_count,
            "broadphase_cell_bound": broadphase_cell_bound,
            "interval_record_bound": interval_record_bound,
            "encoded_state_bound": encoded_state_bound,
            "encoded_scalar_bound": encoded_scalar_bound,
            "max_segments": max_segments,
            "max_l2_candidate_cells": max_l2_candidate_cells,
            "max_l2_interval_records": max_l2_interval_records,
            "max_encoded_state_bound": max_encoded_state_bound,
            "max_encoded_scalar_bound": max_encoded_scalar_bound,
        }
        for name, value in values.items():
            _exact_nonnegative_int(value, name)
        reserve = (
            0.015
            + 0.00025 * min(segment_count, max_segments)
            + 0.000002 * min(broadphase_cell_bound, max_l2_candidate_cells)
            + 0.000001 * min(interval_record_bound, max_l2_interval_records)
            + 0.000002 * min(encoded_state_bound, max_encoded_state_bound)
            + 0.000001 * min(encoded_scalar_bound, max_encoded_scalar_bound)
        )
        if not isfinite(reserve) or reserve < 0.0:
            raise ValueError("wheel_sqp_resource_budget_exceeded")
        return reserve

    def assess(
        self,
        deadline: PlanningDeadlineV2,
        **reserve_kwargs: int,
    ) -> WheelSQPResourceLedgerV1:
        if type(deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")
        remaining = deadline.remaining_s
        if remaining <= 0.0:
            return WheelSQPResourceLedgerV1(0.0, 0.0, False, "planning_deadline_expired")
        try:
            reserve = self.reserve_s(**reserve_kwargs)
        except (TypeError, ValueError, OverflowError):
            return WheelSQPResourceLedgerV1(
                remaining,
                remaining,
                False,
                "wheel_sqp_resource_budget_exceeded",
            )
        if reserve >= remaining:
            return WheelSQPResourceLedgerV1(
                reserve,
                remaining,
                False,
                "wheel_sqp_resource_budget_exceeded",
            )
        return WheelSQPResourceLedgerV1(reserve, remaining, True, None)
