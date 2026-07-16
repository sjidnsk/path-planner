from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isclose, isfinite, pi, remainder
from numbers import Real

import numpy as np

from path_planner.core import Cell
from path_planner.search import (
    MotionPrimitive,
    Pose2D,
    PoseTransition,
    replay_motion_primitive,
)
from path_planner.search.hybrid_astar import MAX_REPLAY_STEPS
from path_planner.v2.contracts import (
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.geometry import (
    conservative_wheel_pose_cells,
    conservative_wheel_sweep_cells,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelProfileV2
from path_planner.v2.providers.wheel import WheelMotionPrimitiveV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    SafetyQueryV2,
    SYNTHETIC_TERRAIN_SOURCE_KIND_V2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


WHEEL_TRANSITION_VALIDATOR_ID_V2 = "path-planner-v2-wheel-transition-l2/v1"
WHEEL_ROUTE_VALIDATOR_ID_V2 = "path-planner-v2-wheel-route-l2/v1"
_MAX_DECLARED_ROUTE_STATES = MAX_REPLAY_STEPS + 1
_EXPECTED_CONTRACT_EXCEPTIONS = (
    TypeError,
    ValueError,
    OverflowError,
    AttributeError,
    IndexError,
)

_TERRAIN_FAILURE_PRIORITY = {
    "terrain_out_of_bounds": 0,
    "terrain_unknown": 1,
    "terrain_hard_obstacle": 2,
    "terrain_not_traversable": 3,
    "terrain_slope_exceeded": 4,
}
_TERRAIN_QUERY_REASON_CODES = frozenset({"terrain_safe", *_TERRAIN_FAILURE_PRIORITY})
_COMMON_FAILURE_REASON_CODES = frozenset(
    {
        "planning_deadline_expired",
        "planning_deadline_contract_mismatch",
        "terrain_out_of_bounds",
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
        "terrain_snapshot_hash_mismatch",
        "terrain_query_contract_mismatch",
        "wheel_profile_contract_mismatch",
        "primitive_structure_mismatch",
        "primitive_flag_mismatch",
        "primitive_duration_mismatch",
        "primitive_zero_motion_mismatch",
        "primitive_replay_mismatch",
        "wheel_reverse_disabled",
        "wheel_turn_in_place_disabled",
        "wheel_speed_limit_exceeded",
        "wheel_angular_speed_limit_exceeded",
        "wheel_min_turning_radius_violated",
    }
)
_VALIDATOR_REASON_CODES = {
    WHEEL_TRANSITION_VALIDATOR_ID_V2: frozenset(
        {"transition_l2_valid", *_COMMON_FAILURE_REASON_CODES}
    ),
    WHEEL_ROUTE_VALIDATOR_ID_V2: frozenset(
        {
            "route_l2_valid",
            *_COMMON_FAILURE_REASON_CODES,
            "terrain_snapshot_identity_mismatch",
            "wheel_platform_identity_mismatch",
            "wheel_profile_identity_mismatch",
            "route_incomplete",
            "route_structure_mismatch",
            "route_start_mismatch",
            "route_start_contract_mismatch",
            "route_connectivity_mismatch",
            "route_state_budget_exceeded",
            "route_goal_tolerance_exceeded",
            "route_goal_contract_mismatch",
            "planning_request_contract_mismatch",
            "wheel_primitive_type_mismatch",
            "primitive_hold_contract_mismatch",
            "primitive_sweep_unavailable",
        }
    ),
}
_PASS_REASON_BY_VALIDATOR = {
    WHEEL_TRANSITION_VALIDATOR_ID_V2: "transition_l2_valid",
    WHEEL_ROUTE_VALIDATOR_ID_V2: "route_l2_valid",
}


@dataclass(frozen=True, slots=True)
class WheelValidationResultV2:
    evidence: ValidationEvidenceV2
    reason_code: str
    timed_out: bool
    failed_cell: Cell | None
    failed_primitive_index: int | None
    checked_cell_count: int

    def __post_init__(self) -> None:
        if type(self.evidence) is not ValidationEvidenceV2:
            raise TypeError("evidence must be exact ValidationEvidenceV2")
        if self.evidence.level is not ValidationLevelV2.L2:
            raise ValueError("evidence must be L2")
        validator_id = self.evidence.validator_id
        if type(validator_id) is not str:
            raise TypeError("evidence validator_id must be exact str")
        if validator_id not in _VALIDATOR_REASON_CODES:
            raise ValueError("evidence validator_id must be a stable wheel L2 validator")
        if type(self.reason_code) is not str or not self.reason_code.strip():
            raise ValueError("reason_code must be a nonempty string")
        if self.reason_code not in _VALIDATOR_REASON_CODES[validator_id]:
            raise ValueError("reason_code is not valid for the wheel L2 validator")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be bool")
        if type(self.evidence.passed) is not bool:
            raise TypeError("evidence passed must be bool")
        if (
            type(self.evidence.checks) is not tuple
            or len(self.evidence.checks) != 1
            or type(self.evidence.checks[0]) is not str
        ):
            raise TypeError("evidence checks must be one exact string in a tuple")
        if self.failed_cell is not None:
            if type(self.failed_cell) is not Cell:
                raise TypeError("failed_cell must be exact Cell or None")
            try:
                exact_coordinates = (
                    type(self.failed_cell.x) is int
                    and type(self.failed_cell.y) is int
                )
            except AttributeError:
                exact_coordinates = False
            if not exact_coordinates:
                raise ValueError("failed_cell coordinates must be exact integers")
        if self.failed_primitive_index is not None:
            if type(self.failed_primitive_index) is not int:
                raise TypeError("failed_primitive_index must be an integer or None")
            if self.failed_primitive_index < 0:
                raise ValueError("failed_primitive_index must be nonnegative")
        if type(self.checked_cell_count) is not int:
            raise TypeError("checked_cell_count must be an integer")
        if self.checked_cell_count < 0:
            raise ValueError("checked_cell_count must be nonnegative")
        if self.timed_out is not (self.reason_code == "planning_deadline_expired"):
            raise ValueError("timeout fields must agree")
        if self.timed_out and (
            self.failed_cell is not None or self.failed_primitive_index is not None
        ):
            raise ValueError("timeout result must not carry failure metadata")
        if self.evidence.checks != (self.reason_code,):
            raise ValueError("evidence checks must contain the exact reason_code")
        if self.evidence.passed is not (
            self.reason_code == _PASS_REASON_BY_VALIDATOR[validator_id]
        ):
            raise ValueError("evidence passed must agree with the reason_code")
        if self.failed_cell is not None and self.failed_primitive_index is None:
            raise ValueError("failed_cell requires failed_primitive_index")
        if self.reason_code in _TERRAIN_FAILURE_PRIORITY:
            if self.failed_cell is None or self.failed_primitive_index is None:
                raise ValueError(
                    "terrain failure requires failed_cell and failed_primitive_index"
                )
            if self.checked_cell_count < 1:
                raise ValueError("terrain failure requires at least one checked cell")
        elif self.failed_cell is not None:
            raise ValueError("non-terrain failure must not carry failed_cell")
        if self.evidence.passed:
            if self.timed_out:
                raise ValueError("passing evidence cannot be timed out")
            if self.failed_cell is not None or self.failed_primitive_index is not None:
                raise ValueError("passing evidence cannot carry failure metadata")
            if self.checked_cell_count == 0:
                raise ValueError("passing L2 evidence must include checked cells")


@dataclass(frozen=True, slots=True)
class _AuditedMotion:
    index: int
    start: Pose2D | None
    control: MotionPrimitive | None
    replay: PoseTransition | None
    hold: bool
    actual_end: Pose2D | None
    route_state_count: int
    reason_code: str | None


_MISSING_FIELD = object()


@dataclass(frozen=True, slots=True)
class _PublicTransitionFields:
    start: object
    primitive: object
    samples: object
    end: object
    distance_m: object
    absolute_heading_change_rad: object


@dataclass(frozen=True, slots=True)
class _MotionControlFields:
    name: object
    v_mps: object
    omega_radps: object
    duration_s: object
    reverse: object
    turn_in_place: object


@dataclass(frozen=True, slots=True)
class _TypedPrimitiveFields:
    kind: object
    start_state: object
    end_state: object
    duration_s: object
    distance_m: object
    energy_cost: object
    observation_contribution: object
    validation_level: object
    control_name: object
    samples: object
    v_mps: object
    omega_radps: object
    reverse: object
    turn_in_place: object


def _read_untrusted_field(instance: object, name: str) -> object:
    try:
        return getattr(instance, name)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return _MISSING_FIELD


def _read_public_transition_fields(
    transition: PoseTransition,
) -> _PublicTransitionFields:
    return _PublicTransitionFields(
        start=_read_untrusted_field(transition, "start"),
        primitive=_read_untrusted_field(transition, "primitive"),
        samples=_read_untrusted_field(transition, "samples"),
        end=_read_untrusted_field(transition, "end"),
        distance_m=_read_untrusted_field(transition, "distance_m"),
        absolute_heading_change_rad=_read_untrusted_field(
            transition,
            "absolute_heading_change_rad",
        ),
    )


def _read_motion_control_fields(primitive: MotionPrimitive) -> _MotionControlFields:
    return _MotionControlFields(
        name=_read_untrusted_field(primitive, "name"),
        v_mps=_read_untrusted_field(primitive, "v_mps"),
        omega_radps=_read_untrusted_field(primitive, "omega_radps"),
        duration_s=_read_untrusted_field(primitive, "duration_s"),
        reverse=_read_untrusted_field(primitive, "reverse"),
        turn_in_place=_read_untrusted_field(primitive, "turn_in_place"),
    )


def _read_typed_primitive_fields(
    primitive: WheelMotionPrimitiveV2,
) -> _TypedPrimitiveFields:
    return _TypedPrimitiveFields(
        kind=_read_untrusted_field(primitive, "kind"),
        start_state=_read_untrusted_field(primitive, "start_state"),
        end_state=_read_untrusted_field(primitive, "end_state"),
        duration_s=_read_untrusted_field(primitive, "duration_s"),
        distance_m=_read_untrusted_field(primitive, "distance_m"),
        energy_cost=_read_untrusted_field(primitive, "energy_cost"),
        observation_contribution=_read_untrusted_field(
            primitive,
            "observation_contribution",
        ),
        validation_level=_read_untrusted_field(primitive, "validation_level"),
        control_name=_read_untrusted_field(primitive, "control_name"),
        samples=_read_untrusted_field(primitive, "samples"),
        v_mps=_read_untrusted_field(primitive, "v_mps"),
        omega_radps=_read_untrusted_field(primitive, "omega_radps"),
        reverse=_read_untrusted_field(primitive, "reverse"),
        turn_in_place=_read_untrusted_field(primitive, "turn_in_place"),
    )


def _fields_missing(*values: object) -> bool:
    return any(value is _MISSING_FIELD for value in values)


@dataclass(frozen=True, slots=True)
class _TerrainFailure:
    reason_code: str
    cell: Cell
    primitive_index: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            _TERRAIN_FAILURE_PRIORITY[self.reason_code],
            self.primitive_index,
            self.cell.y,
            self.cell.x,
        )


class _DeadlineContractError(RuntimeError):
    pass


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


def _deadline_expired(deadline: PlanningDeadlineV2) -> bool:
    try:
        if (
            type(deadline.started_monotonic_s) is not float
            or type(deadline.deadline_monotonic_s) is not float
        ):
            raise TypeError("planning deadline timestamps must be exact floats")
        started = _finite_real(
            deadline.started_monotonic_s,
            "deadline started_monotonic_s",
        )
        cutoff = _finite_real(
            deadline.deadline_monotonic_s,
            "deadline deadline_monotonic_s",
        )
        if cutoff < started or not callable(deadline._monotonic_clock):
            raise ValueError("planning deadline fields are inconsistent")
        expired = deadline.expired
    except _EXPECTED_CONTRACT_EXCEPTIONS as exc:
        raise _DeadlineContractError("planning deadline contract mismatch") from exc
    if type(expired) is not bool:
        raise _DeadlineContractError("planning deadline must return exact bool")
    return expired


def _raise_if_deadline_expired(deadline: PlanningDeadlineV2) -> None:
    if _deadline_expired(deadline):
        raise TimeoutError("wheel validation deadline expired")


def _deadline_contract(
    validator_id: str,
    checked_cell_count: int,
    *,
    failed_primitive_index: int | None = None,
) -> WheelValidationResultV2:
    return _result(
        validator_id,
        "planning_deadline_contract_mismatch",
        failed_primitive_index=failed_primitive_index,
        checked_cell_count=checked_cell_count,
    )


def _pose_from_public_pose(pose: object) -> Pose2D:
    if type(pose) is not Pose2D:
        raise TypeError("pose must be exact Pose2D")
    return Pose2D(
        _finite_real(pose.x_m, "pose x_m"),
        _finite_real(pose.y_m, "pose y_m"),
        _finite_real(pose.theta_rad, "pose theta_rad"),
    )


def _poses_equal(left: Pose2D, right: Pose2D) -> bool:
    return (
        left.x_m == right.x_m
        and left.y_m == right.y_m
        and left.theta_rad == right.theta_rad
    )


_SNAPSHOT_LAYER_DTYPES = (
    ("elevation_m", np.dtype("<f8")),
    ("slope_deg", np.dtype("<f8")),
    ("traversable_mask", np.dtype(np.bool_)),
    ("hard_obstacle_mask", np.dtype(np.bool_)),
    ("observed_mask", np.dtype(np.bool_)),
    ("confidence", np.dtype("<f8")),
)


def _has_immutable_bytes_storage(layer: np.ndarray) -> bool:
    current: object = layer
    while type(current) is np.ndarray:
        if current.flags.writeable:
            return False
        current = current.base
    return type(current) is bytes


def _reaudit_terrain_snapshot(
    snapshot: object,
    deadline: PlanningDeadlineV2,
) -> bool:
    if type(snapshot) is not TerrainSnapshotV2:
        return False
    try:
        _raise_if_deadline_expired(deadline)
        geometry = snapshot.geometry
        provenance = snapshot.provenance
        if type(geometry) is not FineGridGeometryV2:
            return False
        if type(provenance) is not TerrainProvenanceV2:
            return False
        if (
            type(geometry.width) is not int
            or type(geometry.height) is not int
            or type(geometry.origin) is not tuple
            or len(geometry.origin) != 2
            or any(type(value) is not float for value in geometry.origin)
            or type(geometry.frame_id) is not str
            or type(geometry.resolution_m) is not float
        ):
            return False
        replace(geometry)
        if (
            type(provenance.source_kind) is not str
            or type(provenance.source_id) is not str
            or type(provenance.source_hash) is not str
            or type(provenance.physical_obstacle_cells_written) is not bool
            or type(provenance.details) is not tuple
            or not provenance.source_kind.strip()
            or not provenance.source_id.strip()
            or not provenance.source_hash.strip()
            or (
                provenance.source_kind == SYNTHETIC_TERRAIN_SOURCE_KIND_V2
                and provenance.physical_obstacle_cells_written is not False
            )
        ):
            return False
        previous_key: str | None = None
        for detail in provenance.details:
            _raise_if_deadline_expired(deadline)
            if type(detail) is not tuple or len(detail) != 2:
                return False
            key, value = detail
            if (
                type(key) is not str
                or not key.strip()
                or (previous_key is not None and key <= previous_key)
            ):
                return False
            if value is not None and type(value) not in (str, bool, int, float):
                return False
            if type(value) is float and not isfinite(value):
                return False
            previous_key = key
        shape = geometry.shape
        layers: dict[str, np.ndarray] = {}
        for name, dtype in _SNAPSHOT_LAYER_DTYPES:
            _raise_if_deadline_expired(deadline)
            layer = getattr(snapshot, name)
            if (
                type(layer) is not np.ndarray
                or layer.dtype != dtype
                or layer.ndim != 2
                or layer.shape != shape
                or not layer.flags.c_contiguous
                or not _has_immutable_bytes_storage(layer)
            ):
                return False
            layers[name] = layer
            if dtype == np.dtype("<f8") and not bool(np.all(np.isfinite(layer))):
                return False
        if bool(np.any(layers["slope_deg"] < 0.0)):
            return False
        _raise_if_deadline_expired(deadline)
        confidence = layers["confidence"]
        if bool(np.any((confidence < 0.0) | (confidence > 1.0))):
            return False
        _raise_if_deadline_expired(deadline)
        if bool(
            np.any(layers["traversable_mask"] & layers["hard_obstacle_mask"])
        ):
            return False
        _raise_if_deadline_expired(deadline)
    except _DeadlineContractError:
        raise
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return False
    return True


def _safe_snapshot_hash(
    snapshot: object,
    deadline: PlanningDeadlineV2,
) -> str | None:
    if not _reaudit_terrain_snapshot(snapshot, deadline):
        return None
    try:
        _raise_if_deadline_expired(deadline)
        digest = snapshot_hash(snapshot)
        _raise_if_deadline_expired(deadline)
    except _DeadlineContractError:
        raise
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return None
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return digest


def _pose_from_state(state: object) -> Pose2D:
    if type(state) is not PoseStateV2:
        raise TypeError("state must be exact PoseStateV2")
    return Pose2D(
        _finite_real(state.x_m, "state x_m"),
        _finite_real(state.y_m, "state y_m"),
        _finite_real(state.heading_rad, "state heading_rad"),
    )


def _validate_entry_types(
    anchor: object,
    wheel_profile: object,
    deadline: object,
) -> None:
    if type(anchor) is not FineSafetyAnchorV2:
        raise TypeError("anchor must be exact FineSafetyAnchorV2")
    if type(wheel_profile) is not WheelProfileV2:
        raise TypeError("wheel_profile must be exact WheelProfileV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")


def _reaudit_wheel_profile(
    wheel_profile: WheelProfileV2,
) -> WheelProfileV2 | None:
    try:
        if type(wheel_profile.profile) is not PlatformProfileV2:
            return None
        profile = replace(wheel_profile.profile)
        return replace(wheel_profile, profile=profile)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return None


def _route_state_budget(request: PlanningRequestV2) -> int | None:
    try:
        budget = request.resource_budget
        if type(budget) is not ResourceBudgetV2:
            return None
        max_route_states = budget.max_route_states
    except AttributeError:
        return None
    if type(max_route_states) is not int:
        return None
    if max_route_states < 0:
        return None
    return max_route_states


def _declared_route_state_overflow(
    primitives: tuple[object, ...],
    max_route_states: int,
    deadline: PlanningDeadlineV2,
) -> int | None:
    _raise_if_deadline_expired(deadline)
    if len(primitives) > MAX_REPLAY_STEPS:
        return MAX_REPLAY_STEPS
    hard_limit = min(max_route_states, _MAX_DECLARED_ROUTE_STATES)
    declared_route_states = 0
    for index, primitive in enumerate(primitives):
        _raise_if_deadline_expired(deadline)
        declared_count = 0
        if type(primitive) is WheelMotionPrimitiveV2:
            samples = _read_untrusted_field(primitive, "samples")
            if type(samples) is tuple:
                declared_count = len(samples)
                if declared_count > _MAX_DECLARED_ROUTE_STATES:
                    return index
        declared_route_states += (
            declared_count if index == 0 else max(0, declared_count - 1)
        )
        if declared_route_states > hard_limit:
            return index
    return None


def _result(
    validator_id: str,
    reason_code: str,
    *,
    passed: bool = False,
    timed_out: bool = False,
    failed_cell: Cell | None = None,
    failed_primitive_index: int | None = None,
    checked_cell_count: int = 0,
) -> WheelValidationResultV2:
    return WheelValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=validator_id,
            level=ValidationLevelV2.L2,
            passed=passed,
            checks=(reason_code,),
        ),
        reason_code=reason_code,
        timed_out=timed_out,
        failed_cell=failed_cell,
        failed_primitive_index=failed_primitive_index,
        checked_cell_count=checked_cell_count,
    )


def _timeout(validator_id: str, checked_cell_count: int) -> WheelValidationResultV2:
    return _result(
        validator_id,
        "planning_deadline_expired",
        timed_out=True,
        checked_cell_count=checked_cell_count,
    )


def _new_control(
    name: object,
    v_mps: object,
    omega_radps: object,
    duration_s: object,
    reverse: object,
    turn_in_place: object,
    wheel_profile: WheelProfileV2,
) -> tuple[MotionPrimitive | None, str | None]:
    try:
        speed = _finite_real(v_mps, "v_mps")
        angular_speed = _finite_real(omega_radps, "omega_radps")
        duration = _finite_real(duration_s, "duration_s")
    except (TypeError, ValueError):
        return None, "primitive_structure_mismatch"
    if duration <= 0.0:
        return None, "primitive_duration_mismatch"

    expected_reverse = speed < 0.0
    expected_turn = speed == 0.0 and angular_speed != 0.0
    name_valid = type(name) is str and bool(name.strip())
    flags_valid = type(reverse) is bool and type(turn_in_place) is bool
    normalized_name = name if name_valid else "invalid_wheel_control"
    normalized_reverse = reverse if type(reverse) is bool else expected_reverse
    normalized_turn = turn_in_place if type(turn_in_place) is bool else expected_turn

    control = MotionPrimitive(
        normalized_name,
        speed,
        angular_speed,
        duration,
        reverse=normalized_reverse,
        turn_in_place=normalized_turn,
    )
    reason: str | None = None
    if not name_valid:
        reason = "primitive_structure_mismatch"
    if not flags_valid:
        reason = reason or "primitive_flag_mismatch"
    if duration != wheel_profile.primitive_duration_s:
        reason = reason or "primitive_duration_mismatch"
    if speed == 0.0 and angular_speed == 0.0:
        reason = reason or "primitive_zero_motion_mismatch"
    if (
        flags_valid
        and (reverse is not expected_reverse or turn_in_place is not expected_turn)
    ):
        reason = reason or "primitive_flag_mismatch"
    if expected_reverse and not wheel_profile.reverse_enabled:
        reason = reason or "wheel_reverse_disabled"
    if expected_turn and not wheel_profile.turn_in_place_enabled:
        reason = reason or "wheel_turn_in_place_disabled"
    if abs(speed) > wheel_profile.max_speed_mps:
        reason = reason or "wheel_speed_limit_exceeded"
    if abs(angular_speed) > wheel_profile.max_angular_speed_radps:
        reason = reason or "wheel_angular_speed_limit_exceeded"
    if speed != 0.0 and angular_speed != 0.0:
        turning_radius = abs(speed / angular_speed)
        if turning_radius < wheel_profile.min_turning_radius_m:
            reason = reason or "wheel_min_turning_radius_violated"
    return control, reason


def _replay(
    start: Pose2D,
    control: MotionPrimitive,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> PoseTransition:
    return replay_motion_primitive(
        start,
        control,
        wheel_profile.integration_dt_s,
        deadline_checker=lambda: _deadline_expired(deadline),
    )


def _audit_public_transition(
    transition: PoseTransition,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> _AuditedMotion:
    fields = _read_public_transition_fields(transition)
    transition_fields_missing = _fields_missing(
        fields.start,
        fields.primitive,
        fields.samples,
        fields.end,
        fields.distance_m,
        fields.absolute_heading_change_rad,
    )
    try:
        start = _pose_from_public_pose(fields.start)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return _AuditedMotion(
            0,
            None,
            None,
            None,
            False,
            None,
            0,
            "primitive_structure_mismatch",
        )

    primitive = fields.primitive
    if type(primitive) is not MotionPrimitive:
        return _AuditedMotion(
            0,
            start,
            None,
            None,
            False,
            None,
            0,
            "primitive_structure_mismatch",
        )
    control_fields = _read_motion_control_fields(primitive)
    control_fields_missing = _fields_missing(
        control_fields.name,
        control_fields.v_mps,
        control_fields.omega_radps,
        control_fields.duration_s,
        control_fields.reverse,
        control_fields.turn_in_place,
    )
    control, reason = _new_control(
        control_fields.name,
        control_fields.v_mps,
        control_fields.omega_radps,
        control_fields.duration_s,
        control_fields.reverse,
        control_fields.turn_in_place,
        wheel_profile,
    )
    if transition_fields_missing or control_fields_missing:
        reason = "primitive_structure_mismatch"
    if control is None:
        return _AuditedMotion(0, start, None, None, False, None, 0, reason)
    replay = _replay(start, control, wheel_profile, deadline)
    try:
        declared_samples = fields.samples
        exact_samples = (
            type(declared_samples) is tuple
            and len(declared_samples) == len(replay.samples)
        )
        if exact_samples:
            for declared, expected in zip(
                declared_samples,
                replay.samples,
                strict=True,
            ):
                _raise_if_deadline_expired(deadline)
                try:
                    normalized = _pose_from_public_pose(declared)
                except _EXPECTED_CONTRACT_EXCEPTIONS:
                    exact_samples = False
                    break
                if not _poses_equal(normalized, expected):
                    exact_samples = False
                    break
        try:
            declared_end = _pose_from_public_pose(fields.end)
            exact_end = _poses_equal(declared_end, replay.end)
        except _EXPECTED_CONTRACT_EXCEPTIONS:
            exact_end = False
        distance = _finite_real(fields.distance_m, "transition distance_m")
        heading_change = _finite_real(
            fields.absolute_heading_change_rad,
            "transition absolute_heading_change_rad",
        )
        replay_matches = (
            exact_samples
            and exact_end
            and isclose(distance, replay.distance_m, rel_tol=1.0e-12, abs_tol=1.0e-12)
            and isclose(
                heading_change,
                replay.absolute_heading_change_rad,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
    except _DeadlineContractError:
        raise
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        replay_matches = False
    if not replay_matches:
        reason = reason or "primitive_replay_mismatch"
    return _AuditedMotion(
        0,
        start,
        control,
        replay,
        False,
        replay.end,
        len(replay.samples),
        reason,
    )


def _audit_hold(
    fields: _TypedPrimitiveFields,
    index: int,
    deadline: PlanningDeadlineV2,
) -> _AuditedMotion:
    try:
        start = _pose_from_state(fields.start_state)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return _AuditedMotion(
            index,
            None,
            None,
            None,
            True,
            None,
            0,
            "primitive_hold_contract_mismatch",
        )
    try:
        _raise_if_deadline_expired(deadline)
        samples = fields.samples
        sample = (
            _pose_from_state(samples[0])
            if type(samples) is tuple and len(samples) == 1
            else None
        )
        end = _pose_from_state(fields.end_state)
        valid = (
            fields.kind is PrimitiveKindV2.WHEEL_MOTION
            and type(fields.control_name) is str
            and fields.control_name == "hold"
            and sample is not None
            and _poses_equal(sample, start)
            and _poses_equal(end, start)
            and _finite_real(fields.duration_s, "hold duration_s") == 0.0
            and _finite_real(fields.distance_m, "hold distance_m") == 0.0
            and _finite_real(fields.energy_cost, "hold energy_cost") == 0.0
            and _finite_real(
                fields.observation_contribution,
                "hold observation_contribution",
            )
            >= 0.0
            and _finite_real(fields.v_mps, "hold v_mps") == 0.0
            and _finite_real(fields.omega_radps, "hold omega_radps") == 0.0
            and fields.reverse is False
            and fields.turn_in_place is False
            and fields.validation_level is ValidationLevelV2.L2
        )
    except _DeadlineContractError:
        raise
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        valid = False
    return _AuditedMotion(
        index,
        start,
        None,
        None,
        True,
        start,
        1,
        None if valid else "primitive_hold_contract_mismatch",
    )


def _audit_typed_primitive(
    primitive: object,
    index: int,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> _AuditedMotion:
    if type(primitive) is not WheelMotionPrimitiveV2:
        return _AuditedMotion(
            index,
            None,
            None,
            None,
            False,
            None,
            0,
            "wheel_primitive_type_mismatch",
        )
    fields = _read_typed_primitive_fields(primitive)
    samples = fields.samples
    declared_count = len(samples) if type(samples) is tuple else 0
    declared_hold = type(samples) is tuple and declared_count == 1
    if declared_hold:
        return _audit_hold(fields, index, deadline)

    try:
        start = _pose_from_state(fields.start_state)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return _AuditedMotion(
            index,
            None,
            None,
            None,
            False,
            None,
            0,
            "primitive_structure_mismatch",
        )
    control, reason = _new_control(
        fields.control_name,
        fields.v_mps,
        fields.omega_radps,
        fields.duration_s,
        fields.reverse,
        fields.turn_in_place,
        wheel_profile,
    )
    if _fields_missing(
        fields.kind,
        fields.start_state,
        fields.end_state,
        fields.duration_s,
        fields.distance_m,
        fields.energy_cost,
        fields.observation_contribution,
        fields.validation_level,
        fields.control_name,
        fields.samples,
        fields.v_mps,
        fields.omega_radps,
        fields.reverse,
        fields.turn_in_place,
    ):
        reason = "primitive_structure_mismatch"
    if fields.kind is not PrimitiveKindV2.WHEEL_MOTION:
        reason = reason or "primitive_structure_mismatch"
    try:
        if (
            _finite_real(fields.distance_m, "distance_m") < 0.0
            or _finite_real(fields.energy_cost, "energy_cost") < 0.0
            or _finite_real(
                fields.observation_contribution,
                "observation_contribution",
            )
            < 0.0
        ):
            reason = reason or "primitive_structure_mismatch"
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        reason = reason or "primitive_structure_mismatch"
    if control is None:
        return _AuditedMotion(
            index,
            start,
            None,
            None,
            False,
            None,
            declared_count,
            reason,
        )

    try:
        replay = _replay(start, control, wheel_profile, deadline)
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        return _AuditedMotion(
            index,
            start,
            control,
            None,
            False,
            None,
            declared_count,
            reason or "primitive_replay_mismatch",
        )
    try:
        declared_samples = samples
        replay_matches = (
            type(declared_samples) is tuple
            and len(declared_samples) == len(replay.samples)
        )
        if replay_matches:
            for declared, expected in zip(
                declared_samples,
                replay.samples,
                strict=True,
            ):
                _raise_if_deadline_expired(deadline)
                try:
                    normalized = _pose_from_state(declared)
                except _EXPECTED_CONTRACT_EXCEPTIONS:
                    replay_matches = False
                    break
                if not _poses_equal(normalized, expected):
                    replay_matches = False
                    break
        try:
            declared_end = _pose_from_state(fields.end_state)
            end_matches = _poses_equal(declared_end, replay.end)
        except _EXPECTED_CONTRACT_EXCEPTIONS:
            end_matches = False
        replay_matches = (
            replay_matches
            and end_matches
            and isclose(
                _finite_real(fields.distance_m, "distance_m"),
                replay.distance_m,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
    except _DeadlineContractError:
        raise
    except TimeoutError:
        raise
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        replay_matches = False
    if (
        type(fields.control_name) is str
        and fields.control_name == "hold"
    ) or not replay_matches:
        reason = reason or "primitive_replay_mismatch"
    return _AuditedMotion(
        index,
        start,
        control,
        replay,
        False,
        replay.end,
        max(len(replay.samples), declared_count),
        reason,
    )


def _motion_cells(
    audited: _AuditedMotion,
    anchor: FineSafetyAnchorV2,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> tuple[Cell, ...]:
    if audited.start is None:
        return ()
    if audited.hold:
        _raise_if_deadline_expired(deadline)
        cells = conservative_wheel_pose_cells(
            audited.start,
            anchor.snapshot.geometry,
            body_length_m=wheel_profile.body_length_m,
            body_width_m=wheel_profile.body_width_m,
            safety_margin_m=wheel_profile.footprint_safety_margin_m,
        )
        _raise_if_deadline_expired(deadline)
        return cells
    if audited.control is None:
        return ()
    return conservative_wheel_sweep_cells(
        audited.start,
        audited.control,
        anchor.snapshot.geometry,
        body_length_m=wheel_profile.body_length_m,
        body_width_m=wheel_profile.body_width_m,
        safety_margin_m=wheel_profile.footprint_safety_margin_m,
        deadline_checker=lambda: _deadline_expired(deadline),
    )


def _query_cells(
    motions_and_cells: tuple[tuple[_AuditedMotion, tuple[Cell, ...]], ...],
    anchor: FineSafetyAnchorV2,
    expected_snapshot_hash: str,
    validator_id: str,
    deadline: PlanningDeadlineV2,
) -> tuple[_TerrainFailure | None, int, WheelValidationResultV2 | None]:
    best: _TerrainFailure | None = None
    checked = 0
    for audited, cells in motions_and_cells:
        for cell in cells:
            try:
                if _deadline_expired(deadline):
                    return best, checked, _timeout(validator_id, checked)
            except _DeadlineContractError:
                return best, checked, _deadline_contract(validator_id, checked)
            checked += 1
            try:
                query = anchor.query(cell, 30.0)
            except _EXPECTED_CONTRACT_EXCEPTIONS:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_query_contract_mismatch",
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            try:
                if _deadline_expired(deadline):
                    return best, checked, _timeout(validator_id, checked)
            except _DeadlineContractError:
                return best, checked, _deadline_contract(validator_id, checked)
            if type(query) is not SafetyQueryV2:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_query_contract_mismatch",
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            try:
                query_cell = query.cell
                query_cell_matches = (
                    type(query_cell) is Cell
                    and type(query_cell.x) is int
                    and type(query_cell.y) is int
                    and query_cell.x == cell.x
                    and query_cell.y == cell.y
                )
                reason_code = query.reason_code
                returned_hash = query.snapshot_hash
                query_contract_matches = (
                    query_cell_matches
                    and type(query.passed) is bool
                    and type(reason_code) is str
                    and reason_code in _TERRAIN_QUERY_REASON_CODES
                    and query.passed is (reason_code == "terrain_safe")
                    and query.validation_level is ValidationLevelV2.L2
                    and type(returned_hash) is str
                    and len(returned_hash) == 64
                    and not any(
                        character not in "0123456789abcdef"
                        for character in returned_hash
                    )
                )
                if query.slope_deg is not None:
                    slope = _finite_real(query.slope_deg, "query slope_deg")
                    query_contract_matches = query_contract_matches and slope >= 0.0
                if query.confidence is not None:
                    confidence = _finite_real(
                        query.confidence,
                        "query confidence",
                    )
                    query_contract_matches = (
                        query_contract_matches and 0.0 <= confidence <= 1.0
                    )
            except _EXPECTED_CONTRACT_EXCEPTIONS:
                query_contract_matches = False
                returned_hash = None
                reason_code = None
            if not query_contract_matches:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_query_contract_mismatch",
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            if returned_hash != expected_snapshot_hash:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_snapshot_hash_mismatch",
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            if reason_code in _TERRAIN_FAILURE_PRIORITY:
                candidate = _TerrainFailure(reason_code, cell, audited.index)
                if best is None or candidate.key < best.key:
                    best = candidate
    try:
        if _deadline_expired(deadline):
            return best, checked, _timeout(validator_id, checked)
        current_snapshot_hash = _safe_snapshot_hash(anchor.snapshot, deadline)
    except TimeoutError:
        return best, checked, _timeout(validator_id, checked)
    except _DeadlineContractError:
        return best, checked, _deadline_contract(validator_id, checked)
    try:
        cached_snapshot_hash = anchor._snapshot_hash
    except AttributeError:
        cached_snapshot_hash = None
    if (
        current_snapshot_hash != expected_snapshot_hash
        or type(cached_snapshot_hash) is not str
        or cached_snapshot_hash != expected_snapshot_hash
    ):
        return (
            best,
            checked,
            _result(
                validator_id,
                "terrain_snapshot_hash_mismatch",
                checked_cell_count=checked,
            ),
        )
    return best, checked, None


def validate_wheel_transition_l2(
    transition: PoseTransition,
    anchor: FineSafetyAnchorV2,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> WheelValidationResultV2:
    if type(transition) is not PoseTransition:
        raise TypeError("transition must be exact PoseTransition")
    _validate_entry_types(anchor, wheel_profile, deadline)
    try:
        if _deadline_expired(deadline):
            return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    audited_profile = _reaudit_wheel_profile(wheel_profile)
    if audited_profile is None:
        return _result(
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            "wheel_profile_contract_mismatch",
        )
    wheel_profile = audited_profile
    try:
        expected_hash = _safe_snapshot_hash(anchor.snapshot, deadline)
    except TimeoutError:
        return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    if expected_hash is None:
        return _result(
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            "terrain_snapshot_hash_mismatch",
        )
    try:
        audited = _audit_public_transition(transition, wheel_profile, deadline)
        cells = _motion_cells(audited, anchor, wheel_profile, deadline)
    except TimeoutError:
        return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    except _EXPECTED_CONTRACT_EXCEPTIONS:
        audited = _AuditedMotion(
            0,
            None,
            None,
            None,
            False,
            None,
            0,
            "primitive_structure_mismatch",
        )
        cells = ()

    terrain, checked, immediate = _query_cells(
        ((audited, cells),),
        anchor,
        expected_hash,
        WHEEL_TRANSITION_VALIDATOR_ID_V2,
        deadline,
    )
    if immediate is not None:
        return immediate
    if terrain is not None:
        return _result(
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            terrain.reason_code,
            failed_cell=terrain.cell,
            failed_primitive_index=terrain.primitive_index,
            checked_cell_count=checked,
        )
    if audited.reason_code is not None:
        return _result(
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            audited.reason_code,
            failed_primitive_index=0,
            checked_cell_count=checked,
        )
    return _result(
        WHEEL_TRANSITION_VALIDATOR_ID_V2,
        "transition_l2_valid",
        passed=True,
        checked_cell_count=checked,
    )


def validate_route_l2(
    route: TypedRouteV2,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> WheelValidationResultV2:
    if type(route) is not TypedRouteV2:
        raise TypeError("route must be exact TypedRouteV2")
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    _validate_entry_types(anchor, wheel_profile, deadline)
    try:
        if _deadline_expired(deadline):
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    audited_profile = _reaudit_wheel_profile(wheel_profile)
    if audited_profile is None:
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "wheel_profile_contract_mismatch",
        )
    wheel_profile = audited_profile
    max_route_states = _route_state_budget(request)
    if max_route_states is None:
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "planning_request_contract_mismatch",
        )

    if anchor.snapshot is not request.terrain_snapshot:
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "terrain_snapshot_identity_mismatch",
        )
    try:
        request_hash = _safe_snapshot_hash(request.terrain_snapshot, deadline)
        anchor_hash = _safe_snapshot_hash(anchor.snapshot, deadline)
    except TimeoutError:
        return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    if request_hash is None or anchor_hash is None:
        return _result(WHEEL_ROUTE_VALIDATOR_ID_V2, "terrain_snapshot_hash_mismatch")
    if request_hash != anchor_hash:
        return _result(WHEEL_ROUTE_VALIDATOR_ID_V2, "terrain_snapshot_hash_mismatch")
    try:
        if _deadline_expired(deadline):
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)

    structural: tuple[str, int | None] | None = None

    def record(reason_code: str, primitive_index: int | None = None) -> None:
        nonlocal structural
        if structural is None:
            structural = (reason_code, primitive_index)

    if route.platform_kind is not PlatformKindV2.WHEEL:
        record("wheel_platform_identity_mismatch")
    request_profile_id = request.platform_profile_id
    audited_profile_id = wheel_profile.profile.profile_id
    if (
        type(request_profile_id) is not str
        or type(audited_profile_id) is not str
        or request_profile_id != audited_profile_id
    ):
        record("wheel_profile_identity_mismatch")
    if route.is_complete is not True:
        record("route_incomplete")
    try:
        request_start = _pose_from_state(request.start_state)
    except (TypeError, ValueError, OverflowError, AttributeError):
        request_start = None
        record("route_start_contract_mismatch", 0)
    try:
        request_goal = _pose_from_state(request.goal_state)
    except (TypeError, ValueError, OverflowError, AttributeError):
        request_goal = None
        record("route_goal_contract_mismatch")
    primitives = route.primitives
    if type(primitives) is not tuple or not primitives:
        record("route_structure_mismatch")
        primitives = ()
    try:
        overflow_index = _declared_route_state_overflow(
            primitives,
            max_route_states,
            deadline,
        )
    except TimeoutError:
        return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    except _DeadlineContractError:
        return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
    if overflow_index is not None:
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            "route_state_budget_exceeded",
            failed_primitive_index=overflow_index,
        )

    audited_motions: list[_AuditedMotion] = []
    route_state_count = 0
    for index, primitive in enumerate(primitives):
        try:
            _raise_if_deadline_expired(deadline)
            audited = _audit_typed_primitive(
                primitive,
                index,
                wheel_profile,
                deadline,
            )
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except _DeadlineContractError:
            return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        audited_motions.append(audited)
        route_state_count += audited.route_state_count if index == 0 else max(
            0,
            audited.route_state_count - 1,
        )
        if route_state_count > min(max_route_states, _MAX_DECLARED_ROUTE_STATES):
            return _result(
                WHEEL_ROUTE_VALIDATOR_ID_V2,
                "route_state_budget_exceeded",
                failed_primitive_index=index,
            )

    if primitives and request_start is not None:
        first_start = audited_motions[0].start
        if first_start is not None and not _poses_equal(first_start, request_start):
            record("route_start_mismatch", 0)

    for index in range(1, len(audited_motions)):
        try:
            _raise_if_deadline_expired(deadline)
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except _DeadlineContractError:
            return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        previous_actual = audited_motions[index - 1].actual_end
        current_start = audited_motions[index].start
        if (
            previous_actual is not None
            and current_start is not None
            and not _poses_equal(previous_actual, current_start)
        ):
            record("route_connectivity_mismatch", index)

    for audited in audited_motions:
        try:
            _raise_if_deadline_expired(deadline)
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except _DeadlineContractError:
            return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        if audited.reason_code is not None:
            record(audited.reason_code, audited.index)

    if (
        primitives
        and audited_motions[-1].actual_end is not None
        and request_goal is not None
    ):
        final = audited_motions[-1].actual_end
        position_error = hypot(
            final.x_m - request_goal.x_m,
            final.y_m - request_goal.y_m,
        )
        final_heading = remainder(final.theta_rad, 2.0 * pi)
        goal_heading = remainder(request_goal.theta_rad, 2.0 * pi)
        heading_error = abs(remainder(final_heading - goal_heading, 2.0 * pi))
        if (
            position_error > wheel_profile.profile.goal_position_tolerance_m
            or heading_error > wheel_profile.profile.goal_heading_tolerance_rad
        ):
            record("route_goal_tolerance_exceeded", len(primitives) - 1)

    motions_and_cells: list[tuple[_AuditedMotion, tuple[Cell, ...]]] = []
    for audited in audited_motions:
        try:
            _raise_if_deadline_expired(deadline)
            cells = _motion_cells(audited, anchor, wheel_profile, deadline)
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except _DeadlineContractError:
            return _deadline_contract(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except _EXPECTED_CONTRACT_EXCEPTIONS:
            record("primitive_sweep_unavailable", audited.index)
            cells = ()
        motions_and_cells.append((audited, cells))

    terrain, checked, immediate = _query_cells(
        tuple(motions_and_cells),
        anchor,
        request_hash,
        WHEEL_ROUTE_VALIDATOR_ID_V2,
        deadline,
    )
    if immediate is not None:
        return immediate
    if terrain is not None:
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            terrain.reason_code,
            failed_cell=terrain.cell,
            failed_primitive_index=terrain.primitive_index,
            checked_cell_count=checked,
        )
    if structural is not None:
        reason_code, primitive_index = structural
        return _result(
            WHEEL_ROUTE_VALIDATOR_ID_V2,
            reason_code,
            failed_primitive_index=primitive_index,
            checked_cell_count=checked,
        )
    return _result(
        WHEEL_ROUTE_VALIDATOR_ID_V2,
        "route_l2_valid",
        passed=True,
        checked_cell_count=checked,
    )
