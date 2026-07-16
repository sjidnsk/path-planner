from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isclose, isfinite, pi, remainder
from numbers import Real

from path_planner.core import Cell
from path_planner.search import (
    MotionPrimitive,
    Pose2D,
    PoseTransition,
    replay_motion_primitive,
)
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
    FineSafetyAnchorV2,
    SafetyQueryV2,
    snapshot_hash,
)


WHEEL_TRANSITION_VALIDATOR_ID_V2 = "path-planner-v2-wheel-transition-l2/v1"
WHEEL_ROUTE_VALIDATOR_ID_V2 = "path-planner-v2-wheel-route-l2/v1"

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
        if validator_id not in _VALIDATOR_REASON_CODES:
            raise ValueError("evidence validator_id must be a stable wheel L2 validator")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be a nonempty string")
        if self.reason_code not in _VALIDATOR_REASON_CODES[validator_id]:
            raise ValueError("reason_code is not valid for the wheel L2 validator")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be bool")
        if self.failed_cell is not None and type(self.failed_cell) is not Cell:
            raise TypeError("failed_cell must be exact Cell or None")
        if self.failed_primitive_index is not None:
            if (
                isinstance(self.failed_primitive_index, bool)
                or not isinstance(self.failed_primitive_index, int)
            ):
                raise TypeError("failed_primitive_index must be an integer or None")
            if self.failed_primitive_index < 0:
                raise ValueError("failed_primitive_index must be nonnegative")
        if isinstance(self.checked_cell_count, bool) or not isinstance(
            self.checked_cell_count,
            int,
        ):
            raise TypeError("checked_cell_count must be an integer")
        if self.checked_cell_count < 0:
            raise ValueError("checked_cell_count must be nonnegative")
        if self.timed_out is not (self.reason_code == "planning_deadline_expired"):
            raise ValueError("timeout fields must agree")
        if self.timed_out and (
            self.failed_cell is not None or self.failed_primitive_index is not None
        ):
            raise ValueError("timeout result must not carry failure metadata")
        if self.failed_cell is not None and self.failed_primitive_index is None:
            raise ValueError("failed_cell requires failed_primitive_index")
        if self.evidence.checks != (self.reason_code,):
            raise ValueError("evidence checks must contain the exact reason_code")
        if self.evidence.passed is not (
            self.reason_code == _PASS_REASON_BY_VALIDATOR[validator_id]
        ):
            raise ValueError("evidence passed must agree with the reason_code")
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
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None


def _route_state_budget(request: PlanningRequestV2) -> int | None:
    try:
        budget = request.resource_budget
        if type(budget) is not ResourceBudgetV2:
            return None
        max_route_states = budget.max_route_states
    except AttributeError:
        return None
    if isinstance(max_route_states, bool) or not isinstance(max_route_states, int):
        return None
    if max_route_states < 0:
        return None
    return max_route_states


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
    name_valid = isinstance(name, str) and bool(name.strip())
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
        deadline_checker=lambda: deadline.expired,
    )


def _audit_public_transition(
    transition: PoseTransition,
    wheel_profile: WheelProfileV2,
    deadline: PlanningDeadlineV2,
) -> _AuditedMotion:
    try:
        start = Pose2D(
            _finite_real(transition.start.x_m, "transition start x_m"),
            _finite_real(transition.start.y_m, "transition start y_m"),
            _finite_real(transition.start.theta_rad, "transition start theta_rad"),
        )
    except (AttributeError, TypeError, ValueError):
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

    primitive = transition.primitive
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
    control, reason = _new_control(
        primitive.name,
        primitive.v_mps,
        primitive.omega_radps,
        primitive.duration_s,
        primitive.reverse,
        primitive.turn_in_place,
        wheel_profile,
    )
    if control is None:
        return _AuditedMotion(0, start, None, None, False, None, 0, reason)
    replay = _replay(start, control, wheel_profile, deadline)
    try:
        exact_samples = (
            isinstance(transition.samples, tuple)
            and all(type(sample) is Pose2D for sample in transition.samples)
            and transition.samples == replay.samples
        )
        exact_end = type(transition.end) is Pose2D and transition.end == replay.end
        distance = _finite_real(transition.distance_m, "transition distance_m")
        heading_change = _finite_real(
            transition.absolute_heading_change_rad,
            "transition absolute_heading_change_rad",
        )
        replay_matches = (
            transition.start == start
            and exact_samples
            and exact_end
            and isclose(distance, replay.distance_m, rel_tol=1.0e-12, abs_tol=1.0e-12)
            and isclose(
                heading_change,
                replay.absolute_heading_change_rad,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
    except (AttributeError, TypeError, ValueError):
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


def _audit_hold(primitive: WheelMotionPrimitiveV2, index: int) -> _AuditedMotion:
    try:
        start = _pose_from_state(primitive.start_state)
    except (TypeError, ValueError):
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
        valid = (
            primitive.kind is PrimitiveKindV2.WHEEL_MOTION
            and primitive.control_name == "hold"
            and primitive.samples == (primitive.start_state,)
            and primitive.end_state == primitive.start_state
            and _finite_real(primitive.duration_s, "hold duration_s") == 0.0
            and _finite_real(primitive.distance_m, "hold distance_m") == 0.0
            and _finite_real(primitive.energy_cost, "hold energy_cost") == 0.0
            and _finite_real(
                primitive.observation_contribution,
                "hold observation_contribution",
            )
            >= 0.0
            and _finite_real(primitive.v_mps, "hold v_mps") == 0.0
            and _finite_real(primitive.omega_radps, "hold omega_radps") == 0.0
            and primitive.reverse is False
            and primitive.turn_in_place is False
            and primitive.validation_level is ValidationLevelV2.L2
        )
    except (TypeError, ValueError):
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
    if isinstance(primitive.samples, tuple) and len(primitive.samples) == 1:
        return _audit_hold(primitive, index)

    try:
        start = _pose_from_state(primitive.start_state)
    except (TypeError, ValueError):
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
        primitive.control_name,
        primitive.v_mps,
        primitive.omega_radps,
        primitive.duration_s,
        primitive.reverse,
        primitive.turn_in_place,
        wheel_profile,
    )
    if primitive.kind is not PrimitiveKindV2.WHEEL_MOTION:
        reason = reason or "primitive_structure_mismatch"
    try:
        if (
            _finite_real(primitive.distance_m, "distance_m") < 0.0
            or _finite_real(primitive.energy_cost, "energy_cost") < 0.0
            or _finite_real(
                primitive.observation_contribution,
                "observation_contribution",
            )
            < 0.0
        ):
            reason = reason or "primitive_structure_mismatch"
    except (TypeError, ValueError):
        reason = reason or "primitive_structure_mismatch"
    if control is None:
        declared_count = len(primitive.samples) if isinstance(primitive.samples, tuple) else 0
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
    except (TypeError, ValueError):
        declared_count = len(primitive.samples) if isinstance(primitive.samples, tuple) else 0
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
    expected_samples = tuple(
        PoseStateV2(sample.x_m, sample.y_m, sample.theta_rad)
        for sample in replay.samples
    )
    try:
        replay_matches = (
            isinstance(primitive.samples, tuple)
            and all(type(sample) is PoseStateV2 for sample in primitive.samples)
            and primitive.samples == expected_samples
            and type(primitive.end_state) is PoseStateV2
            and primitive.end_state == expected_samples[-1]
            and isclose(
                _finite_real(primitive.distance_m, "distance_m"),
                replay.distance_m,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )
    except (TypeError, ValueError):
        replay_matches = False
    if primitive.control_name == "hold" or not replay_matches:
        reason = reason or "primitive_replay_mismatch"
    return _AuditedMotion(
        index,
        start,
        control,
        replay,
        False,
        replay.end,
        max(
            len(replay.samples),
            len(primitive.samples) if isinstance(primitive.samples, tuple) else 0,
        ),
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
        if deadline.expired:
            raise TimeoutError("wheel validation deadline expired")
        cells = conservative_wheel_pose_cells(
            audited.start,
            anchor.snapshot.geometry,
            body_length_m=wheel_profile.body_length_m,
            body_width_m=wheel_profile.body_width_m,
            safety_margin_m=wheel_profile.footprint_safety_margin_m,
        )
        if deadline.expired:
            raise TimeoutError("wheel validation deadline expired")
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
        deadline_checker=lambda: deadline.expired,
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
            if deadline.expired:
                return best, checked, _timeout(validator_id, checked)
            query = anchor.query(cell, 30.0)
            checked += 1
            if deadline.expired:
                return best, checked, _timeout(validator_id, checked)
            if type(query) is not SafetyQueryV2:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_query_contract_mismatch",
                        failed_cell=cell,
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            if query.snapshot_hash != expected_snapshot_hash:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_snapshot_hash_mismatch",
                        failed_cell=cell,
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            query_contract_matches = (
                query.cell == cell
                and query.validation_level is ValidationLevelV2.L2
                and query.reason_code in _TERRAIN_QUERY_REASON_CODES
                and query.passed is (query.reason_code == "terrain_safe")
            )
            if not query_contract_matches:
                return (
                    best,
                    checked,
                    _result(
                        validator_id,
                        "terrain_query_contract_mismatch",
                        failed_cell=cell,
                        failed_primitive_index=audited.index,
                        checked_cell_count=checked,
                    ),
                )
            if query.reason_code in _TERRAIN_FAILURE_PRIORITY:
                candidate = _TerrainFailure(query.reason_code, cell, audited.index)
                if best is None or candidate.key < best.key:
                    best = candidate
    if deadline.expired:
        return best, checked, _timeout(validator_id, checked)
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
    if deadline.expired:
        return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    audited_profile = _reaudit_wheel_profile(wheel_profile)
    if audited_profile is None:
        return _result(
            WHEEL_TRANSITION_VALIDATOR_ID_V2,
            "wheel_profile_contract_mismatch",
        )
    wheel_profile = audited_profile
    expected_hash = snapshot_hash(anchor.snapshot)
    if deadline.expired:
        return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    try:
        audited = _audit_public_transition(transition, wheel_profile, deadline)
        cells = _motion_cells(audited, anchor, wheel_profile, deadline)
    except TimeoutError:
        return _timeout(WHEEL_TRANSITION_VALIDATOR_ID_V2, 0)
    except (TypeError, ValueError):
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
    if deadline.expired:
        return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
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
        request_hash = snapshot_hash(request.terrain_snapshot)
        anchor_hash = snapshot_hash(anchor.snapshot)
    except (TypeError, ValueError):
        return _result(WHEEL_ROUTE_VALIDATOR_ID_V2, "terrain_snapshot_hash_mismatch")
    if request_hash != anchor_hash:
        return _result(WHEEL_ROUTE_VALIDATOR_ID_V2, "terrain_snapshot_hash_mismatch")
    if deadline.expired:
        return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)

    structural: tuple[str, int | None] | None = None

    def record(reason_code: str, primitive_index: int | None = None) -> None:
        nonlocal structural
        if structural is None:
            structural = (reason_code, primitive_index)

    if route.platform_kind is not PlatformKindV2.WHEEL:
        record("wheel_platform_identity_mismatch")
    if request.platform_profile_id != wheel_profile.profile.profile_id:
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
    if not isinstance(primitives, tuple) or not primitives:
        record("route_structure_mismatch")
        primitives = ()
    if primitives and request_start is not None:
        try:
            if primitives[0].start_state != request.start_state:
                record("route_start_mismatch", 0)
        except AttributeError:
            record("route_start_mismatch", 0)
    for index in range(1, len(primitives)):
        try:
            if primitives[index - 1].end_state != primitives[index].start_state:
                record("route_connectivity_mismatch", index)
        except AttributeError:
            record("route_connectivity_mismatch", index)

    audited_motions: list[_AuditedMotion] = []
    route_state_count = 0
    for index, primitive in enumerate(primitives):
        if deadline.expired:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        try:
            audited = _audit_typed_primitive(
                primitive,
                index,
                wheel_profile,
                deadline,
            )
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        audited_motions.append(audited)
        if audited.reason_code is not None:
            record(audited.reason_code, index)
        route_state_count += audited.route_state_count if index == 0 else max(
            0,
            audited.route_state_count - 1,
        )
        if route_state_count > max_route_states:
            return _result(
                WHEEL_ROUTE_VALIDATOR_ID_V2,
                "route_state_budget_exceeded",
                failed_primitive_index=index,
            )

    for index in range(1, len(primitives)):
        try:
            declared_connected = (
                primitives[index - 1].end_state == primitives[index].start_state
            )
        except AttributeError:
            declared_connected = False
        previous_actual = audited_motions[index - 1].actual_end
        current_start = audited_motions[index].start
        actual_connected = (
            previous_actual is not None
            and current_start is not None
            and previous_actual == current_start
        )
        if not declared_connected or not actual_connected:
            record("route_connectivity_mismatch", index)

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
        heading_error = abs(
            remainder(final.theta_rad - request_goal.theta_rad, 2.0 * pi)
        )
        if (
            position_error > wheel_profile.profile.goal_position_tolerance_m
            or heading_error > wheel_profile.profile.goal_heading_tolerance_rad
        ):
            record("route_goal_tolerance_exceeded", len(primitives) - 1)

    motions_and_cells: list[tuple[_AuditedMotion, tuple[Cell, ...]]] = []
    for audited in audited_motions:
        try:
            cells = _motion_cells(audited, anchor, wheel_profile, deadline)
        except TimeoutError:
            return _timeout(WHEEL_ROUTE_VALIDATOR_ID_V2, 0)
        except (TypeError, ValueError):
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
