from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose, isfinite, nextafter
from numbers import Real

import numpy as np

from path_planner.core import CostGrid, FailureReason, GridSpec
from path_planner.search import (
    HybridAStarPlanner,
    MotionPrimitive,
    Pose2D,
    PosePlanRequest,
    PosePlanResult,
    default_scout_mini_primitives,
    replay_motion_primitive,
)
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    CacheEvidenceV2,
    CostBreakdownV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningOutcomeV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    RoutePrimitiveV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationLevelV2,
)
from path_planner.v2.profiles import PlatformProfileV2, WheelProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineGridGeometryV2, FineSafetyAnchorV2


WHEEL_OBSERVATION_SOURCE_V2 = "wheel_route_samples_gain_not_computed/v1"
WHEEL_CACHE_NAMESPACE_DISABLED_V2 = "path-planner-v2-wheel-cache-disabled/v1"
WHEEL_CACHE_KEY_DISABLED_V2 = "wheel-cache-disabled/v1"


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


def _safe_elapsed(deadline: PlanningDeadlineV2) -> float:
    try:
        elapsed = deadline.elapsed_s
    except Exception:
        return 0.0
    if type(elapsed) not in (int, float):
        return 0.0
    normalized = float(elapsed)
    if not isfinite(normalized) or normalized < 0.0:
        return 0.0
    return normalized


def _hybrid_elapsed(
    result: PosePlanResult | None,
    deadline: PlanningDeadlineV2,
) -> float:
    if result is not None:
        try:
            runtime_ms = result.diagnostics.runtime_ms
        except AttributeError:
            runtime_ms = None
        if type(runtime_ms) in (int, float):
            normalized = float(runtime_ms) / 1000.0
            if isfinite(normalized) and normalized >= 0.0:
                return normalized
    return _safe_elapsed(deadline)


def _telemetry(
    deadline: PlanningDeadlineV2,
    *,
    result: PosePlanResult | None = None,
    termination_reason: str,
    timed_out: bool = False,
) -> SearchTelemetryV2:
    if result is None:
        expanded = generated = rejected_l0 = rejected_l2 = 0
    else:
        expanded = result.expanded_count
        generated = result.audit.generated_primitives
        rejected_l0 = result.audit.rejected_poses
        rejected_l2 = result.audit.rejected_transitions
        if not timed_out:
            timed_out = result.audit.timed_out
            termination_reason = result.audit.termination_reason
    return SearchTelemetryV2(
        expanded_states=expanded,
        generated_primitives=generated,
        rejected_l0=rejected_l0,
        rejected_l1=0,
        rejected_l2=rejected_l2,
        elapsed_s=_hybrid_elapsed(result, deadline),
        timed_out=timed_out,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason=termination_reason,
    )


def _failure(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    *,
    category: FailureCategoryV2,
    reason_code: str,
    stage: str,
    result: PosePlanResult | None = None,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    timed_out: bool = False,
) -> PlanningFailureV2:
    return PlanningFailureV2(
        request_id=request.request_id,
        platform_kind=PlatformKindV2.WHEEL,
        category=category,
        reason_code=reason_code,
        evidence=FailureEvidenceV2(
            stage=stage,
            checks=(reason_code,),
            details=details,
        ),
        search_telemetry=_telemetry(
            deadline,
            result=result,
            termination_reason=reason_code,
            timed_out=timed_out,
        ),
    )


def _timeout_failure(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    stage: str,
    *,
    result: PosePlanResult | None = None,
) -> PlanningFailureV2:
    return _failure(
        request,
        deadline,
        category=FailureCategoryV2.TIMEOUT,
        reason_code="planning_deadline_expired",
        stage=stage,
        result=result,
        timed_out=True,
    )


def _hold(state: PoseStateV2) -> "WheelMotionPrimitiveV2":
    return WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=state,
        end_state=state,
        duration_s=0.0,
        distance_m=0.0,
        energy_cost=0.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        control_name="hold",
        samples=(state,),
        v_mps=0.0,
        omega_radps=0.0,
        reverse=False,
        turn_in_place=False,
    )


def _exact_control(value: object) -> MotionPrimitive:
    if type(value) is not MotionPrimitive:
        raise TypeError("control must be exact MotionPrimitive")
    name = value.name
    if type(name) is not str or not name.strip():
        raise ValueError("control name must be exact nonempty str")
    for field_name in ("v_mps", "omega_radps", "duration_s"):
        field_value = getattr(value, field_name)
        if type(field_value) not in (int, float):
            raise TypeError(f"control {field_name} must be exact int or float")
        normalized = float(field_value)
        if not isfinite(normalized):
            raise ValueError(f"control {field_name} must be finite")
    if type(value.reverse) is not bool or type(value.turn_in_place) is not bool:
        raise TypeError("control flags must be exact bool")
    return MotionPrimitive(
        name=name,
        v_mps=float(value.v_mps),
        omega_radps=float(value.omega_radps),
        duration_s=float(value.duration_s),
        reverse=value.reverse,
        turn_in_place=value.turn_in_place,
    )


def _raw_energy(
    profile: WheelProfileV2,
    control: MotionPrimitive,
    distance_m: float,
    heading_change_rad: float,
) -> float:
    translation = profile.translation_energy_per_m * distance_m
    if control.reverse:
        translation *= profile.reverse_energy_multiplier
    return (
        translation
        + profile.rotation_energy_per_rad * heading_change_rad
        + profile.idle_energy_per_s * control.duration_s
    )


def _candidate_primitive(
    profile: WheelProfileV2,
    control: MotionPrimitive,
    transition,
) -> "WheelMotionPrimitiveV2":
    samples = tuple(_to_pose_state(sample) for sample in transition.samples)
    return WheelMotionPrimitiveV2(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=samples[0],
        end_state=samples[-1],
        duration_s=control.duration_s,
        distance_m=transition.distance_m,
        energy_cost=_raw_energy(
            profile,
            control,
            transition.distance_m,
            transition.absolute_heading_change_rad,
        ),
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L0,
        control_name=control.name,
        samples=samples,
        v_mps=control.v_mps,
        omega_radps=control.omega_radps,
        reverse=control.reverse,
        turn_in_place=control.turn_in_place,
    )


def _constant_grid(geometry: FineGridGeometryV2) -> CostGrid:
    spec = GridSpec(
        width=geometry.width,
        height=geometry.height,
        resolution=geometry.resolution_m,
        origin=geometry.origin,
        frame_id=geometry.frame_id,
    )
    return CostGrid(
        spec=spec,
        cost=np.zeros(spec.shape, dtype=np.float64),
        passable_mask=np.ones(spec.shape, dtype=bool),
    )


def _grid_memory_bytes(geometry: FineGridGeometryV2) -> int:
    cell_count = geometry.width * geometry.height
    return cell_count * (np.dtype(np.float64).itemsize + np.dtype(bool).itemsize)


def _pose_request(
    request: PlanningRequestV2,
    profile: WheelProfileV2,
) -> PosePlanRequest:
    objective = request.objective_profile
    energy_scale = objective.energy_weight / profile.energy_normalization
    return PosePlanRequest(
        start=Pose2D(
            request.start_state.x_m,
            request.start_state.y_m,
            request.start_state.heading_rad,
        ),
        goal=Pose2D(
            request.goal_state.x_m,
            request.goal_state.y_m,
            request.goal_state.heading_rad,
        ),
        theta_bin_count=profile.theta_bin_count,
        position_tolerance_m=profile.profile.goal_position_tolerance_m,
        theta_tolerance_rad=profile.profile.goal_heading_tolerance_rad,
        max_iterations=request.resource_budget.max_expanded_states,
        primitive_duration_s=profile.primitive_duration_s,
        integration_dt_s=profile.integration_dt_s,
        max_speed_mps=profile.max_speed_mps,
        max_angular_speed_radps=profile.max_angular_speed_radps,
        footprint_length_m=profile.body_length_m,
        footprint_width_m=profile.body_width_m,
        footprint_safety_margin_m=profile.footprint_safety_margin_m,
        translation_cost_weight=(
            objective.distance_weight
            + energy_scale * profile.translation_energy_per_m
        ),
        rotation_cost_weight=energy_scale * profile.rotation_energy_per_rad,
        reverse_penalty_weight=(
            energy_scale
            * profile.translation_energy_per_m
            * max(0.0, profile.reverse_energy_multiplier - 1.0)
        ),
        turn_penalty_weight=0.0,
        terrain_cost_weight=0.0,
    )


def _filtered_controls(
    pose_request: PosePlanRequest,
    profile: WheelProfileV2,
) -> tuple[MotionPrimitive, ...]:
    accepted: list[MotionPrimitive] = []
    for control in default_scout_mini_primitives(pose_request):
        if abs(control.v_mps) > profile.max_speed_mps:
            continue
        if abs(control.omega_radps) > profile.max_angular_speed_radps:
            continue
        if (control.reverse or control.v_mps < 0.0) and not profile.reverse_enabled:
            continue
        if control.turn_in_place and not profile.turn_in_place_enabled:
            continue
        if control.v_mps != 0.0 and control.omega_radps != 0.0:
            radius = abs(control.v_mps / control.omega_radps)
            if radius < profile.min_turning_radius_m:
                continue
        accepted.append(control)
    return tuple(accepted)


def _cost_breakdown(
    request: PlanningRequestV2,
    profile: WheelProfileV2,
    primitives: tuple["WheelMotionPrimitiveV2", ...],
) -> CostBreakdownV2:
    raw_distance = sum(primitive.distance_m for primitive in primitives)
    raw_energy = sum(primitive.energy_cost for primitive in primitives)
    raw_duration = sum(primitive.duration_s for primitive in primitives)
    objective = request.objective_profile
    distance_cost = objective.distance_weight * raw_distance
    energy_cost = objective.energy_weight * (
        raw_energy / profile.energy_normalization
    )
    time_cost = objective.time_weight * (
        raw_duration / profile.time_normalization_s
    )
    total_cost = distance_cost + energy_cost + time_cost
    return CostBreakdownV2(
        distance_cost=distance_cost,
        risk_cost=0.0,
        energy_cost=energy_cost,
        time_cost=time_cost,
        total_cost=total_cost,
    )


def _observation_samples(
    primitives: tuple["WheelMotionPrimitiveV2", ...],
) -> tuple[PoseStateV2, ...]:
    samples: list[PoseStateV2] = []
    for primitive in primitives:
        if not samples:
            samples.extend(primitive.samples)
        else:
            samples.extend(primitive.samples[1:])
    return tuple(samples)


def _hybrid_failure_mapping(
    reason: FailureReason | None,
) -> tuple[FailureCategoryV2, str]:
    if reason is FailureReason.TIMEOUT:
        return FailureCategoryV2.TIMEOUT, "planning_deadline_expired"
    if reason is FailureReason.VALIDATOR_ERROR:
        return FailureCategoryV2.VALIDATION_FAILED, "wheel_hybrid_validator_error"
    if reason in (FailureReason.START_OUT_OF_BOUNDS, FailureReason.START_BLOCKED):
        return FailureCategoryV2.UNSAFE_START, "wheel_start_pose_invalid"
    if reason in (FailureReason.GOAL_OUT_OF_BOUNDS, FailureReason.GOAL_BLOCKED):
        return FailureCategoryV2.UNSAFE_GOAL, "wheel_goal_pose_invalid"
    if reason is FailureReason.UNREACHABLE:
        return FailureCategoryV2.GOAL_POSE_UNREACHABLE, "wheel_goal_pose_unreachable"
    if reason is FailureReason.MAX_ITERATIONS:
        return FailureCategoryV2.RESOURCE_LIMIT, "wheel_expansion_budget_exhausted"
    return FailureCategoryV2.NO_COMPLETE_ROUTE, "wheel_no_complete_route"


def _validation_failure_category(
    reason_code: str,
    *,
    hold: bool,
) -> FailureCategoryV2:
    if reason_code == "planning_deadline_expired":
        return FailureCategoryV2.TIMEOUT
    if reason_code == "route_state_budget_exceeded":
        return FailureCategoryV2.RESOURCE_LIMIT
    if reason_code in (
        "route_goal_tolerance_exceeded",
        "route_goal_contract_mismatch",
    ):
        return FailureCategoryV2.GOAL_POSE_UNREACHABLE
    if hold and reason_code in {
        "terrain_out_of_bounds",
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
    }:
        return FailureCategoryV2.UNSAFE_START
    return FailureCategoryV2.VALIDATION_FAILED


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


@dataclass(frozen=True, slots=True)
class WheelPrimitiveProviderV2:
    wheel_profile: WheelProfileV2

    def __post_init__(self) -> None:
        if type(self.wheel_profile) is not WheelProfileV2:
            raise TypeError("wheel_profile must be exact WheelProfileV2")

    @property
    def profile(self) -> PlatformProfileV2:
        return self.wheel_profile.profile

    def plan(
        self,
        request: PlanningRequestV2,
        anchor: FineSafetyAnchorV2,
        deadline: PlanningDeadlineV2,
    ) -> PlanningOutcomeV2:
        if type(request) is not PlanningRequestV2:
            raise TypeError("request must be exact PlanningRequestV2")
        if type(anchor) is not FineSafetyAnchorV2:
            raise TypeError("anchor must be exact FineSafetyAnchorV2")
        if type(deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")

        profile = self.wheel_profile
        stage = "capability_check"
        try:
            if request.objective_profile.risk_weight != 0.0:
                return _failure(
                    request,
                    deadline,
                    category=FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                    reason_code="wheel_risk_objective_unsupported",
                    stage=stage,
                )
            if request.accelerator_policy is AcceleratorPolicyV2.REQUIRED:
                return _failure(
                    request,
                    deadline,
                    category=FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                    reason_code="wheel_accelerator_required_unsupported",
                    stage=stage,
                )

            stage = "resource_check"
            if deadline.expired:
                return _timeout_failure(request, deadline, stage)
            if request.resource_budget.max_expanded_states == 0:
                return _failure(
                    request,
                    deadline,
                    category=FailureCategoryV2.RESOURCE_LIMIT,
                    reason_code="wheel_expansion_budget_exhausted",
                    stage=stage,
                )

            hybrid_result: PosePlanResult | None = None
            exact_hold = request.start_state == request.goal_state
            if exact_hold:
                candidates = (_hold(request.start_state),)
            else:
                geometry = anchor.snapshot.geometry
                memory_limit = request.resource_budget.max_memory_bytes
                if (
                    memory_limit > 0
                    and _grid_memory_bytes(geometry) > memory_limit
                ):
                    return _failure(
                        request,
                        deadline,
                        category=FailureCategoryV2.RESOURCE_LIMIT,
                        reason_code="wheel_search_grid_memory_budget_exceeded",
                        stage=stage,
                    )

                stage = "hybrid_search"
                grid = _constant_grid(geometry)
                pose_request = _pose_request(request, profile)
                controls = _filtered_controls(pose_request, profile)

                def pose_validator(_pose: Pose2D) -> bool:
                    return True

                from path_planner.v2.validation import validate_wheel_transition_l2

                def transition_validator(transition) -> bool:
                    validation = validate_wheel_transition_l2(
                        transition,
                        anchor,
                        profile,
                        deadline,
                    )
                    return validation.evidence.passed is True

                def hybrid_clock() -> float:
                    return (
                        deadline.deadline_monotonic_s
                        if deadline.expired
                        else deadline.started_monotonic_s
                    )

                hybrid_result = HybridAStarPlanner(controls).plan(
                    grid,
                    pose_request,
                    pose_validator=pose_validator,
                    transition_validator=transition_validator,
                    deadline_monotonic_s=deadline.deadline_monotonic_s,
                    monotonic_clock=hybrid_clock,
                )
                if type(hybrid_result) is not PosePlanResult:
                    raise TypeError("Hybrid planner must return exact PosePlanResult")
                if not hybrid_result.success:
                    category, reason_code = _hybrid_failure_mapping(
                        hybrid_result.failure_reason
                    )
                    return _failure(
                        request,
                        deadline,
                        category=category,
                        reason_code=reason_code,
                        stage=stage,
                        result=hybrid_result,
                    )
                if deadline.expired:
                    return _timeout_failure(
                        request,
                        deadline,
                        stage,
                        result=hybrid_result,
                    )

                stage = "independent_replay"
                current = Pose2D(
                    request.start_state.x_m,
                    request.start_state.y_m,
                    request.start_state.heading_rad,
                )
                replayed: list[WheelMotionPrimitiveV2] = []
                for raw_control in hybrid_result.control_sequence:
                    try:
                        control = _exact_control(raw_control)
                        transition = replay_motion_primitive(
                            current,
                            control,
                            profile.integration_dt_s,
                            deadline_checker=lambda: deadline.expired,
                        )
                        replayed.append(
                            _candidate_primitive(profile, control, transition)
                        )
                    except TimeoutError:
                        return _timeout_failure(
                            request,
                            deadline,
                            stage,
                            result=hybrid_result,
                        )
                    except Exception:
                        return _failure(
                            request,
                            deadline,
                            category=FailureCategoryV2.VALIDATION_FAILED,
                            reason_code="wheel_control_replay_failed",
                            stage=stage,
                            result=hybrid_result,
                        )
                    current = transition.end
                if not replayed:
                    return _failure(
                        request,
                        deadline,
                        category=FailureCategoryV2.VALIDATION_FAILED,
                        reason_code="wheel_control_replay_failed",
                        stage=stage,
                        result=hybrid_result,
                    )
                candidates = tuple(replayed)

            costs = _cost_breakdown(request, profile, candidates)
            candidate_route = TypedRouteV2(
                platform_kind=PlatformKindV2.WHEEL,
                primitives=candidates,
                total_cost=costs.total_cost,
                is_complete=True,
            )

            stage = "route_validation"
            from path_planner.v2.validation import validate_route_l2

            validation = validate_route_l2(
                candidate_route,
                request,
                anchor,
                profile,
                deadline,
            )
            if not validation.evidence.passed:
                hold_candidate = (
                    len(candidates) == 1 and candidates[0].control_name == "hold"
                )
                category = _validation_failure_category(
                    validation.reason_code,
                    hold=hold_candidate,
                )
                if category is FailureCategoryV2.TIMEOUT:
                    return _timeout_failure(
                        request,
                        deadline,
                        stage,
                        result=hybrid_result,
                    )
                return _failure(
                    request,
                    deadline,
                    category=category,
                    reason_code=validation.reason_code,
                    stage=stage,
                    result=hybrid_result,
                )

            if deadline.expired:
                return _timeout_failure(
                    request,
                    deadline,
                    stage,
                    result=hybrid_result,
                )

            stage = "route_construction"
            primitives_l2 = tuple(
                primitive
                if primitive.validation_level is ValidationLevelV2.L2
                else replace(primitive, validation_level=ValidationLevelV2.L2)
                for primitive in candidates
            )
            route = TypedRouteV2(
                platform_kind=PlatformKindV2.WHEEL,
                primitives=primitives_l2,
                total_cost=costs.total_cost,
                is_complete=True,
            )
            observation = ObservationProjectionV2(
                source=WHEEL_OBSERVATION_SOURCE_V2,
                sample_states=_observation_samples(primitives_l2),
                expected_new_observed_cells=0.0,
                expected_information_gain=0.0,
            )
            telemetry = _telemetry(
                deadline,
                result=hybrid_result,
                termination_reason=(
                    "start_equals_goal" if hybrid_result is None else "success"
                ),
            )
            return PlanningSuccessV2(
                request_id=request.request_id,
                platform_kind=PlatformKindV2.WHEEL,
                route=route,
                observation_projection=observation,
                cost_breakdown=costs,
                validation_evidence=validation.evidence,
                search_telemetry=telemetry,
                cache_evidence=CacheEvidenceV2(
                    cache_namespace=WHEEL_CACHE_NAMESPACE_DISABLED_V2,
                    cache_key=WHEEL_CACHE_KEY_DISABLED_V2,
                    hit=False,
                ),
            )
        except Exception as exc:
            return _failure(
                request,
                deadline,
                category=FailureCategoryV2.INTERNAL_ERROR,
                reason_code="wheel_provider_exception",
                stage=stage,
                details=(("exception_type", type(exc).__name__),),
            )
