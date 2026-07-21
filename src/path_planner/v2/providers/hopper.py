from __future__ import annotations

import struct
from dataclasses import dataclass
from math import copysign, cos, isfinite, pi, sin

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    CacheEvidenceV2,
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
from path_planner.v2.hopper_authority import (
    HOPPER_RESOURCE_AUTHORITY_V2,
    HopperParameterSetRecordV2,
    HopperProviderAuthorityV2,
    _lookup_hopper_parameter_set_v2,
    _parameter_set_record_is_exact_v2,
)
from path_planner.v2.oracles.hopper import (
    HopperJumpCandidateV2,
    validate_hopper_jump_l2,
)
from path_planner.v2.profiles import audit_hopper_profile_v2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineSafetyAnchorV2


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


def _provider_telemetry_v2(
    deadline: PlanningDeadlineV2,
    reason: str,
    *,
    expanded_states: int = 0,
    generated_primitives: int = 0,
    rejected_l2: int = 0,
    timed_out: bool = False,
) -> SearchTelemetryV2:
    elapsed = deadline.elapsed_s
    return SearchTelemetryV2(
        expanded_states=expanded_states,
        generated_primitives=generated_primitives,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=rejected_l2,
        elapsed_s=0.0 if elapsed == 0.0 else elapsed,
        timed_out=timed_out,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason=reason,
    )


def _provider_failure_v2(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    reason: str,
    category: FailureCategoryV2,
    stage: str,
    *,
    expanded_states: int = 0,
    generated_primitives: int = 0,
    rejected_l2: int = 0,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> PlanningFailureV2:
    return PlanningFailureV2(
        request_id=request.request_id,
        platform_kind=PlatformKindV2.HOPPER,
        category=category,
        reason_code=reason,
        evidence=FailureEvidenceV2(stage, (reason,), details),
        search_telemetry=_provider_telemetry_v2(
            deadline,
            reason,
            expanded_states=expanded_states,
            generated_primitives=generated_primitives,
            rejected_l2=rejected_l2,
            timed_out=reason == "planning_deadline_expired",
        ),
    )


def _provider_record_v2(
    authority: HopperProviderAuthorityV2,
) -> HopperParameterSetRecordV2 | None:
    record = _lookup_hopper_parameter_set_v2(authority.parameter_set_id)
    if type(record) is not HopperParameterSetRecordV2:
        return None
    if not _parameter_set_record_is_exact_v2(record):
        return None
    return record


def _provider_profile_matches_record_v2(
    authority: HopperProviderAuthorityV2,
    record: HopperParameterSetRecordV2,
) -> bool:
    profile = authority.hopper_profile
    return (
        profile.profile.profile_id == record.base_profile_id
        and profile.body_envelope_radius_m == record.body_envelope_radius_m
        and profile.launch_reference_height_m == record.launch_reference_height_m
        and profile.arc_clearance_margin_m == record.arc_clearance_margin_m
        and profile.landing_footprint_radius_m == record.landing_footprint_radius_m
        and profile.stop_condition == record.stop_condition
        and profile.energy_model == record.energy_model
    )


def _provider_direction_v2(index: int) -> tuple[float, float]:
    if index == 0:
        return 1.0, 0.0
    if index == 4:
        return 0.0, 1.0
    if index == 8:
        return -1.0, 0.0
    if index == 12:
        return 0.0, -1.0
    azimuth = 2.0 * pi * index / 16.0
    return cos(azimuth), sin(azimuth)


@dataclass(frozen=True, slots=True)
class HopperPrimitiveProviderV2:
    hopper_authority: HopperProviderAuthorityV2

    def __post_init__(self) -> None:
        if type(self.hopper_authority) is not HopperProviderAuthorityV2:
            raise TypeError("hopper_authority must be exact HopperProviderAuthorityV2")

    @property
    def profile(self) -> PlatformProfileV2:
        return self.hopper_authority.hopper_profile.profile

    @property
    def hopper_resource_authority(self):
        return HOPPER_RESOURCE_AUTHORITY_V2

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
        if anchor.snapshot is not request.terrain_snapshot:
            return _provider_failure_v2(
                request,
                deadline,
                "terrain_snapshot_identity_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "provider_authority",
            )

        profile = self.hopper_authority.hopper_profile
        audit = audit_hopper_profile_v2(profile)
        if not audit.complete:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_proxy_profile_incomplete",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        record = _provider_record_v2(self.hopper_authority)
        if record is None:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_parameter_set_unsupported",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if not _provider_profile_matches_record_v2(self.hopper_authority, record):
            reason = (
                "hopper_stop_condition_unsupported"
                if profile.stop_condition != record.stop_condition
                else "hopper_energy_model_unsupported"
                if profile.energy_model != record.energy_model
                else "hopper_parameter_set_unsupported"
            )
            return _provider_failure_v2(
                request,
                deadline,
                reason,
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if request.objective_profile.risk_weight != 0.0:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_risk_objective_unsupported",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if request.accelerator_policy is AcceleratorPolicyV2.REQUIRED:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_accelerator_required_unsupported",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if deadline.expired:
            return _provider_failure_v2(
                request,
                deadline,
                "planning_deadline_expired",
                FailureCategoryV2.TIMEOUT,
                "capability_preflight",
            )
        if request.start_state.heading_rad != request.goal_state.heading_rad:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_goal_heading_unreachable",
                FailureCategoryV2.GOAL_POSE_UNREACHABLE,
                "goal_preflight",
            )
        if request.start_state == request.goal_state:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_hold_oracle_unavailable",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if request.resource_budget.max_expanded_states == 0:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_expansion_budget_exhausted",
                FailureCategoryV2.RESOURCE_LIMIT,
                "search_setup",
            )
        effective_route_states = min(
            request.resource_budget.max_route_states, 100_001
        )
        if effective_route_states < 2:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_route_state_budget_exceeded",
                FailureCategoryV2.RESOURCE_LIMIT,
                "route_state_admission",
                details=(
                    ("attempted_route_states", 2),
                    ("effective_max_route_states", effective_route_states),
                    (
                        "requested_max_route_states",
                        request.resource_budget.max_route_states,
                    ),
                ),
            )

        expanded_states = 1
        generated_primitives = 0
        rejected_l2 = 0
        start_hopper = HopperSearchStateV2(request.start_state, 0.0)
        best_match: tuple[float, tuple[int, int, int], HopperJumpPrimitiveV2] | None = None
        objective = request.objective_profile

        for speed_index in range(4):
            for elevation_index in range(3):
                for azimuth_index in range(16):
                    generated_primitives += 1
                    if deadline.expired:
                        return _provider_failure_v2(
                            request,
                            deadline,
                            "planning_deadline_expired",
                            FailureCategoryV2.TIMEOUT,
                            "search_expansion",
                            expanded_states=expanded_states,
                            generated_primitives=generated_primitives,
                            rejected_l2=rejected_l2,
                        )
                    candidate = HopperJumpCandidateV2(
                        request.start_state,
                        0.0,
                        profile,
                        record.parameter_set_id,
                        speed_index,
                        elevation_index,
                        azimuth_index,
                        "hopper-jump-candidate/v1",
                    )
                    validation = validate_hopper_jump_l2(candidate, anchor, deadline)
                    if validation.reason_code != "hopper_jump_l2_valid":
                        if validation.category is FailureCategoryV2.VALIDATION_FAILED:
                            rejected_l2 += 1
                            continue
                        return _provider_failure_v2(
                            request,
                            deadline,
                            validation.reason_code,
                            validation.category or FailureCategoryV2.INTERNAL_ERROR,
                            validation.stage,
                            expanded_states=expanded_states,
                            generated_primitives=generated_primitives,
                            rejected_l2=rejected_l2,
                            details=validation.details,
                        )
                    speed = profile.launch_speeds_mps[speed_index]
                    elevation = profile.launch_elevations_rad[elevation_index]
                    vertical_speed = speed * sin(elevation)
                    flight_time = 2.0 * (vertical_speed / profile.gravity_mps2)
                    distance = (speed * cos(elevation)) * flight_time
                    x_direction, y_direction = _provider_direction_v2(azimuth_index)
                    end_pose = PoseStateV2(
                        request.start_state.x_m + distance * x_direction,
                        request.start_state.y_m + distance * y_direction,
                        request.start_state.heading_rad,
                    )
                    end_hopper = HopperSearchStateV2(end_pose, 0.0)
                    energy = record.energy_evaluator(speed)
                    primitive = HopperJumpPrimitiveV2(
                        kind=PrimitiveKindV2.BALLISTIC_JUMP,
                        start_state=request.start_state,
                        end_state=end_pose,
                        duration_s=flight_time,
                        distance_m=distance,
                        energy_cost=energy,
                        observation_contribution=0.0,
                        validation_level=ValidationLevelV2.L2,
                        start_hopper_state=start_hopper,
                        end_hopper_state=end_hopper,
                        speed_index=speed_index,
                        elevation_index=elevation_index,
                        azimuth_index=azimuth_index,
                        parameter_set_id=record.parameter_set_id,
                        selected_landing_mass=validation.selected_landing_mass,
                    )
                    if end_pose != request.goal_state:
                        continue
                    distance_cost = objective.distance_weight * distance
                    energy_cost = objective.energy_weight * energy
                    time_cost = objective.time_weight * flight_time
                    total = sum((distance_cost, 0.0, energy_cost, time_cost))
                    key = (speed_index, elevation_index, azimuth_index)
                    candidate_match = (total, key, primitive)
                    if best_match is None or candidate_match[:2] < best_match[:2]:
                        best_match = candidate_match

        if best_match is None:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_no_complete_route",
                FailureCategoryV2.NO_COMPLETE_ROUTE,
                "hopper_search",
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
            )

        _total, _action_key, primitive = best_match
        route = TypedRouteV2(
            PlatformKindV2.HOPPER,
            (primitive,),
            best_match[0],
            True,
        )
        from path_planner.v2.hopper_route_validation import validate_hopper_route_l2

        l2_result = validate_hopper_route_l2(
            route,
            request,
            anchor,
            self.hopper_authority,
            deadline,
        )
        if not l2_result.passed:
            return _provider_failure_v2(
                request,
                deadline,
                l2_result.reason_code,
                l2_result.category or FailureCategoryV2.INTERNAL_ERROR,
                l2_result.stage,
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
                details=l2_result.details,
            )
        return PlanningSuccessV2(
            request_id=request.request_id,
            platform_kind=PlatformKindV2.HOPPER,
            route=route,
            observation_projection=ObservationProjectionV2(
                source=(
                    "path-planner-v2-hopper-inflight-observation-disabled/v1"
                ),
                sample_states=(primitive.start_state, primitive.end_state),
                expected_new_observed_cells=0.0,
                expected_information_gain=0.0,
            ),
            cost_breakdown=l2_result.cost_breakdown,
            validation_evidence=l2_result.evidence,
            search_telemetry=_provider_telemetry_v2(
                deadline,
                "hopper_route_l2_valid",
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
            ),
            cache_evidence=CacheEvidenceV2(
                "path-planner-v2-hopper-cache-disabled/v1",
                "hopper-cache-disabled/v1",
                False,
            ),
        )


__all__ = (
    "HopperJumpPrimitiveV2",
    "HopperPrimitiveProviderV2",
    "HopperSearchStateV2",
    "hopper_state_key_v2",
    "nominal_hopper_search_state_v2",
)
