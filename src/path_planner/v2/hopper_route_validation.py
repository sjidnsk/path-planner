from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
from math import cos, isfinite, nextafter, pi, sin

from path_planner.v2.contracts import (
    CostBreakdownV2,
    FailureCategoryV2,
    PlanningRequestV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.hopper_authority import (
    HopperGenericInternalSimulationProxyImplementationRecordV2,
    HopperParameterSetRecordV2,
    HopperProviderAuthorityV2,
    _hopper_parameter_set_in_memory_token_v2,
    _hopper_profile_matches_parameter_set_record_v2,
    _lookup_hopper_parameter_set_v2,
    _parameter_set_record_is_exact_v2,
)
from path_planner.v2.oracles.hopper import (
    HOPPER_REPLAY_COUNTER_IDS_V1,
    HopperJumpCandidateV2,
    HopperValidationResultV2,
    validate_hopper_jump_l2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineSafetyAnchorV2


HOPPER_ROUTE_VALIDATOR_ID_V2 = "path-planner-v2-hopper-route-l2/v1"
_PROBABILITY_SCHEMA = "hopper-route-probability-diagnostic/v1"
_RESULT_SCHEMA = "hopper-route-validation-result/v1"
_MAX_ROUTE_STATES = 100_001
_TRUSTED_HOPPER_JUMP_L2 = validate_hopper_jump_l2
_TRUSTED_HOPPER_PARAMETER_LOOKUP = _lookup_hopper_parameter_set_v2
_TRUSTED_HOPPER_PARAMETER_EXACT_CHECK = _parameter_set_record_is_exact_v2
_TRUSTED_HOPPER_PARAMETER_TOKEN = _hopper_parameter_set_in_memory_token_v2
_TRUSTED_HOPPER_PROFILE_RECORD_MATCH = (
    _hopper_profile_matches_parameter_set_record_v2
)


@dataclass(frozen=True, slots=True)
class HopperRouteProbabilityDiagnosticV2:
    per_hop_mass: tuple[float, ...]
    union_bound_lower_mass: float
    schema_version: str = _PROBABILITY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.per_hop_mass) is not tuple or not self.per_hop_mass:
            raise TypeError("per_hop_mass must be a nonempty exact tuple")
        for mass in self.per_hop_mass:
            if type(mass) is not float or not isfinite(mass) or not 0.99 <= mass <= 1.0:
                raise ValueError("per-hop mass must be an exact finite float in [0.99, 1]")
        if (
            type(self.union_bound_lower_mass) is not float
            or not isfinite(self.union_bound_lower_mass)
            or not 0.0 <= self.union_bound_lower_mass <= 1.0
        ):
            raise ValueError("union_bound_lower_mass must be an exact probability")
        if type(self.schema_version) is not str or self.schema_version != _PROBABILITY_SCHEMA:
            raise ValueError("probability diagnostic schema mismatch")


def hopper_route_probability_diagnostic_v2(
    per_hop_mass: tuple[float, ...],
) -> HopperRouteProbabilityDiagnosticV2:
    if type(per_hop_mass) is not tuple or not per_hop_mass:
        raise TypeError("per_hop_mass must be a nonempty exact tuple")
    exact_failure = Fraction(0, 1)
    for mass in per_hop_mass:
        if type(mass) is not float or not isfinite(mass) or not 0.99 <= mass <= 1.0:
            raise ValueError("per-hop mass must be an exact finite float in [0.99, 1]")
        exact_failure += Fraction(1, 1) - Fraction(*mass.as_integer_ratio())
    exact_lower = max(Fraction(0, 1), Fraction(1, 1) - exact_failure)
    rounded = float(exact_lower)
    if Fraction(*rounded.as_integer_ratio()) > exact_lower:
        rounded = nextafter(rounded, float("-inf"))
    return HopperRouteProbabilityDiagnosticV2(per_hop_mass, rounded)


@dataclass(frozen=True, slots=True)
class HopperRouteValidationResultV2:
    passed: bool
    reason_code: str
    category: FailureCategoryV2 | None
    stage: str
    details: tuple[tuple[str, object], ...]
    route_state_count: int
    replay_work_counts: tuple[tuple[str, int], ...]
    l2_peak_accounted_bytes: int
    cost_breakdown: CostBreakdownV2 | None
    probability_diagnostic: HopperRouteProbabilityDiagnosticV2 | None
    route_digest: str | None
    evidence: ValidationEvidenceV2
    schema_version: str = _RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be exact bool")
        if type(self.reason_code) is not str or not self.reason_code:
            raise TypeError("reason_code must be an exact nonempty str")
        if self.category is not None and type(self.category) is not FailureCategoryV2:
            raise TypeError("category must be exact FailureCategoryV2 or None")
        if type(self.stage) is not str or not self.stage:
            raise TypeError("stage must be an exact nonempty str")
        if type(self.details) is not tuple:
            raise TypeError("details must be an exact tuple")
        if type(self.route_state_count) is not int or self.route_state_count < 0:
            raise TypeError("route_state_count must be an exact nonnegative int")
        if type(self.l2_peak_accounted_bytes) is not int or self.l2_peak_accounted_bytes < 0:
            raise TypeError("l2_peak_accounted_bytes must be an exact nonnegative int")
        if type(self.replay_work_counts) is not tuple or len(self.replay_work_counts) != 6:
            raise TypeError("replay_work_counts must contain six counters")
        for position, pair in enumerate(self.replay_work_counts):
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] != HOPPER_REPLAY_COUNTER_IDS_V1[position]
                or type(pair[1]) is not int
                or not 0 <= pair[1] <= 100_000
            ):
                raise ValueError("replay work counter mismatch")
        if type(self.evidence) is not ValidationEvidenceV2:
            raise TypeError("evidence must be exact ValidationEvidenceV2")
        if (
            self.evidence.validator_id != HOPPER_ROUTE_VALIDATOR_ID_V2
            or self.evidence.level is not ValidationLevelV2.L2
            or self.evidence.passed is not self.passed
            or self.evidence.checks != (self.reason_code,)
        ):
            raise ValueError("route evidence mismatch")
        if self.passed:
            if (
                self.reason_code != "hopper_route_l2_valid"
                or self.category is not None
                or self.stage != "route_validation"
                or type(self.cost_breakdown) is not CostBreakdownV2
                or type(self.probability_diagnostic) is not HopperRouteProbabilityDiagnosticV2
                or type(self.route_digest) is not str
                or len(self.route_digest) != 64
            ):
                raise ValueError("route success carrier mismatch")
        elif (
            self.cost_breakdown is not None
            or self.probability_diagnostic is not None
            or self.route_digest is not None
        ):
            raise ValueError("route failure cannot carry success values")
        if type(self.schema_version) is not str or self.schema_version != _RESULT_SCHEMA:
            raise ValueError("route result schema mismatch")


def _zero_counts() -> tuple[tuple[str, int], ...]:
    return tuple((counter_id, 0) for counter_id in HOPPER_REPLAY_COUNTER_IDS_V1)


def _failure(
    reason: str,
    category: FailureCategoryV2,
    stage: str,
    route_state_count: int,
    counts: tuple[tuple[str, int], ...] | None = None,
    *,
    details: tuple[tuple[str, object], ...] = (),
    peak: int = 0,
) -> HopperRouteValidationResultV2:
    return HopperRouteValidationResultV2(
        passed=False,
        reason_code=reason,
        category=category,
        stage=stage,
        details=details,
        route_state_count=route_state_count,
        replay_work_counts=_zero_counts() if counts is None else counts,
        l2_peak_accounted_bytes=peak,
        cost_breakdown=None,
        probability_diagnostic=None,
        route_digest=None,
        evidence=ValidationEvidenceV2(
            HOPPER_ROUTE_VALIDATOR_ID_V2,
            ValidationLevelV2.L2,
            False,
            (reason,),
        ),
    )


def _canonical_direction(index: int) -> tuple[float, float]:
    cardinal = {0: (1.0, 0.0), 4: (0.0, 1.0), 8: (-1.0, 0.0), 12: (0.0, -1.0)}
    if index in cardinal:
        return cardinal[index]
    azimuth = 2.0 * pi * index / 16.0
    return cos(azimuth), sin(azimuth)


def _supported_record(
    authority: object,
) -> tuple[
    object,
    (
        HopperParameterSetRecordV2
        | HopperGenericInternalSimulationProxyImplementationRecordV2
    ),
] | None:
    if type(authority) is not HopperProviderAuthorityV2:
        return None
    try:
        record = _lookup_hopper_parameter_set_v2(authority.parameter_set_id)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return None
    if not _parameter_set_record_is_exact_v2(record):
        return None
    if not _hopper_profile_matches_parameter_set_record_v2(
        authority.hopper_profile,
        record,
    ):
        return None
    try:
        _hopper_parameter_set_in_memory_token_v2(record)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return None
    return authority.hopper_profile, record


def validate_hopper_route_l2(
    route: TypedRouteV2,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    hopper_authority: HopperProviderAuthorityV2,
    deadline: PlanningDeadlineV2,
) -> HopperRouteValidationResultV2:
    route_state_count = len(route.primitives) + 1 if type(route) is TypedRouteV2 else 0
    if (
        type(request) is not PlanningRequestV2
        or type(anchor) is not FineSafetyAnchorV2
        or type(deadline) is not PlanningDeadlineV2
        or anchor.snapshot is not request.terrain_snapshot
    ):
        return _failure(
            "planning_request_contract_mismatch",
            FailureCategoryV2.INTERNAL_ERROR,
            "route_validation",
            route_state_count,
        )
    if (
        validate_hopper_jump_l2 is not _TRUSTED_HOPPER_JUMP_L2
        or _lookup_hopper_parameter_set_v2
        is not _TRUSTED_HOPPER_PARAMETER_LOOKUP
        or _parameter_set_record_is_exact_v2
        is not _TRUSTED_HOPPER_PARAMETER_EXACT_CHECK
        or _hopper_parameter_set_in_memory_token_v2
        is not _TRUSTED_HOPPER_PARAMETER_TOKEN
        or _hopper_profile_matches_parameter_set_record_v2
        is not _TRUSTED_HOPPER_PROFILE_RECORD_MATCH
    ):
        return _failure(
            "hopper_authority_contract_mismatch",
            FailureCategoryV2.INTERNAL_ERROR,
            "route_validation",
            route_state_count,
        )
    supported = _supported_record(hopper_authority)
    if supported is None:
        return _failure(
            "hopper_authority_contract_mismatch",
            FailureCategoryV2.INTERNAL_ERROR,
            "route_validation",
            route_state_count,
        )
    profile, record = supported
    if (
        type(route) is not TypedRouteV2
        or route.platform_kind is not PlatformKindV2.HOPPER
        or route.is_complete is not True
        or not route.primitives
    ):
        return _failure(
            "hopper_route_structure_mismatch",
            FailureCategoryV2.VALIDATION_FAILED,
            "route_validation",
            route_state_count,
        )
    effective_max_states = min(request.resource_budget.max_route_states, _MAX_ROUTE_STATES)
    if route_state_count > effective_max_states:
        return _failure(
            "hopper_route_state_budget_exceeded",
            FailureCategoryV2.RESOURCE_LIMIT,
            "route_validation",
            route_state_count,
            details=(
                ("attempted_route_states", route_state_count),
                ("effective_max_route_states", effective_max_states),
                ("requested_max_route_states", request.resource_budget.max_route_states),
            ),
        )

    provider = import_module("path_planner.v2.providers.hopper")
    primitive_type = provider.HopperJumpPrimitiveV2
    state_type = provider.HopperSearchStateV2
    counts_max = [0] * 6
    masses: list[float] = []
    distance_cost = 0.0
    risk_cost = 0.0
    energy_cost = 0.0
    time_cost = 0.0
    previous_end = request.start_state

    for primitive in route.primitives:
        if type(primitive) is not primitive_type:
            return _failure(
                "hopper_primitive_structure_mismatch",
                FailureCategoryV2.VALIDATION_FAILED,
                "route_validation",
                route_state_count,
                tuple((name, counts_max[i]) for i, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)),
            )
        if (
            primitive.kind is not PrimitiveKindV2.BALLISTIC_JUMP
            or primitive.validation_level is not ValidationLevelV2.L2
            or type(primitive.start_hopper_state) is not state_type
            or type(primitive.end_hopper_state) is not state_type
            or primitive.start_state != primitive.start_hopper_state.nominal_state
            or primitive.end_state != primitive.end_hopper_state.nominal_state
            or primitive.start_state != previous_end
            or primitive.parameter_set_id != record.parameter_set_id
        ):
            return _failure(
                "hopper_primitive_contract_mismatch",
                FailureCategoryV2.VALIDATION_FAILED,
                "route_validation",
                route_state_count,
                tuple((name, counts_max[i]) for i, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)),
            )
        speed = profile.launch_speeds_mps[primitive.speed_index]
        elevation = profile.launch_elevations_rad[primitive.elevation_index]
        horizontal_speed = speed * cos(elevation)
        vertical_speed = speed * sin(elevation)
        vertical_time_scale = vertical_speed / profile.gravity_mps2
        flight_time = 2.0 * vertical_time_scale
        x_direction, y_direction = _canonical_direction(primitive.azimuth_index)
        dx = (horizontal_speed * x_direction) * flight_time
        dy = (horizontal_speed * y_direction) * flight_time
        expected_end = PoseStateV2(
            primitive.start_state.x_m + dx,
            primitive.start_state.y_m + dy,
            primitive.start_state.heading_rad,
        )
        distance = horizontal_speed * flight_time
        try:
            _hopper_parameter_set_in_memory_token_v2(record)
            expected_energy = record.energy_evaluator(speed)
            _hopper_parameter_set_in_memory_token_v2(record)
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            return _failure(
                "hopper_authority_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "route_validation",
                route_state_count,
                tuple(
                    (name, counts_max[index])
                    for index, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)
                ),
            )
        if (
            primitive.end_state != expected_end
            or primitive.start_hopper_state.support_height_m
            != primitive.end_hopper_state.support_height_m
            or primitive.duration_s != flight_time
            or primitive.distance_m != distance
            or primitive.energy_cost != expected_energy
            or primitive.observation_contribution != 0.0
        ):
            return _failure(
                "hopper_primitive_contract_mismatch",
                FailureCategoryV2.VALIDATION_FAILED,
                "route_validation",
                route_state_count,
                tuple((name, counts_max[i]) for i, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)),
            )
        candidate = HopperJumpCandidateV2(
            primitive.start_state,
            primitive.start_hopper_state.support_height_m,
            profile,
            record.parameter_set_id,
            primitive.speed_index,
            primitive.elevation_index,
            primitive.azimuth_index,
            "hopper-jump-candidate/v1",
        )
        oracle = validate_hopper_jump_l2(candidate, anchor, deadline)
        if (
            validate_hopper_jump_l2 is not _TRUSTED_HOPPER_JUMP_L2
            or type(oracle) is not HopperValidationResultV2
        ):
            return _failure(
                "hopper_authority_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "route_validation",
                route_state_count,
                tuple(
                    (name, counts_max[index])
                    for index, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)
                ),
            )
        for index, (_counter_id, count) in enumerate(oracle.replay_work_counts):
            counts_max[index] = max(counts_max[index], count)
        frozen_counts = tuple(
            (name, counts_max[index])
            for index, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)
        )
        if oracle.reason_code != "hopper_jump_l2_valid":
            return _failure(
                oracle.reason_code,
                oracle.category or FailureCategoryV2.VALIDATION_FAILED,
                oracle.stage,
                route_state_count,
                frozen_counts,
                details=oracle.details,
            )
        if (
            type(oracle.selected_landing_mass) is not float
            or primitive.selected_landing_mass != oracle.selected_landing_mass
        ):
            return _failure(
                "hopper_primitive_contract_mismatch",
                FailureCategoryV2.VALIDATION_FAILED,
                "route_validation",
                route_state_count,
                frozen_counts,
            )
        masses.append(oracle.selected_landing_mass)
        objective = request.objective_profile
        distance_cost += objective.distance_weight * distance
        energy_cost += objective.energy_weight * expected_energy
        time_cost += objective.time_weight * flight_time
        previous_end = primitive.end_state

    frozen_counts = tuple(
        (name, counts_max[index])
        for index, name in enumerate(HOPPER_REPLAY_COUNTER_IDS_V1)
    )
    if route.primitives[0].start_state != request.start_state:
        return _failure(
            "hopper_route_start_mismatch",
            FailureCategoryV2.VALIDATION_FAILED,
            "route_validation",
            route_state_count,
            frozen_counts,
        )
    if previous_end != request.goal_state:
        return _failure(
            "hopper_route_goal_mismatch",
            FailureCategoryV2.VALIDATION_FAILED,
            "route_validation",
            route_state_count,
            frozen_counts,
        )
    total_cost = sum((distance_cost, risk_cost, energy_cost, time_cost))
    if route.total_cost != total_cost:
        return _failure(
            "hopper_route_cost_contract_mismatch",
            FailureCategoryV2.VALIDATION_FAILED,
            "route_validation",
            route_state_count,
            frozen_counts,
        )
    costs = CostBreakdownV2(
        distance_cost,
        risk_cost,
        energy_cost,
        time_cost,
        total_cost,
    )
    diagnostic = hopper_route_probability_diagnostic_v2(tuple(masses))
    digest = sha256(canonical_json_bytes(route)).hexdigest()
    peak = 4_096 + 512 * len(route.primitives) + 384_065_536
    return HopperRouteValidationResultV2(
        passed=True,
        reason_code="hopper_route_l2_valid",
        category=None,
        stage="route_validation",
        details=(),
        route_state_count=route_state_count,
        replay_work_counts=frozen_counts,
        l2_peak_accounted_bytes=peak,
        cost_breakdown=costs,
        probability_diagnostic=diagnostic,
        route_digest=digest,
        evidence=ValidationEvidenceV2(
            HOPPER_ROUTE_VALIDATOR_ID_V2,
            ValidationLevelV2.L2,
            True,
            ("hopper_route_l2_valid",),
        ),
    )
