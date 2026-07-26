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
    HopperGenericInternalSimulationProxyImplementationRecordV2,
    HopperParameterSetRecordV2,
    HopperProviderAuthorityV2,
    _hopper_parameter_set_in_memory_token_v2,
    _hopper_profile_matches_parameter_set_record_v2,
    _lookup_hopper_parameter_set_v2,
    _parameter_set_record_is_exact_v2,
    _require_canonical_resource_authority_v2,
)
from path_planner.v2.oracles.hopper import (
    HopperJumpCandidateV2,
    HopperValidationResultV2,
    validate_hopper_jump_l2,
)
from path_planner.v2.profiles import audit_hopper_profile_v2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.search import SearchQueueEntryV2, StableSearchQueueV2
from path_planner.v2.terrain import FineSafetyAnchorV2


HOPPER_SEARCH_STATE_SCHEMA_V2 = "hopper-nominal-mean-search-state/v1"
HOPPER_JUMP_PRIMITIVE_SCHEMA_V2 = "hopper-jump-primitive/v1"
HOPPER_STATE_KEY_TAG_V1 = 0x484F505045525631

_HOPPER_SPEED_COUNT_V2 = 4
_HOPPER_ELEVATION_COUNT_V2 = 3
_HOPPER_AZIMUTH_COUNT_V2 = 16
_MIN_SELECTED_LANDING_MASS_V2 = 0.99
_TRUSTED_HOPPER_JUMP_L2_V2 = validate_hopper_jump_l2
_TRUSTED_HOPPER_PARAMETER_LOOKUP_V2 = _lookup_hopper_parameter_set_v2
_TRUSTED_HOPPER_PARAMETER_EXACT_CHECK_V2 = _parameter_set_record_is_exact_v2
_TRUSTED_HOPPER_PARAMETER_TOKEN_V2 = _hopper_parameter_set_in_memory_token_v2
_TRUSTED_HOPPER_PROFILE_RECORD_MATCH_V2 = (
    _hopper_profile_matches_parameter_set_record_v2
)


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
) -> (
    HopperParameterSetRecordV2
    | HopperGenericInternalSimulationProxyImplementationRecordV2
    | None
):
    record = _lookup_hopper_parameter_set_v2(authority.parameter_set_id)
    if not _parameter_set_record_is_exact_v2(record):
        return None
    _hopper_parameter_set_in_memory_token_v2(record)
    return record


def _provider_profile_matches_record_v2(
    authority: HopperProviderAuthorityV2,
    record: (
        HopperParameterSetRecordV2
        | HopperGenericInternalSimulationProxyImplementationRecordV2
    ),
) -> bool:
    return _hopper_profile_matches_parameter_set_record_v2(
        authority.hopper_profile,
        record,
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
class _HopperSearchMemoryLedgerV2:
    accounting_id: str
    requested_max_memory_bytes: int
    effective_max_memory_bytes: int
    admitted_record_count: int
    persistent_peak_bytes: int
    transient_reservation_peak_bytes: int
    combined_accounted_peak_bytes: int
    route_materialization_peak_bytes: int
    schema_version: str = "hopper-search-memory-ledger/v1"

    def __post_init__(self) -> None:
        if type(self.accounting_id) is not str or not self.accounting_id:
            raise TypeError("accounting_id must be exact nonempty str")
        for name in (
            "requested_max_memory_bytes",
            "effective_max_memory_bytes",
            "admitted_record_count",
            "persistent_peak_bytes",
            "transient_reservation_peak_bytes",
            "combined_accounted_peak_bytes",
            "route_materialization_peak_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative int")
        if self.schema_version != "hopper-search-memory-ledger/v1":
            raise ValueError("memory ledger schema mismatch")


@dataclass(frozen=True, slots=True)
class _HopperNodeRecordV2:
    candidate_id: str
    state: HopperSearchStateV2
    state_key: tuple[int, ...]
    depth: int
    route_state_count: int
    distance_cost: float
    energy_cost: float
    time_cost: float
    total_cost: float
    parent_candidate_id: str | None
    incoming_action: tuple[int, int, int] | None
    selected_landing_mass: float | None
    duration_s: float
    distance_m: float
    energy_resource: float


_HOPPER_SEMANTIC_REASON_RANK_V2 = {
    "hopper_launch_unknown": 10,
    "hopper_launch_unsafe": 11,
    "hopper_arc_boundary_violation": 20,
    "hopper_arc_unknown": 21,
    "hopper_arc_clearance_violation": 22,
    "hopper_landing_probability_below_threshold": 30,
    "hopper_landing_zone_unknown": 40,
    "hopper_landing_zone_unsafe": 41,
    "hopper_landing_slope_exceeded": 42,
    "hopper_landing_height_unreachable": 43,
    "hopper_landing_theta_unreachable": 50,
    "hopper_stop_condition_failed": 60,
}


def _search_persistent_bytes_v2(record_count: int) -> int:
    authority = HOPPER_RESOURCE_AUTHORITY_V2
    return authority.search_base_bytes + authority.search_record_bytes * record_count


def _route_materialization_bytes_v2(hop_count: int) -> int:
    authority = HOPPER_RESOURCE_AUTHORITY_V2
    return authority.route_build_base_bytes + authority.route_primitive_bytes * hop_count


def _action_transient_reservations_v2() -> tuple[tuple[str, int], ...]:
    authority = HOPPER_RESOURCE_AUTHORITY_V2
    ballistic = authority.ballistic_build_base_bytes + (
        authority.ballistic_build_per_sample_bytes
        * authority.max_ballistic_samples
    )
    exact_slot_bytes = authority.exact_integer_slot_overhead_bytes + (
        authority.max_exact_integer_bits + 7
    ) // 8
    arc = authority.ballistic_retained_per_sample_bytes * 2 + (
        authority.exact_base_bytes
        + authority.max_exact_live_integer_slots * exact_slot_bytes
        + authority.max_replay_steps * authority.exact_distinct_cell_bytes
    )
    landing = authority.landing_build_base_bytes + (
        authority.landing_build_per_candidate_bytes
        * authority.max_landing_candidates
    )
    return (
        ("ballistic_build", ballistic),
        ("arc_oracle", arc),
        ("landing_build", landing),
    )


def _updated_memory_ledger_v2(
    ledger: _HopperSearchMemoryLedgerV2,
    *,
    admitted_record_count: int | None = None,
    persistent_bytes: int,
    transient_bytes: int = 0,
    route_materialization_bytes: int = 0,
) -> _HopperSearchMemoryLedgerV2:
    combined = persistent_bytes + transient_bytes
    return _HopperSearchMemoryLedgerV2(
        accounting_id=ledger.accounting_id,
        requested_max_memory_bytes=ledger.requested_max_memory_bytes,
        effective_max_memory_bytes=ledger.effective_max_memory_bytes,
        admitted_record_count=(
            ledger.admitted_record_count
            if admitted_record_count is None
            else admitted_record_count
        ),
        persistent_peak_bytes=max(ledger.persistent_peak_bytes, persistent_bytes),
        transient_reservation_peak_bytes=max(
            ledger.transient_reservation_peak_bytes,
            transient_bytes,
        ),
        combined_accounted_peak_bytes=max(
            ledger.combined_accounted_peak_bytes,
            combined,
        ),
        route_materialization_peak_bytes=max(
            ledger.route_materialization_peak_bytes,
            route_materialization_bytes,
        ),
    )


def _memory_failure_details_v2(
    ledger: _HopperSearchMemoryLedgerV2,
    *,
    persistent_bytes: int,
    transient_bytes: int,
    phase: str,
    attempted_bytes: int | None = None,
) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    return (
        ("accounting_id", ledger.accounting_id),
        ("admitted_record_count", ledger.admitted_record_count),
        (
            "attempted_accounted_bytes",
            persistent_bytes + transient_bytes
            if attempted_bytes is None
            else attempted_bytes,
        ),
        ("effective_max_memory_bytes", ledger.effective_max_memory_bytes),
        ("max_memory_bytes", ledger.requested_max_memory_bytes),
        ("persistent_accounted_bytes", persistent_bytes),
        ("phase", phase),
        ("transient_reserved_bytes", transient_bytes),
    )


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

        try:
            _require_canonical_resource_authority_v2(
                HOPPER_RESOURCE_AUTHORITY_V2
            )
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_authority_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "provider_authority",
                details=(
                    ("actual", "hopper_authority_contract_mismatch"),
                    ("expected", "canonical_hopper_authority"),
                    ("phase", "provider_authority"),
                ),
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
        try:
            record = _provider_record_v2(self.hopper_authority)
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_authority_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "provider_authority",
            )
        if (
            validate_hopper_jump_l2 is not _TRUSTED_HOPPER_JUMP_L2_V2
            or _lookup_hopper_parameter_set_v2
            is not _TRUSTED_HOPPER_PARAMETER_LOOKUP_V2
            or _parameter_set_record_is_exact_v2
            is not _TRUSTED_HOPPER_PARAMETER_EXACT_CHECK_V2
            or _hopper_parameter_set_in_memory_token_v2
            is not _TRUSTED_HOPPER_PARAMETER_TOKEN_V2
            or _hopper_profile_matches_parameter_set_record_v2
            is not _TRUSTED_HOPPER_PROFILE_RECORD_MATCH_V2
        ):
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_authority_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "provider_authority",
            )
        if record is None:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_parameter_set_unsupported",
                FailureCategoryV2.UNSUPPORTED_CAPABILITY,
                "capability_preflight",
            )
        if not _provider_profile_matches_record_v2(self.hopper_authority, record):
            if (
                type(record)
                is HopperGenericInternalSimulationProxyImplementationRecordV2
            ):
                return _provider_failure_v2(
                    request,
                    deadline,
                    "hopper_authority_contract_mismatch",
                    FailureCategoryV2.INTERNAL_ERROR,
                    "provider_authority",
                )
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
                "search_expansion",
                details=(
                    ("attempted_expanded_states", 1),
                    ("max_expanded_states", 0),
                ),
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

        resource_authority = HOPPER_RESOURCE_AUTHORITY_V2
        requested_memory = request.resource_budget.max_memory_bytes
        effective_memory = (
            resource_authority.hard_accounted_memory_bytes
            if requested_memory == 0
            else min(
                requested_memory,
                resource_authority.hard_accounted_memory_bytes,
            )
        )
        memory_ledger = _HopperSearchMemoryLedgerV2(
            resource_authority.memory_accounting_id,
            requested_memory,
            effective_memory,
            0,
            0,
            0,
            0,
            0,
        )
        root_persistent = _search_persistent_bytes_v2(1)
        if root_persistent > effective_memory:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_memory_budget_exceeded",
                FailureCategoryV2.RESOURCE_LIMIT,
                "search_memory",
                details=_memory_failure_details_v2(
                    memory_ledger,
                    persistent_bytes=_search_persistent_bytes_v2(0),
                    transient_bytes=0,
                    phase="root_admission",
                    attempted_bytes=root_persistent,
                ),
            )
        memory_ledger = _updated_memory_ledger_v2(
            memory_ledger,
            admitted_record_count=1,
            persistent_bytes=root_persistent,
        )

        expanded_states = 0
        generated_primitives = 0
        rejected_l2 = 0
        start_hopper = HopperSearchStateV2(request.start_state, 0.0)
        objective = request.objective_profile
        start_key = hopper_state_key_v2(start_hopper)
        root_id = "hopper-node-00000000000000000000"
        root_node = _HopperNodeRecordV2(
            root_id,
            start_hopper,
            start_key,
            0,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            None,
            0.0,
            0.0,
            0.0,
        )
        queue = StableSearchQueueV2()
        try:
            queue.extend(
                (
                    SearchQueueEntryV2(
                        root_id,
                        start_key,
                        "hopper-root/v1",
                        0.0,
                        0.0,
                        (),
                    ),
                )
            )
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_search_cost_contract_mismatch",
                FailureCategoryV2.INTERNAL_ERROR,
                "search_setup",
                details=(
                    ("actual", "root_queue_rejected"),
                    ("expected", "stable_search_queue_entry"),
                    ("phase", "search_setup"),
                ),
            )

        nodes: dict[str, _HopperNodeRecordV2] = {root_id: root_node}
        best_by_state: dict[tuple[int, ...], tuple[float, str]] = {
            start_key: (0.0, root_id)
        }
        closed: set[tuple[int, ...]] = set()
        serial = 1
        goal_node: _HopperNodeRecordV2 | None = None
        semantic_representative: tuple[
            tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], str],
            str,
            str,
            int,
            int | None,
            int | None,
            int | None,
        ] | None = None
        action_reservations = _action_transient_reservations_v2()

        while len(queue) > 0:
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
            try:
                entry = queue.pop_anchor()
                node = nodes[entry.candidate_id]
            except (KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:
                return _provider_failure_v2(
                    request,
                    deadline,
                    "hopper_search_cost_contract_mismatch",
                    FailureCategoryV2.INTERNAL_ERROR,
                    "search_expansion",
                    expanded_states=expanded_states,
                    generated_primitives=generated_primitives,
                    rejected_l2=rejected_l2,
                    details=(
                        ("actual", "queue_node_mismatch"),
                        ("expected", "admitted_node"),
                        ("phase", "search_expansion"),
                    ),
                )
            best = best_by_state.get(entry.state_key)
            if best != (entry.path_cost, entry.candidate_id):
                continue
            if entry.state_key in closed:
                continue
            if node.state.nominal_state == request.goal_state:
                goal_node = node
                break
            if expanded_states >= request.resource_budget.max_expanded_states:
                return _provider_failure_v2(
                    request,
                    deadline,
                    "hopper_expansion_budget_exhausted",
                    FailureCategoryV2.RESOURCE_LIMIT,
                    "search_expansion",
                    expanded_states=expanded_states,
                    generated_primitives=generated_primitives,
                    rejected_l2=rejected_l2,
                    details=(
                        ("attempted_expanded_states", expanded_states + 1),
                        (
                            "max_expanded_states",
                            request.resource_budget.max_expanded_states,
                        ),
                    ),
                )
            closed.add(entry.state_key)
            expanded_states += 1

            for speed_index in range(4):
                for elevation_index in range(3):
                    for azimuth_index in range(16):
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
                        persistent = _search_persistent_bytes_v2(
                            memory_ledger.admitted_record_count
                        )
                        for phase, transient in action_reservations:
                            if persistent + transient > effective_memory:
                                return _provider_failure_v2(
                                    request,
                                    deadline,
                                    "hopper_memory_budget_exceeded",
                                    FailureCategoryV2.RESOURCE_LIMIT,
                                    "search_memory",
                                    expanded_states=expanded_states,
                                    generated_primitives=generated_primitives,
                                    rejected_l2=rejected_l2,
                                    details=_memory_failure_details_v2(
                                        memory_ledger,
                                        persistent_bytes=persistent,
                                        transient_bytes=transient,
                                        phase=phase,
                                    ),
                                )
                            memory_ledger = _updated_memory_ledger_v2(
                                memory_ledger,
                                persistent_bytes=persistent,
                                transient_bytes=transient,
                            )

                        candidate = HopperJumpCandidateV2(
                            node.state.nominal_state,
                            node.state.support_height_m,
                            profile,
                            record.parameter_set_id,
                            speed_index,
                            elevation_index,
                            azimuth_index,
                            "hopper-jump-candidate/v1",
                        )
                        generated_primitives += 1
                        try:
                            validation = validate_hopper_jump_l2(
                                candidate,
                                anchor,
                                deadline,
                            )
                        except (KeyboardInterrupt, MemoryError, SystemExit):
                            raise
                        except Exception:
                            validation = None
                        if validate_hopper_jump_l2 is not _TRUSTED_HOPPER_JUMP_L2_V2:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_authority_contract_mismatch",
                                FailureCategoryV2.INTERNAL_ERROR,
                                "provider_authority",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                            )
                        if type(validation) is not HopperValidationResultV2:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_jump_oracle_contract_mismatch",
                                FailureCategoryV2.INTERNAL_ERROR,
                                "search_expansion",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=(
                                    ("actual", "malformed_or_exception"),
                                    ("expected", "hopper-jump-validation-result/v1"),
                                    ("phase", "search_expansion"),
                                ),
                            )
                        if validation.reason_code != "hopper_jump_l2_valid":
                            if (
                                validation.category
                                is FailureCategoryV2.VALIDATION_FAILED
                            ):
                                rejected_l2 += 1
                                reason_rank = _HOPPER_SEMANTIC_REASON_RANK_V2[
                                    validation.reason_code
                                ]
                                segment = validation.segment_index
                                cell_y = (
                                    None
                                    if validation.failed_cell is None
                                    else validation.failed_cell.y
                                )
                                cell_x = (
                                    None
                                    if validation.failed_cell is None
                                    else validation.failed_cell.x
                                )
                                action_key = (
                                    f"{speed_index:02d}:"
                                    f"{elevation_index:02d}:"
                                    f"{azimuth_index:02d}"
                                )
                                rank_key = (
                                    reason_rank,
                                    (1, 0) if segment is None else (0, segment),
                                    (1, 0) if cell_y is None else (0, cell_y),
                                    (1, 0) if cell_x is None else (0, cell_x),
                                    action_key,
                                )
                                representative = (
                                    rank_key,
                                    validation.reason_code,
                                    action_key,
                                    node.depth,
                                    segment,
                                    cell_y,
                                    cell_x,
                                )
                                if (
                                    semantic_representative is None
                                    or representative[0]
                                    < semantic_representative[0]
                                ):
                                    semantic_representative = representative
                                continue
                            details = validation.details
                            if validation.reason_code == "planning_deadline_expired":
                                details = ()
                            elif validation.reason_code not in (
                                "hopper_replay_work_budget_exceeded",
                            ):
                                details = (
                                    ("actual", validation.reason_code),
                                    ("expected", "hopper_jump_l2_valid"),
                                    ("phase", validation.stage),
                                )
                            return _provider_failure_v2(
                                request,
                                deadline,
                                validation.reason_code,
                                validation.category
                                or FailureCategoryV2.INTERNAL_ERROR,
                                validation.stage,
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=details,
                            )
                        selected_mass = validation.selected_landing_mass
                        if (
                            type(selected_mass) is not float
                            or selected_mass < _MIN_SELECTED_LANDING_MASS_V2
                        ):
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_jump_oracle_contract_mismatch",
                                FailureCategoryV2.INTERNAL_ERROR,
                                "search_expansion",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=(
                                    ("actual", "invalid_selected_landing_mass"),
                                    ("expected", "mass_at_least_0.99"),
                                    ("phase", "search_expansion"),
                                ),
                            )

                        speed = profile.launch_speeds_mps[speed_index]
                        elevation = profile.launch_elevations_rad[elevation_index]
                        horizontal_speed = speed * cos(elevation)
                        vertical_speed = speed * sin(elevation)
                        flight_time = 2.0 * (
                            vertical_speed / profile.gravity_mps2
                        )
                        x_direction, y_direction = _provider_direction_v2(
                            azimuth_index
                        )
                        dx = (horizontal_speed * x_direction) * flight_time
                        dy = (horizontal_speed * y_direction) * flight_time
                        distance = horizontal_speed * flight_time
                        end_pose = PoseStateV2(
                            node.state.nominal_state.x_m + dx,
                            node.state.nominal_state.y_m + dy,
                            node.state.nominal_state.heading_rad,
                        )
                        end_hopper = HopperSearchStateV2(
                            end_pose,
                            node.state.support_height_m,
                        )
                        try:
                            _hopper_parameter_set_in_memory_token_v2(record)
                            energy = record.energy_evaluator(speed)
                            _hopper_parameter_set_in_memory_token_v2(record)
                            distance_cost = node.distance_cost + (
                                objective.distance_weight * distance
                            )
                            energy_cost = node.energy_cost + (
                                objective.energy_weight * energy
                            )
                            time_cost = node.time_cost + (
                                objective.time_weight * flight_time
                            )
                            total = sum(
                                (distance_cost, 0.0, energy_cost, time_cost)
                            )
                            if any(
                                type(value) is not float
                                or not isfinite(value)
                                or value < 0.0
                                for value in (
                                    energy,
                                    distance_cost,
                                    energy_cost,
                                    time_cost,
                                    total,
                                )
                            ):
                                raise ValueError
                        except (KeyboardInterrupt, MemoryError, SystemExit):
                            raise
                        except Exception:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_numeric_contract_mismatch",
                                FailureCategoryV2.INTERNAL_ERROR,
                                "search_expansion",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=(
                                    ("actual", "noncanonical_search_cost"),
                                    ("expected", "finite_nonnegative_binary64"),
                                    ("phase", "search_expansion"),
                                ),
                            )

                        child_route_states = node.route_state_count + 1
                        if child_route_states > effective_route_states:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_route_state_budget_exceeded",
                                FailureCategoryV2.RESOURCE_LIMIT,
                                "route_state_admission",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=(
                                    (
                                        "attempted_route_states",
                                        child_route_states,
                                    ),
                                    (
                                        "effective_max_route_states",
                                        effective_route_states,
                                    ),
                                    (
                                        "requested_max_route_states",
                                        request.resource_budget.max_route_states,
                                    ),
                                ),
                            )
                        child_key = hopper_state_key_v2(end_hopper)
                        candidate_id = f"hopper-node-{serial:020d}"
                        serial += 1
                        action_key = (
                            f"{speed_index:02d}:"
                            f"{elevation_index:02d}:"
                            f"{azimuth_index:02d}"
                        )
                        existing = best_by_state.get(child_key)
                        if child_key in closed:
                            if existing is not None and total < existing[0]:
                                return _provider_failure_v2(
                                    request,
                                    deadline,
                                    "hopper_search_cost_contract_mismatch",
                                    FailureCategoryV2.INTERNAL_ERROR,
                                    "search_expansion",
                                    expanded_states=expanded_states,
                                    generated_primitives=generated_primitives,
                                    rejected_l2=rejected_l2,
                                    details=(
                                        ("actual", "lower_cost_closed_state"),
                                        ("expected", "dijkstra_monotonic_cost"),
                                        ("phase", "search_expansion"),
                                    ),
                                )
                            continue
                        if existing is not None and total >= existing[0]:
                            continue

                        prospective_count = (
                            memory_ledger.admitted_record_count + 1
                        )
                        prospective_persistent = _search_persistent_bytes_v2(
                            prospective_count
                        )
                        if prospective_persistent > effective_memory:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_memory_budget_exceeded",
                                FailureCategoryV2.RESOURCE_LIMIT,
                                "search_memory",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=_memory_failure_details_v2(
                                    memory_ledger,
                                    persistent_bytes=_search_persistent_bytes_v2(
                                        memory_ledger.admitted_record_count
                                    ),
                                    transient_bytes=0,
                                    phase="child_admission",
                                    attempted_bytes=prospective_persistent,
                                ),
                            )
                        memory_ledger = _updated_memory_ledger_v2(
                            memory_ledger,
                            admitted_record_count=prospective_count,
                            persistent_bytes=prospective_persistent,
                        )
                        child_node = _HopperNodeRecordV2(
                            candidate_id,
                            end_hopper,
                            child_key,
                            node.depth + 1,
                            child_route_states,
                            distance_cost,
                            energy_cost,
                            time_cost,
                            total,
                            node.candidate_id,
                            (speed_index, elevation_index, azimuth_index),
                            selected_mass,
                            flight_time,
                            distance,
                            energy,
                        )
                        nodes[candidate_id] = child_node
                        best_by_state[child_key] = (total, candidate_id)
                        try:
                            queue.extend(
                                (
                                    SearchQueueEntryV2(
                                        candidate_id,
                                        child_key,
                                        action_key,
                                        total,
                                        0.0,
                                        (),
                                    ),
                                )
                            )
                        except (KeyboardInterrupt, MemoryError, SystemExit):
                            raise
                        except Exception:
                            return _provider_failure_v2(
                                request,
                                deadline,
                                "hopper_search_cost_contract_mismatch",
                                FailureCategoryV2.INTERNAL_ERROR,
                                "search_expansion",
                                expanded_states=expanded_states,
                                generated_primitives=generated_primitives,
                                rejected_l2=rejected_l2,
                                details=(
                                    ("actual", "child_queue_rejected"),
                                    ("expected", "stable_search_queue_entry"),
                                    ("phase", "search_expansion"),
                                ),
                            )

        if goal_node is None:
            representative_details: tuple[
                tuple[str, str | int | float | bool | None], ...
            ]
            if semantic_representative is None:
                representative_details = (
                    ("action_key", None),
                    ("actual", None),
                    ("candidate_id", None),
                    ("cell_x", None),
                    ("cell_y", None),
                    ("hop_index", None),
                    ("parameter_set_id", record.parameter_set_id),
                    ("phase", "search_exhaustion"),
                    ("reason_rank", None),
                    ("segment_index", None),
                )
            else:
                (
                    rank_key,
                    reason,
                    action_key,
                    hop_index,
                    segment_index,
                    cell_y,
                    cell_x,
                ) = semantic_representative
                representative_details = (
                    ("action_key", action_key),
                    ("actual", reason),
                    ("candidate_id", None),
                    ("cell_x", cell_x),
                    ("cell_y", cell_y),
                    ("hop_index", hop_index),
                    ("parameter_set_id", record.parameter_set_id),
                    ("phase", "search_exhaustion"),
                    ("reason_rank", rank_key[0]),
                    ("segment_index", segment_index),
                )
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_no_complete_route",
                FailureCategoryV2.NO_COMPLETE_ROUTE,
                "hopper_search",
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
                details=representative_details,
            )

        hop_count = goal_node.depth
        persistent = _search_persistent_bytes_v2(
            memory_ledger.admitted_record_count
        )
        route_reservation = _route_materialization_bytes_v2(hop_count)
        if persistent + route_reservation > effective_memory:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_memory_budget_exceeded",
                FailureCategoryV2.RESOURCE_LIMIT,
                "search_memory",
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
                details=_memory_failure_details_v2(
                    memory_ledger,
                    persistent_bytes=persistent,
                    transient_bytes=route_reservation,
                    phase="route_materialization",
                ),
            )
        memory_ledger = _updated_memory_ledger_v2(
            memory_ledger,
            persistent_bytes=persistent,
            transient_bytes=route_reservation,
            route_materialization_bytes=persistent + route_reservation,
        )

        lineage: list[_HopperNodeRecordV2] = []
        cursor = goal_node
        while cursor.parent_candidate_id is not None:
            lineage.append(cursor)
            cursor = nodes[cursor.parent_candidate_id]
        lineage.reverse()
        primitives: list[HopperJumpPrimitiveV2] = []
        parent = root_node
        for child in lineage:
            if child.incoming_action is None or child.selected_landing_mass is None:
                return _provider_failure_v2(
                    request,
                    deadline,
                    "hopper_primitive_structure_mismatch",
                    FailureCategoryV2.INTERNAL_ERROR,
                    "route_validation",
                    expanded_states=expanded_states,
                    generated_primitives=generated_primitives,
                    rejected_l2=rejected_l2,
                    details=(
                        ("actual", "missing_lineage_action"),
                        ("expected", "complete_hopper_lineage"),
                        ("phase", "route_validation"),
                    ),
                )
            speed_index, elevation_index, azimuth_index = child.incoming_action
            primitives.append(
                HopperJumpPrimitiveV2(
                    kind=PrimitiveKindV2.BALLISTIC_JUMP,
                    start_state=parent.state.nominal_state,
                    end_state=child.state.nominal_state,
                    duration_s=child.duration_s,
                    distance_m=child.distance_m,
                    energy_cost=child.energy_resource,
                    observation_contribution=0.0,
                    validation_level=ValidationLevelV2.L2,
                    start_hopper_state=parent.state,
                    end_hopper_state=child.state,
                    speed_index=speed_index,
                    elevation_index=elevation_index,
                    azimuth_index=azimuth_index,
                    parameter_set_id=record.parameter_set_id,
                    selected_landing_mass=child.selected_landing_mass,
                )
            )
            parent = child
        route = TypedRouteV2(
            PlatformKindV2.HOPPER,
            tuple(primitives),
            goal_node.total_cost,
            True,
        )

        complete_reservation = _route_materialization_bytes_v2(hop_count)
        route_l2_transient = max(value for _phase, value in action_reservations)
        if complete_reservation + route_l2_transient > effective_memory:
            return _provider_failure_v2(
                request,
                deadline,
                "hopper_memory_budget_exceeded",
                FailureCategoryV2.RESOURCE_LIMIT,
                "search_memory",
                expanded_states=expanded_states,
                generated_primitives=generated_primitives,
                rejected_l2=rejected_l2,
                details=_memory_failure_details_v2(
                    memory_ledger,
                    persistent_bytes=complete_reservation,
                    transient_bytes=route_l2_transient,
                    phase="route_l2",
                ),
            )
        memory_ledger = _updated_memory_ledger_v2(
            memory_ledger,
            persistent_bytes=complete_reservation,
            transient_bytes=route_l2_transient,
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
                sample_states=(
                    primitives[0].start_state,
                    *(primitive.end_state for primitive in primitives),
                ),
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
