"""Internal Hopper dispatch and API postcondition authority for v2.

This module deliberately has no public exports.  The generic API owns endpoint
ordering and provider resolution; everything after explicit Hopper dispatch is
kept here so wheel and legged policy cannot accidentally inherit Hopper rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import copysign, isfinite
import path_planner.v2.hopper_authority as _authority_module
import path_planner.v2.hopper_route_validation as _route_validation_module
import path_planner.v2.providers.hopper as _provider_module
from path_planner.v2.contracts import (
    CacheEvidenceV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningOutcomeV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    SearchTelemetryV2,
    TypedRouteV2,
)
from path_planner.v2.hopper_authority import (
    HOPPER_RESOURCE_AUTHORITY_V2,
    HopperParameterSetRecordV2,
    HopperProviderAuthorityV2,
    _hopper_parameter_set_in_memory_token_v2,
    _hopper_resource_authority_in_memory_token_v2,
    _lookup_hopper_parameter_set_v2,
    _require_canonical_resource_authority_v2,
)
from path_planner.v2.hopper_route_validation import (
    HopperRouteValidationResultV2,
    validate_hopper_route_l2,
)
from path_planner.v2.profiles import HopperProfileV2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineSafetyAnchorV2, snapshot_hash


__all__ = ()

_COMPOSITE_SCHEMA = "hopper-api-composite-authority/v1"
_TRUSTED_JUMP_ORACLE = _provider_module.validate_hopper_jump_l2
_TRUSTED_ROUTE_L2 = validate_hopper_route_l2

_AUTHORITY_L2_REASONS = frozenset(
    {
        "planning_deadline_contract_mismatch",
        "planning_request_contract_mismatch",
        "hopper_authority_contract_mismatch",
        "hopper_profile_contract_mismatch",
        "terrain_snapshot_identity_mismatch",
        "terrain_snapshot_hash_mismatch",
        "hopper_terrain_geometry_contract_mismatch",
    }
)
_VALIDATOR_L2_REASONS = frozenset(
    {
        "terrain_query_contract_mismatch",
        "hopper_numeric_contract_mismatch",
        "hopper_jump_oracle_contract_mismatch",
        "hopper_route_oracle_contract_mismatch",
    }
)
_PROVIDER_INVALID_L2_REASONS = frozenset(
    {
        "route_hash_contract_mismatch",
        "hopper_primitive_structure_mismatch",
        "hopper_primitive_contract_mismatch",
        "hopper_route_structure_mismatch",
        "hopper_route_start_mismatch",
        "hopper_route_connectivity_mismatch",
        "hopper_route_goal_mismatch",
        "hopper_route_cost_contract_mismatch",
        "hopper_route_probability_contract_mismatch",
        "hopper_launch_unknown",
        "hopper_launch_unsafe",
        "hopper_arc_boundary_violation",
        "hopper_arc_unknown",
        "hopper_arc_clearance_violation",
        "hopper_landing_probability_below_threshold",
        "hopper_landing_zone_unknown",
        "hopper_landing_zone_unsafe",
        "hopper_landing_slope_exceeded",
        "hopper_landing_height_unreachable",
        "hopper_landing_theta_unreachable",
        "hopper_stop_condition_failed",
        "hopper_route_state_budget_exceeded",
        "hopper_memory_budget_exceeded",
        "hopper_replay_work_budget_exceeded",
    }
)

_CAPABILITY_REASONS = frozenset(
    {
        "hopper_proxy_profile_incomplete",
        "hopper_parameter_set_unsupported",
        "hopper_stop_condition_unsupported",
        "hopper_energy_model_unsupported",
        "hopper_risk_objective_unsupported",
        "hopper_accelerator_required_unsupported",
        "hopper_hold_oracle_unavailable",
    }
)
_SEMANTIC_STAGES = {
    "hopper_launch_unknown": "launch_validation",
    "hopper_launch_unsafe": "launch_validation",
    "hopper_arc_boundary_violation": "arc_validation",
    "hopper_arc_unknown": "arc_validation",
    "hopper_arc_clearance_violation": "arc_validation",
    "hopper_landing_probability_below_threshold": "landing_probability",
    "hopper_landing_zone_unknown": "landing_validation",
    "hopper_landing_zone_unsafe": "landing_validation",
    "hopper_landing_slope_exceeded": "landing_validation",
    "hopper_landing_height_unreachable": "landing_validation",
    "hopper_landing_theta_unreachable": "landing_validation",
    "hopper_stop_condition_failed": "stop_validation",
}
_RESOURCE_POLICY = {
    "hopper_expansion_budget_exhausted": "search_expansion",
    "hopper_route_state_budget_exceeded": "route_state_admission",
    "hopper_memory_budget_exceeded": "search_memory",
    "hopper_replay_work_budget_exceeded": "replay_work",
}
_DEADLINE_STAGES = frozenset(
    {
        "capability_preflight",
        "search_setup",
        "search_expansion",
        "launch_validation",
        "arc_candidate_enumeration",
        "arc_validation",
        "landing_probability",
        "landing_validation",
        "stop_validation",
        "replay_work",
        "route_validation",
    }
)
_CONTRACT_REASONS = frozenset(
    {
        "planning_request_contract_mismatch",
        "planning_deadline_contract_mismatch",
        "hopper_authority_contract_mismatch",
        "hopper_profile_contract_mismatch",
        "terrain_snapshot_identity_mismatch",
        "terrain_snapshot_hash_mismatch",
        "terrain_query_contract_mismatch",
        "hopper_terrain_geometry_contract_mismatch",
        "hopper_numeric_contract_mismatch",
        "hopper_primitive_structure_mismatch",
        "hopper_primitive_contract_mismatch",
        "hopper_jump_oracle_contract_mismatch",
        "hopper_route_structure_mismatch",
        "hopper_route_start_mismatch",
        "hopper_route_connectivity_mismatch",
        "hopper_route_goal_mismatch",
        "hopper_route_cost_contract_mismatch",
        "hopper_route_probability_contract_mismatch",
        "hopper_route_oracle_contract_mismatch",
        "route_hash_contract_mismatch",
        "hopper_search_cost_contract_mismatch",
    }
)
_PROVIDER_STAGES = frozenset(
    {
        "provider_authority",
        "capability_preflight",
        "goal_preflight",
        "search_setup",
        "search_expansion",
        "route_state_admission",
        "search_memory",
        "launch_validation",
        "arc_candidate_enumeration",
        "arc_validation",
        "landing_probability",
        "landing_validation",
        "stop_validation",
        "replay_work",
        "hopper_search",
        "route_validation",
    }
)

_DETAIL_KEYS = {
    "capability": ("actual", "expected", "parameter_set_id", "profile_id"),
    "goal": ("actual", "expected"),
    "semantic": (
        "action_key",
        "actual",
        "candidate_id",
        "cell_x",
        "cell_y",
        "hop_index",
        "parameter_set_id",
        "phase",
        "reason_rank",
        "segment_index",
    ),
    "contract": ("actual", "expected", "phase"),
    "expansion": ("attempted_expanded_states", "max_expanded_states"),
    "route": (
        "attempted_route_states",
        "effective_max_route_states",
        "requested_max_route_states",
    ),
    "replay": ("attempted_work_units", "max_work_units", "phase"),
    "memory": (
        "accounting_id",
        "admitted_record_count",
        "attempted_accounted_bytes",
        "effective_max_memory_bytes",
        "max_memory_bytes",
        "persistent_accounted_bytes",
        "phase",
        "transient_reserved_bytes",
    ),
}


@dataclass(frozen=True, slots=True)
class _ApiSeal:
    provider: object
    plan_func: object
    request_token: bytes
    deadline_token: tuple[str, str, object]
    snapshot: object
    snapshot_digest: str
    geometry_token: bytes
    profile_token: bytes
    composite_token: tuple[object, ...]


def _exact_float_word(value: object, *, nonnegative: bool = False) -> str:
    if type(value) is not float or not isfinite(value):
        raise ValueError("exact finite float required")
    if nonnegative and value < 0.0:
        raise ValueError("nonnegative float required")
    if value == 0.0 and copysign(1.0, value) < 0.0:
        raise ValueError("canonical positive zero required")
    return value.hex()


def _profile_token(profile: HopperProfileV2) -> bytes:
    if type(profile) is not HopperProfileV2:
        raise ValueError("hopper_profile_contract_mismatch")
    return canonical_json_bytes(profile)


def _section2_token(authority: HopperProviderAuthorityV2) -> tuple[object, ...]:
    if type(authority) is not HopperProviderAuthorityV2:
        raise ValueError("hopper_authority_contract_mismatch")
    record = _lookup_hopper_parameter_set_v2(authority.parameter_set_id)
    if type(record) is HopperParameterSetRecordV2:
        resolved = _hopper_parameter_set_in_memory_token_v2(record)
    else:
        resolved = record
    registry = _authority_module.HOPPER_PARAMETER_SET_REGISTRY_V2
    if type(registry) is not tuple:
        raise ValueError("hopper_authority_contract_mismatch")
    registry_token = tuple(
        _hopper_parameter_set_in_memory_token_v2(item) for item in registry
    )
    return (
        _profile_token(authority.hopper_profile),
        resolved,
        authority.authority_schema_version,
        registry_token,
    )


def _request_token(request: PlanningRequestV2) -> bytes:
    if type(request) is not PlanningRequestV2:
        raise ValueError("planning_request_contract_mismatch")
    return canonical_json_bytes(
        (
            request.request_id,
            request.platform_profile_id,
            request.start_state,
            request.goal_state,
            request.objective_profile,
            request.resource_budget,
            request.timeout_s,
            request.accelerator_policy,
            request.determinism_seed,
        )
    )


def _capture_seal(
    provider: object,
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> _ApiSeal:
    if (
        type(profile) is not PlatformProfileV2
        or profile.platform_kind is not PlatformKindV2.HOPPER
        or type(anchor) is not FineSafetyAnchorV2
        or type(deadline) is not PlanningDeadlineV2
        or anchor.snapshot is not request.terrain_snapshot
    ):
        raise ValueError("planning_request_contract_mismatch")
    plan = provider.plan
    plan_self = getattr(plan, "__self__", None)
    plan_func = getattr(plan, "__func__", None)
    if plan_self is not provider or plan_func is None:
        raise ValueError("hopper_authority_contract_mismatch")

    # Attribute acquisition order is part of the Hopper API contract.
    authority = provider.hopper_authority
    resource_authority = provider.hopper_resource_authority
    if (
        type(authority) is not HopperProviderAuthorityV2
        or authority.hopper_profile.profile != profile
        or resource_authority is not HOPPER_RESOURCE_AUTHORITY_V2
    ):
        raise ValueError("hopper_authority_contract_mismatch")
    _require_canonical_resource_authority_v2(resource_authority)
    if (
        _provider_module.validate_hopper_jump_l2 is not _TRUSTED_JUMP_ORACLE
        or _route_validation_module.validate_hopper_route_l2 is not _TRUSTED_ROUTE_L2
    ):
        raise ValueError("hopper_authority_contract_mismatch")

    section2 = _section2_token(authority)
    resource = _hopper_resource_authority_in_memory_token_v2(resource_authority)
    composite = (
        _COMPOSITE_SCHEMA,
        section2,
        resource,
        (provider, plan_func),
        _TRUSTED_JUMP_ORACLE,
        _TRUSTED_ROUTE_L2,
    )
    geometry = anchor.snapshot.geometry
    return _ApiSeal(
        provider=provider,
        plan_func=plan_func,
        request_token=_request_token(request),
        deadline_token=(
            _exact_float_word(deadline.started_monotonic_s),
            _exact_float_word(deadline.deadline_monotonic_s),
            deadline._monotonic_clock,
        ),
        snapshot=anchor.snapshot,
        snapshot_digest=snapshot_hash(anchor.snapshot),
        geometry_token=canonical_json_bytes(geometry),
        profile_token=canonical_json_bytes(profile),
        composite_token=composite,
    )


def _drift_reason(
    initial: _ApiSeal,
    provider: object,
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> str | None:
    try:
        current = _capture_seal(provider, request, profile, anchor, deadline)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return "hopper_authority_contract_mismatch"
    if current.deadline_token != initial.deadline_token:
        return "planning_deadline_contract_mismatch"
    if current.request_token != initial.request_token:
        return "planning_request_contract_mismatch"
    if current.snapshot is not initial.snapshot:
        return "terrain_snapshot_identity_mismatch"
    if current.snapshot_digest != initial.snapshot_digest:
        return "terrain_snapshot_hash_mismatch"
    if current.geometry_token != initial.geometry_token:
        return "hopper_terrain_geometry_contract_mismatch"
    if current.profile_token != initial.profile_token:
        return "hopper_profile_contract_mismatch"
    if (
        current.provider is not initial.provider
        or current.plan_func is not initial.plan_func
        or current.composite_token != initial.composite_token
    ):
        return "hopper_authority_contract_mismatch"
    return None


def _zero_telemetry(
    deadline: PlanningDeadlineV2,
    reason: str,
    *,
    timed_out: bool = False,
) -> SearchTelemetryV2:
    elapsed = deadline.elapsed_s
    return SearchTelemetryV2(
        0,
        0,
        0,
        0,
        0,
        0.0 if elapsed == 0.0 else elapsed,
        timed_out,
        False,
        False,
        reason,
    )


def _api_failure(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    reason: str,
    *,
    category: FailureCategoryV2 = FailureCategoryV2.INTERNAL_ERROR,
    stage: str = "provider_postcondition",
    actual: str | None = None,
    expected: str | None = None,
    phase: str | None = None,
    details: tuple[tuple[str, str | int | float | bool | None], ...] | None = None,
    timed_out: bool = False,
) -> PlanningFailureV2:
    if details is None:
        details = () if actual is None else (
            ("actual", actual),
            ("expected", expected),
            ("phase", stage if phase is None else phase),
        )
    return PlanningFailureV2(
        request.request_id,
        PlatformKindV2.HOPPER,
        category,
        reason,
        FailureEvidenceV2(stage, (reason,), details),
        _zero_telemetry(deadline, reason, timed_out=timed_out),
    )


def _authority_failure(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    reason: str,
    stage: str,
) -> PlanningFailureV2:
    return _api_failure(
        request,
        deadline,
        reason,
        stage=stage,
        actual=reason,
        expected="stable_hopper_authority",
        phase=stage,
    )


def _timeout(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    stage: str,
) -> PlanningFailureV2:
    return _api_failure(
        request,
        deadline,
        "planning_deadline_expired",
        category=FailureCategoryV2.TIMEOUT,
        stage=stage,
        details=(),
        timed_out=True,
    )


def _outcome_token(outcome: object) -> tuple[type[object], bytes]:
    if type(outcome) not in (PlanningSuccessV2, PlanningFailureV2):
        raise ValueError("hopper_provider_outcome_contract_mismatch")
    return type(outcome), canonical_json_bytes(outcome)


def _detail_shape(reason: str) -> tuple[str, ...] | None:
    if reason in _CAPABILITY_REASONS:
        return _DETAIL_KEYS["capability"]
    if reason == "hopper_goal_heading_unreachable":
        return _DETAIL_KEYS["goal"]
    if reason in _SEMANTIC_STAGES or reason == "hopper_no_complete_route":
        return _DETAIL_KEYS["semantic"]
    if reason in _CONTRACT_REASONS:
        return _DETAIL_KEYS["contract"]
    if reason == "hopper_expansion_budget_exhausted":
        return _DETAIL_KEYS["expansion"]
    if reason == "hopper_route_state_budget_exceeded":
        return _DETAIL_KEYS["route"]
    if reason == "hopper_replay_work_budget_exceeded":
        return _DETAIL_KEYS["replay"]
    if reason == "hopper_memory_budget_exceeded":
        return _DETAIL_KEYS["memory"]
    if reason == "planning_deadline_expired":
        return ()
    return None


def _failure_policy_matches(failure: PlanningFailureV2) -> bool:
    reason = failure.reason_code
    category = failure.category
    stage = failure.evidence.stage
    if stage not in _PROVIDER_STAGES:
        return False
    if reason in _CAPABILITY_REASONS:
        policy_ok = (
            category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
            and stage == "capability_preflight"
        )
    elif reason == "hopper_goal_heading_unreachable":
        policy_ok = (
            category is FailureCategoryV2.GOAL_POSE_UNREACHABLE
            and stage == "goal_preflight"
        )
    elif reason in _SEMANTIC_STAGES:
        policy_ok = (
            category is FailureCategoryV2.VALIDATION_FAILED
            and stage == _SEMANTIC_STAGES[reason]
        )
    elif reason in _RESOURCE_POLICY:
        policy_ok = (
            category is FailureCategoryV2.RESOURCE_LIMIT
            and stage == _RESOURCE_POLICY[reason]
        )
    elif reason == "hopper_no_complete_route":
        policy_ok = (
            category is FailureCategoryV2.NO_COMPLETE_ROUTE
            and stage == "hopper_search"
        )
    elif reason == "planning_deadline_expired":
        policy_ok = (
            category is FailureCategoryV2.TIMEOUT and stage in _DEADLINE_STAGES
        )
    elif reason in _CONTRACT_REASONS:
        policy_ok = category is FailureCategoryV2.INTERNAL_ERROR
    else:
        return False
    if not policy_ok:
        return False
    expected_keys = _detail_shape(reason)
    details = failure.evidence.details
    if expected_keys is None or tuple(key for key, _value in details) != expected_keys:
        return False
    if failure.evidence.checks != (reason,):
        return False
    return True


def _telemetry_matches(failure: PlanningFailureV2) -> bool:
    telemetry = failure.search_telemetry
    if type(telemetry) is not SearchTelemetryV2:
        return False
    counts = (
        telemetry.expanded_states,
        telemetry.generated_primitives,
        telemetry.rejected_l0,
        telemetry.rejected_l1,
        telemetry.rejected_l2,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        return False
    expanded, generated, rejected_l0, rejected_l1, rejected_l2 = counts
    lower = max(0, 192 * (expanded - 1))
    if (
        rejected_l0 != 0
        or rejected_l1 != 0
        or rejected_l2 > generated
        or not lower <= generated <= 192 * expanded
        or type(telemetry.elapsed_s) is not float
        or not isfinite(telemetry.elapsed_s)
        or telemetry.elapsed_s < 0.0
        or (telemetry.elapsed_s == 0.0 and copysign(1.0, telemetry.elapsed_s) < 0.0)
        or type(telemetry.timed_out) is not bool
        or telemetry.timed_out is not (failure.reason_code == "planning_deadline_expired")
        or type(telemetry.accelerator_used) is not bool
        or telemetry.accelerator_used
        or type(telemetry.ackermann_feasible_claimed) is not bool
        or telemetry.ackermann_feasible_claimed
        or type(telemetry.termination_reason) is not str
        or telemetry.termination_reason != failure.reason_code
    ):
        return False
    if failure.reason_code == "hopper_no_complete_route":
        return expanded >= 1 and generated == 192 * expanded
    return True


def _provider_failure_is_valid(
    outcome: object,
    request: PlanningRequestV2,
) -> bool:
    return (
        type(outcome) is PlanningFailureV2
        and outcome.request_id == request.request_id
        and outcome.platform_kind is PlatformKindV2.HOPPER
        and type(outcome.evidence) is FailureEvidenceV2
        and _failure_policy_matches(outcome)
        and _telemetry_matches(outcome)
    )


def _success_envelope_is_valid(
    outcome: object,
    request: PlanningRequestV2,
) -> bool:
    if (
        type(outcome) is not PlanningSuccessV2
        or outcome.request_id != request.request_id
        or outcome.platform_kind is not PlatformKindV2.HOPPER
        or type(outcome.route) is not TypedRouteV2
        or outcome.route.platform_kind is not PlatformKindV2.HOPPER
        or outcome.route.is_complete is not True
        or not outcome.route.primitives
        or type(outcome.observation_projection) is not ObservationProjectionV2
        or type(outcome.cache_evidence) is not CacheEvidenceV2
        or type(outcome.search_telemetry) is not SearchTelemetryV2
    ):
        return False
    observation = outcome.observation_projection
    cache = outcome.cache_evidence
    telemetry = outcome.search_telemetry
    return (
        observation.source
        == "path-planner-v2-hopper-inflight-observation-disabled/v1"
        and observation.expected_new_observed_cells == 0.0
        and observation.expected_information_gain == 0.0
        and type(observation.sample_states) is tuple
        and len(observation.sample_states) == len(outcome.route.primitives) + 1
        and cache.cache_namespace == "path-planner-v2-hopper-cache-disabled/v1"
        and cache.cache_key == "hopper-cache-disabled/v1"
        and cache.hit is False
        and telemetry.expanded_states >= 1
        and telemetry.generated_primitives == 192 * telemetry.expanded_states
        and telemetry.rejected_l0 == 0
        and telemetry.rejected_l1 == 0
        and telemetry.rejected_l2 <= telemetry.generated_primitives
        and telemetry.timed_out is False
        and telemetry.accelerator_used is False
        and telemetry.ackermann_feasible_claimed is False
        and telemetry.termination_reason == "hopper_route_l2_valid"
    )


def _l2_remap(
    result: object,
    outcome: PlanningSuccessV2,
    deadline: PlanningDeadlineV2,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(reason, actual, expected)``; a None reason means pass."""
    if type(result) is not HopperRouteValidationResultV2:
        return (
            "hopper_route_oracle_contract_mismatch",
            "malformed_or_exception",
            "hopper-route-validation-result/v1",
        )
    reason = result.reason_code
    if result.passed is True:
        expected_digest = sha256(canonical_json_bytes(outcome.route)).hexdigest()
        if (
            reason == "hopper_route_l2_valid"
            and result.schema_version == "hopper-route-validation-result/v1"
            and result.route_digest == expected_digest
            and result.cost_breakdown == outcome.cost_breakdown
            and result.evidence == outcome.validation_evidence
        ):
            return None, None, None
        return (
            "hopper_route_oracle_contract_mismatch",
            reason,
            "hopper-route-validation-result/v1",
        )
    if reason in _AUTHORITY_L2_REASONS:
        return reason, reason, "stable_hopper_authority"
    if reason in _VALIDATOR_L2_REASONS:
        return "hopper_route_oracle_contract_mismatch", reason, "hopper_route_l2_valid"
    if reason in _PROVIDER_INVALID_L2_REASONS:
        return "hopper_provider_outcome_contract_mismatch", reason, "hopper_route_l2_valid"
    if reason == "planning_deadline_expired":
        if deadline.expired:
            return "planning_deadline_expired", reason, "deadline_checkpoint_consistent"
        return (
            "hopper_route_oracle_contract_mismatch",
            reason,
            "deadline_checkpoint_consistent",
        )
    return (
        "hopper_route_oracle_contract_mismatch",
        reason if type(reason) is str else "malformed_or_exception",
        "hopper-route-validation-result/v1",
    )


def dispatch_hopper_provider_v2(
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
    provider: object,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> PlanningOutcomeV2:
    """Run the Hopper-only provider/API seal and postcondition sequence."""
    try:
        initial = _capture_seal(provider, request, profile, anchor, deadline)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:
        return _authority_failure(
            request,
            deadline,
            "hopper_authority_contract_mismatch",
            "provider_authority",
        )
    if deadline.expired:
        return _timeout(request, deadline, "provider_dispatch")

    try:
        outcome = provider.plan(request, anchor, deadline)
    except (KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception as exc:
        drift = _drift_reason(initial, provider, request, profile, anchor, deadline)
        if drift is not None:
            return _authority_failure(request, deadline, drift, "provider_completion")
        if deadline.expired:
            return _timeout(request, deadline, "provider_completion")
        return _api_failure(
            request,
            deadline,
            "primitive_provider_exception",
            stage="provider_execution",
            details=(("exception_type", type(exc).__name__),),
        )

    drift = _drift_reason(initial, provider, request, profile, anchor, deadline)
    if drift is not None:
        return _authority_failure(request, deadline, drift, "provider_completion")
    try:
        outcome_token = _outcome_token(outcome)
    except Exception:
        outcome_token = None
    if deadline.expired:
        return _timeout(request, deadline, "provider_completion")

    l2_result: object | None = None
    l2_exception = False
    if type(outcome) is PlanningSuccessV2 and _success_envelope_is_valid(outcome, request):
        try:
            l2_result = _TRUSTED_ROUTE_L2(
                outcome.route,
                request,
                anchor,
                provider.hopper_authority,
                deadline,
            )
        except (KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:
            l2_exception = True

    # This checkpoint is mandatory even when L2 raised an ordinary exception.
    drift = _drift_reason(initial, provider, request, profile, anchor, deadline)
    if drift is not None:
        return _authority_failure(request, deadline, drift, "provider_postcondition")
    try:
        outcome_stable = outcome_token is not None and _outcome_token(outcome) == outcome_token
    except Exception:
        outcome_stable = False
    if deadline.expired:
        return _timeout(request, deadline, "provider_postcondition")

    drift = _drift_reason(initial, provider, request, profile, anchor, deadline)
    if drift is not None:
        return _authority_failure(request, deadline, drift, "provider_postcondition")
    try:
        outcome_stable = (
            outcome_stable
            and outcome_token is not None
            and _outcome_token(outcome) == outcome_token
        )
    except Exception:
        outcome_stable = False
    if not outcome_stable:
        return _api_failure(
            request,
            deadline,
            "hopper_provider_outcome_contract_mismatch",
            actual="malformed_or_mutated",
            expected="stable_hopper_provider_outcome",
        )

    if type(outcome) is PlanningFailureV2:
        if _provider_failure_is_valid(outcome, request):
            return outcome
        return _api_failure(
            request,
            deadline,
            "hopper_provider_outcome_contract_mismatch",
            actual=outcome.reason_code,
            expected="valid_hopper_provider_failure",
        )
    if type(outcome) is not PlanningSuccessV2 or not _success_envelope_is_valid(
        outcome, request
    ):
        return _api_failure(
            request,
            deadline,
            "hopper_provider_outcome_contract_mismatch",
            actual=type(outcome).__name__,
            expected="valid_hopper_provider_outcome",
        )

    if l2_exception:
        remap = (
            "hopper_route_oracle_contract_mismatch",
            "malformed_or_exception",
            "hopper-route-validation-result/v1",
        )
    else:
        remap = _l2_remap(l2_result, outcome, deadline)
    reason, actual, expected = remap
    if reason is None:
        return outcome
    if reason == "planning_deadline_expired":
        return _timeout(request, deadline, "provider_postcondition")
    stage = "provider_postcondition"
    return _api_failure(
        request,
        deadline,
        reason,
        stage=stage,
        actual=actual,
        expected=expected,
        phase=stage,
    )
