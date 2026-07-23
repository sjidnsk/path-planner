"""Trusted opt-in dispatch for the Scout Mini wheel corridor SQP capability."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, remainder, tau
from types import MethodType

import path_planner.v2.providers.wheel_sqp as _provider_module
import path_planner.v2.wheel_sqp_serialization as _serialization_module
import path_planner.v2.wheel_sqp_validation as _validation_module
from path_planner.v2.contracts import (
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
    ValidationLevelV2,
)
from path_planner.v2.profiles import (
    WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2,
    PlatformProfileV2,
    WheelKinematicSQPProfileV2,
)
from path_planner.v2.providers.wheel_sqp import (
    WHEEL_SQP_CACHE_KEY_DISABLED_V2,
    WHEEL_SQP_CACHE_NAMESPACE_DISABLED_V2,
    WheelKinematicSQPProviderV2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineSafetyAnchorV2, snapshot_hash
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_L2_VALIDATOR_V2,
    WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
    WHEEL_KINEMATIC_SOLVER_CONTRACT_V2,
    WheelKinematicRouteV2,
    WheelKinematicSegmentV2,
    WheelSQPSearchTelemetryV2,
    WheelSQPValidationEvidenceV2,
)
from path_planner.v2.wheel_sqp_serialization import (
    decode_wheel_route_v2,
    encode_wheel_route_v2,
    project_wheel_route_to_candidate_v1,
    wheel_candidate_hash_v2,
    wheel_route_hash_v2,
    wheel_segment_hash_v2,
    wheel_sqp_profile_hash_v2,
    wheel_sqp_request_hash_v2,
)


__all__ = ()

_TRUSTED_PROVIDER_TYPE = WheelKinematicSQPProviderV2
_TRUSTED_PROVIDER_PLAN = WheelKinematicSQPProviderV2.plan
_TRUSTED_PROVIDER_ENGINE = _provider_module._plan_wheel_sqp_v2
_TRUSTED_PROVIDER_DECISION_HASH = _provider_module._decision_hash
_TRUSTED_PROVIDER_TELEMETRY = _provider_module._telemetry
_TRUSTED_DECISION_AUDIT_SINK = _provider_module._DECISION_AUDIT_SINK_V1
_TRUSTED_L2_VALIDATOR = _provider_module.validate_wheel_sqp_candidate_l2
_TRUSTED_ROUTE_ENCODER = encode_wheel_route_v2
_TRUSTED_ROUTE_DECODER = decode_wheel_route_v2
_TRUSTED_ROUTE_HASH = wheel_route_hash_v2
_TRUSTED_ROUTE_TO_CANDIDATE = project_wheel_route_to_candidate_v1
_TRUSTED_CANDIDATE_HASH = wheel_candidate_hash_v2
_TRUSTED_SEGMENT_HASH = wheel_segment_hash_v2

_FAILURE_CATEGORIES = {
    "wheel_sqp_profile_unsupported": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_backend_unavailable": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_objective_unsupported": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_accelerator_required_unsupported": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_start_invalid": FailureCategoryV2.UNSAFE_START,
    "wheel_sqp_goal_invalid": FailureCategoryV2.UNSAFE_GOAL,
    "wheel_sqp_no_2d_corridor": FailureCategoryV2.GOAL_POSE_UNREACHABLE,
    "wheel_sqp_initialization_failed": FailureCategoryV2.NO_COMPLETE_ROUTE,
    "wheel_sqp_infeasible": FailureCategoryV2.NO_COMPLETE_ROUTE,
    "wheel_sqp_candidate_l2_rejected": FailureCategoryV2.VALIDATION_FAILED,
    "wheel_sqp_repair_l2_rejected": FailureCategoryV2.VALIDATION_FAILED,
    "wheel_sqp_goal_tolerance_exceeded": FailureCategoryV2.GOAL_POSE_UNREACHABLE,
    "wheel_sqp_corridor_budget_exceeded": FailureCategoryV2.RESOURCE_LIMIT,
    "wheel_sqp_resource_budget_exceeded": FailureCategoryV2.RESOURCE_LIMIT,
    "planning_deadline_expired": FailureCategoryV2.TIMEOUT,
    "wheel_sqp_numeric_contract_failed": FailureCategoryV2.INTERNAL_ERROR,
    "wheel_sqp_identity_mismatch": FailureCategoryV2.INTERNAL_ERROR,
    "wheel_sqp_internal_error": FailureCategoryV2.INTERNAL_ERROR,
}


@dataclass(frozen=True, slots=True)
class _DispatchSealV1:
    request: PlanningRequestV2
    request_id: str
    request_hash: str
    profile: PlatformProfileV2
    profile_bytes: bytes
    wheel_profile: WheelKinematicSQPProfileV2
    wheel_profile_bytes: bytes
    wheel_profile_hash: str
    snapshot: object
    snapshot_digest: str
    anchor: FineSafetyAnchorV2
    anchor_digest: str
    deadline: PlanningDeadlineV2
    started_monotonic_s: float
    deadline_monotonic_s: float
    clock: object
    provider: WheelKinematicSQPProviderV2


def _trusted_callables_current() -> bool:
    return (
        _provider_module.WheelKinematicSQPProviderV2 is _TRUSTED_PROVIDER_TYPE
        and _provider_module.WheelKinematicSQPProviderV2.plan is _TRUSTED_PROVIDER_PLAN
        and _provider_module._plan_wheel_sqp_v2 is _TRUSTED_PROVIDER_ENGINE
        and _provider_module._decision_hash is _TRUSTED_PROVIDER_DECISION_HASH
        and _provider_module._telemetry is _TRUSTED_PROVIDER_TELEMETRY
        and _provider_module._DECISION_AUDIT_SINK_V1
        is _TRUSTED_DECISION_AUDIT_SINK
        and _provider_module.validate_wheel_sqp_candidate_l2 is _TRUSTED_L2_VALIDATOR
        and _validation_module.validate_wheel_sqp_candidate_l2 is _TRUSTED_L2_VALIDATOR
        and _serialization_module.encode_wheel_route_v2 is _TRUSTED_ROUTE_ENCODER
        and _serialization_module.decode_wheel_route_v2 is _TRUSTED_ROUTE_DECODER
        and _serialization_module.wheel_route_hash_v2 is _TRUSTED_ROUTE_HASH
        and _serialization_module.project_wheel_route_to_candidate_v1
        is _TRUSTED_ROUTE_TO_CANDIDATE
        and _serialization_module.wheel_candidate_hash_v2 is _TRUSTED_CANDIDATE_HASH
        and _serialization_module.wheel_segment_hash_v2 is _TRUSTED_SEGMENT_HASH
    )


def _capture_seal(
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
    provider: object,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> _DispatchSealV1:
    if type(request) is not PlanningRequestV2:
        raise TypeError("request must be exact PlanningRequestV2")
    if type(profile) is not PlatformProfileV2:
        raise TypeError("profile must be exact PlatformProfileV2")
    if type(provider) is not _TRUSTED_PROVIDER_TYPE:
        raise TypeError("provider must be exact WheelKinematicSQPProviderV2")
    if type(anchor) is not FineSafetyAnchorV2:
        raise TypeError("anchor must be exact FineSafetyAnchorV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if not _trusted_callables_current():
        raise ValueError("trusted wheel SQP callable drift")
    bound_plan = provider.plan
    if (
        type(bound_plan) is not MethodType
        or bound_plan.__self__ is not provider
        or bound_plan.__func__ is not _TRUSTED_PROVIDER_PLAN
    ):
        raise ValueError("provider plan binding drift")
    wheel_profile = provider.wheel_sqp_profile
    if type(wheel_profile) is not WheelKinematicSQPProfileV2:
        raise TypeError("provider wheel profile must be exact")
    if provider.profile is not profile or wheel_profile.profile is not profile:
        raise ValueError("provider and registry profile identity mismatch")
    if wheel_profile.profile != profile:
        raise ValueError("provider and registry profile value mismatch")
    if (
        profile.platform_kind is not PlatformKindV2.WHEEL
        or profile.capability_revision
        != WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2
        or request.platform_profile_id != profile.profile_id
    ):
        raise ValueError("wheel SQP capability identity mismatch")
    if anchor.snapshot is not request.terrain_snapshot:
        raise ValueError("anchor snapshot identity mismatch")
    digest = snapshot_hash(request.terrain_snapshot)
    if anchor._snapshot_hash != digest:
        raise ValueError("anchor snapshot digest mismatch")
    return _DispatchSealV1(
        request=request,
        request_id=request.request_id,
        request_hash=wheel_sqp_request_hash_v2(request, digest),
        profile=profile,
        profile_bytes=canonical_json_bytes(profile),
        wheel_profile=wheel_profile,
        wheel_profile_bytes=canonical_json_bytes(wheel_profile),
        wheel_profile_hash=wheel_sqp_profile_hash_v2(wheel_profile),
        snapshot=request.terrain_snapshot,
        snapshot_digest=digest,
        anchor=anchor,
        anchor_digest=anchor._snapshot_hash,
        deadline=deadline,
        started_monotonic_s=deadline.started_monotonic_s,
        deadline_monotonic_s=deadline.deadline_monotonic_s,
        clock=deadline._monotonic_clock,
        provider=provider,
    )


def _seal_is_current(seal: _DispatchSealV1) -> bool:
    try:
        provider = seal.provider
        bound_plan = provider.plan
        current_digest = snapshot_hash(seal.request.terrain_snapshot)
        return (
            _trusted_callables_current()
            and type(seal.request) is PlanningRequestV2
            and seal.request.request_id == seal.request_id
            and seal.request.terrain_snapshot is seal.snapshot
            and current_digest == seal.snapshot_digest
            and wheel_sqp_request_hash_v2(seal.request, current_digest)
            == seal.request_hash
            and type(seal.profile) is PlatformProfileV2
            and canonical_json_bytes(seal.profile) == seal.profile_bytes
            and type(provider) is _TRUSTED_PROVIDER_TYPE
            and provider.wheel_sqp_profile is seal.wheel_profile
            and provider.profile is seal.profile
            and seal.wheel_profile.profile is seal.profile
            and canonical_json_bytes(seal.wheel_profile)
            == seal.wheel_profile_bytes
            and wheel_sqp_profile_hash_v2(seal.wheel_profile)
            == seal.wheel_profile_hash
            and seal.anchor.snapshot is seal.snapshot
            and seal.anchor._snapshot_hash == seal.anchor_digest
            and seal.deadline.started_monotonic_s == seal.started_monotonic_s
            and seal.deadline.deadline_monotonic_s == seal.deadline_monotonic_s
            and seal.deadline._monotonic_clock is seal.clock
            and type(bound_plan) is MethodType
            and bound_plan.__self__ is provider
            and bound_plan.__func__ is _TRUSTED_PROVIDER_PLAN
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return False


def _clock_state(seal: _DispatchSealV1) -> tuple[float, bool]:
    try:
        now = seal.clock()
        if type(now) not in (int, float):
            return 0.0, False
        value = float(now) - seal.started_monotonic_s
        elapsed = value if isfinite(value) and value >= 0.0 else 0.0
        expired = isfinite(float(now)) and float(now) >= seal.deadline_monotonic_s
        return elapsed, expired
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return 0.0, False


def _expired(seal: _DispatchSealV1) -> bool:
    try:
        now = seal.clock()
        return type(now) in (int, float) and isfinite(float(now)) and (
            float(now) >= seal.deadline_monotonic_s
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return False


def _api_failure(
    seal: _DispatchSealV1,
    reason_code: str,
    phase: str,
    *,
    exception_type: str | None = None,
) -> PlanningFailureV2:
    elapsed_s, physically_expired = _clock_state(seal)
    if physically_expired:
        reason_code = "planning_deadline_expired"
        exception_type = None
    timed_out = reason_code == "planning_deadline_expired"
    details = () if exception_type is None else (("exception_type", exception_type),)
    return PlanningFailureV2(
        request_id=seal.request_id,
        platform_kind=PlatformKindV2.WHEEL,
        category=_FAILURE_CATEGORIES[reason_code],
        reason_code=reason_code,
        evidence=FailureEvidenceV2(
            stage="wheel_sqp_api",
            checks=(reason_code,),
            details=(*details, ("phase", phase)),
        ),
        search_telemetry=WheelSQPSearchTelemetryV2(
            expanded_states=0,
            generated_primitives=0,
            rejected_l0=0,
            rejected_l1=0,
            rejected_l2=0,
            elapsed_s=elapsed_s,
            timed_out=timed_out,
            accelerator_used=False,
            ackermann_feasible_claimed=False,
            termination_reason=reason_code,
            corridor_count=0,
            selected_corridor_index=None,
            corridor_expansions=0,
            sqp_iterations=0,
            sqp_function_evaluations=0,
            l2_interval_count=0,
            l2_cell_count=0,
            repair_attempted=False,
            decision_hash=None,
        ),
    )


def _timeout(seal: _DispatchSealV1, phase: str) -> PlanningFailureV2:
    return _api_failure(seal, "planning_deadline_expired", phase)


def _identity_failure(seal: _DispatchSealV1, phase: str) -> PlanningFailureV2:
    return _api_failure(seal, "wheel_sqp_identity_mismatch", phase)


def _joined_samples(candidate: object) -> tuple[object, ...]:
    samples: list[object] = []
    for segment in candidate.segments:
        samples.extend(segment.samples if not samples else segment.samples[1:])
    return tuple(samples)


def _success_is_valid(
    success: object,
    seal: _DispatchSealV1,
    captured_decisions: tuple[tuple[str, str | None], ...],
) -> bool:
    try:
        if (
            type(success) is not PlanningSuccessV2
            or success.request_id != seal.request_id
            or success.platform_kind is not PlatformKindV2.WHEEL
            or type(success.route) is not WheelKinematicRouteV2
            or type(success.cost_breakdown) is not CostBreakdownV2
            or type(success.observation_projection) is not ObservationProjectionV2
            or type(success.validation_evidence) is not WheelSQPValidationEvidenceV2
            or type(success.search_telemetry) is not WheelSQPSearchTelemetryV2
            or type(success.cache_evidence) is not CacheEvidenceV2
        ):
            return False
        route = success.route
        if (
            route.is_complete is not True
            or type(route.primitives) is not tuple
            or not route.primitives
            or any(type(item) is not WheelKinematicSegmentV2 for item in route.primitives)
        ):
            return False
        encoded = _TRUSTED_ROUTE_ENCODER(route)
        decoded = _TRUSTED_ROUTE_DECODER(encoded)
        if (
            type(decoded) is not WheelKinematicRouteV2
            or _TRUSTED_ROUTE_ENCODER(decoded) != encoded
            or decoded != route
            or decoded.route_hash != _TRUSTED_ROUTE_HASH(decoded)
        ):
            return False
        if any(
            segment.segment_hash != _TRUSTED_SEGMENT_HASH(segment)
            for segment in decoded.primitives
        ):
            return False
        projected = _TRUSTED_ROUTE_TO_CANDIDATE(
            decoded,
            seal.request,
            seal.wheel_profile,
            seal.snapshot_digest,
        )
        projected_hash = _TRUSTED_CANDIDATE_HASH(projected)
        evidence = success.validation_evidence
        if not (
            projected_hash == projected.candidate_hash
            == decoded.source_candidate_hash
            == evidence.candidate_hash
            and evidence.route_hash == decoded.route_hash
            and evidence.request_hash == decoded.request_hash == seal.request_hash
            and evidence.profile_hash == decoded.profile_hash
            == seal.wheel_profile_hash
            and evidence.terrain_snapshot_hash == decoded.terrain_snapshot_hash
            == seal.snapshot_digest
            and evidence.validator_id == decoded.validator_contract_id
            == WHEEL_KINEMATIC_L2_VALIDATOR_V2
            and evidence.solver_contract_id == decoded.solver_contract_id
            == WHEEL_KINEMATIC_SOLVER_CONTRACT_V2
            and evidence.level is ValidationLevelV2.L2
            and evidence.passed is True
            and evidence.checks
            == (
                "strict_candidate_replay",
                "continuous_rectangle_sweep",
                "goal_tolerance",
                "strict_route_reseal",
            )
        ):
            return False
        objective = seal.request.objective_profile
        expected_cost = CostBreakdownV2(
            distance_cost=objective.distance_weight * projected.total_distance_m,
            risk_cost=0.0,
            energy_cost=objective.energy_weight
            * (projected.total_relative_energy / seal.wheel_profile.energy_normalization),
            time_cost=objective.time_weight
            * (projected.total_duration_s / seal.wheel_profile.time_normalization_s),
            total_cost=projected.total_cost,
        )
        if success.cost_breakdown != expected_cost:
            return False
        observation = success.observation_projection
        if not (
            observation.source == WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2
            and type(observation.sample_states) is tuple
            and observation.sample_states == _joined_samples(projected)
            and observation.expected_new_observed_cells == 0.0
            and observation.expected_information_gain == 0.0
        ):
            return False
        cache = success.cache_evidence
        if not (
            cache.cache_namespace == WHEEL_SQP_CACHE_NAMESPACE_DISABLED_V2
            and cache.cache_key == WHEEL_SQP_CACHE_KEY_DISABLED_V2
            and cache.hit is False
        ):
            return False
        telemetry = success.search_telemetry
        if not (
            telemetry.timed_out is False
            and telemetry.accelerator_used is False
            and telemetry.ackermann_feasible_claimed is False
            and telemetry.termination_reason == "wheel_sqp_route_selected"
            and type(telemetry.decision_hash) is str
            and len(telemetry.decision_hash) == 64
            and all(ch in "0123456789abcdef" for ch in telemetry.decision_hash)
            and captured_decisions
            == ((telemetry.termination_reason, telemetry.decision_hash),)
        ):
            return False
        endpoint = projected.actual_endpoint
        goal = seal.request.goal_state
        position_error = hypot(endpoint.x_m - goal.x_m, endpoint.y_m - goal.y_m)
        heading_error = abs(remainder(endpoint.heading_rad - goal.heading_rad, tau))
        return (
            projected.start_state == seal.request.start_state
            and position_error <= seal.profile.goal_position_tolerance_m
            and heading_error <= seal.profile.goal_heading_tolerance_rad
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return False


def _failure_is_valid(
    failure: object,
    seal: _DispatchSealV1,
    captured_decisions: tuple[tuple[str, str | None], ...],
) -> bool:
    try:
        if (
            type(failure) is not PlanningFailureV2
            or failure.request_id != seal.request_id
            or failure.platform_kind is not PlatformKindV2.WHEEL
            or failure.reason_code not in _FAILURE_CATEGORIES
            or failure.category is not _FAILURE_CATEGORIES[failure.reason_code]
            or type(failure.evidence) is not FailureEvidenceV2
            or failure.evidence.checks != (failure.reason_code,)
            or type(failure.search_telemetry) is not WheelSQPSearchTelemetryV2
        ):
            return False
        telemetry = failure.search_telemetry
        decision_hash_binding = captured_decisions == (
            (telemetry.termination_reason, telemetry.decision_hash),
        )
        return (
            telemetry.termination_reason == failure.reason_code
            and telemetry.timed_out
            is (failure.reason_code == "planning_deadline_expired")
            and telemetry.accelerator_used is False
            and telemetry.ackermann_feasible_claimed is False
            and decision_hash_binding
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return False


def dispatch_wheel_sqp_provider_v2(
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
    provider: object,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> PlanningOutcomeV2:
    try:
        seal = _capture_seal(request, profile, provider, anchor, deadline)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        # The generic API has already proved the stable request/profile surface.
        # A local minimal seal is safe because no untrusted provider code has run.
        minimal_seal = _DispatchSealV1(
            request=request,
            request_id=request.request_id,
            request_hash="0" * 64,
            profile=profile,
            profile_bytes=b"",
            wheel_profile=object(),
            wheel_profile_bytes=b"",
            wheel_profile_hash="0" * 64,
            snapshot=request.terrain_snapshot,
            snapshot_digest="0" * 64,
            anchor=anchor,
            anchor_digest=getattr(anchor, "_snapshot_hash", "0" * 64),
            deadline=deadline,
            started_monotonic_s=deadline.started_monotonic_s,
            deadline_monotonic_s=deadline.deadline_monotonic_s,
            clock=deadline._monotonic_clock,
            provider=provider,
        )
        if _expired(minimal_seal):
            return _timeout(minimal_seal, "preflight")
        return _identity_failure(minimal_seal, "preflight")
    if _expired(seal):
        return _timeout(seal, "provider_dispatch")
    captured_decisions: list[tuple[str, str | None]] = []
    audit_token = _TRUSTED_DECISION_AUDIT_SINK.set(
        captured_decisions.append
    )
    try:
        outcome = _TRUSTED_PROVIDER_PLAN(provider, request, anchor, deadline)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        if _expired(seal):
            return _timeout(seal, "provider_completion")
        completion_seal_current = _seal_is_current(seal)
        if not completion_seal_current:
            if _expired(seal):
                return _timeout(seal, "provider_completion")
            return _identity_failure(seal, "provider_completion")
        if _expired(seal):
            return _timeout(seal, "provider_completion")
        return _api_failure(
            seal,
            "wheel_sqp_internal_error",
            "provider_execution",
            exception_type=type(exc).__name__,
        )
    finally:
        _TRUSTED_DECISION_AUDIT_SINK.reset(audit_token)
    if _expired(seal):
        return _timeout(seal, "provider_completion")
    completion_seal_current = _seal_is_current(seal)
    if not completion_seal_current:
        if _expired(seal):
            return _timeout(seal, "provider_completion")
        return _identity_failure(seal, "provider_completion")
    valid = (
        _success_is_valid(outcome, seal, tuple(captured_decisions))
        if type(outcome) is PlanningSuccessV2
        else _failure_is_valid(outcome, seal, tuple(captured_decisions))
        if type(outcome) is PlanningFailureV2
        else False
    )
    if _expired(seal):
        return _timeout(seal, "codec_reseal")
    postcondition_seal_current = _seal_is_current(seal)
    if not postcondition_seal_current:
        if _expired(seal):
            return _timeout(seal, "codec_reseal")
        return _identity_failure(seal, "codec_reseal")
    if _expired(seal):
        return _timeout(seal, "codec_reseal")
    if not valid:
        return _identity_failure(seal, "provider_postcondition")
    return outcome
