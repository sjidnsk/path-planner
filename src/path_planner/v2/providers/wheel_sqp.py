from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite

from path_planner.core import WorldPoint
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
)
from path_planner.v2.profiles import (
    PlatformProfileV2,
    WheelKinematicSQPProfileV2,
    audit_wheel_kinematic_sqp_profile_v2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineSafetyAnchorV2, snapshot_hash
from path_planner.v2.wheel_corridors import (
    WheelCorridorGenerationResultV2,
    WheelSQPTerrainGuideV1,
    generate_wheel_corridors_v2,
    wheel_corridor_path_hash_v1,
)
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
    WheelCorridorV2,
    WheelSQPInitialGuessV2,
    WheelSQPOptimizationResultV2,
    WheelSQPRepairConstraintV1,
    WheelSQPRepairNotAllowed,
    WheelSQPSearchTelemetryV2,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    WheelTopologySignatureV1,
    WheelTrajectoryL2ResultV2,
)
from path_planner.v2.wheel_sqp_initialization import (
    WheelSQPInitializationError,
    initialize_wheel_trajectory_v2,
)
from path_planner.v2.wheel_sqp_serialization import (
    CanonicalWheelCandidateV1,
    WheelSQPCodecError,
    encode_wheel_candidate_v2,
    materialize_canonical_wheel_candidate_v2,
    wheel_sqp_profile_hash_v2,
    wheel_sqp_request_hash_v2,
)
from path_planner.v2 import wheel_sqp_solver
from path_planner.v2.wheel_sqp_validation import (
    promote_wheel_candidate_to_route_v2,
    repair_constraint_from_counterexample_v2,
    validate_wheel_sqp_candidate_l2,
    validate_wheel_stationary_footprint_v2,
)


WHEEL_SQP_CACHE_NAMESPACE_DISABLED_V2 = (
    "path-planner-v2-wheel-sqp-cache-disabled/v1"
)
WHEEL_SQP_CACHE_KEY_DISABLED_V2 = "wheel-sqp-cache-disabled/v1"
WHEEL_SQP_DECISION_DOMAIN_V1 = "wheel_sqp_first_passing_route_decision/v1"

_DECISION_AUDIT_SINK_V1: ContextVar[object | None] = ContextVar(
    "wheel_sqp_decision_audit_sink_v1",
    default=None,
)

_ZERO_HASH = "0" * 64
_PRESERVED_L2_REASONS = frozenset(
    {
        "planning_deadline_expired",
        "wheel_sqp_identity_mismatch",
        "wheel_sqp_numeric_contract_failed",
        "wheel_sqp_internal_error",
        "wheel_sqp_corridor_budget_exceeded",
        "wheel_sqp_resource_budget_exceeded",
        "wheel_sqp_goal_tolerance_exceeded",
    }
)
_FAILURE_PRECEDENCE = {
    "planning_deadline_expired": 0,
    "wheel_sqp_identity_mismatch": 1,
    "wheel_sqp_numeric_contract_failed": 2,
    "wheel_sqp_internal_error": 3,
    "wheel_sqp_backend_unavailable": 4,
    "wheel_sqp_objective_unsupported": 5,
    "wheel_sqp_corridor_budget_exceeded": 6,
    "wheel_sqp_resource_budget_exceeded": 7,
    "wheel_sqp_goal_tolerance_exceeded": 8,
    "wheel_sqp_repair_l2_rejected": 9,
    "wheel_sqp_candidate_l2_rejected": 10,
    "wheel_sqp_infeasible": 11,
    "wheel_sqp_initialization_failed": 12,
}
_FAILURE_CATEGORY = {
    "wheel_sqp_profile_unsupported": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_backend_unavailable": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_objective_unsupported": FailureCategoryV2.UNSUPPORTED_CAPABILITY,
    "wheel_sqp_accelerator_required_unsupported": (
        FailureCategoryV2.UNSUPPORTED_CAPABILITY
    ),
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
class WheelKinematicSQPProviderV2:
    wheel_sqp_profile: WheelKinematicSQPProfileV2

    def __post_init__(self) -> None:
        if type(self.wheel_sqp_profile) is not WheelKinematicSQPProfileV2:
            raise TypeError(
                "wheel_sqp_profile must be exact WheelKinematicSQPProfileV2"
            )

    @property
    def profile(self) -> PlatformProfileV2:
        return self.wheel_sqp_profile.profile

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
        return _plan_wheel_sqp_v2(self, request, anchor, deadline)


@dataclass(slots=True)
class _CallStateV1:
    snapshot_digest: str
    exact_hold: bool = False
    public_corridor_count: int = 0
    corridor_expansions: int = 0
    generated_primitives: int = 0
    rejected_l2: int = 0
    sqp_iterations: int = 0
    sqp_function_evaluations: int = 0
    l2_interval_count: int = 0
    l2_cell_count: int = 0
    repair_attempted: bool = False
    corridor_payload: tuple[dict[str, object], ...] = ()
    events: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _AttemptOutcomeV1:
    initial_guess: WheelSQPInitialGuessV2 | None
    optimization: WheelSQPOptimizationResultV2 | None
    candidate: CanonicalWheelCandidateV1 | None
    candidate_bytes: bytes | None
    l2: WheelTrajectoryL2ResultV2 | None
    reason_code: str | None


def _safe_elapsed(deadline: PlanningDeadlineV2) -> float:
    try:
        elapsed = deadline.elapsed_s
    except Exception:
        return 0.0
    if type(elapsed) not in (int, float):
        return 0.0
    normalized = float(elapsed)
    return normalized if isfinite(normalized) and normalized >= 0.0 else 0.0


def _telemetry(
    deadline: PlanningDeadlineV2,
    state: _CallStateV1 | None,
    ledger: WheelSQPWorkLedgerV1 | None,
    *,
    termination_reason: str,
    selected_corridor_index: int | None,
    decision_hash: str | None,
) -> WheelSQPSearchTelemetryV2:
    telemetry = WheelSQPSearchTelemetryV2(
        expanded_states=0 if ledger is None else ledger.expanded_states,
        generated_primitives=0 if state is None else state.generated_primitives,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=0 if state is None else state.rejected_l2,
        elapsed_s=_safe_elapsed(deadline),
        timed_out=termination_reason == "planning_deadline_expired",
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason=termination_reason,
        corridor_count=0 if state is None else state.public_corridor_count,
        selected_corridor_index=selected_corridor_index,
        corridor_expansions=0 if state is None else state.corridor_expansions,
        sqp_iterations=0 if state is None else state.sqp_iterations,
        sqp_function_evaluations=(
            0 if state is None else state.sqp_function_evaluations
        ),
        l2_interval_count=0 if state is None else state.l2_interval_count,
        l2_cell_count=0 if state is None else state.l2_cell_count,
        repair_attempted=False if state is None else state.repair_attempted,
        decision_hash=decision_hash,
    )
    audit_sink = _DECISION_AUDIT_SINK_V1.get()
    if audit_sink is not None:
        if not callable(audit_sink):
            raise TypeError("wheel SQP decision audit sink must be callable")
        audit_sink((termination_reason, decision_hash))
    return telemetry


def _failure(
    request: PlanningRequestV2,
    deadline: PlanningDeadlineV2,
    reason_code: str,
    stage: str,
    *,
    state: _CallStateV1 | None = None,
    ledger: WheelSQPWorkLedgerV1 | None = None,
    decision_hash: str | None = None,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> PlanningFailureV2:
    if deadline.expired:
        reason_code = "planning_deadline_expired"
        stage = "deadline"
        details = ()
    category = _FAILURE_CATEGORY.get(reason_code, FailureCategoryV2.INTERNAL_ERROR)
    if reason_code not in _FAILURE_CATEGORY:
        reason_code = "wheel_sqp_internal_error"
    return PlanningFailureV2(
        request_id=request.request_id,
        platform_kind=PlatformKindV2.WHEEL,
        category=category,
        reason_code=reason_code,
        evidence=FailureEvidenceV2(
            stage=stage,
            checks=(reason_code,),
            details=tuple(sorted(details, key=lambda item: item[0])),
        ),
        search_telemetry=_telemetry(
            deadline,
            state,
            ledger,
            termination_reason=reason_code,
            selected_corridor_index=None,
            decision_hash=decision_hash,
        ),
    )


def _corridor_payload(corridor: WheelCorridorV2) -> dict[str, object]:
    return {
        "corridor_hash": corridor.corridor_hash,
        "corridor_index": corridor.corridor_index,
        "guide_cost": corridor.guide_cost,
        "path_length_m": corridor.path_length_m,
        "topology_signature": tuple(
            (component_hash, crossing_count)
            for component_hash, crossing_count in corridor.topology_signature.entries
        ),
    }


def _counterexample_payload(receipt: WheelTrajectoryL2ResultV2) -> object:
    counterexample = receipt.counterexample
    if counterexample is None:
        return None
    return {
        "candidate_hash": counterexample.candidate_hash,
        "candidate_segment_hash": counterexample.candidate_segment_hash,
        "cell": (counterexample.cell.x, counterexample.cell.y),
        "repairable": counterexample.repairable,
        "segment_index": counterexample.segment_index,
        "snapshot_hash": counterexample.snapshot_hash,
        "terrain_reason": counterexample.terrain_reason,
        "time_fractions": (
            counterexample.time_fraction_lo,
            counterexample.time_fraction_mid,
            counterexample.time_fraction_hi,
        ),
    }


def _repair_payload(repair: WheelSQPRepairConstraintV1 | None) -> object:
    if repair is None:
        return None
    return {
        "cell": (repair.cell.x, repair.cell.y),
        "cell_bounds_m": repair.cell_bounds_m,
        "clearance_m": repair.clearance_m,
        "face": repair.face,
        "segment_index": repair.segment_index,
        "source_candidate_hash": repair.source_candidate_hash,
        "source_segment_hash": repair.source_segment_hash,
        "time_fractions": repair.time_fractions,
    }


def _record_attempt(
    state: _CallStateV1,
    corridor: WheelCorridorV2,
    attempt: _AttemptOutcomeV1,
    *,
    attempt_kind: str,
    repair: WheelSQPRepairConstraintV1 | None,
) -> None:
    optimization = attempt.optimization
    if optimization is not None:
        state.sqp_iterations += optimization.iteration_count
        state.sqp_function_evaluations += optimization.function_evaluation_count
        if attempt.initial_guess is not None:
            state.generated_primitives += len(attempt.initial_guess.segments)
    receipt = attempt.l2
    if receipt is not None:
        state.l2_interval_count += receipt.checked_interval_count
        state.l2_cell_count += receipt.checked_cell_count
        if not receipt.passed:
            state.rejected_l2 += 1
    state.events.append(
        {
            "attempt_kind": attempt_kind,
            "candidate_bytes_hash": (
                None
                if attempt.candidate_bytes is None
                else sha256(attempt.candidate_bytes).hexdigest()
            ),
            "corridor_index": corridor.corridor_index,
            "counterexample": (
                None if receipt is None else _counterexample_payload(receipt)
            ),
            "l2_cell_count": 0 if receipt is None else receipt.checked_cell_count,
            "l2_interval_count": (
                0 if receipt is None else receipt.checked_interval_count
            ),
            "l2_passed": None if receipt is None else receipt.passed,
            "l2_reason": None if receipt is None else receipt.reason_code,
            "mode_sequence": (
                ()
                if attempt.initial_guess is None
                else tuple(mode.value for mode in attempt.initial_guess.modes)
            ),
            "repair": _repair_payload(repair),
            "solver_evaluations": (
                0 if optimization is None else optimization.function_evaluation_count
            ),
            "solver_iterations": (
                0 if optimization is None else optimization.iteration_count
            ),
            "solver_reason": (
                attempt.reason_code
                if optimization is None
                else optimization.reason_code
            ),
            "solver_status": (
                None if optimization is None else optimization.status.value
            ),
        }
    )


def _decision_hash(
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    state: _CallStateV1,
    *,
    final_reason: str,
    route_hash: str | None,
    selected_corridor_index: int | None,
    repair_applied: bool,
) -> str:
    payload = {
        "corridors": state.corridor_payload,
        "domain": WHEEL_SQP_DECISION_DOMAIN_V1,
        "events": tuple(state.events),
        "exact_hold": state.exact_hold,
        "final_reason": final_reason,
        "profile_hash": wheel_sqp_profile_hash_v2(profile),
        "repair_applied": repair_applied,
        "request_hash": wheel_sqp_request_hash_v2(
            request,
            state.snapshot_digest,
        ),
        "route_hash": route_hash,
        "selected_corridor_index": selected_corridor_index,
        "snapshot_hash": state.snapshot_digest,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _exact_hold_corridor(
    request: PlanningRequestV2,
    snapshot_digest: str,
) -> WheelCorridorV2:
    geometry = request.terrain_snapshot.geometry
    cell = geometry.world_to_cell(
        WorldPoint(request.start_state.x_m, request.start_state.y_m)
    )
    cells = (cell,)
    return WheelCorridorV2(
        corridor_index=0,
        cells=cells,
        corridor_hash=wheel_corridor_path_hash_v1(snapshot_digest, cells),
        guide_cost=0.0,
        path_length_m=0.0,
        topology_signature=WheelTopologySignatureV1(()),
    )


def _run_attempt(
    corridor: WheelCorridorV2,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    profile: WheelKinematicSQPProfileV2,
    terrain_guide: WheelSQPTerrainGuideV1,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    snapshot_digest: str,
    *,
    initial_guess: WheelSQPInitialGuessV2 | None,
    repair: WheelSQPRepairConstraintV1 | None,
) -> _AttemptOutcomeV1:
    optimization: WheelSQPOptimizationResultV2 | None = None
    candidate: CanonicalWheelCandidateV1 | None = None
    candidate_bytes: bytes | None = None
    try:
        if initial_guess is None:
            initial_guess = initialize_wheel_trajectory_v2(
                corridor,
                request,
                profile,
                ledger,
                deadline,
            )
        ledger.check_deadline()
        problem = wheel_sqp_solver.build_wheel_sqp_problem_v1(
            corridor,
            initial_guess,
            request,
            profile,
            terrain_guide,
            deadline,
            ledger,
            repair=repair,
        )
        optimization = wheel_sqp_solver.solve_wheel_sqp_v2(
            problem,
            deadline,
            ledger,
        )
        ledger.check_deadline()
        if optimization.candidate is None:
            return _AttemptOutcomeV1(
                initial_guess,
                optimization,
                None,
                None,
                None,
                optimization.reason_code or "wheel_sqp_internal_error",
            )
        candidate = materialize_canonical_wheel_candidate_v2(
            optimization.candidate,
            request=request,
            profile=profile,
            terrain_snapshot_hash=snapshot_digest,
        )
        ledger.check_deadline()
        candidate_bytes = encode_wheel_candidate_v2(candidate)
        ledger.check_deadline()
        receipt = validate_wheel_sqp_candidate_l2(
            candidate_bytes,
            request,
            anchor,
            profile,
            deadline,
            ledger,
        )
        ledger.check_deadline()
        if type(receipt) is not WheelTrajectoryL2ResultV2:
            raise ValueError("L2 validator returned the wrong receipt type")
        return _AttemptOutcomeV1(
            initial_guess,
            optimization,
            candidate,
            candidate_bytes,
            receipt,
            None,
        )
    except WheelSQPInitializationError as exc:
        return _AttemptOutcomeV1(
            initial_guess,
            optimization,
            candidate,
            candidate_bytes,
            None,
            exc.reason_code,
        )
    except WheelSQPWorkLimitError as exc:
        return _AttemptOutcomeV1(
            initial_guess,
            optimization,
            candidate,
            candidate_bytes,
            None,
            exc.reason_code,
        )
    except WheelSQPCodecError:
        return _AttemptOutcomeV1(
            initial_guess,
            optimization,
            candidate,
            candidate_bytes,
            None,
            "wheel_sqp_numeric_contract_failed",
        )
    except (TypeError, ValueError, OverflowError):
        return _AttemptOutcomeV1(
            initial_guess,
            optimization,
            candidate,
            candidate_bytes,
            None,
            "wheel_sqp_identity_mismatch",
        )


def _normalized_l2_reason(
    receipt: WheelTrajectoryL2ResultV2,
    *,
    repair_attempt: bool,
) -> str:
    if receipt.reason_code in _PRESERVED_L2_REASONS:
        return receipt.reason_code
    return (
        "wheel_sqp_repair_l2_rejected"
        if repair_attempt
        else "wheel_sqp_candidate_l2_rejected"
    )


def _select_failure(reasons: list[str]) -> str:
    if not reasons:
        return "wheel_sqp_internal_error"
    normalized = [
        reason if reason in _FAILURE_PRECEDENCE else "wheel_sqp_internal_error"
        for reason in reasons
    ]
    return min(normalized, key=lambda reason: (_FAILURE_PRECEDENCE[reason], reason))


def _observation_samples(candidate: CanonicalWheelCandidateV1):
    samples = []
    for segment in candidate.segments:
        if not samples:
            samples.extend(segment.samples)
        else:
            samples.extend(segment.samples[1:])
    return tuple(samples)


def _build_success(
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    state: _CallStateV1,
    corridor: WheelCorridorV2,
    attempt: _AttemptOutcomeV1,
    *,
    repair_applied: bool,
) -> PlanningSuccessV2:
    if attempt.candidate is None or attempt.l2 is None or not attempt.l2.passed:
        raise ValueError("success requires a canonical candidate and passed L2 receipt")
    ledger.check_deadline()
    promotion = promote_wheel_candidate_to_route_v2(
        attempt.candidate,
        attempt.l2,
        repair_applied=repair_applied,
    )
    ledger.check_deadline()
    objective = request.objective_profile
    distance_cost = objective.distance_weight * attempt.candidate.total_distance_m
    energy_cost = objective.energy_weight * (
        attempt.candidate.total_relative_energy / profile.energy_normalization
    )
    time_cost = objective.time_weight * (
        attempt.candidate.total_duration_s / profile.time_normalization_s
    )
    costs = CostBreakdownV2(
        distance_cost=distance_cost,
        risk_cost=0.0,
        energy_cost=energy_cost,
        time_cost=time_cost,
        total_cost=promotion.route.total_cost,
    )
    selected_index = None if state.exact_hold else corridor.corridor_index
    decision_hash = _decision_hash(
        request,
        profile,
        state,
        final_reason="wheel_sqp_route_selected",
        route_hash=promotion.route.route_hash,
        selected_corridor_index=selected_index,
        repair_applied=repair_applied,
    )
    success = PlanningSuccessV2(
        request_id=request.request_id,
        platform_kind=PlatformKindV2.WHEEL,
        route=promotion.route,
        observation_projection=ObservationProjectionV2(
            source=WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2,
            sample_states=_observation_samples(attempt.candidate),
            expected_new_observed_cells=0.0,
            expected_information_gain=0.0,
        ),
        cost_breakdown=costs,
        validation_evidence=promotion.evidence,
        search_telemetry=_telemetry(
            deadline,
            state,
            ledger,
            termination_reason="wheel_sqp_route_selected",
            selected_corridor_index=selected_index,
            decision_hash=decision_hash,
        ),
        cache_evidence=CacheEvidenceV2(
            cache_namespace=WHEEL_SQP_CACHE_NAMESPACE_DISABLED_V2,
            cache_key=WHEEL_SQP_CACHE_KEY_DISABLED_V2,
            hit=False,
        ),
    )
    ledger.check_deadline()
    return success


def _finish_failure(
    request: PlanningRequestV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    state: _CallStateV1,
    reasons: list[str],
    *,
    stage: str,
) -> PlanningFailureV2:
    reason = (
        "planning_deadline_expired"
        if deadline.expired
        else _select_failure(reasons)
    )
    decision_hash = _decision_hash(
        request,
        profile,
        state,
        final_reason=reason,
        route_hash=None,
        selected_corridor_index=None,
        repair_applied=False,
    )
    return _failure(
        request,
        deadline,
        reason,
        stage,
        state=state,
        ledger=ledger,
        decision_hash=decision_hash,
    )


def _run_corridors(
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    profile: WheelKinematicSQPProfileV2,
    deadline: PlanningDeadlineV2,
    ledger: WheelSQPWorkLedgerV1,
    state: _CallStateV1,
    corridors: tuple[WheelCorridorV2, ...],
    terrain_guide: WheelSQPTerrainGuideV1,
) -> PlanningOutcomeV2:
    reasons: list[str] = []
    for corridor in corridors:
        if deadline.expired:
            reasons.append("planning_deadline_expired")
            break
        original = _run_attempt(
            corridor,
            request,
            anchor,
            profile,
            terrain_guide,
            deadline,
            ledger,
            state.snapshot_digest,
            initial_guess=None,
            repair=None,
        )
        _record_attempt(
            state,
            corridor,
            original,
            attempt_kind="original",
            repair=None,
        )
        if deadline.expired:
            reasons.append("planning_deadline_expired")
            break
        if original.reason_code is not None:
            reasons.append(original.reason_code)
            if original.reason_code == "planning_deadline_expired":
                break
            continue
        assert original.l2 is not None
        if original.l2.passed:
            return _build_success(
                request,
                profile,
                deadline,
                ledger,
                state,
                corridor,
                original,
                repair_applied=False,
            )
        original_l2_reason = _normalized_l2_reason(
            original.l2,
            repair_attempt=False,
        )
        reasons.append(original_l2_reason)
        if original_l2_reason == "planning_deadline_expired":
            break
        counterexample = original.l2.counterexample
        if (
            state.exact_hold
            or counterexample is None
            or counterexample.repairable is not True
        ):
            continue
        assert original.candidate_bytes is not None
        try:
            repair = repair_constraint_from_counterexample_v2(
                counterexample,
                original.candidate_bytes,
                profile,
                geometry=request.terrain_snapshot.geometry,
            )
        except WheelSQPRepairNotAllowed:
            reasons.append("wheel_sqp_identity_mismatch")
            continue
        if deadline.expired:
            reasons.append("planning_deadline_expired")
            break
        state.repair_attempted = True
        repaired = _run_attempt(
            corridor,
            request,
            anchor,
            profile,
            terrain_guide,
            deadline,
            ledger,
            state.snapshot_digest,
            initial_guess=original.initial_guess,
            repair=repair,
        )
        _record_attempt(
            state,
            corridor,
            repaired,
            attempt_kind="repair",
            repair=repair,
        )
        if deadline.expired:
            reasons.append("planning_deadline_expired")
            break
        if repaired.reason_code is not None:
            reasons.append(repaired.reason_code)
            if repaired.reason_code == "planning_deadline_expired":
                break
            continue
        assert repaired.l2 is not None
        if repaired.l2.passed:
            return _build_success(
                request,
                profile,
                deadline,
                ledger,
                state,
                corridor,
                repaired,
                repair_applied=True,
            )
        repaired_l2_reason = _normalized_l2_reason(
            repaired.l2,
            repair_attempt=True,
        )
        reasons.append(repaired_l2_reason)
        if repaired_l2_reason == "planning_deadline_expired":
            break
    return _finish_failure(
        request,
        profile,
        deadline,
        ledger,
        state,
        reasons,
        stage="corridor_loop",
    )


def _plan_wheel_sqp_v2(
    provider: WheelKinematicSQPProviderV2,
    request: PlanningRequestV2,
    anchor: FineSafetyAnchorV2,
    deadline: PlanningDeadlineV2,
) -> PlanningOutcomeV2:
    stage = "preflight"
    state: _CallStateV1 | None = None
    ledger: WheelSQPWorkLedgerV1 | None = None
    try:
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)
        try:
            profile = audit_wheel_kinematic_sqp_profile_v2(
                provider.wheel_sqp_profile
            )
        except (TypeError, ValueError, OverflowError):
            return _failure(
                request,
                deadline,
                "wheel_sqp_profile_unsupported",
                stage,
            )
        if profile != provider.wheel_sqp_profile:
            return _failure(
                request,
                deadline,
                "wheel_sqp_profile_unsupported",
                stage,
            )
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)
        if request.platform_profile_id != profile.profile.profile_id:
            return _failure(
                request,
                deadline,
                "wheel_sqp_profile_unsupported",
                stage,
            )
        if anchor.snapshot is not request.terrain_snapshot:
            return _failure(
                request,
                deadline,
                "wheel_sqp_identity_mismatch",
                stage,
            )
        snapshot_digest = snapshot_hash(request.terrain_snapshot)
        if anchor._snapshot_hash != snapshot_digest:
            return _failure(
                request,
                deadline,
                "wheel_sqp_identity_mismatch",
                stage,
            )
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)
        if wheel_sqp_solver._load_scipy_optimize_v1() is None:
            return _failure(
                request,
                deadline,
                "wheel_sqp_backend_unavailable",
                stage,
            )
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)
        if request.objective_profile.risk_weight != 0.0:
            return _failure(
                request,
                deadline,
                "wheel_sqp_objective_unsupported",
                stage,
            )
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)
        if request.accelerator_policy is AcceleratorPolicyV2.REQUIRED:
            return _failure(
                request,
                deadline,
                "wheel_sqp_accelerator_required_unsupported",
                stage,
            )
        if deadline.expired:
            return _failure(request, deadline, "planning_deadline_expired", stage)

        ledger = WheelSQPWorkLedgerV1(request.resource_budget, deadline)
        stage = "start_footprint"
        start_reason = validate_wheel_stationary_footprint_v2(
            request.start_state,
            anchor,
            profile,
            deadline,
            ledger,
        )
        if deadline.expired:
            return _failure(
                request,
                deadline,
                "planning_deadline_expired",
                stage,
                ledger=ledger,
            )
        if start_reason is not None:
            return _failure(
                request,
                deadline,
                "wheel_sqp_start_invalid",
                stage,
                ledger=ledger,
                details=(("terrain_reason", start_reason),),
            )
        stage = "goal_footprint"
        goal_reason = validate_wheel_stationary_footprint_v2(
            request.goal_state,
            anchor,
            profile,
            deadline,
            ledger,
        )
        if deadline.expired:
            return _failure(
                request,
                deadline,
                "planning_deadline_expired",
                stage,
                ledger=ledger,
            )
        if goal_reason is not None:
            return _failure(
                request,
                deadline,
                "wheel_sqp_goal_invalid",
                stage,
                ledger=ledger,
                details=(("terrain_reason", goal_reason),),
            )

        exact_hold = request.start_state == request.goal_state
        stage = "corridor_generation"
        if exact_hold:
            corridors = (_exact_hold_corridor(request, snapshot_digest),)
            corridor_expansions = 0
            public_corridor_count = 0
            corridor_payload: tuple[dict[str, object], ...] = ()
        else:
            geometry = request.terrain_snapshot.geometry
            start_cell = geometry.world_to_cell(
                WorldPoint(request.start_state.x_m, request.start_state.y_m)
            )
            goal_cell = geometry.world_to_cell(
                WorldPoint(request.goal_state.x_m, request.goal_state.y_m)
            )
            generated = generate_wheel_corridors_v2(
                request.terrain_snapshot,
                start_cell,
                goal_cell,
                request.resource_budget,
                deadline,
                ledger=ledger,
            )
            if type(generated) is not WheelCorridorGenerationResultV2:
                raise ValueError("corridor generator returned the wrong result type")
            corridor_expansions = generated.expanded_states
            if generated.reason_code != "wheel_sqp_corridors_ready":
                state = _CallStateV1(
                    snapshot_digest=snapshot_digest,
                    corridor_expansions=corridor_expansions,
                )
                return _failure(
                    request,
                    deadline,
                    generated.reason_code,
                    stage,
                    state=state,
                    ledger=ledger,
                )
            corridors = generated.corridors
            public_corridor_count = len(corridors)
            corridor_payload = tuple(_corridor_payload(item) for item in corridors)
        if deadline.expired:
            return _failure(
                request,
                deadline,
                "planning_deadline_expired",
                stage,
                ledger=ledger,
            )
        state = _CallStateV1(
            snapshot_digest=snapshot_digest,
            exact_hold=exact_hold,
            public_corridor_count=public_corridor_count,
            corridor_expansions=corridor_expansions,
            corridor_payload=corridor_payload,
        )
        stage = "terrain_guide"
        terrain_guide = WheelSQPTerrainGuideV1.from_snapshot(
            request.terrain_snapshot,
            profile.profile.max_traversable_slope_deg,
            ledger=ledger,
        )
        ledger.check_deadline()
        stage = "corridor_loop"
        return _run_corridors(
            request,
            anchor,
            profile,
            deadline,
            ledger,
            state,
            corridors,
            terrain_guide,
        )
    except WheelSQPWorkLimitError as exc:
        return _failure(
            request,
            deadline,
            exc.reason_code,
            stage,
            state=state,
            ledger=ledger,
        )
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        return _failure(
            request,
            deadline,
            "wheel_sqp_internal_error",
            stage,
            state=state,
            ledger=ledger,
            details=(("exception_type", type(exc).__name__),),
        )
