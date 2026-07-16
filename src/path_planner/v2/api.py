from __future__ import annotations

from collections.abc import Mapping
from math import floor, hypot, remainder, tau
from time import monotonic
from typing import Callable

from path_planner.core import Cell
from path_planner.v2.contracts import (
    FailureCategoryV2,
    FailureEvidenceV2,
    PlanningFailureV2,
    PlanningOutcomeV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    SearchTelemetryV2,
)
from path_planner.v2.profiles import PlatformProfileRegistryV2, PlatformProfileV2
from path_planner.v2.providers import PrimitiveProviderV2
from path_planner.v2.runtime import MonotonicClockV2, PlanningDeadlineV2
from path_planner.v2.terrain import FineSafetyAnchorV2, SafetyQueryV2, TerrainSnapshotV2


PlanV2Fn = Callable[
    [
        PlanningRequestV2,
        PlatformProfileRegistryV2,
        Mapping[str, PrimitiveProviderV2],
    ],
    PlanningOutcomeV2,
]


def _zero_telemetry(
    reason_code: str,
    *,
    elapsed_s: float = 0.0,
    timed_out: bool = False,
) -> SearchTelemetryV2:
    return SearchTelemetryV2(
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
    )


def _failure(
    *,
    request_id: str,
    platform_kind: PlatformKindV2 | None,
    category: FailureCategoryV2,
    reason_code: str,
    stage: str,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    elapsed_s: float = 0.0,
    timed_out: bool = False,
) -> PlanningFailureV2:
    return PlanningFailureV2(
        request_id=request_id,
        platform_kind=platform_kind,
        category=category,
        reason_code=reason_code,
        evidence=FailureEvidenceV2(
            stage=stage,
            checks=(reason_code,),
            details=details,
        ),
        search_telemetry=_zero_telemetry(
            reason_code,
            elapsed_s=elapsed_s,
            timed_out=timed_out,
        ),
    )


def _timeout_failure(
    *,
    request: PlanningRequestV2,
    platform_kind: PlatformKindV2,
    deadline: PlanningDeadlineV2,
    stage: str,
) -> PlanningFailureV2:
    return _failure(
        request_id=request.request_id,
        platform_kind=platform_kind,
        category=FailureCategoryV2.TIMEOUT,
        reason_code="planning_deadline_expired",
        stage=stage,
        elapsed_s=deadline.elapsed_s,
        timed_out=True,
    )


def _pose_cell(snapshot: TerrainSnapshotV2, pose: PoseStateV2) -> Cell:
    geometry = snapshot.geometry
    return Cell(
        int(floor((pose.x_m - geometry.origin[0]) / geometry.resolution_m)),
        int(floor((pose.y_m - geometry.origin[1]) / geometry.resolution_m)),
    )


def _safety_failure(
    *,
    request: PlanningRequestV2,
    platform_kind: PlatformKindV2,
    query: SafetyQueryV2,
    endpoint: str,
) -> PlanningFailureV2:
    category = (
        FailureCategoryV2.UNSAFE_START
        if endpoint == "start"
        else FailureCategoryV2.UNSAFE_GOAL
    )
    return _failure(
        request_id=request.request_id,
        platform_kind=platform_kind,
        category=category,
        reason_code=query.reason_code,
        stage=f"{endpoint}_safety",
        details=(
            ("cell_x", query.cell.x),
            ("cell_y", query.cell.y),
            ("snapshot_hash", query.snapshot_hash),
        ),
    )


def _outcome_is_valid(
    outcome: object,
    request: PlanningRequestV2,
    profile: PlatformProfileV2,
) -> bool:
    if type(outcome) is PlanningSuccessV2:
        success = outcome
        actual_goal = success.route.primitives[-1].end_state
        requested_goal = request.goal_state
        position_error_m = hypot(
            actual_goal.x_m - requested_goal.x_m,
            actual_goal.y_m - requested_goal.y_m,
        )
        raw_heading_error = remainder(
            remainder(actual_goal.heading_rad, tau)
            - remainder(requested_goal.heading_rad, tau),
            tau,
        )
        heading_error_rad = abs(raw_heading_error)
        return (
            success.request_id == request.request_id
            and success.platform_kind is profile.platform_kind
            and success.route.primitives[0].start_state == request.start_state
            and position_error_m <= profile.goal_position_tolerance_m
            and heading_error_rad <= profile.goal_heading_tolerance_rad
        )
    if type(outcome) is PlanningFailureV2:
        failure = outcome
        return (
            failure.request_id == request.request_id
            and failure.platform_kind is profile.platform_kind
        )
    return False


def plan_v2(
    request: PlanningRequestV2,
    *,
    registry: PlatformProfileRegistryV2,
    providers: Mapping[str, PrimitiveProviderV2],
    monotonic_clock: MonotonicClockV2 = monotonic,
) -> PlanningOutcomeV2:
    if not isinstance(request, PlanningRequestV2):
        raise TypeError("request must be PlanningRequestV2")
    if not isinstance(registry, PlatformProfileRegistryV2):
        raise TypeError("registry must be PlatformProfileRegistryV2")
    if not isinstance(providers, Mapping):
        raise TypeError("providers must be a mapping")
    if not callable(monotonic_clock):
        raise TypeError("monotonic_clock must be callable")

    started_monotonic_s = monotonic_clock()
    deadline = PlanningDeadlineV2(
        started_monotonic_s=started_monotonic_s,
        deadline_monotonic_s=started_monotonic_s + min(request.timeout_s, 2.0),
        _monotonic_clock=monotonic_clock,
    )

    profile = registry.resolve(request.platform_profile_id)
    if profile is None:
        return _failure(
            request_id=request.request_id,
            platform_kind=None,
            category=FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            reason_code="platform_profile_unresolved",
            stage="profile_resolution",
            details=(("profile_id", request.platform_profile_id),),
        )

    snapshot = request.terrain_snapshot
    if type(snapshot) is not TerrainSnapshotV2:
        return _failure(
            request_id=request.request_id,
            platform_kind=profile.platform_kind,
            category=FailureCategoryV2.INVALID_REQUEST,
            reason_code="terrain_snapshot_invalid",
            stage="terrain_validation",
            details=(("terrain_type", type(snapshot).__name__),),
        )
    anchor = FineSafetyAnchorV2(snapshot)

    start_query = anchor.query(
        _pose_cell(snapshot, request.start_state),
        max_slope_deg=profile.max_traversable_slope_deg,
    )
    if not start_query.passed:
        return _safety_failure(
            request=request,
            platform_kind=profile.platform_kind,
            query=start_query,
            endpoint="start",
        )

    goal_query = anchor.query(
        _pose_cell(snapshot, request.goal_state),
        max_slope_deg=profile.max_traversable_slope_deg,
    )
    if not goal_query.passed:
        return _safety_failure(
            request=request,
            platform_kind=profile.platform_kind,
            query=goal_query,
            endpoint="goal",
        )

    provider = providers.get(profile.profile_id)
    if provider is None:
        return _failure(
            request_id=request.request_id,
            platform_kind=profile.platform_kind,
            category=FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            reason_code="primitive_provider_unregistered",
            stage="provider_resolution",
            details=(("profile_id", profile.profile_id),),
        )

    try:
        provider_profile = provider.profile
        provider_plan = provider.plan
    except Exception:
        provider_profile = None
        provider_plan = None
    if (
        type(provider_profile) is not type(profile)
        or provider_profile != profile
        or provider_profile.profile_id != profile.profile_id
        or not callable(provider_plan)
    ):
        return _failure(
            request_id=request.request_id,
            platform_kind=profile.platform_kind,
            category=FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            reason_code="primitive_provider_profile_mismatch",
            stage="provider_resolution",
            details=(("profile_id", profile.profile_id),),
        )

    if deadline.expired:
        return _timeout_failure(
            request=request,
            platform_kind=profile.platform_kind,
            deadline=deadline,
            stage="provider_dispatch",
        )

    try:
        outcome = provider_plan(request, anchor, deadline)
    except Exception as exc:
        if deadline.expired:
            return _timeout_failure(
                request=request,
                platform_kind=profile.platform_kind,
                deadline=deadline,
                stage="provider_completion",
            )
        return _failure(
            request_id=request.request_id,
            platform_kind=profile.platform_kind,
            category=FailureCategoryV2.INTERNAL_ERROR,
            reason_code="primitive_provider_exception",
            stage="provider_execution",
            details=(("exception_type", type(exc).__name__),),
        )

    if deadline.expired:
        return _timeout_failure(
            request=request,
            platform_kind=profile.platform_kind,
            deadline=deadline,
            stage="provider_completion",
        )

    outcome_is_valid = _outcome_is_valid(outcome, request, profile)
    if deadline.expired:
        return _timeout_failure(
            request=request,
            platform_kind=profile.platform_kind,
            deadline=deadline,
            stage="provider_postcondition",
        )

    if not outcome_is_valid:
        return _failure(
            request_id=request.request_id,
            platform_kind=profile.platform_kind,
            category=FailureCategoryV2.INTERNAL_ERROR,
            reason_code="primitive_provider_outcome_invalid",
            stage="provider_postcondition",
            details=(("outcome_type", type(outcome).__name__),),
        )
    return outcome
