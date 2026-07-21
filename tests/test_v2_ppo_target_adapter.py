from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi
from typing import get_args

import numpy as np
import pytest

from path_planner.v2.adapters import (
    BuildPpoRequestFn,
    PlanPpoTargetFn,
    PpoTargetV2,
    build_ppo_request_v2,
    plan_ppo_target_v2,
)
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    CacheEvidenceV2,
    CostBreakdownV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObjectiveProfileV2,
    ObservationProjectionV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
    SearchTelemetryV2,
    TypedRouteV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.observation import ObservedTerrainInputV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


@dataclass(frozen=True, slots=True)
class _SampledPrimitive(RoutePrimitiveV2):
    samples: tuple[PoseStateV2, ...]


def _observed(*, hidden_value: float = 5.0, source_hash: str = "truth-a") -> ObservedTerrainInputV2:
    geometry = FineGridGeometryV2(width=6, height=3)
    observed = np.ones(geometry.shape, dtype=bool)
    observed[:, -1] = False
    elevation = np.zeros(geometry.shape, dtype=np.float64)
    slope = np.zeros(geometry.shape, dtype=np.float64)
    confidence = np.ones(geometry.shape, dtype=np.float64)
    elevation[:, -1] = hidden_value
    slope[:, -1] = hidden_value
    confidence[:, -1] = hidden_value / 10.0
    return ObservedTerrainInputV2(
        request_id="ppo-request-001",
        start_state=PoseStateV2(0.25, 0.75, -0.0),
        terrain_snapshot=TerrainSnapshotV2(
            geometry=geometry,
            elevation_m=elevation,
            slope_deg=slope,
            traversable_mask=observed.copy(),
            hard_obstacle_mask=np.zeros(geometry.shape, dtype=bool),
            observed_mask=observed,
            confidence=confidence,
            provenance=TerrainProvenanceV2(
                source_kind="observed-test-source/v1",
                source_id="truth-source",
                source_hash=source_hash,
                physical_obstacle_cells_written=False,
                details=(("truth_hash", source_hash),),
            ),
        ),
    )


def _options():
    return (
        "wheel-safe/v1",
        ObjectiveProfileV2(),
        ResourceBudgetV2(max_expanded_states=100, max_route_states=20),
        1.5,
        AcceleratorPolicyV2.OPTIONAL,
        17,
    )


def _request_bytes(request: PlanningRequestV2) -> bytes:
    return canonical_json_bytes(
        {
            "request_id": request.request_id,
            "platform_profile_id": request.platform_profile_id,
            "start": request.start_state,
            "goal": request.goal_state,
            "terrain_snapshot_hash": snapshot_hash(request.terrain_snapshot),
            "objective": request.objective_profile,
            "resource": request.resource_budget,
            "timeout_s": request.timeout_s,
            "accelerator_policy": request.accelerator_policy,
            "determinism_seed": request.determinism_seed,
        }
    )


def _failure() -> PlanningFailureV2:
    return PlanningFailureV2(
        request_id="ppo-request-001",
        platform_kind=PlatformKindV2.WHEEL,
        category=FailureCategoryV2.GOAL_POSE_UNREACHABLE,
        reason_code="wheel_goal_pose_unreachable",
        evidence=FailureEvidenceV2(
            stage="search",
            checks=("wheel_goal_pose_unreachable",),
            details=(),
        ),
        search_telemetry=_telemetry("wheel_goal_pose_unreachable"),
    )


def _telemetry(reason: str) -> SearchTelemetryV2:
    return SearchTelemetryV2(
        expanded_states=1,
        generated_primitives=1,
        rejected_l0=0,
        rejected_l1=0,
        rejected_l2=0,
        elapsed_s=0.01,
        timed_out=False,
        accelerator_used=False,
        ackermann_feasible_claimed=False,
        termination_reason=reason,
    )


def _success(target: PpoTargetV2) -> PlanningSuccessV2:
    start = PoseStateV2(0.25, 0.75, 0.0)
    end = PoseStateV2(target.x_m, target.y_m, target.theta_rad)
    primitive = _SampledPrimitive(
        kind=PrimitiveKindV2.WHEEL_MOTION,
        start_state=start,
        end_state=end,
        duration_s=1.0,
        distance_m=1.0,
        energy_cost=1.0,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        samples=(start, end),
    )
    route = TypedRouteV2(PlatformKindV2.WHEEL, (primitive,), 1.0)
    return PlanningSuccessV2(
        request_id="ppo-request-001",
        platform_kind=PlatformKindV2.WHEEL,
        route=route,
        observation_projection=ObservationProjectionV2(
            source="provider-placeholder/v1",
            sample_states=(start, end),
            expected_new_observed_cells=0.0,
            expected_information_gain=0.0,
        ),
        cost_breakdown=CostBreakdownV2(0.0, 0.0, 0.5, 0.5, 1.0),
        validation_evidence=ValidationEvidenceV2(
            validator_id="test-route-l2/v1",
            level=ValidationLevelV2.L2,
            passed=True,
            checks=("route_l2_valid",),
        ),
        search_telemetry=_telemetry("success"),
        cache_evidence=CacheEvidenceV2("disabled/v1", "disabled", False),
    )


@pytest.mark.parametrize("value", [inf, -inf, float("nan"), True])
def test_ppo_target_rejects_nonfinite_and_boolean_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="finite real"):
        PpoTargetV2(value, 1.0, 0.0)  # type: ignore[arg-type]


def test_build_request_alias_and_target_values_are_exactly_preserved() -> None:
    target = PpoTargetV2(1.25, 0.75, 3.0 * pi)
    observed = _observed()
    profile_id, objective, budget, timeout, accelerator, seed = _options()
    builder: BuildPpoRequestFn = build_ppo_request_v2

    request = builder(
        target,
        observed,
        profile_id,
        objective,
        budget,
        timeout,
        accelerator,
        seed,
    )

    assert request.goal_state.x_m.hex() == target.x_m.hex()
    assert request.goal_state.y_m.hex() == target.y_m.hex()
    assert request.goal_state.heading_rad.hex() == target.theta_rad.hex()
    assert request.start_state == observed.start_state
    assert request.terrain_snapshot is observed.terrain_snapshot
    callable_args, callable_result = get_args(BuildPpoRequestFn)
    assert tuple(callable_args) == (
        PpoTargetV2,
        ObservedTerrainInputV2,
        str,
        ObjectiveProfileV2,
        ResourceBudgetV2,
        float,
        AcceleratorPolicyV2,
        int,
    )
    assert callable_result is PlanningRequestV2


def test_hidden_truth_and_truth_provenance_changes_do_not_change_request_bytes() -> None:
    target = PpoTargetV2(1.25, 0.75, 0.25)
    options = _options()
    left = build_ppo_request_v2(target, _observed(hidden_value=5.0, source_hash="truth-a"), *options)
    right = build_ppo_request_v2(target, _observed(hidden_value=9.0, source_hash="truth-b"), *options)

    assert _request_bytes(left) == _request_bytes(right)


def test_planner_failure_is_returned_by_identity_without_retry_or_reselection() -> None:
    target = PpoTargetV2(1.25, 0.75, 0.25)
    expected = _failure()
    requests: list[PlanningRequestV2] = []

    def planner(request: PlanningRequestV2):
        requests.append(request)
        return expected

    typed_planner: PlanPpoTargetFn = planner
    outcome = plan_ppo_target_v2(
        target,
        _observed(),
        *_options(),
        plan_request=typed_planner,
    )

    assert outcome is expected
    assert len(requests) == 1
    callable_args, callable_result = get_args(PlanPpoTargetFn)
    assert tuple(callable_args) == (PlanningRequestV2,)
    assert callable_result == PlanningSuccessV2 | PlanningFailureV2


def test_success_replaces_only_observation_projection() -> None:
    target = PpoTargetV2(1.25, 0.75, 2.5 * pi)
    expected = _success(target)
    calls = 0

    def planner(_request: PlanningRequestV2):
        nonlocal calls
        calls += 1
        return expected

    outcome = plan_ppo_target_v2(
        target,
        _observed(),
        *_options(),
        plan_request=planner,
    )

    assert isinstance(outcome, PlanningSuccessV2)
    assert calls == 1
    for name in (
        "request_id",
        "platform_kind",
        "route",
        "cost_breakdown",
        "validation_evidence",
        "search_telemetry",
        "cache_evidence",
        "schema_version",
    ):
        assert getattr(outcome, name) is getattr(expected, name)
    assert outcome.observation_projection is not expected.observation_projection
    assert outcome.observation_projection.sample_states[-1].heading_rad.hex() == target.theta_rad.hex()
