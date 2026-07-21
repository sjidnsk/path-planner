from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from math import cos, sin

import numpy as np

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    FailureCategoryV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.hopper_authority import (
    HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
    HOPPER_RESOURCE_AUTHORITY_V2,
    HopperProviderAuthorityV2,
    hopper_gate5b_algorithm_fixture_v1,
)
from path_planner.v2.profiles import (
    HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
    HopperProfileV2,
    PlatformProfileV2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _module():
    return import_module("path_planner.v2.providers.hopper")


def _anchor() -> FineSafetyAnchorV2:
    shape = (64, 64)
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(64, 64, (-8.0, -8.0), "moon"),
        elevation_m=np.zeros(shape, dtype=np.float64),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="gate5b-provider-fixture",
            source_hash="gate5b-provider-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _deadline(now: float = 0.0) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 100.0, lambda: now)


def _request(
    profile: HopperProfileV2,
    anchor: FineSafetyAnchorV2,
    *,
    goal: PoseStateV2 | None = None,
    objective: ObjectiveProfileV2 | None = None,
    budget: ResourceBudgetV2 | None = None,
    accelerator: AcceleratorPolicyV2 = AcceleratorPolicyV2.DISABLED,
) -> PlanningRequestV2:
    start = PoseStateV2(0.25, 0.25, 0.0)
    if goal is None:
        speed = profile.launch_speeds_mps[1]
        elevation = profile.launch_elevations_rad[1]
        flight_time = 2.0 * ((speed * sin(elevation)) / profile.gravity_mps2)
        distance = (speed * cos(elevation)) * flight_time
        goal = PoseStateV2(start.x_m + distance, start.y_m, 0.0)
    return PlanningRequestV2(
        request_id="gate5b-provider",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=anchor.snapshot,
        objective_profile=ObjectiveProfileV2() if objective is None else objective,
        resource_budget=(
            ResourceBudgetV2(10, 100, 0) if budget is None else budget
        ),
        timeout_s=2.0,
        accelerator_policy=accelerator,
        determinism_seed=17,
    )


def _fixture_provider():
    module = _module()
    profile = hopper_gate5b_algorithm_fixture_v1()
    authority = HopperProviderAuthorityV2(
        profile,
        HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.parameter_set_id,
        "hopper-provider-authority/v1",
    )
    return module.HopperPrimitiveProviderV2(authority), profile


def test_hopper_provider_capability_preflight_rejects_incomplete_and_unknown_authority() -> None:
    module = _module()
    platform = PlatformProfileV2(
        profile_id="hopper-lunar-ballistic-proxy/v1",
        platform_kind=PlatformKindV2.HOPPER,
        capability_revision=HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
        simulation_proxy=True,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=0.0,
        goal_heading_tolerance_rad=0.0,
    )
    incomplete = HopperProfileV2(platform)
    anchor = _anchor()
    provider = module.HopperPrimitiveProviderV2(
        HopperProviderAuthorityV2(
            incomplete, None, "hopper-provider-authority/v1"
        )
    )
    outcome = provider.plan(_request(incomplete, anchor), anchor, _deadline())
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
    assert outcome.reason_code == "hopper_proxy_profile_incomplete"
    assert outcome.evidence.stage == "capability_preflight"
    assert outcome.search_telemetry.generated_primitives == 0

    complete = hopper_gate5b_algorithm_fixture_v1()
    unknown = module.HopperPrimitiveProviderV2(
        HopperProviderAuthorityV2(
            complete, "hopper-unknown/v1", "hopper-provider-authority/v1"
        )
    )
    outcome = unknown.plan(_request(complete, anchor), anchor, _deadline())
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "hopper_parameter_set_unsupported"
    assert outcome.search_telemetry.generated_primitives == 0


def test_hopper_provider_returns_one_hop_fixture_success_envelope() -> None:
    module = _module()
    provider, profile = _fixture_provider()
    anchor = _anchor()
    outcome = provider.plan(_request(profile, anchor), anchor, _deadline())
    assert type(outcome) is PlanningSuccessV2
    assert outcome.platform_kind is PlatformKindV2.HOPPER
    assert len(outcome.route.primitives) == 1
    assert type(outcome.route.primitives[0]) is module.HopperJumpPrimitiveV2
    assert outcome.route.primitives[0].selected_landing_mass >= 0.99
    assert outcome.observation_projection.source == (
        "path-planner-v2-hopper-inflight-observation-disabled/v1"
    )
    assert len(outcome.observation_projection.sample_states) == 2
    assert outcome.validation_evidence.checks == ("hopper_route_l2_valid",)
    assert outcome.cache_evidence.cache_namespace == (
        "path-planner-v2-hopper-cache-disabled/v1"
    )
    assert outcome.cache_evidence.hit is False
    assert outcome.search_telemetry.expanded_states == 1
    assert outcome.search_telemetry.generated_primitives == 192
    assert outcome.search_telemetry.ackermann_feasible_claimed is False
    assert provider.hopper_resource_authority is HOPPER_RESOURCE_AUTHORITY_V2
    assert provider.profile is profile.profile


def test_hopper_provider_repeated_runs_are_byte_deterministic() -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    request = _request(profile, anchor)
    first = provider.plan(request, anchor, _deadline())
    second = provider.plan(request, anchor, _deadline())
    assert type(first) is PlanningSuccessV2
    assert type(second) is PlanningSuccessV2
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_hopper_provider_enforces_risk_accelerator_deadline_and_resource_preflight() -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    cases = (
        (
            _request(
                profile,
                anchor,
                objective=ObjectiveProfileV2(
                    distance_weight=0.0,
                    risk_weight=0.1,
                    energy_weight=0.5,
                    time_weight=0.5,
                ),
            ),
            _deadline(),
            "hopper_risk_objective_unsupported",
        ),
        (
            _request(
                profile,
                anchor,
                accelerator=AcceleratorPolicyV2.REQUIRED,
            ),
            _deadline(),
            "hopper_accelerator_required_unsupported",
        ),
        (
            _request(profile, anchor, budget=ResourceBudgetV2(0, 100, 0)),
            _deadline(),
            "hopper_expansion_budget_exhausted",
        ),
        (
            _request(profile, anchor, budget=ResourceBudgetV2(10, 1, 0)),
            _deadline(),
            "hopper_route_state_budget_exceeded",
        ),
        (
            _request(profile, anchor),
            _deadline(now=101.0),
            "planning_deadline_expired",
        ),
    )
    for request, deadline, reason in cases:
        outcome = provider.plan(request, anchor, deadline)
        assert type(outcome) is PlanningFailureV2
        assert outcome.reason_code == reason
        assert outcome.search_telemetry.generated_primitives == 0


def test_hopper_provider_search_exhaustion_returns_no_partial_route() -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    request = _request(
        profile,
        anchor,
        goal=PoseStateV2(7.25, 7.25, 0.0),
        budget=ResourceBudgetV2(1, 100, 0),
    )
    outcome = provider.plan(request, anchor, _deadline())
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.NO_COMPLETE_ROUTE
    assert outcome.reason_code == "hopper_no_complete_route"
    assert outcome.evidence.stage == "hopper_search"
    assert outcome.search_telemetry.expanded_states == 1
    assert outcome.search_telemetry.generated_primitives == 192
