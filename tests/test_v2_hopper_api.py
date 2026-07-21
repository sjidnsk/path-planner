from __future__ import annotations

from math import cos, sin

import numpy as np

import path_planner.v2 as v2
import path_planner.v2.hopper_api as hopper_api
from path_planner.v2.api import plan_v2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    FailureCategoryV2,
    FailureEvidenceV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
    SearchTelemetryV2,
)
from path_planner.v2.hopper_authority import (
    HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
    HopperProviderAuthorityV2,
    hopper_gate5b_algorithm_fixture_v1,
)
from path_planner.v2.profiles import PlatformProfileRegistryV2
from path_planner.v2.providers.hopper import HopperPrimitiveProviderV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _fixture():
    profile = hopper_gate5b_algorithm_fixture_v1()
    authority = HopperProviderAuthorityV2(
        profile,
        HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.parameter_set_id,
        "hopper-provider-authority/v1",
    )
    provider = HopperPrimitiveProviderV2(authority)
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
            source_id="gate5b-hopper-api-fixture",
            source_hash="gate5b-hopper-api-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    start = PoseStateV2(0.25, 0.25, 0.0)
    speed = profile.launch_speeds_mps[1]
    elevation = profile.launch_elevations_rad[1]
    flight_time = 2.0 * ((speed * sin(elevation)) / profile.gravity_mps2)
    distance = (speed * cos(elevation)) * flight_time
    request = PlanningRequestV2(
        request_id="gate5b-hopper-api-module",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=PoseStateV2(start.x_m + distance, start.y_m, 0.0),
        terrain_snapshot=snapshot,
        objective_profile=ObjectiveProfileV2(),
        resource_budget=ResourceBudgetV2(10, 100, 0),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=29,
    )
    registry = PlatformProfileRegistryV2((profile.profile,))
    return request, registry, provider


def _call(request, registry, provider):
    return plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: provider},
        monotonic_clock=lambda: 0.0,
    )


def test_hopper_api_module_is_internal_and_success_dispatches() -> None:
    assert hopper_api.__all__ == ()
    assert "dispatch_hopper_provider_v2" not in v2.__all__
    assert not hasattr(v2, "dispatch_hopper_provider_v2")
    request, registry, provider = _fixture()
    assert type(_call(request, registry, provider)) is PlanningSuccessV2


def test_hopper_api_rejects_malformed_provider_failure_structurally() -> None:
    request, registry, delegate = _fixture()

    class MalformedFailureProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority
        hopper_resource_authority = delegate.hopper_resource_authority

        def plan(self, request, anchor, deadline):
            return PlanningFailureV2(
                request_id=request.request_id,
                platform_kind=PlatformKindV2.HOPPER,
                category=FailureCategoryV2.NO_COMPLETE_ROUTE,
                reason_code="hopper_no_complete_route",
                evidence=FailureEvidenceV2(
                    "hopper_search",
                    ("hopper_no_complete_route",),
                    (),
                ),
                search_telemetry=SearchTelemetryV2(
                    1,
                    192,
                    0,
                    0,
                    0,
                    0.0,
                    False,
                    False,
                    False,
                    "hopper_no_complete_route",
                ),
            )

    outcome = _call(request, registry, MalformedFailureProvider())
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is FailureCategoryV2.INTERNAL_ERROR
    assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"
    assert outcome.evidence.stage == "provider_postcondition"
    assert outcome.evidence.details == (
        ("actual", "hopper_no_complete_route"),
        ("expected", "valid_hopper_provider_failure"),
        ("phase", "provider_postcondition"),
    )
    assert outcome.search_telemetry.expanded_states == 0
    assert outcome.search_telemetry.generated_primitives == 0


def test_hopper_api_requires_composite_resource_authority() -> None:
    request, registry, delegate = _fixture()

    class MissingResourceAuthorityProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority

        def plan(self, request, anchor, deadline):
            raise AssertionError("provider must not run")

    outcome = _call(request, registry, MissingResourceAuthorityProvider())
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "hopper_authority_contract_mismatch"
    assert outcome.evidence.stage == "provider_authority"
