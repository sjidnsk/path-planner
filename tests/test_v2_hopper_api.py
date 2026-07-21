from __future__ import annotations

from dataclasses import replace
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
    ObservationProjectionV2,
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
    speed = profile.launch_speeds_mps[0]
    elevation = profile.launch_elevations_rad[0]
    flight_time = 2.0 * ((speed * sin(elevation)) / profile.gravity_mps2)
    distance = (speed * cos(elevation)) * flight_time
    request = PlanningRequestV2(
        request_id="gate5b-hopper-api-module",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=PoseStateV2(start.x_m, start.y_m + distance, 0.0),
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


def test_hopper_api_dispatches_the_exact_bound_plan_captured_by_the_seal() -> None:
    request, registry, delegate = _fixture()

    class SwitchingPlanProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority
        hopper_resource_authority = delegate.hopper_resource_authority

        def __init__(self) -> None:
            self.completed = False
            self.plan_access_count = 0
            self.unsealed_plan_called = False

        @property
        def plan(self):
            self.plan_access_count += 1
            if self.completed:
                return self._sealed_plan
            if self.plan_access_count == 3:
                return self._unsealed_plan
            return self._sealed_plan

        def _sealed_plan(self, request, anchor, deadline):
            outcome = delegate.plan(request, anchor, deadline)
            self.completed = True
            return outcome

        def _unsealed_plan(self, request, anchor, deadline):
            self.unsealed_plan_called = True
            raise AssertionError("an uncaptured plan implementation ran")

    provider = SwitchingPlanProvider()
    outcome = _call(request, registry, provider)
    assert type(outcome) is PlanningSuccessV2
    assert provider.unsealed_plan_called is False


def test_hopper_api_rejects_failure_detail_value_and_contract_stage_mismatches() -> None:
    request, registry, delegate = _fixture()

    malformed_outcomes = (
        PlanningFailureV2(
            request.request_id,
            PlatformKindV2.HOPPER,
            FailureCategoryV2.UNSUPPORTED_CAPABILITY,
            "hopper_proxy_profile_incomplete",
            FailureEvidenceV2(
                "capability_preflight",
                ("hopper_proxy_profile_incomplete",),
                (
                    ("actual", True),
                    ("expected", 7),
                    ("parameter_set_id", False),
                    ("profile_id", 11),
                ),
            ),
            SearchTelemetryV2(
                0,
                0,
                0,
                0,
                0,
                0.0,
                False,
                False,
                False,
                "hopper_proxy_profile_incomplete",
            ),
        ),
        PlanningFailureV2(
            request.request_id,
            PlatformKindV2.HOPPER,
            FailureCategoryV2.INTERNAL_ERROR,
            "route_hash_contract_mismatch",
            FailureEvidenceV2(
                "search_setup",
                ("route_hash_contract_mismatch",),
                (
                    ("actual", "route_hash_contract_mismatch"),
                    ("expected", "stable_route_hash"),
                    ("phase", "search_setup"),
                ),
            ),
            SearchTelemetryV2(
                0,
                0,
                0,
                0,
                0,
                0.0,
                False,
                False,
                False,
                "route_hash_contract_mismatch",
            ),
        ),
    )

    for malformed in malformed_outcomes:
        class MalformedProvider:
            profile = delegate.profile
            hopper_authority = delegate.hopper_authority
            hopper_resource_authority = delegate.hopper_resource_authority

            def plan(self, request, anchor, deadline):
                return malformed

        outcome = _call(request, registry, MalformedProvider())
        assert type(outcome) is PlanningFailureV2
        assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"
        assert outcome.evidence.details[1] == (
            "expected",
            "valid_hopper_provider_failure",
        )


def test_hopper_api_rejects_provider_failure_beyond_request_expansion_cap() -> None:
    request, registry, delegate = _fixture()
    attempted = request.resource_budget.max_expanded_states + 1

    class OverBudgetFailureProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority
        hopper_resource_authority = delegate.hopper_resource_authority

        def plan(self, request, anchor, deadline):
            return PlanningFailureV2(
                request.request_id,
                PlatformKindV2.HOPPER,
                FailureCategoryV2.RESOURCE_LIMIT,
                "hopper_expansion_budget_exhausted",
                FailureEvidenceV2(
                    "search_expansion",
                    ("hopper_expansion_budget_exhausted",),
                    (
                        ("attempted_expanded_states", attempted),
                        ("max_expanded_states", request.resource_budget.max_expanded_states),
                    ),
                ),
                SearchTelemetryV2(
                    attempted,
                    192 * attempted,
                    0,
                    0,
                    0,
                    0.0,
                    False,
                    False,
                    False,
                    "hopper_expansion_budget_exhausted",
                ),
            )

    outcome = _call(request, registry, OverBudgetFailureProvider())
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"


def test_hopper_api_rejects_success_with_noncanonical_observation_endpoints() -> None:
    request, registry, delegate = _fixture()

    class ForgedObservationProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority
        hopper_resource_authority = delegate.hopper_resource_authority

        def plan(self, request, anchor, deadline):
            outcome = delegate.plan(request, anchor, deadline)
            observation = replace(
                outcome.observation_projection,
                sample_states=tuple(
                    PoseStateV2(99.0, 99.0, 0.0)
                    for _state in outcome.observation_projection.sample_states
                ),
            )
            assert type(observation) is ObservationProjectionV2
            return replace(outcome, observation_projection=observation)

    outcome = _call(request, registry, ForgedObservationProvider())
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"


def test_hopper_api_rejects_success_telemetry_outside_exact_request_bounds() -> None:
    request, registry, delegate = _fixture()
    attempted = request.resource_budget.max_expanded_states + 1

    class ForgedTelemetryProvider:
        profile = delegate.profile
        hopper_authority = delegate.hopper_authority
        hopper_resource_authority = delegate.hopper_resource_authority

        def plan(self, request, anchor, deadline):
            outcome = delegate.plan(request, anchor, deadline)
            telemetry = replace(
                outcome.search_telemetry,
                expanded_states=attempted,
                generated_primitives=192 * attempted,
            )
            return replace(outcome, search_telemetry=telemetry)

    outcome = _call(request, registry, ForgedTelemetryProvider())
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "hopper_provider_outcome_contract_mismatch"
