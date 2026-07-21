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


def _goal_after_hops(
    profile: HopperProfileV2,
    hop_count: int,
    *,
    speed_index: int = 1,
    elevation_index: int = 1,
    azimuth_index: int = 0,
) -> PoseStateV2:
    if azimuth_index != 0:
        raise AssertionError("the focused fixture helper only derives east hops")
    speed = profile.launch_speeds_mps[speed_index]
    elevation = profile.launch_elevations_rad[elevation_index]
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    flight_time = 2.0 * (vertical_speed / profile.gravity_mps2)
    dx = (horizontal_speed * 1.0) * flight_time
    state = PoseStateV2(0.25, 0.25, 0.0)
    for _ in range(hop_count):
        state = PoseStateV2(state.x_m + dx, state.y_m, state.heading_rad)
    return state


def _install_selective_oracle(monkeypatch, allowed_actions):
    module = _module()
    oracle_module = import_module("path_planner.v2.oracles.hopper")
    real_oracle = module.validate_hopper_jump_l2
    allowed = frozenset(allowed_actions)
    seen: list[tuple[int, int, int]] = []

    def selective(candidate, anchor, deadline):
        action = (
            candidate.speed_index,
            candidate.elevation_index,
            candidate.azimuth_index,
        )
        seen.append(action)
        if action in allowed:
            return real_oracle(candidate, anchor, deadline)
        return oracle_module._result_v2(
            "hopper_stop_condition_failed",
            "stop_validation",
            [0, 0, 0, 0, 0, 0],
        )

    monkeypatch.setattr(module, "validate_hopper_jump_l2", selective)
    return seen


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


def test_hopper_provider_returns_one_hop_fixture_success_envelope(monkeypatch) -> None:
    module = _module()
    provider, profile = _fixture_provider()
    anchor = _anchor()
    _install_selective_oracle(monkeypatch, ((1, 1, 0),))
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


def test_hopper_provider_repeated_runs_are_byte_deterministic(monkeypatch) -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    _install_selective_oracle(monkeypatch, ((1, 1, 0),))
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


def test_hopper_provider_search_exhaustion_returns_no_partial_route(monkeypatch) -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    _install_selective_oracle(monkeypatch, ())
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


def test_hopper_provider_runs_fixed_order_two_hop_dijkstra_and_endpoint_envelope(
    monkeypatch,
) -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    seen = _install_selective_oracle(monkeypatch, ((1, 1, 0),))
    request = _request(
        profile,
        anchor,
        goal=_goal_after_hops(profile, 2),
        budget=ResourceBudgetV2(5, 3, 0),
    )

    outcome = provider.plan(request, anchor, _deadline())

    assert type(outcome) is PlanningSuccessV2
    assert len(outcome.route.primitives) == 2
    assert outcome.route.primitives[0].end_state == outcome.route.primitives[1].start_state
    assert outcome.route.primitives[-1].end_state == request.goal_state
    assert outcome.observation_projection.sample_states == (
        outcome.route.primitives[0].start_state,
        outcome.route.primitives[0].end_state,
        outcome.route.primitives[1].end_state,
    )
    assert outcome.search_telemetry.expanded_states == 2
    assert outcome.search_telemetry.generated_primitives == 384
    assert outcome.search_telemetry.rejected_l2 == 382
    expected_order = tuple(
        (speed_index, elevation_index, azimuth_index)
        for speed_index in range(4)
        for elevation_index in range(3)
        for azimuth_index in range(16)
    )
    assert tuple(seen) == expected_order + expected_order


def test_hopper_provider_enforces_expansion_and_h_plus_one_route_state_caps(
    monkeypatch,
) -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    _install_selective_oracle(monkeypatch, ((1, 1, 0),))
    goal = _goal_after_hops(profile, 2)

    expansion = provider.plan(
        _request(
            profile,
            anchor,
            goal=goal,
            budget=ResourceBudgetV2(1, 3, 0),
        ),
        anchor,
        _deadline(),
    )
    assert type(expansion) is PlanningFailureV2
    assert expansion.category is FailureCategoryV2.RESOURCE_LIMIT
    assert expansion.reason_code == "hopper_expansion_budget_exhausted"
    assert expansion.evidence.details == (
        ("attempted_expanded_states", 2),
        ("max_expanded_states", 1),
    )
    assert expansion.search_telemetry.expanded_states == 1
    assert expansion.search_telemetry.generated_primitives == 192

    route_states = provider.plan(
        _request(
            profile,
            anchor,
            goal=goal,
            budget=ResourceBudgetV2(5, 2, 0),
        ),
        anchor,
        _deadline(),
    )
    assert type(route_states) is PlanningFailureV2
    assert route_states.category is FailureCategoryV2.RESOURCE_LIMIT
    assert route_states.reason_code == "hopper_route_state_budget_exceeded"
    assert route_states.evidence.details == (
        ("attempted_route_states", 3),
        ("effective_max_route_states", 2),
        ("requested_max_route_states", 2),
    )
    assert route_states.search_telemetry.expanded_states == 2
    # The admitted fixture action is index 64 in the second fixed 192-action pass.
    assert route_states.search_telemetry.generated_primitives == 257


def test_hopper_provider_enforces_accounted_memory_root_and_transient_phases(
    monkeypatch,
) -> None:
    provider, profile = _fixture_provider()
    anchor = _anchor()
    _install_selective_oracle(monkeypatch, ((1, 1, 0),))

    root = provider.plan(
        _request(
            profile,
            anchor,
            budget=ResourceBudgetV2(10, 100, 5_119),
        ),
        anchor,
        _deadline(),
    )
    assert type(root) is PlanningFailureV2
    assert root.category is FailureCategoryV2.RESOURCE_LIMIT
    assert root.reason_code == "hopper_memory_budget_exceeded"
    assert dict(root.evidence.details)["phase"] == "root_admission"
    assert dict(root.evidence.details)["attempted_accounted_bytes"] == 5_120
    assert root.search_telemetry.expanded_states == 0
    assert root.search_telemetry.generated_primitives == 0

    transient = provider.plan(
        _request(
            profile,
            anchor,
            budget=ResourceBudgetV2(10, 100, 10_000_000),
        ),
        anchor,
        _deadline(),
    )
    assert type(transient) is PlanningFailureV2
    assert transient.category is FailureCategoryV2.RESOURCE_LIMIT
    assert transient.reason_code == "hopper_memory_budget_exceeded"
    assert dict(transient.evidence.details)["phase"] == "arc_oracle"
    assert transient.search_telemetry.expanded_states == 1
    assert transient.search_telemetry.generated_primitives == 0
