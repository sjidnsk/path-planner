from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module
from math import cos, sin

import numpy as np

from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlatformKindV2,
    PlanningRequestV2,
    PoseStateV2,
    PrimitiveKindV2,
    ResourceBudgetV2,
    RoutePrimitiveV2,
    TypedRouteV2,
    ValidationLevelV2,
)
from path_planner.v2.hopper_authority import (
    HOPPER_GATE5B_ALGORITHM_FIXTURE_V1,
    HopperProviderAuthorityV2,
    hopper_gate5b_algorithm_fixture_v1,
)
from path_planner.v2.oracles.hopper import HopperJumpCandidateV2, validate_hopper_jump_l2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _provider_module():
    return import_module("path_planner.v2.providers.hopper")


def _validation_module():
    return import_module("path_planner.v2.hopper_route_validation")


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
            source_id="gate5b-route-l2-fixture",
            source_hash="gate5b-route-l2-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _deadline() -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 100.0, lambda: 0.0)


def _fixture(*, max_route_states: int = 10_000):
    provider = _provider_module()
    profile = hopper_gate5b_algorithm_fixture_v1()
    authority = HopperProviderAuthorityV2(
        profile,
        HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.parameter_set_id,
        "hopper-provider-authority/v1",
    )
    anchor = _anchor()
    start_pose = PoseStateV2(0.25, 0.25, 0.0)
    speed_index = 1
    elevation_index = 1
    azimuth_index = 0
    speed = profile.launch_speeds_mps[speed_index]
    elevation = profile.launch_elevations_rad[elevation_index]
    vertical_speed = speed * sin(elevation)
    flight_time = 2.0 * (vertical_speed / profile.gravity_mps2)
    distance = (speed * cos(elevation)) * flight_time
    end_pose = PoseStateV2(start_pose.x_m + distance, start_pose.y_m, 0.0)
    candidate = HopperJumpCandidateV2(
        start_pose,
        0.0,
        profile,
        authority.parameter_set_id,
        speed_index,
        elevation_index,
        azimuth_index,
        "hopper-jump-candidate/v1",
    )
    oracle = validate_hopper_jump_l2(candidate, anchor, _deadline())
    assert oracle.reason_code == "hopper_jump_l2_valid"
    start = provider.HopperSearchStateV2(
        start_pose, 0.0, "hopper-nominal-mean-search-state/v1"
    )
    end = provider.HopperSearchStateV2(
        end_pose, 0.0, "hopper-nominal-mean-search-state/v1"
    )
    energy = HOPPER_GATE5B_ALGORITHM_FIXTURE_V1.energy_evaluator(speed)
    primitive = provider.HopperJumpPrimitiveV2(
        kind=PrimitiveKindV2.BALLISTIC_JUMP,
        start_state=start_pose,
        end_state=end_pose,
        duration_s=flight_time,
        distance_m=distance,
        energy_cost=energy,
        observation_contribution=0.0,
        validation_level=ValidationLevelV2.L2,
        start_hopper_state=start,
        end_hopper_state=end,
        speed_index=speed_index,
        elevation_index=elevation_index,
        azimuth_index=azimuth_index,
        parameter_set_id=authority.parameter_set_id,
        selected_landing_mass=oracle.selected_landing_mass,
        primitive_schema_version="hopper-jump-primitive/v1",
    )
    objective = ObjectiveProfileV2()
    total = objective.energy_weight * energy + objective.time_weight * flight_time
    route = TypedRouteV2(PlatformKindV2.HOPPER, (primitive,), total, True)
    request = PlanningRequestV2(
        request_id="gate5b-route-l2",
        platform_profile_id=profile.profile.profile_id,
        start_state=start_pose,
        goal_state=end_pose,
        terrain_snapshot=anchor.snapshot,
        objective_profile=objective,
        resource_budget=ResourceBudgetV2(
            max_expanded_states=100,
            max_route_states=max_route_states,
            max_memory_bytes=0,
        ),
        timeout_s=2.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=7,
    )
    return provider, authority, anchor, request, route, primitive


def test_hopper_search_state_and_exact_key_freeze_canonical_words() -> None:
    provider = _provider_module()
    state = provider.HopperSearchStateV2(
        PoseStateV2(-0.0, -0.0, -0.0),
        -0.0,
        "hopper-nominal-mean-search-state/v1",
    )
    assert tuple(field.name for field in fields(provider.HopperSearchStateV2)) == (
        "nominal_state",
        "support_height_m",
        "schema_version",
    )
    assert provider.hopper_state_key_v2(state) == (
        0x484F505045525631,
        0,
        0,
        0,
        0,
    )
    assert not hasattr(state, "__dict__")
    try:
        state.support_height_m = 1.0
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("state must be frozen")


def test_hopper_jump_primitive_freezes_typed_jump_contract() -> None:
    provider, _authority, _anchor_value, _request, _route, primitive = _fixture()
    assert isinstance(primitive, RoutePrimitiveV2)
    assert type(primitive) is provider.HopperJumpPrimitiveV2
    assert primitive.kind is PrimitiveKindV2.BALLISTIC_JUMP
    assert primitive.validation_level is ValidationLevelV2.L2
    assert primitive.start_state == primitive.start_hopper_state.nominal_state
    assert primitive.end_state == primitive.end_hopper_state.nominal_state
    assert primitive.selected_landing_mass >= 0.99
    assert primitive.observation_contribution == 0.0
    assert not hasattr(primitive, "dt_s")


def test_hopper_probability_diagnostic_uses_per_hop_union_bound() -> None:
    module = _validation_module()
    diagnostic = module.hopper_route_probability_diagnostic_v2((0.99, 0.995))
    assert type(diagnostic) is module.HopperRouteProbabilityDiagnosticV2
    assert diagnostic.per_hop_mass == (0.99, 0.995)
    assert diagnostic.union_bound_lower_mass <= 0.985
    assert diagnostic.union_bound_lower_mass > 0.984999999999
    assert diagnostic.schema_version == "hopper-route-probability-diagnostic/v1"


def test_hopper_route_l2_replays_one_hop_and_returns_complete_evidence() -> None:
    module = _validation_module()
    _provider, authority, anchor, request, route, _primitive = _fixture()
    result = module.validate_hopper_route_l2(
        route, request, anchor, authority, _deadline()
    )
    assert type(result) is module.HopperRouteValidationResultV2
    assert result.passed is True
    assert result.reason_code == "hopper_route_l2_valid"
    assert result.stage == "route_validation"
    assert result.route_state_count == 2
    assert len(result.replay_work_counts) == 6
    assert result.cost_breakdown is not None
    assert result.probability_diagnostic is not None
    assert type(result.route_digest) is str and len(result.route_digest) == 64
    assert result.evidence.passed is True


def test_hopper_route_l2_rejects_primitive_endpoint_or_mass_drift() -> None:
    module = _validation_module()
    _provider, authority, anchor, request, route, primitive = _fixture()
    object.__setattr__(primitive, "selected_landing_mass", 0.999)
    result = module.validate_hopper_route_l2(
        route, request, anchor, authority, _deadline()
    )
    assert result.passed is False
    assert result.reason_code == "hopper_primitive_contract_mismatch"
    assert result.cost_breakdown is None
    assert result.probability_diagnostic is None
    assert result.route_digest is None


def test_hopper_route_l2_enforces_operational_h_plus_one_route_state_bound() -> None:
    module = _validation_module()
    _provider, authority, anchor, request, route, _primitive = _fixture(
        max_route_states=1
    )
    result = module.validate_hopper_route_l2(
        route, request, anchor, authority, _deadline()
    )
    assert result.passed is False
    assert result.reason_code == "hopper_route_state_budget_exceeded"
    assert result.category.value == "resource_limit"
    assert result.route_state_count == 2

