from __future__ import annotations

from math import cos, hypot, inf, nextafter, pi, sin

import numpy as np
import pytest

import path_planner.v2.hopper_authority as hopper_authority
import path_planner.v2.hopper_route_validation as hopper_route_validation
import path_planner.v2.oracles.hopper as hopper_oracle
import path_planner.v2.providers.hopper as hopper_provider
import path_planner.v2.providers.legged as legged_provider
import path_planner.v2.profiles as profiles
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningFailureV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.oracles.legged import (
    LegIdV2,
    validate_legged_step_l2,
)
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


_HOPPER_PARAMETER_SET_ID = (
    "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
)
_LEGGED_PROFILE_ID = "legged-static-crawl-midterm-g2/v2"
_LEGGED_START = PoseStateV2(15.25, 15.25, 0.0)
_HOPPER_SPEED_MPS = 1.5
_HOPPER_ELEVATION_RAD = pi / 4.0
_HOPPER_HORIZONTAL_SPEED_MPS = _HOPPER_SPEED_MPS * cos(_HOPPER_ELEVATION_RAD)
_HOPPER_VERTICAL_SPEED_MPS = _HOPPER_SPEED_MPS * sin(_HOPPER_ELEVATION_RAD)
_HOPPER_FLIGHT_TIME_S = 2.0 * (_HOPPER_VERTICAL_SPEED_MPS / 1.62)
_HOPPER_RANGE_M = _HOPPER_HORIZONTAL_SPEED_MPS * _HOPPER_FLIGHT_TIME_S
_HOPPER_AXIS_1_M = 1.25 + _HOPPER_RANGE_M
_HOPPER_AXIS_2_M = _HOPPER_AXIS_1_M + _HOPPER_RANGE_M
_HOPPER_START = PoseStateV2(1.25, _HOPPER_AXIS_2_M, 0.0)


def _deadline() -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, 1_000.0, lambda: 0.0)


def _anchor(
    elevation_m: np.ndarray | None = None,
    *,
    shape: tuple[int, int] = (60, 60),
) -> FineSafetyAnchorV2:
    if elevation_m is None:
        elevation_m = np.zeros(shape, dtype="<f8")
    else:
        elevation_m = np.asarray(elevation_m, dtype="<f8")
        shape = elevation_m.shape
    snapshot = TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=shape[1], height=shape[0]),
        elevation_m=elevation_m,
        slope_deg=np.zeros(shape, dtype="<f8"),
        traversable_mask=np.ones(shape, dtype=bool),
        hard_obstacle_mask=np.zeros(shape, dtype=bool),
        observed_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype="<f8"),
        provenance=TerrainProvenanceV2(
            source_kind="midterm-provider-repair-test-fixture/v1",
            source_id="midterm-provider-repair-terrain",
            source_hash="b" * 64,
            physical_obstacle_cells_written=False,
        ),
    )
    return FineSafetyAnchorV2(snapshot)


def _distance_only() -> ObjectiveProfileV2:
    return ObjectiveProfileV2(
        distance_weight=1.0,
        risk_weight=0.0,
        energy_weight=0.0,
        time_weight=0.0,
    )


def _legged_profile() -> profiles.LeggedProfileV2:
    return profiles.LeggedProfileV2(
        profile=profiles.PlatformProfileV2(
            profile_id=_LEGGED_PROFILE_ID,
            platform_kind=PlatformKindV2.LEGGED,
            capability_revision=profiles.LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2,
            simulation_proxy=True,
            max_traversable_slope_deg=30.0,
            goal_position_tolerance_m=0.0,
            goal_heading_tolerance_rad=0.0,
        )
    )


def _legged_request(
    anchor: FineSafetyAnchorV2,
    *,
    max_route_states: int = 5,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="midterm-legged-one-graph-hop",
        platform_profile_id=_LEGGED_PROFILE_ID,
        start_state=_LEGGED_START,
        goal_state=PoseStateV2(15.50, 15.25, 0.0),
        terrain_snapshot=anchor.snapshot,
        objective_profile=_distance_only(),
        resource_budget=ResourceBudgetV2(
            max_expanded_states=10_000,
            max_route_states=max_route_states,
            max_memory_bytes=0,
        ),
        timeout_s=1_000.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=11,
    )


def _hopper_profile():
    return hopper_authority.hopper_generic_internal_simulation_proxy_midterm_v1()


def _hopper_provider() -> hopper_provider.HopperPrimitiveProviderV2:
    return hopper_provider.HopperPrimitiveProviderV2(
        hopper_authority.HopperProviderAuthorityV2(
            hopper_profile=_hopper_profile(),
            parameter_set_id=_HOPPER_PARAMETER_SET_ID,
            authority_schema_version="hopper-provider-authority/v1",
        )
    )


def _hopper_r1_goal() -> tuple[PoseStateV2, float, float]:
    profile = _hopper_profile()
    speed = profile.launch_speeds_mps[0]
    elevation = profile.launch_elevations_rad[1]
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    flight_time = 2.0 * (vertical_speed / profile.gravity_mps2)
    horizontal_range = (horizontal_speed * 1.0) * flight_time
    return (
        PoseStateV2(
            _HOPPER_START.x_m + horizontal_range,
            _HOPPER_START.y_m,
            _HOPPER_START.heading_rad,
        ),
        flight_time,
        horizontal_range,
    )


def _hopper_request(
    anchor: FineSafetyAnchorV2,
    goal: PoseStateV2,
    *,
    start: PoseStateV2 = _HOPPER_START,
    objective: ObjectiveProfileV2 | None = None,
    max_expanded_states: int = 1,
) -> PlanningRequestV2:
    profile = _hopper_profile()
    return PlanningRequestV2(
        request_id="midterm-hopper-r1-exact-east-hop",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=anchor.snapshot,
        objective_profile=_distance_only() if objective is None else objective,
        resource_budget=ResourceBudgetV2(
            max_expanded_states=max_expanded_states,
            max_route_states=2,
            max_memory_bytes=536_870_912,
        ),
        timeout_s=1_000.0,
        accelerator_policy=AcceleratorPolicyV2.DISABLED,
        determinism_seed=17,
    )


def _hopper_candidate() -> hopper_oracle.HopperJumpCandidateV2:
    return hopper_oracle.HopperJumpCandidateV2(
        start_state=_HOPPER_START,
        support_height_m=0.0,
        hopper_profile=_hopper_profile(),
        parameter_set_id=_HOPPER_PARAMETER_SET_ID,
        speed_index=0,
        elevation_index=1,
        azimuth_index=0,
    )


def test_legged_phase1_four_step_witness_passes_every_real_step_l2():
    anchor = _anchor()
    request = _legged_request(anchor)
    state = legged_provider._provider_initial_state_v2(request)
    expected = (
        (LegIdV2.REAR_RIGHT, (0.50, 0.0)),
        (LegIdV2.FRONT_RIGHT, (0.25, 0.0)),
        (LegIdV2.REAR_LEFT, (0.50, 0.0)),
        (LegIdV2.FRONT_LEFT, (-0.25, 0.0)),
    )

    assert (
        legged_provider.LEGGED_PROVIDER_INITIAL_CRAWL_PHASE_V2,
        state.sequence_phase,
    ) == (1, 1)
    for expected_leg, offset in expected:
        primitive, candidate = legged_provider._provider_candidate_v2(
            state,
            state.body_state.heading_rad,
            offset,
        )
        result = validate_legged_step_l2(
            candidate,
            anchor,
            _legged_profile(),
            _deadline(),
        )
        assert primitive.moving_leg is expected_leg
        assert result.reason_code == "legged_step_l2_valid"
        assert result.minimum_support_margin_m is not None
        assert result.minimum_support_margin_m >= 0.05
        state = primitive.end_legged_state

    assert state.body_state == PoseStateV2(15.50, 15.25, 0.0)


def test_legged_provider_returns_complete_four_phase_route_from_rotated_initial_phase():
    anchor = _anchor()
    result = legged_provider.LeggedPrimitiveProviderV2(_legged_profile()).plan(
        _legged_request(anchor),
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningSuccessV2
    assert len(result.route.primitives) == 4
    assert tuple(primitive.moving_leg for primitive in result.route.primitives) == (
        LegIdV2.REAR_RIGHT,
        LegIdV2.FRONT_RIGHT,
        LegIdV2.REAR_LEFT,
        LegIdV2.FRONT_LEFT,
    )
    assert result.route.primitives[-1].end_state == PoseStateV2(15.50, 15.25, 0.0)
    assert result.validation_evidence.checks == ("legged_route_l2_valid",)


def test_legged_route_budget_four_steps_requires_exactly_five_states():
    anchor = _anchor()
    provider = legged_provider.LeggedPrimitiveProviderV2(_legged_profile())

    success = provider.plan(_legged_request(anchor, max_route_states=5), anchor, _deadline())
    blocked = provider.plan(_legged_request(anchor, max_route_states=4), anchor, _deadline())

    assert type(success) is PlanningSuccessV2
    assert type(blocked) is PlanningFailureV2
    assert blocked.reason_code == "legged_route_state_budget_exhausted"


def test_legged_provider_capability_identity_changes_with_initial_phase():
    assert profiles.LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2 == (
        "simulation_proxy_static_crawl/v2"
    )
    with pytest.raises(ValueError, match="capability_revision"):
        profiles.LeggedProfileV2(
            profile=profiles.PlatformProfileV2(
                profile_id="legged-static-crawl-midterm-g2/v1",
                platform_kind=PlatformKindV2.LEGGED,
                capability_revision="simulation_proxy_static_crawl/v1",
                simulation_proxy=True,
                max_traversable_slope_deg=30.0,
                goal_position_tolerance_m=0.0,
                goal_heading_tolerance_rad=0.0,
            )
        )


def test_hopper_r1_same_support_45deg_range_uses_two_vz_over_g():
    goal, flight_time, horizontal_range = _hopper_r1_goal()

    assert flight_time.hex() == "0x1.4f3892f7f4a55p+0"
    assert horizontal_range.hex() == "0x1.638e38e38e38fp+0"
    assert goal.x_m.hex() == (_HOPPER_START.x_m + horizontal_range).hex()
    assert goal.y_m.hex() == _HOPPER_START.y_m.hex()


def test_hopper_r1_exact_goal_succeeds_with_one_expansion_and_real_l2():
    anchor = _anchor()
    goal, _, horizontal_range = _hopper_r1_goal()

    result = _hopper_provider().plan(
        _hopper_request(anchor, goal),
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningSuccessV2
    assert result.search_telemetry.expanded_states == 1
    assert result.search_telemetry.generated_primitives == 192
    assert len(result.route.primitives) == 1
    assert result.route.primitives[0].end_state == goal
    assert result.route.primitives[0].distance_m.hex() == horizontal_range.hex()
    endpoint_distance = hypot(
        goal.x_m - _HOPPER_START.x_m,
        goal.y_m - _HOPPER_START.y_m,
    )
    assert result.route.total_cost.hex() == horizontal_range.hex()
    assert result.route.total_cost.hex() == "0x1.638e38e38e38fp+0"
    assert endpoint_distance.hex() == "0x1.638e38e38e390p+0"
    assert nextafter(result.route.total_cost, inf).hex() == endpoint_distance.hex()
    assert nextafter(endpoint_distance, -inf).hex() == result.route.total_cost.hex()
    assert result.validation_evidence.checks == ("hopper_route_l2_valid",)
    replay = hopper_route_validation.validate_hopper_route_l2(
        result.route,
        _hopper_request(anchor, goal),
        anchor,
        _hopper_provider().hopper_authority,
        _deadline(),
    )
    assert replay.passed is True
    assert replay.reason_code == "hopper_route_l2_valid"


def test_hopper_r1_goal_one_ulp_drift_cannot_enter_direct_fast_path():
    anchor = _anchor()
    goal, _, _ = _hopper_r1_goal()
    drifted = PoseStateV2(nextafter(goal.x_m, inf), goal.y_m, goal.heading_rad)

    result = _hopper_provider().plan(
        _hopper_request(anchor, drifted),
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.search_telemetry.termination_reason != "hopper_route_l2_valid"


def test_hopper_non_r1_start_with_fifteen_ulp_delta_falls_back_to_original_search():
    anchor = _anchor(shape=(80, 80))
    profile = _hopper_profile()
    speed = profile.launch_speeds_mps[0]
    elevation = profile.launch_elevations_rad[1]
    horizontal_speed = speed * cos(elevation)
    vertical_speed = speed * sin(elevation)
    flight_time = 2.0 * (vertical_speed / profile.gravity_mps2)
    modeled_range = horizontal_speed * flight_time
    start = PoseStateV2(30.75, 15.25, 0.0)
    goal = PoseStateV2(
        start.x_m + modeled_range,
        start.y_m,
        start.heading_rad,
    )
    endpoint_delta = goal.x_m - start.x_m

    result = _hopper_provider().plan(
        _hopper_request(anchor, goal, start=start),
        anchor,
        _deadline(),
    )

    assert modeled_range.hex() == "0x1.638e38e38e38fp+0"
    assert endpoint_delta.hex() == "0x1.638e38e38e380p+0"
    advanced = endpoint_delta
    for _ in range(15):
        advanced = nextafter(advanced, inf)
    assert advanced.hex() == modeled_range.hex()
    assert nextafter(endpoint_delta, inf).hex() != modeled_range.hex()
    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_expansion_budget_exhausted"
    assert result.search_telemetry.expanded_states == 1
    assert result.search_telemetry.generated_primitives == 192


def test_hopper_direct_goal_non_distance_objective_falls_back():
    anchor = _anchor()
    goal, _, _ = _hopper_r1_goal()
    objective = ObjectiveProfileV2(
        distance_weight=1.0,
        risk_weight=0.0,
        energy_weight=1.0,
        time_weight=0.0,
    )

    result = _hopper_provider().plan(
        _hopper_request(anchor, goal, objective=objective),
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.search_telemetry.expanded_states == 1


def test_hopper_direct_goal_l2_rejection_falls_back_to_original_search():
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 6] = 0.051
    anchor = _anchor(elevation)
    goal, _, _ = _hopper_r1_goal()

    result = _hopper_provider().plan(
        _hopper_request(anchor, goal),
        anchor,
        _deadline(),
    )

    assert type(result) is PlanningFailureV2
    assert result.reason_code == "hopper_expansion_budget_exhausted"
    assert result.search_telemetry.expanded_states == 1
    assert result.search_telemetry.generated_primitives == 192


@pytest.mark.parametrize("residual_m", (0.05, -0.05))
def test_hopper_launch_full_support_envelope_accepts_50mm(residual_m: float):
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 2] = residual_m

    result = hopper_oracle.validate_hopper_jump_l2(
        _hopper_candidate(),
        _anchor(elevation),
        _deadline(),
    )

    assert result.reason_code == "hopper_jump_l2_valid"


@pytest.mark.parametrize("residual_m", (0.051, -0.051))
def test_hopper_launch_full_support_envelope_rejects_51mm(residual_m: float):
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 2] = residual_m

    result = hopper_oracle.validate_hopper_jump_l2(
        _hopper_candidate(),
        _anchor(elevation),
        _deadline(),
    )

    assert result.reason_code == "hopper_launch_unsafe"


@pytest.mark.parametrize("residual_m", (0.05, -0.05))
def test_hopper_landing_full_support_envelope_accepts_50mm(residual_m: float):
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 6] = residual_m

    result = hopper_oracle.validate_hopper_jump_l2(
        _hopper_candidate(),
        _anchor(elevation),
        _deadline(),
    )

    assert result.reason_code == "hopper_jump_l2_valid"


@pytest.mark.parametrize("residual_m", (0.051, -0.051))
def test_hopper_landing_full_support_envelope_rejects_51mm(residual_m: float):
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 6] = residual_m

    result = hopper_oracle.validate_hopper_jump_l2(
        _hopper_candidate(),
        _anchor(elevation),
        _deadline(),
    )

    assert result.reason_code == "hopper_landing_height_unreachable"


def test_hopper_support_residual_does_not_relax_arc_clearance():
    elevation = np.zeros((60, 60), dtype="<f8")
    elevation[8, 4] = 0.80

    result = hopper_oracle.validate_hopper_jump_l2(
        _hopper_candidate(),
        _anchor(elevation),
        _deadline(),
    )

    assert result.reason_code == "hopper_arc_clearance_violation"


def test_hopper_support_contract_bumps_generic_capability_identity():
    record = hopper_authority.HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_IMPLEMENTATION_V1

    assert profiles.HOPPER_GENERIC_INTERNAL_SIMULATION_PROXY_CAPABILITY_REVISION_V2 == (
        "simulation_proxy_generic_internal_lunar_ballistic/v3"
    )
    assert profiles.HOPPER_SUPPORT_PLANE_MODEL_ID_V2 == (
        "hopper_horizontal_same_support_full_envelope_50mm/v1"
    )
    assert profiles.HOPPER_SUPPORT_HEIGHT_TOLERANCE_M_V2.hex() == 0.05.hex()
    assert record.capability_revision.endswith("/v3")
    assert record.support_plane_model_id == profiles.HOPPER_SUPPORT_PLANE_MODEL_ID_V2
    assert (
        record.support_height_tolerance_m.hex()
        == profiles.HOPPER_SUPPORT_HEIGHT_TOLERANCE_M_V2.hex()
    )
    assert record.relief_preservation_required is True
    assert hopper_provider.HOPPER_PROVIDER_ROOT_SUPPORT_HEIGHT_M_V2.hex() == 0.0.hex()
