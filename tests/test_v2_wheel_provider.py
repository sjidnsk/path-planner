from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, pi, tau

import numpy as np
import pytest

import path_planner
import path_planner.v2 as v2
import path_planner.v2.providers as provider_exports
import path_planner.v2.providers.wheel as wheel_module
import path_planner.v2.validation as validation_module
from path_planner.core import Cell, FailureReason, WorldPoint
from path_planner.platform import DEFAULT_PLATFORM_KEY
from path_planner.search import (
    MotionPrimitive,
    Pose2D,
    PoseCostBreakdown,
    PosePathDiagnostics,
    PosePlanResult,
    PoseSearchAudit,
    replay_motion_primitive,
)
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
    ValidationLevelV2,
)
from path_planner.v2.geometry import conservative_wheel_pose_cells
from path_planner.v2.profiles import PlatformProfileV2, WheelProfileV2
from path_planner.v2.providers import PrimitiveProviderV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


def _snapshot(
    *,
    width: int = 20,
    height: int = 20,
    hard_cells: tuple[Cell, ...] = (),
    unknown_cells: tuple[Cell, ...] = (),
) -> TerrainSnapshotV2:
    shape = (height, width)
    traversable = np.ones(shape, dtype=bool)
    hard = np.zeros(shape, dtype=bool)
    observed = np.ones(shape, dtype=bool)
    for cell in hard_cells:
        hard[cell.y, cell.x] = True
        traversable[cell.y, cell.x] = False
    for cell in unknown_cells:
        observed[cell.y, cell.x] = False
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(
            width=width,
            height=height,
            frame_id="moon",
        ),
        elevation_m=np.zeros(shape, dtype=np.float64),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="wheel-provider-fixture",
            source_hash="wheel-provider-fixture-hash",
            physical_obstacle_cells_written=False,
        ),
    )


def _wheel_profile(
    *,
    position_tolerance_m: float = 0.0,
    heading_tolerance_rad: float = 0.0,
    **overrides,
) -> WheelProfileV2:
    generic = PlatformProfileV2(
        profile_id="wheel-provider/v1",
        platform_kind=PlatformKindV2.WHEEL,
        capability_revision="wheel-provider-capability/v1",
        simulation_proxy=False,
        max_traversable_slope_deg=30.0,
        goal_position_tolerance_m=position_tolerance_m,
        goal_heading_tolerance_rad=heading_tolerance_rad,
    )
    return WheelProfileV2(profile=generic, **overrides)


def _request(
    profile: WheelProfileV2,
    snapshot: TerrainSnapshotV2,
    start: PoseStateV2,
    goal: PoseStateV2,
    *,
    objective: ObjectiveProfileV2 | None = None,
    budget: ResourceBudgetV2 | None = None,
    accelerator_policy: AcceleratorPolicyV2 = AcceleratorPolicyV2.DISABLED,
) -> PlanningRequestV2:
    return PlanningRequestV2(
        request_id="wheel-provider-request",
        platform_profile_id=profile.profile.profile_id,
        start_state=start,
        goal_state=goal,
        terrain_snapshot=snapshot,
        objective_profile=objective or ObjectiveProfileV2(),
        resource_budget=budget or ResourceBudgetV2(),
        timeout_s=2.0,
        accelerator_policy=accelerator_policy,
        determinism_seed=17,
    )


def _deadline(*, now: float = 0.0, end: float = 100.0) -> PlanningDeadlineV2:
    return PlanningDeadlineV2(0.0, end, lambda: now)


def _provider(profile: WheelProfileV2):
    return wheel_module.WheelPrimitiveProviderV2(profile)


def _hybrid_failure(
    reason: FailureReason,
    *,
    expanded: int = 0,
    audit: PoseSearchAudit | None = None,
) -> PosePlanResult:
    return PosePlanResult(
        success=False,
        pose_path=(),
        control_sequence=(),
        legacy_cell_path=(),
        total_cost=inf,
        cost_breakdown=PoseCostBreakdown(),
        expanded_count=expanded,
        failure_reason=reason,
        diagnostics=PosePathDiagnostics(ackermann_feasible_claimed=True),
        audit=audit
        or PoseSearchAudit(
            timed_out=reason is FailureReason.TIMEOUT,
            termination_reason=reason.value,
        ),
    )


def _hybrid_success(
    controls: tuple[MotionPrimitive, ...],
    *,
    expanded: int = 7,
    audit: PoseSearchAudit | None = None,
) -> PosePlanResult:
    return PosePlanResult(
        success=True,
        pose_path=(Pose2D(-999.0, -999.0, pi),),
        control_sequence=controls,
        legacy_cell_path=(),
        total_cost=999_999.0,
        cost_breakdown=PoseCostBreakdown(
            translation_cost=111.0,
            rotation_cost=222.0,
            reverse_penalty=333.0,
            turn_penalty=444.0,
        ),
        expanded_count=expanded,
        failure_reason=None,
        diagnostics=PosePathDiagnostics(
            runtime_ms=1234.0,
            ackermann_feasible_claimed=True,
        ),
        audit=audit or PoseSearchAudit(termination_reason="success"),
    )


def _install_hybrid(
    monkeypatch: pytest.MonkeyPatch,
    callback,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    class PublicHybrid:
        def __init__(self, primitives=None) -> None:
            captured["primitives"] = tuple(primitives or ())

        def plan(self, grid, request, **kwargs):
            captured.update(grid=grid, request=request, **kwargs)
            return callback(grid, request, **kwargs)

    monkeypatch.setattr(wheel_module, "HybridAStarPlanner", PublicHybrid, raising=False)
    return captured


def _assert_failure(
    outcome,
    category: FailureCategoryV2,
    reason: str,
    stage: str,
) -> PlanningFailureV2:
    assert type(outcome) is PlanningFailureV2
    assert outcome.category is category
    assert outcome.reason_code == reason
    assert outcome.evidence.stage == stage
    assert not hasattr(outcome, "route")
    return outcome


class _ForbiddenTruthLayer:
    def __array__(self, *_args, **_kwargs):
        raise AssertionError("Hybrid grid read terrain truth")

    def __getitem__(self, _key):
        raise AssertionError("Hybrid grid read terrain truth")


def test_provider_profile_identity_protocol_exports_and_opt_in_boundary() -> None:
    wheel_profile = _wheel_profile()
    provider = _provider(wheel_profile)

    assert v2.WheelPrimitiveProviderV2 is wheel_module.WheelPrimitiveProviderV2
    assert provider_exports.WheelPrimitiveProviderV2 is wheel_module.WheelPrimitiveProviderV2
    assert provider.wheel_profile is wheel_profile
    assert type(provider.wheel_profile) is WheelProfileV2
    assert provider.profile is wheel_profile.profile
    assert type(provider.profile) is PlatformProfileV2
    assert isinstance(provider, PrimitiveProviderV2)
    assert not hasattr(provider, "__dict__")
    with pytest.raises(FrozenInstanceError):
        provider.wheel_profile = _wheel_profile()
    with pytest.raises(TypeError, match="WheelProfileV2"):
        wheel_module.WheelPrimitiveProviderV2(object())
    assert DEFAULT_PLATFORM_KEY == "yutu2"
    assert "plan_v2" not in path_planner.__dict__


def test_hybrid_grid_copies_only_geometry_and_uses_public_deadline_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(unknown_cells=(Cell(0, 0),))
    profile = _wheel_profile()
    start = PoseStateV2(3.25, 3.25, 0.0)
    request = _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0))
    anchor = FineSafetyAnchorV2(snapshot)
    for name in (
        "elevation_m",
        "slope_deg",
        "traversable_mask",
        "hard_obstacle_mask",
        "observed_mask",
        "confidence",
    ):
        object.__setattr__(snapshot, name, _ForbiddenTruthLayer())

    captured = _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_failure(FailureReason.UNREACHABLE),
    )
    deadline = _deadline()
    outcome = _provider(profile).plan(request, anchor, deadline)

    _assert_failure(
        outcome,
        FailureCategoryV2.GOAL_POSE_UNREACHABLE,
        "wheel_goal_pose_unreachable",
        "hybrid_search",
    )
    grid = captured["grid"]
    geometry = snapshot.geometry
    assert grid.spec.width == geometry.width
    assert grid.spec.height == geometry.height
    assert grid.spec.origin == geometry.origin
    assert grid.spec.resolution == geometry.resolution_m
    assert grid.spec.frame_id == geometry.frame_id
    assert np.array_equal(grid.cost, np.zeros(geometry.shape))
    assert np.array_equal(grid.passable_mask, np.ones(geometry.shape, dtype=bool))
    assert captured["deadline_monotonic_s"] == deadline.deadline_monotonic_s
    assert callable(captured["monotonic_clock"])
    assert captured["monotonic_clock"]() < deadline.deadline_monotonic_s
    assert captured["pose_validator"](Pose2D(3.25, 3.25, 0.0)) is True


def test_endpoint_safe_candidate_with_blocked_intermediate_sweep_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(
        primitive_duration_s=2.0,
        integration_dt_s=0.25,
        reverse_enabled=False,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
    )
    start_pose = Pose2D(3.25, 3.25, 0.0)
    control = MotionPrimitive("forward", 1.0, 0.0, 2.0)
    replay = replay_motion_primitive(start_pose, control, profile.integration_dt_s)
    snapshot = _snapshot(hard_cells=(Cell(8, 6),))
    request = _request(
        profile,
        snapshot,
        PoseStateV2(start_pose.x_m, start_pose.y_m, start_pose.theta_rad),
        PoseStateV2(replay.end.x_m, replay.end.y_m, replay.end.theta_rad),
    )
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_success((control,)),
    )

    outcome = _provider(profile).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.VALIDATION_FAILED,
        "terrain_hard_obstacle",
        "route_validation",
    )
    assert failure.search_telemetry.ackermann_feasible_claimed is False


def test_first_l2_rejected_edge_allows_alternate_success_and_direct_audit_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(
        primitive_duration_s=1.0,
        integration_dt_s=0.25,
        reverse_enabled=True,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
    )
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(2.25, 3.25, 0.0)
    snapshot = _snapshot(hard_cells=(Cell(8, 6),))
    request = _request(profile, snapshot, start, goal)
    captured: dict[str, object]

    def search(_grid, pose_request, **kwargs):
        primitives = captured["primitives"]
        forward = next(item for item in primitives if item.name == "forward")
        reverse = next(item for item in primitives if item.name == "reverse")
        rejected = replay_motion_primitive(
            pose_request.start,
            forward,
            pose_request.integration_dt_s,
        )
        accepted = replay_motion_primitive(
            pose_request.start,
            reverse,
            pose_request.integration_dt_s,
        )
        assert kwargs["transition_validator"](rejected) is False
        assert kwargs["transition_validator"](accepted) is True
        return _hybrid_success(
            (reverse,),
            expanded=7,
            audit=PoseSearchAudit(
                generated_primitives=9,
                rejected_poses=2,
                rejected_transitions=1,
                termination_reason="success",
            ),
        )

    captured = _install_hybrid(monkeypatch, search)
    outcome = _provider(profile).plan(
        request,
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    assert len(outcome.route.primitives) == 1
    primitive = outcome.route.primitives[0]
    assert primitive.control_name == "reverse"
    assert primitive.reverse is True
    assert primitive.validation_level is ValidationLevelV2.L2
    assert primitive.start_state == start
    assert primitive.end_state == goal
    telemetry = outcome.search_telemetry
    assert telemetry.expanded_states == 7
    assert telemetry.generated_primitives == 9
    assert telemetry.rejected_l0 == 2
    assert telemetry.rejected_l1 == 0
    assert telemetry.rejected_l2 == 1
    assert telemetry.timed_out is False
    assert telemetry.termination_reason == "success"
    assert telemetry.accelerator_used is False
    assert telemetry.ackermann_feasible_claimed is False
    observation = outcome.observation_projection
    assert observation.source == "wheel_route_samples_gain_not_computed/v1"
    assert observation.sample_states == primitive.samples
    assert observation.expected_new_observed_cells == 0.0
    assert observation.expected_information_gain == 0.0
    assert all(item.observation_contribution == 0.0 for item in outcome.route.primitives)
    assert outcome.cache_evidence.cache_namespace == (
        "path-planner-v2-wheel-cache-disabled/v1"
    )
    assert outcome.cache_evidence.cache_key == "wheel-cache-disabled/v1"
    assert outcome.cache_evidence.hit is False


def test_public_scout_controls_are_filtered_by_wheel_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(
        reverse_enabled=False,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_speed_mps=1.0,
        max_angular_speed_radps=1.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    captured = _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_failure(FailureReason.UNREACHABLE),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.GOAL_POSE_UNREACHABLE,
        "wheel_goal_pose_unreachable",
        "hybrid_search",
    )
    controls = captured["primitives"]
    assert tuple(control.name for control in controls) == ("forward",)
    assert all(control.v_mps >= 0.0 for control in controls)
    assert all(not control.reverse for control in controls)
    assert all(not control.turn_in_place for control in controls)
    assert all(abs(control.v_mps) <= profile.max_speed_mps for control in controls)
    assert all(
        abs(control.omega_radps) <= profile.max_angular_speed_radps
        for control in controls
    )


def test_reverse_enabled_rear_goal_uses_reverse_control() -> None:
    profile = _wheel_profile(
        reverse_enabled=True,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(2.25, 3.25, 0.0)

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, goal),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    assert outcome.route.primitives[-1].end_state == goal
    assert any(primitive.reverse for primitive in outcome.route.primitives)
    assert any(primitive.v_mps < 0.0 for primitive in outcome.route.primitives)


def test_turn_in_place_enabled_reaches_heading_only_goal() -> None:
    profile = _wheel_profile(
        reverse_enabled=False,
        turn_in_place_enabled=True,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=pi / 2.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(3.25, 3.25, pi / 2.0)

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, goal),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    assert any(primitive.turn_in_place for primitive in outcome.route.primitives)
    assert outcome.route.primitives[-1].end_state == goal


def test_turn_in_place_disabled_heading_only_goal_is_unreachable() -> None:
    profile = _wheel_profile(
        reverse_enabled=False,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=pi / 2.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(3.25, 3.25, pi / 2.0)

    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            start,
            goal,
            budget=ResourceBudgetV2(max_expanded_states=50),
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.GOAL_POSE_UNREACHABLE,
        "wheel_goal_pose_unreachable",
        "hybrid_search",
    )


def test_exact_start_goal_hold_bypasses_hybrid_and_has_zero_l2_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(idle_energy_per_s=5.0)
    snapshot = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)

    class ForbiddenHybrid:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("exact hold must bypass Hybrid A*")

    monkeypatch.setattr(
        wheel_module,
        "HybridAStarPlanner",
        ForbiddenHybrid,
        raising=False,
    )
    outcome = _provider(profile).plan(
        _request(profile, snapshot, state, state),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    assert len(outcome.route.primitives) == 1
    hold = outcome.route.primitives[0]
    assert hold.control_name == "hold"
    assert hold.start_state == state
    assert hold.end_state == state
    assert hold.samples == (state,)
    assert hold.duration_s == 0.0
    assert hold.distance_m == 0.0
    assert hold.energy_cost == 0.0
    assert hold.observation_contribution == 0.0
    assert hold.validation_level is ValidationLevelV2.L2
    assert hold.v_mps == 0.0
    assert hold.omega_radps == 0.0
    assert hold.reverse is False
    assert hold.turn_in_place is False
    assert outcome.route.total_cost == 0.0
    assert outcome.cost_breakdown.distance_cost == 0.0
    assert outcome.cost_breakdown.risk_cost == 0.0
    assert outcome.cost_breakdown.energy_cost == 0.0
    assert outcome.cost_breakdown.time_cost == 0.0
    assert outcome.cost_breakdown.total_cost == 0.0
    assert outcome.validation_evidence.passed is True
    assert outcome.validation_evidence.level is ValidationLevelV2.L2


def test_exact_hold_queries_full_footprint_and_unsafe_cell_fails_as_start() -> None:
    profile = _wheel_profile()
    safe = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)
    footprint = conservative_wheel_pose_cells(
        Pose2D(state.x_m, state.y_m, state.heading_rad),
        safe.geometry,
        body_length_m=profile.body_length_m,
        body_width_m=profile.body_width_m,
        safety_margin_m=profile.footprint_safety_margin_m,
    )
    center = safe.geometry.world_to_cell(WorldPoint(state.x_m, state.y_m))
    blocked = next(cell for cell in footprint if cell != center)
    snapshot = _snapshot(hard_cells=(blocked,))

    outcome = _provider(profile).plan(
        _request(profile, snapshot, state, state),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.UNSAFE_START,
        "terrain_hard_obstacle",
        "route_validation",
    )


def test_actual_endpoint_within_wrap_safe_tolerance_is_not_snapped() -> None:
    profile = _wheel_profile(
        heading_tolerance_rad=0.02,
        primitive_duration_s=1.0,
        integration_dt_s=0.25,
        reverse_enabled=False,
        turn_in_place_enabled=True,
        min_turning_radius_m=20.0,
        max_angular_speed_radps=0.1,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    requested_goal = PoseStateV2(3.25, 3.25, tau + 0.11)

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, requested_goal),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    actual = outcome.route.primitives[-1].end_state
    assert actual.heading_rad == pytest.approx(0.1)
    assert actual != requested_goal
    assert abs((actual.heading_rad - requested_goal.heading_rad + pi) % tau - pi) == (
        pytest.approx(0.01)
    )


@pytest.mark.parametrize(
    ("control_name", "goal_x", "expected_energy"),
    [("forward", 4.25, 2.5), ("reverse", 2.25, 3.5)],
)
def test_independent_forward_reverse_energy_and_weighted_cost_breakdown(
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    goal_x: float,
    expected_energy: float,
) -> None:
    profile = _wheel_profile(
        primitive_duration_s=1.0,
        integration_dt_s=0.25,
        reverse_enabled=True,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
        translation_energy_per_m=2.0,
        rotation_energy_per_rad=3.0,
        idle_energy_per_s=0.5,
        reverse_energy_multiplier=1.5,
        energy_normalization=4.0,
        time_normalization_s=2.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    objective = ObjectiveProfileV2(
        distance_weight=0.25,
        risk_weight=0.0,
        energy_weight=0.5,
        time_weight=0.75,
    )
    captured: dict[str, object]

    def search(*_args, **_kwargs):
        control = next(
            item for item in captured["primitives"] if item.name == control_name
        )
        return _hybrid_success((control,))

    captured = _install_hybrid(monkeypatch, search)
    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            start,
            PoseStateV2(goal_x, 3.25, 0.0),
            objective=objective,
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    primitive = outcome.route.primitives[0]
    assert primitive.distance_m == pytest.approx(1.0)
    assert primitive.duration_s == pytest.approx(1.0)
    assert primitive.energy_cost == pytest.approx(expected_energy)
    costs = outcome.cost_breakdown
    assert costs.distance_cost == pytest.approx(0.25)
    assert costs.risk_cost == 0.0
    assert costs.energy_cost == pytest.approx(0.5 * expected_energy / 4.0)
    assert costs.time_cost == pytest.approx(0.75 / 2.0)
    assert costs.total_cost == pytest.approx(
        costs.distance_cost + costs.energy_cost + costs.time_cost
    )
    assert outcome.route.total_cost == costs.total_cost
    assert costs.total_cost != 999_999.0


@pytest.mark.parametrize(
    ("objective", "accelerator", "reason"),
    [
        (
            ObjectiveProfileV2(risk_weight=0.1, energy_weight=0.5, time_weight=0.5),
            AcceleratorPolicyV2.DISABLED,
            "wheel_risk_objective_unsupported",
        ),
        (
            ObjectiveProfileV2(),
            AcceleratorPolicyV2.REQUIRED,
            "wheel_accelerator_required_unsupported",
        ),
    ],
)
def test_unsupported_risk_and_required_accelerator_fail_without_search(
    monkeypatch: pytest.MonkeyPatch,
    objective: ObjectiveProfileV2,
    accelerator: AcceleratorPolicyV2,
    reason: str,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported capability must fail before search")
        ),
    )

    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            start,
            PoseStateV2(4.25, 3.25, 0.0),
            objective=objective,
            accelerator_policy=accelerator,
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.UNSUPPORTED_CAPABILITY,
        reason,
        "capability_check",
    )


def test_optional_accelerator_remains_disabled_on_success() -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)

    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            state,
            state,
            accelerator_policy=AcceleratorPolicyV2.OPTIONAL,
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    assert type(outcome) is PlanningSuccessV2
    assert outcome.search_telemetry.accelerator_used is False


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (
            ResourceBudgetV2(max_expanded_states=0),
            "wheel_expansion_budget_exhausted",
        ),
        (
            ResourceBudgetV2(max_memory_bytes=1),
            "wheel_search_grid_memory_budget_exceeded",
        ),
    ],
)
def test_zero_expansion_or_insufficient_memory_budget_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
    budget: ResourceBudgetV2,
    reason: str,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid resource budget must fail before search")
        ),
    )

    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            start,
            PoseStateV2(4.25, 3.25, 0.0),
            budget=budget,
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.RESOURCE_LIMIT,
        reason,
        "resource_check",
    )


def test_hybrid_max_iteration_exhaustion_maps_to_typed_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    audit = PoseSearchAudit(
        generated_primitives=5,
        rejected_poses=2,
        rejected_transitions=1,
        termination_reason=FailureReason.MAX_ITERATIONS.value,
    )
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_failure(
            FailureReason.MAX_ITERATIONS,
            expanded=3,
            audit=audit,
        ),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.RESOURCE_LIMIT,
        "wheel_expansion_budget_exhausted",
        "hybrid_search",
    )
    assert failure.search_telemetry.expanded_states == 3
    assert failure.search_telemetry.generated_primitives == 5
    assert failure.search_telemetry.rejected_l0 == 2
    assert failure.search_telemetry.rejected_l1 == 0
    assert failure.search_telemetry.rejected_l2 == 1
    assert failure.search_telemetry.termination_reason == "max_iterations"


def test_expired_shared_deadline_is_typed_timeout_without_partial_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("expired deadline must fail before search")
        ),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(now=0.0, end=0.0),
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.TIMEOUT,
        "planning_deadline_expired",
        "resource_check",
    )
    assert failure.search_telemetry.timed_out is True
    assert failure.search_telemetry.termination_reason == "planning_deadline_expired"
    assert failure.search_telemetry.accelerator_used is False
    assert failure.search_telemetry.ackermann_feasible_claimed is False


@pytest.mark.parametrize(
    ("hybrid_reason", "category", "reason"),
    [
        (
            FailureReason.TIMEOUT,
            FailureCategoryV2.TIMEOUT,
            "planning_deadline_expired",
        ),
        (
            FailureReason.VALIDATOR_ERROR,
            FailureCategoryV2.VALIDATION_FAILED,
            "wheel_hybrid_validator_error",
        ),
        (
            FailureReason.START_OUT_OF_BOUNDS,
            FailureCategoryV2.UNSAFE_START,
            "wheel_start_pose_invalid",
        ),
        (
            FailureReason.GOAL_OUT_OF_BOUNDS,
            FailureCategoryV2.UNSAFE_GOAL,
            "wheel_goal_pose_invalid",
        ),
        (
            FailureReason.UNREACHABLE,
            FailureCategoryV2.GOAL_POSE_UNREACHABLE,
            "wheel_goal_pose_unreachable",
        ),
        (
            FailureReason.INVALID_INPUT,
            FailureCategoryV2.NO_COMPLETE_ROUTE,
            "wheel_no_complete_route",
        ),
    ],
)
def test_empty_hybrid_failures_map_to_stable_typed_failures(
    monkeypatch: pytest.MonkeyPatch,
    hybrid_reason: FailureReason,
    category: FailureCategoryV2,
    reason: str,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    audit = PoseSearchAudit(
        generated_primitives=4,
        rejected_poses=2,
        rejected_transitions=1,
        timed_out=hybrid_reason is FailureReason.TIMEOUT,
        termination_reason=hybrid_reason.value,
    )
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_failure(
            hybrid_reason,
            expanded=2,
            audit=audit,
        ),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    failure = _assert_failure(outcome, category, reason, "hybrid_search")
    assert failure.search_telemetry.expanded_states == 2
    assert failure.search_telemetry.generated_primitives == 4
    assert failure.search_telemetry.rejected_l0 == 2
    assert failure.search_telemetry.rejected_l2 == 1
    assert failure.search_telemetry.timed_out is (
        hybrid_reason is FailureReason.TIMEOUT
    )


def test_final_route_state_budget_failure_is_typed_without_hold_partial() -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)

    outcome = _provider(profile).plan(
        _request(
            profile,
            snapshot,
            state,
            state,
            budget=ResourceBudgetV2(max_route_states=0),
        ),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.RESOURCE_LIMIT,
        "route_state_budget_exceeded",
        "route_validation",
    )


def test_final_goal_tolerance_failure_is_typed_without_candidate_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(
        reverse_enabled=False,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    captured: dict[str, object]

    def search(*_args, **_kwargs):
        forward = next(
            item for item in captured["primitives"] if item.name == "forward"
        )
        return _hybrid_success((forward,))

    captured = _install_hybrid(monkeypatch, search)
    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(5.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.GOAL_POSE_UNREACHABLE,
        "route_goal_tolerance_exceeded",
        "route_validation",
    )


def test_independent_replay_exception_is_stable_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    invalid_control = MotionPrimitive("forward", 1.0, 0.0, 1.0)
    object.__setattr__(invalid_control, "v_mps", float("nan"))
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_success((invalid_control,)),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.VALIDATION_FAILED,
        "wheel_control_replay_failed",
        "independent_replay",
    )


def test_unexpected_provider_exception_exposes_only_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)

    class ExplodingHybrid:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("unstable secret must not escape")

    monkeypatch.setattr(
        wheel_module,
        "HybridAStarPlanner",
        ExplodingHybrid,
        raising=False,
    )
    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, PoseStateV2(4.25, 3.25, 0.0)),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.INTERNAL_ERROR,
        "wheel_provider_exception",
        "hybrid_search",
    )
    assert failure.evidence.details == (("exception_type", "RuntimeError"),)
    assert "unstable secret" not in repr(failure)


def test_nonexact_near_goal_empty_hybrid_success_cannot_inject_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(position_tolerance_m=0.2)
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(3.35, 3.25, 0.0)
    _install_hybrid(
        monkeypatch,
        lambda *_args, **_kwargs: _hybrid_success(()),
    )

    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, goal),
        FineSafetyAnchorV2(snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.VALIDATION_FAILED,
        "wheel_control_replay_failed",
        "independent_replay",
    )


def test_exact_hold_snapshot_identity_failure_is_validation_not_unsafe_start() -> None:
    profile = _wheel_profile()
    request_snapshot = _snapshot()
    different_snapshot = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)

    outcome = _provider(profile).plan(
        _request(profile, request_snapshot, state, state),
        FineSafetyAnchorV2(different_snapshot),
        _deadline(),
    )

    _assert_failure(
        outcome,
        FailureCategoryV2.VALIDATION_FAILED,
        "terrain_snapshot_identity_mismatch",
        "route_validation",
    )


def test_deadline_expiring_after_final_l2_pass_replaces_success_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile()
    snapshot = _snapshot()
    state = PoseStateV2(3.25, 3.25, 0.0)

    class SwitchClock:
        expired = False

        def __call__(self) -> float:
            return 2.0 if self.expired else 0.0

    clock = SwitchClock()
    deadline = PlanningDeadlineV2(0.0, 1.0, clock)
    real_validate = validation_module.validate_route_l2

    def validate_then_expire(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        assert result.evidence.passed is True
        clock.expired = True
        return result

    monkeypatch.setattr(validation_module, "validate_route_l2", validate_then_expire)
    outcome = _provider(profile).plan(
        _request(profile, snapshot, state, state),
        FineSafetyAnchorV2(snapshot),
        deadline,
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.TIMEOUT,
        "planning_deadline_expired",
        "route_validation",
    )
    assert failure.search_telemetry.timed_out is True


def test_deadline_expiring_during_route_construction_replaces_success_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _wheel_profile(
        reverse_enabled=False,
        turn_in_place_enabled=False,
        min_turning_radius_m=2.0,
        max_angular_speed_radps=1.0,
    )
    snapshot = _snapshot()
    start = PoseStateV2(3.25, 3.25, 0.0)
    goal = PoseStateV2(4.25, 3.25, 0.0)
    captured: dict[str, object]

    class SwitchClock:
        expired = False

        def __call__(self) -> float:
            return 2.0 if self.expired else 0.0

    clock = SwitchClock()
    deadline = PlanningDeadlineV2(0.0, 1.0, clock)

    def search(*_args, **_kwargs):
        forward = next(
            item for item in captured["primitives"] if item.name == "forward"
        )
        return _hybrid_success((forward,))

    captured = _install_hybrid(monkeypatch, search)
    real_replace = wheel_module.replace

    def replace_then_expire(*args, **kwargs):
        result = real_replace(*args, **kwargs)
        clock.expired = True
        return result

    monkeypatch.setattr(wheel_module, "replace", replace_then_expire)
    outcome = _provider(profile).plan(
        _request(profile, snapshot, start, goal),
        FineSafetyAnchorV2(snapshot),
        deadline,
    )

    failure = _assert_failure(
        outcome,
        FailureCategoryV2.TIMEOUT,
        "planning_deadline_expired",
        "route_construction",
    )
    assert failure.search_telemetry.timed_out is True
