from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import path_planner.search.hybrid_astar as hybrid_astar_module
from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.search import (
    AStarPlanner,
    HybridAStarPlanner,
    MotionPrimitive,
    Pose2D,
    PosePlanRequest,
    PoseSearchAudit,
    PoseTransition,
    replay_motion_primitive,
)


class _TickClock:
    def __init__(self, *, step: float) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class _FlagClock:
    def __init__(self) -> None:
        self.expired = False

    def __call__(self) -> float:
        return 2.0 if self.expired else 0.0


def test_hybrid_astar_emits_pose_route_without_replacing_grid_astar() -> None:
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), np.ones((5, 5), dtype=bool))

    hybrid = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(2.0, 2.0, 0.0), goal=Pose2D(2.0, 2.0, math.pi / 2.0)),
    )
    grid_route = AStarPlanner().plan(
        grid,
        PlanRequest(start=Cell(2, 2), goal=Cell(2, 2)),
    )

    assert hybrid.success
    assert hybrid.to_route_dict(grid.spec)["trajectory_kind"] == "hybrid_astar_pose_path"
    assert grid_route.to_route_dict(grid.spec)["trajectory_kind"] == "geometric_path"
    assert hybrid.diagnostics.ackermann_feasible_claimed is False
    assert hybrid.diagnostics.to_dict()["dominance_key_policy"] == "single_best_pose_per_cell_theta_bin"


def test_hybrid_astar_supports_turn_in_place_primitive() -> None:
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), np.ones((5, 5), dtype=bool))

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(
            start=Pose2D(2.0, 2.0, 0.0),
            goal=Pose2D(2.0, 2.0, math.pi / 2.0),
            max_angular_speed_radps=math.radians(45.0),
        ),
    )

    assert result.success
    assert any(primitive.name == "turn_in_place_left" for primitive in result.control_sequence)
    assert len(result.pose_path) > len(result.control_sequence)
    assert result.cost_breakdown.rotation_cost > 0.0


def test_hybrid_astar_rectangular_footprint_rejects_blocked_goal() -> None:
    passable = np.ones((5, 5), dtype=bool)
    passable[2, 2] = False
    grid = CostGrid(GridSpec(width=5, height=5, resolution=1.0), np.ones((5, 5)), passable)

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(2.0, 2.0, 0.0)),
    )

    assert not result.success
    assert result.failure_reason is FailureReason.GOAL_BLOCKED
    assert result.diagnostics.footprint_length_m == 0.612
    assert result.diagnostics.footprint_width_m == 0.580


def test_public_replay_matches_hybrid_samples_and_includes_exact_boundaries() -> None:
    grid = CostGrid(
        GridSpec(width=8, height=8, resolution=1.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool),
    )
    start = Pose2D(2.0, 2.0, 0.0)
    primitive = MotionPrimitive("public_forward", 1.0, 0.0, 1.0)

    transition = replay_motion_primitive(start, primitive, 0.25)
    result = HybridAStarPlanner((primitive,)).plan(
        grid,
        PosePlanRequest(
            start=start,
            goal=Pose2D(3.0, 2.0, 0.0),
            position_tolerance_m=1.0e-9,
            theta_tolerance_rad=1.0e-9,
        ),
    )

    assert transition.samples[0] == start.normalized()
    assert transition.samples[-1] == transition.end
    assert transition.samples == (
        Pose2D(2.0, 2.0, 0.0),
        Pose2D(2.25, 2.0, 0.0),
        Pose2D(2.5, 2.0, 0.0),
        Pose2D(2.75, 2.0, 0.0),
        Pose2D(3.0, 2.0, 0.0),
    )
    assert transition.distance_m == 1.0
    assert transition.absolute_heading_change_rad == 0.0
    assert result.success
    assert result.pose_path == transition.samples


def test_public_replay_preserves_non_normalized_start_exactly() -> None:
    start = Pose2D(2.0, 3.0, 2.0 * math.pi + 0.25)

    transition = replay_motion_primitive(
        start,
        MotionPrimitive("forward", 1.0, 0.0, 0.25),
        0.25,
    )

    assert transition.start is start
    assert transition.samples[0] is start
    assert transition.end.theta_rad == pytest.approx(0.25)


@pytest.mark.parametrize("omega", [1.5 * math.pi, 2.0 * math.pi])
def test_public_replay_preserves_large_absolute_heading_change(omega: float) -> None:
    transition = replay_motion_primitive(
        Pose2D(1.0, 1.0, 0.0),
        MotionPrimitive("large_turn", 0.0, omega, 1.0, turn_in_place=True),
        1.0,
    )

    assert transition.absolute_heading_change_rad == pytest.approx(abs(omega))


def test_hybrid_cost_preserves_v1_wrap_safe_rotation_for_large_primitive_without_validators() -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )
    primitive = MotionPrimitive(
        "large_turn",
        0.0,
        3.0 * math.pi,
        1.0,
        turn_in_place=True,
    )
    request = PosePlanRequest(
        start=Pose2D(2.0, 2.0, 0.0),
        goal=Pose2D(2.0, 2.0, math.pi),
        integration_dt_s=1.0,
        position_tolerance_m=1.0e-9,
        theta_tolerance_rad=1.0e-9,
    )

    transition = replay_motion_primitive(request.start, primitive, request.integration_dt_s)
    result = HybridAStarPlanner((primitive,)).plan(grid, request)
    expected_search_heading_change = math.pi
    expected_total = expected_search_heading_change * (
        request.rotation_cost_weight + request.turn_penalty_weight
    )

    assert transition.absolute_heading_change_rad == pytest.approx(3.0 * math.pi)
    assert result.success
    assert result.cost_breakdown.rotation_cost == pytest.approx(
        expected_search_heading_change * request.rotation_cost_weight
    )
    assert result.cost_breakdown.turn_penalty == pytest.approx(
        expected_search_heading_change * request.turn_penalty_weight
    )
    assert result.total_cost == pytest.approx(expected_total)
    assert result.to_route_dict(grid.spec)["path_cost"] == pytest.approx(expected_total)


def test_non_bool_transition_validator_result_fails_closed() -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )
    result = HybridAStarPlanner((MotionPrimitive("forward", 1.0, 0.0, 1.0),)).plan(
        grid,
        PosePlanRequest(
            start=Pose2D(1.0, 2.0, 0.0),
            goal=Pose2D(2.0, 2.0, 0.0),
            position_tolerance_m=1.0e-9,
            theta_tolerance_rad=1.0e-9,
        ),
        transition_validator=lambda transition: 1,
    )

    assert not result.success
    assert result.failure_reason is not None
    assert result.failure_reason.value == "validator_error"
    assert result.pose_path == ()
    assert result.control_sequence == ()


@pytest.mark.parametrize(
    "validator",
    [lambda pose: 1, lambda pose: (_ for _ in ()).throw(RuntimeError("unstable"))],
)
def test_pose_validator_contract_errors_fail_closed_without_partial_route(validator) -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(2.0, 1.0, 0.0)),
        pose_validator=validator,
    )

    assert result.failure_reason is not None
    assert result.failure_reason.value == "validator_error"
    assert result.pose_path == ()
    assert result.control_sequence == ()
    assert result.audit.termination_reason == "validator_error"


def test_transition_validator_exception_fails_closed_without_partial_route() -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )

    def invalid(_transition):
        raise LookupError("unstable")

    result = HybridAStarPlanner((MotionPrimitive("forward", 1.0, 0.0, 1.0),)).plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(2.0, 1.0, 0.0)),
        transition_validator=invalid,
    )

    assert result.failure_reason is not None
    assert result.failure_reason.value == "validator_error"
    assert result.pose_path == ()
    assert result.control_sequence == ()


@pytest.mark.parametrize(
    ("reject_goal", "reason"),
    [(False, FailureReason.START_BLOCKED), (True, FailureReason.GOAL_BLOCKED)],
)
def test_start_and_goal_pose_validator_rejections_are_audited(reject_goal, reason) -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )
    start = Pose2D(1.0, 1.0, 0.0)
    goal = Pose2D(2.0, 1.0, 0.0)

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=start, goal=goal),
        pose_validator=(
            (lambda pose: pose != goal)
            if reject_goal
            else (lambda pose: pose != start)
        ),
    )

    assert result.failure_reason is reason
    assert result.audit.generated_primitives == 0
    assert result.audit.rejected_poses == 1
    assert result.audit.rejected_transitions == 0


def test_replay_deadline_checker_aborts_inside_integration_loop() -> None:
    checks = iter((False, True))

    with pytest.raises(TimeoutError, match="deadline"):
        replay_motion_primitive(
            Pose2D(1.0, 1.0, 0.0),
            MotionPrimitive("forward", 1.0, 0.0, 1.0),
            0.1,
            deadline_checker=lambda: next(checks),
        )


def test_replay_rejects_unbounded_step_count_before_allocating_samples(monkeypatch) -> None:
    monkeypatch.setattr(hybrid_astar_module, "MAX_REPLAY_STEPS", 2)
    with pytest.raises(ValueError, match="steps"):
        replay_motion_primitive(
            Pose2D(1.0, 1.0, 0.0),
            MotionPrimitive("forward", 1.0, 0.0, 1.0),
            0.25,
        )


def test_public_transition_and_audit_enforce_invariants() -> None:
    transition = replay_motion_primitive(
        Pose2D(1.0, 1.0, 0.0),
        MotionPrimitive("forward", 1.0, 0.0, 1.0),
        0.5,
    )

    with pytest.raises(TypeError, match="samples"):
        replace(transition, samples=list(transition.samples))
    with pytest.raises(ValueError, match="start"):
        replace(transition, start=Pose2D(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="distance"):
        replace(transition, distance_m=-1.0)
    with pytest.raises(ValueError, match="heading"):
        replace(transition, absolute_heading_change_rad=float("nan"))
    with pytest.raises(ValueError, match="heading"):
        replace(transition, absolute_heading_change_rad=0.25)
    with pytest.raises(ValueError, match="nonnegative"):
        PoseSearchAudit(generated_primitives=-1)
    with pytest.raises(ValueError, match="timeout"):
        PoseSearchAudit(timed_out=True, termination_reason="success")


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf")])
def test_hybrid_rejects_invalid_absolute_deadlines(deadline) -> None:
    grid = CostGrid(
        GridSpec(width=3, height=3, resolution=1.0),
        np.ones((3, 3)),
        np.ones((3, 3), dtype=bool),
    )
    with pytest.raises((TypeError, ValueError), match="deadline"):
        HybridAStarPlanner().plan(
            grid,
            PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(1.0, 1.0, 0.0)),
            deadline_monotonic_s=deadline,
        )


def test_hybrid_validates_clock_and_validator_callables() -> None:
    grid = CostGrid(
        GridSpec(width=3, height=3, resolution=1.0),
        np.ones((3, 3)),
        np.ones((3, 3), dtype=bool),
    )
    request = PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(1.0, 1.0, 0.0))

    with pytest.raises(TypeError, match="clock"):
        HybridAStarPlanner().plan(grid, request, monotonic_clock=None)
    with pytest.raises(TypeError, match="pose_validator"):
        HybridAStarPlanner().plan(grid, request, pose_validator=object())
    with pytest.raises(TypeError, match="transition_validator"):
        HybridAStarPlanner().plan(grid, request, transition_validator=object())
    with pytest.raises(ValueError, match="clock"):
        HybridAStarPlanner().plan(
            grid,
            request,
            deadline_monotonic_s=1.0,
            monotonic_clock=lambda: float("nan"),
        )


def test_rejected_transition_does_not_poison_same_dominance_key() -> None:
    grid = CostGrid(
        GridSpec(width=8, height=8, resolution=1.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool),
    )
    reject = MotionPrimitive("reject_low_cost", 1.0, 0.0, 1.0)
    accept = MotionPrimitive("accept_higher_cost", 1.2, 0.0, 1.0)
    request = PosePlanRequest(
        start=Pose2D(1.0, 2.0, 0.0),
        goal=Pose2D(2.2, 2.0, 0.0),
        position_tolerance_m=1.0e-9,
        theta_tolerance_rad=1.0e-9,
        closed_key_xy_resolution_m=2.0,
    )

    result = HybridAStarPlanner((reject, accept)).plan(
        grid,
        request,
        transition_validator=lambda transition: transition.primitive.name != "reject_low_cost",
    )

    assert result.success
    assert tuple(item.name for item in result.control_sequence) == ("accept_higher_cost",)
    assert result.audit.generated_primitives == 2
    assert result.audit.rejected_poses == 0
    assert result.audit.rejected_transitions == 1
    assert result.audit.termination_reason == "success"


def test_opt_in_pose_validator_owns_feasibility_but_not_center_bounds() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    grid = CostGrid(GridSpec(width=3, height=3, resolution=1.0), np.ones((3, 3)), blocked)
    request = PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(1.0, 1.0, 0.0))

    default = HybridAStarPlanner().plan(grid, request)
    opted_in = HybridAStarPlanner().plan(grid, request, pose_validator=lambda pose: True)
    out_of_bounds = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(-1.0, 1.0, 0.0), goal=Pose2D(1.0, 1.0, 0.0)),
        pose_validator=lambda pose: True,
    )

    assert default.failure_reason is FailureReason.START_BLOCKED
    assert opted_in.success
    assert out_of_bounds.failure_reason is FailureReason.START_OUT_OF_BOUNDS


def test_hybrid_timeout_during_preprocessing_has_no_partial_route_and_stable_audit() -> None:
    grid = CostGrid(
        GridSpec(width=20, height=20, resolution=1.0),
        np.ones((20, 20)),
        np.ones((20, 20), dtype=bool),
    )
    request = PosePlanRequest(start=Pose2D(2.0, 2.0, 0.0), goal=Pose2D(17.0, 17.0, 0.0))

    def run_once():
        clock = _TickClock(step=0.01)
        return HybridAStarPlanner().plan(
            grid,
            request,
            deadline_monotonic_s=0.08,
            monotonic_clock=clock,
        )

    first = run_once()
    second = run_once()

    assert first.failure_reason is FailureReason.TIMEOUT
    assert first.expanded_count == 0
    assert first.pose_path == ()
    assert first.control_sequence == ()
    assert first.audit.generated_primitives == 0
    assert first.audit.timed_out is True
    assert first.audit.termination_reason == FailureReason.TIMEOUT.value
    assert first.audit == second.audit


def test_hybrid_replay_timeout_returns_typed_empty_failure(monkeypatch) -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )
    clock = _FlagClock()
    original_replay = hybrid_astar_module.replay_motion_primitive

    def expire_then_replay(start, primitive, integration_dt_s, *, deadline_checker=None):
        assert deadline_checker is not None
        clock.expired = True
        return original_replay(
            start,
            primitive,
            integration_dt_s,
            deadline_checker=deadline_checker,
        )

    monkeypatch.setattr(hybrid_astar_module, "replay_motion_primitive", expire_then_replay)
    result = HybridAStarPlanner((MotionPrimitive("forward", 1.0, 0.0, 1.0),)).plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 2.0, 0.0), goal=Pose2D(2.0, 2.0, 0.0)),
        deadline_monotonic_s=1.0,
        monotonic_clock=clock,
    )

    assert result.failure_reason is FailureReason.TIMEOUT
    assert result.pose_path == ()
    assert result.control_sequence == ()
    assert result.audit.generated_primitives == 1
    assert result.audit.timed_out is True


def test_hybrid_timeout_after_reconstruct_discards_late_success(monkeypatch) -> None:
    grid = CostGrid(
        GridSpec(width=5, height=5, resolution=1.0),
        np.ones((5, 5)),
        np.ones((5, 5), dtype=bool),
    )
    clock = _FlagClock()
    original_reconstruct = hybrid_astar_module._reconstruct

    def reconstruct_then_expire(nodes, key):
        reconstructed = original_reconstruct(nodes, key)
        clock.expired = True
        return reconstructed

    monkeypatch.setattr(hybrid_astar_module, "_reconstruct", reconstruct_then_expire)
    result = HybridAStarPlanner((MotionPrimitive("forward", 1.0, 0.0, 1.0),)).plan(
        grid,
        PosePlanRequest(
            start=Pose2D(1.0, 2.0, 0.0),
            goal=Pose2D(2.0, 2.0, 0.0),
            position_tolerance_m=1.0e-9,
            theta_tolerance_rad=1.0e-9,
        ),
        deadline_monotonic_s=1.0,
        monotonic_clock=clock,
    )

    assert result.failure_reason is FailureReason.TIMEOUT
    assert result.pose_path == ()
    assert result.control_sequence == ()
    assert result.audit.timed_out is True


def test_hybrid_default_serialization_does_not_expose_new_audit_fields() -> None:
    grid = CostGrid(
        GridSpec(width=3, height=3, resolution=1.0),
        np.ones((3, 3)),
        np.ones((3, 3), dtype=bool),
    )
    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(start=Pose2D(1.0, 1.0, 0.0), goal=Pose2D(1.0, 1.0, 0.0)),
    )

    route = result.to_route_dict(grid.spec)
    assert "audit" not in route
    assert "generated_primitives" not in route["diagnostics"]


def test_hybrid_default_v1_route_control_and_serialization_golden(monkeypatch) -> None:
    monkeypatch.setattr(hybrid_astar_module.time, "perf_counter", lambda: 10.0)
    grid = CostGrid(
        GridSpec(width=8, height=8, resolution=1.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool),
    )

    result = HybridAStarPlanner().plan(
        grid,
        PosePlanRequest(
            start=Pose2D(2.0, 2.0, 0.0),
            goal=Pose2D(3.0, 2.0, 0.0),
            position_tolerance_m=1.0e-9,
            theta_tolerance_rad=1.0e-9,
        ),
    )

    assert result.pose_path == (
        Pose2D(2.0, 2.0, 0.0),
        Pose2D(2.25, 2.0, 0.0),
        Pose2D(2.5, 2.0, 0.0),
        Pose2D(2.75, 2.0, 0.0),
        Pose2D(3.0, 2.0, 0.0),
    )
    assert tuple(primitive.to_dict() for primitive in result.control_sequence) == (
        {
            "name": "forward",
            "v_mps": 1.0,
            "omega_radps": 0.0,
            "duration_s": 1.0,
            "reverse": False,
            "turn_in_place": False,
        },
    )
    assert result.to_route_dict(grid.spec) == {
        "schema_version": "path-planner-hybrid-pose-route/v1",
        "trajectory_kind": "hybrid_astar_pose_path",
        "reachable": True,
        "path_cost": 1.0,
        "failure_reason": None,
        "grid": {
            "width": 8,
            "height": 8,
            "resolution": 1.0,
            "origin": [0.0, 0.0],
            "frame_id": "map",
        },
        "pose_path": [
            {"x_m": 2.0, "y_m": 2.0, "theta_rad": 0.0, "theta_deg": 0.0},
            {"x_m": 2.25, "y_m": 2.0, "theta_rad": 0.0, "theta_deg": 0.0},
            {"x_m": 2.5, "y_m": 2.0, "theta_rad": 0.0, "theta_deg": 0.0},
            {"x_m": 2.75, "y_m": 2.0, "theta_rad": 0.0, "theta_deg": 0.0},
            {"x_m": 3.0, "y_m": 2.0, "theta_rad": 0.0, "theta_deg": 0.0},
        ],
        "control_sequence": [
            {
                "name": "forward",
                "v_mps": 1.0,
                "omega_radps": 0.0,
                "duration_s": 1.0,
                "reverse": False,
                "turn_in_place": False,
            }
        ],
        "legacy_cell_path": [[2, 2], [3, 2]],
        "cost_breakdown": {
            "translation_cost": 1.0,
            "rotation_cost": 0.0,
            "reverse_penalty": 0.0,
            "turn_penalty": 0.0,
            "slope_cost": 0.0,
            "clearance_cost": 0.0,
            "total": 1.0,
        },
        "expanded_count": 2,
        "diagnostics": {
            "runtime_ms": 0.0,
            "expanded_pose_count": 2,
            "max_frontier_size": 8,
            "theta_bin_count": 72,
            "heuristic_policy": "max_2d_grid_cost_euclidean_heading",
            "search_mode": "hybrid_astar_pose_path",
            "platform_model": "differential_skid_steer",
            "ackermann_feasible_claimed": False,
            "footprint_length_m": 0.612,
            "footprint_width_m": 0.58,
            "footprint_safety_margin_m": 0.0,
            "primitives": [
                "forward",
                "forward_left",
                "forward_right",
                "reverse",
                "reverse_left",
                "reverse_right",
                "turn_in_place_left",
                "turn_in_place_right",
            ],
            "dominance_key_policy": "single_best_pose_per_cell_theta_bin",
            "hard_obstacle_sources": [
                "physical_obstacle_cells",
                "slope_blocked_cells",
                "blocked_cells",
            ],
        },
    }
