from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.search import (
    AStarPlanner,
    HybridAStarPlanner,
    MotionPrimitive,
    Pose2D,
    PosePlanRequest,
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


def test_hybrid_timeout_returns_no_partial_route_and_stable_audit_counts() -> None:
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
    assert first.pose_path == ()
    assert first.control_sequence == ()
    assert first.audit.timed_out is True
    assert first.audit.termination_reason == FailureReason.TIMEOUT.value
    assert first.audit == second.audit


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
