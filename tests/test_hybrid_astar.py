from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.search import AStarPlanner, HybridAStarPlanner, Pose2D, PosePlanRequest


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
