import math

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.optimization import TrajectoryOptimizationConfig, optimize_trajectory, resample_world_points
from path_planner.postprocess import build_corridor
from path_planner.postprocess.models import CorridorResult
from path_planner.trajectory import build_trackable_path


def make_grid(cost, mask=None, resolution=1.0):
    cost_array = np.asarray(cost, dtype=float)
    passable = np.ones(cost_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable)


def test_optimized_path_stays_inside_fixed_corridor_boxes():
    grid = make_grid(np.ones((5, 7)), resolution=0.5)
    cells = (Cell(0, 2), Cell(2, 2), Cell(4, 2), Cell(6, 2))
    trackable = build_trackable_path(
        grid,
        cells,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )
    corridor = build_corridor(grid, cells, radius_cells=1)

    result = optimize_trajectory(
        grid,
        trackable,
        corridor,
        platform_profile=None,
        config=TrajectoryOptimizationConfig(max_iterations=8),
    )

    assert result.fallback_status.used_reference_path is False
    assert len(result.optimized_path) == len(trackable.waypoints)
    for point, box in zip(result.optimized_path, result.corridor_boxes):
        assert box.min_x <= point.x <= box.max_x
        assert box.min_y <= point.y <= box.max_y
    assert result.metrics.is_within_corridor is True


def test_optimization_keeps_start_and_goal_fixed():
    grid = make_grid(np.ones((5, 7)), resolution=0.5)
    cells = (Cell(0, 2), Cell(2, 3), Cell(4, 2), Cell(6, 2))
    trackable = build_trackable_path(
        grid,
        cells,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )
    corridor = build_corridor(grid, cells, radius_cells=1)

    result = optimize_trajectory(
        grid,
        trackable,
        corridor,
        platform_profile=None,
        config=TrajectoryOptimizationConfig(max_iterations=12),
    )

    assert math.isclose(result.optimized_path[0].x, trackable.waypoints[0].world.x)
    assert math.isclose(result.optimized_path[0].y, trackable.waypoints[0].world.y)
    assert math.isclose(result.optimized_path[-1].x, trackable.waypoints[-1].world.x)
    assert math.isclose(result.optimized_path[-1].y, trackable.waypoints[-1].world.y)


def test_resample_world_points_keeps_endpoints_and_limits_spacing():
    grid = make_grid(np.ones((3, 8)), resolution=0.5)
    points = (
        grid.spec.cell_to_world(Cell(0, 1)),
        grid.spec.cell_to_world(Cell(3, 1)),
        grid.spec.cell_to_world(Cell(7, 1)),
    )

    resampled = resample_world_points(points, spacing_m=0.4)

    assert resampled[0] == points[0]
    assert resampled[-1] == points[-1]
    for previous, current in zip(resampled[:-1], resampled[1:]):
        assert math.hypot(current.x - previous.x, current.y - previous.y) <= 0.400001
    assert len(resampled) > len(points)


def test_optimization_outputs_execution_aware_metrics_and_resampled_path():
    grid = make_grid(np.ones((5, 7)), resolution=0.5)
    cells = (Cell(0, 2), Cell(2, 3), Cell(4, 2), Cell(6, 2))
    trackable = build_trackable_path(
        grid,
        cells,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )
    corridor = build_corridor(grid, cells, radius_cells=1)

    result = optimize_trajectory(
        grid,
        trackable,
        corridor,
        platform_profile=None,
        config=TrajectoryOptimizationConfig(
            max_iterations=12,
            resample_spacing_m=0.35,
            weight_tracking=2.0,
            weight_spacing=1.0,
            weight_speed_smoothness=0.5,
        ),
    )
    payload = result.to_dict()

    assert "resampled_optimized_path" in payload
    assert len(result.resampled_optimized_path) > len(result.optimized_path)
    assert result.resampled_optimized_path[0] == result.optimized_path[0]
    assert result.resampled_optimized_path[-1] == result.optimized_path[-1]
    assert result.metrics.waypoint_spacing_max_m <= 0.350001
    assert result.metrics.is_within_corridor is True
    for point, box in zip(result.resampled_optimized_path, result.resampled_corridor_boxes):
        assert box.contains(point)
    for key in (
        "waypoint_spacing_mean_m",
        "waypoint_spacing_max_m",
        "heading_change_max_deg",
        "tracking_error_proxy",
        "speed_smoothness_cost",
    ):
        assert key in payload["metrics"]
        assert payload["metrics"][key] >= 0.0
        assert key in payload["metrics"]["baseline_vs_optimized"]
    assert "warnings" in payload


def test_optimization_can_reduce_high_cost_exposure_inside_corridor():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 5.0, 5.0, 5.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        resolution=0.5,
    )
    cells = (Cell(0, 1), Cell(2, 1), Cell(4, 1))
    trackable = build_trackable_path(
        grid,
        cells,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )
    corridor = build_corridor(grid, cells, radius_cells=1)

    result = optimize_trajectory(
        grid,
        trackable,
        corridor,
        platform_profile=None,
        config=TrajectoryOptimizationConfig(max_iterations=20),
    )

    assert result.metrics.optimized_high_cost_exposure < result.metrics.reference_high_cost_exposure


def test_optimization_falls_back_when_corridor_is_unusable():
    grid = make_grid(np.ones((3, 4)), resolution=0.5)
    cells = (Cell(0, 1), Cell(1, 1), Cell(3, 1))
    trackable = build_trackable_path(
        grid,
        cells,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )
    corridor = CorridorResult(
        status="failed",
        radius_cells=1,
        sections=(),
        failure_reason="test_unusable_corridor",
    )

    result = optimize_trajectory(
        grid,
        trackable,
        corridor,
        platform_profile=None,
        config=TrajectoryOptimizationConfig(max_iterations=5),
    )

    assert result.solver_status == "fallback"
    assert result.fallback_status.used_reference_path is True
    assert result.fallback_status.reason == "test_unusable_corridor"
    assert [point.to_list() for point in result.optimized_path] == [
        waypoint.world.to_list() for waypoint in trackable.waypoints
    ]
