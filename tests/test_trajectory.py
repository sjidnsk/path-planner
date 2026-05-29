import math

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.platform import PlannerPlatformProfile
from path_planner.trajectory import build_trackable_path, evaluate_tracking_safety


def make_grid(cost, mask=None, resolution=1.0):
    cost_array = np.asarray(cost, dtype=float)
    passable = np.ones(cost_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable)


def make_profile(*, footprint_radius_m=0.5):
    return PlannerPlatformProfile(
        platform_key="test-rover",
        platform_name="Test Rover",
        config_path=None,
        safety_margin_m=0.0,
        body_length_m=0.8,
        body_width_m=0.6,
        footprint_radius_m=footprint_radius_m,
        max_slope_deg=15.0,
        max_obstacle_height_m=0.2,
        ground_clearance_m=0.3,
        raw_min_turning_radius_m=0.0,
        effective_min_turning_radius_m=None,
        energy_model=None,
        parameter_sources={},
        constraint_sources={"footprint_radius_m": "derived"},
        constraint_warnings=(),
    )


def test_trackable_path_computes_heading_and_segment_length_for_straight_path():
    grid = make_grid(np.ones((1, 3)), resolution=0.5)
    path = (Cell(0, 0), Cell(1, 0), Cell(2, 0))

    trackable = build_trackable_path(
        grid,
        path,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    assert trackable.source_path == "smoothed_path"
    assert trackable.length_m == 1.0
    assert len(trackable.waypoints) == 3
    assert [waypoint.segment_length_m for waypoint in trackable.waypoints] == [0.0, 0.5, 0.5]
    assert all(math.isclose(waypoint.heading_rad, 0.0) for waypoint in trackable.waypoints)
    assert all(waypoint.turn_angle_deg == 0.0 for waypoint in trackable.waypoints)
    assert all(waypoint.turning_radius_m is None for waypoint in trackable.waypoints)


def test_trackable_path_computes_turn_angle_curvature_and_turning_radius():
    grid = make_grid(np.ones((2, 2)), resolution=1.0)
    path = (Cell(0, 0), Cell(1, 0), Cell(1, 1))

    trackable = build_trackable_path(
        grid,
        path,
        source_path="raw_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    turn = trackable.waypoints[1]
    assert trackable.source_path == "raw_path"
    assert math.isclose(turn.turn_angle_deg, 90.0)
    assert math.isclose(turn.curvature, math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(turn.turning_radius_m, 1.0 / math.sqrt(2.0), rel_tol=1e-6)


def test_speed_profile_slows_down_at_sharp_turns_and_high_cost_cells():
    cost = np.array(
        [
            [1.0, 1.0],
            [1.0, 5.0],
        ]
    )
    grid = make_grid(cost, resolution=1.0)
    path = (Cell(0, 0), Cell(1, 0), Cell(1, 1))

    trackable = build_trackable_path(
        grid,
        path,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    speeds = trackable.speed_profile
    assert speeds[0] == 0.2
    assert speeds[1] < speeds[0]
    assert speeds[2] < speeds[0]
    assert min(speeds) >= 0.05


def test_tracking_safety_reports_violation_when_error_tube_reaches_obstacle():
    mask = np.ones((3, 5), dtype=bool)
    mask[1, 3] = False
    grid = make_grid(np.ones((3, 5)), mask=mask, resolution=0.5)
    path = (Cell(0, 1), Cell(1, 1), Cell(2, 1))
    trackable = build_trackable_path(
        grid,
        path,
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    report = evaluate_tracking_safety(
        grid,
        trackable,
        platform_profile=make_profile(footprint_radius_m=0.5),
        tracking_error_bound_m=0.5,
    )

    assert report.is_safe is False
    assert report.tracking_error_bound_m == 0.5
    assert report.checked_radius_m == 1.0
    assert report.violation_indices == (2,)
    assert report.min_clearance_m == 0.0
    assert "tracking safety violations" in report.summary
