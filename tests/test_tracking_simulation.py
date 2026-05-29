import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.platform import PlannerPlatformProfile
from path_planner.tracking import TrackingSimulationConfig, simulate_tracking
from path_planner.trajectory import build_trackable_path


def make_grid(cost, mask=None, resolution=1.0):
    cost_array = np.asarray(cost, dtype=float)
    passable = np.ones(cost_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable)


def make_profile(*, footprint_radius_m=0.25):
    return PlannerPlatformProfile(
        platform_key="test-rover",
        platform_name="Test Rover",
        config_path=None,
        safety_margin_m=0.0,
        body_length_m=0.4,
        body_width_m=0.3,
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
        speed_max_mps=0.2,
    )


def test_straight_path_tracking_keeps_cross_track_error_near_zero():
    grid = make_grid(np.ones((3, 8)), resolution=0.5)
    trackable = build_trackable_path(
        grid,
        (Cell(0, 1), Cell(3, 1), Cell(7, 1)),
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    result = simulate_tracking(
        grid,
        trackable,
        platform_profile=make_profile(),
        config=TrackingSimulationConfig(lookahead_m=0.75, time_step_s=0.2, max_sim_time_s=30.0),
    )

    assert result.safety_report.is_safe is True
    assert result.metrics.max_cross_track_error_m < 1e-6
    assert result.metrics.simulated_length_m > 0.0
    assert result.states[-1].target_waypoint_index == len(trackable.waypoints) - 1


def test_corner_path_tracking_produces_measurable_cross_track_error():
    grid = make_grid(np.ones((8, 8)), resolution=0.5)
    trackable = build_trackable_path(
        grid,
        (Cell(0, 2), Cell(5, 2), Cell(5, 7)),
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    result = simulate_tracking(
        grid,
        trackable,
        platform_profile=make_profile(),
        config=TrackingSimulationConfig(lookahead_m=0.75, time_step_s=0.2, max_sim_time_s=40.0),
    )

    assert result.metrics.max_cross_track_error_m > 0.05
    assert result.metrics.path_length_m == trackable.length_m
    assert result.metrics.mean_speed_mps > 0.0


def test_tracking_simulation_reports_safety_violation_near_obstacle():
    mask = np.ones((3, 8), dtype=bool)
    mask[1, 4] = False
    grid = make_grid(np.ones((3, 8)), mask=mask, resolution=0.5)
    trackable = build_trackable_path(
        grid,
        (Cell(0, 1), Cell(3, 1), Cell(7, 1)),
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    result = simulate_tracking(
        grid,
        trackable,
        platform_profile=make_profile(footprint_radius_m=0.5),
        config=TrackingSimulationConfig(lookahead_m=0.75, time_step_s=0.2, max_sim_time_s=30.0),
    )

    assert result.safety_report.is_safe is False
    assert result.safety_report.violation_indices
    assert result.metrics.safety_violation_count == len(result.safety_report.violation_indices)
    assert result.metrics.min_clearance_m == 0.0


def test_tracking_simulation_serializes_experiment_metrics():
    grid = make_grid(np.array([[1.0, 1.0, 5.0, 1.0]]), resolution=0.5)
    trackable = build_trackable_path(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0)),
        source_path="smoothed_path",
        max_speed_mps=0.2,
        min_speed_mps=0.05,
    )

    result = simulate_tracking(
        grid,
        trackable,
        platform_profile=make_profile(),
        config=TrackingSimulationConfig(lookahead_m=0.5, time_step_s=0.2, max_sim_time_s=20.0),
    )
    payload = result.to_dict()

    assert "simulated_path" in payload
    assert "metrics" in payload
    assert payload["metrics"]["path_length_m"] == trackable.length_m
    assert payload["metrics"]["high_cost_exposure"] > 0.0
    assert payload["config"]["lookahead_m"] == 0.5
