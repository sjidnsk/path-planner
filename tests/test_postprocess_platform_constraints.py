import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess import build_corridor, has_line_of_sight, run_postprocess, smooth_path
from path_planner.search import AStarPlanner


def make_profile(*, footprint_radius_m=0.75, min_turning_radius_m=None):
    return PlannerPlatformProfile(
        platform_key="test",
        platform_name="Test Platform",
        config_path=None,
        safety_margin_m=0.0,
        body_length_m=1.2,
        body_width_m=0.9,
        footprint_radius_m=footprint_radius_m,
        max_slope_deg=15.0,
        max_obstacle_height_m=0.2,
        ground_clearance_m=0.3,
        raw_min_turning_radius_m=min_turning_radius_m or 0.0,
        effective_min_turning_radius_m=min_turning_radius_m,
        energy_model=None,
        parameter_sources={},
        constraint_sources={"footprint_radius_m": "derived"},
        constraint_warnings=(),
    )


def make_grid(mask, resolution=0.5):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_corridor_fails_when_path_cell_violates_vehicle_footprint():
    grid = make_grid(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    profile = make_profile(footprint_radius_m=0.75)

    corridor = build_corridor(grid, (Cell(1, 1),), radius_cells=1, platform_profile=profile)

    assert corridor.status == "failed"
    assert corridor.failure_reason == "path_cell_violates_platform_footprint"
    assert corridor.inflated_blocked_count > corridor.original_blocked_count


def test_smoothing_rejects_shortcut_inside_vehicle_footprint_clearance():
    grid = make_grid(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )
    profile = make_profile(footprint_radius_m=0.75)
    raw_path = (Cell(0, 1), Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0), Cell(4, 1))

    result = smooth_path(grid, raw_path, platform_profile=profile)

    assert has_line_of_sight(grid, Cell(0, 1), Cell(4, 1), platform_profile=profile) is False
    assert result.cells != (Cell(0, 1), Cell(4, 1))
    assert result.fallback_reason is None


def test_run_postprocess_uses_platform_min_turning_radius_when_effective():
    grid = make_grid(np.ones((3, 3), dtype=bool), resolution=1.0)
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))
    profile = make_profile(footprint_radius_m=0.0, min_turning_radius_m=2.0)

    postprocess = run_postprocess(grid, result, platform_profile=profile, max_shortcut_cost=None)

    assert postprocess.platform_profile == profile
    assert postprocess.curvature_report.constraint_min_turning_radius == 2.0
    assert postprocess.constraint_warnings == ()
