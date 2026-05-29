import numpy as np

from path_planner.adapters import route_result_to_json_dict
from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.platform import PlannerPlatformProfile
from path_planner.search import AStarPlanner, SearchTerrainLayers, build_planning_grid


def make_profile(*, footprint_radius_m=1.0):
    return PlannerPlatformProfile(
        platform_key="test-rover",
        platform_name="Test Rover",
        config_path=None,
        safety_margin_m=0.0,
        body_length_m=1.6,
        body_width_m=1.2,
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


def make_grid(mask, resolution=1.0):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_planning_grid_uses_vehicle_inflated_passable_mask_for_search():
    grid = make_grid(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ]
    )

    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))

    assert grid.is_passable(Cell(1, 0)) is True
    assert planning_grid.is_passable(Cell(1, 0)) is False
    assert planning_grid.search_mode == "platform_aware_astar"
    assert planning_grid.passable_source == "inflated_passable_mask"
    assert planning_grid.original_blocked_count == 1
    assert planning_grid.inflated_blocked_count > planning_grid.original_blocked_count
    assert planning_grid.footprint_radius_m == 1.0
    assert planning_grid.constraints.footprint_radius_m == 1.0
    assert planning_grid.constraints.safety_margin_m == 0.0
    assert planning_grid.constraints.max_slope_deg == 15.0
    assert planning_grid.constraints.max_obstacle_height_m == 0.2
    assert planning_grid.constraints.min_turning_radius_m is None
    assert planning_grid.terrain_cost.shape == grid.spec.shape


def test_platform_aware_astar_rejects_gap_that_only_centerline_can_cross():
    mask = np.ones((5, 7), dtype=bool)
    mask[1, 3] = False
    mask[3, 3] = False
    grid = make_grid(mask)
    request = PlanRequest(start=Cell(0, 2), goal=Cell(6, 2))

    standard = AStarPlanner().plan(grid, request)
    platform_aware = AStarPlanner().plan(
        build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0)),
        request,
    )

    assert standard.success is True
    assert Cell(3, 2) in standard.path_cells
    assert platform_aware.success is False
    assert platform_aware.failure_reason is FailureReason.UNREACHABLE
    assert platform_aware.diagnostics.search_mode == "platform_aware_astar"
    assert platform_aware.diagnostics.passable_source == "inflated_passable_mask"


def test_route_json_records_platform_aware_search_metadata():
    mask = np.ones((5, 7), dtype=bool)
    mask[1, 3] = False
    mask[3, 3] = False
    grid = make_grid(mask)
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))

    result = AStarPlanner().plan(planning_grid, PlanRequest(start=Cell(0, 2), goal=Cell(6, 2)))
    payload = route_result_to_json_dict(result, grid.spec)

    assert payload["diagnostics"]["search_mode"] == "platform_aware_astar"
    assert payload["diagnostics"]["passable_source"] == "inflated_passable_mask"
    assert payload["diagnostics"]["platform_key"] == "test-rover"
    assert payload["diagnostics"]["footprint_radius_m"] == 1.0
    assert payload["diagnostics"]["inflated_blocked_count"] > payload["diagnostics"]["original_blocked_count"]


def test_planning_grid_accepts_structured_terrain_layers():
    grid = make_grid(np.ones((2, 3), dtype=bool))
    terrain_layers = SearchTerrainLayers(
        slope=np.zeros(grid.spec.shape),
        roughness=np.full(grid.spec.shape, 0.2),
        illumination=np.ones(grid.spec.shape),
        confidence=np.ones(grid.spec.shape),
    )

    planning_grid = build_planning_grid(
        grid,
        platform_profile=make_profile(footprint_radius_m=0.0),
        terrain_layers=terrain_layers,
    )

    assert planning_grid.terrain_layers.layer_names() == ("slope", "roughness", "illumination", "confidence")


def test_planning_grid_uses_original_passable_mask_when_footprint_is_not_effective():
    grid = make_grid(np.ones((2, 3), dtype=bool))

    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=0.0))

    assert planning_grid.search_mode == "standard_grid_astar"
    assert planning_grid.passable_source == "original_passable_mask"
