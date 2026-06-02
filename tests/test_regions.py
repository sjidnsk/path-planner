import numpy as np

from path_planner.adapters import route_result_to_json_dict
from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.drake_backend import IrisRegion, IrisRegionReport
from path_planner.postprocess import build_corridor
from path_planner.regions import (
    ConvexRegion,
    ObstaclePrimitive,
    build_blocked_cell_obstacles,
    build_grid_box_regions,
    build_merged_blocked_cell_obstacles,
    build_region_graph_report,
)
from path_planner.search import AStarPlanner


def make_grid(mask, resolution=0.5):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_obstacle_primitive_serializes_blocked_cell_box_bounds():
    obstacle = ObstaclePrimitive(
        obstacle_id=2,
        source="blocked_cell_box",
        min_cell=Cell(3, 4),
        max_cell=Cell(3, 4),
        min_world=GridSpec(width=6, height=6, resolution=0.5).cell_to_world(Cell(3, 4)),
        max_world=GridSpec(width=6, height=6, resolution=0.5).cell_to_world(Cell(4, 5)),
    )

    payload = obstacle.to_dict()

    assert payload["id"] == 2
    assert payload["source"] == "blocked_cell_box"
    assert payload["cell_bounds"] == {"min": [3, 4], "max": [3, 4]}
    assert payload["world_bounds"] == {"min": [1.5, 2.0], "max": [2.0, 2.5]}


def test_build_blocked_cell_obstacles_from_passable_mask():
    grid = make_grid(
        [
            [True, False, True],
            [True, True, False],
        ],
        resolution=1.0,
    )

    obstacles = build_blocked_cell_obstacles(grid)

    assert len(obstacles) == 2
    assert [obstacle.min_cell.to_list() for obstacle in obstacles] == [[1, 0], [2, 1]]
    assert all(obstacle.source == "blocked_cell_box" for obstacle in obstacles)
    assert obstacles[0].to_dict()["world_bounds"] == {"min": [1.0, 0.0], "max": [2.0, 1.0]}


def test_merged_blocked_cell_obstacles_cover_only_unsafe_cells():
    grid = make_grid(
        [
            [True, False, False, True],
            [True, False, False, True],
            [True, True, False, False],
        ],
        resolution=1.0,
    )

    obstacles = build_merged_blocked_cell_obstacles(grid)
    payloads = [obstacle.to_dict() for obstacle in obstacles]

    assert len(obstacles) == 2
    assert all(obstacle.source == "merged_blocked_rectangle" for obstacle in obstacles)
    assert payloads[0]["cell_bounds"] == {"min": [1, 0], "max": [2, 1]}
    assert payloads[1]["cell_bounds"] == {"min": [2, 2], "max": [3, 2]}


def test_grid_box_regions_serialize_corridor_sections():
    grid = make_grid(np.ones((4, 5), dtype=bool), resolution=0.5)
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1)), radius_cells=1)

    regions = build_grid_box_regions(grid, corridor)
    payload = regions[0].to_dict()

    assert len(regions) == 2
    assert isinstance(regions[0], ConvexRegion)
    assert payload["source"] == "grid_box"
    assert payload["center_cell"] == [1, 1]
    assert payload["cell_bounds"] == {"min": [0, 0], "max": [2, 2]}
    assert payload["world_bounds"] == {"min": [0.0, 0.0], "max": [1.5, 1.5]}
    assert payload["validation_status"] == "valid"


def test_region_graph_report_builds_vertices_edges_and_summary():
    grid = make_grid(np.ones((4, 6), dtype=bool), resolution=0.5)
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)

    report = build_region_graph_report(grid, corridor)
    payload = report.to_dict()

    assert payload["status"] == "ok"
    assert payload["vertex_count"] == 3
    assert payload["edge_count"] >= 2
    assert payload["region_source"] == "grid_box"
    assert payload["obstacle_source"] == "blocked_cell_box"
    assert payload["fallback_used"] is False
    assert payload["motion_feasibility_status"] == "diagnostic_only"
    assert payload["graph"]["regions"]
    assert payload["graph"]["edges"]


def test_region_graph_report_is_optional_route_json_field_without_changing_route_semantics():
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=0.5)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    region_graph_report = build_region_graph_report(grid, corridor)

    payload = route_result_to_json_dict(plan, grid.spec, region_graph_report=region_graph_report)

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["region_graph_report"]["status"] == "ok"
    assert payload["region_graph_report"]["vertex_count"] == len(plan.path_cells)


def test_region_graph_report_can_use_valid_iris_regions_as_graph_source():
    grid = make_grid(np.ones((4, 6), dtype=bool), resolution=0.5)
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)
    grid_regions = build_grid_box_regions(grid, corridor)
    iris_report = IrisRegionReport(
        backend="workspace_iris",
        status="ok",
        seed_source="postprocess_corridor_centers",
        domain_source="postprocess_corridor_grid_box",
        obstacle_source="blocked_cell_box",
        regions=tuple(_iris_region_from_grid_region(region) for region in grid_regions),
        obstacle_count=0,
        validation_status="valid",
        fallback_used=False,
    )

    report = build_region_graph_report(grid, corridor, iris_region_report=iris_report)
    payload = report.to_dict()

    assert payload["status"] == "ok"
    assert payload["region_source"] == "iris"
    assert payload["fallback_used"] is False
    assert payload["quality_metrics"]["requested_region_source"] == "iris"
    assert payload["quality_metrics"]["graph_source"] == "iris"
    assert payload["quality_metrics"]["iris_region_count"] == len(grid_regions)
    assert payload["quality_metrics"]["grid_fallback_region_count"] == 0
    assert payload["quality_metrics"]["connected_component_count"] == 1
    assert payload["quality_metrics"]["start_goal_connected"] is True
    assert payload["graph"]["edges"][0]["source"] == "sampled_connectivity"


def test_region_graph_report_falls_back_to_grid_box_when_iris_report_is_unavailable():
    grid = make_grid(np.ones((4, 6), dtype=bool), resolution=0.5)
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)
    iris_report = IrisRegionReport(
        backend="workspace_iris",
        status="fallback",
        seed_source="postprocess_corridor_centers",
        domain_source="postprocess_corridor_grid_box",
        obstacle_source="blocked_cell_box",
        regions=(),
        obstacle_count=0,
        validation_status="not_evaluated",
        failure_status="backend_unavailable",
        failure_reason="backend_unavailable: simulated missing pydrake",
        fallback_used=True,
    )

    report = build_region_graph_report(grid, corridor, iris_region_report=iris_report)
    payload = report.to_dict()

    assert payload["status"] == "ok"
    assert payload["region_source"] == "grid_box"
    assert payload["fallback_used"] is True
    assert "backend_unavailable" in payload["failure_reason"]
    assert payload["quality_metrics"]["requested_region_source"] == "iris"
    assert payload["quality_metrics"]["graph_source"] == "grid_box"
    assert payload["quality_metrics"]["grid_fallback_region_count"] == len(corridor.sections)
    assert payload["quality_metrics"]["fallback_ratio"] == 1.0
    assert payload["quality_metrics"]["start_goal_connected"] is True


def test_region_graph_report_falls_back_when_iris_graph_is_disconnected():
    grid = make_grid(np.ones((3, 7), dtype=bool), resolution=0.5)
    corridor = build_corridor(grid, (Cell(1, 1), Cell(2, 1), Cell(3, 1)), radius_cells=1)
    disconnected_regions = (
        _single_cell_iris_region(0, Cell(0, 1), grid),
        _single_cell_iris_region(1, Cell(3, 1), grid),
        _single_cell_iris_region(2, Cell(6, 1), grid),
    )
    iris_report = IrisRegionReport(
        backend="workspace_iris",
        status="ok",
        seed_source="postprocess_corridor_centers",
        domain_source="postprocess_corridor_grid_box",
        obstacle_source="blocked_cell_box",
        regions=disconnected_regions,
        obstacle_count=0,
        validation_status="valid",
        fallback_used=False,
    )

    report = build_region_graph_report(grid, corridor, iris_region_report=iris_report)
    payload = report.to_dict()

    assert payload["region_source"] == "grid_box"
    assert payload["fallback_used"] is True
    assert "start_goal_not_connected" in payload["failure_reason"]
    assert payload["quality_metrics"]["fallback_reason"] == payload["failure_reason"]


def _iris_region_from_grid_region(region):
    return IrisRegion(
        region_id=region.region_id,
        source="iris",
        seed_cell=region.center_cell,
        seed_world=region.min_world,
        min_cell=region.min_cell,
        max_cell=region.max_cell,
        min_world=region.min_world,
        max_world=region.max_world,
        domain_min_cell=region.min_cell,
        domain_max_cell=region.max_cell,
        domain_min_world=region.min_world,
        domain_max_world=region.max_world,
        hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        hpolyhedron_b=(region.max_world.x, -region.min_world.x, region.max_world.y, -region.min_world.y),
        validation_status="valid",
    )


def _single_cell_iris_region(region_id, cell, grid):
    min_world = grid.spec.cell_to_world(cell)
    max_world = grid.spec.cell_to_world(Cell(cell.x + 1, cell.y + 1))
    return IrisRegion(
        region_id=region_id,
        source="iris",
        seed_cell=cell,
        seed_world=min_world,
        min_cell=cell,
        max_cell=cell,
        min_world=min_world,
        max_world=max_world,
        domain_min_cell=cell,
        domain_max_cell=cell,
        domain_min_world=min_world,
        domain_max_world=max_world,
        hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        hpolyhedron_b=(max_world.x, -min_world.x, max_world.y, -min_world.y),
        validation_status="valid",
    )
