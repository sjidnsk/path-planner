import math

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, NeighborPolicy, PlanDiagnostics, PlanRequest, PlanResult
from path_planner.regions import ConvexRegion, RegionEdge, RegionGraph, RegionGraphReport
from path_planner.search import AStarPlanner, RegionGraphGuidedPlanner


def make_grid(cost, passable=None):
    cost_array = np.asarray(cost, dtype=float)
    passable_mask = np.ones(cost_array.shape, dtype=bool) if passable is None else np.asarray(passable, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=1.0)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable_mask)


def make_region(region_id, cell, grid):
    world = grid.spec.cell_to_world(cell)
    max_world = grid.spec.cell_to_world(Cell(cell.x + 1, cell.y + 1))
    return ConvexRegion(
        region_id=region_id,
        source="manual",
        center_cell=cell,
        min_cell=cell,
        max_cell=cell,
        min_world=world,
        max_world=max_world,
        cell_count=1,
    )


def make_box_region(region_id, min_cell, max_cell, center, grid):
    min_world = grid.spec.cell_to_world(min_cell)
    max_world = grid.spec.cell_to_world(Cell(max_cell.x + 1, max_cell.y + 1))
    return ConvexRegion(
        region_id=region_id,
        source="manual",
        center_cell=center,
        min_cell=min_cell,
        max_cell=max_cell,
        min_world=min_world,
        max_world=max_world,
        cell_count=(max_cell.x - min_cell.x + 1) * (max_cell.y - min_cell.y + 1),
    )


def make_report(grid, centers, edges, *, connected=True):
    regions = tuple(make_region(index, center, grid) for index, center in enumerate(centers))
    return make_report_from_regions(regions, edges, connected=connected)


def make_report_from_regions(regions, edges, *, connected=True):
    graph_edges = tuple(
        RegionEdge(
            edge_id=index,
            source="manual",
            from_region_id=left,
            to_region_id=right,
            connection_kind="manual_test_edge",
        )
        for index, (left, right) in enumerate(edges)
    )
    return RegionGraphReport(
        status="ok",
        region_source="manual",
        obstacle_source="blocked_cell_box",
        graph=RegionGraph(regions=regions, edges=graph_edges, obstacles=()),
        failure_reason=None,
        fallback_used=False,
        motion_feasibility_status="diagnostic_only",
        quality_metrics={
            "connected_component_count": 1 if connected else len(regions),
            "start_goal_connected": connected,
        },
    )


def make_baseline_result(grid, path_cells, total_cost):
    return PlanResult(
        success=True,
        path_cells=tuple(path_cells),
        path_world=tuple(grid.spec.cell_to_world(cell) for cell in path_cells),
        total_cost=total_cost,
        expanded_count=len(path_cells),
        failure_reason=None,
        diagnostics=PlanDiagnostics(
            path_length_m=float(len(path_cells) - 1),
            expanded_cells=tuple(path_cells),
            search_mode="platform_aware_astar",
        ),
    )


def test_region_graph_guided_selects_better_sampled_region_candidate():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 1.0],
            [6.0, 6.0, 6.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(3, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(0, 1), Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2), Cell(3, 1), Cell(3, 0)),
        total_cost=31.0,
    )
    report = make_report(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0)),
        ((0, 1), (1, 2), (2, 3)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result.path_cells == (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0))
    assert outcome.result.total_cost < baseline.total_cost
    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.report.fallback_reason is None
    assert outcome.report.segment_count == 3
    payload = outcome.report.to_dict()
    assert payload["region_graph_candidate"]["status"] == "selected"
    assert payload["comparison"]["path_changed"] is True
    assert payload["comparison"]["candidate_cost_delta"] < 0.0


def test_region_path_anchors_to_regions_containing_start_and_goal_not_graph_order():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0)),
        total_cost=20.0,
    )
    start_region = make_region(10, Cell(0, 0), grid)
    middle_region = make_region(20, Cell(2, 0), grid)
    goal_region = make_region(30, Cell(4, 0), grid)
    decoy_region = make_region(5, Cell(5, 0), grid)
    report = make_report_from_regions(
        (decoy_region, goal_region, middle_region, start_region),
        ((10, 20), (20, 30)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["region_sequence"] == [10, 20, 30]
    assert sampled["start_goal_anchoring"]["start_region_id"] == 10
    assert sampled["start_goal_anchoring"]["goal_region_id"] == 30


def test_sampled_region_path_uses_edge_adjacent_multi_sample_candidate():
    grid = make_grid(
        [
            [1.0, 9.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0)),
        total_cost=12.0,
    )
    report = make_report_from_regions(
        (
            make_box_region(0, Cell(0, 0), Cell(0, 1), Cell(0, 0), grid),
            make_box_region(1, Cell(1, 0), Cell(3, 1), Cell(2, 0), grid),
            make_box_region(2, Cell(4, 0), Cell(4, 1), Cell(4, 0), grid),
        ),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.result.total_cost < baseline.total_cost
    assert Cell(1, 1) in outcome.result.path_cells
    assert Cell(3, 1) in outcome.result.path_cells
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["sample_attempt_count"] > sampled["sample_count"]
    assert sampled["candidate_rankings"][0]["status"] == "selected"
    assert sampled["candidate_rankings"][0]["candidate_cost_delta"] < 0.0


def test_region_graph_guided_selects_sampled_region_path_candidate():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 8.0, 8.0, 8.0, 1.0],
            [1.0, 8.0, 1.0, 8.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(
        grid,
        (
            Cell(0, 0),
            Cell(0, 1),
            Cell(0, 2),
            Cell(1, 2),
            Cell(2, 2),
            Cell(3, 2),
            Cell(4, 2),
            Cell(4, 1),
            Cell(4, 0),
        ),
        total_cost=30.0,
    )
    report = make_report(
        grid,
        (Cell(0, 0), Cell(2, 0), Cell(4, 0)),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result.path_cells == (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0))
    assert outcome.result.total_cost < baseline.total_cost
    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    payload = outcome.report.to_dict()
    sampled = payload["sampled_region_path_report"]
    assert sampled["schema_version"] == "sampled_region_path_report/v1"
    assert sampled["status"] == "selected"
    assert sampled["fallback_reason"] is None
    assert sampled["region_sequence"] == [0, 1, 2]
    assert sampled["sample_count"] == 3
    assert sampled["safety_checks"]["collision_free"] is True
    assert sampled["candidate_comparison"]["candidate_cost_delta"] < 0.0
    assert payload["comparison"]["path_changed"] is True


def test_region_graph_guided_reports_sampled_path_collision_blocker():
    grid = make_grid(
        [[1.0, 1.0, 1.0]],
        passable=[[True, False, True]],
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = make_baseline_result(grid, (Cell(0, 0), Cell(2, 0)), total_cost=10.0)
    report = make_report(grid, (Cell(0, 0), Cell(2, 0)), ((0, 1),))

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    payload = outcome.report.to_dict()
    sampled = payload["sampled_region_path_report"]
    assert sampled["status"] == "fallback"
    assert sampled["fallback_reason"] == "sampled_path_collision"
    assert sampled["safety_checks"]["collision_free"] is False


def test_region_graph_guided_reports_goal_anchor_unconnected_blocker():
    grid = make_grid([[1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions((make_region(0, Cell(0, 0), grid),), ())

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "goal_anchor_region_unconnected"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "goal_anchor_region_unconnected"
    assert sampled["start_goal_anchoring"]["start_region_id"] == 0
    assert sampled["start_goal_anchoring"]["goal_region_id"] == 1
    assert sampled["start_goal_anchoring"]["goal_anchor_region_added"] is True
    assert sampled["start_goal_anchoring"]["goal_anchor_region_connected"] is False


def test_passable_goal_outside_region_gets_connected_anchor_region():
    grid = make_grid([[1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0)),
        total_cost=10.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 0), grid),
            make_region(1, Cell(1, 0), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] is None
    assert sampled["region_sequence"] == [0, 1, 2]
    anchoring = sampled["start_goal_anchoring"]
    assert anchoring["goal_classification"] == "goal_outside_region_coverage"
    assert anchoring["goal_anchor_region_added"] is True
    assert anchoring["goal_anchor_region_connected"] is True
    assert anchoring["goal_anchor_failure_reason"] is None
    assert anchoring["goal_region_id"] == 2


def test_blocked_goal_without_region_reports_goal_not_passable():
    grid = make_grid([[1.0, 1.0, 1.0]], passable=[[True, True, False]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions((make_region(0, Cell(0, 0), grid),), ())

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "goal_not_passable"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "goal_not_passable"
    anchoring = sampled["start_goal_anchoring"]
    assert anchoring["goal_classification"] == "goal_not_passable"
    assert anchoring["goal_anchor_region_added"] is False
    assert anchoring["goal_anchor_failure_reason"] == "goal_not_passable"


def test_passable_goal_anchor_unconnected_reports_specific_blocker():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions((make_region(0, Cell(0, 0), grid),), ())

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "goal_anchor_region_unconnected"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "goal_anchor_region_unconnected"
    anchoring = sampled["start_goal_anchoring"]
    assert anchoring["goal_classification"] == "goal_outside_region_coverage"
    assert anchoring["goal_anchor_region_added"] is True
    assert anchoring["goal_anchor_region_connected"] is False
    assert anchoring["goal_anchor_failure_reason"] == "goal_anchor_region_unconnected"


def test_cost_aware_connector_selects_lower_cost_route_inside_region_union():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 1), goal=Cell(4, 1), prevent_corner_cutting=False)
    baseline = make_baseline_result(
        grid,
        (Cell(0, 1), Cell(1, 1), Cell(2, 1), Cell(3, 1), Cell(4, 1)),
        total_cost=28.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 1), grid),
            make_box_region(1, Cell(1, 0), Cell(3, 2), Cell(2, 1), grid),
            make_region(2, Cell(4, 1), grid),
        ),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert Cell(2, 1) not in outcome.result.path_cells
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["candidate_rankings"][0]["strategy"] == "cost_aware_constrained_astar"
    assert sampled["candidate_rankings"][0]["status"] == "selected"
    assert sampled["candidate_rankings"][0]["candidate_cost_delta"] < 0.0


def test_higher_cost_constrained_connector_falls_back_with_cost_dominated_reason():
    grid = make_grid(
        [
            [1.0, 9.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 1), goal=Cell(4, 1), prevent_corner_cutting=False)
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 1), grid),
            make_box_region(1, Cell(1, 0), Cell(3, 0), Cell(2, 0), grid),
            make_region(2, Cell(4, 1), grid),
        ),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "region_sequence_cost_dominated"
    assert sampled["candidate_rankings"][0]["strategy"] == "cost_aware_constrained_astar"
    assert sampled["candidate_rankings"][0]["candidate_cost_delta"] > 0.0


def test_region_graph_guided_candidate_preserves_request_neighbor_policy_in_diagnostics():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(3, 0), neighbor_policy=NeighborPolicy.FOUR)
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(0, 0), Cell(0, 0), Cell(0, 0), Cell(3, 0)),
        total_cost=20.0,
    )
    report = make_report(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0)),
        ((0, 1), (1, 2), (2, 3)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.result.diagnostics.neighbor_policy == "4-neighbor"


def test_region_graph_guided_falls_back_when_graph_is_disconnected():
    grid = make_grid([[1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report(grid, (Cell(0, 0), Cell(2, 0)), (), connected=False)

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    assert outcome.report.selected_backend == "astar"
    assert outcome.report.fallback_reason == "region_graph_disconnected"
    assert outcome.report.segment_count == 0
    assert outcome.report.to_dict()["region_graph_candidate"]["status"] == "fallback"


def test_region_graph_guided_falls_back_when_candidate_is_not_better_than_baseline():
    grid = make_grid([[1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report(grid, (Cell(0, 0), Cell(2, 0)), ((0, 1),))

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert math.isclose(outcome.report.candidate_path_cost, baseline.total_cost)
    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "region_graph_candidate_not_better"
    assert outcome.report.segment_count == 1
    assert outcome.report.to_dict()["comparison"]["path_changed"] is False
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "sampled_candidate_equal_cost_no_quality_gain"
