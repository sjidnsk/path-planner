import math

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, NeighborPolicy, PlanDiagnostics, PlanRequest, PlanResult
from path_planner.platform import PlannerPlatformProfile
from path_planner.regions import ConvexRegion, RegionEdge, RegionGraph, RegionGraphReport
from path_planner.search import AStarPlanner, RegionGraphGuidedPlanner, build_planning_grid


def make_grid(cost, passable=None):
    cost_array = np.asarray(cost, dtype=float)
    passable_mask = np.ones(cost_array.shape, dtype=bool) if passable is None else np.asarray(passable, dtype=bool)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=1.0)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable_mask)


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
    assert sampled["fallback_reason"] == "target_component_disconnected"
    assert sampled["start_goal_anchoring"]["reachable_component_report"]["status"] == "disconnected"
    assert sampled["safety_checks"]["collision_free"] is False


def test_region_graph_guided_reports_goal_anchor_unconnected_blocker():
    grid = make_grid([[1.0, 1.0, 1.0]], passable=[[True, False, True]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions((make_region(0, Cell(0, 0), grid),), ())

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "anchor_component_disconnected"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "anchor_component_disconnected"
    assert sampled["start_goal_anchoring"]["start_region_id"] == 0
    assert sampled["start_goal_anchoring"]["goal_region_id"] == 1
    assert sampled["start_goal_anchoring"]["goal_anchor_region_added"] is True
    assert sampled["start_goal_anchoring"]["goal_anchor_region_connected"] is False
    assert sampled["start_goal_anchoring"]["goal_anchor_failure_reason"] == "anchor_component_disconnected"
    assert sampled["start_goal_anchoring"]["reachable_component_report"]["status"] == "disconnected"
    closure = sampled["start_goal_anchoring"]["anchor_connectivity_closure"]
    assert closure["reason_counts"]["safe_bridge_blocked"] > 0


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


def test_passable_goal_anchor_uses_component_proxy_when_raw_anchor_beyond_radius():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = AStarPlanner().plan(grid, request)
    report = make_report_from_regions((make_region(0, Cell(0, 0), grid),), ())

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.result.path_cells[-1] == Cell(2, 0)
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] is None
    terminal = sampled["terminal_adjustment_report"]
    assert terminal["status"] == "selected"
    assert terminal["reason_code"] == "proxy_goal_anchor_selected"
    assert terminal["proxy_goal_anchor_selected"] is True
    assert terminal["adjusted_goal_cell"] == [2, 0]
    assert terminal["original_goal_cell"] == [4, 0]
    anchoring = sampled["start_goal_anchoring"]
    assert anchoring["goal_cell"] == [2, 0]
    assert anchoring["requested_goal_cell"] == [4, 0]
    assert anchoring["goal_classification"] == "goal_outside_region_coverage"
    assert anchoring["goal_anchor_region_added"] is True
    assert anchoring["goal_anchor_region_connected"] is True
    assert anchoring["goal_anchor_failure_reason"] is None
    closure = anchoring["anchor_connectivity_closure"]
    assert closure["attempt_count"] > 0
    assert closure["connected_count"] > 0
    assert closure["reason_counts"]["safe_bridge_found"] > 0


def test_goal_anchor_uses_bounded_safe_bridge_to_reach_nearby_region():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0)),
        total_cost=20.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 0), grid),
            make_region(1, Cell(2, 0), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    anchoring = sampled["start_goal_anchoring"]
    assert sampled["fallback_reason"] is None
    assert sampled["region_sequence"] == [0, 1, 2]
    assert anchoring["goal_anchor_region_added"] is True
    assert anchoring["goal_anchor_region_connected"] is True
    assert anchoring["goal_anchor_failure_reason"] is None
    closure = anchoring["anchor_connectivity_closure"]
    assert closure["attempt_count"] > 0
    assert closure["connected_count"] >= 1
    assert closure["connection_kind_counts"]["anchor_region_safe_bridge"] >= 1
    assert closure["reason_counts"]["safe_bridge_found"] >= 1


def test_safe_bridge_cells_feed_bridge_aware_constrained_connector():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(3, 0), Cell(4, 0)),
        total_cost=20.0,
    )
    report = make_report_from_regions(
        (
            make_box_region(0, Cell(0, 0), Cell(1, 0), Cell(0, 0), grid),
            make_region(1, Cell(2, 0), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    rankings = sampled["candidate_rankings"]
    selected_ranking = next(item for item in rankings if item["status"] == "selected")
    assert selected_ranking["strategy"] == "bridge_aware_constrained_astar"
    assert selected_ranking["bridge_aware"] is True
    assert selected_ranking["bridge_connection_count"] >= 1
    assert selected_ranking["bridge_cell_count"] >= 3
    assert [3, 0] in selected_ranking["bridge_cells"]
    connector_attempts = [
        attempt
        for attempt in sampled["sample_attempts"]
        if attempt.get("kind") == "connector_attempt"
        and attempt.get("strategy") == "bridge_aware_constrained_astar"
    ]
    assert connector_attempts
    assert connector_attempts[0]["status"] == "available"
    assert connector_attempts[0]["bridge_connection_count"] >= 1
    assert [3, 0] in connector_attempts[0]["bridge_cells"]


def test_bridge_corridor_expansion_connects_safe_bridge_through_bounded_passable_gap():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 1.0, 1.0, 1.0],
        ],
        passable=[
            [True, True, True, True, True],
            [True, False, True, True, True],
        ],
    )
    request = PlanRequest(start=Cell(0, 1), goal=Cell(4, 1), prevent_corner_cutting=False)
    baseline = make_baseline_result(
        grid,
        (Cell(0, 1), Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(2, 1), Cell(3, 1), Cell(4, 1)),
        total_cost=20.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 1), grid),
            make_region(1, Cell(2, 1), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert Cell(1, 0) in outcome.result.path_cells
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    selected_ranking = next(item for item in sampled["candidate_rankings"] if item["status"] == "selected")
    assert selected_ranking["strategy"] == "bridge_corridor_constrained_astar"
    assert selected_ranking["bridge_corridor_expanded"] is True
    assert selected_ranking["bridge_corridor_radius_cells"] == 1
    assert selected_ranking["bridge_corridor_added_cell_count"] > 0
    assert selected_ranking["bridge_corridor_start_goal_connected"] is True
    assert selected_ranking["bridge_corridor_failure_reason"] is None
    rejected_bridge = next(
        item
        for item in sampled["candidate_rankings"]
        if item["strategy"] == "bridge_aware_constrained_astar"
    )
    assert rejected_bridge["fallback_reason"] == "bridge_aware_connector_path_unavailable"
    corridor_attempts = [
        attempt
        for attempt in sampled["sample_attempts"]
        if attempt.get("kind") == "connector_attempt"
        and attempt.get("strategy") == "bridge_corridor_constrained_astar"
    ]
    assert corridor_attempts
    assert corridor_attempts[0]["status"] == "available"
    assert corridor_attempts[0]["bridge_corridor_start_goal_connected"] is True


def test_bridge_corridor_reports_blocked_when_real_passable_mask_has_no_connector_path():
    grid = make_grid(
        [[1.0, 9.0, 1.0, 1.0, 1.0]],
        passable=[[True, False, True, True, True]],
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(4, 0))
    baseline = make_baseline_result(grid, (Cell(0, 0), Cell(4, 0)), total_cost=20.0)
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 0), grid),
            make_region(1, Cell(2, 0), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "target_component_disconnected"
    component = sampled["start_goal_anchoring"]["reachable_component_report"]
    assert component["status"] == "disconnected"
    assert component["reason"] == "target_component_disconnected"
    component_filter = next(
        item
        for item in sampled["candidate_rankings"]
        if item["strategy"] == "reachable_component_filter"
    )
    assert component_filter["status"] == "rejected"
    assert component_filter["fallback_reason"] == "target_component_disconnected"
    assert not any(
        item["strategy"] == "bridge_corridor_constrained_astar"
        for item in sampled["candidate_rankings"]
    )


def test_platform_anchor_regions_use_bounded_safe_neighborhood_for_connectivity():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))
    request = PlanRequest(start=Cell(0, 0), goal=Cell(6, 0))
    baseline = AStarPlanner().plan(planning_grid, request)
    report = make_report_from_regions(
        (
            make_box_region(0, Cell(2, 0), Cell(4, 0), Cell(3, 0), grid),
        ),
        (),
    )

    outcome = RegionGraphGuidedPlanner().plan(
        planning_grid,
        request,
        baseline_result=baseline,
        region_graph_report=report,
    )

    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    anchoring = sampled["start_goal_anchoring"]
    assert anchoring["region_sequence_found"] is True
    assert anchoring["start_anchor_region_added"] is True
    assert anchoring["goal_anchor_region_added"] is True
    assert anchoring["start_anchor_region_connected"] is True
    assert anchoring["goal_anchor_region_connected"] is True
    assert sampled["fallback_reason"] == "sampled_candidate_path_duplicate"
    assert sampled["candidate_comparison"]["path_duplicate_with_baseline"] is True


def test_footprint_unsafe_goal_gets_bounded_terminal_adjustment():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0, 5.0, 5.0],
            [1.0, 1.0, 1.0, 5.0, 5.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 5.0, 5.0],
            [1.0, 1.0, 1.0, 5.0, 5.0],
        ],
        passable=[
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
    )
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))
    request = PlanRequest(start=Cell(0, 2), goal=Cell(3, 2))
    baseline = AStarPlanner().plan(planning_grid, request)
    report = make_report_from_regions(
        (
            make_box_region(0, Cell(0, 2), Cell(2, 2), Cell(1, 2), grid),
        ),
        (),
    )

    outcome = RegionGraphGuidedPlanner().plan(
        planning_grid,
        request,
        baseline_result=baseline,
        region_graph_report=report,
    )

    assert baseline.success is False
    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.result.path_cells[-1] == Cell(2, 2)
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    terminal = sampled["terminal_adjustment_report"]
    assert terminal["schema_version"] == "terminal_adjustment_report/v1"
    assert terminal["status"] == "selected"
    assert terminal["target_adjusted"] is True
    assert terminal["reason"] == "goal_footprint_unsafe"
    assert terminal["original_goal_cell"] == [3, 2]
    assert terminal["adjusted_goal_cell"] == [2, 2]
    assert terminal["candidate_count"] > 0
    assert sampled["start_goal_anchoring"]["goal_cell"] == [2, 2]
    assert sampled["start_goal_anchoring"]["requested_goal_cell"] == [3, 2]
    assert sampled["start_goal_anchoring"]["goal_classification"] == "covered"


def test_terminal_adjustment_prefers_reachable_component_over_closer_disconnected_candidate():
    passable = np.ones((5, 7), dtype=bool)
    passable[:, 3] = False
    grid = make_grid(np.ones((5, 7), dtype=float), passable=passable)
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))
    request = PlanRequest(start=Cell(0, 2), goal=Cell(4, 2))
    baseline = AStarPlanner().plan(planning_grid, request)
    report = make_report_from_regions(
        (
            make_box_region(0, Cell(0, 2), Cell(1, 2), Cell(0, 2), grid),
        ),
        (),
    )

    outcome = RegionGraphGuidedPlanner().plan(
        planning_grid,
        request,
        baseline_result=baseline,
        region_graph_report=report,
    )

    assert baseline.success is False
    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.result.path_cells[-1] == Cell(1, 2)
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    terminal = sampled["terminal_adjustment_report"]
    assert terminal["status"] == "selected"
    assert terminal["reason_code"] == "terminal_adjustment_selected"
    assert terminal["adjusted_goal_cell"] == [1, 2]
    assert terminal["reachable_component_replacement_selected"] is True
    assert terminal["reachable_candidate_count"] == 1
    assert terminal["candidate_count"] > terminal["reachable_candidate_count"]
    component = terminal["reachable_component_report"]
    assert component["status"] == "adjusted_connected"
    assert component["reason"] == "reachable_component_replacement_selected"
    assert component["start_component_id"] == component["adjusted_goal_component_id"]
    assert component["raw_goal_component_id"] is None
    assert sampled["start_goal_anchoring"]["reachable_component_report"]["status"] == "adjusted_connected"


def test_terminal_adjustment_rescues_reachable_terminal_beyond_default_radius():
    passable = np.ones((5, 8), dtype=bool)
    passable[:, 4] = False
    grid = make_grid(np.ones((5, 8), dtype=float), passable=passable)
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=2.0))
    request = PlanRequest(start=Cell(0, 2), goal=Cell(6, 2))
    baseline = AStarPlanner().plan(planning_grid, request)
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 2), grid),
        ),
        (),
    )

    outcome = RegionGraphGuidedPlanner().plan(
        planning_grid,
        request,
        baseline_result=baseline,
        region_graph_report=report,
    )

    assert baseline.success is False
    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    assert outcome.result.path_cells[-1] == Cell(1, 2)
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    terminal = sampled["terminal_adjustment_report"]
    assert terminal["status"] == "selected"
    assert terminal["reason_code"] == "reachable_terminal_selected_by_component_projection"
    assert terminal["target_adjusted"] is True
    assert terminal["adjusted_goal_cell"] == [1, 2]
    assert terminal["candidate_count"] > 0
    assert terminal["reachable_candidate_count"] == 0
    assert terminal["reachable_terminal_rescue_used"] is True
    assert terminal["distance_cells"] > terminal["max_radius_cells"]
    assert terminal["reachable_component_report"]["reason"] == "reachable_terminal_selected_by_component_projection"


def test_terminal_adjustment_reports_distance_budget_when_component_projection_too_far():
    passable = np.ones((5, 30), dtype=bool)
    passable[:, 4] = False
    passable[2, 23] = False
    grid = make_grid(np.ones((5, 30), dtype=float), passable=passable)
    planning_grid = build_planning_grid(grid, platform_profile=make_profile(footprint_radius_m=1.0))
    request = PlanRequest(start=Cell(0, 2), goal=Cell(24, 2))
    baseline = AStarPlanner().plan(planning_grid, request)
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 2), grid),
        ),
        (),
    )

    outcome = RegionGraphGuidedPlanner().plan(
        planning_grid,
        request,
        baseline_result=baseline,
        region_graph_report=report,
    )

    assert baseline.success is False
    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    assert outcome.report.fallback_reason == "reachable_terminal_distance_budget_exceeded"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "reachable_terminal_distance_budget_exceeded"
    terminal = sampled["terminal_adjustment_report"]
    assert terminal["status"] == "unavailable"
    assert terminal["reason_code"] == "reachable_terminal_distance_budget_exceeded"
    assert terminal["candidate_count"] > 0
    assert terminal["reachable_candidate_count"] == 0
    assert terminal["rescue_candidate_count"] == 0
    assert terminal["reachable_component_report"]["reason"] == "reachable_terminal_distance_budget_exceeded"


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


def test_failed_constrained_connector_with_higher_cost_region_sequence_is_cost_dominated():
    grid = make_grid(
        [
            [1.0, 9.0, 9.0, 9.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 1), goal=Cell(4, 1))
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

    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "region_sequence_cost_dominated"
    assert sampled["candidate_rankings"][0]["strategy"] == "cost_aware_constrained_astar"
    assert sampled["candidate_rankings"][0]["fallback_reason"] == "constrained_connector_failed"


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


def test_equal_cost_connector_selects_execution_quality_tie_break():
    grid = make_grid([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    request = PlanRequest(start=Cell(0, 1), goal=Cell(4, 1), prevent_corner_cutting=False)
    baseline = make_baseline_result(
        grid,
        (
            Cell(0, 1),
            Cell(1, 1),
            Cell(1, 0),
            Cell(2, 0),
            Cell(3, 0),
            Cell(3, 1),
            Cell(4, 1),
        ),
        total_cost=4.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 1), grid),
            make_box_region(1, Cell(1, 1), Cell(3, 1), Cell(2, 1), grid),
            make_region(2, Cell(4, 1), grid),
        ),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.report.status == "selected"
    assert outcome.report.selected_backend == "sampled_region_path"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["execution_tie_break"]["reason"] == "execution_tie_break_improved"
    assert sampled["candidate_comparison"]["turn_count_delta"] < 0
    assert sampled["candidate_rankings"][0]["selection_reason"] == "execution_tie_break_improved"


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
    assert sampled["fallback_reason"] == "sampled_candidate_path_duplicate"
    assert sampled["execution_tie_break"]["reason"] == "execution_tie_break_no_alternative"
    comparison = sampled["candidate_comparison"]
    assert comparison["path_duplicate_with_baseline"] is True
    assert comparison["baseline_path_overlap_ratio"] == 1.0
    assert comparison["benefit_surface_present"] is False
    assert comparison["complexity_reason"] == "sampled_candidate_path_duplicate"
    ranking = next(item for item in sampled["candidate_rankings"] if item.get("rank") == 1)
    assert ranking["path_duplicate_with_baseline"] is True
    assert ranking["baseline_path_overlap_ratio"] == 1.0
    assert ranking["complexity_reason"] == "sampled_candidate_path_duplicate"


def test_sampled_candidate_reports_no_quality_gain_when_equal_cost_path_is_worse():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0), prevent_corner_cutting=False)
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(1, 0), Cell(2, 0)),
        total_cost=math.sqrt(2.0) * 2.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 0), grid),
            make_region(1, Cell(1, 1), grid),
            make_region(2, Cell(2, 0), grid),
        ),
        ((0, 1), (1, 2)),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "sampled_candidate_no_quality_gain"
    comparison = sampled["candidate_comparison"]
    assert comparison["candidate_cost_delta"] == 0.0
    assert comparison["path_duplicate_with_baseline"] is False
    assert comparison["baseline_path_overlap_ratio"] < 1.0
    assert comparison["path_length_delta_m"] > 0.0
    assert comparison["benefit_surface_present"] is True
    assert comparison["complexity_reason"] == "sampled_candidate_no_quality_gain"


def test_constrained_connector_failure_is_not_reported_as_cost_dominated():
    grid = make_grid(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        passable=[
            [True, False, True],
            [True, True, True],
        ],
    )
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 0))
    baseline = make_baseline_result(
        grid,
        (Cell(0, 0), Cell(0, 1), Cell(1, 1), Cell(2, 1), Cell(2, 0)),
        total_cost=2.0,
    )
    report = make_report_from_regions(
        (
            make_region(0, Cell(0, 0), grid),
            make_region(1, Cell(2, 0), grid),
        ),
        ((0, 1),),
    )

    outcome = RegionGraphGuidedPlanner().plan(grid, request, baseline_result=baseline, region_graph_report=report)

    assert outcome.result is baseline
    assert outcome.report.status == "fallback"
    sampled = outcome.report.to_dict()["sampled_region_path_report"]
    assert sampled["fallback_reason"] == "constrained_connector_failed"
    assert sampled["fallback_reason"] != "region_sequence_cost_dominated"
    assert any(
        item["strategy"] == "cost_aware_constrained_astar"
        and item["fallback_reason"] == "constrained_connector_failed"
        for item in sampled["candidate_rankings"]
    )
