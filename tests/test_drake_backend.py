import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from path_planner.adapters import route_result_to_json_dict
from path_planner.core import Cell, CostGrid, GridSpec, PlanDiagnostics, PlanRequest, PlanResult, WorldPoint
from path_planner.drake_backend import (
    ConvexRegionSequenceItem,
    ConvexRegionSequenceReport,
    GcsTrajectoryReport,
    IrisRegion,
    IrisRegionReport,
    build_convex_region_sequence_report,
    build_gcs_control_point_trajectory_report,
    build_gcs_curvature_constrained_candidate_report,
    build_gcs_geometric_candidate_report,
    build_gcs_motion_feasibility_report,
    build_gcs_trajectory_report,
    build_workspace_iris_region_report,
)
from path_planner.drake_backend.gcs_diagnostics import build_direction_cone_constraint_summary
from path_planner.drake_backend.gcs_control_point_trajectory import GcsControlPointSolverConfig
from path_planner.drake_backend.gcs_scenario_matrix import build_gcs_scenario_matrix_summary
from path_planner.postprocess import build_corridor, run_postprocess
from path_planner.postprocess.models import CorridorResult, CorridorSection
from path_planner.search import AStarPlanner

DEMO_MAP_CORRIDOR = str(Path(__file__).resolve().parents[1] / "examples" / "demo_map_corridor.json")


def make_grid(mask, resolution=1.0):
    passable = np.asarray(mask, dtype=bool)
    spec = GridSpec(width=passable.shape[1], height=passable.shape[0], resolution=resolution)
    return CostGrid(spec=spec, cost=np.ones(passable.shape), passable_mask=passable)


def test_workspace_iris_backend_unavailable_returns_fallback_report(monkeypatch):
    import path_planner.drake_backend.iris as iris_backend

    grid = make_grid(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    corridor = build_corridor(grid, (Cell(0, 1), Cell(1, 1), Cell(2, 1)), radius_cells=1)

    def unavailable():
        raise ImportError("simulated missing pydrake")

    monkeypatch.setattr(iris_backend, "_load_geometry_optimization", unavailable)

    report = build_workspace_iris_region_report(grid, corridor)
    payload = report.to_dict()

    assert payload["backend"] == "workspace_iris"
    assert payload["status"] == "fallback"
    assert payload["failure_status"] == "backend_unavailable"
    assert "simulated missing pydrake" in payload["failure_reason"]
    assert payload["fallback_used"] is True
    assert payload["seed_source"] == "postprocess_corridor_centers"
    assert payload["domain_source"] == "postprocess_corridor_grid_box"
    assert payload["obstacle_source"] == "blocked_cell_box"
    assert payload["region_count"] == len(corridor.sections)
    assert payload["regions"][0]["source"] == "grid_box"
    assert payload["regions"][0]["hpolyhedron"]["A"]
    assert payload["regions"][0]["hpolyhedron"]["b"]


def test_iris_region_report_is_optional_route_json_field():
    grid = make_grid(np.ones((3, 4), dtype=bool))
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    report = build_workspace_iris_region_report(grid, corridor)

    without_report = route_result_to_json_dict(plan, grid.spec)
    with_report = route_result_to_json_dict(plan, grid.spec, iris_region_report=report)

    assert "iris_region_report" not in without_report
    assert with_report["trajectory_kind"] == "geometric_path"
    assert with_report["reachable"] is True
    assert with_report["iris_region_report"]["backend"] == "workspace_iris"
    assert with_report["iris_region_report"]["region_count"] == len(corridor.sections)


def test_convex_region_sequence_fallback_box_covers_astar_path_without_blocked_cells():
    grid = make_grid(
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        resolution=1.0,
    )
    request = PlanRequest(start=Cell(0, 2), goal=Cell(4, 2))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)

    report = build_convex_region_sequence_report(grid, plan, corridor)
    payload = report.to_route_fields()

    assert payload["convex_region_backend"] == "fallback_box"
    assert payload["convex_region_fallback_used"] is True
    assert payload["convex_region_count"] > 0
    assert payload["convex_region_start_contained"] is True
    assert payload["convex_region_goal_contained"] is True
    assert payload["convex_region_blocked_cell_violation_count"] == 0
    assert payload["convex_region_coverage_status"] == "covered"
    assert payload["gcs_ready"] is True
    assert payload["gcs_ready_reason"] == "convex_region_sequence_ready"
    assert (
        payload["convex_region_adjacent_overlap_count"] + payload["convex_region_portal_count"]
        >= payload["convex_region_count"] - 1
    )
    covered_indices = {
        index
        for region in payload["convex_region_sequence"]
        for index in region["covered_path_indices"]
    }
    assert covered_indices == set(range(len(plan.path_cells)))
    for region in payload["convex_region_sequence"]:
        assert region["backend"] == "fallback_box"
        assert region["source"] == "fallback_box"
        assert region["hpolyhedron"]["A"]
        assert region["hpolyhedron"]["b"]
        bounds = region["bounds"]
        assert not (
            bounds["min"][0] <= 2 <= bounds["max"][0]
            and bounds["min"][1] <= 2 <= bounds["max"][1]
        )


def test_convex_region_sequence_can_use_valid_workspace_iris_regions():
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    iris_regions = tuple(_iris_region_for_cell(index, cell, grid) for index, cell in enumerate(plan.path_cells))
    iris_report = IrisRegionReport(
        backend="workspace_iris",
        status="ok",
        seed_source="astar_path_cells",
        domain_source="astar_corridor_box",
        obstacle_source="blocked_cell_box",
        regions=iris_regions,
        obstacle_count=0,
        validation_status="valid",
        fallback_used=False,
    )

    report = build_convex_region_sequence_report(grid, plan, corridor, iris_region_report=iris_report)
    payload = report.to_route_fields()

    assert payload["convex_region_backend"] == "workspace_iris"
    assert payload["convex_region_fallback_used"] is False
    assert payload["convex_region_count"] == len(plan.path_cells)
    assert payload["gcs_ready"] is True
    assert {region["source"] for region in payload["convex_region_sequence"]} == {"iris"}
    assert {region["backend"] for region in payload["convex_region_sequence"]} == {"workspace_iris"}


def test_convex_region_sequence_is_optional_route_json_field_without_changing_route_semantics():
    grid = make_grid(np.ones((3, 4), dtype=bool))
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    report = build_convex_region_sequence_report(grid, plan, corridor)

    payload = route_result_to_json_dict(plan, grid.spec, convex_region_sequence_report=report)

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["convex_region_count"] > 0
    assert payload["convex_region_sequence"]
    assert payload["gcs_ready"] is True


def test_gcs_trajectory_report_solves_smoke_path_from_convex_region_sequence():
    pytest.importorskip("pydrake")
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    report = build_gcs_trajectory_report(grid, convex_report, sample_count=7)
    payload = report.to_route_fields()

    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_trajectory_attempted"] is True
    assert payload["gcs_trajectory_success"] is True
    assert payload["gcs_trajectory_backend"] == "pydrake_direction_cone_program"
    assert payload["gcs_trajectory_reason"] == "direction_cone_solution_found"
    assert payload["gcs_trajectory_sample_count"] == 7
    assert payload["gcs_trajectory_collision_count"] == 0
    assert payload["gcs_trajectory_path_length"] > 0.0
    assert payload["gcs_trajectory_region_count"] == convex_report.region_count
    assert len(payload["gcs_trajectory_sampled_points"]) == 7
    assert payload["gcs_trajectory_constraint_summary"]["constraint_model"] == "direction_cone"
    assert payload["gcs_trajectory_constraint_summary"]["schema_version"] == "gcs_direction_cone_constraint/v1"
    assert payload["gcs_trajectory_constraint_summary"]["attempted"] is True
    assert payload["gcs_trajectory_constraint_summary"]["evaluated"] is True
    assert payload["gcs_trajectory_constraint_summary"]["backend_enforced"] is True
    assert payload["gcs_trajectory_constraint_summary"]["enforcing_backend"] == "pydrake_mathematical_program"
    assert payload["gcs_trajectory_constraint_summary"]["fallback_reason"] is None
    assert payload["gcs_trajectory_constraint_summary"]["solver_constraint_count"] > 0
    assert payload["gcs_trajectory_constraint_summary"]["violation_count"] == 0
    assert payload["gcs_trajectory_constraint_summary"]["eta"] > 0.0
    assert payload["gcs_trajectory_constraint_summary"]["rho_min"] is not None
    assert payload["gcs_trajectory_constraint_summary"]["parameter_count"] > 0
    assert payload["gcs_trajectory_constraint_summary"]["parameters"][0]["tangent"]
    assert payload["gcs_trajectory_constraint_summary"]["parameters"][0]["normal"]
    assert payload["gcs_trajectory_cost_summary"]["schema_version"] == "gcs_cost_summary/v1"
    assert payload["gcs_trajectory_cost_summary"]["path_length"] == pytest.approx(
        payload["gcs_trajectory_path_length"]
    )
    assert payload["gcs_trajectory_cost_summary"]["terrain_path_cost"] > 0.0
    assert payload["gcs_trajectory_cost_summary"]["high_cost_exposure"] == 0.0
    assert payload["gcs_trajectory_cost_summary"]["smoothness_proxy"] >= 0.0


def test_gcs_trajectory_report_classifies_sample_collision():
    pytest.importorskip("pydrake")
    grid = make_grid(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ],
        resolution=1.0,
    )
    convex_report = _unsafe_two_region_report(grid)

    report = build_gcs_trajectory_report(grid, convex_report, sample_count=9)
    payload = report.to_route_fields()

    assert payload["gcs_trajectory_attempted"] is True
    assert payload["gcs_trajectory_success"] is False
    assert payload["gcs_trajectory_reason"] == "sampled_trajectory_collision"
    assert payload["gcs_trajectory_collision_count"] > 0


def test_direction_cone_summary_reports_portal_support_and_rho_parameters():
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)
    points = tuple(region.seed_world for region in convex_report.regions)

    summary = build_direction_cone_constraint_summary(points, convex_report.regions)

    assert summary["evaluated"] is True
    assert summary["portal_width_min_m"] is not None
    assert summary["support_width_min_m"] is not None
    assert summary["rho_lower_bound_min_m"] is not None
    assert summary["rho_source_counts"]["seed_distance_portal_support_min"] > 0
    assert summary["constraint_tightness_min"] > 1.0
    first_parameter = summary["parameters"][0]
    assert first_parameter["portal_width_m"] is not None
    assert first_parameter["support_width_m"] is not None
    assert first_parameter["rho_lower_bound_m"] > 0.0
    assert first_parameter["rho_source"] == "seed_distance_portal_support_min"
    assert first_parameter["constraint_tightness"] > 1.0


def test_direction_cone_summary_flags_degenerate_portal_geometry():
    grid = make_grid(np.ones((2, 2), dtype=bool), resolution=1.0)
    first = _convex_region_for_bounds(
        grid,
        region_id=0,
        seed=Cell(0, 0),
        min_cell=Cell(0, 0),
        max_cell=Cell(0, 0),
    )
    second = _convex_region_for_bounds(
        grid,
        region_id=1,
        seed=Cell(1, 1),
        min_cell=Cell(1, 1),
        max_cell=Cell(1, 1),
    )

    summary = build_direction_cone_constraint_summary(
        (first.seed_world, second.seed_world),
        (first, second),
    )

    assert summary["evaluated"] is True
    assert summary["portal_width_min_m"] == 0.0
    assert "degenerate_portal_width" in summary["risk_flags"]
    assert summary["parameters"][0]["portal_width_m"] == 0.0
    assert "degenerate_portal_width" in summary["parameters"][0]["risk_flags"]


def test_gcs_trajectory_report_handles_unavailable_pydrake(monkeypatch):
    import path_planner.drake_backend.gcs_trajectory as gcs_backend

    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    def unavailable():
        raise ImportError("simulated missing pydrake")

    monkeypatch.setattr(gcs_backend, "_load_gcs_dependencies", unavailable)

    payload = build_gcs_trajectory_report(grid, convex_report).to_route_fields()

    assert payload["gcs_trajectory_attempted"] is False
    assert payload["gcs_trajectory_success"] is False
    assert payload["gcs_trajectory_reason"] == "pydrake_unavailable"
    assert "simulated missing pydrake" in payload["gcs_trajectory_result_status"]
    assert payload["gcs_trajectory_constraint_summary"]["constraint_model"] == "direction_cone"
    assert payload["gcs_trajectory_constraint_summary"]["evaluated"] is False
    assert payload["gcs_trajectory_cost_summary"]["schema_version"] == "gcs_cost_summary/v1"
    assert payload["gcs_trajectory_cost_summary"]["terrain_path_cost"] is None


def test_gcs_trajectory_report_classifies_solver_infeasible(monkeypatch):
    pytest.importorskip("pydrake")
    import path_planner.drake_backend.gcs_trajectory as gcs_backend

    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    def infeasible(*args, **kwargs):
        raise RuntimeError("solver reported infeasible constraints")

    monkeypatch.setattr(gcs_backend, "_solve_gcs_path", infeasible)

    payload = build_gcs_trajectory_report(grid, convex_report).to_route_fields()

    assert payload["gcs_trajectory_attempted"] is True
    assert payload["gcs_trajectory_success"] is False
    assert payload["gcs_trajectory_reason"] == "solver_infeasible"
    assert "infeasible" in payload["gcs_trajectory_result_status"]


def test_gcs_trajectory_report_is_optional_route_json_field_without_changing_route_semantics():
    pytest.importorskip("pydrake")
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)
    gcs_report = build_gcs_trajectory_report(grid, convex_report, sample_count=5)

    payload = route_result_to_json_dict(
        plan,
        grid.spec,
        convex_region_sequence_report=convex_report,
        gcs_trajectory_report=gcs_report,
    )

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_attempted"] is True
    assert payload["gcs_trajectory_success"] is True
    assert payload["gcs_trajectory_sample_count"] == 5


def test_gcs_control_point_report_solves_derivative_direction_cone_path():
    pytest.importorskip("pydrake")
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    report = build_gcs_control_point_trajectory_report(grid, convex_report, sample_count=7)
    payload = report.to_route_fields()
    summary = payload["gcs_trajectory_constraint_summary"]

    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_trajectory_attempted"] is True
    assert payload["gcs_trajectory_success"] is True
    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_reason"] == "control_point_direction_cone_solution_found"
    assert payload["gcs_trajectory_sample_count"] == 7
    assert payload["gcs_trajectory_collision_count"] == 0
    assert payload["gcs_trajectory_region_count"] == convex_report.region_count
    assert summary["constraint_model"] == "direction_cone"
    assert summary["trajectory_parameterization"] == "control_point_derivative_proxy"
    assert summary["backend_enforced"] is True
    assert summary["enforcing_backend"] == "pydrake_control_point_mathematical_program"
    assert summary["solver_constraint_count"] > 0
    assert summary["control_point_count"] == convex_report.region_count
    assert summary["derivative_constraint_count"] == 3 * (convex_report.region_count - 1)
    assert summary["control_point_region_containment_count"] >= convex_report.region_count
    assert summary["derivative_proxy"] == "successive_control_point_difference"
    assert summary["objective_terms"] == [
        "segment_length_quadratic",
        "low_cost_anchor_quadratic",
        "control_point_terrain_anchor_quadratic",
        "control_point_second_difference_quadratic",
    ]
    assert payload["gcs_trajectory_cost_summary"]["schema_version"] == "gcs_cost_summary/v1"
    assert payload["gcs_trajectory_cost_summary"]["terrain_path_cost"] > 0.0


@pytest.mark.drake
def test_gcs_control_point_report_exposes_terrain_cost_objective_proxy():
    pytest.importorskip("pydrake")
    grid = _candidate_grid(
        [
            [1.0, 8.0, 1.0, 1.0],
            [1.0, 6.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 2), goal=Cell(3, 0))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    report = build_gcs_control_point_trajectory_report(grid, convex_report, sample_count=9)
    payload = report.to_route_fields()
    constraint_summary = payload["gcs_trajectory_constraint_summary"]
    cost_summary = payload["gcs_trajectory_cost_summary"]

    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_success"] is True
    assert "control_point_terrain_anchor_quadratic" in constraint_summary["objective_terms"]
    assert constraint_summary["objective_term_weights"]["control_point_terrain_anchor_quadratic"] > 0.0
    assert constraint_summary["objective_term_weights"]["control_point_terrain_anchor_quadratic"] == 0.05
    assert constraint_summary["objective_term_weights"]["control_point_second_difference_quadratic"] == 0.2
    assert constraint_summary["max_allowed_direction_error_deg"] == 45.0
    assert constraint_summary["terrain_objective_source"] == (
        "region_inverse_cost_weighted_passable_cell_centroid"
    )
    assert cost_summary["terrain_objective_source"] == (
        "region_inverse_cost_weighted_passable_cell_centroid"
    )
    assert cost_summary["terrain_objective_weight"] > 0.0
    assert cost_summary["terrain_objective_weight"] == 0.05
    assert cost_summary["terrain_objective_anchor_count"] == convex_report.region_count
    assert cost_summary["control_point_terrain_cost"] > 0.0
    assert cost_summary["sampled_terrain_cost"] == cost_summary["terrain_path_cost"]
    assert cost_summary["terrain_objective_boundary"] == "proxy_not_continuous_field_integral"
    assert "control_point_high_cost_exposure_proxy_quadratic" not in constraint_summary["objective_terms"]
    assert "high_cost_exposure_objective_weight" not in cost_summary


def test_gcs_control_point_config_rejects_negative_high_cost_exposure_weight():
    with pytest.raises(ValueError, match="high_cost_exposure_weight must be non-negative"):
        GcsControlPointSolverConfig(high_cost_exposure_weight=-0.1)


@pytest.mark.drake
def test_gcs_control_point_report_accepts_explicit_calibration_config():
    pytest.importorskip("pydrake")
    grid = _candidate_grid(
        [
            [1.0, 8.0, 1.0, 1.0],
            [1.0, 6.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    request = PlanRequest(start=Cell(0, 2), goal=Cell(3, 0))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    report = build_gcs_control_point_trajectory_report(
        grid,
        convex_report,
        sample_count=9,
        config=GcsControlPointSolverConfig(
            terrain_objective_weight=0.08,
            second_difference_weight=0.35,
            high_cost_exposure_weight=0.45,
            direction_cone_max_error_deg=35.0,
            direction_cone_rho_floor_m=0.04,
            direction_cone_seed_rho_ratio=0.08,
        ),
    )
    payload = report.to_route_fields()
    constraint_summary = payload["gcs_trajectory_constraint_summary"]
    cost_summary = payload["gcs_trajectory_cost_summary"]

    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert constraint_summary["objective_term_weights"]["control_point_terrain_anchor_quadratic"] == 0.08
    assert constraint_summary["objective_term_weights"]["control_point_second_difference_quadratic"] == 0.35
    assert constraint_summary["objective_term_weights"]["control_point_high_cost_exposure_proxy_quadratic"] == 0.45
    assert "control_point_high_cost_exposure_proxy_quadratic" in constraint_summary["objective_terms"]
    assert constraint_summary["terrain_objective_weight"] == 0.08
    assert constraint_summary["high_cost_exposure_objective_weight"] == 0.45
    assert constraint_summary["high_cost_exposure_proxy_source"] == "region_high_cost_exposure_proxy"
    assert cost_summary["terrain_objective_weight"] == 0.08
    assert cost_summary["high_cost_exposure_objective_weight"] == 0.45
    assert cost_summary["high_cost_exposure_proxy_source"] == "region_high_cost_exposure_proxy"
    assert cost_summary["high_cost_exposure_proxy_cost"] is not None
    assert cost_summary["high_cost_exposure_proxy_boundary"] == "proxy_not_continuous_field_integral"
    assert constraint_summary["max_allowed_direction_error_deg"] == 35.0
    assert constraint_summary["rho_lower_bound_min_m"] >= 0.04
    assert constraint_summary["parameters"][0]["rho_seed_distance_m"] >= 0.04


def test_gcs_control_point_report_handles_unavailable_pydrake(monkeypatch):
    import path_planner.drake_backend.gcs_control_point_trajectory as control_backend

    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)

    def unavailable():
        raise ImportError("simulated missing pydrake")

    monkeypatch.setattr(control_backend, "_load_gcs_dependencies", unavailable)

    payload = build_gcs_control_point_trajectory_report(grid, convex_report).to_route_fields()

    assert payload["gcs_trajectory_attempted"] is False
    assert payload["gcs_trajectory_success"] is False
    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_reason"] == "pydrake_unavailable"
    assert "simulated missing pydrake" in payload["gcs_trajectory_result_status"]
    assert payload["gcs_trajectory_constraint_summary"]["evaluated"] is False
    assert payload["gcs_trajectory_constraint_summary"]["trajectory_parameterization"] == (
        "control_point_derivative_proxy"
    )
    assert payload["gcs_trajectory_cost_summary"]["terrain_objective_source"] == "not_evaluated"
    assert payload["gcs_trajectory_cost_summary"]["sampled_terrain_cost"] is None


def test_gcs_control_point_report_is_optional_route_json_field_without_changing_route_semantics():
    pytest.importorskip("pydrake")
    grid = make_grid(np.ones((3, 4), dtype=bool), resolution=1.0)
    request = PlanRequest(start=Cell(0, 1), goal=Cell(3, 1))
    plan = AStarPlanner().plan(grid, request)
    corridor = build_corridor(grid, plan.path_cells, radius_cells=1)
    convex_report = build_convex_region_sequence_report(grid, plan, corridor)
    gcs_report = build_gcs_control_point_trajectory_report(grid, convex_report, sample_count=5)

    payload = route_result_to_json_dict(
        plan,
        grid.spec,
        convex_region_sequence_report=convex_report,
        gcs_trajectory_report=gcs_report,
    )

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_constraint_summary"]["trajectory_parameterization"] == (
        "control_point_derivative_proxy"
    )


def test_gcs_geometric_candidate_selects_lower_cost_collision_free_sampled_path():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5))
    )

    report = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report)
    payload = report.to_route_fields()

    assert payload["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
    assert payload["gcs_candidate_attempted"] is True
    assert payload["gcs_candidate_available"] is True
    assert payload["gcs_candidate_selected"] is True
    assert payload["gcs_candidate_selection_reason"] == "gcs_candidate_quality_improved"
    assert payload["gcs_candidate_fallback_reason"] is None
    assert payload["gcs_candidate_collision_count"] == 0
    assert payload["gcs_candidate_path_cost"] < plan.total_cost
    assert payload["gcs_candidate_cost_delta_vs_baseline"] < 0.0
    assert payload["gcs_candidate_baseline_overlap_ratio"] == 0.0
    assert payload["gcs_candidate_constraint_summary"]["direction_cone"]["evaluated"] is True
    assert payload["gcs_candidate_constraint_summary"]["direction_cone"]["violation_count"] == 0
    assert payload["gcs_candidate_cost_summary"]["schema_version"] == "gcs_candidate_cost_summary/v1"
    assert payload["gcs_candidate_cost_summary"]["terrain_path_cost"] == payload["gcs_candidate_path_cost"]
    assert payload["gcs_candidate_cost_summary"]["cost_delta_vs_baseline"] < 0.0
    assert payload["gcs_candidate_cost_summary"]["smoothness_proxy"] >= 0.0
    assert payload["gcs_candidate_cost_summary"]["candidate_decision"] == "selected"
    assert payload["gcs_candidate_cost_summary"]["decision_reason"] == "gcs_candidate_quality_improved"
    assert payload["gcs_candidate_cost_summary"]["quality_gate"]["baseline_delta_improved"] is True
    assert payload["gcs_candidate_cost_summary"]["terrain_cost_source"] == "sampled_unique_passable_cells"


def test_gcs_geometric_candidate_blocks_when_direction_cone_was_not_evaluated():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5)),
        direction_cone_evaluated=False,
    )

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_attempted"] is True
    assert payload["gcs_candidate_available"] is False
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "direction_cone_not_evaluated"
    assert payload["gcs_candidate_constraint_summary"]["direction_cone"]["evaluated"] is False


def test_gcs_geometric_candidate_blocks_when_direction_cone_was_not_backend_enforced():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5)),
        direction_cone_backend_enforced=False,
    )

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_available"] is False
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "direction_cone_not_backend_enforced"
    assert payload["gcs_candidate_constraint_summary"]["direction_cone"]["backend_enforced"] is False


def test_gcs_geometric_candidate_blocks_direction_cone_violation_without_replacing_route():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5)),
        direction_cone_violation_count=1,
    )

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_available"] is False
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "direction_cone_constraint_violation"
    assert payload["gcs_candidate_constraint_summary"]["direction_cone"]["violation_count"] == 1


def test_gcs_geometric_candidate_blocks_motion_infeasible_path_without_replacing_route():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(1.5, 1.5),
            WorldPoint(2.5, 1.5),
        )
    )
    motion_report = build_gcs_motion_feasibility_report(
        gcs_report,
        min_turning_radius_m=5.0,
        max_heading_change_deg=30.0,
    )

    payload = build_gcs_geometric_candidate_report(
        grid,
        plan,
        postprocess,
        gcs_report,
        motion_report,
    ).to_route_fields()

    assert payload["gcs_candidate_available"] is False
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "motion_infeasible"
    assert payload["gcs_candidate_constraint_summary"]["motion_feasibility"]["status"] == "infeasible"


def test_gcs_geometric_candidate_reports_cost_dominated_path_without_replacing_route():
    grid = _candidate_grid(
        [
            [10.0, 10.0, 10.0, 10.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=4.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5))
    )

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_attempted"] is True
    assert payload["gcs_candidate_available"] is True
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "cost_dominated"
    assert payload["gcs_candidate_cost_delta_vs_baseline"] > 0.0
    assert payload["gcs_candidate_cost_summary"]["candidate_decision"] == "blocked"
    assert payload["gcs_candidate_cost_summary"]["decision_reason"] == "cost_dominated"
    assert payload["gcs_candidate_cost_summary"]["quality_gate"]["baseline_delta_improved"] is False


def test_gcs_geometric_candidate_blocks_when_high_cost_exposure_worsens():
    grid = _candidate_grid(
        [
            [10.0, 10.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    baseline_cells = (Cell(0, 1), Cell(1, 1), Cell(2, 1), Cell(3, 1))
    plan = _plan_from_cells(grid, baseline_cells, total_cost=50.0)
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(2.5, 0.5),
            WorldPoint(3.5, 0.5),
        )
    )

    payload = build_gcs_geometric_candidate_report(
        grid,
        plan,
        None,
        gcs_report,
        high_cost_threshold=3.0,
    ).to_route_fields()

    assert payload["gcs_candidate_available"] is True
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "high_cost_exposure"
    assert payload["gcs_candidate_cost_delta_vs_baseline"] < 0.0
    assert payload["gcs_candidate_high_cost_exposure"] > 0.0
    assert payload["gcs_candidate_cost_summary"]["baseline_high_cost_exposure"] == 0.0
    assert payload["gcs_candidate_cost_summary"]["high_cost_exposure_delta_vs_baseline"] > 0.0
    assert payload["gcs_candidate_cost_summary"]["quality_gate"]["high_cost_exposure_not_worse"] is False


def test_gcs_geometric_candidate_reports_duplicate_baseline_path():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    path_cells = (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2))
    plan = _plan_from_cells(grid, path_cells, total_cost=4.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points(tuple(_cell_center(grid, cell) for cell in path_cells))

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_available"] is True
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "path_duplicate_with_baseline"
    assert payload["gcs_candidate_baseline_overlap_ratio"] == 1.0


def test_gcs_direction_cone_scenario_matrix_summarizes_expected_outcomes():
    clean_grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )
    collision_grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        blocked=(Cell(1, 0),),
    )
    plan = _plan_from_cells(clean_grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)), total_cost=40.0)
    postprocess = run_postprocess(clean_grid, plan)
    selected = build_gcs_geometric_candidate_report(
        clean_grid,
        plan,
        postprocess,
        _gcs_report_from_points(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5))
        ),
    ).to_route_fields()
    expensive_grid = _candidate_grid(
        [
            [10.0, 10.0, 10.0, 10.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    cost_dominated_plan = _plan_from_cells(
        expensive_grid,
        (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)),
        total_cost=4.0,
    )
    cost_dominated = build_gcs_geometric_candidate_report(
        expensive_grid,
        cost_dominated_plan,
        run_postprocess(expensive_grid, cost_dominated_plan),
        _gcs_report_from_points(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5), WorldPoint(3.5, 0.5))
        ),
    ).to_route_fields()
    duplicate_grid = _candidate_grid(np.ones((3, 4), dtype=float))
    duplicate_plan = _plan_from_cells(
        duplicate_grid,
        (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)),
        total_cost=4.0,
    )
    duplicate = build_gcs_geometric_candidate_report(
        duplicate_grid,
        duplicate_plan,
        run_postprocess(duplicate_grid, duplicate_plan),
        _gcs_report_from_points(
            tuple(_cell_center(duplicate_grid, cell) for cell in (Cell(0, 2), Cell(1, 2), Cell(2, 2), Cell(3, 2)))
        ),
    ).to_route_fields()
    collision = build_gcs_geometric_candidate_report(
        collision_grid,
        plan,
        postprocess,
        _gcs_report_from_points((WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5))),
    ).to_route_fields()
    not_evaluated = build_gcs_geometric_candidate_report(
        clean_grid,
        plan,
        postprocess,
        _gcs_report_from_points(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5)),
            direction_cone_evaluated=False,
        ),
    ).to_route_fields()
    not_enforced = build_gcs_geometric_candidate_report(
        clean_grid,
        plan,
        postprocess,
        _gcs_report_from_points(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5)),
            direction_cone_backend_enforced=False,
        ),
    ).to_route_fields()
    violation = build_gcs_geometric_candidate_report(
        clean_grid,
        plan,
        postprocess,
        _gcs_report_from_points(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5)),
            direction_cone_violation_count=1,
        ),
    ).to_route_fields()
    turn_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(1.5, 1.5), WorldPoint(2.5, 1.5))
    )
    motion_infeasible = build_gcs_geometric_candidate_report(
        clean_grid,
        plan,
        postprocess,
        turn_report,
        build_gcs_motion_feasibility_report(turn_report, min_turning_radius_m=5.0, max_heading_change_deg=30.0),
    ).to_route_fields()
    degenerate_portal = {
        "gcs_trajectory_constraint_summary": build_direction_cone_constraint_summary(
            (WorldPoint(0.5, 0.5), WorldPoint(1.5, 1.5)),
            (
                _convex_region_for_bounds(
                    clean_grid,
                    region_id=0,
                    seed=Cell(0, 0),
                    min_cell=Cell(0, 0),
                    max_cell=Cell(0, 0),
                ),
                _convex_region_for_bounds(
                    clean_grid,
                    region_id=1,
                    seed=Cell(1, 1),
                    min_cell=Cell(1, 1),
                    max_cell=Cell(1, 1),
                ),
            ),
        )
    }
    pydrake_unavailable = {
        "gcs_trajectory_attempted": False,
        "gcs_trajectory_reason": "pydrake_unavailable",
        "gcs_trajectory_constraint_summary": _direction_cone_summary(evaluated=False),
    }

    summary = build_gcs_scenario_matrix_summary(
        [
            _scenario_case("open_corridor", selected, "selected", "gcs_candidate_quality_improved"),
            _scenario_case("cost_dominated", cost_dominated, "blocked", "cost_dominated"),
            _scenario_case("duplicate_baseline", duplicate, "blocked", "path_duplicate_with_baseline"),
            _scenario_case("sample_collision", collision, "blocked", "sampled_trajectory_collision"),
            _scenario_case("direction_cone_not_evaluated", not_evaluated, "blocked", "direction_cone_not_evaluated"),
            _scenario_case("direction_cone_not_enforced", not_enforced, "blocked", "direction_cone_not_backend_enforced"),
            _scenario_case("direction_cone_violation", violation, "blocked", "direction_cone_constraint_violation"),
            _scenario_case("motion_infeasible", motion_infeasible, "blocked", "motion_infeasible"),
            _scenario_case("degenerate_portal", degenerate_portal, "blocked", "degenerate_portal_width"),
            _scenario_case("pydrake_unavailable", pydrake_unavailable, "blocked", "pydrake_unavailable"),
        ]
    )

    assert summary["schema_version"] == "gcs_direction_cone_scenario_matrix/v1"
    assert summary["case_count"] == 10
    assert summary["selected_count"] == 1
    assert summary["blocked_count"] == 9
    assert summary["expectation_failures"] == []
    assert summary["decision_reason_counts"]["gcs_candidate_quality_improved"] == 1
    assert summary["decision_reason_counts"]["cost_dominated"] == 1
    assert summary["decision_reason_counts"]["degenerate_portal_width"] == 1
    assert summary["decision_reason_counts"]["pydrake_unavailable"] == 1
    case_ids = {case["case_id"] for case in summary["cases"]}
    assert case_ids == {
        "open_corridor",
        "cost_dominated",
        "duplicate_baseline",
        "sample_collision",
        "direction_cone_not_evaluated",
        "direction_cone_not_enforced",
        "direction_cone_violation",
        "motion_infeasible",
        "degenerate_portal",
        "pydrake_unavailable",
    }


def test_gcs_geometric_candidate_rechecks_sampled_path_collision():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        blocked=(Cell(1, 0),),
    )
    plan = _plan_from_cells(grid, (Cell(0, 2), Cell(1, 2), Cell(2, 2)), total_cost=3.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points((WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5)))

    payload = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report).to_route_fields()

    assert payload["gcs_candidate_attempted"] is True
    assert payload["gcs_candidate_available"] is False
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "sampled_trajectory_collision"
    assert payload["gcs_candidate_collision_count"] == 1


def test_gcs_geometric_candidate_classifies_missing_and_failed_gcs_reports():
    grid = _candidate_grid([[1.0, 1.0], [1.0, 1.0]])
    plan = _plan_from_cells(grid, (Cell(0, 1), Cell(1, 1)), total_cost=2.0)
    postprocess = run_postprocess(grid, plan)
    failed_gcs = GcsTrajectoryReport(
        attempted=True,
        success=False,
        backend="pydrake_gcs",
        result_status="solver failed",
        reason="solver_failed",
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=2,
    )

    missing = build_gcs_geometric_candidate_report(grid, plan, postprocess, None).to_route_fields()
    failed = build_gcs_geometric_candidate_report(grid, plan, postprocess, failed_gcs).to_route_fields()

    assert missing["gcs_candidate_attempted"] is True
    assert missing["gcs_candidate_available"] is False
    assert missing["gcs_candidate_fallback_reason"] == "gcs_report_missing"
    assert failed["gcs_candidate_available"] is False
    assert failed["gcs_candidate_fallback_reason"] == "gcs_trajectory_failed"


def test_gcs_motion_feasibility_report_classifies_curvature_bounded_sampled_path():
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(2.5, 0.5),
            WorldPoint(3.5, 0.5),
        )
    )

    payload = build_gcs_motion_feasibility_report(
        gcs_report,
        min_turning_radius_m=0.5,
        max_heading_change_deg=45.0,
    ).to_route_fields()

    assert payload["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
    assert payload["gcs_motion_feasibility_evaluated"] is True
    assert payload["gcs_motion_feasibility_trajectory_source"] == "gcs_trajectory_sampled_points"
    assert payload["gcs_motion_feasibility_motion_model"] == "curvature_bounded"
    assert payload["gcs_motion_feasibility_feasibility_status"] == "feasible"
    assert payload["gcs_motion_feasibility_fallback_reason"] is None
    assert payload["gcs_motion_feasibility_min_turning_radius_m"] == 0.5
    assert payload["gcs_motion_feasibility_max_heading_change_deg"] == 45.0
    assert payload["gcs_motion_feasibility_curvature_violation_count"] == 0
    assert payload["gcs_motion_feasibility_heading_violation_count"] == 0
    assert payload["gcs_motion_feasibility_violation_indices"] == []
    assert payload["gcs_motion_feasibility_sample_count"] == 4
    assert payload["gcs_motion_feasibility_path_length"] > 0.0
    assert payload["gcs_motion_feasibility_constraint_summary"]["motion_model"] == "curvature_bounded"


def test_gcs_motion_feasibility_report_splits_heading_and_curvature_violations():
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(1.5, 1.5),
            WorldPoint(1.5, 2.5),
        )
    )

    payload = build_gcs_motion_feasibility_report(
        gcs_report,
        min_turning_radius_m=5.0,
        max_heading_change_deg=30.0,
    ).to_route_fields()

    assert payload["gcs_motion_feasibility_evaluated"] is True
    assert payload["gcs_motion_feasibility_feasibility_status"] == "infeasible"
    assert payload["gcs_motion_feasibility_fallback_reason"] == "motion_constraint_violation"
    assert payload["gcs_motion_feasibility_curvature_violation_count"] > 0
    assert payload["gcs_motion_feasibility_heading_violation_count"] > 0
    assert 1 in payload["gcs_motion_feasibility_violation_indices"]
    assert payload["gcs_motion_feasibility_constraint_summary"]["max_observed_heading_change_deg"] >= 90.0


def test_gcs_motion_feasibility_report_classifies_missing_failed_and_short_gcs_reports():
    failed_gcs = GcsTrajectoryReport(
        attempted=True,
        success=False,
        backend="pydrake_gcs",
        result_status="solver failed",
        reason="solver_failed",
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=2,
    )
    short_gcs = _gcs_report_from_points((WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5)))

    missing = build_gcs_motion_feasibility_report(None).to_route_fields()
    failed = build_gcs_motion_feasibility_report(failed_gcs).to_route_fields()
    short = build_gcs_motion_feasibility_report(short_gcs).to_route_fields()

    assert missing["gcs_motion_feasibility_evaluated"] is False
    assert missing["gcs_motion_feasibility_feasibility_status"] == "diagnostic_only"
    assert missing["gcs_motion_feasibility_fallback_reason"] == "gcs_report_missing"
    assert failed["gcs_motion_feasibility_evaluated"] is False
    assert failed["gcs_motion_feasibility_fallback_reason"] == "gcs_trajectory_failed"
    assert short["gcs_motion_feasibility_evaluated"] is False
    assert short["gcs_motion_feasibility_fallback_reason"] == "insufficient_samples"


def test_gcs_motion_feasibility_report_is_optional_route_json_field_without_changing_route_semantics():
    grid = _candidate_grid([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    plan = _plan_from_cells(grid, (Cell(0, 1), Cell(1, 1), Cell(2, 1)), total_cost=30.0)
    gcs_report = _gcs_report_from_points(
        (WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5))
    )
    motion_report = build_gcs_motion_feasibility_report(
        gcs_report,
        min_turning_radius_m=0.5,
        max_heading_change_deg=45.0,
    )

    payload = route_result_to_json_dict(
        plan,
        grid.spec,
        gcs_trajectory_report=gcs_report,
        gcs_motion_feasibility_report=motion_report,
    )

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
    assert payload["gcs_motion_feasibility_feasibility_status"] == "feasible"


def test_gcs_curvature_constrained_candidate_repairs_curvature_violation_within_regions():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    plan = _plan_from_cells(grid, (Cell(0, 0), Cell(1, 0), Cell(1, 1), Cell(1, 2)), total_cost=4.0)
    convex_report = _wide_convex_region_report(grid)
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(1.5, 1.5),
            WorldPoint(1.5, 2.5),
        )
    )
    motion_report = build_gcs_motion_feasibility_report(
        gcs_report,
        max_curvature=1.0,
        max_heading_change_deg=120.0,
    )

    payload = build_gcs_curvature_constrained_candidate_report(
        grid,
        plan,
        convex_report,
        gcs_report,
        motion_report,
        max_curvature=1.0,
        max_heading_change_deg=120.0,
    ).to_route_fields()

    assert payload["gcs_curvature_constrained_report_schema_version"] == (
        "gcs_curvature_constrained_candidate_report/v1"
    )
    assert payload["gcs_curvature_constrained_attempted"] is True
    assert payload["gcs_curvature_constrained_available"] is True
    assert payload["gcs_curvature_constrained_selected"] is True
    assert payload["gcs_curvature_constrained_repair_success"] is True
    assert payload["gcs_curvature_constrained_repair_strategy"] == "moving_average_smoothing"
    assert payload["gcs_curvature_constrained_fallback_reason"] is None
    assert payload["gcs_curvature_constrained_status_before"] == "infeasible"
    assert payload["gcs_curvature_constrained_status_after"] == "feasible"
    assert payload["gcs_curvature_constrained_curvature_violation_count_before"] > 0
    assert payload["gcs_curvature_constrained_curvature_violation_count_after"] == 0
    assert payload["gcs_curvature_constrained_heading_violation_count_after"] == 0
    assert payload["gcs_curvature_constrained_collision_count"] == 0
    assert payload["gcs_curvature_constrained_region_containment_violation_count"] == 0
    assert payload["gcs_curvature_constrained_path_length"] > 0.0
    assert payload["gcs_curvature_constrained_path_cost"] > 0.0
    assert payload["gcs_curvature_constrained_cost_delta_vs_baseline"] is not None
    assert payload["gcs_curvature_constrained_constraint_summary"]["max_curvature"] == 1.0
    assert payload["gcs_curvature_constrained_constraint_summary"]["repair_passes"] > 0


def test_gcs_curvature_constrained_candidate_rechecks_repaired_path_collision():
    grid = _candidate_grid(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        blocked=(Cell(1, 1),),
    )
    plan = _plan_from_cells(grid, (Cell(0, 0), Cell(1, 0), Cell(1, 1), Cell(1, 2)), total_cost=4.0)
    convex_report = _wide_convex_region_report(grid)
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(1.5, 1.5),
            WorldPoint(1.5, 2.5),
        )
    )
    motion_report = build_gcs_motion_feasibility_report(gcs_report, max_curvature=1.0)

    payload = build_gcs_curvature_constrained_candidate_report(
        grid,
        plan,
        convex_report,
        gcs_report,
        motion_report,
        max_curvature=1.0,
    ).to_route_fields()

    assert payload["gcs_curvature_constrained_attempted"] is True
    assert payload["gcs_curvature_constrained_available"] is False
    assert payload["gcs_curvature_constrained_selected"] is False
    assert payload["gcs_curvature_constrained_repair_success"] is False
    assert payload["gcs_curvature_constrained_fallback_reason"] == "candidate_collision"
    assert payload["gcs_curvature_constrained_collision_count"] > 0


def test_gcs_curvature_constrained_candidate_is_optional_route_json_field_without_changing_route_semantics():
    grid = _candidate_grid([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    plan = _plan_from_cells(grid, (Cell(0, 0), Cell(1, 0), Cell(1, 1), Cell(1, 2)), total_cost=4.0)
    convex_report = _wide_convex_region_report(grid)
    gcs_report = _gcs_report_from_points(
        (
            WorldPoint(0.5, 0.5),
            WorldPoint(1.5, 0.5),
            WorldPoint(1.5, 1.5),
            WorldPoint(1.5, 2.5),
        )
    )
    motion_report = build_gcs_motion_feasibility_report(gcs_report, max_curvature=1.0)
    constrained_report = build_gcs_curvature_constrained_candidate_report(
        grid,
        plan,
        convex_report,
        gcs_report,
        motion_report,
        max_curvature=1.0,
    )

    payload = route_result_to_json_dict(
        plan,
        grid.spec,
        convex_region_sequence_report=convex_report,
        gcs_trajectory_report=gcs_report,
        gcs_motion_feasibility_report=motion_report,
        gcs_curvature_constrained_candidate_report=constrained_report,
    )

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["geometric_path"]["cells"] == [[0, 0], [1, 0], [1, 1], [1, 2]]
    assert payload["gcs_curvature_constrained_report_schema_version"] == (
        "gcs_curvature_constrained_candidate_report/v1"
    )
    assert payload["gcs_curvature_constrained_available"] is True
    assert payload["gcs_curvature_constrained_selected"] is True


def test_gcs_geometric_candidate_is_optional_route_json_field_without_changing_route_semantics():
    grid = _candidate_grid([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    plan = _plan_from_cells(grid, (Cell(0, 1), Cell(1, 1), Cell(2, 1)), total_cost=30.0)
    postprocess = run_postprocess(grid, plan)
    gcs_report = _gcs_report_from_points((WorldPoint(0.5, 0.5), WorldPoint(1.5, 0.5), WorldPoint(2.5, 0.5)))
    candidate_report = build_gcs_geometric_candidate_report(grid, plan, postprocess, gcs_report)

    payload = route_result_to_json_dict(
        plan,
        grid.spec,
        postprocess=postprocess,
        gcs_trajectory_report=gcs_report,
        gcs_candidate_report=candidate_report,
    )

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["geometric_path"]["cells"] == [[0, 1], [1, 1], [2, 1]]
    assert payload["gcs_candidate_available"] is True
    assert payload["gcs_candidate_selected"] is True
    assert payload["gcs_candidate_path_cost"] < payload["path_cost"]


def test_workspace_iris_cell_bounds_do_not_hide_unsafe_cells(monkeypatch):
    import path_planner.drake_backend.iris as iris_backend

    class FakeHPolyhedron:
        @staticmethod
        def MakeBox(min_bounds, max_bounds):
            return (tuple(min_bounds), tuple(max_bounds))

    class FakeIrisOptions:
        require_sample_point_is_contained = False

    class FakeRegion:
        def IsEmpty(self):
            return False

        def PointInSet(self, sample, tolerance):
            x = int(float(sample[0]))
            y = int(float(sample[1]))
            return (x, y) in {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (2, 2)}

        def A(self):
            return np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))

        def b(self):
            return np.asarray((3.0, 0.0, 3.0, 0.0))

    class FakeGeometryOptimization:
        HPolyhedron = FakeHPolyhedron
        IrisOptions = FakeIrisOptions

        @staticmethod
        def Iris(obstacles, sample, domain, options):
            return FakeRegion()

    monkeypatch.setattr(iris_backend, "_load_geometry_optimization", lambda: FakeGeometryOptimization)
    grid = make_grid(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ],
        resolution=1.0,
    )
    corridor = CorridorResult(
        status="ok",
        radius_cells=1,
        sections=(
            CorridorSection(
                center=Cell(0, 0),
                cells=(Cell(0, 0), Cell(1, 0), Cell(2, 0), Cell(0, 1), Cell(0, 2), Cell(2, 2)),
            ),
        ),
        failure_reason=None,
        original_blocked_count=1,
        inflated_blocked_count=1,
    )

    report = build_workspace_iris_region_report(grid, corridor)
    region = report.to_dict()["regions"][0]
    bounds = region["cell_bounds"]

    assert report.to_dict()["status"] == "ok"
    assert not (
        bounds["min"][0] <= 1 <= bounds["max"][0]
        and bounds["min"][1] <= 1 <= bounds["max"][1]
    )


def test_cli_drake_iris_regions_writes_optional_report_without_changing_route_semantics(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    report = payload["iris_region_report"]
    graph_report = payload["region_graph_report"]

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert report["backend"] == "workspace_iris"
    assert report["status"] in {"ok", "fallback"}
    assert report["region_count"] > 0
    assert payload["convex_region_count"] > 0
    assert payload["convex_region_sequence"]
    assert payload["convex_region_backend"] in {"workspace_iris", "fallback_box"}
    assert payload["convex_region_blocked_cell_violation_count"] == 0
    assert payload["gcs_ready"] is True
    assert report["seed_source"] == "postprocess_corridor_centers"
    assert report["obstacle_source"] in {"blocked_cell_box", "merged_blocked_rectangle"}
    assert graph_report["quality_metrics"]["requested_region_source"] == "iris"
    assert graph_report["quality_metrics"]["graph_source"] in {"iris", "grid_box"}
    assert graph_report["quality_metrics"]["start_goal_connected"] is True
    if report["status"] == "fallback":
        assert report["domain_source"] == "postprocess_corridor_grid_box"
        assert graph_report["region_source"] == "grid_box"
        assert graph_report["fallback_used"] is True
        assert "iris_region_graph_fallback" in graph_report["failure_reason"]
    else:
        assert report["domain_source"] == "postprocess_corridor_safe_component_box"
    html = (output_dir / "diagnostics.html").read_text(encoding="utf-8")
    assert "IRIS / Region Graph Summary" in html
    assert "not a GCS trajectory" in html
    assert "not an Ackermann/skid-steer feasibility proof" in html
    assert "iris_region_status" in completed.stdout


def test_cli_gcs_trajectory_smoke_writes_optional_report_without_changing_route_semantics(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-trajectory-smoke",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_ready"] is True
    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_trajectory_constraint_summary"]["constraint_model"] == "direction_cone"
    assert payload["gcs_trajectory_cost_summary"]["schema_version"] == "gcs_cost_summary/v1"
    if payload["gcs_trajectory_attempted"]:
        assert payload["gcs_trajectory_success"] is True
        assert payload["gcs_trajectory_collision_count"] == 0
        assert payload["gcs_trajectory_region_count"] == payload["convex_region_count"]
    else:
        assert payload["gcs_trajectory_success"] is False
        assert payload["gcs_trajectory_reason"] == "pydrake_unavailable"
        assert payload["gcs_trajectory_constraint_summary"]["evaluated"] is False
    assert "gcs_trajectory_success" in completed.stdout


def test_cli_gcs_geometric_candidate_is_opt_in_and_writes_candidate_report(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-geometric-candidate",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
    assert payload["gcs_candidate_attempted"] is True
    assert "gcs_candidate_available" in payload
    assert "gcs_candidate_selected" in payload
    direction_cone = payload["gcs_trajectory_constraint_summary"]
    cost_summary = payload["gcs_candidate_cost_summary"]
    assert "rho_source_counts" in direction_cone
    assert "portal_width_min_m" in direction_cone
    assert "support_width_min_m" in direction_cone
    assert "constraint_tightness_min" in direction_cone
    assert "candidate_decision" in cost_summary
    assert "decision_reason" in cost_summary
    assert "quality_gate" in cost_summary
    if payload["gcs_trajectory_attempted"]:
        assert direction_cone["backend_enforced"] is True
        assert direction_cone["rho_source_counts"]
        assert cost_summary["candidate_decision"] in {"selected", "blocked"}
    else:
        assert payload["gcs_trajectory_reason"] == "pydrake_unavailable"
        assert direction_cone["evaluated"] is False
    assert "gcs_candidate_available" in completed.stdout


@pytest.mark.drake
def test_cli_gcs_control_point_candidate_is_opt_in_and_writes_reports(tmp_path):
    pytest.importorskip("pydrake")
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-control-point-candidate",
            "--gcs-motion-feasibility",
            "--max-heading-change-deg",
            "120",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_constraint_summary"]["trajectory_parameterization"] == (
        "control_point_derivative_proxy"
    )
    assert payload["gcs_trajectory_constraint_summary"]["backend_enforced"] is True
    assert payload["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
    assert payload["gcs_candidate_cost_summary"]["candidate_decision"] in {"selected", "blocked"}
    assert "decision_reason" in payload["gcs_candidate_cost_summary"]
    assert payload["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
    assert "gcs_trajectory_backend" in completed.stdout
    assert "gcs_candidate_available" in completed.stdout


def test_cli_gcs_control_point_candidate_forwards_calibration_parameters(tmp_path):
    pytest.importorskip("pydrake")
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-control-point-candidate",
            "--gcs-control-point-terrain-weight",
            "0.08",
            "--gcs-control-point-second-difference-weight",
            "0.35",
            "--gcs-control-point-high-cost-exposure-weight",
            "0.45",
            "--gcs-control-point-direction-cone-max-error-deg",
            "35",
            "--gcs-control-point-direction-cone-rho-floor-m",
            "0.04",
            "--gcs-control-point-direction-cone-seed-rho-ratio",
            "0.08",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    constraint_summary = payload["gcs_trajectory_constraint_summary"]
    cost_summary = payload["gcs_trajectory_cost_summary"]

    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert constraint_summary["objective_term_weights"]["control_point_terrain_anchor_quadratic"] == 0.08
    assert constraint_summary["objective_term_weights"]["control_point_second_difference_quadratic"] == 0.35
    assert constraint_summary["objective_term_weights"]["control_point_high_cost_exposure_proxy_quadratic"] == 0.45
    assert constraint_summary["high_cost_exposure_objective_weight"] == 0.45
    assert cost_summary["high_cost_exposure_objective_weight"] == 0.45
    assert cost_summary["high_cost_exposure_proxy_cost"] is not None
    assert constraint_summary["max_allowed_direction_error_deg"] == 35.0
    assert constraint_summary["rho_lower_bound_min_m"] >= 0.04
    assert cost_summary["terrain_objective_weight"] == 0.08


def test_cli_gcs_control_point_candidate_forces_pydrake_unavailable(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PATH_PLANNER_FORCE_PYDRAKE_UNAVAILABLE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--gcs-control-point-candidate",
            "--gcs-motion-feasibility",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_trajectory_attempted"] is False
    assert payload["gcs_trajectory_reason"] == "pydrake_unavailable"
    assert payload["gcs_motion_feasibility_feasibility_status"] == "diagnostic_only"
    assert payload["gcs_candidate_fallback_reason"] == "gcs_trajectory_failed"
    assert "pydrake_unavailable" in completed.stdout


@pytest.mark.drake
def test_cli_gcs_control_point_candidate_still_blocks_motion_infeasible_candidate(tmp_path):
    pytest.importorskip("pydrake")
    request_json = tmp_path / "request.json"
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    request_json.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-request/v1",
                "grid": {
                    "width": 8,
                    "height": 4,
                    "resolution": 1.0,
                    "origin": [0.0, 0.0],
                    "frame_id": "control_point_motion_blocked",
                },
                "cost": [[1 for _ in range(8)] for _ in range(4)],
                "passable_mask": [[True for _ in range(8)] for _ in range(4)],
                "start": [0, 3],
                "goal": [7, 0],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            str(request_json),
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--gcs-control-point-candidate",
            "--gcs-motion-feasibility",
            "--max-heading-change-deg",
            "1",
            "--max-shortcut-cost",
            "0",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_motion_feasibility_feasibility_status"] == "infeasible"
    assert payload["gcs_motion_feasibility_heading_violation_count"] > 0
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "motion_infeasible"


@pytest.mark.drake
def test_cli_gcs_control_point_candidate_blocks_tight_turning_radius_candidate(tmp_path):
    pytest.importorskip("pydrake")
    request_json = tmp_path / "request.json"
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    request_json.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-request/v1",
                "grid": {
                    "width": 8,
                    "height": 4,
                    "resolution": 1.0,
                    "origin": [0.0, 0.0],
                    "frame_id": "control_point_tight_radius_blocked",
                },
                "cost": [[1 for _ in range(8)] for _ in range(4)],
                "passable_mask": [[True for _ in range(8)] for _ in range(4)],
                "start": [0, 3],
                "goal": [7, 0],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            str(request_json),
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--gcs-control-point-candidate",
            "--gcs-motion-feasibility",
            "--max-heading-change-deg",
            "120",
            "--min-turning-radius",
            "20",
            "--max-shortcut-cost",
            "0",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert payload["gcs_motion_feasibility_feasibility_status"] == "infeasible"
    assert payload["gcs_motion_feasibility_fallback_reason"] == "curvature_constraint_violation"
    assert payload["gcs_motion_feasibility_curvature_violation_count"] > 0
    assert payload["gcs_candidate_selected"] is False
    assert payload["gcs_candidate_fallback_reason"] == "motion_infeasible"


@pytest.mark.drake
def test_gcs_cli_scenario_batch_summarizes_route_json_cases(tmp_path):
    pytest.importorskip("pydrake")
    from path_planner.drake_backend.gcs_cli_batch import run_gcs_cli_scenario_batch

    output_dir = tmp_path / "batch"
    summary_json = output_dir / "summary.json"

    summary = run_gcs_cli_scenario_batch(
        output_dir=output_dir,
        summary_json=summary_json,
        python_executable=sys.executable,
    )

    assert summary_json.exists()
    persisted = json.loads(summary_json.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["schema_version"] == "gcs_direction_cone_cli_scenario_batch/v1"
    assert summary["case_count"] == 7
    assert summary["selected_count"] == 1
    assert summary["blocked_count"] == 6
    assert summary["expectation_failures"] == []
    assert summary["decision_reason_counts"]["gcs_candidate_quality_improved"] == 1
    assert summary["decision_reason_counts"]["cost_dominated"] == 1
    assert summary["decision_reason_counts"]["path_duplicate_with_baseline"] == 1
    assert summary["decision_reason_counts"]["direction_cone_constraint_violation"] == 1
    assert summary["decision_reason_counts"]["motion_infeasible"] == 1
    assert summary["decision_reason_counts"]["degenerate_portal_width"] == 1
    assert summary["decision_reason_counts"]["pydrake_unavailable"] == 1
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert set(cases) == {
        "open_corridor_selected",
        "cost_dominated_diagonal",
        "duplicate_baseline",
        "direction_cone_obstacle_detour",
        "motion_infeasible_turn",
        "degenerate_portal",
        "pydrake_unavailable",
    }
    assert cases["open_corridor_selected"]["outcome"] == "selected"
    assert cases["open_corridor_selected"]["fallback_reason"] is None
    assert cases["direction_cone_obstacle_detour"]["fallback_reason"] == "direction_cone_constraint_violation"
    assert cases["direction_cone_obstacle_detour"]["direction_cone_status"] == "violated"
    assert cases["degenerate_portal"]["fallback_reason"] == "direction_cone_constraint_violation"
    assert "degenerate_portal_width" in cases["degenerate_portal"]["direction_cone_risk_flags"]
    for case in summary["cases"]:
        route_json = output_dir / case["route_json_path"]
        route = json.loads(route_json.read_text(encoding="utf-8"))
        assert route["trajectory_kind"] == "geometric_path"
        assert route["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
        assert route["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
        assert "gcs_trajectory_constraint_summary" in route
        assert "gcs_candidate_cost_summary" in route
        assert "rho_source_counts" in route["gcs_trajectory_constraint_summary"]
        assert "candidate_decision" in route["gcs_candidate_cost_summary"]
        assert "decision_reason" in route["gcs_candidate_cost_summary"]
        assert "quality_gate" in route["gcs_candidate_cost_summary"]


def test_gcs_cli_scenario_batch_cli_forces_pydrake_unavailable_case(tmp_path):
    summary_json = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.drake_backend.gcs_cli_batch",
            "--output-dir",
            str(tmp_path / "batch"),
            "--summary-json",
            str(summary_json),
            "--case",
            "pydrake_unavailable",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
        capture_output=True,
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "gcs_direction_cone_cli_scenario_batch/v1"
    assert summary["case_count"] == 1
    assert summary["selected_count"] == 0
    assert summary["blocked_count"] == 1
    assert summary["expectation_failures"] == []
    case = summary["cases"][0]
    assert case["case_id"] == "pydrake_unavailable"
    assert case["outcome"] == "blocked"
    assert case["decision_reason"] == "pydrake_unavailable"
    assert case["trajectory_attempted"] is False
    route = json.loads((tmp_path / "batch" / case["route_json_path"]).read_text(encoding="utf-8"))
    assert route["gcs_trajectory_attempted"] is False
    assert route["gcs_trajectory_reason"] == "pydrake_unavailable"
    assert route["gcs_trajectory_constraint_summary"]["evaluated"] is False
    assert "pydrake_unavailable" in completed.stdout


def test_gcs_motion_feasibility_batch_summary_flags_expected_mismatches():
    from path_planner.drake_backend.gcs_cli_batch import build_gcs_motion_feasibility_batch_summary

    feasible_payload = {
        "gcs_trajectory_attempted": True,
        "gcs_trajectory_success": True,
        "gcs_trajectory_reason": "direction_cone_solution_found",
        "gcs_candidate_selected": True,
        "gcs_candidate_available": True,
        "gcs_candidate_fallback_reason": None,
        "gcs_motion_feasibility_report_schema_version": "gcs_motion_feasibility_report/v1",
        "gcs_motion_feasibility_evaluated": True,
        "gcs_motion_feasibility_feasibility_status": "feasible",
        "gcs_motion_feasibility_fallback_reason": None,
        "gcs_motion_feasibility_motion_model": "curvature_bounded",
        "gcs_motion_feasibility_min_turning_radius_m": None,
        "gcs_motion_feasibility_max_heading_change_deg": 120.0,
        "gcs_motion_feasibility_curvature_violation_count": 0,
        "gcs_motion_feasibility_heading_violation_count": 0,
        "gcs_motion_feasibility_violation_indices": [],
        "gcs_motion_feasibility_sample_count": 4,
        "gcs_motion_feasibility_path_length": 3.0,
        "gcs_motion_feasibility_constraint_summary": {
            "max_observed_curvature": 0.0,
            "min_observed_turning_radius_m": None,
            "max_observed_heading_change_deg": 0.0,
        },
    }
    unavailable_payload = {
        "gcs_trajectory_attempted": False,
        "gcs_trajectory_success": False,
        "gcs_trajectory_reason": "pydrake_unavailable",
        "gcs_candidate_selected": False,
        "gcs_candidate_available": False,
        "gcs_candidate_fallback_reason": "gcs_trajectory_failed",
        "gcs_motion_feasibility_report_schema_version": "gcs_motion_feasibility_report/v1",
        "gcs_motion_feasibility_evaluated": False,
        "gcs_motion_feasibility_feasibility_status": "diagnostic_only",
        "gcs_motion_feasibility_fallback_reason": "gcs_trajectory_failed",
        "gcs_motion_feasibility_motion_model": "curvature_bounded",
        "gcs_motion_feasibility_min_turning_radius_m": None,
        "gcs_motion_feasibility_max_heading_change_deg": 120.0,
        "gcs_motion_feasibility_curvature_violation_count": 0,
        "gcs_motion_feasibility_heading_violation_count": 0,
        "gcs_motion_feasibility_violation_indices": [],
        "gcs_motion_feasibility_sample_count": 0,
        "gcs_motion_feasibility_path_length": 0.0,
        "gcs_motion_feasibility_constraint_summary": {
            "max_observed_curvature": None,
            "min_observed_turning_radius_m": None,
            "max_observed_heading_change_deg": None,
        },
    }

    summary = build_gcs_motion_feasibility_batch_summary(
        [
            {
                "case_id": "feasible_case",
                "payload": feasible_payload,
                "expected": {"outcome": "feasible", "decision_reason": "motion_feasible"},
            },
            {
                "case_id": "mismatch_case",
                "payload": unavailable_payload,
                "expected": {"outcome": "feasible", "decision_reason": "motion_feasible"},
            },
        ]
    )

    assert summary["schema_version"] == "gcs_motion_feasibility_cli_batch/v1"
    assert summary["case_count"] == 2
    assert summary["feasible_count"] == 1
    assert summary["infeasible_count"] == 0
    assert summary["diagnostic_only_count"] == 1
    assert summary["candidate_selected_count"] == 1
    assert summary["candidate_blocked_count"] == 1
    assert summary["decision_reason_counts"]["motion_feasible"] == 1
    assert summary["decision_reason_counts"]["pydrake_unavailable"] == 1
    assert summary["expectation_failures"] == [
        {"case_id": "mismatch_case", "mismatch_fields": ["outcome", "decision_reason"]}
    ]


def test_gcs_control_point_terrain_cost_batch_summary_flags_expected_mismatches():
    from path_planner.drake_backend.gcs_cli_batch import (
        build_gcs_control_point_terrain_cost_batch_summary,
    )

    selected_payload = {
        "gcs_trajectory_attempted": True,
        "gcs_trajectory_success": True,
        "gcs_trajectory_backend": "pydrake_control_point_direction_cone_program",
        "gcs_trajectory_reason": "control_point_direction_cone_solution_found",
        "gcs_candidate_selected": True,
        "gcs_candidate_available": True,
        "gcs_candidate_fallback_reason": None,
        "gcs_candidate_selection_reason": "gcs_candidate_quality_improved",
        "gcs_candidate_cost_delta_vs_baseline": -4.0,
        "gcs_candidate_high_cost_exposure": 0.0,
        "gcs_candidate_cost_summary": {
            "candidate_decision": "selected",
            "decision_reason": "gcs_candidate_quality_improved",
            "cost_delta_vs_baseline": -4.0,
            "high_cost_exposure": 0.0,
            "sampled_terrain_cost": 6.0,
            "terrain_objective_source": "region_inverse_cost_weighted_passable_cell_centroid",
            "terrain_objective_weight": 0.05,
            "control_point_terrain_cost": 6.0,
        },
        "gcs_trajectory_cost_summary": {
            "sampled_terrain_cost": 6.0,
            "terrain_objective_source": "region_inverse_cost_weighted_passable_cell_centroid",
            "terrain_objective_weight": 0.05,
            "control_point_terrain_cost": 6.0,
        },
    }
    unavailable_payload = {
        "gcs_trajectory_attempted": False,
        "gcs_trajectory_success": False,
        "gcs_trajectory_backend": "pydrake_control_point_direction_cone_program",
        "gcs_trajectory_reason": "pydrake_unavailable",
        "gcs_candidate_selected": False,
        "gcs_candidate_available": False,
        "gcs_candidate_fallback_reason": "gcs_trajectory_failed",
        "gcs_candidate_cost_summary": {
            "candidate_decision": "blocked",
            "decision_reason": "cost_not_evaluated",
        },
        "gcs_trajectory_cost_summary": {
            "sampled_terrain_cost": None,
            "terrain_objective_source": "not_evaluated",
            "terrain_objective_weight": None,
            "control_point_terrain_cost": None,
        },
    }

    summary = build_gcs_control_point_terrain_cost_batch_summary(
        [
            {
                "case_id": "selected_case",
                "payload": selected_payload,
                "expected": {"outcome": "selected", "decision_reason": "gcs_candidate_quality_improved"},
            },
            {
                "case_id": "mismatch_case",
                "payload": unavailable_payload,
                "expected": {"outcome": "selected", "decision_reason": "gcs_candidate_quality_improved"},
            },
        ]
    )

    assert summary["schema_version"] == "gcs_control_point_terrain_cost_cli_batch/v1"
    assert summary["case_count"] == 2
    assert summary["selected_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["terrain_objective_evaluated_count"] == 1
    assert summary["decision_reason_counts"]["gcs_candidate_quality_improved"] == 1
    assert summary["decision_reason_counts"]["pydrake_unavailable"] == 1
    assert summary["expectation_failures"] == [
        {"case_id": "mismatch_case", "mismatch_fields": ["outcome", "decision_reason"]}
    ]
    selected = summary["cases"][0]
    assert selected["trajectory_backend"] == "pydrake_control_point_direction_cone_program"
    assert selected["terrain_objective_source"] == (
        "region_inverse_cost_weighted_passable_cell_centroid"
    )
    assert selected["sampled_terrain_cost"] == 6.0
    assert selected["control_point_terrain_cost"] == 6.0


@pytest.mark.drake
def test_gcs_control_point_terrain_cost_cli_batch_summarizes_route_json_cases(tmp_path):
    pytest.importorskip("pydrake")
    from path_planner.drake_backend.gcs_cli_batch import run_gcs_control_point_terrain_cost_cli_batch

    output_dir = tmp_path / "control-point-terrain-batch"
    summary_json = output_dir / "summary.json"

    summary = run_gcs_control_point_terrain_cost_cli_batch(
        output_dir=output_dir,
        summary_json=summary_json,
        python_executable=sys.executable,
    )

    assert summary_json.exists()
    persisted = json.loads(summary_json.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["schema_version"] == "gcs_control_point_terrain_cost_cli_batch/v1"
    assert summary["case_count"] == 5
    assert summary["selected_count"] >= 1
    assert summary["blocked_count"] >= 3
    assert summary["terrain_objective_evaluated_count"] >= 3
    assert summary["expectation_failures"] == []
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert set(cases) == {
        "control_point_low_cost_selected",
        "control_point_cost_dominated",
        "control_point_high_cost_exposure_blocked",
        "control_point_motion_infeasible",
        "control_point_pydrake_unavailable",
    }
    assert cases["control_point_low_cost_selected"]["outcome"] == "selected"
    assert cases["control_point_cost_dominated"]["decision_reason"] == "cost_dominated"
    assert cases["control_point_high_cost_exposure_blocked"]["outcome"] == "blocked"
    assert cases["control_point_high_cost_exposure_blocked"]["high_cost_exposure"] > 0.0
    assert (
        cases["control_point_high_cost_exposure_blocked"][
            "high_cost_exposure_delta_vs_baseline"
        ]
        > 0.0
    )
    assert cases["control_point_motion_infeasible"]["decision_reason"] == "motion_infeasible"
    assert cases["control_point_pydrake_unavailable"]["decision_reason"] == "pydrake_unavailable"
    for case in summary["cases"]:
        route_json = output_dir / case["route_json_path"]
        route = json.loads(route_json.read_text(encoding="utf-8"))
        assert route["trajectory_kind"] == "geometric_path"
        assert route["gcs_trajectory_backend"] == "pydrake_control_point_direction_cone_program"
        assert route["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
        cost_summary = route["gcs_trajectory_cost_summary"]
        if route["gcs_trajectory_attempted"]:
            assert cost_summary["terrain_objective_source"] == (
                "region_inverse_cost_weighted_passable_cell_centroid"
            )
            assert cost_summary["terrain_objective_weight"] > 0.0
            assert cost_summary["control_point_terrain_cost"] is not None
            assert cost_summary["sampled_terrain_cost"] == cost_summary["terrain_path_cost"]
        else:
            assert route["gcs_trajectory_reason"] == "pydrake_unavailable"
            assert cost_summary["terrain_objective_source"] == "not_evaluated"


@pytest.mark.drake
def test_gcs_motion_feasibility_cli_batch_summarizes_route_json_cases(tmp_path):
    pytest.importorskip("pydrake")
    from path_planner.drake_backend.gcs_cli_batch import run_gcs_motion_feasibility_cli_batch

    output_dir = tmp_path / "motion-batch"
    summary_json = output_dir / "summary.json"

    summary = run_gcs_motion_feasibility_cli_batch(
        output_dir=output_dir,
        summary_json=summary_json,
        python_executable=sys.executable,
    )

    assert summary_json.exists()
    persisted = json.loads(summary_json.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["schema_version"] == "gcs_motion_feasibility_cli_batch/v1"
    assert summary["case_count"] == 6
    assert summary["feasible_count"] == 2
    assert summary["infeasible_count"] == 3
    assert summary["diagnostic_only_count"] == 1
    assert summary["candidate_selected_count"] == 1
    assert summary["expectation_failures"] == []
    assert summary["decision_reason_counts"]["motion_feasible"] == 2
    assert summary["decision_reason_counts"]["heading_constraint_violation"] == 2
    assert summary["decision_reason_counts"]["curvature_constraint_violation"] == 1
    assert summary["decision_reason_counts"]["pydrake_unavailable"] == 1
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert set(cases) == {
        "straight_feasible",
        "gentle_turn_feasible",
        "sharp_turn_blocked",
        "tight_radius_blocked",
        "direction_cone_selected_but_motion_blocked",
        "pydrake_unavailable",
    }
    assert cases["straight_feasible"]["outcome"] == "feasible"
    assert cases["straight_feasible"]["heading_violation_count"] == 0
    assert cases["gentle_turn_feasible"]["candidate_selected"] is True
    assert cases["sharp_turn_blocked"]["outcome"] == "infeasible"
    assert cases["sharp_turn_blocked"]["heading_violation_count"] > 0
    assert cases["tight_radius_blocked"]["curvature_violation_count"] > 0
    assert cases["direction_cone_selected_but_motion_blocked"]["motion_gate_blocked_candidate"] is True
    assert cases["pydrake_unavailable"]["outcome"] == "diagnostic_only"
    assert cases["pydrake_unavailable"]["decision_reason"] == "pydrake_unavailable"
    for case in summary["cases"]:
        route_json = output_dir / case["route_json_path"]
        route = json.loads(route_json.read_text(encoding="utf-8"))
        assert route["trajectory_kind"] == "geometric_path"
        assert route["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
        assert route["gcs_candidate_report_schema_version"] == "gcs_geometric_candidate_report/v1"
        assert route["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
        assert "gcs_motion_feasibility_constraint_summary" in route


def test_gcs_motion_feasibility_batch_cli_forces_pydrake_unavailable_case(tmp_path):
    summary_json = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.drake_backend.gcs_cli_batch",
            "--batch-kind",
            "motion-feasibility",
            "--output-dir",
            str(tmp_path / "motion-batch"),
            "--summary-json",
            str(summary_json),
            "--case",
            "pydrake_unavailable",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
        capture_output=True,
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "gcs_motion_feasibility_cli_batch/v1"
    assert summary["case_count"] == 1
    assert summary["feasible_count"] == 0
    assert summary["infeasible_count"] == 0
    assert summary["diagnostic_only_count"] == 1
    assert summary["expectation_failures"] == []
    case = summary["cases"][0]
    assert case["case_id"] == "pydrake_unavailable"
    assert case["outcome"] == "diagnostic_only"
    assert case["decision_reason"] == "pydrake_unavailable"
    assert case["trajectory_attempted"] is False
    route = json.loads((tmp_path / "motion-batch" / case["route_json_path"]).read_text(encoding="utf-8"))
    assert route["gcs_trajectory_attempted"] is False
    assert route["gcs_trajectory_reason"] == "pydrake_unavailable"
    assert route["gcs_motion_feasibility_feasibility_status"] == "diagnostic_only"
    assert "pydrake_unavailable" in completed.stdout


def test_cli_gcs_motion_feasibility_is_opt_in_and_writes_report(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-motion-feasibility",
            "--max-heading-change-deg",
            "120",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
    assert payload["gcs_motion_feasibility_trajectory_source"] == "gcs_trajectory_sampled_points"
    assert payload["gcs_motion_feasibility_motion_model"] == "curvature_bounded"
    if payload["gcs_trajectory_attempted"]:
        assert payload["gcs_motion_feasibility_feasibility_status"] in {"feasible", "infeasible"}
    else:
        assert payload["gcs_motion_feasibility_feasibility_status"] == "diagnostic_only"
        assert payload["gcs_motion_feasibility_fallback_reason"] == "gcs_trajectory_failed"
    assert "gcs_motion_feasibility_status" in completed.stdout


def test_cli_gcs_curvature_constrained_candidate_is_opt_in_and_writes_report(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            DEMO_MAP_CORRIDOR,
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
            "--drake-iris-regions",
            "--gcs-curvature-constrained-candidate",
            "--max-heading-change-deg",
            "120",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert payload["gcs_trajectory_report_schema_version"] == "gcs_trajectory_report/v1"
    assert payload["gcs_motion_feasibility_report_schema_version"] == "gcs_motion_feasibility_report/v1"
    assert payload["gcs_curvature_constrained_report_schema_version"] == (
        "gcs_curvature_constrained_candidate_report/v1"
    )
    assert payload["gcs_curvature_constrained_attempted"] is True
    assert "gcs_curvature_constrained_selected" in completed.stdout


def test_pydrake_imports_are_confined_to_optional_backend_and_drake_tests():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    patterns = ("from " + "pydrake", "import " + "pydrake")
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in patterns):
            relative = path.relative_to(root).as_posix()
            if not relative.startswith("src/path_planner/drake_backend/"):
                offenders.append(relative)
    assert offenders == []


def _iris_region_for_cell(region_id, cell, grid):
    min_world = grid.spec.cell_to_world(cell)
    max_world = grid.spec.cell_to_world(Cell(cell.x + 1, cell.y + 1))
    return IrisRegion(
        region_id=region_id,
        source="iris",
        seed_cell=cell,
        seed_world=grid.spec.cell_to_world(cell),
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


def _unsafe_two_region_report(grid):
    min_cell = Cell(0, 1)
    max_cell = Cell(2, 1)
    min_world = grid.spec.cell_to_world(min_cell)
    max_world = grid.spec.cell_to_world(Cell(max_cell.x + 1, max_cell.y + 1))
    regions = (
        ConvexRegionSequenceItem(
            region_id=0,
            backend="fallback_box",
            source="fallback_box",
            seed_cell=Cell(0, 1),
            seed_world=grid.spec.cell_to_world(Cell(0, 1)),
            min_cell=min_cell,
            max_cell=max_cell,
            min_world=min_world,
            max_world=max_world,
            hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            hpolyhedron_b=(max_world.x, -min_world.x, max_world.y, -min_world.y),
            covered_path_indices=(0,),
        ),
        ConvexRegionSequenceItem(
            region_id=1,
            backend="fallback_box",
            source="fallback_box",
            seed_cell=Cell(2, 1),
            seed_world=grid.spec.cell_to_world(Cell(2, 1)),
            min_cell=min_cell,
            max_cell=max_cell,
            min_world=min_world,
            max_world=max_world,
            hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            hpolyhedron_b=(max_world.x, -min_world.x, max_world.y, -min_world.y),
            covered_path_indices=(1,),
        ),
    )
    return ConvexRegionSequenceReport(
        backend="fallback_box",
        fallback_used=True,
        coverage_status="covered",
        start_contained=True,
        goal_contained=True,
        adjacent_overlap_count=1,
        portal_count=0,
        blocked_cell_violation_count=0,
        gcs_ready=True,
        gcs_ready_reason="convex_region_sequence_ready",
        regions=regions,
        pydrake_available=True,
    )


def _convex_region_for_bounds(grid, *, region_id, seed, min_cell, max_cell):
    min_world = grid.spec.cell_to_world(min_cell)
    max_world = grid.spec.cell_to_world(Cell(max_cell.x + 1, max_cell.y + 1))
    return ConvexRegionSequenceItem(
        region_id=region_id,
        backend="fallback_box",
        source="fallback_box",
        seed_cell=seed,
        seed_world=_cell_center(grid, seed),
        min_cell=min_cell,
        max_cell=max_cell,
        min_world=min_world,
        max_world=max_world,
        hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        hpolyhedron_b=(max_world.x, -min_world.x, max_world.y, -min_world.y),
        covered_path_indices=(region_id,),
    )


def _wide_convex_region_report(grid):
    min_cell = Cell(0, 0)
    max_cell = Cell(grid.spec.width - 1, grid.spec.height - 1)
    min_world = grid.spec.cell_to_world(min_cell)
    max_world = grid.spec.cell_to_world(Cell(max_cell.x + 1, max_cell.y + 1))
    region = ConvexRegionSequenceItem(
        region_id=0,
        backend="fallback_box",
        source="fallback_box",
        seed_cell=min_cell,
        seed_world=grid.spec.cell_to_world(min_cell),
        min_cell=min_cell,
        max_cell=max_cell,
        min_world=min_world,
        max_world=max_world,
        hpolyhedron_a=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        hpolyhedron_b=(max_world.x, -min_world.x, max_world.y, -min_world.y),
        covered_path_indices=(0,),
    )
    return ConvexRegionSequenceReport(
        backend="fallback_box",
        fallback_used=True,
        coverage_status="covered",
        start_contained=True,
        goal_contained=True,
        adjacent_overlap_count=0,
        portal_count=0,
        blocked_cell_violation_count=0,
        gcs_ready=True,
        gcs_ready_reason="convex_region_sequence_ready",
        regions=(region,),
        pydrake_available=True,
    )


def _candidate_grid(cost_rows, *, blocked=()):
    cost = np.asarray(cost_rows, dtype=float)
    passable = np.ones(cost.shape, dtype=bool)
    for cell in blocked:
        passable[cell.y, cell.x] = False
    return CostGrid(
        spec=GridSpec(width=cost.shape[1], height=cost.shape[0], resolution=1.0),
        cost=cost,
        passable_mask=passable,
    )


def _plan_from_cells(grid, cells, *, total_cost):
    world = tuple(_cell_center(grid, cell) for cell in cells)
    return PlanResult(
        success=True,
        path_cells=tuple(cells),
        path_world=world,
        total_cost=float(total_cost),
        expanded_count=len(cells),
        failure_reason=None,
        diagnostics=PlanDiagnostics(path_length_m=float(max(len(cells) - 1, 0))),
    )


def _cell_center(grid, cell):
    return WorldPoint(cell.x + 0.5, cell.y + 0.5)


def _gcs_report_from_points(
    points,
    *,
    direction_cone_evaluated=True,
    direction_cone_backend_enforced=True,
    direction_cone_violation_count=0,
):
    return GcsTrajectoryReport(
        attempted=True,
        success=True,
        backend="pydrake_gcs",
        result_status="SolutionResult.kSolutionFound",
        reason="gcs_trajectory_solution_found",
        sample_count=len(points),
        collision_count=0,
        path_length=float(max(len(points) - 1, 0)),
        region_count=2,
        sampled_points=tuple(points),
        constraint_summary=_direction_cone_summary(
            evaluated=direction_cone_evaluated,
            backend_enforced=direction_cone_backend_enforced,
            violation_count=direction_cone_violation_count,
        ),
        cost_summary={
            "schema_version": "gcs_cost_summary/v1",
            "path_length": float(max(len(points) - 1, 0)),
            "terrain_path_cost": float(len(points)),
            "high_cost_exposure": 0.0,
            "energy_proxy": float(max(len(points) - 1, 0)),
            "smoothness_proxy": 0.0,
        },
    )


def _direction_cone_summary(*, evaluated=True, backend_enforced=True, violation_count=0):
    actual_backend_enforced = bool(evaluated and backend_enforced)
    return {
        "schema_version": "gcs_direction_cone_constraint/v1",
        "constraint_model": "direction_cone",
        "attempted": True,
        "evaluated": evaluated,
        "status": "enforced" if actual_backend_enforced else "evaluated" if evaluated else "not_evaluated",
        "backend_enforced": actual_backend_enforced,
        "enforcing_backend": "pydrake_mathematical_program" if actual_backend_enforced else None,
        "fallback_reason": (
            None
            if actual_backend_enforced
            else "direction_cone_backend_constraint_not_supported"
            if evaluated
            else "insufficient_direction_cone_samples"
        ),
        "max_allowed_direction_error_deg": 45.0,
        "eta": 1.0,
        "rho_min": 1.0 if evaluated else None,
        "region_width_min_m": 1.0 if evaluated else None,
        "parameter_count": 1 if evaluated else 0,
        "solver_constraint_count": 4 if actual_backend_enforced else 0,
        "violation_count": int(violation_count),
        "risk_flags": ["direction_cone_violation"] if violation_count else [],
        "parameters": [
            {
                "edge_index": 0,
                "tangent": [1.0, 0.0],
                "normal": [-0.0, 1.0],
                "rho": 1.0,
                "eta": 1.0,
                "region_width_m": 1.0,
                "risk_flags": [],
            }
        ]
        if evaluated
        else [],
    }


def _scenario_case(case_id, payload, expected_outcome, expected_decision_reason):
    return {
        "case_id": case_id,
        "payload": payload,
        "expected": {
            "outcome": expected_outcome,
            "decision_reason": expected_decision_reason,
        },
    }
