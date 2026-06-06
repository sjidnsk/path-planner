from __future__ import annotations

from typing import Any

import numpy as np

from path_planner.core import CostGrid, WorldPoint

from .gcs_diagnostics import (
    build_direction_cone_constraint_summary,
    build_gcs_cost_summary,
    direction_cone_edge_parameters,
    direction_cone_not_evaluated_summary,
    empty_gcs_cost_summary,
    mark_direction_cone_backend_enforced,
)
from .gcs_trajectory import (
    _has_region_adjacency,
    _load_gcs_dependencies,
    _low_cost_anchor,
    _region_matrices,
    _resample_polyline,
    _sample_collision_count,
    _sampled_path_length,
    _solution_status,
    _solution_success,
    _solver_exception_reason,
    _solver_result_reason,
    _validate_region_hpolyhedra,
)
from .models import ConvexRegionSequenceItem, ConvexRegionSequenceReport, GcsTrajectoryReport

PYDRAKE_CONTROL_POINT_BACKEND = "pydrake_control_point_direction_cone_program"
CONTROL_POINT_ENFORCING_BACKEND = "pydrake_control_point_mathematical_program"
CONTROL_POINT_PARAMETERIZATION = "control_point_derivative_proxy"
DERIVATIVE_PROXY = "successive_control_point_difference"
OBJECTIVE_TERMS = (
    "segment_length_quadratic",
    "low_cost_anchor_quadratic",
    "control_point_second_difference_quadratic",
)


def build_gcs_control_point_trajectory_report(
    grid: CostGrid,
    convex_region_sequence_report: ConvexRegionSequenceReport | None,
    *,
    sample_count: int = 25,
) -> GcsTrajectoryReport:
    if convex_region_sequence_report is None:
        return _not_attempted("convex_region_report_missing", "convex_region_report_missing")
    if not convex_region_sequence_report.gcs_ready:
        return _not_attempted(
            "convex_region_not_gcs_ready",
            convex_region_sequence_report.gcs_ready_reason,
            region_count=convex_region_sequence_report.region_count,
        )
    regions = convex_region_sequence_report.regions
    if not regions:
        return _not_attempted("convex_region_not_gcs_ready", "convex_region_sequence_empty")
    if not _has_region_adjacency(regions):
        return _not_attempted(
            "region_overlap_missing",
            "region_overlap_missing",
            region_count=convex_region_sequence_report.region_count,
        )

    try:
        deps = _load_gcs_dependencies()
    except Exception as exc:
        return _not_attempted(
            f"{type(exc).__name__}: {exc}",
            "pydrake_unavailable",
            region_count=convex_region_sequence_report.region_count,
        )

    sample_count = max(int(sample_count), 2)
    try:
        _validate_region_hpolyhedra(regions)
    except Exception as exc:
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason="invalid_hpolyhedron",
            region_count=convex_region_sequence_report.region_count,
        )

    try:
        sampled_points, result, solver_counts = _solve_control_point_path(
            deps,
            grid,
            regions,
            sample_count=sample_count,
        )
    except Exception as exc:
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason=_solver_exception_reason(exc),
            region_count=convex_region_sequence_report.region_count,
        )

    result_status = _solution_status(result)
    if not _solution_success(result):
        return _attempted_failure(
            result_status=result_status,
            reason=_solver_result_reason(result_status),
            region_count=convex_region_sequence_report.region_count,
        )

    collision_count = _sample_collision_count(grid, sampled_points)
    path_length = _sampled_path_length(sampled_points)
    constraint_summary = _control_point_constraint_summary(
        sampled_points,
        regions,
        solver_counts=solver_counts,
    )
    cost_summary = build_gcs_cost_summary(grid, sampled_points)
    if collision_count > 0:
        return GcsTrajectoryReport(
            attempted=True,
            success=False,
            backend=PYDRAKE_CONTROL_POINT_BACKEND,
            result_status="sampled_trajectory_collision",
            reason="sampled_trajectory_collision",
            sample_count=len(sampled_points),
            collision_count=collision_count,
            path_length=path_length,
            region_count=convex_region_sequence_report.region_count,
            sampled_points=sampled_points,
            constraint_summary=constraint_summary,
            cost_summary=cost_summary,
        )
    return GcsTrajectoryReport(
        attempted=True,
        success=True,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason="control_point_direction_cone_solution_found",
        sample_count=len(sampled_points),
        collision_count=0,
        path_length=path_length,
        region_count=convex_region_sequence_report.region_count,
        sampled_points=sampled_points,
        constraint_summary=constraint_summary,
        cost_summary=cost_summary,
    )


def _not_attempted(result_status: str, reason: str, *, region_count: int = 0) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=False,
        success=False,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=_not_evaluated_summary(reason),
        cost_summary=empty_gcs_cost_summary(),
    )


def _attempted_failure(*, result_status: str, reason: str, region_count: int) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=True,
        success=False,
        backend=PYDRAKE_CONTROL_POINT_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=_not_evaluated_summary(reason),
        cost_summary=empty_gcs_cost_summary(),
    )


def _solve_control_point_path(
    deps: dict[str, Any],
    grid: CostGrid,
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    sample_count: int,
) -> tuple[tuple[WorldPoint, ...], Any, dict[str, int]]:
    if len(regions) < 2:
        raise ValueError("control-point direction_cone backend requires at least two regions")

    prog = deps["MathematicalProgram"]()
    control_points = prog.NewContinuousVariables(len(regions), 2, "cp")
    counts = {
        "solver_constraint_count": 0,
        "control_point_region_containment_count": 0,
        "start_goal_constraint_count": 0,
        "derivative_constraint_count": 0,
    }

    for index, region in enumerate(regions):
        a_matrix, b_vector = _region_matrices(region)
        for row, bound in zip(a_matrix, b_vector):
            prog.AddLinearConstraint(
                row[0] * control_points[index, 0] + row[1] * control_points[index, 1],
                -np.inf,
                bound,
            )
            counts["solver_constraint_count"] += 1
            counts["control_point_region_containment_count"] += 1

    first_seed = regions[0].seed_world
    last_seed = regions[-1].seed_world
    for variable, value in (
        (control_points[0, 0], first_seed.x),
        (control_points[0, 1], first_seed.y),
        (control_points[-1, 0], last_seed.x),
        (control_points[-1, 1], last_seed.y),
    ):
        prog.AddBoundingBoxConstraint(value, value, variable)
        counts["solver_constraint_count"] += 1
        counts["start_goal_constraint_count"] += 1

    for index, (first, second) in enumerate(zip(regions[:-1], regions[1:])):
        edge_parameters = direction_cone_edge_parameters(first, second)
        if edge_parameters["seed_distance_m"] <= 0.0:
            raise ValueError("direction_cone reference segment is degenerate")
        tangent = np.asarray(edge_parameters["tangent"], dtype=float)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        derivative_x = control_points[index + 1, 0] - control_points[index, 0]
        derivative_y = control_points[index + 1, 1] - control_points[index, 1]
        forward = tangent[0] * derivative_x + tangent[1] * derivative_y
        lateral = normal[0] * derivative_x + normal[1] * derivative_y
        eta = float(edge_parameters["eta"])
        rho = float(edge_parameters["rho_lower_bound_m"])

        prog.AddLinearConstraint(forward, rho, np.inf)
        prog.AddLinearConstraint(lateral - eta * forward, -np.inf, 0.0)
        prog.AddLinearConstraint(-lateral - eta * forward, -np.inf, 0.0)
        counts["solver_constraint_count"] += 3
        counts["derivative_constraint_count"] += 3

    objective = 0.0
    for index in range(len(regions) - 1):
        dx = control_points[index + 1, 0] - control_points[index, 0]
        dy = control_points[index + 1, 1] - control_points[index, 1]
        objective += dx * dx + dy * dy
    for index, region in enumerate(regions):
        anchor = _low_cost_anchor(grid, region)
        objective += 0.1 * (
            (control_points[index, 0] - anchor.x) ** 2
            + (control_points[index, 1] - anchor.y) ** 2
        )
    for index in range(1, len(regions) - 1):
        ddx = control_points[index + 1, 0] - 2.0 * control_points[index, 0] + control_points[index - 1, 0]
        ddy = control_points[index + 1, 1] - 2.0 * control_points[index, 1] + control_points[index - 1, 1]
        objective += 0.2 * (ddx * ddx + ddy * ddy)
    prog.AddQuadraticCost(objective)

    for index, region in enumerate(regions):
        prog.SetInitialGuess(control_points[index, 0], region.seed_world.x)
        prog.SetInitialGuess(control_points[index, 1], region.seed_world.y)

    result = deps["Solve"](prog)
    if not _solution_success(result):
        return (), result, counts

    solution = result.GetSolution(control_points)
    control_polyline = tuple(WorldPoint(float(point[0]), float(point[1])) for point in solution)
    return _resample_polyline(control_polyline, sample_count=sample_count), result, counts


def _control_point_constraint_summary(
    sampled_points: tuple[WorldPoint, ...],
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    solver_counts: dict[str, int],
) -> dict[str, Any]:
    summary = mark_direction_cone_backend_enforced(
        build_direction_cone_constraint_summary(sampled_points, regions),
        enforcing_backend=CONTROL_POINT_ENFORCING_BACKEND,
        solver_constraint_count=solver_counts["solver_constraint_count"],
    )
    summary.update(
        {
            "trajectory_parameterization": CONTROL_POINT_PARAMETERIZATION,
            "control_point_count": len(regions),
            "derivative_proxy": DERIVATIVE_PROXY,
            "derivative_constraint_count": solver_counts["derivative_constraint_count"],
            "control_point_region_containment_count": solver_counts[
                "control_point_region_containment_count"
            ],
            "start_goal_constraint_count": solver_counts["start_goal_constraint_count"],
            "objective_terms": list(OBJECTIVE_TERMS),
        }
    )
    return summary


def _not_evaluated_summary(reason: str) -> dict[str, Any]:
    summary = direction_cone_not_evaluated_summary(reason)
    summary.update(
        {
            "trajectory_parameterization": CONTROL_POINT_PARAMETERIZATION,
            "control_point_count": 0,
            "derivative_proxy": DERIVATIVE_PROXY,
            "derivative_constraint_count": 0,
            "control_point_region_containment_count": 0,
            "start_goal_constraint_count": 0,
            "objective_terms": list(OBJECTIVE_TERMS),
        }
    )
    return summary
