from __future__ import annotations

from math import hypot
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint

from .gcs_diagnostics import (
    build_direction_cone_constraint_summary,
    build_gcs_cost_summary,
    direction_cone_edge_parameters,
    direction_cone_not_evaluated_summary,
    empty_gcs_cost_summary,
    mark_direction_cone_backend_enforced,
)
from .models import ConvexRegionSequenceItem, ConvexRegionSequenceReport, GcsTrajectoryReport

PYDRAKE_GCS_BACKEND = "pydrake_direction_cone_program"
DIRECTION_CONE_ENFORCING_BACKEND = "pydrake_mathematical_program"


def build_gcs_trajectory_report(
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
        sampled_points, result, solver_constraint_count = _solve_gcs_path(
            deps,
            grid,
            regions,
            sample_count=sample_count,
        )
    except Exception as exc:
        reason = _solver_exception_reason(exc)
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason=reason,
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
    constraint_summary = mark_direction_cone_backend_enforced(
        build_direction_cone_constraint_summary(sampled_points, regions),
        enforcing_backend=DIRECTION_CONE_ENFORCING_BACKEND,
        solver_constraint_count=solver_constraint_count,
    )
    cost_summary = build_gcs_cost_summary(grid, sampled_points)
    if collision_count > 0:
        return GcsTrajectoryReport(
            attempted=True,
            success=False,
            backend=PYDRAKE_GCS_BACKEND,
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
        backend=PYDRAKE_GCS_BACKEND,
        result_status=result_status,
        reason="direction_cone_solution_found",
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
        backend=PYDRAKE_GCS_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=direction_cone_not_evaluated_summary(reason),
        cost_summary=empty_gcs_cost_summary(),
    )


def _attempted_failure(*, result_status: str, reason: str, region_count: int) -> GcsTrajectoryReport:
    return GcsTrajectoryReport(
        attempted=True,
        success=False,
        backend=PYDRAKE_GCS_BACKEND,
        result_status=result_status,
        reason=reason,
        sample_count=0,
        collision_count=0,
        path_length=0.0,
        region_count=region_count,
        constraint_summary=direction_cone_not_evaluated_summary(reason),
        cost_summary=empty_gcs_cost_summary(),
    )


def _load_gcs_dependencies() -> dict[str, Any]:
    from pydrake.solvers import MathematicalProgram, Solve

    return {
        "MathematicalProgram": MathematicalProgram,
        "Solve": Solve,
    }


def _validate_region_hpolyhedra(regions: tuple[ConvexRegionSequenceItem, ...]) -> None:
    for region in regions:
        _region_matrices(region)


def _region_matrices(region: ConvexRegionSequenceItem) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = np.asarray(region.hpolyhedron_a, dtype=float, order="F")
    b_vector = np.asarray(region.hpolyhedron_b, dtype=float)
    if a_matrix.ndim != 2 or a_matrix.shape[1] != 2 or a_matrix.shape[0] != b_vector.shape[0]:
        raise ValueError("region hpolyhedron must have A shape (m, 2) and b shape (m,)")
    if not np.all(np.isfinite(a_matrix)) or not np.all(np.isfinite(b_vector)):
        raise ValueError("region hpolyhedron must be finite")
    return a_matrix, b_vector


def _solve_gcs_path(
    deps: dict[str, Any],
    grid: CostGrid,
    regions: tuple[ConvexRegionSequenceItem, ...],
    *,
    sample_count: int,
) -> tuple[tuple[WorldPoint, ...], Any, int]:
    if len(regions) < 2:
        raise ValueError("direction_cone backend requires at least two regions")

    prog = deps["MathematicalProgram"]()
    variables = prog.NewContinuousVariables(len(regions), 2, "p")
    constraint_count = 0

    for index, region in enumerate(regions):
        a_matrix, b_vector = _region_matrices(region)
        for row, bound in zip(a_matrix, b_vector):
            prog.AddLinearConstraint(row[0] * variables[index, 0] + row[1] * variables[index, 1], -np.inf, bound)
            constraint_count += 1

    first_seed = regions[0].seed_world
    last_seed = regions[-1].seed_world
    for variable, value in (
        (variables[0, 0], first_seed.x),
        (variables[0, 1], first_seed.y),
        (variables[-1, 0], last_seed.x),
        (variables[-1, 1], last_seed.y),
    ):
        prog.AddBoundingBoxConstraint(value, value, variable)
        constraint_count += 1

    for index, (first, second) in enumerate(zip(regions[:-1], regions[1:])):
        edge_parameters = direction_cone_edge_parameters(first, second)
        if edge_parameters["seed_distance_m"] <= 0.0:
            raise ValueError("direction_cone reference segment is degenerate")
        tangent = np.asarray(edge_parameters["tangent"], dtype=float)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        delta_x = variables[index + 1, 0] - variables[index, 0]
        delta_y = variables[index + 1, 1] - variables[index, 1]
        forward = tangent[0] * delta_x + tangent[1] * delta_y
        lateral = normal[0] * delta_x + normal[1] * delta_y
        eta = float(edge_parameters["eta"])
        rho = float(edge_parameters["rho_lower_bound_m"])

        prog.AddLinearConstraint(forward, rho, np.inf)
        prog.AddLinearConstraint(lateral - eta * forward, -np.inf, 0.0)
        prog.AddLinearConstraint(-lateral - eta * forward, -np.inf, 0.0)
        constraint_count += 3

    objective = 0.0
    for index in range(len(regions) - 1):
        dx = variables[index + 1, 0] - variables[index, 0]
        dy = variables[index + 1, 1] - variables[index, 1]
        objective += dx * dx + dy * dy
    for index, region in enumerate(regions):
        anchor = _low_cost_anchor(grid, region)
        objective += 0.1 * ((variables[index, 0] - anchor.x) ** 2 + (variables[index, 1] - anchor.y) ** 2)
    for index in range(1, len(regions) - 1):
        ddx = variables[index + 1, 0] - 2.0 * variables[index, 0] + variables[index - 1, 0]
        ddy = variables[index + 1, 1] - 2.0 * variables[index, 1] + variables[index - 1, 1]
        objective += 0.2 * (ddx * ddx + ddy * ddy)
    prog.AddQuadraticCost(objective)

    for index, region in enumerate(regions):
        prog.SetInitialGuess(variables[index, 0], region.seed_world.x)
        prog.SetInitialGuess(variables[index, 1], region.seed_world.y)

    result = deps["Solve"](prog)
    if not _solution_success(result):
        return (), result, constraint_count

    solution = result.GetSolution(variables)
    waypoints = tuple(WorldPoint(float(point[0]), float(point[1])) for point in solution)
    return _resample_polyline(waypoints, sample_count=sample_count), result, constraint_count


def _region_seed_vector(region: ConvexRegionSequenceItem) -> np.ndarray:
    return np.asarray([region.seed_world.x, region.seed_world.y], dtype=float)


def _seed_distance(first: ConvexRegionSequenceItem, second: ConvexRegionSequenceItem) -> float:
    return hypot(second.seed_world.x - first.seed_world.x, second.seed_world.y - first.seed_world.y)


def _seed_tangent(first: ConvexRegionSequenceItem, second: ConvexRegionSequenceItem) -> np.ndarray | None:
    distance = _seed_distance(first, second)
    if distance <= 0.0:
        return None
    return np.asarray(
        [
            (second.seed_world.x - first.seed_world.x) / distance,
            (second.seed_world.y - first.seed_world.y) / distance,
        ],
        dtype=float,
    )


def _low_cost_anchor(grid: CostGrid, region: ConvexRegionSequenceItem) -> WorldPoint:
    best_cell: Cell | None = None
    best_cost: float | None = None
    for y in range(region.min_cell.y, region.max_cell.y + 1):
        for x in range(region.min_cell.x, region.max_cell.x + 1):
            cell = Cell(x, y)
            if not grid.spec.in_bounds(cell) or not grid.is_passable(cell):
                continue
            point = grid.spec.cell_to_world(cell)
            if not _point_in_region(point, region):
                continue
            cost = grid.cost_at(cell)
            if best_cost is None or cost < best_cost:
                best_cell = cell
                best_cost = cost
    if best_cell is None:
        return region.seed_world
    return grid.spec.cell_to_world(best_cell)


def _point_in_region(point: WorldPoint, region: ConvexRegionSequenceItem, *, tolerance: float = 1.0e-9) -> bool:
    a_matrix, b_vector = _region_matrices(region)
    vector = np.asarray([point.x, point.y], dtype=float)
    return bool(np.all(a_matrix @ vector <= b_vector + tolerance))


def _resample_polyline(points: tuple[WorldPoint, ...], *, sample_count: int) -> tuple[WorldPoint, ...]:
    sample_count = max(int(sample_count), 2)
    if len(points) <= 1:
        return points
    segment_lengths = [
        hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(points[:-1], points[1:])
    ]
    total_length = sum(segment_lengths)
    if total_length <= 0.0:
        return tuple(points[0] for _ in range(sample_count))

    targets = np.linspace(0.0, total_length, sample_count)
    sampled: list[WorldPoint] = []
    segment_index = 0
    segment_start_distance = 0.0
    for target in targets:
        while (
            segment_index < len(segment_lengths) - 1
            and segment_start_distance + segment_lengths[segment_index] < target
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        first = points[segment_index]
        second = points[segment_index + 1]
        length = segment_lengths[segment_index]
        alpha = 0.0 if length <= 0.0 else (target - segment_start_distance) / length
        sampled.append(
            WorldPoint(
                first.x + alpha * (second.x - first.x),
                first.y + alpha * (second.y - first.y),
            )
        )
    return tuple(sampled)


def _has_region_adjacency(regions: tuple[ConvexRegionSequenceItem, ...]) -> bool:
    for first, second in zip(regions[:-1], regions[1:]):
        if (
            first.max_world.x < second.min_world.x
            or second.max_world.x < first.min_world.x
            or first.max_world.y < second.min_world.y
            or second.max_world.y < first.min_world.y
        ):
            return False
    return True


def _sample_trajectory(trajectory: Any, *, sample_count: int) -> tuple[WorldPoint, ...]:
    start_time = float(trajectory.start_time())
    end_time = float(trajectory.end_time())
    if end_time <= start_time:
        times = [start_time for _ in range(sample_count)]
    else:
        times = np.linspace(start_time, end_time, sample_count)
    points: list[WorldPoint] = []
    for time in times:
        value = np.asarray(trajectory.value(float(time)), dtype=float).reshape(-1)
        if value.shape[0] < 2:
            raise ValueError("GCS trajectory value must contain at least 2 coordinates")
        points.append(WorldPoint(float(value[0]), float(value[1])))
    return tuple(points)


def _sample_collision_count(grid: CostGrid, points: tuple[WorldPoint, ...]) -> int:
    collisions = 0
    for point in points:
        cell = grid.spec.world_to_cell(point)
        if not grid.is_passable(cell):
            collisions += 1
    return collisions


def _sampled_path_length(points: tuple[WorldPoint, ...]) -> float:
    total = 0.0
    for first, second in zip(points[:-1], points[1:]):
        total += hypot(second.x - first.x, second.y - first.y)
    return float(total)


def _solution_success(result: Any) -> bool:
    try:
        return bool(result.is_success())
    except Exception:
        return False


def _solution_status(result: Any) -> str:
    try:
        return str(result.get_solution_result())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _solver_exception_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "infeasible" in text:
        return "solver_infeasible"
    if "solver" in text and ("unavailable" in text or "not available" in text or "not enabled" in text):
        return "solver_unavailable"
    if "continuity" in text:
        return "unsupported_continuity"
    return "solver_failed"


def _solver_result_reason(result_status: str) -> str:
    text = result_status.lower()
    if "infeasible" in text:
        return "solver_infeasible"
    if "solver" in text and ("unavailable" in text or "not available" in text or "not enabled" in text):
        return "solver_unavailable"
    return "solver_failed"
