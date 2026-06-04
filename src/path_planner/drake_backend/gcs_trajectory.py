from __future__ import annotations

from math import hypot
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint

from .models import ConvexRegionSequenceItem, ConvexRegionSequenceReport, GcsTrajectoryReport

PYDRAKE_GCS_BACKEND = "pydrake_gcs"


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
        hpolyhedra = [_to_hpolyhedron(deps, region) for region in regions]
    except Exception as exc:
        return _attempted_failure(
            result_status=f"{type(exc).__name__}: {exc}",
            reason="invalid_hpolyhedron",
            region_count=convex_region_sequence_report.region_count,
        )

    start = _region_seed_vector(regions[0])
    goal = _region_seed_vector(regions[-1])
    try:
        trajectory, result = _solve_gcs_path(deps, hpolyhedra, start, goal)
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

    sampled_points = _sample_trajectory(trajectory, sample_count=sample_count)
    collision_count = _sample_collision_count(grid, sampled_points)
    path_length = _sampled_path_length(sampled_points)
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
        )
    return GcsTrajectoryReport(
        attempted=True,
        success=True,
        backend=PYDRAKE_GCS_BACKEND,
        result_status=result_status,
        reason="gcs_trajectory_solution_found",
        sample_count=len(sampled_points),
        collision_count=0,
        path_length=path_length,
        region_count=convex_region_sequence_report.region_count,
        sampled_points=sampled_points,
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
    )


def _load_gcs_dependencies() -> dict[str, Any]:
    from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point
    from pydrake.planning import GcsTrajectoryOptimization

    return {
        "GraphOfConvexSetsOptions": GraphOfConvexSetsOptions,
        "GcsTrajectoryOptimization": GcsTrajectoryOptimization,
        "HPolyhedron": HPolyhedron,
        "Point": Point,
    }


def _to_hpolyhedron(deps: dict[str, Any], region: ConvexRegionSequenceItem) -> Any:
    a_matrix = np.asarray(region.hpolyhedron_a, dtype=float, order="F")
    b_vector = np.asarray(region.hpolyhedron_b, dtype=float)
    if a_matrix.ndim != 2 or a_matrix.shape[1] != 2 or a_matrix.shape[0] != b_vector.shape[0]:
        raise ValueError("region hpolyhedron must have A shape (m, 2) and b shape (m,)")
    hpolyhedron = deps["HPolyhedron"](a_matrix, b_vector)
    if hpolyhedron.IsEmpty():
        raise ValueError("region hpolyhedron is empty")
    return hpolyhedron


def _solve_gcs_path(deps: dict[str, Any], hpolyhedra: list[Any], start: np.ndarray, goal: np.ndarray) -> tuple[Any, Any]:
    optimizer = deps["GcsTrajectoryOptimization"](2)
    source = optimizer.AddRegions([deps["Point"](start)], 0, name="start")
    edges = [(index, index + 1) for index in range(len(hpolyhedra) - 1)]
    corridor = optimizer.AddRegions(hpolyhedra, edges, 1, name="corridor")
    target = optimizer.AddRegions([deps["Point"](goal)], 0, name="goal")
    optimizer.AddEdges(source, corridor, edges_between_regions=[(0, 0)])
    optimizer.AddEdges(corridor, target, edges_between_regions=[(len(hpolyhedra) - 1, 0)])
    optimizer.AddPathLengthCost(1.0)
    optimizer.AddPathEnergyCost(0.05)
    return optimizer.SolvePath(source, target, deps["GraphOfConvexSetsOptions"]())


def _region_seed_vector(region: ConvexRegionSequenceItem) -> np.ndarray:
    return np.asarray([region.seed_world.x, region.seed_world.y], dtype=float)


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
