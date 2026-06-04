from __future__ import annotations

from math import hypot, isfinite

from path_planner.core import Cell, CostGrid, PlanResult, WorldPoint
from path_planner.postprocess import PostprocessResult

from .models import GcsGeometricCandidateReport, GcsTrajectoryReport

GCS_CANDIDATE_SELECTED_REASON = "gcs_candidate_quality_improved"


def build_gcs_geometric_candidate_report(
    grid: CostGrid,
    result: PlanResult,
    postprocess: PostprocessResult | None,
    gcs_trajectory_report: GcsTrajectoryReport | None,
    *,
    high_cost_threshold: float = 3.0,
    duplicate_overlap_threshold: float = 0.95,
    improvement_epsilon: float = 1e-6,
) -> GcsGeometricCandidateReport:
    if gcs_trajectory_report is None:
        return _unavailable("gcs_report_missing")
    if not result.success or not result.path_cells:
        return _unavailable("unsupported_route_replacement")
    if not gcs_trajectory_report.success:
        if gcs_trajectory_report.reason == "sampled_trajectory_collision":
            return _unavailable(
                "sampled_trajectory_collision",
                collision_count=gcs_trajectory_report.collision_count,
            )
        return _unavailable("gcs_trajectory_failed")

    sampled_points = gcs_trajectory_report.sampled_points
    if not sampled_points:
        return _unavailable("gcs_trajectory_failed")

    candidate = _path_metrics(
        grid,
        sampled_points,
        high_cost_threshold=high_cost_threshold,
    )
    collision_count = max(candidate.collision_count, gcs_trajectory_report.collision_count)
    if collision_count > 0:
        return GcsGeometricCandidateReport(
            attempted=True,
            available=False,
            selected=False,
            selection_reason=None,
            fallback_reason="sampled_trajectory_collision",
            path_length=candidate.path_length,
            path_cost=candidate.path_cost,
            collision_count=collision_count,
            high_cost_exposure=candidate.high_cost_exposure,
            baseline_overlap_ratio=_baseline_overlap_ratio(candidate.cells, result.path_cells),
            cost_delta_vs_baseline=None,
            cost_delta_vs_postprocess=None,
        )

    baseline_cost = _finite_float(result.total_cost)
    if baseline_cost is None:
        baseline_cost = _cell_path_cost(grid, result.path_cells)
    postprocess_cost = _postprocess_path_cost(grid, postprocess)
    cost_delta_vs_baseline = candidate.path_cost - baseline_cost
    cost_delta_vs_postprocess = None if postprocess_cost is None else candidate.path_cost - postprocess_cost
    overlap_ratio = _baseline_overlap_ratio(candidate.cells, result.path_cells)

    if (
        cost_delta_vs_baseline < -improvement_epsilon
        and (cost_delta_vs_postprocess is None or cost_delta_vs_postprocess < -improvement_epsilon)
    ):
        return GcsGeometricCandidateReport(
            attempted=True,
            available=True,
            selected=True,
            selection_reason=GCS_CANDIDATE_SELECTED_REASON,
            fallback_reason=None,
            path_length=candidate.path_length,
            path_cost=candidate.path_cost,
            collision_count=0,
            high_cost_exposure=candidate.high_cost_exposure,
            baseline_overlap_ratio=overlap_ratio,
            cost_delta_vs_baseline=cost_delta_vs_baseline,
            cost_delta_vs_postprocess=cost_delta_vs_postprocess,
        )

    fallback_reason = _fallback_reason(
        cost_delta_vs_baseline=cost_delta_vs_baseline,
        cost_delta_vs_postprocess=cost_delta_vs_postprocess,
        overlap_ratio=overlap_ratio,
        duplicate_overlap_threshold=duplicate_overlap_threshold,
        improvement_epsilon=improvement_epsilon,
    )
    return GcsGeometricCandidateReport(
        attempted=True,
        available=True,
        selected=False,
        selection_reason=None,
        fallback_reason=fallback_reason,
        path_length=candidate.path_length,
        path_cost=candidate.path_cost,
        collision_count=0,
        high_cost_exposure=candidate.high_cost_exposure,
        baseline_overlap_ratio=overlap_ratio,
        cost_delta_vs_baseline=cost_delta_vs_baseline,
        cost_delta_vs_postprocess=cost_delta_vs_postprocess,
    )


class _PathMetrics:
    def __init__(
        self,
        *,
        cells: tuple[Cell, ...],
        path_length: float,
        path_cost: float,
        collision_count: int,
        high_cost_exposure: float,
    ) -> None:
        self.cells = cells
        self.path_length = path_length
        self.path_cost = path_cost
        self.collision_count = collision_count
        self.high_cost_exposure = high_cost_exposure


def _unavailable(reason: str, *, collision_count: int = 0) -> GcsGeometricCandidateReport:
    return GcsGeometricCandidateReport(
        attempted=True,
        available=False,
        selected=False,
        selection_reason=None,
        fallback_reason=reason,
        path_length=None,
        path_cost=None,
        collision_count=collision_count,
        high_cost_exposure=None,
        baseline_overlap_ratio=None,
        cost_delta_vs_baseline=None,
        cost_delta_vs_postprocess=None,
    )


def _path_metrics(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    *,
    high_cost_threshold: float,
) -> _PathMetrics:
    cells: list[Cell] = []
    collision_count = 0
    path_cost = 0.0
    high_cost_exposure = 0.0
    previous_cell: Cell | None = None
    for point in points:
        cell = grid.spec.world_to_cell(point)
        if not grid.is_passable(cell):
            collision_count += 1
            continue
        if previous_cell == cell:
            continue
        cells.append(cell)
        previous_cell = cell
        cost = grid.cost_at(cell)
        path_cost += cost
        high_cost_exposure += max(cost - high_cost_threshold, 0.0)
    return _PathMetrics(
        cells=tuple(cells),
        path_length=_world_path_length(points),
        path_cost=float(path_cost),
        collision_count=collision_count,
        high_cost_exposure=float(high_cost_exposure),
    )


def _world_path_length(points: tuple[WorldPoint, ...]) -> float:
    total = 0.0
    for first, second in zip(points[:-1], points[1:]):
        total += hypot(second.x - first.x, second.y - first.y)
    return float(total)


def _cell_path_cost(grid: CostGrid, cells: tuple[Cell, ...]) -> float:
    total = 0.0
    previous: Cell | None = None
    for cell in cells:
        if cell == previous:
            continue
        previous = cell
        if grid.is_passable(cell):
            total += grid.cost_at(cell)
    return float(total)


def _postprocess_path_cost(grid: CostGrid, postprocess: PostprocessResult | None) -> float | None:
    if postprocess is None:
        return None
    if postprocess.smoothed_path.cells:
        return _cell_path_cost(grid, postprocess.smoothed_path.cells)
    if postprocess.smoothed_path.world:
        metrics = _path_metrics(
            grid,
            postprocess.smoothed_path.world,
            high_cost_threshold=float("inf"),
        )
        return metrics.path_cost
    return None


def _baseline_overlap_ratio(candidate_cells: tuple[Cell, ...], baseline_cells: tuple[Cell, ...]) -> float:
    if not candidate_cells:
        return 0.0
    baseline = set(baseline_cells)
    overlap = sum(1 for cell in set(candidate_cells) if cell in baseline)
    return float(overlap / len(set(candidate_cells)))


def _fallback_reason(
    *,
    cost_delta_vs_baseline: float,
    cost_delta_vs_postprocess: float | None,
    overlap_ratio: float,
    duplicate_overlap_threshold: float,
    improvement_epsilon: float,
) -> str:
    if overlap_ratio >= duplicate_overlap_threshold and abs(cost_delta_vs_baseline) <= improvement_epsilon:
        return "path_duplicate_with_baseline"
    if cost_delta_vs_baseline > improvement_epsilon:
        return "cost_dominated"
    if cost_delta_vs_postprocess is not None and cost_delta_vs_postprocess > improvement_epsilon:
        return "cost_dominated"
    return "no_quality_gain"


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return number
