from __future__ import annotations

from math import hypot, isfinite

from path_planner.core import Cell, CostGrid, PlanResult, WorldPoint
from path_planner.postprocess import PostprocessResult

from .gcs_diagnostics import build_gcs_cost_summary
from .models import GcsGeometricCandidateReport, GcsMotionFeasibilityReport, GcsTrajectoryReport

GCS_CANDIDATE_SELECTED_REASON = "gcs_candidate_quality_improved"


def build_gcs_geometric_candidate_report(
    grid: CostGrid,
    result: PlanResult,
    postprocess: PostprocessResult | None,
    gcs_trajectory_report: GcsTrajectoryReport | None,
    gcs_motion_feasibility_report: GcsMotionFeasibilityReport | None = None,
    *,
    high_cost_threshold: float = 3.0,
    duplicate_overlap_threshold: float = 0.95,
    improvement_epsilon: float = 1e-6,
) -> GcsGeometricCandidateReport:
    if gcs_trajectory_report is None:
        return _unavailable("gcs_report_missing")
    if not result.success or not result.path_cells:
        return _unavailable(
            "unsupported_route_replacement",
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
        )
    if not gcs_trajectory_report.success:
        if gcs_trajectory_report.reason == "sampled_trajectory_collision":
            return _unavailable(
                "sampled_trajectory_collision",
                collision_count=gcs_trajectory_report.collision_count,
                constraint_summary=_constraint_summary(
                    gcs_trajectory_report,
                    gcs_motion_feasibility_report,
                ),
            )
        return _unavailable(
            "gcs_trajectory_failed",
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
        )

    sampled_points = gcs_trajectory_report.sampled_points
    if not sampled_points:
        return _unavailable(
            "gcs_trajectory_failed",
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
        )
    direction_cone_blocker = _direction_cone_blocker(gcs_trajectory_report)
    if direction_cone_blocker is not None:
        return _unavailable(
            direction_cone_blocker,
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
        )
    if (
        gcs_motion_feasibility_report is not None
        and gcs_motion_feasibility_report.evaluated
        and gcs_motion_feasibility_report.feasibility_status == "infeasible"
    ):
        return _unavailable(
            "motion_infeasible",
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
        )

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
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
            cost_summary=_candidate_cost_summary(
                grid,
                sampled_points,
                candidate,
                high_cost_threshold=high_cost_threshold,
                baseline_cost=None,
                postprocess_cost=None,
                cost_delta_vs_baseline=None,
                cost_delta_vs_postprocess=None,
                baseline_high_cost_exposure=None,
                postprocess_high_cost_exposure=None,
                high_cost_exposure_delta_vs_baseline=None,
                high_cost_exposure_delta_vs_postprocess=None,
                candidate_decision="blocked",
                decision_reason="sampled_trajectory_collision",
                quality_gate=_quality_gate_summary(
                    cost_delta_vs_baseline=None,
                    cost_delta_vs_postprocess=None,
                    overlap_ratio=None,
                    high_cost_exposure_delta_vs_baseline=None,
                    high_cost_exposure_delta_vs_postprocess=None,
                    duplicate_overlap_threshold=duplicate_overlap_threshold,
                    improvement_epsilon=improvement_epsilon,
                ),
            ),
        )

    baseline_cost = _finite_float(result.total_cost)
    if baseline_cost is None:
        baseline_cost = _cell_path_cost(grid, result.path_cells)
    postprocess_cost = _postprocess_path_cost(grid, postprocess)
    baseline_high_cost_exposure = _cell_path_high_cost_exposure(
        grid,
        result.path_cells,
        high_cost_threshold=high_cost_threshold,
    )
    postprocess_high_cost_exposure = _postprocess_high_cost_exposure(
        grid,
        postprocess,
        high_cost_threshold=high_cost_threshold,
    )
    cost_delta_vs_baseline = candidate.path_cost - baseline_cost
    cost_delta_vs_postprocess = None if postprocess_cost is None else candidate.path_cost - postprocess_cost
    high_cost_exposure_delta_vs_baseline = candidate.high_cost_exposure - baseline_high_cost_exposure
    high_cost_exposure_delta_vs_postprocess = (
        None
        if postprocess_high_cost_exposure is None
        else candidate.high_cost_exposure - postprocess_high_cost_exposure
    )
    overlap_ratio = _baseline_overlap_ratio(candidate.cells, result.path_cells)
    candidate_cost_improved = (
        cost_delta_vs_baseline < -improvement_epsilon
        and (cost_delta_vs_postprocess is None or cost_delta_vs_postprocess < -improvement_epsilon)
    )

    high_cost_exposure_worse = (
        high_cost_exposure_delta_vs_baseline > improvement_epsilon
        or (
            high_cost_exposure_delta_vs_postprocess is not None
            and high_cost_exposure_delta_vs_postprocess > improvement_epsilon
        )
    )
    if candidate_cost_improved and high_cost_exposure_worse:
        return GcsGeometricCandidateReport(
            attempted=True,
            available=True,
            selected=False,
            selection_reason=None,
            fallback_reason="high_cost_exposure",
            path_length=candidate.path_length,
            path_cost=candidate.path_cost,
            collision_count=0,
            high_cost_exposure=candidate.high_cost_exposure,
            baseline_overlap_ratio=overlap_ratio,
            cost_delta_vs_baseline=cost_delta_vs_baseline,
            cost_delta_vs_postprocess=cost_delta_vs_postprocess,
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
            cost_summary=_candidate_cost_summary(
                grid,
                sampled_points,
                candidate,
                high_cost_threshold=high_cost_threshold,
                baseline_cost=baseline_cost,
                postprocess_cost=postprocess_cost,
                cost_delta_vs_baseline=cost_delta_vs_baseline,
                cost_delta_vs_postprocess=cost_delta_vs_postprocess,
                baseline_high_cost_exposure=baseline_high_cost_exposure,
                postprocess_high_cost_exposure=postprocess_high_cost_exposure,
                high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
                high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
                candidate_decision="blocked",
                decision_reason="high_cost_exposure",
                quality_gate=_quality_gate_summary(
                    cost_delta_vs_baseline=cost_delta_vs_baseline,
                    cost_delta_vs_postprocess=cost_delta_vs_postprocess,
                    overlap_ratio=overlap_ratio,
                    high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
                    high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
                    duplicate_overlap_threshold=duplicate_overlap_threshold,
                    improvement_epsilon=improvement_epsilon,
                ),
            ),
        )

    if candidate_cost_improved:
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
            constraint_summary=_constraint_summary(
                gcs_trajectory_report,
                gcs_motion_feasibility_report,
            ),
            cost_summary=_candidate_cost_summary(
                grid,
                sampled_points,
                candidate,
                high_cost_threshold=high_cost_threshold,
                baseline_cost=baseline_cost,
                postprocess_cost=postprocess_cost,
                cost_delta_vs_baseline=cost_delta_vs_baseline,
                cost_delta_vs_postprocess=cost_delta_vs_postprocess,
                baseline_high_cost_exposure=baseline_high_cost_exposure,
                postprocess_high_cost_exposure=postprocess_high_cost_exposure,
                high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
                high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
                candidate_decision="selected",
                decision_reason=GCS_CANDIDATE_SELECTED_REASON,
                quality_gate=_quality_gate_summary(
                    cost_delta_vs_baseline=cost_delta_vs_baseline,
                    cost_delta_vs_postprocess=cost_delta_vs_postprocess,
                    overlap_ratio=overlap_ratio,
                    high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
                    high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
                    duplicate_overlap_threshold=duplicate_overlap_threshold,
                    improvement_epsilon=improvement_epsilon,
                ),
            ),
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
        constraint_summary=_constraint_summary(
            gcs_trajectory_report,
            gcs_motion_feasibility_report,
        ),
        cost_summary=_candidate_cost_summary(
            grid,
            sampled_points,
            candidate,
            high_cost_threshold=high_cost_threshold,
            baseline_cost=baseline_cost,
            postprocess_cost=postprocess_cost,
            cost_delta_vs_baseline=cost_delta_vs_baseline,
            cost_delta_vs_postprocess=cost_delta_vs_postprocess,
            baseline_high_cost_exposure=baseline_high_cost_exposure,
            postprocess_high_cost_exposure=postprocess_high_cost_exposure,
            high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
            high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
            candidate_decision="blocked",
            decision_reason=fallback_reason,
            quality_gate=_quality_gate_summary(
                cost_delta_vs_baseline=cost_delta_vs_baseline,
                cost_delta_vs_postprocess=cost_delta_vs_postprocess,
                overlap_ratio=overlap_ratio,
                high_cost_exposure_delta_vs_baseline=high_cost_exposure_delta_vs_baseline,
                high_cost_exposure_delta_vs_postprocess=high_cost_exposure_delta_vs_postprocess,
                duplicate_overlap_threshold=duplicate_overlap_threshold,
                improvement_epsilon=improvement_epsilon,
            ),
        ),
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


def _unavailable(
    reason: str,
    *,
    collision_count: int = 0,
    constraint_summary: dict | None = None,
    cost_summary: dict | None = None,
) -> GcsGeometricCandidateReport:
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
        constraint_summary=constraint_summary or _empty_constraint_summary(),
        cost_summary=cost_summary or _empty_candidate_cost_summary(),
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


def _cell_path_high_cost_exposure(
    grid: CostGrid,
    cells: tuple[Cell, ...],
    *,
    high_cost_threshold: float,
) -> float:
    total = 0.0
    previous: Cell | None = None
    for cell in cells:
        if cell == previous:
            continue
        previous = cell
        if grid.is_passable(cell):
            total += max(grid.cost_at(cell) - high_cost_threshold, 0.0)
    return float(total)


def _postprocess_high_cost_exposure(
    grid: CostGrid,
    postprocess: PostprocessResult | None,
    *,
    high_cost_threshold: float,
) -> float | None:
    if postprocess is None:
        return None
    if postprocess.smoothed_path.cells:
        return _cell_path_high_cost_exposure(
            grid,
            postprocess.smoothed_path.cells,
            high_cost_threshold=high_cost_threshold,
        )
    if postprocess.smoothed_path.world:
        metrics = _path_metrics(
            grid,
            postprocess.smoothed_path.world,
            high_cost_threshold=high_cost_threshold,
        )
        return metrics.high_cost_exposure
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


def _direction_cone_blocker(gcs_trajectory_report: GcsTrajectoryReport) -> str | None:
    direction_cone = gcs_trajectory_report.constraint_summary
    if direction_cone.get("constraint_model") != "direction_cone":
        return "direction_cone_not_evaluated"
    if not direction_cone.get("evaluated", False):
        return "direction_cone_not_evaluated"
    if not direction_cone.get("backend_enforced", False):
        return "direction_cone_not_backend_enforced"
    if int(direction_cone.get("violation_count") or 0) > 0:
        return "direction_cone_constraint_violation"
    return None


def _constraint_summary(
    gcs_trajectory_report: GcsTrajectoryReport | None,
    gcs_motion_feasibility_report: GcsMotionFeasibilityReport | None,
) -> dict:
    motion = {"evaluated": False, "status": "not_requested", "fallback_reason": None}
    if gcs_motion_feasibility_report is not None:
        motion = {
            "evaluated": gcs_motion_feasibility_report.evaluated,
            "status": gcs_motion_feasibility_report.feasibility_status,
            "fallback_reason": gcs_motion_feasibility_report.fallback_reason,
            "motion_model": gcs_motion_feasibility_report.motion_model,
        }
    return {
        "schema_version": "gcs_candidate_constraint_summary/v1",
        "direction_cone": dict(gcs_trajectory_report.constraint_summary) if gcs_trajectory_report is not None else {},
        "motion_feasibility": motion,
    }


def _empty_constraint_summary() -> dict:
    return {
        "schema_version": "gcs_candidate_constraint_summary/v1",
        "direction_cone": {},
        "motion_feasibility": {"evaluated": False, "status": "not_requested", "fallback_reason": None},
    }


def _candidate_cost_summary(
    grid: CostGrid,
    sampled_points: tuple[WorldPoint, ...],
    candidate: _PathMetrics,
    *,
    high_cost_threshold: float,
    baseline_cost: float | None,
    postprocess_cost: float | None,
    cost_delta_vs_baseline: float | None,
    cost_delta_vs_postprocess: float | None,
    baseline_high_cost_exposure: float | None,
    postprocess_high_cost_exposure: float | None,
    high_cost_exposure_delta_vs_baseline: float | None,
    high_cost_exposure_delta_vs_postprocess: float | None,
    candidate_decision: str,
    decision_reason: str,
    quality_gate: dict,
) -> dict:
    summary = build_gcs_cost_summary(grid, sampled_points, high_cost_threshold=high_cost_threshold)
    summary["schema_version"] = "gcs_candidate_cost_summary/v1"
    summary["terrain_path_cost"] = candidate.path_cost
    summary["path_length"] = candidate.path_length
    summary["high_cost_exposure"] = candidate.high_cost_exposure
    summary["baseline_path_cost"] = baseline_cost
    summary["postprocess_path_cost"] = postprocess_cost
    summary["cost_delta_vs_baseline"] = cost_delta_vs_baseline
    summary["cost_delta_vs_postprocess"] = cost_delta_vs_postprocess
    summary["baseline_high_cost_exposure"] = baseline_high_cost_exposure
    summary["postprocess_high_cost_exposure"] = postprocess_high_cost_exposure
    summary["high_cost_exposure_delta_vs_baseline"] = high_cost_exposure_delta_vs_baseline
    summary["high_cost_exposure_delta_vs_postprocess"] = high_cost_exposure_delta_vs_postprocess
    summary["candidate_decision"] = candidate_decision
    summary["decision_reason"] = decision_reason
    summary["quality_gate"] = quality_gate
    return summary


def _empty_candidate_cost_summary() -> dict:
    return {
        "schema_version": "gcs_candidate_cost_summary/v1",
        "path_length": None,
        "terrain_path_cost": None,
        "high_cost_exposure": None,
        "terrain_cost_source": "not_evaluated",
        "high_cost_threshold": None,
        "sampled_cell_count": 0,
        "blocked_sample_count": 0,
        "energy_proxy": None,
        "smoothness_proxy": None,
        "baseline_path_cost": None,
        "postprocess_path_cost": None,
        "cost_delta_vs_baseline": None,
        "cost_delta_vs_postprocess": None,
        "baseline_high_cost_exposure": None,
        "postprocess_high_cost_exposure": None,
        "high_cost_exposure_delta_vs_baseline": None,
        "high_cost_exposure_delta_vs_postprocess": None,
        "candidate_decision": "blocked",
        "decision_reason": "cost_not_evaluated",
        "quality_gate": {},
    }


def _quality_gate_summary(
    *,
    cost_delta_vs_baseline: float | None,
    cost_delta_vs_postprocess: float | None,
    overlap_ratio: float | None,
    high_cost_exposure_delta_vs_baseline: float | None,
    high_cost_exposure_delta_vs_postprocess: float | None,
    duplicate_overlap_threshold: float,
    improvement_epsilon: float,
) -> dict:
    baseline_delta_improved = (
        None if cost_delta_vs_baseline is None else cost_delta_vs_baseline < -improvement_epsilon
    )
    postprocess_delta_improved = (
        None if cost_delta_vs_postprocess is None else cost_delta_vs_postprocess < -improvement_epsilon
    )
    duplicate_with_baseline = (
        None
        if overlap_ratio is None or cost_delta_vs_baseline is None
        else overlap_ratio >= duplicate_overlap_threshold
        and abs(cost_delta_vs_baseline) <= improvement_epsilon
    )
    high_cost_exposure_not_worse = (
        None
        if high_cost_exposure_delta_vs_baseline is None
        else high_cost_exposure_delta_vs_baseline <= improvement_epsilon
        and (
            high_cost_exposure_delta_vs_postprocess is None
            or high_cost_exposure_delta_vs_postprocess <= improvement_epsilon
        )
    )
    return {
        "baseline_delta_improved": baseline_delta_improved,
        "postprocess_delta_improved": postprocess_delta_improved,
        "high_cost_exposure_not_worse": high_cost_exposure_not_worse,
        "duplicate_with_baseline": duplicate_with_baseline,
        "baseline_overlap_ratio": overlap_ratio,
        "duplicate_overlap_threshold": float(duplicate_overlap_threshold),
        "improvement_epsilon": float(improvement_epsilon),
    }


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return number
