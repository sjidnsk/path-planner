from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest, PlanResult
from path_planner.optimization import TrajectoryOptimizationResult, merge_tracking_comparison
from path_planner.postprocess import PostprocessResult
from path_planner.regions import RegionGraphReport
from path_planner.tracking import TrackingSimulationResult

if TYPE_CHECKING:
    from path_planner.drake_backend import (
        ConvexRegionSequenceReport,
        GcsCurvatureConstrainedCandidateReport,
        GcsGeometricCandidateReport,
        GcsMotionFeasibilityReport,
        GcsTrajectoryReport,
        IrisRegionReport,
    )

REQUEST_SCHEMA_VERSION = "path-planner-request/v1"


def load_plan_input(path: str | Path) -> tuple[CostGrid, PlanRequest]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {REQUEST_SCHEMA_VERSION}")
    grid_raw = raw["grid"]
    origin = grid_raw.get("origin", [0.0, 0.0])
    spec = GridSpec(
        width=int(grid_raw["width"]),
        height=int(grid_raw["height"]),
        resolution=float(grid_raw["resolution"]),
        origin=(float(origin[0]), float(origin[1])),
        frame_id=str(grid_raw.get("frame_id", "map")),
    )
    grid = CostGrid(
        spec=spec,
        cost=np.asarray(raw["cost"], dtype=float),
        passable_mask=np.asarray(raw["passable_mask"], dtype=bool),
        metadata={"adapter": "json", "source": str(path)},
    )
    request = PlanRequest(
        start=Cell(int(raw["start"][0]), int(raw["start"][1])),
        goal=Cell(int(raw["goal"][0]), int(raw["goal"][1])),
        max_iterations=int(raw.get("max_iterations", 100_000)),
    )
    return grid, request


def route_result_to_json_dict(
    result: PlanResult,
    spec: GridSpec,
    *,
    postprocess: PostprocessResult | None = None,
    tracking_simulation: TrackingSimulationResult | None = None,
    trajectory_optimization: TrajectoryOptimizationResult | None = None,
    optimized_tracking_simulation: TrackingSimulationResult | None = None,
    region_graph_report: RegionGraphReport | None = None,
    iris_region_report: IrisRegionReport | None = None,
    convex_region_sequence_report: ConvexRegionSequenceReport | None = None,
    gcs_trajectory_report: GcsTrajectoryReport | None = None,
    gcs_candidate_report: GcsGeometricCandidateReport | None = None,
    gcs_motion_feasibility_report: GcsMotionFeasibilityReport | None = None,
    gcs_curvature_constrained_candidate_report: GcsCurvatureConstrainedCandidateReport | None = None,
    planning_backend_report: Any | None = None,
) -> dict[str, Any]:
    payload = result.to_route_dict(spec)
    if planning_backend_report is not None:
        payload["planning_backend_report"] = planning_backend_report.to_dict()
    if postprocess is not None:
        payload["postprocess"] = postprocess.to_dict()
    if region_graph_report is not None:
        payload["region_graph_report"] = region_graph_report.to_dict()
    if iris_region_report is not None:
        payload["iris_region_report"] = iris_region_report.to_dict()
    if convex_region_sequence_report is not None:
        payload.update(convex_region_sequence_report.to_route_fields())
    if gcs_trajectory_report is not None:
        payload.update(gcs_trajectory_report.to_route_fields())
    if gcs_candidate_report is not None:
        payload.update(gcs_candidate_report.to_route_fields())
    if gcs_motion_feasibility_report is not None:
        payload.update(gcs_motion_feasibility_report.to_route_fields())
    if gcs_curvature_constrained_candidate_report is not None:
        payload.update(gcs_curvature_constrained_candidate_report.to_route_fields())
    if tracking_simulation is not None:
        payload["tracking_simulation_report"] = tracking_simulation.to_dict()
    if trajectory_optimization is not None:
        payload["trajectory_optimization_report"] = merge_tracking_comparison(
            trajectory_optimization.to_dict(),
            tracking_simulation,
            optimized_tracking_simulation,
        )
    if optimized_tracking_simulation is not None:
        payload["optimized_tracking_simulation_report"] = optimized_tracking_simulation.to_dict()
    return payload
