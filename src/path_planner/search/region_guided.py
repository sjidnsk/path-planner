from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, PlanDiagnostics, PlanRequest, PlanResult
from path_planner.regions import RegionGraphReport
from path_planner.search.astar import AStarPlanner
from path_planner.search.planning_grid import PlanningGrid, STANDARD_GRID_ASTAR

ASTAR_BACKEND = "astar"
REGION_GRAPH_GUIDED_BACKEND = "region_graph_guided"


@dataclass(frozen=True)
class RegionGraphGuidedPlanReport:
    requested_backend: str
    selected_backend: str
    status: str
    fallback_reason: str | None
    segment_count: int
    skeleton_cells: tuple[Cell, ...]
    baseline_path_cost: float | None
    candidate_path_cost: float | None
    baseline_cell_count: int
    candidate_cell_count: int
    region_graph_status: str | None
    region_graph_source: str | None
    start_goal_connected: bool | None
    candidate_status: str

    def to_dict(self) -> dict[str, Any]:
        candidate_delta = (
            None
            if self.baseline_path_cost is None or self.candidate_path_cost is None
            else self.candidate_path_cost - self.baseline_path_cost
        )
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "segment_count": self.segment_count,
            "skeleton_cells": [cell.to_list() for cell in self.skeleton_cells],
            "baseline_path": {
                "path_cost": self.baseline_path_cost,
                "cell_count": self.baseline_cell_count,
            },
            "candidate_path": {
                "path_cost": self.candidate_path_cost,
                "cell_count": self.candidate_cell_count,
            },
            "comparison": {
                "candidate_cost_delta": candidate_delta,
                "candidate_cell_count_delta": self.candidate_cell_count - self.baseline_cell_count,
                "path_changed": self.selected_backend == REGION_GRAPH_GUIDED_BACKEND,
            },
            "region_graph_candidate": {
                "status": self.candidate_status,
                "fallback_reason": self.fallback_reason,
                "region_graph_status": self.region_graph_status,
                "region_graph_source": self.region_graph_source,
                "start_goal_connected": self.start_goal_connected,
            },
        }


@dataclass(frozen=True)
class RegionGraphGuidedPlanOutcome:
    result: PlanResult
    report: RegionGraphGuidedPlanReport


class RegionGraphGuidedPlanner:
    def __init__(self, *, planner: AStarPlanner | None = None, improvement_epsilon: float = 1e-9) -> None:
        self._planner = planner or AStarPlanner()
        self._improvement_epsilon = improvement_epsilon

    def plan(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        *,
        baseline_result: PlanResult,
        region_graph_report: RegionGraphReport,
    ) -> RegionGraphGuidedPlanOutcome:
        skeleton, fallback_reason = self._build_skeleton(request, region_graph_report)
        if fallback_reason is not None:
            return self._fallback(
                baseline_result,
                region_graph_report,
                fallback_reason=fallback_reason,
                skeleton_cells=skeleton,
                segment_count=0,
                candidate=None,
                candidate_status="fallback",
            )

        candidate, segment_count, segment_failure = self._run_segments(grid, request, skeleton)
        if segment_failure is not None:
            return self._fallback(
                baseline_result,
                region_graph_report,
                fallback_reason=segment_failure,
                skeleton_cells=skeleton,
                segment_count=segment_count,
                candidate=candidate,
                candidate_status="fallback",
            )

        if candidate is None:
            return self._fallback(
                baseline_result,
                region_graph_report,
                fallback_reason="region_graph_invalid",
                skeleton_cells=skeleton,
                segment_count=segment_count,
                candidate=None,
                candidate_status="fallback",
            )

        if self._candidate_is_better(candidate, baseline_result):
            return RegionGraphGuidedPlanOutcome(
                result=candidate,
                report=self._report(
                    baseline_result,
                    region_graph_report,
                    selected_backend=REGION_GRAPH_GUIDED_BACKEND,
                    status="selected",
                    fallback_reason=None,
                    skeleton_cells=skeleton,
                    segment_count=segment_count,
                    candidate=candidate,
                    candidate_status="selected",
                ),
            )

        return self._fallback(
            baseline_result,
            region_graph_report,
            fallback_reason="region_graph_candidate_not_better",
            skeleton_cells=skeleton,
            segment_count=segment_count,
            candidate=candidate,
            candidate_status="fallback",
        )

    def _build_skeleton(
        self,
        request: PlanRequest,
        report: RegionGraphReport,
    ) -> tuple[tuple[Cell, ...], str | None]:
        if report.status != "ok" or report.graph is None or not report.graph.regions:
            return (), "region_graph_invalid"
        if report.quality_metrics.get("start_goal_connected") is False:
            return (), "region_graph_disconnected"
        region_path = self._region_path(report)
        if not region_path:
            return (), "region_graph_disconnected"

        cells: list[Cell] = [request.start]
        for region in region_path:
            if region.center_cell != cells[-1] and region.center_cell != request.goal:
                cells.append(region.center_cell)
        if request.goal != cells[-1]:
            cells.append(request.goal)
        if len(cells) < 2:
            return tuple(cells), "region_graph_invalid"
        return tuple(cells), None

    def _region_path(self, report: RegionGraphReport):
        graph = report.graph
        if graph is None or not graph.regions:
            return ()
        region_by_id = {region.region_id: region for region in graph.regions}
        start_id = graph.regions[0].region_id
        goal_id = graph.regions[-1].region_id
        adjacency: dict[int, set[int]] = {region.region_id: set() for region in graph.regions}
        for edge in graph.edges:
            adjacency.setdefault(edge.from_region_id, set()).add(edge.to_region_id)
            adjacency.setdefault(edge.to_region_id, set()).add(edge.from_region_id)

        frontier: deque[int] = deque([start_id])
        came_from: dict[int, int | None] = {start_id: None}
        while frontier:
            current = frontier.popleft()
            if current == goal_id:
                break
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    frontier.append(neighbor)
        if goal_id not in came_from:
            return ()

        ids: list[int] = [goal_id]
        while came_from[ids[-1]] is not None:
            ids.append(came_from[ids[-1]])
        ids.reverse()
        return tuple(region_by_id[region_id] for region_id in ids)

    def _run_segments(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        skeleton: tuple[Cell, ...],
    ) -> tuple[PlanResult | None, int, str | None]:
        segments: list[PlanResult] = []
        stitched: list[Cell] = []
        for start, goal in zip(skeleton[:-1], skeleton[1:]):
            segment_request = PlanRequest(
                start=start,
                goal=goal,
                neighbor_policy=request.neighbor_policy,
                prevent_corner_cutting=request.prevent_corner_cutting,
                max_iterations=request.max_iterations,
            )
            segment = self._planner.plan(grid, segment_request)
            if not segment.success:
                return None, len(segments), "segment_astar_failed"
            segments.append(segment)
            stitched.extend(segment.path_cells if not stitched else segment.path_cells[1:])

        if not stitched:
            return None, 0, "region_graph_invalid"
        return self._candidate_from_segments(grid, request, tuple(stitched), segments), len(segments), None

    def _candidate_from_segments(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        path_cells: tuple[Cell, ...],
        segments: list[PlanResult],
    ) -> PlanResult:
        expanded_cells: list[Cell] = []
        max_frontier_size = 0
        runtime_ms = 0.0
        for segment in segments:
            expanded_cells.extend(segment.diagnostics.expanded_cells)
            max_frontier_size = max(max_frontier_size, segment.diagnostics.max_frontier_size)
            runtime_ms += segment.diagnostics.runtime_ms
        path_world = tuple(grid.spec.cell_to_world(cell) for cell in path_cells)
        diagnostics = self._diagnostics_from_segments(
            grid,
            request=request,
            path_cells=path_cells,
            expanded_cells=tuple(expanded_cells),
            max_frontier_size=max_frontier_size,
            runtime_ms=runtime_ms,
        )
        return PlanResult(
            success=True,
            path_cells=path_cells,
            path_world=path_world,
            total_cost=sum(segment.total_cost for segment in segments),
            expanded_count=len(expanded_cells),
            failure_reason=None,
            diagnostics=diagnostics,
        )

    def _diagnostics_from_segments(
        self,
        grid: CostGrid | PlanningGrid,
        *,
        request: PlanRequest,
        path_cells: tuple[Cell, ...],
        expanded_cells: tuple[Cell, ...],
        max_frontier_size: int,
        runtime_ms: float,
    ) -> PlanDiagnostics:
        passable_cost = grid.cost[grid.passable_mask]
        metadata = self._search_metadata(grid)
        return PlanDiagnostics(
            runtime_ms=runtime_ms,
            max_frontier_size=max_frontier_size,
            path_length_m=self._path_length(path_cells, grid.spec.resolution),
            expanded_cells=expanded_cells,
            cost_min=float(np.min(passable_cost)) if passable_cost.size else None,
            cost_max=float(np.max(passable_cost)) if passable_cost.size else None,
            cost_mean=float(np.mean(passable_cost)) if passable_cost.size else None,
            neighbor_policy=request.neighbor_policy.value,
            prevent_corner_cutting=request.prevent_corner_cutting,
            search_mode=REGION_GRAPH_GUIDED_BACKEND,
            passable_source=metadata["passable_source"],
            platform_key=metadata["platform_key"],
            original_blocked_count=metadata["original_blocked_count"],
            inflated_blocked_count=metadata["inflated_blocked_count"],
            footprint_radius_m=metadata["footprint_radius_m"],
            terrain_layers=tuple(metadata["terrain_layers"]),
        )

    def _search_metadata(self, grid: CostGrid | PlanningGrid) -> dict[str, Any]:
        if isinstance(grid, PlanningGrid):
            payload = grid.search_metadata()
            return dict(payload)
        blocked_count = int(np.count_nonzero(~grid.passable_mask))
        return {
            "passable_source": "original_passable_mask",
            "platform_key": None,
            "original_blocked_count": blocked_count,
            "inflated_blocked_count": blocked_count,
            "footprint_radius_m": None,
            "terrain_layers": (),
            "search_mode": STANDARD_GRID_ASTAR,
        }

    def _candidate_is_better(self, candidate: PlanResult, baseline: PlanResult) -> bool:
        if not baseline.success:
            return True
        if candidate.path_cells == baseline.path_cells:
            return False
        return candidate.total_cost < baseline.total_cost - self._improvement_epsilon

    def _fallback(
        self,
        baseline_result: PlanResult,
        report: RegionGraphReport,
        *,
        fallback_reason: str,
        skeleton_cells: tuple[Cell, ...],
        segment_count: int,
        candidate: PlanResult | None,
        candidate_status: str,
    ) -> RegionGraphGuidedPlanOutcome:
        return RegionGraphGuidedPlanOutcome(
            result=baseline_result,
            report=self._report(
                baseline_result,
                report,
                selected_backend=ASTAR_BACKEND,
                status="fallback",
                fallback_reason=fallback_reason,
                skeleton_cells=skeleton_cells,
                segment_count=segment_count,
                candidate=candidate,
                candidate_status=candidate_status,
            ),
        )

    def _report(
        self,
        baseline_result: PlanResult,
        report: RegionGraphReport,
        *,
        selected_backend: str,
        status: str,
        fallback_reason: str | None,
        skeleton_cells: tuple[Cell, ...],
        segment_count: int,
        candidate: PlanResult | None,
        candidate_status: str,
    ) -> RegionGraphGuidedPlanReport:
        return RegionGraphGuidedPlanReport(
            requested_backend=REGION_GRAPH_GUIDED_BACKEND,
            selected_backend=selected_backend,
            status=status,
            fallback_reason=fallback_reason,
            segment_count=segment_count,
            skeleton_cells=skeleton_cells,
            baseline_path_cost=baseline_result.total_cost if math.isfinite(baseline_result.total_cost) else None,
            candidate_path_cost=(
                candidate.total_cost if candidate is not None and math.isfinite(candidate.total_cost) else None
            ),
            baseline_cell_count=len(baseline_result.path_cells),
            candidate_cell_count=0 if candidate is None else len(candidate.path_cells),
            region_graph_status=report.status,
            region_graph_source=report.region_source,
            start_goal_connected=report.quality_metrics.get("start_goal_connected"),
            candidate_status=candidate_status,
        )

    def _path_length(self, path: tuple[Cell, ...], resolution: float) -> float:
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(b.x - a.x, b.y - a.y) * resolution for a, b in zip(path[:-1], path[1:]))
