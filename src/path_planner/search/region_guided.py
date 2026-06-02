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
SAMPLED_REGION_PATH_BACKEND = "sampled_region_path"
SAMPLED_REGION_PATH_REPORT_SCHEMA_VERSION = "sampled_region_path_report/v1"


@dataclass(frozen=True)
class SampledRegionPathReport:
    schema_version: str
    status: str
    fallback_reason: str | None
    region_sequence: tuple[int, ...]
    sample_cells: tuple[Cell, ...]
    path_cells: tuple[Cell, ...]
    edge_transition_count: int
    collision_free: bool
    baseline_path_cost: float | None
    candidate_path_cost: float | None
    baseline_path_length_m: float | None
    candidate_path_length_m: float | None
    baseline_high_cost_exposure: float | None
    candidate_high_cost_exposure: float | None
    baseline_tracking_proxy: float | None
    candidate_tracking_proxy: float | None

    @property
    def sample_count(self) -> int:
        return len(self.sample_cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "region_sequence": list(self.region_sequence),
            "sample_count": self.sample_count,
            "sample_cells": [cell.to_list() for cell in self.sample_cells],
            "path_cells": [cell.to_list() for cell in self.path_cells],
            "edge_transition_count": self.edge_transition_count,
            "safety_checks": {
                "collision_free": self.collision_free,
                "passable_cell_count": len(self.path_cells) if self.collision_free else 0,
                "blocked_cell_count": 0 if self.collision_free else 1,
            },
            "candidate_comparison": {
                "baseline_path_cost": self.baseline_path_cost,
                "candidate_path_cost": self.candidate_path_cost,
                "candidate_cost_delta": _delta(self.candidate_path_cost, self.baseline_path_cost),
                "baseline_path_length_m": self.baseline_path_length_m,
                "candidate_path_length_m": self.candidate_path_length_m,
                "path_length_delta_m": _delta(self.candidate_path_length_m, self.baseline_path_length_m),
                "baseline_high_cost_exposure": self.baseline_high_cost_exposure,
                "candidate_high_cost_exposure": self.candidate_high_cost_exposure,
                "high_cost_exposure_delta": _delta(
                    self.candidate_high_cost_exposure,
                    self.baseline_high_cost_exposure,
                ),
                "baseline_tracking_proxy": self.baseline_tracking_proxy,
                "candidate_tracking_proxy": self.candidate_tracking_proxy,
                "tracking_proxy_delta": _delta(self.candidate_tracking_proxy, self.baseline_tracking_proxy),
            },
        }


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
    sampled_region_path_report: SampledRegionPathReport | None = None

    def to_dict(self) -> dict[str, Any]:
        candidate_delta = (
            None
            if self.baseline_path_cost is None or self.candidate_path_cost is None
            else self.candidate_path_cost - self.baseline_path_cost
        )
        payload = {
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
                "path_changed": self.selected_backend != ASTAR_BACKEND,
            },
            "region_graph_candidate": {
                "status": self.candidate_status,
                "fallback_reason": self.fallback_reason,
                "region_graph_status": self.region_graph_status,
                "region_graph_source": self.region_graph_source,
                "start_goal_connected": self.start_goal_connected,
            },
        }
        if self.sampled_region_path_report is not None:
            payload["sampled_region_path_report"] = self.sampled_region_path_report.to_dict()
        return payload


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
            sampled_report = self._sampled_report(
                grid,
                baseline_result,
                status="fallback",
                fallback_reason=fallback_reason,
                region_path=(),
                sample_cells=(),
                path_cells=(),
                collision_free=False,
                candidate=None,
            )
            return self._fallback(
                baseline_result,
                region_graph_report,
                fallback_reason=fallback_reason,
                skeleton_cells=skeleton,
                segment_count=0,
                candidate=None,
                candidate_status="fallback",
                sampled_region_path_report=sampled_report,
            )

        region_path = self._region_path(region_graph_report)
        sampled_candidate, sampled_report = self._run_sampled_region_path(
            grid,
            request,
            baseline_result=baseline_result,
            region_path=region_path,
        )
        if sampled_candidate is not None and self._sampled_candidate_is_selectable(sampled_report, baseline_result):
            return RegionGraphGuidedPlanOutcome(
                result=sampled_candidate,
                report=self._report(
                    baseline_result,
                    region_graph_report,
                    selected_backend=SAMPLED_REGION_PATH_BACKEND,
                    status="selected",
                    fallback_reason=None,
                    skeleton_cells=skeleton,
                    segment_count=max(len(sampled_report.sample_cells) - 1, 0),
                    candidate=sampled_candidate,
                    candidate_status="selected",
                    sampled_region_path_report=sampled_report,
                ),
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
                sampled_region_path_report=sampled_report,
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
                sampled_region_path_report=sampled_report,
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
                    sampled_region_path_report=sampled_report,
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
            sampled_region_path_report=sampled_report,
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

    def _run_sampled_region_path(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        *,
        baseline_result: PlanResult,
        region_path: tuple[Any, ...],
    ) -> tuple[PlanResult | None, SampledRegionPathReport]:
        if not region_path:
            return None, self._sampled_report(
                grid,
                baseline_result,
                status="fallback",
                fallback_reason="region_sequence_missing",
                region_path=(),
                sample_cells=(),
                path_cells=(),
                collision_free=False,
                candidate=None,
            )

        samples: list[Cell] = []
        for index, region in enumerate(region_path):
            preferred = request.start if index == 0 else request.goal if index == len(region_path) - 1 else None
            sample = self._sample_region_cell(grid, region, preferred=preferred)
            if sample is None:
                return None, self._sampled_report(
                    grid,
                    baseline_result,
                    status="fallback",
                    fallback_reason="region_sample_unavailable",
                    region_path=region_path,
                    sample_cells=tuple(samples),
                    path_cells=(),
                    collision_free=False,
                    candidate=None,
                )
            if not samples or sample != samples[-1]:
                samples.append(sample)

        path_cells, collision_reason = self._stitch_sampled_path(grid, request, tuple(samples))
        if collision_reason is not None:
            return None, self._sampled_report(
                grid,
                baseline_result,
                status="fallback",
                fallback_reason=collision_reason,
                region_path=region_path,
                sample_cells=tuple(samples),
                path_cells=path_cells,
                collision_free=False,
                candidate=None,
            )

        candidate = self._candidate_from_sampled_path(grid, request, path_cells)
        sampled_report = self._sampled_report(
            grid,
            baseline_result,
            status="selected",
            fallback_reason=None,
            region_path=region_path,
            sample_cells=tuple(samples),
            path_cells=path_cells,
            collision_free=True,
            candidate=candidate,
        )
        if self._sampled_candidate_is_selectable(sampled_report, baseline_result):
            return candidate, sampled_report
        return None, self._sampled_report(
            grid,
            baseline_result,
            status="fallback",
            fallback_reason="sampled_candidate_not_better",
            region_path=region_path,
            sample_cells=tuple(samples),
            path_cells=path_cells,
            collision_free=True,
            candidate=candidate,
        )

    def _sample_region_cell(
        self,
        grid: CostGrid | PlanningGrid,
        region: Any,
        *,
        preferred: Cell | None,
    ) -> Cell | None:
        if preferred is not None and self._cell_in_region(preferred, region) and grid.is_passable(preferred):
            return preferred
        candidates: list[Cell] = []
        for y in range(region.min_cell.y, region.max_cell.y + 1):
            for x in range(region.min_cell.x, region.max_cell.x + 1):
                cell = Cell(x, y)
                if grid.spec.in_bounds(cell) and grid.is_passable(cell):
                    candidates.append(cell)
        if not candidates:
            return None
        anchor = preferred if preferred is not None else region.center_cell
        return min(
            candidates,
            key=lambda cell: (
                grid.cost_at(cell),
                math.hypot(cell.x - anchor.x, cell.y - anchor.y),
                cell.y,
                cell.x,
            ),
        )

    def _stitch_sampled_path(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        samples: tuple[Cell, ...],
    ) -> tuple[tuple[Cell, ...], str | None]:
        if len(samples) < 2:
            return samples, "edge_transition_sample_failed"
        stitched: list[Cell] = []
        for start, goal in zip(samples[:-1], samples[1:]):
            segment = self._line_cells(start, goal)
            if not segment:
                return tuple(stitched), "edge_transition_sample_failed"
            for previous, current in zip(segment[:-1], segment[1:]):
                if not self._transition_is_safe(grid, request, previous, current):
                    return tuple(stitched), "sampled_path_collision"
            for cell in segment:
                if not grid.is_passable(cell):
                    return tuple(stitched), "sampled_path_collision"
            stitched.extend(segment if not stitched else segment[1:])
        return tuple(stitched), None

    def _candidate_from_sampled_path(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        path_cells: tuple[Cell, ...],
    ) -> PlanResult:
        diagnostics = self._diagnostics_from_segments(
            grid,
            request=request,
            path_cells=path_cells,
            expanded_cells=path_cells,
            max_frontier_size=max(len(path_cells), 1),
            runtime_ms=0.0,
        )
        return PlanResult(
            success=True,
            path_cells=path_cells,
            path_world=tuple(grid.spec.cell_to_world(cell) for cell in path_cells),
            total_cost=self._path_cost(grid, path_cells),
            expanded_count=len(path_cells),
            failure_reason=None,
            diagnostics=diagnostics,
        )

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

    def _sampled_candidate_is_selectable(
        self,
        report: SampledRegionPathReport,
        baseline: PlanResult,
    ) -> bool:
        if report.status != "selected" or report.candidate_path_cost is None:
            return False
        if not baseline.success:
            return True
        if report.candidate_path_cost < baseline.total_cost - self._improvement_epsilon:
            return True
        if report.candidate_path_cost > baseline.total_cost + self._improvement_epsilon:
            return False
        exposure_delta = _delta(report.candidate_high_cost_exposure, report.baseline_high_cost_exposure)
        tracking_delta = _delta(report.candidate_tracking_proxy, report.baseline_tracking_proxy)
        length_delta = _delta(report.candidate_path_length_m, report.baseline_path_length_m)
        return any(
            delta is not None and delta < -self._improvement_epsilon
            for delta in (exposure_delta, tracking_delta, length_delta)
        )

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
        sampled_region_path_report: SampledRegionPathReport | None,
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
                sampled_region_path_report=sampled_region_path_report,
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
        sampled_region_path_report: SampledRegionPathReport | None,
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
            sampled_region_path_report=sampled_region_path_report,
        )

    def _path_length(self, path: tuple[Cell, ...], resolution: float) -> float:
        if len(path) < 2:
            return 0.0
        return sum(math.hypot(b.x - a.x, b.y - a.y) * resolution for a, b in zip(path[:-1], path[1:]))

    def _path_cost(self, grid: CostGrid | PlanningGrid, path: tuple[Cell, ...]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for previous, current in zip(path[:-1], path[1:]):
            total += math.hypot(current.x - previous.x, current.y - previous.y) * grid.cost_at(current)
        return float(total)

    def _line_cells(self, start: Cell, goal: Cell) -> tuple[Cell, ...]:
        steps = max(abs(goal.x - start.x), abs(goal.y - start.y))
        if steps == 0:
            return (start,)
        cells: list[Cell] = []
        for index in range(steps + 1):
            x = round(start.x + (goal.x - start.x) * index / steps)
            y = round(start.y + (goal.y - start.y) * index / steps)
            cell = Cell(int(x), int(y))
            if not cells or cell != cells[-1]:
                cells.append(cell)
        return tuple(cells)

    def _transition_is_safe(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        previous: Cell,
        current: Cell,
    ) -> bool:
        if not grid.is_passable(current):
            return False
        dx = current.x - previous.x
        dy = current.y - previous.y
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        if dx != 0 and dy != 0 and request.prevent_corner_cutting:
            side_a = Cell(previous.x + dx, previous.y)
            side_b = Cell(previous.x, previous.y + dy)
            if not grid.is_passable(side_a) and not grid.is_passable(side_b):
                return False
        return True

    def _cell_in_region(self, cell: Cell, region: Any) -> bool:
        return (
            region.min_cell.x <= cell.x <= region.max_cell.x
            and region.min_cell.y <= cell.y <= region.max_cell.y
        )

    def _sampled_report(
        self,
        grid: CostGrid | PlanningGrid,
        baseline_result: PlanResult,
        *,
        status: str,
        fallback_reason: str | None,
        region_path: tuple[Any, ...],
        sample_cells: tuple[Cell, ...],
        path_cells: tuple[Cell, ...],
        collision_free: bool,
        candidate: PlanResult | None,
    ) -> SampledRegionPathReport:
        baseline_path_cost = baseline_result.total_cost if math.isfinite(baseline_result.total_cost) else None
        candidate_path_cost = candidate.total_cost if candidate is not None and math.isfinite(candidate.total_cost) else None
        baseline_path_length = baseline_result.diagnostics.path_length_m if baseline_result.success else None
        candidate_path_length = candidate.diagnostics.path_length_m if candidate is not None else None
        return SampledRegionPathReport(
            schema_version=SAMPLED_REGION_PATH_REPORT_SCHEMA_VERSION,
            status=status,
            fallback_reason=fallback_reason,
            region_sequence=tuple(int(region.region_id) for region in region_path),
            sample_cells=sample_cells,
            path_cells=path_cells,
            edge_transition_count=max(len(sample_cells) - 1, 0),
            collision_free=collision_free,
            baseline_path_cost=baseline_path_cost,
            candidate_path_cost=candidate_path_cost,
            baseline_path_length_m=baseline_path_length,
            candidate_path_length_m=candidate_path_length,
            baseline_high_cost_exposure=(
                self._high_cost_exposure(grid, baseline_result.path_cells) if baseline_result.success else None
            ),
            candidate_high_cost_exposure=(
                self._high_cost_exposure(grid, candidate.path_cells) if candidate is not None else None
            ),
            baseline_tracking_proxy=(
                self._tracking_proxy(baseline_result.path_cells) if baseline_result.success else None
            ),
            candidate_tracking_proxy=self._tracking_proxy(candidate.path_cells) if candidate is not None else None,
        )

    def _high_cost_exposure(self, grid: CostGrid | PlanningGrid, path: tuple[Cell, ...]) -> float:
        threshold = 3.0
        exposure = 0.0
        for previous, current in zip(path[:-1], path[1:]):
            midpoint = Cell(
                int(round((previous.x + current.x) / 2.0)),
                int(round((previous.y + current.y) / 2.0)),
            )
            if grid.spec.in_bounds(midpoint) and grid.cost_at(midpoint) >= threshold:
                exposure += math.hypot(current.x - previous.x, current.y - previous.y) * grid.spec.resolution
        return float(exposure)

    def _tracking_proxy(self, path: tuple[Cell, ...]) -> float:
        if len(path) < 3:
            return 0.0
        headings = [
            math.atan2(current.y - previous.y, current.x - previous.x)
            for previous, current in zip(path[:-1], path[1:])
        ]
        total = 0.0
        for previous, current in zip(headings[:-1], headings[1:]):
            delta = (current - previous + math.pi) % (2.0 * math.pi) - math.pi
            total += delta * delta
        return float(total)


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)
