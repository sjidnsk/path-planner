from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, PlanDiagnostics, PlanRequest, PlanResult
from path_planner.regions import ConvexRegion, RegionEdge, RegionGraphReport
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
    start_goal_anchoring: dict[str, Any] = field(default_factory=dict)
    sample_attempts: tuple[dict[str, Any], ...] = ()
    candidate_rankings: tuple[dict[str, Any], ...] = ()

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
            "start_goal_anchoring": dict(self.start_goal_anchoring),
            "sample_attempt_count": len(self.sample_attempts),
            "sample_attempts": [dict(attempt) for attempt in self.sample_attempts],
            "candidate_rankings": [dict(ranking) for ranking in self.candidate_rankings],
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


@dataclass(frozen=True)
class _RegionPathResolution:
    region_path: tuple[Any, ...]
    start_region_id: int | None
    goal_region_id: int | None
    start_region_candidates: tuple[int, ...]
    goal_region_candidates: tuple[int, ...]
    fallback_reason: str | None
    start_classification: str
    goal_classification: str
    start_anchor_region_added: bool = False
    goal_anchor_region_added: bool = False
    start_anchor_region_connected: bool = False
    goal_anchor_region_connected: bool = False
    start_anchor_failure_reason: str | None = None
    goal_anchor_failure_reason: str | None = None


@dataclass(frozen=True)
class _SampleOption:
    region_id: int
    cell: Cell
    strategy: str
    cost: float


@dataclass(frozen=True)
class _SampledCandidate:
    strategy: str
    sample_cells: tuple[Cell, ...]
    path_cells: tuple[Cell, ...]
    candidate: PlanResult


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
        region_resolution = self._resolve_region_path(grid, request, region_graph_report)
        anchoring = self._anchoring_payload(request, region_resolution)
        skeleton, fallback_reason = self._build_skeleton(request, region_resolution)
        if fallback_reason is not None:
            sampled_report = self._sampled_report(
                grid,
                baseline_result,
                status="fallback",
                fallback_reason=fallback_reason,
                region_path=region_resolution.region_path,
                sample_cells=(),
                path_cells=(),
                collision_free=False,
                candidate=None,
                anchoring=anchoring,
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

        sampled_candidate, sampled_report = self._run_sampled_region_path(
            grid,
            request,
            baseline_result=baseline_result,
            region_path=region_resolution.region_path,
            anchoring=anchoring,
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
        resolution: _RegionPathResolution,
    ) -> tuple[tuple[Cell, ...], str | None]:
        if resolution.fallback_reason is not None:
            return (), resolution.fallback_reason
        region_path = resolution.region_path
        if not region_path:
            return (), "region_sequence_missing"

        cells: list[Cell] = [request.start]
        for region in region_path:
            if region.center_cell != cells[-1] and region.center_cell != request.goal:
                cells.append(region.center_cell)
        if request.goal != cells[-1]:
            cells.append(request.goal)
        if len(cells) < 2:
            return tuple(cells), "region_sequence_missing"
        return tuple(cells), None

    def _resolve_region_path(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        report: RegionGraphReport,
    ) -> _RegionPathResolution:
        graph = report.graph
        regions: list[Any] = list(graph.regions) if graph is not None else []
        edges: list[RegionEdge] = list(graph.edges) if graph is not None else []
        start_regions = tuple(region for region in regions if self._cell_in_region(request.start, region))
        goal_regions = tuple(region for region in regions if self._cell_in_region(request.goal, region))
        start_candidates = tuple(region.region_id for region in start_regions)
        goal_candidates = tuple(region.region_id for region in goal_regions)

        next_region_id = max((int(region.region_id) for region in regions), default=-1) + 1
        start_classification = "covered" if start_regions else self._endpoint_classification(grid, request.start, "start")
        goal_classification = "covered" if goal_regions else self._endpoint_classification(grid, request.goal, "goal")
        start_anchor_added = False
        goal_anchor_added = False
        start_anchor_connected = False
        goal_anchor_connected = False
        start_anchor_failure_reason: str | None = None
        goal_anchor_failure_reason: str | None = None

        if not start_regions:
            if start_classification != "start_outside_region_coverage":
                return _RegionPathResolution(
                    region_path=(),
                    start_region_id=None,
                    goal_region_id=None,
                    start_region_candidates=(),
                    goal_region_candidates=goal_candidates,
                    fallback_reason=start_classification,
                    start_classification=start_classification,
                    goal_classification=goal_classification,
                    start_anchor_failure_reason=start_classification,
                )
            start_anchor = self._anchor_region(grid, next_region_id, request.start)
            next_region_id += 1
            regions.append(start_anchor)
            start_regions = (start_anchor,)
            start_candidates = (start_anchor.region_id,)
            start_anchor_added = True

        if not goal_regions:
            if goal_classification != "goal_outside_region_coverage":
                return _RegionPathResolution(
                    region_path=(),
                    start_region_id=(
                        self._choose_anchor_region(request.start, start_regions).region_id if start_regions else None
                    ),
                    goal_region_id=None,
                    start_region_candidates=start_candidates,
                    goal_region_candidates=(),
                    fallback_reason=goal_classification,
                    start_classification=start_classification,
                    goal_classification=goal_classification,
                    start_anchor_region_added=start_anchor_added,
                    start_anchor_region_connected=start_anchor_connected,
                    goal_anchor_failure_reason=goal_classification,
                )
            goal_anchor = self._anchor_region(grid, next_region_id, request.goal)
            regions.append(goal_anchor)
            goal_regions = (goal_anchor,)
            goal_candidates = (goal_anchor.region_id,)
            goal_anchor_added = True

        added_anchor_ids = {
            region.region_id
            for region in (*start_regions, *goal_regions)
            if (start_anchor_added and region.center_cell == request.start)
            or (goal_anchor_added and region.center_cell == request.goal)
        }
        edges, anchor_connected = self._connect_anchor_regions(grid, request, tuple(regions), tuple(edges), added_anchor_ids)
        if start_anchor_added:
            start_anchor_connected = anchor_connected.get(start_candidates[0], False)
            if not start_anchor_connected:
                start_anchor_failure_reason = "start_anchor_region_unconnected"
        if goal_anchor_added:
            goal_anchor_connected = anchor_connected.get(goal_candidates[0], False)
            if not goal_anchor_connected:
                goal_anchor_failure_reason = "goal_anchor_region_unconnected"

        region_by_id = {region.region_id: region for region in regions}
        if not start_regions:
            goal_region_id = None
            if goal_regions:
                goal_region_id = self._choose_anchor_region(request.goal, goal_regions).region_id
            return _RegionPathResolution(
                region_path=(),
                start_region_id=None,
                goal_region_id=goal_region_id,
                start_region_candidates=(),
                goal_region_candidates=goal_candidates,
                fallback_reason="start_region_missing",
                start_classification=start_classification,
                goal_classification=goal_classification,
            )
        if not goal_regions:
            return _RegionPathResolution(
                region_path=(),
                start_region_id=self._choose_anchor_region(request.start, start_regions).region_id,
                goal_region_id=None,
                start_region_candidates=start_candidates,
                goal_region_candidates=(),
                fallback_reason="goal_region_missing",
                start_classification=start_classification,
                goal_classification=goal_classification,
            )
        start_region = self._choose_anchor_region(request.start, start_regions)
        goal_region = self._choose_anchor_region(request.goal, goal_regions)
        start_id = start_region.region_id
        goal_id = goal_region.region_id
        adjacency: dict[int, set[int]] = {region.region_id: set() for region in regions}
        for edge in edges:
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
            if goal_anchor_added:
                fallback_reason = "goal_anchor_region_unconnected"
                goal_anchor_failure_reason = fallback_reason
            elif start_anchor_added:
                fallback_reason = "start_anchor_region_unconnected"
                start_anchor_failure_reason = fallback_reason
            else:
                fallback_reason = "region_graph_disconnected"
            return _RegionPathResolution(
                region_path=(),
                start_region_id=start_id,
                goal_region_id=goal_id,
                start_region_candidates=start_candidates,
                goal_region_candidates=goal_candidates,
                fallback_reason=fallback_reason,
                start_classification=start_classification,
                goal_classification=goal_classification,
                start_anchor_region_added=start_anchor_added,
                goal_anchor_region_added=goal_anchor_added,
                start_anchor_region_connected=start_anchor_connected,
                goal_anchor_region_connected=goal_anchor_connected,
                start_anchor_failure_reason=start_anchor_failure_reason,
                goal_anchor_failure_reason=goal_anchor_failure_reason,
            )

        ids: list[int] = [goal_id]
        while came_from[ids[-1]] is not None:
            ids.append(came_from[ids[-1]])
        ids.reverse()
        region_path = tuple(region_by_id[region_id] for region_id in ids)
        return _RegionPathResolution(
            region_path=region_path,
            start_region_id=start_id,
            goal_region_id=goal_id,
            start_region_candidates=start_candidates,
            goal_region_candidates=goal_candidates,
            fallback_reason=None if region_path else "region_sequence_missing",
            start_classification=start_classification,
            goal_classification=goal_classification,
            start_anchor_region_added=start_anchor_added,
            goal_anchor_region_added=goal_anchor_added,
            start_anchor_region_connected=start_anchor_connected,
            goal_anchor_region_connected=goal_anchor_connected,
            start_anchor_failure_reason=start_anchor_failure_reason,
            goal_anchor_failure_reason=goal_anchor_failure_reason,
        )

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
        anchoring: dict[str, Any],
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
                anchoring=anchoring,
            )

        sample_attempts = self._sample_attempts(grid, request, region_path)
        candidates, rejected_rankings = self._sampled_candidates(
            grid,
            request,
            region_path=region_path,
        )
        if not candidates:
            fallback_reason = self._candidate_generation_fallback_reason(rejected_rankings)
            return None, self._sampled_report(
                grid,
                baseline_result,
                status="fallback",
                fallback_reason=fallback_reason,
                region_path=region_path,
                sample_cells=(),
                path_cells=(),
                collision_free=False,
                candidate=None,
                anchoring=anchoring,
                sample_attempts=sample_attempts,
                candidate_rankings=tuple(rejected_rankings),
            )

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item.candidate.total_cost,
                self._high_cost_exposure(grid, item.candidate.path_cells),
                self._tracking_proxy(item.candidate.path_cells),
                item.candidate.diagnostics.path_length_m,
                0 if item.strategy == "cost_aware_constrained_astar" else 1,
                len(item.sample_cells),
                item.strategy,
            ),
        )
        ranking_payload = self._candidate_rankings(
            grid,
            baseline_result,
            ranked_candidates,
            selected_index=None,
            prefix=tuple(rejected_rankings),
        )
        best = ranked_candidates[0]
        sampled_report = self._sampled_report(
            grid,
            baseline_result,
            status="selected",
            fallback_reason=None,
            region_path=region_path,
            sample_cells=best.sample_cells,
            path_cells=best.path_cells,
            collision_free=True,
            candidate=best.candidate,
            anchoring=anchoring,
            sample_attempts=sample_attempts,
            candidate_rankings=ranking_payload,
        )
        if self._sampled_candidate_is_selectable(sampled_report, baseline_result):
            selected_ranking_payload = self._candidate_rankings(
                grid,
                baseline_result,
                ranked_candidates,
                selected_index=0,
                prefix=tuple(rejected_rankings),
            )
            return best.candidate, self._sampled_report(
                grid,
                baseline_result,
                status="selected",
                fallback_reason=None,
                region_path=region_path,
                sample_cells=best.sample_cells,
                path_cells=best.path_cells,
                collision_free=True,
                candidate=best.candidate,
                anchoring=anchoring,
                sample_attempts=sample_attempts,
                candidate_rankings=selected_ranking_payload,
            )
        return None, self._sampled_report(
            grid,
            baseline_result,
            status="fallback",
            fallback_reason=self._sampled_not_better_reason(sampled_report, baseline_result),
            region_path=region_path,
            sample_cells=best.sample_cells,
            path_cells=best.path_cells,
            collision_free=True,
            candidate=best.candidate,
            anchoring=anchoring,
            sample_attempts=sample_attempts,
            candidate_rankings=ranking_payload,
        )

    def _sampled_candidates(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        *,
        region_path: tuple[Any, ...],
    ) -> tuple[list[_SampledCandidate], list[dict[str, Any]]]:
        candidates: list[_SampledCandidate] = []
        rejected: list[dict[str, Any]] = []
        cost_aware = self._cost_aware_constrained_candidate(grid, request, region_path=region_path)
        if cost_aware is None:
            rejected.append(
                {
                    "rank": None,
                    "strategy": "cost_aware_constrained_astar",
                    "status": "rejected",
                    "fallback_reason": "constrained_connector_failed",
                    "sample_count": 0,
                    "sample_cells": [],
                    "edge_transition_count": 0,
                }
            )
        else:
            candidates.append(cost_aware)
        sample_sequences = self._sample_sequences(grid, request, region_path)
        if len(region_path) > 1 and not any(strategy == "edge_adjacent" for strategy, _ in sample_sequences):
            rejected.append(
                {
                    "rank": None,
                    "strategy": "edge_adjacent",
                    "status": "rejected",
                    "fallback_reason": "edge_transition_unavailable",
                    "sample_count": 0,
                    "sample_cells": [],
                    "edge_transition_count": 0,
                }
            )
        for strategy, samples in sample_sequences:
            path_cells, collision_reason = self._stitch_sampled_path(grid, request, samples)
            if collision_reason is not None:
                rejected.append(
                    {
                        "rank": None,
                        "strategy": strategy,
                        "status": "rejected",
                        "fallback_reason": collision_reason,
                        "sample_count": len(samples),
                        "sample_cells": [cell.to_list() for cell in samples],
                        "edge_transition_count": max(len(samples) - 1, 0),
                    }
                )
                continue
            candidate = self._candidate_from_sampled_path(grid, request, path_cells)
            candidates.append(
                _SampledCandidate(
                    strategy=strategy,
                    sample_cells=samples,
                    path_cells=path_cells,
                    candidate=candidate,
                )
            )
        return candidates, rejected

    def _sample_sequences(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        region_path: tuple[Any, ...],
    ) -> tuple[tuple[str, tuple[Cell, ...]], ...]:
        sequences: list[tuple[str, tuple[Cell, ...]]] = []
        primary = self._one_sample_per_region_sequence(grid, request, region_path, strategy="preferred_low_cost")
        if primary:
            sequences.append(("preferred_low_cost", primary))
        centerline = self._one_sample_per_region_sequence(grid, request, region_path, strategy="center")
        if centerline and centerline != primary:
            sequences.append(("center", centerline))
        edge_adjacent = self._edge_adjacent_sequence(grid, request, region_path)
        if edge_adjacent and edge_adjacent not in {sample for _, sample in sequences}:
            sequences.append(("edge_adjacent", edge_adjacent))
        return tuple(sequences)

    def _one_sample_per_region_sequence(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        region_path: tuple[Any, ...],
        *,
        strategy: str,
    ) -> tuple[Cell, ...]:
        samples: list[Cell] = []
        for index, region in enumerate(region_path):
            preferred = request.start if index == 0 else request.goal if index == len(region_path) - 1 else None
            if strategy == "center":
                sample = (
                    region.center_cell
                    if grid.spec.in_bounds(region.center_cell) and grid.is_passable(region.center_cell)
                    else None
                )
                if sample is None:
                    sample = self._sample_region_cell(grid, region, preferred=preferred)
            else:
                sample = self._sample_region_cell(grid, region, preferred=preferred)
            if sample is None:
                return ()
            if not samples or sample != samples[-1]:
                samples.append(sample)
        return tuple(samples)

    def _edge_adjacent_sequence(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        region_path: tuple[Any, ...],
    ) -> tuple[Cell, ...]:
        if len(region_path) < 2:
            return self._one_sample_per_region_sequence(grid, request, region_path, strategy="preferred_low_cost")
        samples: list[Cell] = [request.start]
        for left, right in zip(region_path[:-1], region_path[1:]):
            pair = self._best_edge_transition_pair(grid, request, left, right)
            if pair is None:
                return ()
            left_cell, right_cell = pair
            if left_cell != samples[-1]:
                samples.append(left_cell)
            if right_cell != samples[-1]:
                samples.append(right_cell)
        if request.goal != samples[-1]:
            samples.append(request.goal)
        return tuple(samples)

    def _best_edge_transition_pair(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        left: Any,
        right: Any,
    ) -> tuple[Cell, Cell] | None:
        pairs: list[tuple[float, float, int, int, Cell, Cell]] = []
        for left_cell in self._region_passable_cells(grid, left):
            for right_cell in self._region_passable_cells(grid, right):
                if not self._transition_is_safe(grid, request, left_cell, right_cell):
                    continue
                pairs.append(
                    (
                        grid.cost_at(left_cell) + grid.cost_at(right_cell),
                        math.hypot(right_cell.x - left_cell.x, right_cell.y - left_cell.y),
                        left_cell.y + right_cell.y,
                        left_cell.x + right_cell.x,
                        left_cell,
                        right_cell,
                    )
            )
        if not pairs:
            return None
        _, _, _, _, left_cell, right_cell = min(pairs)
        return left_cell, right_cell

    def _sample_attempts(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        region_path: tuple[Any, ...],
    ) -> tuple[dict[str, Any], ...]:
        attempts: list[dict[str, Any]] = []
        for index, region in enumerate(region_path):
            preferred = request.start if index == 0 else request.goal if index == len(region_path) - 1 else None
            for option in self._sample_options(grid, region, preferred=preferred):
                attempts.append(
                    {
                        "kind": "region_sample",
                        "region_id": int(region.region_id),
                        "strategy": option.strategy,
                        "cell": option.cell.to_list(),
                        "cost": option.cost,
                    }
                )
        for left, right in zip(region_path[:-1], region_path[1:]):
            pair = self._best_edge_transition_pair(grid, request, left, right)
            attempts.append(
                {
                    "kind": "edge_transition",
                    "from_region_id": int(left.region_id),
                    "to_region_id": int(right.region_id),
                    "status": "available" if pair is not None else "unavailable",
                    "from_cell": None if pair is None else pair[0].to_list(),
                    "to_cell": None if pair is None else pair[1].to_list(),
                }
            )
        connector = self._cost_aware_constrained_candidate(grid, request, region_path=region_path)
        attempts.append(
            {
                "kind": "connector_attempt",
                "strategy": "cost_aware_constrained_astar",
                "status": "available" if connector is not None else "unavailable",
                "path_cell_count": 0 if connector is None else len(connector.path_cells),
            }
        )
        return tuple(attempts)

    def _cost_aware_constrained_candidate(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        *,
        region_path: tuple[Any, ...],
    ) -> _SampledCandidate | None:
        mask = self._region_sequence_mask(grid, region_path)
        if mask is None:
            return None
        constrained_grid = CostGrid(
            spec=grid.spec,
            cost=np.array(grid.cost, dtype=float, copy=True),
            passable_mask=mask,
        )
        result = self._planner.plan(
            constrained_grid,
            PlanRequest(
                start=request.start,
                goal=request.goal,
                neighbor_policy=request.neighbor_policy,
                prevent_corner_cutting=request.prevent_corner_cutting,
                max_iterations=request.max_iterations,
            ),
        )
        if not result.success:
            return None
        candidate = self._candidate_from_sampled_path(grid, request, result.path_cells)
        return _SampledCandidate(
            strategy="cost_aware_constrained_astar",
            sample_cells=result.path_cells,
            path_cells=result.path_cells,
            candidate=candidate,
        )

    def _region_sequence_mask(
        self,
        grid: CostGrid | PlanningGrid,
        region_path: tuple[Any, ...],
    ) -> np.ndarray | None:
        if not region_path:
            return None
        mask = np.zeros(grid.spec.shape, dtype=bool)
        for region in region_path:
            min_x = max(0, int(region.min_cell.x))
            max_x = min(grid.spec.width - 1, int(region.max_cell.x))
            min_y = max(0, int(region.min_cell.y))
            max_y = min(grid.spec.height - 1, int(region.max_cell.y))
            if min_x > max_x or min_y > max_y:
                continue
            mask[min_y : max_y + 1, min_x : max_x + 1] = True
        mask &= np.asarray(grid.passable_mask, dtype=bool)
        return mask if np.any(mask) else None

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

    def _sample_options(
        self,
        grid: CostGrid | PlanningGrid,
        region: Any,
        *,
        preferred: Cell | None,
    ) -> tuple[_SampleOption, ...]:
        options: list[_SampleOption] = []
        seen: set[Cell] = set()

        def add(cell: Cell, strategy: str) -> None:
            if cell in seen or not grid.spec.in_bounds(cell) or not grid.is_passable(cell):
                return
            seen.add(cell)
            options.append(
                _SampleOption(
                    region_id=int(region.region_id),
                    cell=cell,
                    strategy=strategy,
                    cost=float(grid.cost_at(cell)),
                )
            )

        if preferred is not None and self._cell_in_region(preferred, region):
            add(preferred, "preferred")
        add(region.center_cell, "center")
        anchor = preferred if preferred is not None else region.center_cell
        ranked = sorted(
            self._region_passable_cells(grid, region),
            key=lambda cell: (
                grid.cost_at(cell),
                math.hypot(cell.x - anchor.x, cell.y - anchor.y),
                cell.y,
                cell.x,
            ),
        )
        for cell in ranked[:4]:
            add(cell, "low_cost")
        return tuple(options)

    def _region_passable_cells(self, grid: CostGrid | PlanningGrid, region: Any) -> tuple[Cell, ...]:
        candidates: list[Cell] = []
        for y in range(region.min_cell.y, region.max_cell.y + 1):
            for x in range(region.min_cell.x, region.max_cell.x + 1):
                cell = Cell(x, y)
                if grid.spec.in_bounds(cell) and grid.is_passable(cell):
                    candidates.append(cell)
        return tuple(candidates)

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

    def _sampled_not_better_reason(
        self,
        report: SampledRegionPathReport,
        baseline: PlanResult,
    ) -> str:
        if report.candidate_path_cost is None:
            return "sampled_candidate_missing_metrics"
        if not baseline.success:
            return "sampled_candidate_not_selectable"
        if report.candidate_path_cost > baseline.total_cost + self._improvement_epsilon:
            if self._has_cost_aware_connector_attempt(report):
                return "region_sequence_cost_dominated"
            return "sampled_candidate_higher_cost"
        exposure_delta = _delta(report.candidate_high_cost_exposure, report.baseline_high_cost_exposure)
        tracking_delta = _delta(report.candidate_tracking_proxy, report.baseline_tracking_proxy)
        length_delta = _delta(report.candidate_path_length_m, report.baseline_path_length_m)
        quality_deltas = tuple(delta for delta in (exposure_delta, tracking_delta, length_delta) if delta is not None)
        if any(delta > self._improvement_epsilon for delta in quality_deltas):
            if self._has_cost_aware_connector_attempt(report):
                return "constrained_connector_not_better"
            return "sampled_candidate_quality_regression"
        if self._has_cost_aware_connector_attempt(report):
            return "constrained_connector_not_better"
        return "sampled_candidate_equal_cost_no_quality_gain"

    def _has_cost_aware_connector_attempt(self, report: SampledRegionPathReport) -> bool:
        for ranking in report.candidate_rankings:
            if ranking.get("strategy") == "cost_aware_constrained_astar":
                return True
        for attempt in report.sample_attempts:
            if (
                attempt.get("kind") == "connector_attempt"
                and attempt.get("strategy") == "cost_aware_constrained_astar"
            ):
                return True
        return False

    def _candidate_rankings(
        self,
        grid: CostGrid | PlanningGrid,
        baseline_result: PlanResult,
        candidates: list[_SampledCandidate],
        *,
        selected_index: int | None,
        prefix: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        rankings: list[dict[str, Any]] = [dict(item) for item in prefix]
        baseline_cost = baseline_result.total_cost if math.isfinite(baseline_result.total_cost) else None
        baseline_length = baseline_result.diagnostics.path_length_m if baseline_result.success else None
        baseline_exposure = self._high_cost_exposure(grid, baseline_result.path_cells) if baseline_result.success else None
        baseline_tracking = self._tracking_proxy(baseline_result.path_cells) if baseline_result.success else None
        for index, item in enumerate(candidates):
            candidate = item.candidate
            high_cost_exposure = self._high_cost_exposure(grid, candidate.path_cells)
            tracking_proxy = self._tracking_proxy(candidate.path_cells)
            rankings.append(
                {
                    "rank": index + 1,
                    "strategy": item.strategy,
                    "status": "selected" if selected_index == index else "candidate",
                    "fallback_reason": None,
                    "sample_count": len(item.sample_cells),
                    "sample_cells": [cell.to_list() for cell in item.sample_cells],
                    "edge_transition_count": max(len(item.sample_cells) - 1, 0),
                    "candidate_path_cost": candidate.total_cost,
                    "candidate_cost_delta": _delta(candidate.total_cost, baseline_cost),
                    "candidate_path_length_m": candidate.diagnostics.path_length_m,
                    "path_length_delta_m": _delta(candidate.diagnostics.path_length_m, baseline_length),
                    "candidate_high_cost_exposure": high_cost_exposure,
                    "high_cost_exposure_delta": _delta(high_cost_exposure, baseline_exposure),
                    "candidate_tracking_proxy": tracking_proxy,
                    "tracking_proxy_delta": _delta(tracking_proxy, baseline_tracking),
                }
            )
        return tuple(rankings)

    def _candidate_generation_fallback_reason(self, rankings: list[dict[str, Any]]) -> str:
        reasons = [str(item.get("fallback_reason")) for item in rankings if item.get("fallback_reason")]
        if any(reason == "sampled_path_collision" for reason in reasons):
            return "sampled_path_collision"
        if any(reason in {"edge_transition_sample_failed", "edge_transition_unavailable"} for reason in reasons):
            return "edge_transition_unavailable"
        if reasons:
            return reasons[0]
        return "region_sample_unavailable"

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

    def _choose_anchor_region(self, cell: Cell, regions: tuple[Any, ...]) -> Any:
        return min(
            regions,
            key=lambda region: (
                math.hypot(region.center_cell.x - cell.x, region.center_cell.y - cell.y),
                int(region.region_id),
            ),
        )

    def _endpoint_classification(
        self,
        grid: CostGrid | PlanningGrid,
        cell: Cell,
        label: str,
    ) -> str:
        if not grid.spec.in_bounds(cell):
            return f"{label}_not_passable"
        if grid.is_passable(cell):
            return f"{label}_outside_region_coverage"
        if isinstance(grid, PlanningGrid):
            original_safe = bool(grid.original_passable_mask[cell.y, cell.x])
            inflated_safe = bool(grid.inflated_passable_mask[cell.y, cell.x])
            if original_safe and not inflated_safe:
                return f"{label}_footprint_unsafe"
        return f"{label}_not_passable"

    def _anchor_region(self, grid: CostGrid | PlanningGrid, region_id: int, cell: Cell) -> ConvexRegion:
        min_world = grid.spec.cell_to_world(cell)
        max_world = grid.spec.cell_to_world(Cell(cell.x + 1, cell.y + 1))
        return ConvexRegion(
            region_id=region_id,
            source="grid_box",
            center_cell=cell,
            min_cell=cell,
            max_cell=cell,
            min_world=min_world,
            max_world=max_world,
            cell_count=1,
            fallback_reason="anchor_region",
        )

    def _connect_anchor_regions(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        regions: tuple[Any, ...],
        edges: tuple[RegionEdge, ...],
        anchor_region_ids: set[int],
    ) -> tuple[list[RegionEdge], dict[int, bool]]:
        result = list(edges)
        connected = {region_id: False for region_id in anchor_region_ids}
        existing_pairs = {
            tuple(sorted((int(edge.from_region_id), int(edge.to_region_id))))
            for edge in result
        }
        next_edge_id = max((int(edge.edge_id) for edge in result), default=-1) + 1
        for anchor in regions:
            if int(anchor.region_id) not in anchor_region_ids:
                continue
            for other in regions:
                if int(other.region_id) == int(anchor.region_id):
                    continue
                pair_key = tuple(sorted((int(anchor.region_id), int(other.region_id))))
                if pair_key in existing_pairs:
                    connected[int(anchor.region_id)] = True
                    continue
                connection_kind = self._anchor_connection_kind(grid, request, anchor, other)
                if connection_kind is None:
                    continue
                result.append(
                    RegionEdge(
                        edge_id=next_edge_id,
                        source="sampled_connectivity",
                        from_region_id=int(anchor.region_id),
                        to_region_id=int(other.region_id),
                        connection_kind=connection_kind,
                    )
                )
                next_edge_id += 1
                existing_pairs.add(pair_key)
                connected[int(anchor.region_id)] = True
                if int(other.region_id) in connected:
                    connected[int(other.region_id)] = True
        return result, connected

    def _anchor_connection_kind(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        anchor: Any,
        other: Any,
    ) -> str | None:
        if not anchor.overlaps_or_touches(other):
            return None
        for anchor_cell in self._region_passable_cells(grid, anchor):
            for other_cell in self._region_passable_cells(grid, other):
                if anchor_cell == other_cell:
                    return "anchor_region_overlap"
                if self._transition_is_safe(grid, request, anchor_cell, other_cell):
                    return "anchor_region_touching"
        return None

    def _missing_region_reason(self, failure_reason: str | None) -> str:
        reason = (failure_reason or "").lower()
        if "goal" in reason:
            return "goal_region_missing"
        if "start" in reason:
            return "start_region_missing"
        return "region_sequence_missing"

    def _anchoring_payload(
        self,
        request: PlanRequest,
        resolution: _RegionPathResolution,
    ) -> dict[str, Any]:
        return {
            "start_cell": request.start.to_list(),
            "goal_cell": request.goal.to_list(),
            "start_region_id": resolution.start_region_id,
            "goal_region_id": resolution.goal_region_id,
            "start_region_candidates": list(resolution.start_region_candidates),
            "goal_region_candidates": list(resolution.goal_region_candidates),
            "region_sequence_found": bool(resolution.region_path),
            "fallback_reason": resolution.fallback_reason,
            "start_classification": resolution.start_classification,
            "goal_classification": resolution.goal_classification,
            "start_anchor_region_added": resolution.start_anchor_region_added,
            "goal_anchor_region_added": resolution.goal_anchor_region_added,
            "start_anchor_region_connected": resolution.start_anchor_region_connected,
            "goal_anchor_region_connected": resolution.goal_anchor_region_connected,
            "start_anchor_failure_reason": resolution.start_anchor_failure_reason,
            "goal_anchor_failure_reason": resolution.goal_anchor_failure_reason,
        }

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
        anchoring: dict[str, Any],
        sample_attempts: tuple[dict[str, Any], ...] = (),
        candidate_rankings: tuple[dict[str, Any], ...] = (),
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
            start_goal_anchoring=anchoring,
            sample_attempts=sample_attempts,
            candidate_rankings=candidate_rankings,
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
