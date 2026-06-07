from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

from path_planner.core import Cell, CostGrid, FailureReason, PlanRequest, PlanResult
from path_planner.search.astar import AStarPlanner
from path_planner.search.planning_grid import PlanningGrid

ASTAR_BACKEND = "astar"
CHANNEL_AWARE_ASTAR_BACKEND = "channel_aware_astar"


@dataclass(frozen=True)
class ChannelAwareAStarConfig:
    neighborhood_radius_cells: int = 1
    center_cell_weight: float = 1.0
    neighborhood_mean_weight: float = 0.25
    neighborhood_max_weight: float = 0.1
    high_cost_exposure_weight: float = 0.5
    blocked_nearby_weight: float = 0.5
    clearance_weight: float = 0.25
    smoothness_weight: float = 0.05
    high_cost_threshold: float = 4.0

    def __post_init__(self) -> None:
        if self.neighborhood_radius_cells < 1:
            raise ValueError("neighborhood_radius_cells must be at least 1")
        for name in (
            "center_cell_weight",
            "neighborhood_mean_weight",
            "neighborhood_max_weight",
            "high_cost_exposure_weight",
            "blocked_nearby_weight",
            "clearance_weight",
            "smoothness_weight",
            "high_cost_threshold",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "neighborhood_radius_cells": self.neighborhood_radius_cells,
            "weights": {
                "center_cell_cost": self.center_cell_weight,
                "neighborhood_mean_cost": self.neighborhood_mean_weight,
                "neighborhood_max_cost": self.neighborhood_max_weight,
                "high_cost_exposure_proxy": self.high_cost_exposure_weight,
                "blocked_nearby_penalty": self.blocked_nearby_weight,
                "clearance_penalty": self.clearance_weight,
                "smoothness_or_direction_proxy": self.smoothness_weight,
            },
            "high_cost_threshold": self.high_cost_threshold,
        }


@dataclass(frozen=True)
class ChannelAwarePlanReport:
    requested_backend: str
    selected_backend: str
    status: str
    fallback_reason: str | None
    config: ChannelAwareAStarConfig
    baseline: dict[str, Any]
    channel_candidate: dict[str, Any]
    comparison: dict[str, Any]
    execution_alignment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "config": self.config.to_dict(),
            "baseline": self.baseline,
            "channel_candidate": self.channel_candidate,
            "comparison": self.comparison,
            "execution_alignment": dict(self.execution_alignment),
        }


@dataclass(frozen=True)
class ChannelAwarePlanOutcome:
    result: PlanResult
    report: ChannelAwarePlanReport


@dataclass(order=True)
class _QueueItem:
    priority: float
    order: int
    cell: Cell = field(compare=False)


class ChannelAwareAStarPlanner:
    def __init__(self, config: ChannelAwareAStarConfig | None = None) -> None:
        self.config = config or ChannelAwareAStarConfig()

    def plan(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        *,
        baseline_result: PlanResult,
    ) -> ChannelAwarePlanOutcome:
        candidate = self._search(grid, request)
        report = self._build_report(grid, baseline_result, candidate)
        if report.status == "selected":
            return ChannelAwarePlanOutcome(result=candidate, report=report)
        return ChannelAwarePlanOutcome(result=baseline_result, report=report)

    def _search(self, grid: CostGrid | PlanningGrid, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        early_failure = AStarPlanner()._validate_request(grid, request, started)
        if early_failure is not None:
            return self._failure_from_result(early_failure)

        frontier: list[_QueueItem] = []
        counter = 0
        heapq.heappush(frontier, _QueueItem(0.0, counter, request.start))
        came_from: dict[Cell, Cell | None] = {request.start: None}
        cost_so_far: dict[Cell, float] = {request.start: 0.0}
        expanded: list[Cell] = []
        max_frontier_size = 1
        min_cost = grid.min_passable_cost()

        while frontier:
            if len(expanded) >= request.max_iterations:
                return self._failure(
                    grid,
                    request,
                    FailureReason.MAX_ITERATIONS,
                    expanded,
                    max_frontier_size,
                    started,
                )

            current = heapq.heappop(frontier).cell
            expanded.append(current)

            if current == request.goal:
                path = self._reconstruct_path(came_from, current)
                return self._success(grid, request, path, expanded, max_frontier_size, started)

            previous = came_from[current]
            for neighbor, step_distance in AStarPlanner()._neighbors(grid, request, current):
                step_cost = self._channel_step_cost(grid, previous, current, neighbor, step_distance)
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    counter += 1
                    priority = new_cost + AStarPlanner()._heuristic(neighbor, request.goal, min_cost)
                    heapq.heappush(frontier, _QueueItem(priority, counter, neighbor))
                    came_from[neighbor] = current
            max_frontier_size = max(max_frontier_size, len(frontier))

        return self._failure(grid, request, FailureReason.UNREACHABLE, expanded, max_frontier_size, started)

    def _success(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        path: tuple[Cell, ...],
        expanded: list[Cell],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = AStarPlanner()._diagnostics(grid, request, path, expanded, max_frontier_size, started)
        diagnostics = replace(diagnostics, search_mode=CHANNEL_AWARE_ASTAR_BACKEND)
        return PlanResult(
            success=True,
            path_cells=path,
            path_world=tuple(grid.spec.cell_to_world(cell) for cell in path),
            total_cost=self._path_cost(grid, path),
            expanded_count=len(expanded),
            failure_reason=None,
            diagnostics=diagnostics,
        )

    def _failure(
        self,
        grid: CostGrid | PlanningGrid,
        request: PlanRequest,
        reason: FailureReason,
        expanded: list[Cell],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = AStarPlanner()._diagnostics(grid, request, (), expanded, max_frontier_size, started)
        diagnostics = replace(diagnostics, search_mode=CHANNEL_AWARE_ASTAR_BACKEND)
        return PlanResult(
            success=False,
            path_cells=(),
            path_world=(),
            total_cost=math.inf,
            expanded_count=len(expanded),
            failure_reason=reason,
            diagnostics=diagnostics,
        )

    def _failure_from_result(self, result: PlanResult) -> PlanResult:
        return PlanResult(
            success=False,
            path_cells=(),
            path_world=(),
            total_cost=math.inf,
            expanded_count=result.expanded_count,
            failure_reason=result.failure_reason,
            diagnostics=replace(result.diagnostics, search_mode=CHANNEL_AWARE_ASTAR_BACKEND),
        )

    def _build_report(
        self,
        grid: CostGrid | PlanningGrid,
        baseline_result: PlanResult,
        candidate: PlanResult,
    ) -> ChannelAwarePlanReport:
        baseline_summary = self._path_summary(grid, baseline_result)
        candidate_summary = self._path_summary(grid, candidate)
        comparison = {
            "path_changed": bool(candidate.success and baseline_result.path_cells != candidate.path_cells),
            "path_cost_delta": _delta(candidate_summary.get("path_cost"), baseline_summary.get("path_cost")),
            "channel_cost_delta": _delta(candidate_summary.get("channel_cost"), baseline_summary.get("channel_cost")),
            "high_cost_exposure_delta": _delta(
                candidate_summary.get("high_cost_exposure"),
                baseline_summary.get("high_cost_exposure"),
            ),
        }
        status = "fallback"
        selected_backend = ASTAR_BACKEND
        fallback_reason = self._fallback_reason(baseline_result, candidate, comparison)
        if fallback_reason is None:
            status = "selected"
            selected_backend = CHANNEL_AWARE_ASTAR_BACKEND

        return ChannelAwarePlanReport(
            requested_backend=CHANNEL_AWARE_ASTAR_BACKEND,
            selected_backend=selected_backend,
            status=status,
            fallback_reason=fallback_reason,
            config=self.config,
            baseline=baseline_summary,
            channel_candidate=candidate_summary,
            comparison=comparison,
            execution_alignment={
                "schema_version": "channel-aware-route-execution-alignment/v1",
                "postprocess_seed": (
                    "selected_backend_result"
                    if selected_backend == CHANNEL_AWARE_ASTAR_BACKEND
                    else "baseline_result"
                ),
                "selected_seed_postprocess_rebuilt": selected_backend == CHANNEL_AWARE_ASTAR_BACKEND,
                "default_route_replacement_verified": False,
                "verification_scope": "opt_in_audit_only",
            },
        )

    def _fallback_reason(
        self,
        baseline_result: PlanResult,
        candidate: PlanResult,
        comparison: dict[str, Any],
    ) -> str | None:
        if not candidate.success:
            reason = candidate.failure_reason.value if candidate.failure_reason else "unknown"
            return f"channel_search_failed:{reason}"
        if not baseline_result.success:
            return None
        if not comparison["path_changed"]:
            return "channel_candidate_same_as_baseline"
        channel_delta = comparison["channel_cost_delta"]
        exposure_delta = comparison["high_cost_exposure_delta"]
        if channel_delta is not None and channel_delta < -1.0e-9:
            return None
        if (
            exposure_delta is not None
            and exposure_delta < -1.0e-9
            and (channel_delta is None or channel_delta <= 1.0e-9)
        ):
            return None
        if exposure_delta is not None and exposure_delta < -1.0e-9:
            return "channel_candidate_quality_regression"
        return "channel_candidate_not_lower_risk"

    def _path_summary(self, grid: CostGrid | PlanningGrid, result: PlanResult) -> dict[str, Any]:
        if not result.success:
            return {
                "status": "failed",
                "failure_reason": result.failure_reason.value if result.failure_reason else None,
                "path_cost": None,
                "channel_cost": None,
                "high_cost_exposure": None,
                "cost_terms": self._empty_terms(),
            }
        terms = self._path_terms(grid, result.path_cells)
        return {
            "status": "success",
            "failure_reason": None,
            "path_cost": self._path_cost(grid, result.path_cells),
            "channel_cost": self._weighted_terms(terms),
            "high_cost_exposure": terms["high_cost_exposure_proxy"],
            "cost_terms": terms,
        }

    def _path_terms(self, grid: CostGrid | PlanningGrid, path: tuple[Cell, ...]) -> dict[str, float]:
        totals = self._empty_terms()
        if len(path) < 2:
            return totals
        previous: Cell | None = None
        for current, neighbor in zip(path[:-1], path[1:]):
            step_distance = math.hypot(neighbor.x - current.x, neighbor.y - current.y)
            terms = self._cell_terms(grid, neighbor)
            terms["smoothness_or_direction_proxy"] = self._smoothness_penalty(previous, current, neighbor)
            for key, value in terms.items():
                totals[key] += value * step_distance if key != "smoothness_or_direction_proxy" else value
            previous = current
        return {key: float(value) for key, value in totals.items()}

    def _empty_terms(self) -> dict[str, float]:
        return {
            "center_cell_cost": 0.0,
            "neighborhood_mean_cost": 0.0,
            "neighborhood_max_cost": 0.0,
            "high_cost_exposure_proxy": 0.0,
            "blocked_nearby_penalty": 0.0,
            "clearance_penalty": 0.0,
            "smoothness_or_direction_proxy": 0.0,
        }

    def _weighted_terms(self, terms: dict[str, float]) -> float:
        return float(
            self.config.center_cell_weight * terms["center_cell_cost"]
            + self.config.neighborhood_mean_weight * terms["neighborhood_mean_cost"]
            + self.config.neighborhood_max_weight * terms["neighborhood_max_cost"]
            + self.config.high_cost_exposure_weight * terms["high_cost_exposure_proxy"]
            + self.config.blocked_nearby_weight * terms["blocked_nearby_penalty"]
            + self.config.clearance_weight * terms["clearance_penalty"]
            + self.config.smoothness_weight * terms["smoothness_or_direction_proxy"]
        )

    def _channel_step_cost(
        self,
        grid: CostGrid | PlanningGrid,
        previous: Cell | None,
        current: Cell,
        neighbor: Cell,
        step_distance: float,
    ) -> float:
        terms = self._cell_terms(grid, neighbor)
        terms["smoothness_or_direction_proxy"] = self._smoothness_penalty(previous, current, neighbor) / max(
            step_distance,
            1.0,
        )
        return max(self._weighted_terms(terms) * step_distance, 0.0)

    def _cell_terms(self, grid: CostGrid | PlanningGrid, cell: Cell) -> dict[str, float]:
        radius = self.config.neighborhood_radius_cells
        costs: list[float] = []
        blocked_count = 0
        total_count = 0
        min_blocked_distance: float | None = None

        for y in range(cell.y - radius, cell.y + radius + 1):
            for x in range(cell.x - radius, cell.x + radius + 1):
                total_count += 1
                item = Cell(x, y)
                if not grid.spec.in_bounds(item) or not grid.is_passable(item):
                    blocked_count += 1
                    distance = math.hypot(x - cell.x, y - cell.y)
                    if min_blocked_distance is None or distance < min_blocked_distance:
                        min_blocked_distance = distance
                    continue
                costs.append(grid.cost_at(item))

        center_cost = grid.cost_at(cell)
        mean_cost = sum(costs) / len(costs) if costs else center_cost
        max_cost = max(costs) if costs else center_cost
        high_cost_exposure = sum(max(cost - self.config.high_cost_threshold, 0.0) for cost in costs)
        blocked_nearby = blocked_count / total_count if total_count else 0.0
        clearance_penalty = 0.0 if min_blocked_distance is None else 1.0 / max(min_blocked_distance, 1.0)
        return {
            "center_cell_cost": float(center_cost),
            "neighborhood_mean_cost": float(mean_cost),
            "neighborhood_max_cost": float(max_cost),
            "high_cost_exposure_proxy": float(high_cost_exposure),
            "blocked_nearby_penalty": float(blocked_nearby),
            "clearance_penalty": float(clearance_penalty),
            "smoothness_or_direction_proxy": 0.0,
        }

    def _smoothness_penalty(self, previous: Cell | None, current: Cell, neighbor: Cell) -> float:
        if previous is None:
            return 0.0
        ax = current.x - previous.x
        ay = current.y - previous.y
        bx = neighbor.x - current.x
        by = neighbor.y - current.y
        a_norm = math.hypot(ax, ay)
        b_norm = math.hypot(bx, by)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (a_norm * b_norm)))
        return float(math.acos(cosine) / math.pi)

    def _path_cost(self, grid: CostGrid | PlanningGrid, path: tuple[Cell, ...]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for current, neighbor in zip(path[:-1], path[1:]):
            total += math.hypot(neighbor.x - current.x, neighbor.y - current.y) * grid.cost_at(neighbor)
        return float(total)

    def _reconstruct_path(self, came_from: dict[Cell, Cell | None], current: Cell) -> tuple[Cell, ...]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return tuple(path)


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)
