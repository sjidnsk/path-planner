from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from path_planner.core import (
    Cell,
    CostGrid,
    FailureReason,
    NeighborPolicy,
    PlanDiagnostics,
    PlanRequest,
    PlanResult,
)


@dataclass(order=True)
class _QueueItem:
    priority: float
    order: int
    cell: Cell = field(compare=False)


class AStarPlanner:
    def plan(self, grid: CostGrid, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        early_failure = self._validate_request(grid, request, started)
        if early_failure is not None:
            return early_failure

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
                return self._failure(grid, request, FailureReason.MAX_ITERATIONS, expanded, max_frontier_size, started)

            current = heapq.heappop(frontier).cell
            expanded.append(current)

            if current == request.goal:
                path = self._reconstruct_path(came_from, current)
                total_cost = cost_so_far[current]
                return self._success(grid, request, path, total_cost, expanded, max_frontier_size, started)

            for neighbor, step_distance in self._neighbors(grid, request, current):
                new_cost = cost_so_far[current] + step_distance * grid.cost_at(neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    counter += 1
                    priority = new_cost + self._heuristic(neighbor, request.goal, min_cost)
                    heapq.heappush(frontier, _QueueItem(priority, counter, neighbor))
                    came_from[neighbor] = current
            max_frontier_size = max(max_frontier_size, len(frontier))

        return self._failure(grid, request, FailureReason.UNREACHABLE, expanded, max_frontier_size, started)

    def _validate_request(self, grid: CostGrid, request: PlanRequest, started: float) -> PlanResult | None:
        if not grid.spec.in_bounds(request.start):
            return self._failure(grid, request, FailureReason.START_OUT_OF_BOUNDS, (), 0, started)
        if not grid.spec.in_bounds(request.goal):
            return self._failure(grid, request, FailureReason.GOAL_OUT_OF_BOUNDS, (), 0, started)
        if not grid.is_passable(request.start):
            return self._failure(grid, request, FailureReason.START_BLOCKED, (), 0, started)
        if not grid.is_passable(request.goal):
            return self._failure(grid, request, FailureReason.GOAL_BLOCKED, (), 0, started)
        return None

    def _neighbors(self, grid: CostGrid, request: PlanRequest, cell: Cell) -> list[tuple[Cell, float]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if request.neighbor_policy is NeighborPolicy.EIGHT:
            directions.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])

        result: list[tuple[Cell, float]] = []
        for dx, dy in directions:
            neighbor = Cell(cell.x + dx, cell.y + dy)
            if not grid.is_passable(neighbor):
                continue
            is_diagonal = dx != 0 and dy != 0
            if is_diagonal and request.prevent_corner_cutting:
                side_a = Cell(cell.x + dx, cell.y)
                side_b = Cell(cell.x, cell.y + dy)
                if not grid.is_passable(side_a) and not grid.is_passable(side_b):
                    continue
            result.append((neighbor, math.sqrt(2.0) if is_diagonal else 1.0))
        return result

    def _heuristic(self, cell: Cell, goal: Cell, min_cost: float) -> float:
        dx = abs(goal.x - cell.x)
        dy = abs(goal.y - cell.y)
        return min_cost * ((dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy))

    def _reconstruct_path(self, came_from: dict[Cell, Cell | None], current: Cell) -> tuple[Cell, ...]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return tuple(path)

    def _success(
        self,
        grid: CostGrid,
        request: PlanRequest,
        path: tuple[Cell, ...],
        total_cost: float,
        expanded: list[Cell],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = self._diagnostics(grid, request, path, expanded, max_frontier_size, started)
        return PlanResult(
            success=True,
            path_cells=path,
            path_world=tuple(grid.spec.cell_to_world(cell) for cell in path),
            total_cost=float(total_cost),
            expanded_count=len(expanded),
            failure_reason=None,
            diagnostics=diagnostics,
        )

    def _failure(
        self,
        grid: CostGrid,
        request: PlanRequest,
        reason: FailureReason,
        expanded: list[Cell] | tuple[Cell, ...],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = self._diagnostics(grid, request, (), tuple(expanded), max_frontier_size, started)
        return PlanResult(
            success=False,
            path_cells=(),
            path_world=(),
            total_cost=math.inf,
            expanded_count=len(expanded),
            failure_reason=reason,
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        grid: CostGrid,
        request: PlanRequest,
        path: tuple[Cell, ...],
        expanded: list[Cell] | tuple[Cell, ...],
        max_frontier_size: int,
        started: float,
    ) -> PlanDiagnostics:
        passable_cost = grid.cost[grid.passable_mask]
        path_length = self._path_length(path, grid.spec.resolution)
        return PlanDiagnostics(
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_frontier_size=max_frontier_size,
            path_length_m=path_length,
            expanded_cells=tuple(expanded),
            cost_min=float(np.min(passable_cost)) if passable_cost.size else None,
            cost_max=float(np.max(passable_cost)) if passable_cost.size else None,
            cost_mean=float(np.mean(passable_cost)) if passable_cost.size else None,
            neighbor_policy=request.neighbor_policy.value,
            prevent_corner_cutting=request.prevent_corner_cutting,
        )

    def _path_length(self, path: tuple[Cell, ...], resolution: float) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y) * resolution
        return total
