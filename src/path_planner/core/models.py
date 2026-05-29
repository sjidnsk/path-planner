from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import floor
from typing import Any

import numpy as np

ROUTE_SCHEMA_VERSION = "path-planner-route/v1"


@dataclass(frozen=True, order=True)
class Cell:
    x: int
    y: int

    def to_list(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float

    def to_list(self) -> list[float]:
        return [float(self.x), float(self.y)]


class NeighborPolicy(str, Enum):
    FOUR = "4-neighbor"
    EIGHT = "8-neighbor"


class FailureReason(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_COST = "invalid_cost"
    START_OUT_OF_BOUNDS = "start_out_of_bounds"
    GOAL_OUT_OF_BOUNDS = "goal_out_of_bounds"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    UNREACHABLE = "unreachable"
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if len(self.origin) != 2:
            raise ValueError("origin must contain x and y")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def cell_to_world(self, cell: Cell) -> WorldPoint:
        return WorldPoint(
            self.origin[0] + cell.x * self.resolution,
            self.origin[1] + cell.y * self.resolution,
        )

    def world_to_cell(self, point: WorldPoint) -> Cell:
        return Cell(
            int(floor((point.x - self.origin[0]) / self.resolution)),
            int(floor((point.y - self.origin[1]) / self.resolution)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": [self.origin[0], self.origin[1]],
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class CostGrid:
    spec: GridSpec
    cost: np.ndarray
    passable_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cost = np.asarray(self.cost, dtype=float)
        mask = np.asarray(self.passable_mask, dtype=bool)
        if cost.shape != self.spec.shape:
            raise ValueError(f"cost shape {cost.shape} must match grid shape {self.spec.shape}")
        if mask.shape != self.spec.shape:
            raise ValueError(f"passable_mask shape {mask.shape} must match grid shape {self.spec.shape}")
        passable_cost = cost[mask]
        if passable_cost.size and (not np.all(np.isfinite(passable_cost)) or np.any(passable_cost < 0.0)):
            raise ValueError("passable cells must have finite nonnegative cost")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "passable_mask", mask)

    def is_passable(self, cell: Cell) -> bool:
        return self.spec.in_bounds(cell) and bool(self.passable_mask[cell.y, cell.x])

    def cost_at(self, cell: Cell) -> float:
        return float(self.cost[cell.y, cell.x])

    def min_passable_cost(self) -> float:
        values = self.cost[self.passable_mask]
        if values.size == 0:
            return 0.0
        return float(max(np.min(values), 0.0))


@dataclass(frozen=True)
class PlanRequest:
    start: Cell
    goal: Cell
    neighbor_policy: NeighborPolicy = NeighborPolicy.EIGHT
    prevent_corner_cutting: bool = True
    max_iterations: int = 100_000

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class PlanDiagnostics:
    runtime_ms: float = 0.0
    max_frontier_size: int = 0
    path_length_m: float = 0.0
    expanded_cells: tuple[Cell, ...] = ()
    cost_min: float | None = None
    cost_max: float | None = None
    cost_mean: float | None = None
    neighbor_policy: str = NeighborPolicy.EIGHT.value
    prevent_corner_cutting: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_ms": self.runtime_ms,
            "max_frontier_size": self.max_frontier_size,
            "path_length_m": self.path_length_m,
            "expanded_cells": [cell.to_list() for cell in self.expanded_cells],
            "cost_min": self.cost_min,
            "cost_max": self.cost_max,
            "cost_mean": self.cost_mean,
            "neighbor_policy": self.neighbor_policy,
            "prevent_corner_cutting": self.prevent_corner_cutting,
        }


@dataclass(frozen=True)
class PlanResult:
    success: bool
    path_cells: tuple[Cell, ...]
    path_world: tuple[WorldPoint, ...]
    total_cost: float
    expanded_count: int
    failure_reason: FailureReason | None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)

    def __post_init__(self) -> None:
        if self.success and self.failure_reason is not None:
            raise ValueError("successful result cannot have failure_reason")
        if not self.success and self.failure_reason is None:
            raise ValueError("failed result must have failure_reason")

    def to_route_dict(self, spec: GridSpec) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "trajectory_kind": "geometric_path",
            "reachable": self.success,
            "path_cost": self.total_cost if self.success else None,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "grid": spec.to_dict(),
            "geometric_path": {
                "cells": [cell.to_list() for cell in self.path_cells],
                "world": [point.to_list() for point in self.path_world],
            },
            "expanded_count": self.expanded_count,
            "diagnostics": self.diagnostics.to_dict(),
        }
