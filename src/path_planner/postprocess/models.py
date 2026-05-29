from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from path_planner.core import Cell, WorldPoint


def _cells_to_lists(cells: tuple[Cell, ...]) -> list[list[int]]:
    return [cell.to_list() for cell in cells]


def _world_to_lists(points: tuple[WorldPoint, ...]) -> list[list[float]]:
    return [point.to_list() for point in points]


@dataclass(frozen=True)
class CorridorSection:
    center: Cell
    cells: tuple[Cell, ...]

    def to_dict(self) -> dict[str, Any]:
        xs = [cell.x for cell in self.cells]
        ys = [cell.y for cell in self.cells]
        bounds = {
            "min": [min(xs), min(ys)],
            "max": [max(xs), max(ys)],
        } if self.cells else {"min": self.center.to_list(), "max": self.center.to_list()}
        return {
            "center": self.center.to_list(),
            "cells": _cells_to_lists(self.cells),
            "bounds": bounds,
        }


@dataclass(frozen=True)
class CorridorResult:
    status: str
    radius_cells: int
    sections: tuple[CorridorSection, ...]
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "radius_cells": self.radius_cells,
            "sections": [section.to_dict() for section in self.sections],
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SmoothedPathResult:
    status: str
    cells: tuple[Cell, ...]
    world: tuple[WorldPoint, ...]
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "cells": _cells_to_lists(self.cells),
            "world": _world_to_lists(self.world),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class CurvatureReport:
    is_feasible: bool
    max_curvature: float
    min_turning_radius: float | None
    violation_indices: tuple[int, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "max_curvature": self.max_curvature,
            "min_turning_radius": self.min_turning_radius,
            "violation_indices": list(self.violation_indices),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class FallbackStatus:
    used_raw_path: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_raw_path": self.used_raw_path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PostprocessResult:
    raw_path_cells: tuple[Cell, ...]
    raw_path_world: tuple[WorldPoint, ...]
    corridor: CorridorResult
    smoothed_path: SmoothedPathResult
    curvature_report: CurvatureReport
    fallback_status: FallbackStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_path": {
                "cells": _cells_to_lists(self.raw_path_cells),
                "world": _world_to_lists(self.raw_path_world),
            },
            "corridor": self.corridor.to_dict(),
            "smoothed_path": self.smoothed_path.to_dict(),
            "curvature_report": self.curvature_report.to_dict(),
            "fallback_status": self.fallback_status.to_dict(),
        }
