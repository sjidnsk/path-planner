from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from path_planner.core import Cell, WorldPoint
from path_planner.platform import PlannerPlatformProfile


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
    original_blocked_count: int = 0
    inflated_blocked_count: int = 0
    footprint_radius_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "radius_cells": self.radius_cells,
            "sections": [section.to_dict() for section in self.sections],
            "failure_reason": self.failure_reason,
            "original_blocked_count": self.original_blocked_count,
            "inflated_blocked_count": self.inflated_blocked_count,
            "footprint_radius_m": self.footprint_radius_m,
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
class CurvatureSample:
    point_index: int
    turn_angle_deg: float
    curvature: float
    turning_radius: float | None
    violates: bool
    violates_curvature: bool
    violates_min_turning_radius: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "turn_angle_deg": self.turn_angle_deg,
            "curvature": self.curvature,
            "turning_radius": self.turning_radius,
            "violates": self.violates,
            "violates_curvature": self.violates_curvature,
            "violates_min_turning_radius": self.violates_min_turning_radius,
        }


@dataclass(frozen=True)
class CurvatureReport:
    is_feasible: bool
    max_curvature: float
    min_turning_radius: float | None
    violation_indices: tuple[int, ...]
    summary: str
    constraint_min_turning_radius: float | None = None
    samples: tuple[CurvatureSample, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "max_curvature": self.max_curvature,
            "min_turning_radius": self.min_turning_radius,
            "constraint_min_turning_radius": self.constraint_min_turning_radius,
            "violation_indices": list(self.violation_indices),
            "summary": self.summary,
            "samples": [sample.to_dict() for sample in self.samples],
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
    platform_profile: PlannerPlatformProfile | None = None
    constraint_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_profile": self.platform_profile.to_dict() if self.platform_profile is not None else None,
            "constraint_warnings": list(self.constraint_warnings),
            "corridor_report": {
                "status": self.corridor.status,
                "section_count": len(self.corridor.sections),
                "radius_cells": self.corridor.radius_cells,
                "failure_reason": self.corridor.failure_reason,
                "original_blocked_count": self.corridor.original_blocked_count,
                "inflated_blocked_count": self.corridor.inflated_blocked_count,
                "footprint_radius_m": self.corridor.footprint_radius_m,
            },
            "raw_path": {
                "cells": _cells_to_lists(self.raw_path_cells),
                "world": _world_to_lists(self.raw_path_world),
            },
            "corridor": self.corridor.to_dict(),
            "smoothed_path": self.smoothed_path.to_dict(),
            "curvature_report": self.curvature_report.to_dict(),
            "fallback_status": self.fallback_status.to_dict(),
        }
