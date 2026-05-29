from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from path_planner.core import Cell, WorldPoint


@dataclass(frozen=True)
class TrackableWaypoint:
    index: int
    cell: Cell
    world: WorldPoint
    heading_rad: float
    segment_length_m: float
    turn_angle_deg: float
    curvature: float
    turning_radius_m: float | None
    recommended_speed_mps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cell": self.cell.to_list(),
            "world": self.world.to_list(),
            "heading_rad": self.heading_rad,
            "segment_length_m": self.segment_length_m,
            "turn_angle_deg": self.turn_angle_deg,
            "curvature": self.curvature,
            "turning_radius_m": self.turning_radius_m,
            "recommended_speed_mps": self.recommended_speed_mps,
        }


@dataclass(frozen=True)
class TrackablePath:
    source_path: str
    waypoints: tuple[TrackableWaypoint, ...]
    length_m: float
    max_curvature: float
    min_turning_radius_m: float | None

    @property
    def speed_profile(self) -> tuple[float, ...]:
        return tuple(waypoint.recommended_speed_mps for waypoint in self.waypoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
            "speed_profile": list(self.speed_profile),
            "length_m": self.length_m,
            "max_curvature": self.max_curvature,
            "min_turning_radius_m": self.min_turning_radius_m,
        }


@dataclass(frozen=True)
class TrackingSafetyReport:
    is_safe: bool
    tracking_error_bound_m: float
    checked_radius_m: float
    min_clearance_m: float | None
    violation_indices: tuple[int, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "tracking_error_bound_m": self.tracking_error_bound_m,
            "checked_radius_m": self.checked_radius_m,
            "min_clearance_m": self.min_clearance_m,
            "violation_indices": list(self.violation_indices),
            "summary": self.summary,
        }
