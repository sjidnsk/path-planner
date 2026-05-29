from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrackingSimulationConfig:
    lookahead_m: float = 0.75
    time_step_s: float = 0.2
    max_sim_time_s: float = 600.0

    def __post_init__(self) -> None:
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead_m must be positive")
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if self.max_sim_time_s <= 0.0:
            raise ValueError("max_sim_time_s must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookahead_m": self.lookahead_m,
            "time_step_s": self.time_step_s,
            "max_sim_time_s": self.max_sim_time_s,
        }


@dataclass(frozen=True)
class TrackingState:
    time_s: float
    x: float
    y: float
    heading_rad: float
    target_waypoint_index: int
    speed_mps: float
    cross_track_error_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "x": self.x,
            "y": self.y,
            "heading_rad": self.heading_rad,
            "target_waypoint_index": self.target_waypoint_index,
            "speed_mps": self.speed_mps,
            "cross_track_error_m": self.cross_track_error_m,
        }


@dataclass(frozen=True)
class TrackingSimulationSafetyReport:
    is_safe: bool
    min_clearance_m: float | None
    violation_indices: tuple[int, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "min_clearance_m": self.min_clearance_m,
            "violation_indices": list(self.violation_indices),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class TrackingSimulationMetrics:
    path_length_m: float
    simulated_length_m: float
    max_cross_track_error_m: float
    min_clearance_m: float | None
    safety_violation_count: int
    curvature_violation_count: int
    mean_speed_mps: float
    high_cost_exposure: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_length_m": self.path_length_m,
            "simulated_length_m": self.simulated_length_m,
            "max_cross_track_error_m": self.max_cross_track_error_m,
            "min_clearance_m": self.min_clearance_m,
            "safety_violation_count": self.safety_violation_count,
            "curvature_violation_count": self.curvature_violation_count,
            "mean_speed_mps": self.mean_speed_mps,
            "high_cost_exposure": self.high_cost_exposure,
        }


@dataclass(frozen=True)
class TrackingSimulationResult:
    config: TrackingSimulationConfig
    states: tuple[TrackingState, ...]
    metrics: TrackingSimulationMetrics
    safety_report: TrackingSimulationSafetyReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "simulated_path": [state.to_dict() for state in self.states],
            "metrics": self.metrics.to_dict(),
            "safety_report": self.safety_report.to_dict(),
        }
