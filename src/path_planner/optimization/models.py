from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from path_planner.core import WorldPoint
from path_planner.trajectory import TrackablePath


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    weight_smoothness: float = 2.0
    weight_cost: float = 2.0
    weight_reference: float = 1.0
    weight_length: float = 1.0
    weight_curvature_proxy: float = 1.0
    weight_tracking: float = 1.0
    weight_spacing: float = 0.5
    weight_speed_smoothness: float = 0.2
    resample_spacing_m: float | None = None
    max_iterations: int = 40
    convergence_tolerance: float = 1e-5
    high_cost_threshold: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "weight_smoothness",
            "weight_cost",
            "weight_reference",
            "weight_length",
            "weight_curvature_proxy",
            "weight_tracking",
            "weight_spacing",
            "weight_speed_smoothness",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.resample_spacing_m is not None and self.resample_spacing_m <= 0.0:
            raise ValueError("resample_spacing_m must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        if self.high_cost_threshold < 0.0:
            raise ValueError("high_cost_threshold must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_smoothness": self.weight_smoothness,
            "weight_cost": self.weight_cost,
            "weight_reference": self.weight_reference,
            "weight_length": self.weight_length,
            "weight_curvature_proxy": self.weight_curvature_proxy,
            "weight_tracking": self.weight_tracking,
            "weight_spacing": self.weight_spacing,
            "weight_speed_smoothness": self.weight_speed_smoothness,
            "resample_spacing_m": self.resample_spacing_m,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "high_cost_threshold": self.high_cost_threshold,
        }


@dataclass(frozen=True)
class CorridorBox:
    point_index: int
    section_index: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, point: WorldPoint) -> bool:
        return self.min_x <= point.x <= self.max_x and self.min_y <= point.y <= self.max_y

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "section_index": self.section_index,
            "min": [self.min_x, self.min_y],
            "max": [self.max_x, self.max_y],
        }


@dataclass(frozen=True)
class TrajectoryOptimizationFallbackStatus:
    used_reference_path: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_reference_path": self.used_reference_path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrajectoryOptimizationMetrics:
    reference_path_length_m: float
    optimized_path_length_m: float
    length_delta_m: float
    reference_high_cost_exposure: float
    optimized_high_cost_exposure: float
    high_cost_exposure_delta: float
    max_curvature: float
    min_turning_radius_m: float | None
    curvature_violation_count: int
    waypoint_spacing_mean_m: float
    waypoint_spacing_max_m: float
    heading_change_max_deg: float
    tracking_error_proxy: float
    speed_smoothness_cost: float
    objective_initial: float
    objective_final: float
    solver_iterations: int
    is_within_corridor: bool
    baseline_vs_optimized: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_path_length_m": self.reference_path_length_m,
            "optimized_path_length_m": self.optimized_path_length_m,
            "length_delta_m": self.length_delta_m,
            "reference_high_cost_exposure": self.reference_high_cost_exposure,
            "optimized_high_cost_exposure": self.optimized_high_cost_exposure,
            "high_cost_exposure_delta": self.high_cost_exposure_delta,
            "max_curvature": self.max_curvature,
            "min_turning_radius_m": self.min_turning_radius_m,
            "curvature_violation_count": self.curvature_violation_count,
            "waypoint_spacing_mean_m": self.waypoint_spacing_mean_m,
            "waypoint_spacing_max_m": self.waypoint_spacing_max_m,
            "heading_change_max_deg": self.heading_change_max_deg,
            "tracking_error_proxy": self.tracking_error_proxy,
            "speed_smoothness_cost": self.speed_smoothness_cost,
            "objective_initial": self.objective_initial,
            "objective_final": self.objective_final,
            "solver_iterations": self.solver_iterations,
            "is_within_corridor": self.is_within_corridor,
            "baseline_vs_optimized": self.baseline_vs_optimized,
        }


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    config: TrajectoryOptimizationConfig
    solver_status: str
    fallback_status: TrajectoryOptimizationFallbackStatus
    optimized_path: tuple[WorldPoint, ...]
    resampled_optimized_path: tuple[WorldPoint, ...]
    optimized_trackable_path: TrackablePath
    resampled_trackable_path: TrackablePath
    corridor_boxes: tuple[CorridorBox, ...]
    resampled_corridor_boxes: tuple[CorridorBox, ...]
    metrics: TrajectoryOptimizationMetrics
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "solver_status": self.solver_status,
            "fallback_status": self.fallback_status.to_dict(),
            "optimized_path": [point.to_list() for point in self.optimized_path],
            "resampled_optimized_path": [point.to_list() for point in self.resampled_optimized_path],
            "optimized_trackable_path": self.optimized_trackable_path.to_dict(),
            "resampled_trackable_path": self.resampled_trackable_path.to_dict(),
            "corridor_boxes": [box.to_dict() for box in self.corridor_boxes],
            "resampled_corridor_boxes": [box.to_dict() for box in self.resampled_corridor_boxes],
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
        }
