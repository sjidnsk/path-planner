from __future__ import annotations

import numpy as np

from path_planner.core import CostGrid, WorldPoint
from path_planner.postprocess.models import CorridorResult

from .corridor_mapping import build_resampled_points, project_points_to_corridor
from .geometry import max_spacing
from .models import TrajectoryOptimizationConfig
from .objective import high_cost_exposure


HIGH_COST_EXPOSURE_TOLERANCE_M = 1e-3


def apply_high_cost_guard(
    grid: CostGrid,
    points: np.ndarray,
    reference: np.ndarray,
    config: TrajectoryOptimizationConfig,
) -> np.ndarray:
    reference_exposure = high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in reference),
        config.high_cost_threshold,
    )
    current_exposure = high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in points),
        config.high_cost_threshold,
    )
    if current_exposure <= reference_exposure:
        return points

    guarded = points.copy()
    for index in range(len(points) - 1):
        segment = (
            WorldPoint(float(guarded[index, 0]), float(guarded[index, 1])),
            WorldPoint(float(guarded[index + 1, 0]), float(guarded[index + 1, 1])),
        )
        reference_segment = (
            WorldPoint(float(reference[index, 0]), float(reference[index, 1])),
            WorldPoint(float(reference[index + 1, 0]), float(reference[index + 1, 1])),
        )
        if high_cost_exposure(grid, segment, config.high_cost_threshold) <= 0.0:
            continue
        if high_cost_exposure(grid, reference_segment, config.high_cost_threshold) > 0.0:
            continue
        if 0 < index < len(points) - 1:
            guarded[index] = reference[index]
        if 0 < index + 1 < len(points) - 1:
            guarded[index + 1] = reference[index + 1]

    guarded_exposure = high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in guarded),
        config.high_cost_threshold,
    )
    return guarded if guarded_exposure <= reference_exposure else reference.copy()


def apply_resampled_high_cost_guard(
    grid: CostGrid,
    candidate_points: tuple[WorldPoint, ...],
    reference_points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    config: TrajectoryOptimizationConfig,
) -> tuple[WorldPoint, ...]:
    reference_resampled = build_resampled_points(grid, reference_points, corridor, config)
    reference_exposure = high_cost_exposure(grid, reference_resampled, config.high_cost_threshold)
    candidate_exposure = high_cost_exposure(grid, candidate_points, config.high_cost_threshold)
    if candidate_exposure <= reference_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
        return candidate_points

    if len(candidate_points) == len(reference_resampled):
        for candidate_weight in (0.75, 0.5, 0.25, 0.1, 0.0):
            blended = _blend_world_points(candidate_points, reference_resampled, candidate_weight)
            blended = project_points_to_corridor(grid, blended, corridor, max_step_m=config.resample_spacing_m)
            if config.resample_spacing_m is not None and max_spacing(blended) > config.resample_spacing_m + 1e-9:
                continue
            if high_cost_exposure(grid, blended, config.high_cost_threshold) <= reference_exposure + HIGH_COST_EXPOSURE_TOLERANCE_M:
                return blended

    return reference_resampled


def _blend_world_points(
    candidate_points: tuple[WorldPoint, ...],
    reference_points: tuple[WorldPoint, ...],
    candidate_weight: float,
) -> tuple[WorldPoint, ...]:
    reference_weight = 1.0 - candidate_weight
    return tuple(
        WorldPoint(
            candidate_weight * candidate.x + reference_weight * reference.x,
            candidate_weight * candidate.y + reference_weight * reference.y,
        )
        for candidate, reference in zip(candidate_points, reference_points)
    )
