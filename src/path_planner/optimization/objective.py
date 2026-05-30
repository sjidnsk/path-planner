from __future__ import annotations

import math

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint

from .geometry import distance, wrap_angle
from .models import TrajectoryOptimizationConfig


def objective(
    grid: CostGrid,
    points: np.ndarray,
    reference: np.ndarray,
    config: TrajectoryOptimizationConfig,
) -> float:
    if len(points) == 0:
        return 0.0
    length = _array_path_length(points)
    smoothness = 0.0
    if len(points) > 2:
        second = points[:-2] - 2.0 * points[1:-1] + points[2:]
        smoothness = float(np.sum(second * second))
    reference_error = float(np.sum((points - reference) * (points - reference)))
    tracking_proxy = _tracking_error_proxy_for_array(points)
    spacing_cost = _spacing_uniformity_cost(points, config.resample_spacing_m)
    speed_smoothness = _speed_smoothness_proxy_for_array(points)
    high_cost = high_cost_exposure(
        grid,
        tuple(WorldPoint(float(x), float(y)) for x, y in points),
        config.high_cost_threshold,
    )
    return float(
        config.weight_length * length
        + config.weight_smoothness * smoothness
        + config.weight_reference * reference_error
        + config.weight_curvature_proxy * smoothness
        + config.weight_tracking * tracking_proxy
        + config.weight_spacing * spacing_cost
        + config.weight_speed_smoothness * speed_smoothness
        + config.weight_cost * high_cost
    )


def cost_gradient(grid: CostGrid, point: np.ndarray) -> np.ndarray:
    cell = grid.spec.world_to_cell(WorldPoint(float(point[0]), float(point[1])))
    if not grid.spec.in_bounds(cell):
        return np.zeros(2, dtype=float)
    left = Cell(max(cell.x - 1, 0), cell.y)
    right = Cell(min(cell.x + 1, grid.spec.width - 1), cell.y)
    up = Cell(cell.x, max(cell.y - 1, 0))
    down = Cell(cell.x, min(cell.y + 1, grid.spec.height - 1))
    scale_x = max((right.x - left.x) * grid.spec.resolution, grid.spec.resolution)
    scale_y = max((down.y - up.y) * grid.spec.resolution, grid.spec.resolution)
    return np.asarray(
        [
            (grid.cost_at(right) - grid.cost_at(left)) / scale_x,
            (grid.cost_at(down) - grid.cost_at(up)) / scale_y,
        ],
        dtype=float,
    )


def high_cost_exposure(grid: CostGrid, points: tuple[WorldPoint, ...], threshold: float) -> float:
    exposure = 0.0
    for previous, current in zip(points[:-1], points[1:]):
        segment_length = distance(previous, current)
        midpoint = WorldPoint((previous.x + current.x) * 0.5, (previous.y + current.y) * 0.5)
        cell = grid.spec.world_to_cell(midpoint)
        if grid.spec.in_bounds(cell) and grid.cost_at(cell) >= threshold:
            exposure += segment_length
    return float(exposure)


def _tracking_error_proxy_for_array(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    headings = [
        math.atan2(float(points[index + 1, 1] - points[index, 1]), float(points[index + 1, 0] - points[index, 0]))
        for index in range(len(points) - 1)
    ]
    heading_changes = [
        abs(wrap_angle(current - previous))
        for previous, current in zip(headings[:-1], headings[1:])
    ]
    return float(sum(change * change for change in heading_changes))


def _spacing_uniformity_cost(points: np.ndarray, target_spacing: float | None) -> float:
    if len(points) < 3:
        return 0.0
    spacings = [
        math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
        for previous, current in zip(points[:-1], points[1:])
    ]
    target = target_spacing if target_spacing is not None else sum(spacings) / len(spacings)
    return float(sum((spacing - target) * (spacing - target) for spacing in spacings))


def _speed_smoothness_proxy_for_array(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    spacings = [
        math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
        for previous, current in zip(points[:-1], points[1:])
    ]
    deltas = [current - previous for previous, current in zip(spacings[:-1], spacings[1:])]
    return float(sum(delta * delta for delta in deltas))


def _array_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = points[1:] - points[:-1]
    return float(np.sum(np.sqrt(np.sum(deltas * deltas, axis=1))))
