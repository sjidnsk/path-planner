from __future__ import annotations

import numpy as np

from path_planner.core import Cell, CostGrid, WorldPoint
from path_planner.postprocess.models import CorridorResult, CorridorSection

from .geometry import densify_world_points, distance, max_spacing
from .models import CorridorBox, TrajectoryOptimizationConfig
from .resampling import resample_world_points


def build_corridor_boxes(
    grid: CostGrid,
    trackable_path,
    corridor: CorridorResult,
) -> tuple[CorridorBox, ...]:
    boxes: list[CorridorBox] = []
    for point_index, waypoint in enumerate(trackable_path.waypoints):
        section_index, section = nearest_section_for_point(grid, waypoint.world, corridor.sections)
        boxes.append(section_to_box(grid, point_index, section_index, section))
    return tuple(boxes)


def build_corridor_boxes_for_points(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
) -> tuple[CorridorBox, ...]:
    boxes: list[CorridorBox] = []
    if corridor.status != "ok" or not corridor.sections:
        return ()
    for point_index, point in enumerate(points):
        section_index, section = nearest_section_for_point(grid, point, corridor.sections)
        boxes.append(section_to_box(grid, point_index, section_index, section))
    return tuple(boxes)


def nearest_section_for_point(
    grid: CostGrid,
    point: WorldPoint,
    sections: tuple[CorridorSection, ...],
) -> tuple[int, CorridorSection]:
    cell = grid.spec.world_to_cell(point)
    candidates = [index for index, section in enumerate(sections) if section_contains_point(grid, section, point)]
    if candidates:
        best_index = min(candidates, key=lambda index: (section_distance_sq(sections[index], cell), -index))
        return best_index, sections[best_index]
    return nearest_section(cell, sections)


def nearest_section(cell: Cell, sections: tuple[CorridorSection, ...]) -> tuple[int, CorridorSection]:
    candidates = [index for index, section in enumerate(sections) if cell in section.cells]
    search_indices = candidates or list(range(len(sections)))
    best_index = min(search_indices, key=lambda index: (section_distance_sq(sections[index], cell), -index))
    return best_index, sections[best_index]


def section_distance_sq(section: CorridorSection, cell: Cell) -> int:
    return (section.center.x - cell.x) ** 2 + (section.center.y - cell.y) ** 2


def section_contains_point(grid: CostGrid, section: CorridorSection, point: WorldPoint) -> bool:
    cells = section.cells or (section.center,)
    min_x = min(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    max_x = max(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    min_y = min(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    max_y = max(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    return min_x <= point.x <= max_x and min_y <= point.y <= max_y


def section_to_box(grid: CostGrid, point_index: int, section_index: int, section: CorridorSection) -> CorridorBox:
    cells = section.cells or (section.center,)
    min_x = min(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    max_x = max(cell.x for cell in cells) * grid.spec.resolution + grid.spec.origin[0]
    min_y = min(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    max_y = max(cell.y for cell in cells) * grid.spec.resolution + grid.spec.origin[1]
    return CorridorBox(
        point_index=point_index,
        section_index=section_index,
        min_x=float(min_x),
        max_x=float(max_x),
        min_y=float(min_y),
        max_y=float(max_y),
    )


def project_to_box(point: np.ndarray, box: CorridorBox) -> np.ndarray:
    return np.asarray(
        [
            min(max(float(point[0]), box.min_x), box.max_x),
            min(max(float(point[1]), box.min_y), box.max_y),
        ],
        dtype=float,
    )


def build_resampled_points(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    config: TrajectoryOptimizationConfig,
) -> tuple[WorldPoint, ...]:
    if config.resample_spacing_m is None or len(points) <= 1:
        return points
    spacing = config.resample_spacing_m
    projected = project_points_to_corridor(
        grid,
        resample_world_points(points, spacing_m=spacing),
        corridor,
        max_step_m=spacing,
    )
    best = projected
    internal_spacing = spacing
    for _ in range(8):
        if max_spacing(projected) <= spacing + 1e-9:
            return projected
        if max_spacing(projected) < max_spacing(best):
            best = projected
        internal_spacing *= 0.5
        densified = densify_world_points(projected, internal_spacing)
        projected = project_points_to_corridor(grid, densified, corridor, max_step_m=spacing)
    return projected if max_spacing(projected) <= max_spacing(best) else best


def project_points_to_corridor(
    grid: CostGrid,
    points: tuple[WorldPoint, ...],
    corridor: CorridorResult,
    *,
    max_step_m: float | None = None,
) -> tuple[WorldPoint, ...]:
    boxes = build_corridor_boxes_for_points(grid, points, corridor)
    if len(boxes) != len(points):
        return points
    projected: list[WorldPoint] = []
    for index, (point, box) in enumerate(zip(points, boxes)):
        if index == 0 or index == len(points) - 1:
            projected.append(point)
            continue
        clipped = project_to_box(np.asarray([point.x, point.y], dtype=float), box)
        candidate = WorldPoint(float(clipped[0]), float(clipped[1]))
        previous = projected[-1]
        if max_step_m is not None and distance(previous, candidate) > max_step_m:
            segment_length = distance(previous, candidate)
            ratio = max_step_m / max(segment_length, 1e-12)
            stepped = np.asarray(
                [
                    previous.x + ratio * (candidate.x - previous.x),
                    previous.y + ratio * (candidate.y - previous.y),
                ],
                dtype=float,
            )
            connected = project_to_box(stepped, box)
            connected_candidate = WorldPoint(float(connected[0]), float(connected[1]))
            if distance(previous, connected_candidate) < distance(previous, candidate):
                candidate = connected_candidate
        projected.append(candidate)
    return tuple(projected)


def lowest_cost_point_in_box(grid: CostGrid, box: CorridorBox, reference_point: np.ndarray) -> np.ndarray:
    best_cell: Cell | None = None
    best_key: tuple[float, float] | None = None
    for y in range(grid.spec.height):
        for x in range(grid.spec.width):
            cell = Cell(x, y)
            point = grid.spec.cell_to_world(cell)
            if not box.contains(point) or not grid.is_passable(cell):
                continue
            distance_sq = (point.x - float(reference_point[0])) ** 2 + (point.y - float(reference_point[1])) ** 2
            key = (grid.cost_at(cell), distance_sq)
            if best_key is None or key < best_key:
                best_cell = cell
                best_key = key
    if best_cell is None:
        return reference_point
    point = grid.spec.cell_to_world(best_cell)
    return np.asarray([point.x, point.y], dtype=float)
