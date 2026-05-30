from __future__ import annotations

from path_planner.core import CostGrid, WorldPoint
from path_planner.trajectory import TrackablePath, TrackableWaypoint

from .geometry import curvature, distance, heading_at, turn_angle_deg


def build_trackable_path_from_world(
    grid: CostGrid,
    reference: TrackablePath,
    points: tuple[WorldPoint, ...],
) -> TrackablePath:
    waypoints: list[TrackableWaypoint] = []
    for index, point in enumerate(points):
        point_curvature = curvature(points, index)
        turning_radius = 1.0 / point_curvature if point_curvature > 0.0 else None
        reference_speed = _recommended_speed_for_point(grid, reference, point, point_curvature, index)
        waypoints.append(
            TrackableWaypoint(
                index=index,
                cell=grid.spec.world_to_cell(point),
                world=point,
                heading_rad=heading_at(points, index),
                segment_length_m=0.0 if index == 0 else distance(points[index - 1], point),
                turn_angle_deg=turn_angle_deg(points, index),
                curvature=point_curvature,
                turning_radius_m=turning_radius,
                recommended_speed_mps=reference_speed,
            )
        )
    radii = [waypoint.turning_radius_m for waypoint in waypoints if waypoint.turning_radius_m is not None]
    return TrackablePath(
        source_path="optimized_path",
        waypoints=tuple(waypoints),
        length_m=sum(waypoint.segment_length_m for waypoint in waypoints),
        max_curvature=max((waypoint.curvature for waypoint in waypoints), default=0.0),
        min_turning_radius_m=min(radii) if radii else None,
    )


def _recommended_speed_for_point(
    grid: CostGrid,
    reference: TrackablePath,
    point: WorldPoint,
    point_curvature: float,
    index: int,
) -> float:
    speeds = reference.speed_profile
    if not speeds:
        return 0.05
    max_speed = max(speeds)
    min_speed = min(speeds)
    cell = grid.spec.world_to_cell(point)
    cost = grid.cost_at(cell) if grid.spec.in_bounds(cell) else max(1.0, max_speed / max(min_speed, 1e-9))
    cost_factor = 1.0 / max(1.0, cost)
    curvature_factor = 1.0 / (1.0 + 4.0 * point_curvature)
    reference_speed = speeds[min(index, len(speeds) - 1)]
    speed = min(max_speed, reference_speed, max_speed * min(cost_factor, curvature_factor))
    return float(max(min_speed, speed))
