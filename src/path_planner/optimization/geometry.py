from __future__ import annotations

import math

from path_planner.core import WorldPoint


def distance(a: WorldPoint, b: WorldPoint) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def max_spacing(points: tuple[WorldPoint, ...]) -> float:
    return max((distance(previous, current) for previous, current in zip(points[:-1], points[1:])), default=0.0)


def densify_world_points(points: tuple[WorldPoint, ...], spacing_m: float) -> tuple[WorldPoint, ...]:
    if len(points) <= 1:
        return points
    densified: list[WorldPoint] = [points[0]]
    for start, goal in zip(points[:-1], points[1:]):
        segment_length = distance(start, goal)
        steps = max(1, int(math.ceil(segment_length / spacing_m)))
        for step in range(1, steps):
            ratio = step / steps
            densified.append(
                WorldPoint(
                    start.x + ratio * (goal.x - start.x),
                    start.y + ratio * (goal.y - start.y),
                )
            )
        densified.append(goal)
    return tuple(densified)


def heading_at(points: tuple[WorldPoint, ...], index: int) -> float:
    if len(points) < 2:
        return 0.0
    if index < len(points) - 1:
        start = points[index]
        goal = points[index + 1]
    else:
        start = points[index - 1]
        goal = points[index]
    return math.atan2(goal.y - start.y, goal.x - start.x)


def turn_angle_deg(points: tuple[WorldPoint, ...], index: int) -> float:
    if index <= 0 or index >= len(points) - 1:
        return 0.0
    ab_x = points[index].x - points[index - 1].x
    ab_y = points[index].y - points[index - 1].y
    bc_x = points[index + 1].x - points[index].x
    bc_y = points[index + 1].y - points[index].y
    len_ab = math.hypot(ab_x, ab_y)
    len_bc = math.hypot(bc_x, bc_y)
    if len_ab == 0.0 or len_bc == 0.0:
        return 0.0
    cosine = (ab_x * bc_x + ab_y * bc_y) / (len_ab * len_bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def curvature(points: tuple[WorldPoint, ...], index: int) -> float:
    if index <= 0 or index >= len(points) - 1:
        return 0.0
    a = points[index - 1]
    b = points[index]
    c = points[index + 1]
    side_ab = distance(a, b)
    side_bc = distance(b, c)
    side_ac = distance(a, c)
    denominator = side_ab * side_bc * side_ac
    if denominator == 0.0:
        return 0.0
    doubled_area = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))
    if doubled_area == 0.0:
        return 0.0
    return 2.0 * doubled_area / denominator


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
