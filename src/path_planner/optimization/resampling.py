from __future__ import annotations

import math

from path_planner.core import WorldPoint


def resample_world_points(points: tuple[WorldPoint, ...], *, spacing_m: float) -> tuple[WorldPoint, ...]:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if len(points) <= 1:
        return points

    cumulative = _cumulative_lengths(points)
    total_length = cumulative[-1]
    if total_length == 0.0:
        return (points[0], points[-1])

    samples = [points[0]]
    sample_s = spacing_m
    while sample_s < total_length:
        samples.append(_point_at_s(points, cumulative, sample_s))
        sample_s += spacing_m
    if samples[-1] != points[-1]:
        samples.append(points[-1])
    return tuple(samples)


def _cumulative_lengths(points: tuple[WorldPoint, ...]) -> tuple[float, ...]:
    values = [0.0]
    for previous, current in zip(points[:-1], points[1:]):
        values.append(values[-1] + math.hypot(current.x - previous.x, current.y - previous.y))
    return tuple(values)


def _point_at_s(points: tuple[WorldPoint, ...], cumulative: tuple[float, ...], path_s: float) -> WorldPoint:
    clamped_s = max(0.0, min(cumulative[-1], path_s))
    for index in range(len(points) - 1):
        if cumulative[index] <= clamped_s <= cumulative[index + 1]:
            segment_length = cumulative[index + 1] - cumulative[index]
            if segment_length == 0.0:
                return points[index]
            ratio = (clamped_s - cumulative[index]) / segment_length
            return WorldPoint(
                points[index].x + ratio * (points[index + 1].x - points[index].x),
                points[index].y + ratio * (points[index + 1].y - points[index].y),
            )
    return points[-1]
