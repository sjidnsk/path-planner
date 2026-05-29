from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

import numpy as np

from path_planner.core import CostGrid
from path_planner.platform import PlannerPlatformProfile


@dataclass(frozen=True)
class FootprintMaskResult:
    safe_mask: np.ndarray
    original_blocked_count: int
    inflated_blocked_count: int
    footprint_radius_m: float | None


def build_footprint_safe_mask(
    grid: CostGrid,
    platform_profile: PlannerPlatformProfile | None = None,
) -> FootprintMaskResult:
    original_blocked_count = int(np.count_nonzero(~grid.passable_mask))
    safe_mask = np.array(grid.passable_mask, dtype=bool, copy=True)
    footprint_radius_m = None if platform_profile is None else platform_profile.footprint_radius_m
    if footprint_radius_m is None or footprint_radius_m <= 0.0:
        return FootprintMaskResult(
            safe_mask=safe_mask,
            original_blocked_count=original_blocked_count,
            inflated_blocked_count=original_blocked_count,
            footprint_radius_m=footprint_radius_m,
        )

    radius_cells = int(ceil(footprint_radius_m / grid.spec.resolution))
    blocked_cells = np.argwhere(~grid.passable_mask)
    for blocked_y, blocked_x in blocked_cells:
        min_y = max(0, int(blocked_y) - radius_cells)
        max_y = min(grid.spec.height - 1, int(blocked_y) + radius_cells)
        min_x = max(0, int(blocked_x) - radius_cells)
        max_x = min(grid.spec.width - 1, int(blocked_x) + radius_cells)
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance_m = hypot((x - int(blocked_x)) * grid.spec.resolution, (y - int(blocked_y)) * grid.spec.resolution)
                if distance_m <= footprint_radius_m:
                    safe_mask[y, x] = False

    return FootprintMaskResult(
        safe_mask=safe_mask,
        original_blocked_count=original_blocked_count,
        inflated_blocked_count=int(np.count_nonzero(~safe_mask)),
        footprint_radius_m=footprint_radius_m,
    )
