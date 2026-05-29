from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec
from path_planner.platform import PlannerPlatformProfile
from path_planner.postprocess.footprint import FootprintMaskResult, build_footprint_safe_mask

STANDARD_GRID_ASTAR = "standard_grid_astar"
PLATFORM_AWARE_ASTAR = "platform_aware_astar"


@dataclass(frozen=True)
class SearchTerrainLayers:
    slope: np.ndarray | None = None
    roughness: np.ndarray | None = None
    illumination: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def layer_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for name in ("slope", "roughness", "illumination", "confidence"):
            if getattr(self, name) is not None:
                names.append(name)
        return tuple(names)

    def validate(self, spec: GridSpec) -> None:
        for name in self.layer_names():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != spec.shape:
                raise ValueError(f"terrain layer {name} shape {array.shape} must match grid shape {spec.shape}")


@dataclass(frozen=True)
class PlanningConstraints:
    platform_key: str | None
    footprint_radius_m: float | None
    safety_margin_m: float | None
    max_slope_deg: float | None
    max_obstacle_height_m: float | None
    ground_clearance_m: float | None
    min_turning_radius_m: float | None

    @classmethod
    def from_platform_profile(cls, profile: PlannerPlatformProfile | None) -> PlanningConstraints:
        if profile is None:
            return cls(
                platform_key=None,
                footprint_radius_m=None,
                safety_margin_m=None,
                max_slope_deg=None,
                max_obstacle_height_m=None,
                ground_clearance_m=None,
                min_turning_radius_m=None,
            )
        return cls(
            platform_key=profile.platform_key,
            footprint_radius_m=profile.footprint_radius_m,
            safety_margin_m=profile.safety_margin_m,
            max_slope_deg=profile.max_slope_deg,
            max_obstacle_height_m=profile.max_obstacle_height_m,
            ground_clearance_m=profile.ground_clearance_m,
            min_turning_radius_m=profile.effective_min_turning_radius_m,
        )


@dataclass(frozen=True)
class PlanningGrid:
    cost_grid: CostGrid
    original_passable_mask: np.ndarray
    inflated_passable_mask: np.ndarray
    constraints: PlanningConstraints
    platform_profile: PlannerPlatformProfile | None
    terrain_layers: SearchTerrainLayers
    search_mode: str
    passable_source: str
    original_blocked_count: int
    inflated_blocked_count: int
    footprint_radius_m: float | None

    @property
    def spec(self) -> GridSpec:
        return self.cost_grid.spec

    @property
    def cost(self) -> np.ndarray:
        return self.cost_grid.cost

    @property
    def terrain_cost(self) -> np.ndarray:
        return self.cost_grid.cost

    @property
    def passable_mask(self) -> np.ndarray:
        return self.inflated_passable_mask

    @property
    def platform_key(self) -> str | None:
        if self.platform_profile is None:
            return None
        return self.platform_profile.platform_key

    def is_passable(self, cell: Cell) -> bool:
        return self.spec.in_bounds(cell) and bool(self.inflated_passable_mask[cell.y, cell.x])

    def cost_at(self, cell: Cell) -> float:
        return self.cost_grid.cost_at(cell)

    def min_passable_cost(self) -> float:
        values = self.cost[self.inflated_passable_mask]
        if values.size == 0:
            return 0.0
        return float(max(np.min(values), 0.0))

    def search_metadata(self) -> dict[str, Any]:
        return {
            "search_mode": self.search_mode,
            "passable_source": self.passable_source,
            "platform_key": self.platform_key,
            "original_blocked_count": self.original_blocked_count,
            "inflated_blocked_count": self.inflated_blocked_count,
            "footprint_radius_m": self.footprint_radius_m,
            "terrain_layers": self.terrain_layers.layer_names(),
        }


def build_planning_grid(
    grid: CostGrid,
    *,
    platform_profile: PlannerPlatformProfile | None = None,
    terrain_layers: SearchTerrainLayers | None = None,
) -> PlanningGrid:
    terrain_layers = terrain_layers or SearchTerrainLayers()
    terrain_layers.validate(grid.spec)
    footprint = _build_search_footprint(grid, platform_profile)
    uses_platform_footprint = (
        platform_profile is not None
        and footprint.footprint_radius_m is not None
        and footprint.footprint_radius_m > 0.0
    )
    search_mode = PLATFORM_AWARE_ASTAR if uses_platform_footprint else STANDARD_GRID_ASTAR
    passable_source = "inflated_passable_mask" if uses_platform_footprint else "original_passable_mask"
    return PlanningGrid(
        cost_grid=grid,
        original_passable_mask=np.array(grid.passable_mask, dtype=bool, copy=True),
        inflated_passable_mask=np.array(footprint.safe_mask, dtype=bool, copy=True),
        constraints=PlanningConstraints.from_platform_profile(platform_profile),
        platform_profile=platform_profile,
        terrain_layers=terrain_layers,
        search_mode=search_mode,
        passable_source=passable_source,
        original_blocked_count=footprint.original_blocked_count,
        inflated_blocked_count=footprint.inflated_blocked_count,
        footprint_radius_m=footprint.footprint_radius_m,
    )


def _build_search_footprint(
    grid: CostGrid,
    platform_profile: PlannerPlatformProfile | None,
) -> FootprintMaskResult:
    return build_footprint_safe_mask(grid, platform_profile)
