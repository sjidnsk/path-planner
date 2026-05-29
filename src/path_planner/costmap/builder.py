from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from path_planner.core import CostGrid, GridSpec
from path_planner.platform import PlannerPlatformProfile


@dataclass(frozen=True)
class CostmapWeights:
    base_cost: float = 1.0
    slope: float = 2.0
    roughness: float = 1.5
    shadow: float = 1.0
    low_confidence: float = 1.0


@dataclass(frozen=True)
class PlatformLimits:
    max_slope_deg: float
    obstacle_threshold: float


@dataclass(frozen=True)
class SemanticLayers:
    cost: np.ndarray | None = None
    passable_mask: np.ndarray | None = None
    slope: np.ndarray | None = None
    roughness: np.ndarray | None = None
    illumination: np.ndarray | None = None
    confidence: np.ndarray | None = None
    obstacle: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


def _layer_or_default(layer: np.ndarray | None, shape: tuple[int, int], value: float) -> np.ndarray:
    if layer is None:
        return np.full(shape, value, dtype=float)
    array = np.asarray(layer, dtype=float)
    if array.shape != shape:
        raise ValueError(f"semantic layer shape {array.shape} must match grid shape {shape}")
    return array


def _mask_or_default(mask: np.ndarray | None, shape: tuple[int, int], value: bool) -> np.ndarray:
    if mask is None:
        return np.full(shape, value, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"mask shape {array.shape} must match grid shape {shape}")
    return array


def build_cost_grid(
    spec: GridSpec,
    layers: SemanticLayers,
    platform: PlatformLimits | None = None,
    weights: CostmapWeights | None = None,
) -> CostGrid:
    weights = weights or CostmapWeights()
    shape = spec.shape

    if layers.cost is not None and layers.passable_mask is not None:
        return CostGrid(
            spec=spec,
            cost=np.asarray(layers.cost, dtype=float),
            passable_mask=np.asarray(layers.passable_mask, dtype=bool),
            metadata={"source": "cost_passable_mask"},
        )

    if platform is None:
        raise ValueError("platform limits are required when building a semantic layer costmap")

    slope = _layer_or_default(layers.slope, shape, 0.0)
    roughness = np.clip(_layer_or_default(layers.roughness, shape, 0.0), 0.0, 1.0)
    illumination = np.clip(_layer_or_default(layers.illumination, shape, 1.0), 0.0, 1.0)
    confidence = np.clip(_layer_or_default(layers.confidence, shape, 1.0), 0.0, 1.0)
    obstacle = _layer_or_default(layers.obstacle, shape, 0.0)
    valid_mask = _mask_or_default(layers.valid_mask, shape, True)

    slope_ratio = np.clip(slope / max(platform.max_slope_deg, 1e-9), 0.0, None)
    shadow_risk = 1.0 - illumination
    low_confidence_risk = 1.0 - confidence

    passable = (
        valid_mask
        & np.isfinite(slope)
        & (slope <= platform.max_slope_deg)
        & (obstacle < platform.obstacle_threshold)
    )
    cost = (
        weights.base_cost
        + weights.slope * np.clip(slope_ratio, 0.0, 1.0)
        + weights.roughness * roughness
        + weights.shadow * shadow_risk
        + weights.low_confidence * low_confidence_risk
    )
    cost = np.where(np.isfinite(cost), cost, weights.base_cost)
    cost = np.maximum(cost, 0.0)

    return CostGrid(
        spec=spec,
        cost=cost,
        passable_mask=passable,
        metadata={
            "source": "semantic_layers",
            "weights": weights.__dict__,
            "platform": platform.__dict__,
        },
    )


def platform_limits_from_profile(profile: PlannerPlatformProfile) -> PlatformLimits:
    missing: list[str] = []
    if profile.max_slope_deg is None:
        missing.append("max_slope_deg")
    if profile.max_obstacle_height_m is None:
        missing.append("max_obstacle_height")
    if missing:
        raise ValueError(f"platform profile missing costmap limits: {', '.join(missing)}")
    return PlatformLimits(
        max_slope_deg=float(profile.max_slope_deg),
        obstacle_threshold=float(profile.max_obstacle_height_m),
    )
