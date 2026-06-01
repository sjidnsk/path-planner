from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from path_planner.core import Cell, WorldPoint

REGION_SOURCES = frozenset({"grid_box", "iris", "manual", "fallback"})
OBSTACLE_SOURCES = frozenset(
    {
        "blocked_cell_box",
        "merged_blocked_rectangle",
        "manual_convex_obstacle",
        "scene_graph_obstacle",
    }
)
EDGE_SOURCES = frozenset({"grid_adjacency", "sampled_connectivity", "manual", "gcs", "fallback"})


def _validate_source(value: str, allowed: frozenset[str], name: str) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")


def _cell_bounds_to_dict(min_cell: Cell, max_cell: Cell) -> dict[str, list[int]]:
    return {"min": min_cell.to_list(), "max": max_cell.to_list()}


def _world_bounds_to_dict(min_world: WorldPoint, max_world: WorldPoint) -> dict[str, list[float]]:
    return {"min": min_world.to_list(), "max": max_world.to_list()}


@dataclass(frozen=True)
class ObstaclePrimitive:
    obstacle_id: int
    source: str
    min_cell: Cell
    max_cell: Cell
    min_world: WorldPoint
    max_world: WorldPoint

    def __post_init__(self) -> None:
        if self.obstacle_id < 0:
            raise ValueError("obstacle_id must be nonnegative")
        _validate_source(self.source, OBSTACLE_SOURCES, "source")
        if self.min_cell.x > self.max_cell.x or self.min_cell.y > self.max_cell.y:
            raise ValueError("min_cell must not exceed max_cell")
        if self.min_world.x > self.max_world.x or self.min_world.y > self.max_world.y:
            raise ValueError("min_world must not exceed max_world")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.obstacle_id,
            "source": self.source,
            "cell_bounds": _cell_bounds_to_dict(self.min_cell, self.max_cell),
            "world_bounds": _world_bounds_to_dict(self.min_world, self.max_world),
        }


@dataclass(frozen=True)
class ConvexRegion:
    region_id: int
    source: str
    center_cell: Cell
    min_cell: Cell
    max_cell: Cell
    min_world: WorldPoint
    max_world: WorldPoint
    cell_count: int
    validation_status: str = "valid"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be nonnegative")
        _validate_source(self.source, REGION_SOURCES, "source")
        if self.min_cell.x > self.max_cell.x or self.min_cell.y > self.max_cell.y:
            raise ValueError("min_cell must not exceed max_cell")
        if self.min_world.x > self.max_world.x or self.min_world.y > self.max_world.y:
            raise ValueError("min_world must not exceed max_world")
        if self.cell_count < 0:
            raise ValueError("cell_count must be nonnegative")

    def overlaps_or_touches(self, other: ConvexRegion) -> bool:
        x_touch = self.min_cell.x <= other.max_cell.x + 1 and other.min_cell.x <= self.max_cell.x + 1
        y_touch = self.min_cell.y <= other.max_cell.y + 1 and other.min_cell.y <= self.max_cell.y + 1
        return x_touch and y_touch

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "source": self.source,
            "center_cell": self.center_cell.to_list(),
            "cell_bounds": _cell_bounds_to_dict(self.min_cell, self.max_cell),
            "world_bounds": _world_bounds_to_dict(self.min_world, self.max_world),
            "cell_count": self.cell_count,
            "validation_status": self.validation_status,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RegionEdge:
    edge_id: int
    source: str
    from_region_id: int
    to_region_id: int
    connection_kind: str

    def __post_init__(self) -> None:
        if self.edge_id < 0:
            raise ValueError("edge_id must be nonnegative")
        _validate_source(self.source, EDGE_SOURCES, "source")
        if self.from_region_id < 0 or self.to_region_id < 0:
            raise ValueError("region ids must be nonnegative")
        if self.from_region_id == self.to_region_id:
            raise ValueError("edge cannot connect a region to itself")
        if not self.connection_kind:
            raise ValueError("connection_kind must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.edge_id,
            "source": self.source,
            "from_region_id": self.from_region_id,
            "to_region_id": self.to_region_id,
            "connection_kind": self.connection_kind,
        }


@dataclass(frozen=True)
class RegionGraph:
    regions: tuple[ConvexRegion, ...]
    edges: tuple[RegionEdge, ...]
    obstacles: tuple[ObstaclePrimitive, ...] = ()

    @property
    def vertex_count(self) -> int:
        return len(self.regions)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [region.to_dict() for region in self.regions],
            "edges": [edge.to_dict() for edge in self.edges],
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }


@dataclass(frozen=True)
class RegionGraphReport:
    status: str
    region_source: str
    obstacle_source: str
    graph: RegionGraph | None = None
    failure_reason: str | None = None
    fallback_used: bool = False
    motion_feasibility_status: str = "not_evaluated"
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.region_source:
            _validate_source(self.region_source, REGION_SOURCES, "region_source")
        if self.obstacle_source:
            _validate_source(self.obstacle_source, OBSTACLE_SOURCES, "obstacle_source")
        if self.status == "ok" and self.graph is None:
            raise ValueError("ok report must include graph")
        if self.status != "ok" and self.failure_reason is None:
            raise ValueError("failed report must include failure_reason")

    @property
    def vertex_count(self) -> int:
        return 0 if self.graph is None else self.graph.vertex_count

    @property
    def edge_count(self) -> int:
        return 0 if self.graph is None else self.graph.edge_count

    @property
    def obstacle_count(self) -> int:
        return 0 if self.graph is None else self.graph.obstacle_count

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "vertex_count": self.vertex_count,
            "edge_count": self.edge_count,
            "obstacle_count": self.obstacle_count,
            "region_source": self.region_source,
            "obstacle_source": self.obstacle_source,
            "failure_reason": self.failure_reason,
            "fallback_used": self.fallback_used,
            "motion_feasibility_status": self.motion_feasibility_status,
            "quality_metrics": dict(self.quality_metrics),
        }
        if self.graph is not None:
            payload["graph"] = self.graph.to_dict()
        return payload
