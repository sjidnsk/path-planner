from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from path_planner.core import Cell, WorldPoint

IRIS_BACKENDS = frozenset({"workspace_iris"})
IRIS_REPORT_STATUSES = frozenset({"ok", "fallback", "failed"})
IRIS_REGION_SOURCES = frozenset({"iris", "grid_box", "fallback"})
IRIS_FAILURE_STATUSES = frozenset(
    {
        "backend_unavailable",
        "invalid_region_input",
        "infeasible",
        "solver_error",
        "fallback_used",
    }
)


def _validate_choice(value: str | None, allowed: frozenset[str], name: str) -> None:
    if value is None:
        return
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")


def _cell_bounds_to_dict(min_cell: Cell, max_cell: Cell) -> dict[str, list[int]]:
    return {"min": min_cell.to_list(), "max": max_cell.to_list()}


def _world_bounds_to_dict(min_world: WorldPoint, max_world: WorldPoint) -> dict[str, list[float]]:
    return {"min": min_world.to_list(), "max": max_world.to_list()}


@dataclass(frozen=True)
class IrisRegion:
    region_id: int
    source: str
    seed_cell: Cell
    seed_world: WorldPoint
    min_cell: Cell
    max_cell: Cell
    min_world: WorldPoint
    max_world: WorldPoint
    domain_min_cell: Cell
    domain_max_cell: Cell
    domain_min_world: WorldPoint
    domain_max_world: WorldPoint
    hpolyhedron_a: tuple[tuple[float, ...], ...]
    hpolyhedron_b: tuple[float, ...]
    validation_status: str
    fallback_reason: str | None = None
    failure_status: str | None = None

    def __post_init__(self) -> None:
        if self.region_id < 0:
            raise ValueError("region_id must be nonnegative")
        _validate_choice(self.source, IRIS_REGION_SOURCES, "source")
        _validate_choice(self.failure_status, IRIS_FAILURE_STATUSES, "failure_status")
        if self.min_cell.x > self.max_cell.x or self.min_cell.y > self.max_cell.y:
            raise ValueError("min_cell must not exceed max_cell")
        if self.domain_min_cell.x > self.domain_max_cell.x or self.domain_min_cell.y > self.domain_max_cell.y:
            raise ValueError("domain_min_cell must not exceed domain_max_cell")
        if self.min_world.x > self.max_world.x or self.min_world.y > self.max_world.y:
            raise ValueError("min_world must not exceed max_world")
        if self.domain_min_world.x > self.domain_max_world.x or self.domain_min_world.y > self.domain_max_world.y:
            raise ValueError("domain_min_world must not exceed domain_max_world")
        if not self.hpolyhedron_a:
            raise ValueError("hpolyhedron_a must be nonempty")
        if len(self.hpolyhedron_a) != len(self.hpolyhedron_b):
            raise ValueError("hpolyhedron_a and hpolyhedron_b length mismatch")
        for row in self.hpolyhedron_a:
            if len(row) != 2:
                raise ValueError("hpolyhedron_a rows must be 2D")
        if self.validation_status == "":
            raise ValueError("validation_status must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "source": self.source,
            "seed_cell": self.seed_cell.to_list(),
            "seed_world": self.seed_world.to_list(),
            "cell_bounds": _cell_bounds_to_dict(self.min_cell, self.max_cell),
            "world_bounds": _world_bounds_to_dict(self.min_world, self.max_world),
            "domain_cell_bounds": _cell_bounds_to_dict(self.domain_min_cell, self.domain_max_cell),
            "domain_world_bounds": _world_bounds_to_dict(self.domain_min_world, self.domain_max_world),
            "hpolyhedron": {
                "A": [list(row) for row in self.hpolyhedron_a],
                "b": list(self.hpolyhedron_b),
            },
            "validation_status": self.validation_status,
            "failure_status": self.failure_status,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class IrisRegionReport:
    backend: str
    status: str
    seed_source: str
    domain_source: str
    obstacle_source: str
    regions: tuple[IrisRegion, ...] = ()
    obstacle_count: int = 0
    validation_status: str = "not_evaluated"
    failure_reason: str | None = None
    failure_status: str | None = None
    fallback_used: bool = False

    def __post_init__(self) -> None:
        _validate_choice(self.backend, IRIS_BACKENDS, "backend")
        _validate_choice(self.status, IRIS_REPORT_STATUSES, "status")
        _validate_choice(self.failure_status, IRIS_FAILURE_STATUSES, "failure_status")
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be nonnegative")
        if self.status == "ok" and (self.failure_reason is not None or self.failure_status is not None):
            raise ValueError("ok report cannot include failure reason/status")
        if self.status != "ok" and self.failure_reason is None:
            raise ValueError("non-ok report must include failure_reason")
        if self.status == "ok" and not self.regions:
            raise ValueError("ok report must include regions")
        if not self.seed_source:
            raise ValueError("seed_source must be nonempty")
        if not self.domain_source:
            raise ValueError("domain_source must be nonempty")
        if not self.obstacle_source:
            raise ValueError("obstacle_source must be nonempty")
        if not self.validation_status:
            raise ValueError("validation_status must be nonempty")

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "region_count": self.region_count,
            "seed_source": self.seed_source,
            "domain_source": self.domain_source,
            "obstacle_source": self.obstacle_source,
            "obstacle_count": self.obstacle_count,
            "validation_status": self.validation_status,
            "failure_status": self.failure_status,
            "failure_reason": self.failure_reason,
            "fallback_used": self.fallback_used,
            "regions": [region.to_dict() for region in self.regions],
        }
