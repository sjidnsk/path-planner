# path-planner Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working `path_planner` package: clean core data models, optional lunar costmap synthesis, adapter boundaries, 2D A* route planning, CLI JSON output, and PNG/HTML diagnostic reports.

**Architecture:** The package uses a `src/` layout. `core` owns stable internal models, `search` consumes only `CostGrid` and `PlanRequest`, `costmap` converts semantic lunar layers into `CostGrid`, `adapters` isolates external contracts, and `diagnostics`/`cli` provide user-facing outputs. GCS/Ackermann, Drake, exploration target selection, and observation updates remain outside Phase 1.

**Tech Stack:** Python 3.12, NumPy, Matplotlib with non-interactive `Agg` backend, pytest, standard-library `argparse`, `json`, `dataclasses`, `enum`, `heapq`, and `pathlib`.

---

## File Structure

Create this structure under `D:\codex\project\lunar-path-planning\path-planner`:

- `pyproject.toml`: package metadata, runtime dependencies, pytest config, CLI entry point.
- `README.md`: Phase 1 usage, scope, verification commands, and non-goals.
- `src/path_planner/__init__.py`: package exports and version.
- `src/path_planner/core/__init__.py`: core model exports.
- `src/path_planner/core/models.py`: `Cell`, `WorldPoint`, `GridSpec`, `CostGrid`, `PlanRequest`, `PlanDiagnostics`, `PlanResult`, `FailureReason`, `NeighborPolicy`, and schema helpers.
- `src/path_planner/costmap/__init__.py`: costmap exports.
- `src/path_planner/costmap/builder.py`: `CostmapWeights`, `PlatformLimits`, `SemanticLayers`, and `build_cost_grid(...)`.
- `src/path_planner/adapters/__init__.py`: adapter exports.
- `src/path_planner/adapters/dev_platform.py`: `DevPlatformAdapter` for arrays compatible with `dev-platform-constraints`.
- `src/path_planner/adapters/json_io.py`: JSON load/dump helpers for CLI and `model-explorer` contract-style exchange.
- `src/path_planner/search/__init__.py`: search exports.
- `src/path_planner/search/astar.py`: `AStarPlanner` and internal priority queue records.
- `src/path_planner/diagnostics/__init__.py`: diagnostics exports.
- `src/path_planner/diagnostics/render.py`: PNG/HTML diagnostic renderer.
- `src/path_planner/cli.py`: CLI entry point.
- `tests/test_package_imports.py`: package import smoke tests.
- `tests/test_core_models.py`: core model validation tests.
- `tests/test_costmap_builder.py`: semantic costmap tests.
- `tests/test_astar.py`: success, cost preference, corner-cut, and failure tests.
- `tests/test_adapters_json.py`: adapter and route schema tests.
- `tests/test_cli_diagnostics.py`: CLI and diagnostics smoke tests.
- `examples/demo_map.json`: small deterministic demo map for CLI.

Do not create GCS, IRIS, Ackermann, Drake, service API, exploration-loop, or observation-update modules in Phase 1.

---

### Task 1: Project Scaffold And Import Contract

**Files:**
- Create: `pyproject.toml`
- Create: `src/path_planner/__init__.py`
- Create: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing package import test**

Create `tests/test_package_imports.py`:

```python
def test_package_imports_version():
    import path_planner

    assert path_planner.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the import test and verify it fails**

Run:

```powershell
python -m pytest tests/test_package_imports.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'path_planner'`.

- [ ] **Step 3: Add package metadata and minimal package**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "path-planner"
version = "0.1.0"
description = "Phase 1 lunar path planner with costmap-aware A* and diagnostics"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26,<2.3",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
]

[project.scripts]
path-planner = "path_planner.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

Create `src/path_planner/__init__.py`:

```python
"""Lunar path planning package."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Run the import test and verify it passes**

Run:

```powershell
python -m pytest tests/test_package_imports.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

Run:

```powershell
git add pyproject.toml src/path_planner/__init__.py tests/test_package_imports.py
git commit -m "chore: scaffold path planner package"
```

Expected: commit succeeds.

---

### Task 2: Core Models And Validation

**Files:**
- Create: `src/path_planner/core/__init__.py`
- Create: `src/path_planner/core/models.py`
- Create: `tests/test_core_models.py`

- [ ] **Step 1: Write failing core model tests**

Create `tests/test_core_models.py`:

```python
import math

import numpy as np
import pytest

from path_planner.core import (
    Cell,
    CostGrid,
    FailureReason,
    GridSpec,
    NeighborPolicy,
    PlanDiagnostics,
    PlanRequest,
    PlanResult,
    WorldPoint,
)


def test_grid_spec_converts_between_cell_and_world():
    spec = GridSpec(width=4, height=3, resolution=0.5, origin=(10.0, -2.0), frame_id="moon")

    assert spec.in_bounds(Cell(2, 1))
    assert not spec.in_bounds(Cell(4, 1))
    assert spec.cell_to_world(Cell(2, 1)) == WorldPoint(11.0, -1.5)
    assert spec.world_to_cell(WorldPoint(11.2, -1.2)) == Cell(2, 1)


def test_cost_grid_rejects_invalid_passable_costs():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    cost = np.array([[1.0, math.inf], [1.0, 1.0]])
    mask = np.array([[True, True], [True, False]])

    with pytest.raises(ValueError, match="passable cells must have finite nonnegative cost"):
        CostGrid(spec=spec, cost=cost, passable_mask=mask)


def test_plan_result_route_dict_marks_geometric_path():
    spec = GridSpec(width=3, height=3, resolution=2.0)
    result = PlanResult(
        success=True,
        path_cells=(Cell(0, 0), Cell(1, 1)),
        path_world=(WorldPoint(0.0, 0.0), WorldPoint(2.0, 2.0)),
        total_cost=2.5,
        expanded_count=4,
        failure_reason=None,
        diagnostics=PlanDiagnostics(runtime_ms=1.0, max_frontier_size=2),
    )

    route = result.to_route_dict(spec)

    assert route["schema_version"] == "path-planner-route/v1"
    assert route["trajectory_kind"] == "geometric_path"
    assert route["reachable"] is True
    assert route["path_cost"] == 2.5
    assert route["geometric_path"]["cells"] == [[0, 0], [1, 1]]


def test_plan_request_defaults_to_8_neighbor_with_corner_cut_protection():
    request = PlanRequest(start=Cell(0, 0), goal=Cell(2, 2))

    assert request.neighbor_policy is NeighborPolicy.EIGHT
    assert request.prevent_corner_cutting is True
    assert request.max_iterations == 100_000
    assert FailureReason.UNREACHABLE.value == "unreachable"
```

- [ ] **Step 2: Run core tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_core_models.py -q
```

Expected: FAIL with import errors for `path_planner.core`.

- [ ] **Step 3: Implement core models**

Create `src/path_planner/core/__init__.py`:

```python
"""Core path planner data models."""

from .models import (
    Cell,
    CostGrid,
    FailureReason,
    GridSpec,
    NeighborPolicy,
    PlanDiagnostics,
    PlanRequest,
    PlanResult,
    WorldPoint,
)

__all__ = [
    "Cell",
    "CostGrid",
    "FailureReason",
    "GridSpec",
    "NeighborPolicy",
    "PlanDiagnostics",
    "PlanRequest",
    "PlanResult",
    "WorldPoint",
]
```

Create `src/path_planner/core/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import floor
from typing import Any

import numpy as np

ROUTE_SCHEMA_VERSION = "path-planner-route/v1"


@dataclass(frozen=True, order=True)
class Cell:
    x: int
    y: int

    def to_list(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float

    def to_list(self) -> list[float]:
        return [float(self.x), float(self.y)]


class NeighborPolicy(str, Enum):
    FOUR = "4-neighbor"
    EIGHT = "8-neighbor"


class FailureReason(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_COST = "invalid_cost"
    START_OUT_OF_BOUNDS = "start_out_of_bounds"
    GOAL_OUT_OF_BOUNDS = "goal_out_of_bounds"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    UNREACHABLE = "unreachable"
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if len(self.origin) != 2:
            raise ValueError("origin must contain x and y")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def cell_to_world(self, cell: Cell) -> WorldPoint:
        return WorldPoint(
            self.origin[0] + cell.x * self.resolution,
            self.origin[1] + cell.y * self.resolution,
        )

    def world_to_cell(self, point: WorldPoint) -> Cell:
        return Cell(
            int(floor((point.x - self.origin[0]) / self.resolution)),
            int(floor((point.y - self.origin[1]) / self.resolution)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": [self.origin[0], self.origin[1]],
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class CostGrid:
    spec: GridSpec
    cost: np.ndarray
    passable_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cost = np.asarray(self.cost, dtype=float)
        mask = np.asarray(self.passable_mask, dtype=bool)
        if cost.shape != self.spec.shape:
            raise ValueError(f"cost shape {cost.shape} must match grid shape {self.spec.shape}")
        if mask.shape != self.spec.shape:
            raise ValueError(f"passable_mask shape {mask.shape} must match grid shape {self.spec.shape}")
        passable_cost = cost[mask]
        if passable_cost.size and (not np.all(np.isfinite(passable_cost)) or np.any(passable_cost < 0.0)):
            raise ValueError("passable cells must have finite nonnegative cost")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "passable_mask", mask)

    def is_passable(self, cell: Cell) -> bool:
        return self.spec.in_bounds(cell) and bool(self.passable_mask[cell.y, cell.x])

    def cost_at(self, cell: Cell) -> float:
        return float(self.cost[cell.y, cell.x])

    def min_passable_cost(self) -> float:
        values = self.cost[self.passable_mask]
        if values.size == 0:
            return 0.0
        return float(max(np.min(values), 0.0))


@dataclass(frozen=True)
class PlanRequest:
    start: Cell
    goal: Cell
    neighbor_policy: NeighborPolicy = NeighborPolicy.EIGHT
    prevent_corner_cutting: bool = True
    max_iterations: int = 100_000

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class PlanDiagnostics:
    runtime_ms: float = 0.0
    max_frontier_size: int = 0
    path_length_m: float = 0.0
    expanded_cells: tuple[Cell, ...] = ()
    cost_min: float | None = None
    cost_max: float | None = None
    cost_mean: float | None = None
    neighbor_policy: str = NeighborPolicy.EIGHT.value
    prevent_corner_cutting: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_ms": self.runtime_ms,
            "max_frontier_size": self.max_frontier_size,
            "path_length_m": self.path_length_m,
            "expanded_cells": [cell.to_list() for cell in self.expanded_cells],
            "cost_min": self.cost_min,
            "cost_max": self.cost_max,
            "cost_mean": self.cost_mean,
            "neighbor_policy": self.neighbor_policy,
            "prevent_corner_cutting": self.prevent_corner_cutting,
        }


@dataclass(frozen=True)
class PlanResult:
    success: bool
    path_cells: tuple[Cell, ...]
    path_world: tuple[WorldPoint, ...]
    total_cost: float
    expanded_count: int
    failure_reason: FailureReason | None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)

    def __post_init__(self) -> None:
        if self.success and self.failure_reason is not None:
            raise ValueError("successful result cannot have failure_reason")
        if not self.success and self.failure_reason is None:
            raise ValueError("failed result must have failure_reason")

    def to_route_dict(self, spec: GridSpec) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "trajectory_kind": "geometric_path",
            "reachable": self.success,
            "path_cost": self.total_cost if self.success else None,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "grid": spec.to_dict(),
            "geometric_path": {
                "cells": [cell.to_list() for cell in self.path_cells],
                "world": [point.to_list() for point in self.path_world],
            },
            "expanded_count": self.expanded_count,
            "diagnostics": self.diagnostics.to_dict(),
        }
```

- [ ] **Step 4: Run core tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_core_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit core models**

Run:

```powershell
git add src/path_planner/core tests/test_core_models.py
git commit -m "feat: add core planning models"
```

Expected: commit succeeds.

---

### Task 3: Costmap Builder

**Files:**
- Create: `src/path_planner/costmap/__init__.py`
- Create: `src/path_planner/costmap/builder.py`
- Create: `tests/test_costmap_builder.py`

- [ ] **Step 1: Write failing costmap tests**

Create `tests/test_costmap_builder.py`:

```python
import numpy as np

from path_planner.core import GridSpec
from path_planner.costmap import CostmapWeights, PlatformLimits, SemanticLayers, build_cost_grid


def test_build_cost_grid_applies_hard_constraints_and_soft_costs():
    spec = GridSpec(width=3, height=2, resolution=1.0)
    layers = SemanticLayers(
        slope=np.array([[0.0, 10.0, 40.0], [5.0, 5.0, 5.0]]),
        roughness=np.array([[0.0, 0.5, 0.0], [0.2, 0.0, 0.0]]),
        illumination=np.array([[1.0, 0.5, 1.0], [0.0, 1.0, 1.0]]),
        confidence=np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]]),
        obstacle=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        valid_mask=np.array([[True, True, True], [True, True, False]]),
    )

    grid = build_cost_grid(spec, layers, PlatformLimits(max_slope_deg=30.0), CostmapWeights())

    assert grid.passable_mask.tolist() == [[True, True, False], [True, False, False]]
    assert grid.cost[0, 0] == 1.0
    assert grid.cost[0, 1] > grid.cost[0, 0]
    assert grid.metadata["source"] == "semantic_layers"


def test_build_cost_grid_can_wrap_existing_cost_and_mask():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [True, True]])

    grid = build_cost_grid(spec, SemanticLayers(cost=cost, passable_mask=mask))

    assert grid.cost.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert grid.passable_mask.tolist() == [[True, False], [True, True]]
    assert grid.metadata["source"] == "cost_passable_mask"
```

- [ ] **Step 2: Run costmap tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_costmap_builder.py -q
```

Expected: FAIL with import errors for `path_planner.costmap`.

- [ ] **Step 3: Implement the costmap builder**

Create `src/path_planner/costmap/__init__.py`:

```python
"""Costmap construction utilities."""

from .builder import CostmapWeights, PlatformLimits, SemanticLayers, build_cost_grid

__all__ = ["CostmapWeights", "PlatformLimits", "SemanticLayers", "build_cost_grid"]
```

Create `src/path_planner/costmap/builder.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from path_planner.core import CostGrid, GridSpec


@dataclass(frozen=True)
class CostmapWeights:
    base_cost: float = 1.0
    slope: float = 2.0
    roughness: float = 1.5
    shadow: float = 1.0
    low_confidence: float = 1.0


@dataclass(frozen=True)
class PlatformLimits:
    max_slope_deg: float = 30.0
    obstacle_threshold: float = 0.5


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
    platform = platform or PlatformLimits()
    weights = weights or CostmapWeights()
    shape = spec.shape

    if layers.cost is not None and layers.passable_mask is not None:
        return CostGrid(
            spec=spec,
            cost=np.asarray(layers.cost, dtype=float),
            passable_mask=np.asarray(layers.passable_mask, dtype=bool),
            metadata={"source": "cost_passable_mask"},
        )

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
```

- [ ] **Step 4: Run costmap tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_costmap_builder.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit costmap builder**

Run:

```powershell
git add src/path_planner/costmap tests/test_costmap_builder.py
git commit -m "feat: add semantic costmap builder"
```

Expected: commit succeeds.

---

### Task 4: A* Search Success Cases

**Files:**
- Create: `src/path_planner/search/__init__.py`
- Create: `src/path_planner/search/astar.py`
- Create: `tests/test_astar.py`

- [ ] **Step 1: Write failing A* success tests**

Create `tests/test_astar.py`:

```python
import numpy as np

from path_planner.core import Cell, CostGrid, FailureReason, GridSpec, PlanRequest
from path_planner.search import AStarPlanner


def grid_from(cost, mask=None):
    cost_array = np.asarray(cost, dtype=float)
    spec = GridSpec(width=cost_array.shape[1], height=cost_array.shape[0], resolution=1.0)
    passable = np.ones(cost_array.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return CostGrid(spec=spec, cost=cost_array, passable_mask=passable)


def test_astar_finds_diagonal_path_on_empty_grid():
    grid = grid_from(np.ones((3, 3)))
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))

    assert result.success is True
    assert result.failure_reason is None
    assert result.path_cells == (Cell(0, 0), Cell(1, 1), Cell(2, 2))
    assert result.total_cost > 0.0
    assert result.expanded_count > 0


def test_astar_prefers_lower_weighted_route():
    cost = np.array(
        [
            [1.0, 20.0, 1.0],
            [1.0, 20.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    grid = grid_from(cost)
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 0)))

    assert result.success is True
    assert Cell(1, 0) not in result.path_cells
    assert result.path_cells[-1] == Cell(2, 0)
```

- [ ] **Step 2: Run A* tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_astar.py::test_astar_finds_diagonal_path_on_empty_grid tests/test_astar.py::test_astar_prefers_lower_weighted_route -q
```

Expected: FAIL with import errors for `path_planner.search`.

- [ ] **Step 3: Implement A* for success paths**

Create `src/path_planner/search/__init__.py`:

```python
"""Search algorithms."""

from .astar import AStarPlanner

__all__ = ["AStarPlanner"]
```

Create `src/path_planner/search/astar.py`:

```python
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from path_planner.core import (
    Cell,
    CostGrid,
    FailureReason,
    NeighborPolicy,
    PlanDiagnostics,
    PlanRequest,
    PlanResult,
)


@dataclass(order=True)
class _QueueItem:
    priority: float
    order: int
    cell: Cell = field(compare=False)


class AStarPlanner:
    def plan(self, grid: CostGrid, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        early_failure = self._validate_request(grid, request, started)
        if early_failure is not None:
            return early_failure

        frontier: list[_QueueItem] = []
        counter = 0
        heapq.heappush(frontier, _QueueItem(0.0, counter, request.start))
        came_from: dict[Cell, Cell | None] = {request.start: None}
        cost_so_far: dict[Cell, float] = {request.start: 0.0}
        expanded: list[Cell] = []
        max_frontier_size = 1

        min_cost = grid.min_passable_cost()

        while frontier:
            if len(expanded) >= request.max_iterations:
                return self._failure(grid, request, FailureReason.MAX_ITERATIONS, expanded, max_frontier_size, started)

            current = heapq.heappop(frontier).cell
            expanded.append(current)

            if current == request.goal:
                path = self._reconstruct_path(came_from, current)
                total_cost = cost_so_far[current]
                return self._success(grid, request, path, total_cost, expanded, max_frontier_size, started)

            for neighbor, step_distance in self._neighbors(grid, request, current):
                new_cost = cost_so_far[current] + step_distance * grid.cost_at(neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    counter += 1
                    priority = new_cost + self._heuristic(neighbor, request.goal, min_cost)
                    heapq.heappush(frontier, _QueueItem(priority, counter, neighbor))
                    came_from[neighbor] = current
            max_frontier_size = max(max_frontier_size, len(frontier))

        return self._failure(grid, request, FailureReason.UNREACHABLE, expanded, max_frontier_size, started)

    def _validate_request(self, grid: CostGrid, request: PlanRequest, started: float) -> PlanResult | None:
        if not grid.spec.in_bounds(request.start):
            return self._failure(grid, request, FailureReason.START_OUT_OF_BOUNDS, (), 0, started)
        if not grid.spec.in_bounds(request.goal):
            return self._failure(grid, request, FailureReason.GOAL_OUT_OF_BOUNDS, (), 0, started)
        if not grid.is_passable(request.start):
            return self._failure(grid, request, FailureReason.START_BLOCKED, (), 0, started)
        if not grid.is_passable(request.goal):
            return self._failure(grid, request, FailureReason.GOAL_BLOCKED, (), 0, started)
        return None

    def _neighbors(self, grid: CostGrid, request: PlanRequest, cell: Cell) -> list[tuple[Cell, float]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if request.neighbor_policy is NeighborPolicy.EIGHT:
            directions.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])

        result: list[tuple[Cell, float]] = []
        for dx, dy in directions:
            neighbor = Cell(cell.x + dx, cell.y + dy)
            if not grid.is_passable(neighbor):
                continue
            is_diagonal = dx != 0 and dy != 0
            if is_diagonal and request.prevent_corner_cutting:
                side_a = Cell(cell.x + dx, cell.y)
                side_b = Cell(cell.x, cell.y + dy)
                if not grid.is_passable(side_a) and not grid.is_passable(side_b):
                    continue
            result.append((neighbor, math.sqrt(2.0) if is_diagonal else 1.0))
        return result

    def _heuristic(self, cell: Cell, goal: Cell, min_cost: float) -> float:
        dx = abs(goal.x - cell.x)
        dy = abs(goal.y - cell.y)
        return min_cost * ((dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy))

    def _reconstruct_path(self, came_from: dict[Cell, Cell | None], current: Cell) -> tuple[Cell, ...]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return tuple(path)

    def _success(
        self,
        grid: CostGrid,
        request: PlanRequest,
        path: tuple[Cell, ...],
        total_cost: float,
        expanded: list[Cell],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = self._diagnostics(grid, request, path, expanded, max_frontier_size, started)
        return PlanResult(
            success=True,
            path_cells=path,
            path_world=tuple(grid.spec.cell_to_world(cell) for cell in path),
            total_cost=float(total_cost),
            expanded_count=len(expanded),
            failure_reason=None,
            diagnostics=diagnostics,
        )

    def _failure(
        self,
        grid: CostGrid,
        request: PlanRequest,
        reason: FailureReason,
        expanded: list[Cell] | tuple[Cell, ...],
        max_frontier_size: int,
        started: float,
    ) -> PlanResult:
        diagnostics = self._diagnostics(grid, request, (), tuple(expanded), max_frontier_size, started)
        return PlanResult(
            success=False,
            path_cells=(),
            path_world=(),
            total_cost=math.inf,
            expanded_count=len(expanded),
            failure_reason=reason,
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        grid: CostGrid,
        request: PlanRequest,
        path: tuple[Cell, ...],
        expanded: list[Cell] | tuple[Cell, ...],
        max_frontier_size: int,
        started: float,
    ) -> PlanDiagnostics:
        passable_cost = grid.cost[grid.passable_mask]
        path_length = self._path_length(path, grid.spec.resolution)
        return PlanDiagnostics(
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_frontier_size=max_frontier_size,
            path_length_m=path_length,
            expanded_cells=tuple(expanded),
            cost_min=float(np.min(passable_cost)) if passable_cost.size else None,
            cost_max=float(np.max(passable_cost)) if passable_cost.size else None,
            cost_mean=float(np.mean(passable_cost)) if passable_cost.size else None,
            neighbor_policy=request.neighbor_policy.value,
            prevent_corner_cutting=request.prevent_corner_cutting,
        )

    def _path_length(self, path: tuple[Cell, ...], resolution: float) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y) * resolution
        return total
```

- [ ] **Step 4: Run A* success tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_astar.py::test_astar_finds_diagonal_path_on_empty_grid tests/test_astar.py::test_astar_prefers_lower_weighted_route -q
```

Expected: PASS.

- [ ] **Step 5: Commit A* success path implementation**

Run:

```powershell
git add src/path_planner/search tests/test_astar.py
git commit -m "feat: add weighted astar search"
```

Expected: commit succeeds.

---

### Task 5: A* Failure Cases And Corner-Cut Protection

**Files:**
- Modify: `tests/test_astar.py`
- Modify: `src/path_planner/search/astar.py`

- [ ] **Step 1: Add failing failure-mode tests**

Append to `tests/test_astar.py`:

```python

def test_astar_prevents_diagonal_corner_cutting():
    mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )
    grid = grid_from(np.ones((2, 2)), mask)
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(1, 1)))

    assert result.success is False
    assert result.failure_reason is FailureReason.UNREACHABLE


def test_astar_reports_blocked_start_and_goal():
    mask = np.array([[False, True], [True, False]])
    grid = grid_from(np.ones((2, 2)), mask)

    start_blocked = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(1, 0)))
    goal_blocked = AStarPlanner().plan(grid, PlanRequest(start=Cell(1, 0), goal=Cell(1, 1)))

    assert start_blocked.failure_reason is FailureReason.START_BLOCKED
    assert goal_blocked.failure_reason is FailureReason.GOAL_BLOCKED


def test_astar_reports_out_of_bounds():
    grid = grid_from(np.ones((2, 2)))

    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(-1, 0), goal=Cell(1, 1)))

    assert result.success is False
    assert result.failure_reason is FailureReason.START_OUT_OF_BOUNDS


def test_astar_reports_max_iterations():
    grid = grid_from(np.ones((5, 5)))
    result = AStarPlanner().plan(
        grid,
        PlanRequest(start=Cell(0, 0), goal=Cell(4, 4), max_iterations=1),
    )

    assert result.success is False
    assert result.failure_reason is FailureReason.MAX_ITERATIONS
    assert result.expanded_count == 1
```

- [ ] **Step 2: Run failure-mode tests**

Run:

```powershell
python -m pytest tests/test_astar.py -q
```

Expected: PASS if Task 4 implementation already covered these cases. If `test_astar_reports_max_iterations` fails because the check happens after expanding too many nodes, adjust the loop as shown in Step 3.

- [ ] **Step 3: Adjust max-iteration accounting only if the test fails**

In `src/path_planner/search/astar.py`, keep the loop guard before appending `current`:

```python
while frontier:
    if len(expanded) >= request.max_iterations:
        return self._failure(grid, request, FailureReason.MAX_ITERATIONS, expanded, max_frontier_size, started)

    current = heapq.heappop(frontier).cell
    expanded.append(current)
```

- [ ] **Step 4: Run all A* tests**

Run:

```powershell
python -m pytest tests/test_astar.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit failure handling tests**

Run:

```powershell
git add src/path_planner/search/astar.py tests/test_astar.py
git commit -m "test: cover astar failure modes"
```

Expected: commit succeeds.

---

### Task 6: External Adapters And JSON Contract

**Files:**
- Create: `src/path_planner/adapters/__init__.py`
- Create: `src/path_planner/adapters/dev_platform.py`
- Create: `src/path_planner/adapters/json_io.py`
- Create: `tests/test_adapters_json.py`
- Create: `examples/demo_map.json`

- [ ] **Step 1: Write failing adapter and JSON tests**

Create `tests/test_adapters_json.py`:

```python
import json

import numpy as np

from path_planner.adapters import DevPlatformAdapter, load_plan_input, route_result_to_json_dict
from path_planner.core import Cell, FailureReason, GridSpec, PlanDiagnostics, PlanResult, WorldPoint


def test_dev_platform_adapter_wraps_cost_and_mask_with_metadata():
    adapter = DevPlatformAdapter()

    grid = adapter.from_arrays(
        cost=np.array([[1.0, 2.0], [3.0, 4.0]]),
        passable_mask=np.array([[True, False], [True, True]]),
        resolution=0.5,
        origin=(1.0, 2.0),
        frame_id="moon",
        layers={"slope": np.array([[0.0, 5.0], [10.0, 20.0]])},
    )

    assert grid.spec == GridSpec(width=2, height=2, resolution=0.5, origin=(1.0, 2.0), frame_id="moon")
    assert grid.metadata["adapter"] == "dev-platform-constraints"
    assert grid.metadata["layers"] == ["slope"]


def test_load_plan_input_reads_internal_json_contract(tmp_path):
    payload = {
        "schema_version": "path-planner-request/v1",
        "grid": {"width": 2, "height": 2, "resolution": 1.0, "origin": [0.0, 0.0], "frame_id": "map"},
        "cost": [[1.0, 1.0], [1.0, 1.0]],
        "passable_mask": [[True, True], [True, True]],
        "start": [0, 0],
        "goal": [1, 1],
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    grid, request = load_plan_input(path)

    assert grid.spec.width == 2
    assert request.start == Cell(0, 0)
    assert request.goal == Cell(1, 1)


def test_route_result_to_json_dict_preserves_failure_reason():
    spec = GridSpec(width=2, height=2, resolution=1.0)
    result = PlanResult(
        success=False,
        path_cells=(),
        path_world=(),
        total_cost=float("inf"),
        expanded_count=0,
        failure_reason=FailureReason.UNREACHABLE,
        diagnostics=PlanDiagnostics(),
    )

    payload = route_result_to_json_dict(result, spec)

    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["reachable"] is False
    assert payload["failure_reason"] == "unreachable"
```

- [ ] **Step 2: Run adapter tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_adapters_json.py -q
```

Expected: FAIL with import errors for `path_planner.adapters`.

- [ ] **Step 3: Implement adapters and JSON helpers**

Create `src/path_planner/adapters/__init__.py`:

```python
"""External contract adapters."""

from .dev_platform import DevPlatformAdapter
from .json_io import load_plan_input, route_result_to_json_dict

__all__ = ["DevPlatformAdapter", "load_plan_input", "route_result_to_json_dict"]
```

Create `src/path_planner/adapters/dev_platform.py`:

```python
from __future__ import annotations

from typing import Mapping

import numpy as np

from path_planner.core import CostGrid, GridSpec


class DevPlatformAdapter:
    def from_arrays(
        self,
        *,
        cost: np.ndarray,
        passable_mask: np.ndarray,
        resolution: float,
        origin: tuple[float, float] = (0.0, 0.0),
        frame_id: str = "map",
        layers: Mapping[str, np.ndarray] | None = None,
    ) -> CostGrid:
        cost_array = np.asarray(cost, dtype=float)
        if cost_array.ndim != 2:
            raise ValueError("cost must be a 2D array")
        spec = GridSpec(
            width=cost_array.shape[1],
            height=cost_array.shape[0],
            resolution=resolution,
            origin=origin,
            frame_id=frame_id,
        )
        layer_names = sorted((layers or {}).keys())
        return CostGrid(
            spec=spec,
            cost=cost_array,
            passable_mask=np.asarray(passable_mask, dtype=bool),
            metadata={
                "adapter": "dev-platform-constraints",
                "layers": layer_names,
            },
        )
```

Create `src/path_planner/adapters/json_io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest, PlanResult

REQUEST_SCHEMA_VERSION = "path-planner-request/v1"


def load_plan_input(path: str | Path) -> tuple[CostGrid, PlanRequest]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {REQUEST_SCHEMA_VERSION}")
    grid_raw = raw["grid"]
    spec = GridSpec(
        width=int(grid_raw["width"]),
        height=int(grid_raw["height"]),
        resolution=float(grid_raw["resolution"]),
        origin=(float(grid_raw.get("origin", [0.0, 0.0])[0]), float(grid_raw.get("origin", [0.0, 0.0])[1])),
        frame_id=str(grid_raw.get("frame_id", "map")),
    )
    grid = CostGrid(
        spec=spec,
        cost=np.asarray(raw["cost"], dtype=float),
        passable_mask=np.asarray(raw["passable_mask"], dtype=bool),
        metadata={"adapter": "json", "source": str(path)},
    )
    request = PlanRequest(
        start=Cell(int(raw["start"][0]), int(raw["start"][1])),
        goal=Cell(int(raw["goal"][0]), int(raw["goal"][1])),
        max_iterations=int(raw.get("max_iterations", 100_000)),
    )
    return grid, request


def route_result_to_json_dict(result: PlanResult, spec: GridSpec) -> dict[str, Any]:
    return result.to_route_dict(spec)
```

Create `examples/demo_map.json`:

```json
{
  "schema_version": "path-planner-request/v1",
  "grid": {
    "width": 6,
    "height": 5,
    "resolution": 1.0,
    "origin": [0.0, 0.0],
    "frame_id": "demo_map"
  },
  "cost": [
    [1, 1, 1, 1, 1, 1],
    [1, 5, 5, 5, 1, 1],
    [1, 1, 1, 5, 1, 1],
    [1, 5, 1, 1, 1, 1],
    [1, 1, 1, 5, 5, 1]
  ],
  "passable_mask": [
    [true, true, true, true, true, true],
    [true, true, false, false, true, true],
    [true, true, true, false, true, true],
    [true, false, true, true, true, true],
    [true, true, true, true, true, true]
  ],
  "start": [0, 0],
  "goal": [5, 4]
}
```

- [ ] **Step 4: Run adapter tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_adapters_json.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit adapters**

Run:

```powershell
git add src/path_planner/adapters tests/test_adapters_json.py examples/demo_map.json
git commit -m "feat: add external planning adapters"
```

Expected: commit succeeds.

---

### Task 7: Diagnostics Renderer

**Files:**
- Create: `src/path_planner/diagnostics/__init__.py`
- Create: `src/path_planner/diagnostics/render.py`
- Create: `tests/test_cli_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics test**

Create `tests/test_cli_diagnostics.py`:

```python
import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.diagnostics import render_diagnostics
from path_planner.search import AStarPlanner


def test_render_diagnostics_writes_png_and_html(tmp_path):
    spec = GridSpec(width=3, height=3, resolution=1.0)
    grid = CostGrid(spec=spec, cost=np.ones((3, 3)), passable_mask=np.ones((3, 3), dtype=bool))
    result = AStarPlanner().plan(grid, PlanRequest(start=Cell(0, 0), goal=Cell(2, 2)))

    png_path = tmp_path / "diagnostics.png"
    html_path = tmp_path / "diagnostics.html"
    render_diagnostics(grid, result, png_path=png_path, html_path=html_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    html = html_path.read_text(encoding="utf-8")
    assert "Cost + Path" in html
    assert "trajectory_kind" in html
    assert "geometric_path" in html
```

- [ ] **Step 2: Run diagnostics test and verify it fails**

Run:

```powershell
python -m pytest tests/test_cli_diagnostics.py::test_render_diagnostics_writes_png_and_html -q
```

Expected: FAIL with import errors for `path_planner.diagnostics`.

- [ ] **Step 3: Implement PNG/HTML renderer**

Create `src/path_planner/diagnostics/__init__.py`:

```python
"""Diagnostic rendering."""

from .render import render_diagnostics

__all__ = ["render_diagnostics"]
```

Create `src/path_planner/diagnostics/render.py`:

```python
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from path_planner.core import CostGrid, PlanResult


def render_diagnostics(
    grid: CostGrid,
    result: PlanResult,
    *,
    png_path: str | Path,
    html_path: str | Path,
) -> None:
    png = Path(png_path)
    page = Path(html_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    page.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    _plot_cost_path(axes[0, 0], grid, result)
    _plot_mask(axes[0, 1], grid)
    _plot_expanded(axes[1, 0], grid, result)
    _plot_metrics(axes[1, 1], result)
    fig.savefig(png, dpi=140)
    plt.close(fig)

    route = result.to_route_dict(grid.spec)
    summary = html.escape(json.dumps(route, ensure_ascii=False, indent=2))
    page.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>Path Planner Diagnostics</title></head><body>",
                "<h1>Path Planner Diagnostics</h1>",
                "<p>trajectory_kind: <strong>geometric_path</strong></p>",
                f"<img src=\"{html.escape(png.name)}\" alt=\"diagnostics\" style=\"max-width:100%;height:auto\">",
                "<h2>Route JSON</h2>",
                f"<pre>{summary}</pre>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def _plot_cost_path(ax, grid: CostGrid, result: PlanResult) -> None:
    ax.imshow(grid.cost, cmap="viridis", origin="upper")
    if result.path_cells:
        xs = [cell.x for cell in result.path_cells]
        ys = [cell.y for cell in result.path_cells]
        ax.plot(xs, ys, color="white", linewidth=2)
        ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c=["lime", "red"], s=36)
    ax.set_title("Cost + Path")


def _plot_mask(ax, grid: CostGrid) -> None:
    ax.imshow(grid.passable_mask, cmap="gray", origin="upper")
    ax.set_title("Passable Mask")


def _plot_expanded(ax, grid: CostGrid, result: PlanResult) -> None:
    expanded = np.zeros(grid.spec.shape, dtype=float)
    for cell in result.diagnostics.expanded_cells:
        if grid.spec.in_bounds(cell):
            expanded[cell.y, cell.x] = 1.0
    ax.imshow(expanded, cmap="magma", origin="upper")
    ax.set_title("Expanded Nodes")


def _plot_metrics(ax, result: PlanResult) -> None:
    ax.axis("off")
    lines = [
        "Summary Metrics",
        f"success: {result.success}",
        f"failure_reason: {result.failure_reason.value if result.failure_reason else ''}",
        f"total_cost: {result.total_cost}",
        f"expanded_count: {result.expanded_count}",
        f"path_nodes: {len(result.path_cells)}",
        f"runtime_ms: {result.diagnostics.runtime_ms:.3f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")
```

- [ ] **Step 4: Run diagnostics test and verify it passes**

Run:

```powershell
python -m pytest tests/test_cli_diagnostics.py::test_render_diagnostics_writes_png_and_html -q
```

Expected: PASS.

- [ ] **Step 5: Commit diagnostics renderer**

Run:

```powershell
git add src/path_planner/diagnostics tests/test_cli_diagnostics.py
git commit -m "feat: add path diagnostics renderer"
```

Expected: commit succeeds.

---

### Task 8: CLI Demo And JSON Output

**Files:**
- Create: `src/path_planner/cli.py`
- Modify: `tests/test_cli_diagnostics.py`

- [ ] **Step 1: Add failing CLI smoke test**

Append to `tests/test_cli_diagnostics.py`:

```python
import json
import subprocess
import sys


def test_cli_demo_writes_json_png_and_html(tmp_path):
    output_json = tmp_path / "route.json"
    output_dir = tmp_path / "report"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "path_planner.cli",
            "--input",
            "examples/demo_map.json",
            "--output-json",
            str(output_json),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "path-planner-route/v1"
    assert payload["trajectory_kind"] == "geometric_path"
    assert payload["reachable"] is True
    assert (output_dir / "diagnostics.png").exists()
    assert (output_dir / "diagnostics.html").exists()
    assert "reachable" in completed.stdout
```

- [ ] **Step 2: Run CLI test and verify it fails**

Run:

```powershell
python -m pytest tests/test_cli_diagnostics.py::test_cli_demo_writes_json_png_and_html -q
```

Expected: FAIL with `No module named path_planner.cli`.

- [ ] **Step 3: Implement CLI**

Create `src/path_planner/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_planner.adapters import load_plan_input, route_result_to_json_dict
from path_planner.diagnostics import render_diagnostics
from path_planner.search import AStarPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 path planner demo")
    parser.add_argument("--input", required=True, help="Path to path-planner-request/v1 JSON")
    parser.add_argument("--output-json", required=True, help="Path to write route JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostics.png and diagnostics.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid, request = load_plan_input(args.input)
    result = AStarPlanner().plan(grid, request)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = route_result_to_json_dict(result, grid.spec)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = Path(args.output_dir)
    render_diagnostics(
        grid,
        result,
        png_path=output_dir / "diagnostics.png",
        html_path=output_dir / "diagnostics.html",
    )

    print(json.dumps({"reachable": result.success, "failure_reason": payload["failure_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI smoke test and verify it passes**

Run:

```powershell
python -m pytest tests/test_cli_diagnostics.py::test_cli_demo_writes_json_png_and_html -q
```

Expected: PASS.

- [ ] **Step 5: Manually run CLI demo once**

Run:

```powershell
python -m path_planner.cli --input examples/demo_map.json --output-json outputs/demo/route.json --output-dir outputs/demo
```

Expected: stdout includes `"reachable": true`; files exist at `outputs/demo/route.json`, `outputs/demo/diagnostics.png`, and `outputs/demo/diagnostics.html`.

- [ ] **Step 6: Commit CLI**

Run:

```powershell
git add src/path_planner/cli.py tests/test_cli_diagnostics.py
git commit -m "feat: add planning cli demo"
```

Expected: commit succeeds.

---

### Task 9: README And Final Verification

**Files:**
- Create: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add output ignores**

Modify `.gitignore` by adding:

```gitignore
outputs/
```

- [ ] **Step 2: Write README**

Create `README.md`:

```markdown
# path-planner

Phase 1 lunar path planner rebuilt from scratch. This package does not copy or depend on `a_gcs_ws-2.0.1`; that project is only a reference for problem framing.

## Scope

Phase 1 provides:

- core route planning models;
- semantic costmap to `cost/passable_mask` conversion;
- adapters for JSON and `dev-platform-constraints`-style arrays;
- 2D A* with 8-neighbor search and corner-cut protection;
- CLI JSON output;
- PNG/HTML diagnostics.

Phase 1 does not implement GCS, IRIS, Ackermann trajectory optimization, Drake integration, exploration target selection, observation updates, or an online planning service.

The planner returns a `geometric_path`, not a vehicle-executable trajectory.

## Development Environment

```powershell
conda activate D:\conda_envs\lunar-explorer
$env:PYTHONPATH='src'
```

The shared `lunar-explorer` Conda environment provides Python 3.12, NumPy,
Matplotlib, and pytest. This project does not need to be installed into that
environment for local development.

## Run Tests

```powershell
python -m pytest
```

Linux shell equivalent:

```bash
PYTHONPATH=src python -m pytest
```

## Run Demo

```powershell
python -m path_planner.cli --input examples/demo_map.json --output-json outputs/demo/route.json --output-dir outputs/demo
```

Expected outputs:

- `outputs/demo/route.json`
- `outputs/demo/diagnostics.png`
- `outputs/demo/diagnostics.html`

## External Interface Direction

`dev-platform-constraints` can provide `cost` and `passable_mask` through `DevPlatformAdapter`. `model-explorer` can consume the route JSON fields `reachable`, `geometric_path`, `path_cost`, `diagnostics`, and `failure_reason`.
```

- [ ] **Step 3: Run full test suite**

Run:

```powershell
python -m pytest
```

Expected: all tests PASS.

- [ ] **Step 4: Run CLI demo**

Run:

```powershell
python -m path_planner.cli --input examples/demo_map.json --output-json outputs/demo/route.json --output-dir outputs/demo
```

Expected: stdout includes `"reachable": true`; output JSON and diagnostics files are generated.

- [ ] **Step 5: Check git status**

Run:

```powershell
git status --short --ignored
```

Expected: tracked changes include `README.md` and `.gitignore`; `outputs/` appears ignored if present.

- [ ] **Step 6: Commit documentation and ignore rules**

Run:

```powershell
git add README.md .gitignore
git commit -m "docs: add phase 1 usage guide"
```

Expected: commit succeeds.

- [ ] **Step 7: Final verification commit**

If any generated outputs were accidentally staged, unstage them before committing with:

```powershell
git restore --staged outputs/demo/route.json outputs/demo/diagnostics.png outputs/demo/diagnostics.html
```

Expected: command succeeds only if those generated files were staged. Generated outputs remain untracked or ignored.

---

## Self-Review

Spec coverage:

- Core models are implemented in Tasks 1 and 2.
- Costmap synthesis is implemented in Task 3.
- A* with 8-neighbor and corner-cut protection is implemented in Tasks 4 and 5.
- Adapters for `dev-platform-constraints` and JSON/model-explorer-style route exchange are implemented in Task 6.
- PNG/HTML diagnostics are implemented in Task 7.
- CLI demo and route JSON output are implemented in Task 8.
- README, verification commands, and generated-output hygiene are implemented in Task 9.

The plan intentionally excludes GCS, IRIS, Ackermann, Drake, exploration target selection, observation updates, and service APIs, matching the Phase 1 spec.
