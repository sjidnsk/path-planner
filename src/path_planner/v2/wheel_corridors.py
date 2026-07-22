from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from math import hypot, sqrt

from path_planner.core import Cell
from path_planner.v2.contracts import ResourceBudgetV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import TerrainSnapshotV2, snapshot_hash
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2,
    WheelCorridorV2,
    WheelTopologySignatureV1,
)


WHEEL_CORRIDOR_GRAPH_SAFETY_SCOPE_V1 = "current_observed_center_cell_passable/v1"
WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1 = "wheel_blocked_component_quarter_cut/v1"
WHEEL_CORRIDOR_MEMORY_ACCOUNTING_V1 = "wheel_corridor_deterministic_records/v1"
WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1 = 30.0

_CARDINAL_LENGTH_M = 0.5
_DIAGONAL_LENGTH_M = 0.5 * sqrt(2.0)
_HARD_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024
_GRAPH_CELL_BYTES = 48
_SEARCH_RECORD_BYTES = 128
_PATH_CELL_BYTES = 16

# N, E, S, W, NE, SE, SW, NW.  The rank is part of the stable A* authority.
_NEIGHBOR_STEPS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
    (1, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
)
_STEP_RANK = {step: rank for rank, step in enumerate(_NEIGHBOR_STEPS)}
_COMPONENT_STEPS = _NEIGHBOR_STEPS[:4]
_BOUNDARY_NAMES = ("left", "top", "right", "bottom")


def _exact_cell(value: object, name: str) -> Cell:
    if type(value) is not Cell:
        raise TypeError(f"{name} must be exact Cell")
    if type(value.x) is not int or type(value.y) is not int:
        raise TypeError(f"{name} coordinates must be exact ints")
    return value


def _exact_slope_threshold(value: object) -> float:
    if type(value) is not float:
        raise TypeError("max_slope_deg must be exact built-in float")
    if value != WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1:
        raise ValueError("max_slope_deg must be exactly 30.0")
    return value


def _canonical_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class WheelCorridorGraphV1:
    snapshot: TerrainSnapshotV2
    max_slope_deg: float
    _passable_cells: frozenset[Cell]
    _clearance: tuple[int, ...]
    snapshot_identity: str
    safety_scope_id: str = WHEEL_CORRIDOR_GRAPH_SAFETY_SCOPE_V1

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TerrainSnapshotV2,
        max_slope_deg: float = WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1,
        *,
        ledger: _CorridorLedger | None = None,
    ) -> WheelCorridorGraphV1:
        if type(snapshot) is not TerrainSnapshotV2:
            raise TypeError("snapshot must be exact TerrainSnapshotV2")
        threshold = _exact_slope_threshold(max_slope_deg)
        if ledger is not None and type(ledger) is not _CorridorLedger:
            raise TypeError("ledger must be exact _CorridorLedger or None")
        geometry = snapshot.geometry
        if ledger is not None:
            ledger.check_deadline()
        passable_cells: set[Cell] = set()
        invalid_cells: list[Cell] = []
        for y in range(geometry.height):
            if ledger is not None:
                ledger.check_deadline()
            for x in range(geometry.width):
                cell = Cell(x, y)
                is_passable = (
                    bool(snapshot.observed_mask[y, x])
                    and not bool(snapshot.hard_obstacle_mask[y, x])
                    and bool(snapshot.traversable_mask[y, x])
                    and float(snapshot.slope_deg[y, x]) <= threshold
                )
                if is_passable:
                    passable_cells.add(cell)
                else:
                    invalid_cells.append(cell)
        passable = frozenset(passable_cells)
        invalid = tuple(invalid_cells)
        clearance = cls._clearance_pass(
            geometry.width,
            geometry.height,
            invalid,
            ledger=ledger,
        )
        if ledger is not None:
            ledger.check_deadline()
        identity = snapshot_hash(snapshot)
        if ledger is not None:
            ledger.check_deadline()
        return cls(
            snapshot=snapshot,
            max_slope_deg=threshold,
            _passable_cells=passable,
            _clearance=clearance,
            snapshot_identity=identity,
        )

    @staticmethod
    def _clearance_pass(
        width: int,
        height: int,
        invalid: tuple[Cell, ...],
        *,
        ledger: _CorridorLedger | None = None,
    ) -> tuple[int, ...]:
        if ledger is not None:
            ledger.check_deadline()
        if not invalid:
            clearance = (max(width, height),) * (width * height)
            if ledger is not None:
                ledger.check_deadline()
            return clearance
        sentinel = width + height + 1
        distances = [sentinel] * (width * height)
        queue: deque[Cell] = deque()
        for cell in invalid:
            distances[cell.y * width + cell.x] = 0
            queue.append(cell)
        while queue:
            if ledger is not None:
                ledger.check_deadline()
            current = queue.popleft()
            next_distance = distances[current.y * width + current.x] + 1
            if ledger is not None:
                ledger.check_deadline()
            for dx, dy in _NEIGHBOR_STEPS:
                x = current.x + dx
                y = current.y + dy
                if not (0 <= x < width and 0 <= y < height):
                    continue
                index = y * width + x
                if next_distance >= distances[index]:
                    continue
                distances[index] = next_distance
                queue.append(Cell(x, y))
        clearance = tuple(distances)
        if ledger is not None:
            ledger.check_deadline()
        return clearance

    @property
    def width(self) -> int:
        return self.snapshot.geometry.width

    @property
    def height(self) -> int:
        return self.snapshot.geometry.height

    def passable(self, cell: Cell) -> bool:
        return _exact_cell(cell, "cell") in self._passable_cells

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        cell = _exact_cell(cell, "cell")
        if not self.passable(cell):
            return ()
        neighbors: list[Cell] = []
        for rank, (dx, dy) in enumerate(_NEIGHBOR_STEPS):
            neighbor = Cell(cell.x + dx, cell.y + dy)
            if not self.passable(neighbor):
                continue
            if rank >= 4:
                if not self.passable(Cell(cell.x + dx, cell.y)):
                    continue
                if not self.passable(Cell(cell.x, cell.y + dy)):
                    continue
            neighbors.append(neighbor)
        return tuple(neighbors)

    def clearance_cells(self, cell: Cell) -> int:
        cell = _exact_cell(cell, "cell")
        if not self.snapshot.geometry.in_bounds(cell):
            raise ValueError("cell is outside corridor graph")
        return self._clearance[cell.y * self.width + cell.x]

    def edge_length_m(self, source: Cell, destination: Cell) -> float:
        source = _exact_cell(source, "source")
        destination = _exact_cell(destination, "destination")
        dx = abs(destination.x - source.x)
        dy = abs(destination.y - source.y)
        if (dx, dy) in ((1, 0), (0, 1)):
            return _CARDINAL_LENGTH_M
        if (dx, dy) == (1, 1):
            return _DIAGONAL_LENGTH_M
        raise ValueError("corridor edge must join adjacent cells")

    def edge_cost(self, source: Cell, destination: Cell) -> float:
        if destination not in self.neighbors(source):
            raise ValueError("corridor edge must be a valid graph neighbor")
        slope_deg = float(self.snapshot.slope_deg[destination.y, destination.x])
        clearance_cells = self.clearance_cells(destination)
        return self.edge_length_m(source, destination) * (
            1.0
            + 0.05 * (slope_deg / WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1) ** 2
            + 0.10 / (1.0 + clearance_cells)
        )


@dataclass(frozen=True, slots=True)
class WheelBlockedComponentV1:
    cells: tuple[Cell, ...]
    component_hash: str
    boundary_cell: Cell
    boundary_rank: int
    boundary_name: str
    cut_q4: tuple[tuple[int, int], tuple[int, int]]
    schema_id: str = WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1

    def __post_init__(self) -> None:
        if type(self.cells) is not tuple or not self.cells:
            raise TypeError("component cells must be a nonempty exact tuple")
        if any(type(cell) is not Cell for cell in self.cells):
            raise TypeError("component cells must contain exact Cell values")
        if self.cells != tuple(sorted(self.cells, key=lambda cell: (cell.y, cell.x))):
            raise ValueError("component cells must be sorted by (y, x)")
        if (
            type(self.component_hash) is not str
            or len(self.component_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.component_hash)
        ):
            raise ValueError("component_hash must be a lowercase SHA-256 digest")
        _exact_cell(self.boundary_cell, "boundary_cell")
        if type(self.boundary_rank) is not int or not 0 <= self.boundary_rank < 4:
            raise ValueError("boundary_rank must be an exact rank in [0, 4)")
        if self.boundary_name != _BOUNDARY_NAMES[self.boundary_rank]:
            raise ValueError("boundary_name must match boundary_rank")
        if (
            type(self.cut_q4) is not tuple
            or len(self.cut_q4) != 2
            or any(
                type(point) is not tuple
                or len(point) != 2
                or any(type(coordinate) is not int for coordinate in point)
                for point in self.cut_q4
            )
        ):
            raise TypeError("cut_q4 must contain two exact integer points")
        if self.cut_q4[0] == self.cut_q4[1]:
            raise ValueError("component cut must have positive length")
        if self.schema_id != WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1:
            raise ValueError("component schema_id mismatch")

    @property
    def cell_order_key(self) -> tuple[tuple[int, int], ...]:
        return tuple((cell.y, cell.x) for cell in self.cells)


@dataclass(frozen=True, slots=True)
class WheelCorridorGenerationResultV2:
    corridors: tuple[WheelCorridorV2, ...]
    reason_code: str
    expanded_states: int
    terrain_snapshot_hash: str | None

    def __post_init__(self) -> None:
        if type(self.corridors) is not tuple or any(
            type(corridor) is not WheelCorridorV2 for corridor in self.corridors
        ):
            raise TypeError("corridors must contain exact WheelCorridorV2 values")
        if type(self.reason_code) is not str or not self.reason_code:
            raise TypeError("reason_code must be exact nonempty str")
        if type(self.expanded_states) is not int or self.expanded_states < 0:
            raise ValueError("expanded_states must be an exact nonnegative int")
        if self.terrain_snapshot_hash is not None and (
            type(self.terrain_snapshot_hash) is not str
            or len(self.terrain_snapshot_hash) != 64
        ):
            raise ValueError("terrain_snapshot_hash must be None or a SHA-256 digest")
        if self.reason_code == "wheel_sqp_corridors_ready":
            if not 1 <= len(self.corridors) <= 3:
                raise ValueError("ready result requires one to three corridors")
            if self.terrain_snapshot_hash is None:
                raise ValueError("ready result requires terrain_snapshot_hash")
        elif self.corridors:
            raise ValueError("failed generation must not expose partial corridors")


@dataclass(frozen=True, slots=True)
class _CorridorCandidate:
    cells: tuple[Cell, ...]
    path_hash: str
    guide_cost: float
    path_length_m: float
    topology_signature: WheelTopologySignatureV1

    @property
    def order_key(self) -> tuple[float, float, WheelTopologySignatureV1, str]:
        return (
            self.guide_cost,
            self.path_length_m,
            self.topology_signature,
            self.path_hash,
        )


class _CorridorAbort(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(slots=True)
class _CorridorLedger:
    budget: ResourceBudgetV2
    deadline: PlanningDeadlineV2
    expanded_states: int = 0
    accounted_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.budget) is not ResourceBudgetV2:
            raise TypeError("budget must be exact ResourceBudgetV2")
        if type(self.deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")

    @property
    def effective_memory_limit_bytes(self) -> int:
        requested = self.budget.max_memory_bytes
        return _HARD_MEMORY_LIMIT_BYTES if requested == 0 else min(
            requested,
            _HARD_MEMORY_LIMIT_BYTES,
        )

    def check_deadline(self) -> None:
        if self.deadline.expired:
            raise _CorridorAbort("planning_deadline_expired")

    def charge_expansion(self) -> None:
        self.check_deadline()
        if self.expanded_states >= self.budget.max_expanded_states:
            raise _CorridorAbort("wheel_sqp_corridor_budget_exceeded")
        self.expanded_states += 1

    def charge_memory(self, amount: int) -> None:
        self.check_deadline()
        if type(amount) is not int or amount < 0:
            raise ValueError("memory charge must be an exact nonnegative int")
        attempted = self.accounted_bytes + amount
        if attempted > self.effective_memory_limit_bytes:
            raise _CorridorAbort("wheel_sqp_resource_budget_exceeded")
        self.accounted_bytes = attempted


def _component_boundary(
    cells: tuple[Cell, ...],
    width: int,
    height: int,
) -> tuple[Cell, int]:
    candidates: list[tuple[int, int, int, int, Cell]] = []
    for cell in cells:
        distances = (cell.x, cell.y, width - 1 - cell.x, height - 1 - cell.y)
        for rank, distance in enumerate(distances):
            candidates.append((distance, rank, cell.y, cell.x, cell))
    _, rank, _, _, cell = min(candidates)
    return cell, rank


def _quarter_cut(
    cell: Cell,
    boundary_rank: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if boundary_rank == 0:
        start = (4 * cell.x + 1, 4 * cell.y + 1)
        end = (0, start[1])
    elif boundary_rank == 1:
        start = (4 * cell.x + 3, 4 * cell.y + 1)
        end = (start[0], 0)
    elif boundary_rank == 2:
        start = (4 * cell.x + 3, 4 * cell.y + 3)
        end = (4 * width, start[1])
    else:
        start = (4 * cell.x + 1, 4 * cell.y + 3)
        end = (start[0], 4 * height)
    return start, end


def build_blocked_components_v1(
    graph: WheelCorridorGraphV1,
    *,
    ledger: _CorridorLedger | None = None,
) -> tuple[WheelBlockedComponentV1, ...]:
    if type(graph) is not WheelCorridorGraphV1:
        raise TypeError("graph must be exact WheelCorridorGraphV1")
    if ledger is not None and type(ledger) is not _CorridorLedger:
        raise TypeError("ledger must be exact _CorridorLedger or None")
    remaining: set[Cell] = set()
    for y in range(graph.height):
        if ledger is not None:
            ledger.check_deadline()
        for x in range(graph.width):
            cell = Cell(x, y)
            if graph.passable(cell):
                continue
            if ledger is not None:
                ledger.charge_memory(_PATH_CELL_BYTES)
            remaining.add(cell)
    components: list[WheelBlockedComponentV1] = []
    while remaining:
        if ledger is not None:
            ledger.check_deadline()
        seed = min(remaining, key=lambda cell: (cell.y, cell.x))
        remaining.remove(seed)
        queue: deque[Cell] = deque((seed,))
        cells: list[Cell] = []
        while queue:
            if ledger is not None:
                ledger.check_deadline()
            current = queue.popleft()
            cells.append(current)
            if ledger is not None:
                ledger.check_deadline()
            for dx, dy in _COMPONENT_STEPS:
                neighbor = Cell(current.x + dx, current.y + dy)
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
        if ledger is not None:
            ledger.check_deadline()
        ordered = tuple(sorted(cells, key=lambda cell: (cell.y, cell.x)))
        component_hash = _canonical_sha256(
            {
                "cells": [[cell.x, cell.y] for cell in ordered],
                "schema_id": WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1,
            }
        )
        boundary_cell, boundary_rank = _component_boundary(
            ordered,
            graph.width,
            graph.height,
        )
        if ledger is not None:
            ledger.check_deadline()
        components.append(
            WheelBlockedComponentV1(
                cells=ordered,
                component_hash=component_hash,
                boundary_cell=boundary_cell,
                boundary_rank=boundary_rank,
                boundary_name=_BOUNDARY_NAMES[boundary_rank],
                cut_q4=_quarter_cut(
                    boundary_cell,
                    boundary_rank,
                    graph.width,
                    graph.height,
                ),
            )
        )
    if ledger is not None:
        ledger.check_deadline()
    ordered_components = tuple(
        sorted(
            components,
            key=lambda component: (component.cell_order_key, component.component_hash),
        )
    )
    if ledger is not None:
        ledger.check_deadline()
    return ordered_components


def _orientation(
    first: tuple[int, int],
    second: tuple[int, int],
    point: tuple[int, int],
) -> int:
    return (
        (second[0] - first[0]) * (point[1] - first[1])
        - (second[1] - first[1]) * (point[0] - first[0])
    )


def _crossing_sign(
    path_start: tuple[int, int],
    path_end: tuple[int, int],
    cut_start: tuple[int, int],
    cut_end: tuple[int, int],
) -> int:
    side_start = _orientation(cut_start, cut_end, path_start)
    side_end = _orientation(cut_start, cut_end, path_end)
    if side_start == 0 or side_end == 0:
        raise ValueError("topology cut has a path vertex ambiguity")
    if (side_start > 0) is (side_end > 0):
        return 0
    cut_side_start = _orientation(path_start, path_end, cut_start)
    cut_side_end = _orientation(path_start, path_end, cut_end)
    if cut_side_start == 0 or cut_side_end == 0:
        raise ValueError("topology cut has an endpoint ambiguity")
    if (cut_side_start > 0) is (cut_side_end > 0):
        return 0
    # Positive is left-to-right relative to the oriented component cut.
    return 1 if side_start > 0 and side_end < 0 else -1


def topology_signature_v1(
    cells: tuple[Cell, ...],
    components: tuple[WheelBlockedComponentV1, ...],
    *,
    ledger: _CorridorLedger | None = None,
) -> WheelTopologySignatureV1:
    if type(cells) is not tuple or not cells:
        raise TypeError("cells must be a nonempty exact tuple")
    if any(type(cell) is not Cell for cell in cells):
        raise TypeError("cells must contain exact Cell values")
    if type(components) is not tuple or any(
        type(component) is not WheelBlockedComponentV1 for component in components
    ):
        raise TypeError("components must contain exact WheelBlockedComponentV1 values")
    if ledger is not None and type(ledger) is not _CorridorLedger:
        raise TypeError("ledger must be exact _CorridorLedger or None")
    entries: list[tuple[str, int]] = []
    for component in components:
        if ledger is not None:
            ledger.charge_memory(_PATH_CELL_BYTES)
        crossing_count = 0
        if ledger is not None:
            ledger.check_deadline()
        for source, destination in zip(cells, cells[1:]):
            path_start = (4 * source.x + 2, 4 * source.y + 2)
            path_end = (4 * destination.x + 2, 4 * destination.y + 2)
            crossing_count += _crossing_sign(
                path_start,
                path_end,
                component.cut_q4[0],
                component.cut_q4[1],
            )
        entries.append((component.component_hash, crossing_count))
    if ledger is not None:
        ledger.check_deadline()
    signature = WheelTopologySignatureV1(tuple(entries))
    if ledger is not None:
        ledger.check_deadline()
    return signature


def _reconstruct_path(came_from: dict[Cell, Cell], goal: Cell) -> tuple[Cell, ...]:
    reversed_path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        reversed_path.append(current)
    return tuple(reversed(reversed_path))


def _stable_astar(
    graph: WheelCorridorGraphV1,
    start: Cell,
    goal: Cell,
    ledger: _CorridorLedger,
    *,
    banned_nodes: frozenset[Cell] = frozenset(),
    banned_edges: frozenset[tuple[Cell, Cell]] = frozenset(),
) -> tuple[Cell, ...] | None:
    if start in banned_nodes or goal in banned_nodes:
        return None
    if not graph.passable(start) or not graph.passable(goal):
        return None
    insertion_id = 0
    start_predecessor = (-1, -1, -1)
    best_g: dict[Cell, float] = {start: 0.0}
    best_predecessor: dict[Cell, tuple[int, int, int]] = {
        start: start_predecessor
    }
    came_from: dict[Cell, Cell] = {}
    queue: list[
        tuple[float, float, int, int, int, int, Cell, tuple[int, int, int]]
    ] = []
    heuristic = _CARDINAL_LENGTH_M * hypot(goal.x - start.x, goal.y - start.y)
    ledger.charge_memory(_SEARCH_RECORD_BYTES)
    heapq.heappush(
        queue,
        (
            heuristic,
            0.0,
            start.y,
            start.x,
            -1,
            insertion_id,
            start,
            start_predecessor,
        ),
    )
    while queue:
        ledger.check_deadline()
        _, g_cost, _, _, _, _, current, predecessor_key = heapq.heappop(queue)
        if best_g.get(current) != g_cost:
            continue
        if best_predecessor.get(current) != predecessor_key:
            continue
        ledger.charge_expansion()
        if current == goal:
            return _reconstruct_path(came_from, goal)
        ledger.check_deadline()
        for neighbor in graph.neighbors(current):
            if neighbor in banned_nodes or (current, neighbor) in banned_edges:
                continue
            tentative_g = g_cost + graph.edge_cost(current, neighbor)
            rank = _STEP_RANK[(neighbor.x - current.x, neighbor.y - current.y)]
            candidate_predecessor = (current.y, current.x, rank)
            prior_g = best_g.get(neighbor)
            prior_predecessor = best_predecessor.get(neighbor)
            if prior_g is not None and tentative_g > prior_g:
                continue
            if (
                prior_g is not None
                and tentative_g == prior_g
                and prior_predecessor is not None
                and candidate_predecessor >= prior_predecessor
            ):
                continue
            best_g[neighbor] = tentative_g
            best_predecessor[neighbor] = candidate_predecessor
            came_from[neighbor] = current
            insertion_id += 1
            estimate = tentative_g + _CARDINAL_LENGTH_M * hypot(
                goal.x - neighbor.x,
                goal.y - neighbor.y,
            )
            ledger.charge_memory(_SEARCH_RECORD_BYTES)
            heapq.heappush(
                queue,
                (
                    estimate,
                    tentative_g,
                    neighbor.y,
                    neighbor.x,
                    rank,
                    insertion_id,
                    neighbor,
                    candidate_predecessor,
                ),
            )
    return None


def stable_astar_v1(
    graph: WheelCorridorGraphV1,
    start: Cell,
    goal: Cell,
    *,
    budget: ResourceBudgetV2,
    deadline: PlanningDeadlineV2,
) -> tuple[tuple[Cell, ...] | None, int]:
    if type(graph) is not WheelCorridorGraphV1:
        raise TypeError("graph must be exact WheelCorridorGraphV1")
    start = _exact_cell(start, "start")
    goal = _exact_cell(goal, "goal")
    ledger = _CorridorLedger(budget, deadline)
    ledger.check_deadline()
    ledger.charge_memory(graph.width * graph.height * _GRAPH_CELL_BYTES)
    path = _stable_astar(graph, start, goal, ledger)
    return path, ledger.expanded_states


def _path_cost(
    graph: WheelCorridorGraphV1,
    cells: tuple[Cell, ...],
    *,
    ledger: _CorridorLedger | None = None,
) -> float:
    cost = 0.0
    for source, destination in zip(cells, cells[1:]):
        if ledger is not None:
            ledger.check_deadline()
        cost += graph.edge_cost(source, destination)
    return cost


def _path_length(
    graph: WheelCorridorGraphV1,
    cells: tuple[Cell, ...],
    *,
    ledger: _CorridorLedger | None = None,
) -> float:
    length = 0.0
    for source, destination in zip(cells, cells[1:]):
        if ledger is not None:
            ledger.check_deadline()
        length += graph.edge_length_m(source, destination)
    return length


def _cell_path_key(cells: tuple[Cell, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((cell.y, cell.x) for cell in cells)


def yen_k_shortest_v1(
    graph: WheelCorridorGraphV1,
    start: Cell,
    goal: Cell,
    *,
    candidate_cap: int,
    ledger: _CorridorLedger,
) -> tuple[tuple[Cell, ...], ...]:
    if type(candidate_cap) is not int or not 1 <= candidate_cap <= 24:
        raise ValueError("candidate_cap must be an exact int in [1, 24]")
    first = _stable_astar(graph, start, goal, ledger)
    if first is None:
        return ()
    accepted: list[tuple[Cell, ...]] = [first]
    accepted_set = {first}
    queued_set: set[tuple[Cell, ...]] = set()
    candidates: list[
        tuple[
            float,
            float,
            tuple[tuple[int, int], ...],
            int,
            tuple[Cell, ...],
        ]
    ] = []
    insertion_id = 0
    while len(accepted) < candidate_cap:
        previous = accepted[-1]
        for spur_index in range(len(previous) - 1):
            ledger.check_deadline()
            root = previous[: spur_index + 1]
            banned_edges = frozenset(
                (path[spur_index], path[spur_index + 1])
                for path in accepted
                if len(path) > spur_index + 1 and path[: spur_index + 1] == root
            )
            spur = _stable_astar(
                graph,
                root[-1],
                goal,
                ledger,
                banned_nodes=frozenset(root[:-1]),
                banned_edges=banned_edges,
            )
            if spur is None:
                continue
            candidate = root[:-1] + spur
            if len(set(candidate)) != len(candidate):
                continue
            if candidate in accepted_set or candidate in queued_set:
                continue
            insertion_id += 1
            ledger.charge_memory(len(candidate) * _PATH_CELL_BYTES)
            heapq.heappush(
                candidates,
                (
                    _path_cost(graph, candidate, ledger=ledger),
                    _path_length(graph, candidate, ledger=ledger),
                    _cell_path_key(candidate),
                    insertion_id,
                    candidate,
                ),
            )
            queued_set.add(candidate)
        if not candidates:
            break
        _, _, _, _, selected = heapq.heappop(candidates)
        queued_set.remove(selected)
        accepted.append(selected)
        accepted_set.add(selected)
    return tuple(accepted)


def _path_hash(
    snapshot_identity: str,
    cells: tuple[Cell, ...],
    *,
    ledger: _CorridorLedger | None = None,
) -> str:
    if ledger is not None:
        ledger.charge_memory(len(cells) * _PATH_CELL_BYTES)
    digest = _canonical_sha256(
        {
            "cells": [[cell.x, cell.y] for cell in cells],
            "snapshot_hash": snapshot_identity,
            "source_id": WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2,
        }
    )
    if ledger is not None:
        ledger.check_deadline()
    return digest


def corridor_order_key_v2(
    corridor: WheelCorridorV2,
) -> tuple[float, float, WheelTopologySignatureV1, str]:
    if type(corridor) is not WheelCorridorV2:
        raise TypeError("corridor must be exact WheelCorridorV2")
    return (
        corridor.guide_cost,
        corridor.path_length_m,
        corridor.topology_signature,
        corridor.path_hash,
    )


def _generation_result(
    *,
    corridors: tuple[WheelCorridorV2, ...],
    reason_code: str,
    ledger: _CorridorLedger,
    graph: WheelCorridorGraphV1 | None,
) -> WheelCorridorGenerationResultV2:
    return WheelCorridorGenerationResultV2(
        corridors=corridors,
        reason_code=reason_code,
        expanded_states=ledger.expanded_states,
        terrain_snapshot_hash=None if graph is None else graph.snapshot_identity,
    )


def generate_wheel_corridors_v2(
    snapshot: TerrainSnapshotV2,
    start_cell: Cell,
    goal_cell: Cell,
    resource_budget: ResourceBudgetV2,
    deadline: PlanningDeadlineV2,
) -> WheelCorridorGenerationResultV2:
    if type(snapshot) is not TerrainSnapshotV2:
        raise TypeError("snapshot must be exact TerrainSnapshotV2")
    start_cell = _exact_cell(start_cell, "start_cell")
    goal_cell = _exact_cell(goal_cell, "goal_cell")
    ledger = _CorridorLedger(resource_budget, deadline)
    graph: WheelCorridorGraphV1 | None = None
    try:
        geometry = snapshot.geometry
        ledger.charge_memory(geometry.width * geometry.height * _GRAPH_CELL_BYTES)
        graph = WheelCorridorGraphV1.from_snapshot(
            snapshot,
            max_slope_deg=WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1,
            ledger=ledger,
        )
        components = build_blocked_components_v1(graph, ledger=ledger)
        raw_paths = yen_k_shortest_v1(
            graph,
            start_cell,
            goal_cell,
            candidate_cap=24,
            ledger=ledger,
        )
        if not raw_paths:
            return _generation_result(
                corridors=(),
                reason_code="wheel_sqp_no_2d_corridor",
                ledger=ledger,
                graph=graph,
            )
        first_per_signature: dict[WheelTopologySignatureV1, _CorridorCandidate] = {}
        for cells in raw_paths:
            ledger.check_deadline()
            signature = topology_signature_v1(cells, components, ledger=ledger)
            if signature in first_per_signature:
                continue
            path_hash = _path_hash(graph.snapshot_identity, cells, ledger=ledger)
            ledger.charge_memory(_PATH_CELL_BYTES)
            first_per_signature[signature] = _CorridorCandidate(
                cells=cells,
                path_hash=path_hash,
                guide_cost=_path_cost(graph, cells, ledger=ledger),
                path_length_m=_path_length(graph, cells, ledger=ledger),
                topology_signature=signature,
            )
        ledger.charge_memory(len(first_per_signature) * _PATH_CELL_BYTES)
        ordered = sorted(
            first_per_signature.values(),
            key=lambda candidate: candidate.order_key,
        )[:3]
        corridors = tuple(
            WheelCorridorV2(
                corridor_index=index,
                cells=corridor.cells,
                corridor_hash=corridor.path_hash,
                guide_cost=corridor.guide_cost,
                path_length_m=corridor.path_length_m,
                topology_signature=corridor.topology_signature,
            )
            for index, corridor in enumerate(ordered)
        )
        ledger.check_deadline()
        return _generation_result(
            corridors=corridors,
            reason_code="wheel_sqp_corridors_ready",
            ledger=ledger,
            graph=graph,
        )
    except _CorridorAbort as abort:
        return _generation_result(
            corridors=(),
            reason_code=abort.reason_code,
            ledger=ledger,
            graph=graph,
        )
