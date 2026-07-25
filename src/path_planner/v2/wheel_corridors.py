from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from math import ceil, floor, hypot, isfinite, sqrt
from numbers import Real
from struct import Struct

from path_planner.core import Cell
from path_planner.v2.contracts import ResourceBudgetV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import FineGridGeometryV2, TerrainSnapshotV2, snapshot_hash
from path_planner.v2.wheel_sqp_contracts import (
    WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2,
    WheelCorridorV2,
    WheelSQPWorkLedgerV1,
    WheelSQPWorkLimitError,
    WheelTopologySignatureV1,
)


WHEEL_CORRIDOR_GRAPH_SAFETY_SCOPE_V1 = "current_observed_center_cell_passable/v1"
WHEEL_CORRIDOR_COMPONENT_SCHEMA_V1 = "wheel_blocked_component_quarter_cut/v1"
WHEEL_CORRIDOR_MEMORY_ACCOUNTING_V1 = "wheel_corridor_deterministic_records/v1"
WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1 = 30.0

_CARDINAL_LENGTH_M = 0.5
_DIAGONAL_LENGTH_M = 0.5 * sqrt(2.0)
_GRAPH_CELL_BYTES = 48
_SEARCH_RECORD_BYTES = 128
_PATH_CELL_BYTES = 16
_TERRAIN_RECORD = Struct("<IIB")
_TERRAIN_RECORD_PEAK_BYTES = 2 * _TERRAIN_RECORD.size
_U63_MAX = (1 << 63) - 1
_WHEEL_SQP_TERRAIN_GUIDE_AUTHORITY = object()

_TERRAIN_REASON_BY_RANK = {
    1: "terrain_unknown",
    2: "terrain_hard_obstacle",
    3: "terrain_not_traversable",
    4: "terrain_slope_exceeded",
}

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


def _finite_coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _invalid_reason_rank(snapshot: TerrainSnapshotV2, row: int, column: int) -> int:
    if not bool(snapshot.observed_mask[row, column]):
        return 1
    if bool(snapshot.hard_obstacle_mask[row, column]):
        return 2
    if not bool(snapshot.traversable_mask[row, column]):
        return 3
    if float(snapshot.slope_deg[row, column]) > WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1:
        return 4
    return 0


@dataclass(frozen=True, slots=True)
class WheelSQPInvalidTerrainRecordV1:
    row: int
    column: int
    reason_rank: int
    reason_code: str

    @property
    def cell(self) -> Cell:
        return Cell(self.column, self.row)


@dataclass(frozen=True, slots=True)
class WheelSQPTerrainClearanceV1:
    signed_distance_m: float
    gradient_x: float
    gradient_y: float
    reason_code: str
    cell: Cell | None
    tie_count: int


def _signed_invalid_rectangle_clearance(
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float, float, int]:
    dx = xmin - x if x < xmin else x - xmax if x > xmax else 0.0
    dy = ymin - y if y < ymin else y - ymax if y > ymax else 0.0
    if dx != 0.0 or dy != 0.0:
        distance = hypot(dx, dy)
        gradient_x = (
            -dx / distance if x < xmin else dx / distance if x > xmax else 0.0
        )
        gradient_y = (
            -dy / distance if y < ymin else dy / distance if y > ymax else 0.0
        )
        return distance, gradient_x, gradient_y, 0
    faces = (
        (x - xmin, -1.0, 0.0, 0),
        (xmax - x, 1.0, 0.0, 1),
        (y - ymin, 0.0, -1.0, 2),
        (ymax - y, 0.0, 1.0, 3),
    )
    depth, gradient_x, gradient_y, face_rank = min(faces, key=lambda item: (item[0], item[3]))
    signed = 0.0 if depth == 0.0 else -depth
    return signed, gradient_x, gradient_y, face_rank


def _signed_map_interior_clearance(
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float, float, int]:
    outside_x = xmin - x if x < xmin else x - xmax if x > xmax else 0.0
    outside_y = ymin - y if y < ymin else y - ymax if y > ymax else 0.0
    if outside_x != 0.0 or outside_y != 0.0:
        distance = hypot(outside_x, outside_y)
        gradient_x = (
            outside_x / distance
            if x < xmin
            else -outside_x / distance
            if x > xmax
            else 0.0
        )
        gradient_y = (
            outside_y / distance
            if y < ymin
            else -outside_y / distance
            if y > ymax
            else 0.0
        )
        return -distance, gradient_x, gradient_y, 0
    faces = (
        (x - xmin, 1.0, 0.0, 0),
        (xmax - x, -1.0, 0.0, 1),
        (y - ymin, 0.0, 1.0, 2),
        (ymax - y, 0.0, -1.0, 3),
    )
    return min(faces, key=lambda item: (item[0], item[3]))


@dataclass(frozen=True, slots=True, init=False)
class WheelSQPTerrainGuideV1:
    terrain_snapshot_hash: str
    snapshot: TerrainSnapshotV2
    invalid_count: int
    _records: bytes
    max_slope_deg: float = WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1

    def __init__(
        self,
        *,
        _authority: object,
        terrain_snapshot_hash: str,
        snapshot: TerrainSnapshotV2,
        invalid_count: int,
        _records: bytes,
        max_slope_deg: float = WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1,
    ) -> None:
        if _authority is not _WHEEL_SQP_TERRAIN_GUIDE_AUTHORITY:
            raise TypeError("WheelSQPTerrainGuideV1 is factory-only")
        object.__setattr__(self, "terrain_snapshot_hash", terrain_snapshot_hash)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "invalid_count", invalid_count)
        object.__setattr__(self, "_records", _records)
        object.__setattr__(self, "max_slope_deg", max_slope_deg)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            type(self.terrain_snapshot_hash) is not str
            or len(self.terrain_snapshot_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.terrain_snapshot_hash
            )
        ):
            raise ValueError("terrain_snapshot_hash must be a SHA-256 digest")
        if type(self.snapshot) is not TerrainSnapshotV2:
            raise TypeError("snapshot must be exact TerrainSnapshotV2")
        if type(self.invalid_count) is not int or self.invalid_count < 0:
            raise TypeError("invalid_count must be an exact nonnegative int")
        if type(self._records) is not bytes:
            raise TypeError("terrain records must be immutable bytes")
        if len(self._records) != self.invalid_count * _TERRAIN_RECORD.size:
            raise ValueError("terrain record byte length mismatch")
        _exact_slope_threshold(self.max_slope_deg)

    @property
    def geometry(self) -> FineGridGeometryV2:
        return self.snapshot.geometry

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TerrainSnapshotV2,
        max_slope_deg: float,
        *,
        ledger: WheelSQPWorkLedgerV1,
    ) -> WheelSQPTerrainGuideV1:
        if type(snapshot) is not TerrainSnapshotV2:
            raise TypeError("snapshot must be exact TerrainSnapshotV2")
        _exact_slope_threshold(max_slope_deg)
        if type(ledger) is not WheelSQPWorkLedgerV1:
            raise TypeError("ledger must be exact WheelSQPWorkLedgerV1")
        ledger.check_deadline()
        if snapshot.geometry.width > 0xFFFFFFFF or snapshot.geometry.height > 0xFFFFFFFF:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        count = 0
        for row in range(snapshot.geometry.height):
            ledger.check_deadline()
            for column in range(snapshot.geometry.width):
                if _invalid_reason_rank(snapshot, row, column) != 0:
                    count += 1
        if count > _U63_MAX // _TERRAIN_RECORD_PEAK_BYTES:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        ledger.check_deadline()
        ledger.charge_memory(count * _TERRAIN_RECORD_PEAK_BYTES)
        ledger.check_deadline()
        mutable = bytearray(count * _TERRAIN_RECORD.size)
        offset = 0
        for row in range(snapshot.geometry.height):
            ledger.check_deadline()
            for column in range(snapshot.geometry.width):
                rank = _invalid_reason_rank(snapshot, row, column)
                if rank == 0:
                    continue
                _TERRAIN_RECORD.pack_into(mutable, offset, row, column, rank)
                offset += _TERRAIN_RECORD.size
        ledger.check_deadline()
        records = bytes(mutable)
        ledger.check_deadline()
        identity = snapshot_hash(snapshot)
        ledger.check_deadline()
        result = cls(
            _authority=_WHEEL_SQP_TERRAIN_GUIDE_AUTHORITY,
            terrain_snapshot_hash=identity,
            snapshot=snapshot,
            invalid_count=count,
            _records=records,
            max_slope_deg=max_slope_deg,
        )
        ledger.check_deadline()
        return result

    def _iter_record_values(self):
        for offset in range(0, len(self._records), _TERRAIN_RECORD.size):
            yield _TERRAIN_RECORD.unpack_from(self._records, offset)

    @property
    def invalid_records(self) -> tuple[WheelSQPInvalidTerrainRecordV1, ...]:
        return tuple(
            WheelSQPInvalidTerrainRecordV1(
                row=row,
                column=column,
                reason_rank=rank,
                reason_code=_TERRAIN_REASON_BY_RANK[rank],
            )
            for row, column, rank in self._iter_record_values()
        )

    @property
    def invalid_cells(self) -> tuple[Cell, ...]:
        return tuple(Cell(column, row) for row, column, _ in self._iter_record_values())

    def nearest_signed_clearance(self, x_m: float, y_m: float) -> WheelSQPTerrainClearanceV1:
        x = _finite_coordinate(x_m, "x_m")
        y = _finite_coordinate(y_m, "y_m")
        origin_x, origin_y = self.geometry.origin
        resolution = self.geometry.resolution_m
        upper_x = origin_x + self.geometry.width * resolution
        upper_y = origin_y + self.geometry.height * resolution
        distance, gradient_x, gradient_y, face_rank = _signed_map_interior_clearance(
            x, y, origin_x, upper_x, origin_y, upper_y
        )
        best_key = (0, -1, -1, face_rank)
        best_reason = "terrain_out_of_bounds"
        best_cell: Cell | None = None
        tie_count = 0
        for row, column, rank in self._iter_record_values():
            xmin = origin_x + column * resolution
            xmax = xmin + resolution
            ymin = origin_y + row * resolution
            ymax = ymin + resolution
            candidate = _signed_invalid_rectangle_clearance(x, y, xmin, xmax, ymin, ymax)
            candidate_distance, candidate_gx, candidate_gy, candidate_face = candidate
            candidate_key = (rank, row, column, candidate_face)
            if candidate_distance < distance:
                distance = candidate_distance
                gradient_x = candidate_gx
                gradient_y = candidate_gy
                best_key = candidate_key
                best_reason = _TERRAIN_REASON_BY_RANK[rank]
                best_cell = Cell(column, row)
                tie_count = 0
            elif candidate_distance == distance:
                tie_count += 1
                if candidate_key < best_key:
                    gradient_x = candidate_gx
                    gradient_y = candidate_gy
                    best_key = candidate_key
                    best_reason = _TERRAIN_REASON_BY_RANK[rank]
                    best_cell = Cell(column, row)
        return WheelSQPTerrainClearanceV1(
            signed_distance_m=float(distance),
            gradient_x=float(gradient_x),
            gradient_y=float(gradient_y),
            reason_code=best_reason,
            cell=best_cell,
            tie_count=tie_count,
        )


def wheel_sqp_corridor_broadphase_cell_bound_v1(
    corridor: WheelCorridorV2,
    geometry: FineGridGeometryV2,
    footprint_radius_m: float,
    *,
    ledger: WheelSQPWorkLedgerV1,
    cap: int = 1_000_000,
) -> int:
    if type(corridor) is not WheelCorridorV2:
        raise TypeError("corridor must be exact WheelCorridorV2")
    if type(geometry) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    radius = _finite_coordinate(footprint_radius_m, "footprint_radius_m")
    if radius < 0.0:
        raise ValueError("footprint_radius_m must be nonnegative")
    if type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1")
    if type(cap) is not int or cap < 0:
        raise TypeError("cap must be an exact nonnegative int")
    total = 0
    leg_count = max(1, len(corridor.cells) - 1)
    for index in range(leg_count):
        ledger.check_deadline()
        left = corridor.cells[index]
        right = corridor.cells[index if len(corridor.cells) == 1 else index + 1]
        left_center = geometry.cell_center(left)
        right_center = geometry.cell_center(right)
        min_x = min(left_center.x, right_center.x) - radius
        max_x = max(left_center.x, right_center.x) + radius
        min_y = min(left_center.y, right_center.y) - radius
        max_y = max(left_center.y, right_center.y) + radius
        x0 = max(
            0,
            ceil((min_x - geometry.origin[0]) / geometry.resolution_m) - 1,
        )
        x1 = min(
            geometry.width - 1,
            floor((max_x - geometry.origin[0]) / geometry.resolution_m),
        )
        y0 = max(
            0,
            ceil((min_y - geometry.origin[1]) / geometry.resolution_m) - 1,
        )
        y1 = min(
            geometry.height - 1,
            floor((max_y - geometry.origin[1]) / geometry.resolution_m),
        )
        leg_cells = 0 if x1 < x0 or y1 < y0 else (x1 - x0 + 1) * (y1 - y0 + 1)
        if leg_cells > cap - total:
            raise WheelSQPWorkLimitError("wheel_sqp_resource_budget_exceeded")
        total += leg_cells
    ledger.check_deadline()
    return total


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
        ledger: WheelSQPWorkLedgerV1 | None = None,
    ) -> WheelCorridorGraphV1:
        if type(snapshot) is not TerrainSnapshotV2:
            raise TypeError("snapshot must be exact TerrainSnapshotV2")
        threshold = _exact_slope_threshold(max_slope_deg)
        if ledger is not None and type(ledger) is not WheelSQPWorkLedgerV1:
            raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
        geometry = snapshot.geometry
        if ledger is not None:
            ledger.charge_memory(geometry.width * geometry.height * _GRAPH_CELL_BYTES)
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
        ledger: WheelSQPWorkLedgerV1 | None = None,
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
    ledger: WheelSQPWorkLedgerV1 | None = None,
) -> tuple[WheelBlockedComponentV1, ...]:
    if type(graph) is not WheelCorridorGraphV1:
        raise TypeError("graph must be exact WheelCorridorGraphV1")
    if ledger is not None and type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
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
    ledger: WheelSQPWorkLedgerV1 | None = None,
) -> WheelTopologySignatureV1:
    if type(cells) is not tuple or not cells:
        raise TypeError("cells must be a nonempty exact tuple")
    for cell in cells:
        _exact_cell(cell, "cells")
    if type(components) is not tuple or any(
        type(component) is not WheelBlockedComponentV1 for component in components
    ):
        raise TypeError("components must contain exact WheelBlockedComponentV1 values")
    if ledger is not None and type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
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
    ledger: WheelSQPWorkLedgerV1,
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
    ledger = WheelSQPWorkLedgerV1(budget, deadline)
    ledger.check_deadline()
    ledger.charge_memory(graph.width * graph.height * _GRAPH_CELL_BYTES)
    path = _stable_astar(graph, start, goal, ledger)
    return path, ledger.expanded_states


def _path_cost(
    graph: WheelCorridorGraphV1,
    cells: tuple[Cell, ...],
    *,
    ledger: WheelSQPWorkLedgerV1 | None = None,
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
    ledger: WheelSQPWorkLedgerV1 | None = None,
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
    ledger: WheelSQPWorkLedgerV1,
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


def wheel_corridor_path_hash_v1(
    snapshot_identity: str,
    cells: tuple[Cell, ...],
    *,
    ledger: WheelSQPWorkLedgerV1 | None = None,
) -> str:
    if (
        type(snapshot_identity) is not str
        or len(snapshot_identity) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_identity)
    ):
        raise ValueError("snapshot_identity must be a lowercase SHA-256 digest")
    if type(cells) is not tuple or not cells:
        raise TypeError("cells must be a nonempty exact tuple")
    for cell in cells:
        _exact_cell(cell, "cells")
    if ledger is not None and type(ledger) is not WheelSQPWorkLedgerV1:
        raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
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
    ledger: WheelSQPWorkLedgerV1,
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
    *,
    ledger: WheelSQPWorkLedgerV1 | None = None,
) -> WheelCorridorGenerationResultV2:
    if type(snapshot) is not TerrainSnapshotV2:
        raise TypeError("snapshot must be exact TerrainSnapshotV2")
    start_cell = _exact_cell(start_cell, "start_cell")
    goal_cell = _exact_cell(goal_cell, "goal_cell")
    if type(resource_budget) is not ResourceBudgetV2:
        raise TypeError("resource_budget must be exact ResourceBudgetV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    if ledger is None:
        work_ledger = WheelSQPWorkLedgerV1(resource_budget, deadline)
    else:
        if type(ledger) is not WheelSQPWorkLedgerV1:
            raise TypeError("ledger must be exact WheelSQPWorkLedgerV1 or None")
        work_ledger = ledger
    graph: WheelCorridorGraphV1 | None = None
    try:
        if ledger is not None:
            if work_ledger.resource_budget != resource_budget:
                raise ValueError("ledger resource_budget must equal resource_budget")
            if work_ledger.deadline is not deadline:
                raise ValueError("ledger deadline must be the exact shared deadline object")
        work_ledger.check_deadline()
        graph = WheelCorridorGraphV1.from_snapshot(
            snapshot,
            max_slope_deg=WHEEL_CORRIDOR_MAX_SLOPE_DEG_V1,
            ledger=work_ledger,
        )
        components = build_blocked_components_v1(graph, ledger=work_ledger)
        raw_paths = yen_k_shortest_v1(
            graph,
            start_cell,
            goal_cell,
            candidate_cap=24,
            ledger=work_ledger,
        )
        if not raw_paths:
            return _generation_result(
                corridors=(),
                reason_code="wheel_sqp_no_2d_corridor",
                ledger=work_ledger,
                graph=graph,
            )
        first_per_signature: dict[WheelTopologySignatureV1, _CorridorCandidate] = {}
        for cells in raw_paths:
            work_ledger.check_deadline()
            signature = topology_signature_v1(cells, components, ledger=work_ledger)
            if signature in first_per_signature:
                continue
            path_hash = wheel_corridor_path_hash_v1(
                graph.snapshot_identity,
                cells,
                ledger=work_ledger,
            )
            work_ledger.charge_memory(_PATH_CELL_BYTES)
            first_per_signature[signature] = _CorridorCandidate(
                cells=cells,
                path_hash=path_hash,
                guide_cost=_path_cost(graph, cells, ledger=work_ledger),
                path_length_m=_path_length(graph, cells, ledger=work_ledger),
                topology_signature=signature,
            )
        work_ledger.charge_memory(len(first_per_signature) * _PATH_CELL_BYTES)
        ordered = sorted(
            first_per_signature.values(),
            key=lambda candidate: candidate.order_key,
        )[:3]
        for corridor in ordered:
            work_ledger.charge_route_states(len(corridor.cells))
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
        work_ledger.check_deadline()
        return _generation_result(
            corridors=corridors,
            reason_code="wheel_sqp_corridors_ready",
            ledger=work_ledger,
            graph=graph,
        )
    except WheelSQPWorkLimitError as abort:
        return _generation_result(
            corridors=(),
            reason_code=abort.reason_code,
            ledger=work_ledger,
            graph=graph,
        )
