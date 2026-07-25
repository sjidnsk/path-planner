from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import floor, isfinite
from numbers import Real
from typing import Iterable, TypeAlias

import numpy as np

from path_planner.core import Cell, WorldPoint
from path_planner.v2.contracts import ValidationEvidenceV2, ValidationLevelV2


FINE_GRID_RESOLUTION_M_V2 = 0.5
SYNTHETIC_TERRAIN_SOURCE_KIND_V2 = "synthetic_terrain_obstacle_proxy/v1"
TERRAIN_SNAPSHOT_HASH_DOMAIN_V2 = b"path-planner-v2-terrain-snapshot/v1"
FINE_SAFETY_VALIDATOR_ID_V2 = "path-planner-v2-fine-safety-anchor/v1"

ScalarDetailV2: TypeAlias = str | int | float | bool | None

_FLOAT64_LE = np.dtype("<f8")
_BOOL = np.dtype(np.bool_)
_LAYER_DTYPES = (
    ("elevation_m", _FLOAT64_LE),
    ("slope_deg", _FLOAT64_LE),
    ("traversable_mask", _BOOL),
    ("hard_obstacle_mask", _BOOL),
    ("observed_mask", _BOOL),
    ("confidence", _FLOAT64_LE),
)
_SAFETY_REASON_CODES = frozenset(
    {
        "terrain_out_of_bounds",
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
        "terrain_safe",
    }
)


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_finite_real(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and must not be bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_cell(cell: object) -> Cell:
    if not isinstance(cell, Cell):
        raise TypeError("cell must be Cell")
    for coordinate in (cell.x, cell.y):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise TypeError("Cell coordinates must be integers and must not be bool")
    return cell


def _validate_scalar_detail(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("provenance detail floats must be finite")
        return
    raise TypeError("provenance detail values must be scalar values")


@dataclass(frozen=True, slots=True)
class FineGridGeometryV2:
    width: int
    height: int
    origin: tuple[float, float] = (0.0, 0.0)
    frame_id: str = "map"
    resolution_m: float = FINE_GRID_RESOLUTION_M_V2

    def __post_init__(self) -> None:
        _positive_integer(self.width, "width")
        _positive_integer(self.height, "height")
        resolution_m = _finite_real(self.resolution_m, "resolution_m")
        if resolution_m != FINE_GRID_RESOLUTION_M_V2:
            raise ValueError("resolution_m must be exactly 0.5")
        object.__setattr__(self, "resolution_m", FINE_GRID_RESOLUTION_M_V2)

        try:
            origin_length = len(self.origin)
        except TypeError as exc:
            raise ValueError("origin must contain exactly x and y") from exc
        if origin_length != 2:
            raise ValueError("origin must contain exactly x and y")
        origin_x = _finite_real(self.origin[0], "origin x")
        origin_y = _finite_real(self.origin[1], "origin y")
        object.__setattr__(self, "origin", (origin_x, origin_y))
        _nonempty_string(self.frame_id, "frame_id")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def in_bounds(self, cell: Cell) -> bool:
        cell = _require_cell(cell)
        return 0 <= cell.x < self.width and 0 <= cell.y < self.height

    def cell_center(self, cell: Cell) -> WorldPoint:
        cell = _require_cell(cell)
        if not self.in_bounds(cell):
            raise ValueError("cell is outside fine-grid bounds")
        return WorldPoint(
            self.origin[0] + (cell.x + 0.5) * self.resolution_m,
            self.origin[1] + (cell.y + 0.5) * self.resolution_m,
        )

    def world_to_cell(self, point: WorldPoint) -> Cell:
        if not isinstance(point, WorldPoint):
            raise TypeError("point must be WorldPoint")
        point_x = _finite_real(point.x, "world x")
        point_y = _finite_real(point.y, "world y")
        upper_x = self.origin[0] + self.width * self.resolution_m
        upper_y = self.origin[1] + self.height * self.resolution_m
        if not (
            self.origin[0] <= point_x < upper_x
            and self.origin[1] <= point_y < upper_y
        ):
            raise ValueError("world point is outside fine-grid bounds")
        return Cell(
            int(floor((point_x - self.origin[0]) / self.resolution_m)),
            int(floor((point_y - self.origin[1]) / self.resolution_m)),
        )


@dataclass(frozen=True, slots=True)
class TerrainProvenanceV2:
    source_kind: str
    source_id: str
    source_hash: str
    physical_obstacle_cells_written: bool
    details: tuple[tuple[str, ScalarDetailV2], ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string(self.source_kind, "source_kind")
        _nonempty_string(self.source_id, "source_id")
        _nonempty_string(self.source_hash, "source_hash")
        if not isinstance(self.physical_obstacle_cells_written, bool):
            raise TypeError("physical_obstacle_cells_written must be bool")
        if not isinstance(self.details, tuple):
            raise TypeError("details must be a tuple")

        keys: list[str] = []
        for detail in self.details:
            if not isinstance(detail, tuple) or len(detail) != 2:
                raise TypeError("details must contain immutable (key, scalar) tuples")
            key, value = detail
            _nonempty_string(key, "detail key")
            _validate_scalar_detail(value)
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("provenance detail keys must be unique")
        if keys != sorted(keys):
            raise ValueError("provenance detail keys must be sorted")
        if (
            self.source_kind == SYNTHETIC_TERRAIN_SOURCE_KIND_V2
            and self.physical_obstacle_cells_written is not False
        ):
            raise ValueError(
                "synthetic terrain provenance requires physical_obstacle_cells_written=False"
            )


def _canonical_layer_copy(
    name: str,
    value: object,
    dtype: np.dtype,
    shape: tuple[int, int],
) -> np.ndarray:
    source = np.asarray(value)
    if dtype == _BOOL and source.dtype != _BOOL:
        raise TypeError(f"{name} must have exact bool dtype")
    layer = np.array(source, dtype=dtype, order="C", copy=True)
    if layer.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if layer.shape != shape:
        raise ValueError(f"{name} shape {layer.shape} must match geometry shape {shape}")
    if dtype == _FLOAT64_LE and not np.all(np.isfinite(layer)):
        raise ValueError(f"{name} values must be finite")
    immutable_bytes = layer.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=dtype).reshape(shape)


@dataclass(frozen=True, slots=True)
class TerrainSnapshotV2:
    geometry: FineGridGeometryV2
    elevation_m: np.ndarray
    slope_deg: np.ndarray
    traversable_mask: np.ndarray
    hard_obstacle_mask: np.ndarray
    observed_mask: np.ndarray
    confidence: np.ndarray
    provenance: TerrainProvenanceV2

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FineGridGeometryV2):
            raise TypeError("geometry must be FineGridGeometryV2")
        if not isinstance(self.provenance, TerrainProvenanceV2):
            raise TypeError("provenance must be TerrainProvenanceV2")

        for name, dtype in _LAYER_DTYPES:
            canonical = _canonical_layer_copy(
                name,
                getattr(self, name),
                dtype,
                self.geometry.shape,
            )
            object.__setattr__(self, name, canonical)

        if np.any(self.slope_deg < 0.0):
            raise ValueError("slope_deg values must be nonnegative")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("confidence values must be in [0, 1]")
        if np.any(self.traversable_mask & self.hard_obstacle_mask):
            raise ValueError("traversable_mask and hard_obstacle_mask must not overlap")


def _zero_normalized_scalar(value: ScalarDetailV2) -> ScalarDetailV2:
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _canonical_snapshot_metadata(snapshot: TerrainSnapshotV2) -> bytes:
    geometry = snapshot.geometry
    provenance = snapshot.provenance
    payload = {
        "geometry": {
            "frame_id": geometry.frame_id,
            "height": geometry.height,
            "origin": [
                _zero_normalized_scalar(geometry.origin[0]),
                _zero_normalized_scalar(geometry.origin[1]),
            ],
            "resolution_m": geometry.resolution_m,
            "width": geometry.width,
        },
        "provenance": {
            "details": [
                [key, _zero_normalized_scalar(value)]
                for key, value in provenance.details
            ],
            "physical_obstacle_cells_written": provenance.physical_obstacle_cells_written,
            "source_hash": provenance.source_hash,
            "source_id": provenance.source_id,
            "source_kind": provenance.source_kind,
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _update_hash_chunk(hasher, chunk: bytes) -> None:
    hasher.update(len(chunk).to_bytes(8, byteorder="big", signed=False))
    hasher.update(chunk)


def snapshot_hash(snapshot: TerrainSnapshotV2) -> str:
    if not isinstance(snapshot, TerrainSnapshotV2):
        raise TypeError("snapshot must be TerrainSnapshotV2")
    hasher = sha256()
    _update_hash_chunk(hasher, TERRAIN_SNAPSHOT_HASH_DOMAIN_V2)
    _update_hash_chunk(hasher, _canonical_snapshot_metadata(snapshot))
    for name, dtype in _LAYER_DTYPES:
        canonical = np.array(getattr(snapshot, name), dtype=dtype, order="C", copy=True)
        if dtype == _FLOAT64_LE:
            canonical[canonical == 0.0] = 0.0
        _update_hash_chunk(hasher, name.encode("ascii"))
        _update_hash_chunk(hasher, dtype.str.encode("ascii"))
        _update_hash_chunk(
            hasher,
            json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
        )
        _update_hash_chunk(hasher, canonical.tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class SafetyQueryV2:
    cell: Cell
    passed: bool
    reason_code: str
    slope_deg: float | None
    confidence: float | None
    snapshot_hash: str
    validation_level: ValidationLevelV2 = ValidationLevelV2.L2

    def __post_init__(self) -> None:
        _require_cell(self.cell)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool")
        if self.reason_code not in _SAFETY_REASON_CODES:
            raise ValueError("reason_code must be a stable terrain safety reason")
        if self.slope_deg is not None:
            object.__setattr__(
                self,
                "slope_deg",
                _nonnegative_finite_real(self.slope_deg, "slope_deg"),
            )
        if self.confidence is not None:
            confidence = _finite_real(self.confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
            object.__setattr__(self, "confidence", confidence)
        if (
            not isinstance(self.snapshot_hash, str)
            or len(self.snapshot_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.snapshot_hash)
        ):
            raise ValueError("snapshot_hash must be a lowercase SHA-256 hex digest")
        if self.validation_level is not ValidationLevelV2.L2:
            raise TypeError("validation_level must be ValidationLevelV2.L2")
        if self.passed is not (self.reason_code == "terrain_safe"):
            raise ValueError("passed must be true exactly for terrain_safe")


@dataclass(frozen=True, slots=True)
class FineSafetyAnchorV2:
    snapshot: TerrainSnapshotV2
    _snapshot_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, TerrainSnapshotV2):
            raise TypeError("snapshot must be TerrainSnapshotV2")
        object.__setattr__(self, "_snapshot_hash", snapshot_hash(self.snapshot))

    def _result(
        self,
        cell: Cell,
        reason_code: str,
        *,
        slope_deg: float | None = None,
        confidence: float | None = None,
    ) -> SafetyQueryV2:
        return SafetyQueryV2(
            cell=cell,
            passed=reason_code == "terrain_safe",
            reason_code=reason_code,
            slope_deg=slope_deg,
            confidence=confidence,
            snapshot_hash=self._snapshot_hash,
            validation_level=ValidationLevelV2.L2,
        )

    def _query_validated_threshold(
        self,
        cell: Cell,
        max_slope_deg: float,
    ) -> SafetyQueryV2:
        cell = _require_cell(cell)
        if not self.snapshot.geometry.in_bounds(cell):
            return self._result(cell, "terrain_out_of_bounds")

        row, column = cell.y, cell.x
        if not bool(self.snapshot.observed_mask[row, column]):
            return self._result(cell, "terrain_unknown")

        slope_deg = float(self.snapshot.slope_deg[row, column])
        confidence = float(self.snapshot.confidence[row, column])
        if bool(self.snapshot.hard_obstacle_mask[row, column]):
            return self._result(
                cell,
                "terrain_hard_obstacle",
                slope_deg=slope_deg,
                confidence=confidence,
            )
        if not bool(self.snapshot.traversable_mask[row, column]):
            return self._result(
                cell,
                "terrain_not_traversable",
                slope_deg=slope_deg,
                confidence=confidence,
            )
        if slope_deg > max_slope_deg:
            return self._result(
                cell,
                "terrain_slope_exceeded",
                slope_deg=slope_deg,
                confidence=confidence,
            )
        return self._result(
            cell,
            "terrain_safe",
            slope_deg=slope_deg,
            confidence=confidence,
        )

    def query(self, cell: Cell, max_slope_deg: float = 30.0) -> SafetyQueryV2:
        normalized_threshold = _nonnegative_finite_real(
            max_slope_deg,
            "max_slope_deg",
        )
        return self._query_validated_threshold(cell, normalized_threshold)

    def validate_cells(
        self,
        cells: Iterable[Cell],
        max_slope_deg: float = 30.0,
    ) -> ValidationEvidenceV2:
        normalized_threshold = _nonnegative_finite_real(
            max_slope_deg,
            "max_slope_deg",
        )
        checks: list[str] = []
        for cell in cells:
            query = self._query_validated_threshold(cell, normalized_threshold)
            checks.append(query.reason_code)
            if not query.passed:
                return ValidationEvidenceV2(
                    validator_id=FINE_SAFETY_VALIDATOR_ID_V2,
                    level=ValidationLevelV2.L2,
                    passed=False,
                    checks=tuple(checks),
                )
        if not checks:
            return ValidationEvidenceV2(
                validator_id=FINE_SAFETY_VALIDATOR_ID_V2,
                level=ValidationLevelV2.L2,
                passed=False,
                checks=("terrain_empty_input",),
            )
        return ValidationEvidenceV2(
            validator_id=FINE_SAFETY_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=True,
            checks=tuple(checks),
        )
