from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from math import ceil, isfinite
from numbers import Real

from path_planner.core import Cell
from path_planner.v2.contracts import ValidationLevelV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    SafetyQueryV2,
    snapshot_hash,
)


HIERARCHY_SCALES_V2 = (1, 2, 4)
_HIERARCHY_BUILD_TOKEN = object()

_TERRAIN_REASON_CODES = frozenset(
    {
        "terrain_safe",
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
    }
)
_BLOCKED_REASON_CODES = frozenset(
    {
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
    }
)
_HINT_REASON_CODES = _TERRAIN_REASON_CODES | {"hierarchy_incomplete_block"}
_HEX_DIGITS = frozenset("0123456789abcdef")


class HierarchyContractErrorV2(RuntimeError):
    """Stable failure boundary for malformed optional hierarchy authority."""


class HierarchyHintStatusV2(str, Enum):
    SAFE_HINT = "safe_hint"
    UNKNOWN = "unknown"
    BLOCKED_HINT = "blocked_hint"


def _require_scale(scale: object) -> int:
    if type(scale) is not int:
        raise TypeError("scale must be an exact integer")
    if scale not in HIERARCHY_SCALES_V2:
        raise ValueError("scale must be one of 1, 2, or 4")
    return scale


def _require_cell(cell: object, *, name: str = "cell") -> Cell:
    if type(cell) is not Cell:
        raise TypeError(f"{name} must be exact Cell")
    if type(cell.x) is not int or type(cell.y) is not int:
        raise TypeError(f"{name} coordinates must be exact integers")
    if cell.x < 0 or cell.y < 0:
        raise ValueError(f"{name} coordinates must be nonnegative")
    return cell


def _normalized_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("max_slope_deg must be a finite real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError("max_slope_deg must be finite") from exc
    if not isfinite(normalized):
        raise ValueError("max_slope_deg must be finite")
    if normalized < 0.0:
        raise ValueError("max_slope_deg must be nonnegative")
    return normalized


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _check_deadline(deadline: PlanningDeadlineV2 | None) -> None:
    if deadline is not None and deadline.expired:
        raise TimeoutError("planning deadline expired during hierarchy construction")


def _anchor_snapshot_identity(anchor: FineSafetyAnchorV2) -> str:
    try:
        computed = snapshot_hash(anchor.snapshot)
        cached = anchor._snapshot_hash
    except Exception as exc:
        raise HierarchyContractErrorV2(
            "fine safety anchor snapshot identity contract mismatch"
        ) from exc
    if not _is_sha256(computed) or not _is_sha256(cached) or computed != cached:
        raise HierarchyContractErrorV2(
            "fine safety anchor snapshot identity contract mismatch"
        )
    return computed


def _coarse_shape(geometry: FineGridGeometryV2, scale: int) -> tuple[int, int]:
    return (ceil(geometry.width / scale), ceil(geometry.height / scale))


def _covered_cells(
    geometry: FineGridGeometryV2,
    scale: int,
    coarse_cell: Cell,
) -> tuple[Cell, ...]:
    start_x = coarse_cell.x * scale
    start_y = coarse_cell.y * scale
    return tuple(
        Cell(x, y)
        for y in range(start_y, min(start_y + scale, geometry.height))
        for x in range(start_x, min(start_x + scale, geometry.width))
    )


@dataclass(frozen=True, slots=True)
class HierarchyCellHintV2:
    scale: int
    cell: Cell
    fine_cells: tuple[Cell, ...]
    status: HierarchyHintStatusV2
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        scale = _require_scale(self.scale)
        coarse_cell = _require_cell(self.cell, name="cell")
        if type(self.fine_cells) is not tuple or not self.fine_cells:
            raise TypeError("fine_cells must be a nonempty exact tuple")
        normalized_cells = tuple(
            _require_cell(cell, name="fine cell") for cell in self.fine_cells
        )
        if normalized_cells != tuple(sorted(normalized_cells, key=lambda c: (c.y, c.x))):
            raise ValueError("fine_cells must be row-major")
        if len(normalized_cells) != len(set(normalized_cells)):
            raise ValueError("fine_cells must be unique")
        if len(normalized_cells) > scale * scale:
            raise ValueError("fine_cells exceed the coarse block")
        if any(
            fine.x // scale != coarse_cell.x or fine.y // scale != coarse_cell.y
            for fine in normalized_cells
        ):
            raise ValueError("fine_cells must map to the exact coarse cell")
        if type(self.status) is not HierarchyHintStatusV2:
            raise TypeError("status must be exact HierarchyHintStatusV2")
        if type(self.reason_codes) is not tuple or not self.reason_codes:
            raise TypeError("reason_codes must be a nonempty exact tuple")
        if any(
            type(reason) is not str or reason not in _HINT_REASON_CODES
            for reason in self.reason_codes
        ):
            raise ValueError("reason_codes must contain stable hierarchy reasons")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("reason_codes must be sorted and unique")
        if self.status is HierarchyHintStatusV2.SAFE_HINT:
            if len(normalized_cells) != scale * scale:
                raise ValueError("safe_hint requires a complete coarse block")
            if self.reason_codes != ("terrain_safe",):
                raise ValueError("safe_hint requires only terrain_safe")
        if self.status is HierarchyHintStatusV2.BLOCKED_HINT:
            if len(normalized_cells) != scale * scale:
                raise ValueError("blocked_hint requires a complete coarse block")
            if not set(self.reason_codes) <= _BLOCKED_REASON_CODES:
                raise ValueError("blocked_hint requires only known blocked reasons")


def _validated_query(
    anchor: FineSafetyAnchorV2,
    cell: Cell,
    max_slope_deg: float,
) -> SafetyQueryV2:
    expected_x = cell.x
    expected_y = cell.y
    try:
        query = anchor.query(cell, max_slope_deg=max_slope_deg)
    except TimeoutError:
        raise
    except Exception as exc:
        raise HierarchyContractErrorV2("fine safety anchor query failed") from exc

    if type(query) is not SafetyQueryV2:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    try:
        query_cell = query.cell
        passed = query.passed
        reason_code = query.reason_code
        terrain_slope = query.slope_deg
        confidence = query.confidence
        identity = query.snapshot_hash
        level = query.validation_level
    except Exception as exc:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch") from exc

    if type(query_cell) is not Cell:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    try:
        query_x = query_cell.x
        query_y = query_cell.y
        current_x = cell.x
        current_y = cell.y
    except Exception as exc:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch") from exc
    if (
        type(query_x) is not int
        or type(query_y) is not int
        or type(current_x) is not int
        or type(current_y) is not int
    ):
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    if (
        query_x != expected_x
        or query_y != expected_y
        or current_x != expected_x
        or current_y != expected_y
    ):
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")

    invalid = (
        type(passed) is not bool
        or type(reason_code) is not str
        or reason_code not in _TERRAIN_REASON_CODES
        or level is not ValidationLevelV2.L2
        or not _is_sha256(identity)
        or passed is not (reason_code == "terrain_safe")
        or reason_code == "terrain_out_of_bounds"
    )
    if invalid:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")

    if reason_code == "terrain_unknown":
        if terrain_slope is not None or confidence is not None:
            raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
        return query

    terrain_slope_value = _query_numeric(terrain_slope)
    confidence_value = _query_numeric(confidence)
    if terrain_slope_value < 0.0 or not 0.0 <= confidence_value <= 1.0:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    if reason_code == "terrain_safe" and terrain_slope_value > max_slope_deg:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    if reason_code == "terrain_slope_exceeded" and terrain_slope_value <= max_slope_deg:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    return query


def _query_numeric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    try:
        normalized = float(value)
    except Exception as exc:
        raise HierarchyContractErrorV2(
            "fine safety anchor query contract mismatch"
        ) from exc
    if not isfinite(normalized):
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    return normalized


def _status_for(
    fine_cells: tuple[Cell, ...],
    scale: int,
    reasons: tuple[str, ...],
) -> HierarchyHintStatusV2:
    if len(fine_cells) != scale * scale:
        return HierarchyHintStatusV2.UNKNOWN
    if all(reason == "terrain_safe" for reason in reasons):
        return HierarchyHintStatusV2.SAFE_HINT
    if all(reason in _BLOCKED_REASON_CODES for reason in reasons):
        return HierarchyHintStatusV2.BLOCKED_HINT
    return HierarchyHintStatusV2.UNKNOWN


def _defensive_hint_copy(hint: object) -> HierarchyCellHintV2:
    try:
        if type(hint) is not HierarchyCellHintV2:
            raise TypeError("hint type mismatch")
        copied = HierarchyCellHintV2(
            scale=hint.scale,
            cell=Cell(hint.cell.x, hint.cell.y),
            fine_cells=tuple(Cell(cell.x, cell.y) for cell in hint.fine_cells),
            status=hint.status,
            reason_codes=tuple(hint.reason_codes),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise HierarchyContractErrorV2("hierarchy hint contract mismatch") from exc
    return copied


def _defensive_geometry_copy(geometry: object) -> FineGridGeometryV2:
    try:
        if type(geometry) is not FineGridGeometryV2:
            raise TypeError("geometry type mismatch")
        return FineGridGeometryV2(
            width=geometry.width,
            height=geometry.height,
            origin=(geometry.origin[0], geometry.origin[1]),
            frame_id=geometry.frame_id,
            resolution_m=geometry.resolution_m,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise HierarchyContractErrorV2("hierarchy geometry contract mismatch") from exc


class ConservativeHierarchyV2(Mapping[str, object]):
    __slots__ = ("_geometry", "_snapshot_hash", "_max_slope_deg", "_hints")

    _SERIALIZATION_KEYS = (
        "geometry",
        "snapshot_hash",
        "max_slope_deg",
        "hints",
    )

    def __init__(
        self,
        geometry: FineGridGeometryV2,
        snapshot_hash: str,
        max_slope_deg: float,
        hints: tuple[HierarchyCellHintV2, ...],
        *,
        _build_token: object = None,
    ) -> None:
        if _build_token is not _HIERARCHY_BUILD_TOKEN:
            raise TypeError("ConservativeHierarchyV2 must be created with build()")
        object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(self, "_snapshot_hash", snapshot_hash)
        object.__setattr__(self, "_max_slope_deg", max_slope_deg)
        object.__setattr__(self, "_hints", hints)
        self.__post_init__()

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    @property
    def geometry(self) -> FineGridGeometryV2:
        return _defensive_geometry_copy(self._geometry)

    @property
    def snapshot_hash(self) -> str:
        return self._snapshot_hash

    @property
    def max_slope_deg(self) -> float:
        return self._max_slope_deg

    @property
    def hints(self) -> tuple[HierarchyCellHintV2, ...]:
        return tuple(_defensive_hint_copy(hint) for hint in self._hints)

    def __iter__(self) -> Iterator[str]:
        return iter(self._SERIALIZATION_KEYS)

    def __len__(self) -> int:
        return len(self._SERIALIZATION_KEYS)

    def __getitem__(self, key: str) -> object:
        if key == "geometry":
            return self.geometry
        if key == "snapshot_hash":
            return self.snapshot_hash
        if key == "max_slope_deg":
            return self.max_slope_deg
        if key == "hints":
            return self.hints
        raise KeyError(key)

    def __eq__(self, other: object) -> bool:
        if type(other) is not ConservativeHierarchyV2:
            return NotImplemented
        return (
            self._geometry == other._geometry
            and self._snapshot_hash == other._snapshot_hash
            and self._max_slope_deg == other._max_slope_deg
            and self._hints == other._hints
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._geometry,
                self._snapshot_hash,
                self._max_slope_deg,
                self._hints,
            )
        )

    def __repr__(self) -> str:
        return (
            "ConservativeHierarchyV2("
            f"geometry={self.geometry!r}, "
            f"snapshot_hash={self.snapshot_hash!r}, "
            f"max_slope_deg={self.max_slope_deg!r}, "
            f"hints={self.hints!r})"
        )

    def __post_init__(self) -> None:
        if type(self._geometry) is not FineGridGeometryV2:
            raise TypeError("geometry must be exact FineGridGeometryV2")
        object.__setattr__(
            self,
            "_geometry",
            _defensive_geometry_copy(self._geometry),
        )
        if not _is_sha256(self.snapshot_hash):
            raise ValueError("snapshot_hash must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "_max_slope_deg",
            _normalized_threshold(self.max_slope_deg),
        )
        if type(self._hints) is not tuple or any(
            type(hint) is not HierarchyCellHintV2 for hint in self._hints
        ):
            raise TypeError("hints must be an exact tuple of HierarchyCellHintV2")
        object.__setattr__(
            self,
            "_hints",
            tuple(_defensive_hint_copy(hint) for hint in self._hints),
        )

        expected_keys: list[tuple[int, int, int]] = []
        for scale in HIERARCHY_SCALES_V2:
            coarse_width, coarse_height = _coarse_shape(self._geometry, scale)
            expected_keys.extend(
                (scale, y, x)
                for y in range(coarse_height)
                for x in range(coarse_width)
            )
        actual_keys = [(hint.scale, hint.cell.y, hint.cell.x) for hint in self._hints]
        if actual_keys != expected_keys:
            raise ValueError("hints must cover every coarse cell in stable order")
        for hint in self._hints:
            expected_cells = _covered_cells(self._geometry, hint.scale, hint.cell)
            if hint.fine_cells != expected_cells:
                raise ValueError("hint fine_cells must match geometry exactly")

    @classmethod
    def build(
        cls,
        anchor: FineSafetyAnchorV2,
        *,
        max_slope_deg: float,
        deadline: PlanningDeadlineV2 | None = None,
    ) -> ConservativeHierarchyV2:
        if type(anchor) is not FineSafetyAnchorV2:
            raise TypeError("anchor must be exact FineSafetyAnchorV2")
        if deadline is not None and type(deadline) is not PlanningDeadlineV2:
            raise TypeError("deadline must be exact PlanningDeadlineV2")
        threshold = _normalized_threshold(max_slope_deg)
        geometry = anchor.snapshot.geometry
        if type(geometry) is not FineGridGeometryV2:
            raise HierarchyContractErrorV2("fine safety anchor geometry contract mismatch")
        _check_deadline(deadline)
        expected_snapshot_hash = _anchor_snapshot_identity(anchor)
        _check_deadline(deadline)

        queries: dict[Cell, SafetyQueryV2] = {}
        for y in range(geometry.height):
            for x in range(geometry.width):
                _check_deadline(deadline)
                cell = Cell(x, y)
                query = _validated_query(anchor, cell, threshold)
                _check_deadline(deadline)
                if query.snapshot_hash != expected_snapshot_hash:
                    raise HierarchyContractErrorV2(
                        "fine safety anchor snapshot identity contract mismatch"
                    )
                queries[cell] = query

        hints: list[HierarchyCellHintV2] = []
        for scale in HIERARCHY_SCALES_V2:
            coarse_width, coarse_height = _coarse_shape(geometry, scale)
            for coarse_y in range(coarse_height):
                for coarse_x in range(coarse_width):
                    _check_deadline(deadline)
                    coarse_cell = Cell(coarse_x, coarse_y)
                    fine_cells = _covered_cells(geometry, scale, coarse_cell)
                    reasons = tuple(queries[cell].reason_code for cell in fine_cells)
                    reason_codes = set(reasons)
                    if len(fine_cells) != scale * scale:
                        reason_codes.add("hierarchy_incomplete_block")
                    hints.append(
                        HierarchyCellHintV2(
                            scale=scale,
                            cell=coarse_cell,
                            fine_cells=fine_cells,
                            status=_status_for(fine_cells, scale, reasons),
                            reason_codes=tuple(sorted(reason_codes)),
                        )
                    )
        _check_deadline(deadline)
        built = cls(
            geometry=geometry,
            snapshot_hash=expected_snapshot_hash,
            max_slope_deg=threshold,
            hints=tuple(hints),
            _build_token=_HIERARCHY_BUILD_TOKEN,
        )
        if _anchor_snapshot_identity(anchor) != expected_snapshot_hash:
            raise HierarchyContractErrorV2(
                "fine safety anchor snapshot identity drift"
            )
        _check_deadline(deadline)
        return built

    def _validated_lookup(self, scale: object, cell: object) -> HierarchyCellHintV2:
        normalized_scale = _require_scale(scale)
        normalized_cell = _require_cell(cell)
        coarse_width, coarse_height = _coarse_shape(self._geometry, normalized_scale)
        if normalized_cell.x >= coarse_width or normalized_cell.y >= coarse_height:
            raise ValueError("cell is outside hierarchy scale bounds")
        for hint in self._hints:
            if hint.scale == normalized_scale and hint.cell == normalized_cell:
                return hint
        raise HierarchyContractErrorV2("hierarchy hint coverage contract mismatch")

    def hint(self, scale: int, cell: Cell) -> HierarchyCellHintV2:
        return _defensive_hint_copy(self._validated_lookup(scale, cell))

    def iter_hints(self, scale: int) -> tuple[HierarchyCellHintV2, ...]:
        normalized_scale = _require_scale(scale)
        return tuple(
            _defensive_hint_copy(hint)
            for hint in self._hints
            if hint.scale == normalized_scale
        )

    def fine_cells(self, scale: int, cell: Cell) -> tuple[Cell, ...]:
        return self.hint(scale, cell).fine_cells
