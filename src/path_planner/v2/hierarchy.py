from __future__ import annotations

from dataclasses import dataclass
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
    try:
        query = anchor.query(cell, max_slope_deg=max_slope_deg)
    except TimeoutError:
        raise
    except Exception as exc:
        raise HierarchyContractErrorV2("fine safety anchor query failed") from exc

    try:
        query_cell = query.cell
        passed = query.passed
        reason_code = query.reason_code
        terrain_slope = query.slope_deg
        confidence = query.confidence
        identity = query.snapshot_hash
        level = query.validation_level
    except (AttributeError, TypeError, ValueError) as exc:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch") from exc

    invalid = (
        type(query) is not SafetyQueryV2
        or type(query_cell) is not Cell
        or query_cell != cell
        or type(query_cell.x) is not int
        or type(query_cell.y) is not int
        or type(passed) is not bool
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

    if (
        isinstance(terrain_slope, bool)
        or not isinstance(terrain_slope, Real)
        or not isfinite(float(terrain_slope))
        or float(terrain_slope) < 0.0
        or isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    if reason_code == "terrain_safe" and float(terrain_slope) > max_slope_deg:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    if reason_code == "terrain_slope_exceeded" and float(terrain_slope) <= max_slope_deg:
        raise HierarchyContractErrorV2("fine safety anchor query contract mismatch")
    return query


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


@dataclass(frozen=True, slots=True)
class ConservativeHierarchyV2:
    geometry: FineGridGeometryV2
    snapshot_hash: str
    max_slope_deg: float
    hints: tuple[HierarchyCellHintV2, ...]

    def __post_init__(self) -> None:
        if type(self.geometry) is not FineGridGeometryV2:
            raise TypeError("geometry must be exact FineGridGeometryV2")
        if not _is_sha256(self.snapshot_hash):
            raise ValueError("snapshot_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "max_slope_deg", _normalized_threshold(self.max_slope_deg))
        if type(self.hints) is not tuple or any(
            type(hint) is not HierarchyCellHintV2 for hint in self.hints
        ):
            raise TypeError("hints must be an exact tuple of HierarchyCellHintV2")

        expected_keys: list[tuple[int, int, int]] = []
        for scale in HIERARCHY_SCALES_V2:
            coarse_width, coarse_height = _coarse_shape(self.geometry, scale)
            expected_keys.extend(
                (scale, y, x)
                for y in range(coarse_height)
                for x in range(coarse_width)
            )
        actual_keys = [(hint.scale, hint.cell.y, hint.cell.x) for hint in self.hints]
        if actual_keys != expected_keys:
            raise ValueError("hints must cover every coarse cell in stable order")
        for hint in self.hints:
            expected_cells = _covered_cells(self.geometry, hint.scale, hint.cell)
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
        coarse_width, coarse_height = _coarse_shape(self.geometry, normalized_scale)
        if normalized_cell.x >= coarse_width or normalized_cell.y >= coarse_height:
            raise ValueError("cell is outside hierarchy scale bounds")
        for hint in self.hints:
            if hint.scale == normalized_scale and hint.cell == normalized_cell:
                return hint
        raise HierarchyContractErrorV2("hierarchy hint coverage contract mismatch")

    def hint(self, scale: int, cell: Cell) -> HierarchyCellHintV2:
        return _defensive_hint_copy(self._validated_lookup(scale, cell))

    def iter_hints(self, scale: int) -> tuple[HierarchyCellHintV2, ...]:
        normalized_scale = _require_scale(scale)
        return tuple(
            _defensive_hint_copy(hint)
            for hint in self.hints
            if hint.scale == normalized_scale
        )

    def fine_cells(self, scale: int, cell: Cell) -> tuple[Cell, ...]:
        return self.hint(scale, cell).fine_cells
