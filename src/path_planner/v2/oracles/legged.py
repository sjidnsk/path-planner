from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, cos, floor, hypot, isfinite, nextafter, pi, sin
from numbers import Real

import numpy as np

from path_planner.core import Cell, WorldPoint
from path_planner.search.hybrid_astar import MAX_REPLAY_STEPS
from path_planner.v2.contracts import (
    PlatformKindV2,
    PoseStateV2,
    ValidationEvidenceV2,
    ValidationLevelV2,
)
from path_planner.v2.geometry import (
    convex_hull_xy,
    oriented_rectangle_cells,
    point_margin_to_convex_polygon,
)
from path_planner.v2.profiles import LeggedProfileV2, PlatformProfileV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    SafetyQueryV2,
    SYNTHETIC_TERRAIN_SOURCE_KIND_V2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
    snapshot_hash,
)


LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2 = (
    "path-planner-v2-legged-static-stability/v1"
)


class LegIdV2(str, Enum):
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"


LEGGED_FOOT_STORAGE_ORDER_V2 = (
    LegIdV2.FRONT_LEFT,
    LegIdV2.FRONT_RIGHT,
    LegIdV2.REAR_LEFT,
    LegIdV2.REAR_RIGHT,
)
LEGGED_CRAWL_SEQUENCE_V2 = (
    LegIdV2.FRONT_LEFT,
    LegIdV2.REAR_RIGHT,
    LegIdV2.FRONT_RIGHT,
    LegIdV2.REAR_LEFT,
)


def _canonical_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, RuntimeError):
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _canonical_point(value: object, name: str) -> WorldPoint:
    if type(value) is not WorldPoint:
        raise TypeError(f"{name} must be exact WorldPoint")
    return WorldPoint(
        _canonical_float(value.x, f"{name}.x"),
        _canonical_float(value.y, f"{name}.y"),
    )


def _canonical_state(value: object, name: str) -> PoseStateV2:
    if type(value) is not PoseStateV2:
        raise TypeError(f"{name} must be exact PoseStateV2")
    return PoseStateV2(
        _canonical_float(value.x_m, f"{name}.x_m"),
        _canonical_float(value.y_m, f"{name}.y_m"),
        _canonical_float(value.heading_rad, f"{name}.heading_rad"),
    )


@dataclass(frozen=True, slots=True)
class LeggedFootContactV2:
    leg_id: LegIdV2
    foothold: WorldPoint

    def __post_init__(self) -> None:
        if type(self.leg_id) is not LegIdV2:
            raise TypeError("leg_id must be exact LegIdV2")
        object.__setattr__(self, "foothold", _canonical_point(self.foothold, "foothold"))


@dataclass(frozen=True, slots=True)
class LeggedStepCandidateV2:
    start_body_state: PoseStateV2
    lift_body_state: PoseStateV2
    end_body_state: PoseStateV2
    foot_contacts: tuple[LeggedFootContactV2, ...]
    moving_leg: LegIdV2
    sequence_phase: int
    target_foothold: WorldPoint

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_body_state",
            _canonical_state(self.start_body_state, "start_body_state"),
        )
        object.__setattr__(
            self,
            "lift_body_state",
            _canonical_state(self.lift_body_state, "lift_body_state"),
        )
        object.__setattr__(
            self,
            "end_body_state",
            _canonical_state(self.end_body_state, "end_body_state"),
        )
        if type(self.foot_contacts) is not tuple:
            raise TypeError("foot_contacts must be an exact tuple")
        if len(self.foot_contacts) != 4:
            raise ValueError("foot_contacts must contain exactly four contacts")
        rebuilt: list[LeggedFootContactV2] = []
        for contact in self.foot_contacts:
            if type(contact) is not LeggedFootContactV2:
                raise TypeError("foot_contacts must contain exact LeggedFootContactV2")
            rebuilt.append(LeggedFootContactV2(contact.leg_id, contact.foothold))
        contacts = tuple(rebuilt)
        if tuple(contact.leg_id for contact in contacts) != LEGGED_FOOT_STORAGE_ORDER_V2:
            raise ValueError("foot_contacts must follow exact storage order")
        object.__setattr__(self, "foot_contacts", contacts)
        if type(self.moving_leg) is not LegIdV2:
            raise TypeError("moving_leg must be exact LegIdV2")
        if type(self.sequence_phase) is not int:
            raise TypeError("sequence_phase must be an exact int")
        if not 0 <= self.sequence_phase <= 3:
            raise ValueError("sequence_phase must be in [0, 3]")
        object.__setattr__(
            self,
            "target_foothold",
            _canonical_point(self.target_foothold, "target_foothold"),
        )


class _DeadlineContractError(RuntimeError):
    pass


class _DeadlineExpired(RuntimeError):
    pass


class _SnapshotHashError(RuntimeError):
    pass


class _QueryContractError(RuntimeError):
    pass


class _StructureError(RuntimeError):
    pass


def _propagate_critical(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        raise exc


@dataclass(frozen=True, slots=True)
class _DeadlineGuard:
    deadline: PlanningDeadlineV2
    started: float
    cutoff: float
    clock: object

    @classmethod
    def capture(cls, deadline: PlanningDeadlineV2) -> _DeadlineGuard:
        try:
            started = deadline.started_monotonic_s
            cutoff = deadline.deadline_monotonic_s
            clock = deadline._monotonic_clock
        except BaseException as exc:
            _propagate_critical(exc)
            raise _DeadlineContractError from exc
        if (
            type(started) is not float
            or type(cutoff) is not float
            or not isfinite(started)
            or not isfinite(cutoff)
            or cutoff < started
            or not callable(clock)
        ):
            raise _DeadlineContractError
        return cls(deadline, started, cutoff, clock)

    def _fields_match(self) -> bool:
        try:
            return (
                type(self.deadline.started_monotonic_s) is float
                and type(self.deadline.deadline_monotonic_s) is float
                and self.deadline.started_monotonic_s.hex() == self.started.hex()
                and self.deadline.deadline_monotonic_s.hex() == self.cutoff.hex()
                and self.deadline._monotonic_clock is self.clock
            )
        except BaseException as exc:
            _propagate_critical(exc)
            return False

    def check(self) -> None:
        if not self._fields_match():
            raise _DeadlineContractError
        try:
            now_value = self.clock()  # type: ignore[operator]
            if isinstance(now_value, bool) or not isinstance(now_value, Real):
                raise TypeError("clock must return a non-bool Real")
            now = float(now_value)
            if not isfinite(now):
                raise ValueError("clock result must be finite")
        except BaseException as exc:
            _propagate_critical(exc)
            raise _DeadlineContractError from exc
        if not self._fields_match():
            raise _DeadlineContractError
        if now >= self.cutoff:
            raise _DeadlineExpired


def _reaudit_profile(profile: LeggedProfileV2) -> LeggedProfileV2 | None:
    try:
        base = profile.profile
        if type(base) is not PlatformProfileV2:
            return None
        profile_id = base.profile_id
        platform_kind = base.platform_kind
        capability_revision = base.capability_revision
        simulation_proxy = base.simulation_proxy
        max_slope = base.max_traversable_slope_deg
        position_tolerance = base.goal_position_tolerance_m
        heading_tolerance = base.goal_heading_tolerance_rad
        schema_version = base.schema_version
        if (
            type(profile_id) is not str
            or type(platform_kind) is not PlatformKindV2
            or type(capability_revision) is not str
            or type(simulation_proxy) is not bool
            or type(max_slope) is not float
            or not isfinite(max_slope)
            or type(position_tolerance) is not float
            or not isfinite(position_tolerance)
            or type(heading_tolerance) is not float
            or not isfinite(heading_tolerance)
            or type(schema_version) is not str
        ):
            return None
        fixed_values = (
            (profile.body_length_m, 0.60),
            (profile.body_width_m, 0.40),
            (profile.nominal_foot_rectangle_length_m, 0.70),
            (profile.nominal_foot_rectangle_width_m, 0.50),
            (profile.max_step_length_m, 0.50),
            (profile.max_step_height_m, 0.25),
            (profile.max_foothold_slope_deg, 25.0),
            (profile.min_support_margin_m, 0.05),
            (profile.local_foothold_grid_spacing_m, 0.25),
        )
        if any(type(value) is not float or value != expected for value, expected in fixed_values):
            return None
        rebuilt_base = PlatformProfileV2(
            profile_id=profile_id,
            platform_kind=platform_kind,
            capability_revision=capability_revision,
            simulation_proxy=simulation_proxy,
            max_traversable_slope_deg=max_slope,
            goal_position_tolerance_m=position_tolerance,
            goal_heading_tolerance_rad=heading_tolerance,
            schema_version=schema_version,
        )
        rebuilt = LeggedProfileV2(
            profile=rebuilt_base,
            body_length_m=profile.body_length_m,
            body_width_m=profile.body_width_m,
            nominal_foot_rectangle_length_m=profile.nominal_foot_rectangle_length_m,
            nominal_foot_rectangle_width_m=profile.nominal_foot_rectangle_width_m,
            max_step_length_m=profile.max_step_length_m,
            max_step_height_m=profile.max_step_height_m,
            max_foothold_slope_deg=profile.max_foothold_slope_deg,
            min_support_margin_m=profile.min_support_margin_m,
            local_foothold_grid_spacing_m=profile.local_foothold_grid_spacing_m,
        )
    except BaseException as exc:
        _propagate_critical(exc)
        return None
    return rebuilt if type(rebuilt) is LeggedProfileV2 else None


def _reaudit_candidate(
    candidate: LeggedStepCandidateV2,
) -> LeggedStepCandidateV2 | None:
    try:
        def exact_canonical_float(value: object) -> bool:
            return (
                type(value) is float
                and isfinite(value)
                and (value != 0.0 or value.hex() == "0x0.0p+0")
            )

        def exact_state(state: object) -> bool:
            return (
                type(state) is PoseStateV2
                and exact_canonical_float(state.x_m)
                and exact_canonical_float(state.y_m)
                and exact_canonical_float(state.heading_rad)
            )

        def exact_point(point: object) -> bool:
            return (
                type(point) is WorldPoint
                and exact_canonical_float(point.x)
                and exact_canonical_float(point.y)
            )

        if (
            not exact_state(candidate.start_body_state)
            or not exact_state(candidate.lift_body_state)
            or not exact_state(candidate.end_body_state)
            or type(candidate.foot_contacts) is not tuple
            or len(candidate.foot_contacts) != 4
            or type(candidate.moving_leg) is not LegIdV2
            or type(candidate.sequence_phase) is not int
            or not 0 <= candidate.sequence_phase <= 3
            or not exact_point(candidate.target_foothold)
        ):
            return None
        for expected_leg, contact in zip(
            LEGGED_FOOT_STORAGE_ORDER_V2,
            candidate.foot_contacts,
            strict=True,
        ):
            if (
                type(contact) is not LeggedFootContactV2
                or contact.leg_id is not expected_leg
                or not exact_point(contact.foothold)
            ):
                return None
        rebuilt = LeggedStepCandidateV2(
            start_body_state=candidate.start_body_state,
            lift_body_state=candidate.lift_body_state,
            end_body_state=candidate.end_body_state,
            foot_contacts=candidate.foot_contacts,
            moving_leg=candidate.moving_leg,
            sequence_phase=candidate.sequence_phase,
            target_foothold=candidate.target_foothold,
        )
    except BaseException as exc:
        _propagate_critical(exc)
        return None
    return rebuilt


def _shortest_heading_delta(start: float, end: float) -> float:
    try:
        raw = end - start
        if not isfinite(raw):
            raise ValueError
        delta = (raw + pi) % (2.0 * pi) - pi
        if delta == -pi and raw > 0.0:
            delta = pi
    except (ArithmeticError, ValueError) as exc:
        raise _StructureError from exc
    if not isfinite(delta):
        raise _StructureError
    return delta


@dataclass(frozen=True, slots=True)
class _SweepSpec:
    start: PoseStateV2
    end: PoseStateV2
    dx: float
    dy: float
    heading_delta: float
    steps: int
    interval_expansion: float


def _sweep_spec(start: PoseStateV2, end: PoseStateV2) -> _SweepSpec:
    try:
        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        if not isfinite(dx) or not isfinite(dy):
            raise ValueError
        heading_delta = _shortest_heading_delta(start.heading_rad, end.heading_rad)
        rho = hypot(0.30, 0.20)
        bound = hypot(dx, dy) + rho * abs(heading_delta)
        if not isfinite(bound):
            raise ValueError
        ratio = bound / 0.0625
        if not isfinite(ratio) or ratio > MAX_REPLAY_STEPS:
            raise ValueError
        steps = max(1, int(ceil(ratio)))
        interval_expansion = nextafter(bound / steps, float("inf"))
        if not isfinite(interval_expansion) or interval_expansion < 0.0:
            raise ValueError
    except (ArithmeticError, ValueError, TypeError, OverflowError) as exc:
        raise _StructureError from exc
    return _SweepSpec(start, end, dx, dy, heading_delta, steps, interval_expansion)


def _candidate_sweeps(candidate: LeggedStepCandidateV2) -> tuple[_SweepSpec, _SweepSpec]:
    first = _sweep_spec(candidate.start_body_state, candidate.lift_body_state)
    second = _sweep_spec(candidate.lift_body_state, candidate.end_body_state)
    if first.steps + second.steps > MAX_REPLAY_STEPS:
        raise _StructureError
    return first, second


def _pose_at(spec: _SweepSpec, index: int) -> PoseStateV2:
    fraction = index / spec.steps
    try:
        x = spec.start.x_m + spec.dx * fraction
        y = spec.start.y_m + spec.dy * fraction
        heading = spec.start.heading_rad + spec.heading_delta * fraction
        if not all(isfinite(value) for value in (x, y, heading)):
            raise ValueError
    except (ArithmeticError, ValueError) as exc:
        raise _StructureError from exc
    return PoseStateV2(x, y, heading)


_LAYER_SPECS = (
    ("elevation_m", np.dtype("<f8")),
    ("slope_deg", np.dtype("<f8")),
    ("traversable_mask", np.dtype(np.bool_)),
    ("hard_obstacle_mask", np.dtype(np.bool_)),
    ("observed_mask", np.dtype(np.bool_)),
    ("confidence", np.dtype("<f8")),
)


def _immutable_layer(layer: object, dtype: np.dtype, shape: tuple[int, int]) -> bool:
    if (
        type(layer) is not np.ndarray
        or layer.dtype != dtype
        or layer.ndim != 2
        or layer.shape != shape
        or not layer.flags.c_contiguous
    ):
        return False
    current: object = layer
    while type(current) is np.ndarray:
        if current.flags.writeable:
            return False
        current = current.base
    return type(current) is bytes


def _metadata_token(value: object) -> object:
    if type(value) is float:
        return ("float", value.hex())
    if type(value) in (str, bool, int) or value is None:
        return (type(value).__name__, value)
    if type(value) is tuple:
        return ("tuple", tuple(_metadata_token(item) for item in value))
    return ("invalid", type(value).__qualname__)


def _exact_provenance_details(details: object) -> bool:
    if type(details) is not tuple:
        return False
    previous_key: str | None = None
    for detail in details:
        if type(detail) is not tuple or len(detail) != 2:
            return False
        key, value = detail
        if (
            type(key) is not str
            or not key.strip()
            or (previous_key is not None and key <= previous_key)
        ):
            return False
        if value is not None and type(value) not in (str, bool, int, float):
            return False
        if type(value) is float and not isfinite(value):
            return False
        previous_key = key
    return True


@dataclass(frozen=True, slots=True)
class _PinnedTerrain:
    anchor: FineSafetyAnchorV2
    snapshot: TerrainSnapshotV2
    authority_geometry: FineGridGeometryV2
    geometry: FineGridGeometryV2
    provenance: TerrainProvenanceV2
    layers: tuple[np.ndarray, ...]
    geometry_metadata: object
    provenance_metadata: object
    expected_hash: str

    @classmethod
    def capture(
        cls,
        anchor: FineSafetyAnchorV2,
        guard: _DeadlineGuard,
    ) -> _PinnedTerrain:
        guard.check()
        try:
            snapshot = anchor.snapshot
            if type(snapshot) is not TerrainSnapshotV2:
                raise ValueError
            geometry = snapshot.geometry
            provenance = snapshot.provenance
            if type(geometry) is not FineGridGeometryV2 or type(provenance) is not TerrainProvenanceV2:
                raise ValueError
            width = geometry.width
            height = geometry.height
            origin = geometry.origin
            frame_id = geometry.frame_id
            resolution_m = geometry.resolution_m
            source_kind = provenance.source_kind
            source_id = provenance.source_id
            source_hash = provenance.source_hash
            physical_written = provenance.physical_obstacle_cells_written
            details = provenance.details
            if (
                type(width) is not int
                or width <= 0
                or type(height) is not int
                or height <= 0
                or type(origin) is not tuple
                or len(origin) != 2
                or any(type(value) is not float or not isfinite(value) for value in origin)
                or type(frame_id) is not str
                or not frame_id.strip()
                or type(resolution_m) is not float
                or resolution_m != 0.5
                or type(source_kind) is not str
                or not source_kind.strip()
                or type(source_id) is not str
                or not source_id.strip()
                or type(source_hash) is not str
                or not source_hash.strip()
                or type(physical_written) is not bool
                or not _exact_provenance_details(details)
                or (
                    source_kind == SYNTHETIC_TERRAIN_SOURCE_KIND_V2
                    and physical_written is not False
                )
            ):
                raise ValueError
            canonical_geometry = FineGridGeometryV2(
                width=width,
                height=height,
                origin=origin,
                frame_id=frame_id,
                resolution_m=resolution_m,
            )
            TerrainProvenanceV2(
                source_kind=source_kind,
                source_id=source_id,
                source_hash=source_hash,
                physical_obstacle_cells_written=physical_written,
                details=details,
            )
            geometry_metadata = _metadata_token(
                (
                    width,
                    height,
                    origin,
                    frame_id,
                    resolution_m,
                )
            )
            provenance_metadata = _metadata_token(
                (
                    source_kind,
                    source_id,
                    source_hash,
                    physical_written,
                    details,
                )
            )
            layers = tuple(getattr(snapshot, name) for name, _dtype in _LAYER_SPECS)
            if any(
                not _immutable_layer(layer, dtype, canonical_geometry.shape)
                for layer, (_name, dtype) in zip(layers, _LAYER_SPECS, strict=True)
            ):
                raise ValueError
            expected_hash = anchor._snapshot_hash
            if (
                type(expected_hash) is not str
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ValueError
            digest = snapshot_hash(snapshot)
        except BaseException as exc:
            _propagate_critical(exc)
            raise _SnapshotHashError from exc
        guard.check()
        pinned = cls(
            anchor,
            snapshot,
            geometry,
            canonical_geometry,
            provenance,
            layers,
            geometry_metadata,
            provenance_metadata,
            expected_hash,
        )
        if digest != expected_hash or not pinned.identity_matches():
            raise _SnapshotHashError
        return pinned

    def identity_matches(self) -> bool:
        try:
            return (
                self.anchor.snapshot is self.snapshot
                and self.snapshot.geometry is self.authority_geometry
                and self.snapshot.provenance is self.provenance
                and all(
                    getattr(self.snapshot, name) is pinned_layer
                    for (name, _dtype), pinned_layer in zip(
                        _LAYER_SPECS,
                        self.layers,
                        strict=True,
                    )
                )
                and _metadata_token(
                    (
                        self.authority_geometry.width,
                        self.authority_geometry.height,
                        self.authority_geometry.origin,
                        self.authority_geometry.frame_id,
                        self.authority_geometry.resolution_m,
                    )
                )
                == self.geometry_metadata
                and _metadata_token(
                    (
                        self.provenance.source_kind,
                        self.provenance.source_id,
                        self.provenance.source_hash,
                        self.provenance.physical_obstacle_cells_written,
                        self.provenance.details,
                    )
                )
                == self.provenance_metadata
                and type(self.anchor._snapshot_hash) is str
                and self.anchor._snapshot_hash is self.expected_hash
            )
        except BaseException as exc:
            _propagate_critical(exc)
            return False

    def verify_hash(self, guard: _DeadlineGuard) -> None:
        guard.check()
        if not self.identity_matches():
            raise _SnapshotHashError
        try:
            digest = snapshot_hash(self.snapshot)
        except BaseException as exc:
            _propagate_critical(exc)
            raise _SnapshotHashError from exc
        guard.check()
        if digest != self.expected_hash or not self.identity_matches():
            raise _SnapshotHashError

    def seal_without_clock(self) -> None:
        if not self.identity_matches():
            raise _SnapshotHashError
        try:
            digest = snapshot_hash(self.snapshot)
        except BaseException as exc:
            _propagate_critical(exc)
            raise _SnapshotHashError from exc
        if digest != self.expected_hash or not self.identity_matches():
            raise _SnapshotHashError

    def layer(self, name: str) -> np.ndarray:
        index = next(index for index, (layer_name, _dtype) in enumerate(_LAYER_SPECS) if layer_name == name)
        return self.layers[index]


_TERRAIN_QUERY_REASONS = frozenset(
    {
        "terrain_out_of_bounds",
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_slope_exceeded",
        "terrain_safe",
    }
)


def _local_terrain_semantics(
    pinned: _PinnedTerrain,
    cell: Cell,
    threshold: float,
) -> tuple[str, float | None, float | None]:
    geometry = pinned.geometry
    if not (0 <= cell.x < geometry.width and 0 <= cell.y < geometry.height):
        return "terrain_out_of_bounds", None, None
    row, column = cell.y, cell.x
    observed = bool(pinned.layer("observed_mask")[row, column])
    if not observed:
        return "terrain_unknown", None, None
    hard = bool(pinned.layer("hard_obstacle_mask")[row, column])
    traversable = bool(pinned.layer("traversable_mask")[row, column])
    slope = float(pinned.layer("slope_deg")[row, column])
    confidence = float(pinned.layer("confidence")[row, column])
    if not isfinite(slope) or slope < 0.0 or not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise _QueryContractError
    if hard:
        reason = "terrain_hard_obstacle"
    elif not traversable:
        reason = "terrain_not_traversable"
    elif slope > threshold:
        reason = "terrain_slope_exceeded"
    else:
        reason = "terrain_safe"
    return reason, slope, confidence


def _query_cell(
    pinned: _PinnedTerrain,
    cell: Cell,
    threshold: float,
    guard: _DeadlineGuard,
    checked: list[int],
) -> SafetyQueryV2:
    guard.check()
    if not pinned.identity_matches():
        raise _QueryContractError
    try:
        checked[0] += 1
        query = pinned.anchor.query(cell, threshold)
    except BaseException as exc:
        _propagate_critical(exc)
        raise _QueryContractError from exc
    guard.check()
    if not pinned.identity_matches() or type(query) is not SafetyQueryV2:
        raise _QueryContractError
    try:
        query_cell = query.cell
        if (
            type(query_cell) is not Cell
            or type(query_cell.x) is not int
            or type(query_cell.y) is not int
            or query_cell != cell
            or type(query.passed) is not bool
            or type(query.reason_code) is not str
            or query.reason_code not in _TERRAIN_QUERY_REASONS
            or query.passed is not (query.reason_code == "terrain_safe")
            or query.validation_level is not ValidationLevelV2.L2
            or type(query.snapshot_hash) is not str
            or len(query.snapshot_hash) != 64
            or any(character not in "0123456789abcdef" for character in query.snapshot_hash)
        ):
            raise ValueError
        if query.snapshot_hash != pinned.expected_hash:
            raise _SnapshotHashError
        expected_reason, expected_slope, expected_confidence = _local_terrain_semantics(
            pinned,
            cell,
            threshold,
        )
        if (
            query.reason_code != expected_reason
            or query.slope_deg != expected_slope
            or query.confidence != expected_confidence
            or (query.slope_deg is not None and type(query.slope_deg) is not float)
            or (query.confidence is not None and type(query.confidence) is not float)
        ):
            raise ValueError
    except BaseException as exc:
        _propagate_critical(exc)
        if isinstance(exc, (_QueryContractError, _SnapshotHashError)):
            raise
        raise _QueryContractError from exc
    return query


def _point_cells(point: WorldPoint, geometry: FineGridGeometryV2) -> tuple[Cell, ...]:
    try:
        axes: list[tuple[int, ...]] = []
        for coordinate, origin in zip((point.x, point.y), geometry.origin, strict=True):
            offset = coordinate - origin
            q = offset / 0.5
            if not isfinite(offset) or not isfinite(q):
                raise ValueError
            nearest = round(q)
            boundary = nearest * 0.5
            if not isfinite(boundary):
                raise ValueError
            if abs(offset - boundary) <= 1.0e-12:
                axes.append((nearest - 1, nearest))
            else:
                axes.append((floor(q),))
        cells = {Cell(x, y) for y in axes[1] for x in axes[0]}
    except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
        raise _StructureError from exc
    return tuple(sorted(cells, key=lambda cell: (cell.y, cell.x)))


def _grid_aligned(candidate: LeggedStepCandidateV2) -> bool:
    source = next(
        contact.foothold
        for contact in candidate.foot_contacts
        if contact.leg_id is candidate.moving_leg
    )
    target = candidate.target_foothold
    try:
        theta = (candidate.start_body_state.heading_rad + pi) % (2.0 * pi) - pi
        if theta == -pi and candidate.start_body_state.heading_rad > 0.0:
            theta = pi
        dx = target.x - source.x
        dy = target.y - source.y
        dx_local = cos(theta) * dx + sin(theta) * dy
        dy_local = -sin(theta) * dx + cos(theta) * dy
        if not all(isfinite(value) for value in (theta, dx, dy, dx_local, dy_local)):
            raise ValueError
        return all(
            abs(component - round(component / 0.25) * 0.25) <= 1.0e-12
            for component in (dx_local, dy_local)
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
        raise _StructureError from exc


def _support_margin(
    candidate: LeggedStepCandidateV2,
    sweeps: tuple[_SweepSpec, _SweepSpec],
    minimum: float,
    guard: _DeadlineGuard,
) -> float | None:
    four_points = tuple(contact.foothold for contact in candidate.foot_contacts)
    three_points = tuple(
        contact.foothold
        for contact in candidate.foot_contacts
        if contact.leg_id is not candidate.moving_leg
    )
    try:
        four_hull = convex_hull_xy(four_points)
        three_hull = convex_hull_xy(three_points)
    except BaseException as exc:
        _propagate_critical(exc)
        raise _StructureError from exc
    if len(four_hull) < 3 or len(three_hull) < 3:
        return None
    best = float("inf")
    for spec, hull in ((sweeps[0], four_hull), (sweeps[1], three_hull)):
        for index in range(spec.steps + 1):
            guard.check()
            pose = _pose_at(spec, index)
            try:
                margin = point_margin_to_convex_polygon(
                    WorldPoint(pose.x_m, pose.y_m),
                    hull,
                )
            except BaseException as exc:
                _propagate_critical(exc)
                raise _StructureError from exc
            if not isfinite(margin):
                raise _StructureError
            best = min(best, margin)
            if margin < minimum:
                return None
    if not isfinite(best):
        raise _StructureError
    return float(best)


def _body_cells(
    sweeps: tuple[_SweepSpec, _SweepSpec],
    profile: LeggedProfileV2,
    geometry: FineGridGeometryV2,
    guard: _DeadlineGuard,
) -> tuple[Cell, ...]:
    cells: set[Cell] = set()
    try:
        for spec in sweeps:
            expanded_length = profile.body_length_m + 2.0 * spec.interval_expansion
            expanded_width = profile.body_width_m + 2.0 * spec.interval_expansion
            if not isfinite(expanded_length) or not isfinite(expanded_width):
                raise _StructureError
            for index in range(spec.steps):
                guard.check()
                pose = _pose_at(spec, index)
                sample_cells = oriented_rectangle_cells(
                    WorldPoint(pose.x_m, pose.y_m),
                    pose.heading_rad,
                    expanded_length,
                    expanded_width,
                    geometry,
                )
                for cell in sample_cells:
                    guard.check()
                    cells.add(cell)
                    if len(cells) > MAX_REPLAY_STEPS:
                        raise _StructureError
            guard.check()
            endpoint = _pose_at(spec, spec.steps)
            for cell in oriented_rectangle_cells(
                WorldPoint(endpoint.x_m, endpoint.y_m),
                endpoint.heading_rad,
                profile.body_length_m,
                profile.body_width_m,
                geometry,
            ):
                guard.check()
                cells.add(cell)
                if len(cells) > MAX_REPLAY_STEPS:
                    raise _StructureError
    except _StructureError:
        raise
    except BaseException as exc:
        _propagate_critical(exc)
        raise _StructureError from exc
    return tuple(sorted(cells, key=lambda cell: (cell.y, cell.x)))


_REASONS = frozenset(
    {
        "planning_deadline_expired",
        "planning_deadline_contract_mismatch",
        "terrain_snapshot_hash_mismatch",
        "terrain_query_contract_mismatch",
        "legged_profile_contract_mismatch",
        "legged_step_structure_mismatch",
        "legged_foothold_grid_misaligned",
        "legged_foothold_unknown",
        "legged_foothold_hard_obstacle",
        "legged_foothold_not_traversable",
        "legged_foothold_slope_exceeded",
        "legged_step_length_exceeded",
        "legged_step_height_exceeded",
        "legged_support_margin_insufficient",
        "legged_body_sweep_unknown",
        "legged_body_sweep_collision",
        "legged_foot_sequence_invalid",
        "legged_step_l2_valid",
    }
)


@dataclass(frozen=True, slots=True)
class LeggedValidationResultV2:
    evidence: ValidationEvidenceV2
    reason_code: str
    timed_out: bool
    failed_cell: Cell | None
    failed_leg: LegIdV2 | None
    checked_cell_count: int
    minimum_support_margin_m: float | None

    def __post_init__(self) -> None:
        if type(self.evidence) is not ValidationEvidenceV2:
            raise TypeError("evidence must be exact ValidationEvidenceV2")
        if (
            type(self.evidence.validator_id) is not str
            or self.evidence.validator_id != LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2
        ):
            raise ValueError("evidence validator_id must be the legged L2 validator")
        if (
            type(self.evidence.level) is not ValidationLevelV2
            or self.evidence.level is not ValidationLevelV2.L2
        ):
            raise ValueError("evidence must be L2")
        if type(self.evidence.passed) is not bool:
            raise TypeError("evidence passed must be exact bool")
        if (
            type(self.evidence.checks) is not tuple
            or len(self.evidence.checks) != 1
            or type(self.evidence.checks[0]) is not str
        ):
            raise TypeError("evidence checks must be one exact string in an exact tuple")
        if type(self.reason_code) is not str or self.reason_code not in _REASONS:
            raise ValueError("reason_code must be a stable legged L2 reason")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be exact bool")
        if type(self.checked_cell_count) is not int:
            raise TypeError("checked_cell_count must be an exact int")
        if self.checked_cell_count < 0:
            raise ValueError("checked_cell_count must be nonnegative")
        post_contact_reasons = {
            "legged_foothold_grid_misaligned",
            "legged_foothold_unknown",
            "legged_foothold_hard_obstacle",
            "legged_foothold_not_traversable",
            "legged_foothold_slope_exceeded",
            "legged_step_length_exceeded",
            "legged_step_height_exceeded",
            "legged_support_margin_insufficient",
            "legged_body_sweep_unknown",
            "legged_body_sweep_collision",
            "legged_foot_sequence_invalid",
            "legged_step_l2_valid",
        }
        if self.reason_code in post_contact_reasons and self.checked_cell_count <= 0:
            raise ValueError("post-contact result requires checked cells")
        if self.failed_cell is not None:
            if (
                type(self.failed_cell) is not Cell
                or type(self.failed_cell.x) is not int
                or type(self.failed_cell.y) is not int
            ):
                raise TypeError("failed_cell must be an exact Cell with exact int coordinates")
        if self.failed_leg is not None and type(self.failed_leg) is not LegIdV2:
            raise TypeError("failed_leg must be exact LegIdV2 or None")
        if self.minimum_support_margin_m is not None and (
            type(self.minimum_support_margin_m) is not float
            or not isfinite(self.minimum_support_margin_m)
        ):
            raise TypeError("minimum_support_margin_m must be an exact finite float")
        if (
            self.minimum_support_margin_m is not None
            and self.minimum_support_margin_m < 0.05
        ):
            raise ValueError("minimum_support_margin_m must meet the fixed 0.05m boundary")
        if self.evidence.checks != (self.reason_code,):
            raise ValueError("evidence checks must contain exactly reason_code")
        passed = self.reason_code == "legged_step_l2_valid"
        if self.evidence.passed is not passed:
            raise ValueError("evidence passed must agree with reason_code")
        if self.timed_out is not (self.reason_code == "planning_deadline_expired"):
            raise ValueError("timeout fields must agree")
        foothold_reasons = {
            "legged_foothold_unknown",
            "legged_foothold_hard_obstacle",
            "legged_foothold_not_traversable",
            "legged_foothold_slope_exceeded",
        }
        body_reasons = {"legged_body_sweep_unknown", "legged_body_sweep_collision"}
        leg_only_early = {
            "legged_foothold_grid_misaligned",
            "legged_step_length_exceeded",
            "legged_step_height_exceeded",
        }
        if self.timed_out and (
            self.failed_cell is not None
            or self.failed_leg is not None
            or self.minimum_support_margin_m is not None
        ):
            raise ValueError("timeout result must not carry failure metadata")
        if self.reason_code in foothold_reasons:
            if (
                self.failed_cell is None
                or self.failed_leg is None
                or self.checked_cell_count <= 0
                or self.minimum_support_margin_m is not None
            ):
                raise ValueError("foothold failure metadata is inconsistent")
        elif self.reason_code in body_reasons:
            if (
                self.failed_cell is None
                or self.failed_leg is not None
                or self.checked_cell_count <= 0
                or self.minimum_support_margin_m is None
            ):
                raise ValueError("body failure metadata is inconsistent")
        elif self.reason_code in leg_only_early:
            if (
                self.failed_cell is not None
                or self.failed_leg is None
                or self.minimum_support_margin_m is not None
            ):
                raise ValueError("legged step failure metadata is inconsistent")
        elif self.reason_code == "legged_foot_sequence_invalid":
            if (
                self.failed_cell is not None
                or self.failed_leg is None
                or self.minimum_support_margin_m is None
            ):
                raise ValueError("sequence failure metadata is inconsistent")
        elif passed:
            if (
                self.failed_cell is not None
                or self.failed_leg is not None
                or self.checked_cell_count <= 0
                or self.minimum_support_margin_m is None
                or self.timed_out
            ):
                raise ValueError("passing result metadata is inconsistent")
        elif (
            self.failed_cell is not None
            or self.failed_leg is not None
            or self.minimum_support_margin_m is not None
        ):
            raise ValueError("contract/profile/support failure must not carry metadata")


def _result(
    reason_code: str,
    *,
    passed: bool = False,
    timed_out: bool = False,
    failed_cell: Cell | None = None,
    failed_leg: LegIdV2 | None = None,
    checked_cell_count: int = 0,
    minimum_support_margin_m: float | None = None,
) -> LeggedValidationResultV2:
    return LeggedValidationResultV2(
        evidence=ValidationEvidenceV2(
            validator_id=LEGGED_STATIC_STABILITY_VALIDATOR_ID_V2,
            level=ValidationLevelV2.L2,
            passed=passed,
            checks=(reason_code,),
        ),
        reason_code=reason_code,
        timed_out=timed_out,
        failed_cell=failed_cell,
        failed_leg=failed_leg,
        checked_cell_count=checked_cell_count,
        minimum_support_margin_m=minimum_support_margin_m,
    )


def validate_legged_step_l2(
    candidate: LeggedStepCandidateV2,
    anchor: FineSafetyAnchorV2,
    legged_profile: LeggedProfileV2,
    deadline: PlanningDeadlineV2,
) -> LeggedValidationResultV2:
    if type(candidate) is not LeggedStepCandidateV2:
        raise TypeError("candidate must be exact LeggedStepCandidateV2")
    if type(anchor) is not FineSafetyAnchorV2:
        raise TypeError("anchor must be exact FineSafetyAnchorV2")
    if type(legged_profile) is not LeggedProfileV2:
        raise TypeError("legged_profile must be exact LeggedProfileV2")
    if type(deadline) is not PlanningDeadlineV2:
        raise TypeError("deadline must be exact PlanningDeadlineV2")
    checked = [0]
    guard: _DeadlineGuard | None = None
    pinned: _PinnedTerrain | None = None

    def finish(
        reason_code: str,
        *,
        passed: bool = False,
        timed_out: bool = False,
        failed_cell: Cell | None = None,
        failed_leg: LegIdV2 | None = None,
        minimum_support_margin_m: float | None = None,
        check_deadline: bool = True,
    ) -> LeggedValidationResultV2:
        if check_deadline and guard is not None:
            guard.check()
        if check_deadline and pinned is not None:
            try:
                pinned.seal_without_clock()
            except _SnapshotHashError:
                return _result(
                    "terrain_snapshot_hash_mismatch",
                    checked_cell_count=checked[0],
                )
        return _result(
            reason_code,
            passed=passed,
            timed_out=timed_out,
            failed_cell=failed_cell,
            failed_leg=failed_leg,
            checked_cell_count=checked[0],
            minimum_support_margin_m=minimum_support_margin_m,
        )

    try:
        guard = _DeadlineGuard.capture(deadline)
        guard.check()

        audited_profile = _reaudit_profile(legged_profile)
        if audited_profile is None:
            return finish("legged_profile_contract_mismatch")

        audited_candidate = _reaudit_candidate(candidate)
        if audited_candidate is None:
            return finish("legged_step_structure_mismatch")
        sweeps = _candidate_sweeps(audited_candidate)

        pinned = _PinnedTerrain.capture(anchor, guard)
        geometry = pinned.geometry
        leg_index = {
            leg: index for index, leg in enumerate(LEGGED_FOOT_STORAGE_ORDER_V2)
        }
        contact_cells: dict[LegIdV2, tuple[Cell, ...]] = {}
        pair_set: set[tuple[LegIdV2, Cell]] = set()
        for contact in audited_candidate.foot_contacts:
            guard.check()
            cells = _point_cells(contact.foothold, geometry)
            contact_cells[contact.leg_id] = cells
            pair_set.update((contact.leg_id, cell) for cell in cells)
        target_cells = _point_cells(audited_candidate.target_foothold, geometry)
        pair_set.update((audited_candidate.moving_leg, cell) for cell in target_cells)

        foothold_failures: list[tuple[int, int, int, int, str, LegIdV2, Cell]] = []
        foothold_reason = {
            "terrain_out_of_bounds": (0, "legged_foothold_unknown"),
            "terrain_unknown": (0, "legged_foothold_unknown"),
            "terrain_hard_obstacle": (1, "legged_foothold_hard_obstacle"),
            "terrain_not_traversable": (2, "legged_foothold_not_traversable"),
            "terrain_slope_exceeded": (3, "legged_foothold_slope_exceeded"),
        }
        ordered_pairs = sorted(
            pair_set,
            key=lambda item: (leg_index[item[0]], item[1].y, item[1].x),
        )
        for leg, cell in ordered_pairs:
            query = _query_cell(
                pinned,
                cell,
                audited_profile.max_foothold_slope_deg,
                guard,
                checked,
            )
            mapped = foothold_reason.get(query.reason_code)
            if mapped is not None:
                rank, reason = mapped
                foothold_failures.append(
                    (rank, leg_index[leg], cell.y, cell.x, reason, leg, cell)
                )
        pinned.verify_hash(guard)

        if not _grid_aligned(audited_candidate):
            return finish(
                "legged_foothold_grid_misaligned",
                failed_leg=audited_candidate.moving_leg,
            )
        if foothold_failures:
            _rank, _leg_rank, _y, _x, reason, leg, cell = min(foothold_failures)
            return finish(reason, failed_cell=cell, failed_leg=leg)

        moving_source = next(
            contact.foothold
            for contact in audited_candidate.foot_contacts
            if contact.leg_id is audited_candidate.moving_leg
        )
        dx = audited_candidate.target_foothold.x - moving_source.x
        dy = audited_candidate.target_foothold.y - moving_source.y
        horizontal_distance = hypot(dx, dy)
        if not isfinite(horizontal_distance):
            raise _StructureError
        if horizontal_distance > audited_profile.max_step_length_m:
            return finish(
                "legged_step_length_exceeded",
                failed_leg=audited_candidate.moving_leg,
            )

        source_elevations: list[float] = []
        target_elevations: list[float] = []
        elevation_layer = pinned.layer("elevation_m")
        for collection, cells in (
            (source_elevations, contact_cells[audited_candidate.moving_leg]),
            (target_elevations, target_cells),
        ):
            for cell in cells:
                guard.check()
                if not pinned.identity_matches() or not geometry.in_bounds(cell):
                    raise _SnapshotHashError
                value = float(elevation_layer[cell.y, cell.x])
                if not isfinite(value):
                    raise _SnapshotHashError
                collection.append(value)
        step_height = max(
            abs(source - target)
            for source in source_elevations
            for target in target_elevations
        )
        if not isfinite(step_height):
            raise _StructureError
        if step_height > audited_profile.max_step_height_m:
            return finish(
                "legged_step_height_exceeded",
                failed_leg=audited_candidate.moving_leg,
            )

        support_margin = _support_margin(
            audited_candidate,
            sweeps,
            audited_profile.min_support_margin_m,
            guard,
        )
        if support_margin is None:
            return finish("legged_support_margin_insufficient")

        cells = _body_cells(sweeps, audited_profile, geometry, guard)
        body_failures: list[tuple[int, int, int, str, Cell]] = []
        for cell in cells:
            query = _query_cell(
                pinned,
                cell,
                audited_profile.profile.max_traversable_slope_deg,
                guard,
                checked,
            )
            if query.reason_code in {"terrain_out_of_bounds", "terrain_unknown"}:
                body_failures.append((0, cell.y, cell.x, "legged_body_sweep_unknown", cell))
            elif query.reason_code in {
                "terrain_hard_obstacle",
                "terrain_not_traversable",
                "terrain_slope_exceeded",
            }:
                body_failures.append((1, cell.y, cell.x, "legged_body_sweep_collision", cell))
        pinned.verify_hash(guard)
        if body_failures:
            _rank, _y, _x, reason, cell = min(body_failures)
            return finish(
                reason,
                failed_cell=cell,
                minimum_support_margin_m=support_margin,
            )

        if LEGGED_CRAWL_SEQUENCE_V2[audited_candidate.sequence_phase] is not audited_candidate.moving_leg:
            return finish(
                "legged_foot_sequence_invalid",
                failed_leg=audited_candidate.moving_leg,
                minimum_support_margin_m=support_margin,
            )
        pinned.verify_hash(guard)
        return finish(
            "legged_step_l2_valid",
            passed=True,
            minimum_support_margin_m=support_margin,
        )
    except _DeadlineContractError:
        return finish("planning_deadline_contract_mismatch", check_deadline=False)
    except _DeadlineExpired:
        return finish(
            "planning_deadline_expired",
            timed_out=True,
            check_deadline=False,
        )
    except _SnapshotHashError:
        try:
            if guard is not None:
                guard.check()
        except _DeadlineContractError:
            return finish("planning_deadline_contract_mismatch", check_deadline=False)
        except _DeadlineExpired:
            return finish(
                "planning_deadline_expired", timed_out=True, check_deadline=False
            )
        return finish("terrain_snapshot_hash_mismatch", check_deadline=False)
    except _QueryContractError:
        try:
            if guard is not None:
                guard.check()
        except _DeadlineContractError:
            return finish("planning_deadline_contract_mismatch", check_deadline=False)
        except _DeadlineExpired:
            return finish(
                "planning_deadline_expired", timed_out=True, check_deadline=False
            )
        return finish("terrain_query_contract_mismatch", check_deadline=False)
    except _StructureError:
        try:
            if guard is not None:
                guard.check()
        except _DeadlineContractError:
            return finish("planning_deadline_contract_mismatch", check_deadline=False)
        except _DeadlineExpired:
            return finish(
                "planning_deadline_expired", timed_out=True, check_deadline=False
            )
        return finish("legged_step_structure_mismatch", check_deadline=False)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        try:
            if guard is not None:
                guard.check()
        except _DeadlineContractError:
            return finish("planning_deadline_contract_mismatch", check_deadline=False)
        except _DeadlineExpired:
            return finish(
                "planning_deadline_expired", timed_out=True, check_deadline=False
            )
        return finish("legged_step_structure_mismatch", check_deadline=False)
