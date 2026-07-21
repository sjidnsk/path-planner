from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import cos, hypot, inf, isclose, radians, remainder, sin, tau
from typing import Iterator

import numpy as np

from path_planner.core import Cell, WorldPoint
from path_planner.v2.contracts import (
    ObservationProjectionV2,
    PlatformKindV2,
    PoseStateV2,
    TypedRouteV2,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)


OBSERVATION_PROJECTION_SOURCE_V2 = "path-planner-v2-observed-only-fov-los/v1"
OBSERVATION_ROUTE_SAMPLE_STEP_M_V2 = 1.0
OBSERVATION_RANGE_M_V2 = 20.0
OBSERVATION_FOV_DEG_V2 = 90.0
OBSERVATION_RAY_STEP_DEG_V2 = 1.0
OBSERVED_TERRAIN_CANONICALIZATION_V2 = "path-planner-v2-observed-only-mask/v1"
OBSERVED_TERRAIN_SOURCE_KIND_V2 = "path-planner-v2-observed-terrain/v1"
OBSERVED_TERRAIN_SOURCE_ID_V2 = "path-planner-v2-observed-only-snapshot/v1"

_OBSERVED_HASH_DOMAIN_V2 = b"path-planner-v2-observed-terrain-content/v1\0"
_FLOAT64_LE = np.dtype("<f8")
_BOOL = np.dtype("|b1")


def _update_hash_chunk(hasher, chunk: bytes) -> None:
    hasher.update(len(chunk).to_bytes(8, byteorder="big", signed=False))
    hasher.update(chunk)


def _observed_content_hash(
    snapshot: TerrainSnapshotV2,
    *,
    elevation_m: np.ndarray,
    slope_deg: np.ndarray,
    traversable_mask: np.ndarray,
    hard_obstacle_mask: np.ndarray,
    observed_mask: np.ndarray,
    confidence: np.ndarray,
) -> str:
    geometry = snapshot.geometry
    metadata = {
        "geometry": {
            "frame_id": geometry.frame_id,
            "height": geometry.height,
            "origin": [
                0.0 if geometry.origin[0] == 0.0 else geometry.origin[0],
                0.0 if geometry.origin[1] == 0.0 else geometry.origin[1],
            ],
            "resolution_m": geometry.resolution_m,
            "width": geometry.width,
        },
    }
    hasher = sha256()
    _update_hash_chunk(hasher, _OBSERVED_HASH_DOMAIN_V2)
    _update_hash_chunk(hasher, canonical_json_bytes(metadata))
    for name, value, dtype in (
        ("elevation_m", elevation_m, _FLOAT64_LE),
        ("slope_deg", slope_deg, _FLOAT64_LE),
        ("traversable_mask", traversable_mask, _BOOL),
        ("hard_obstacle_mask", hard_obstacle_mask, _BOOL),
        ("observed_mask", observed_mask, _BOOL),
        ("confidence", confidence, _FLOAT64_LE),
    ):
        canonical = np.array(value, dtype=dtype, order="C", copy=True)
        if dtype == _FLOAT64_LE:
            canonical[canonical == 0.0] = 0.0
        _update_hash_chunk(hasher, name.encode("ascii"))
        _update_hash_chunk(hasher, dtype.str.encode("ascii"))
        _update_hash_chunk(hasher, canonical_json_bytes(list(canonical.shape)))
        _update_hash_chunk(hasher, canonical.tobytes(order="C"))
    return hasher.hexdigest()


def _canonical_observed_snapshot(snapshot: TerrainSnapshotV2) -> TerrainSnapshotV2:
    if type(snapshot) is not TerrainSnapshotV2:
        raise TypeError("terrain_snapshot must be exact TerrainSnapshotV2")
    observed = np.array(snapshot.observed_mask, dtype=_BOOL, order="C", copy=True)
    elevation = np.where(observed, snapshot.elevation_m, 0.0).astype(
        _FLOAT64_LE,
        copy=False,
    )
    slope = np.where(observed, snapshot.slope_deg, 0.0).astype(
        _FLOAT64_LE,
        copy=False,
    )
    confidence = np.where(observed, snapshot.confidence, 0.0).astype(
        _FLOAT64_LE,
        copy=False,
    )
    traversable = np.logical_and(observed, snapshot.traversable_mask)
    obstacle = np.logical_and(observed, snapshot.hard_obstacle_mask)
    source_hash = _observed_content_hash(
        snapshot,
        elevation_m=elevation,
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=obstacle,
        observed_mask=observed,
        confidence=confidence,
    )
    provenance = TerrainProvenanceV2(
        source_kind=OBSERVED_TERRAIN_SOURCE_KIND_V2,
        source_id=OBSERVED_TERRAIN_SOURCE_ID_V2,
        source_hash=source_hash,
        physical_obstacle_cells_written=False,
        details=(("canonicalization", OBSERVED_TERRAIN_CANONICALIZATION_V2),),
    )
    return TerrainSnapshotV2(
        geometry=snapshot.geometry,
        elevation_m=elevation,
        slope_deg=slope,
        traversable_mask=traversable,
        hard_obstacle_mask=obstacle,
        observed_mask=observed,
        confidence=confidence,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class ObservedTerrainInputV2:
    request_id: str
    start_state: PoseStateV2
    terrain_snapshot: TerrainSnapshotV2

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id.strip():
            raise ValueError("request_id must be an exact nonempty string")
        if type(self.start_state) is not PoseStateV2:
            raise TypeError("start_state must be exact PoseStateV2")
        object.__setattr__(
            self,
            "terrain_snapshot",
            _canonical_observed_snapshot(self.terrain_snapshot),
        )


def _append_without_pose_seam(
    output: list[PoseStateV2],
    values: tuple[PoseStateV2, ...],
) -> None:
    if not values or any(type(value) is not PoseStateV2 for value in values):
        raise ValueError("route observation samples must contain exact PoseStateV2 values")
    start_index = 1 if output and output[-1] == values[0] else 0
    output.extend(values[start_index:])


def _route_waypoints(route: TypedRouteV2) -> tuple[PoseStateV2, ...]:
    points: list[PoseStateV2] = []
    if route.platform_kind is PlatformKindV2.WHEEL:
        for primitive in route.primitives:
            samples = getattr(primitive, "samples", None)
            if type(samples) is not tuple:
                raise TypeError("wheel primitive must expose an exact samples tuple")
            _append_without_pose_seam(points, samples)
    elif route.platform_kind is PlatformKindV2.LEGGED:
        for primitive in route.primitives:
            lift = getattr(primitive, "lift_body_state", None)
            if type(lift) is not PoseStateV2:
                raise TypeError("legged primitive must expose an exact lift_body_state")
            _append_without_pose_seam(
                points,
                (primitive.start_state, lift, primitive.end_state),
            )
    else:
        raise ValueError("arc-length route samples apply only to wheel and legged routes")
    if not points:
        raise ValueError("route observation waypoints must be nonempty")
    return tuple(points)


def _arc_length_samples(
    waypoints: tuple[PoseStateV2, ...],
    endpoint_theta_rad: float,
) -> tuple[PoseStateV2, ...]:
    segments: list[tuple[PoseStateV2, PoseStateV2, float, float, float]] = []
    turn_events: list[tuple[float, int, PoseStateV2]] = []
    cumulative_distance = 0.0
    for sequence_index, (left, right) in enumerate(
        zip(waypoints[:-1], waypoints[1:], strict=True)
    ):
        length = hypot(right.x_m - left.x_m, right.y_m - left.y_m)
        if length == 0.0:
            if left.heading_rad != right.heading_rad:
                turn_events.append((cumulative_distance, sequence_index, left))
            continue
        start_distance = cumulative_distance
        cumulative_distance += length
        segments.append(
            (left, right, length, start_distance, cumulative_distance)
        )

    regular_events: list[tuple[float, int, PoseStateV2]] = []
    distance = 0.0
    total_length = cumulative_distance
    while distance < total_length - 1e-12:
        selected: tuple[PoseStateV2, PoseStateV2, float, float, float] | None = None
        for index, segment in enumerate(segments):
            if distance < segment[4] - 1e-12 or index == len(segments) - 1:
                selected = segment
                break
        if selected is None:
            raise ValueError("route arc-length sampling could not select a segment")
        left, right, length, segment_start_distance, _end_distance = selected
        fraction = (distance - segment_start_distance) / length
        raw_heading_delta = right.heading_rad - left.heading_rad
        heading_delta = remainder(raw_heading_delta, tau)
        if heading_delta == -tau / 2.0 and raw_heading_delta > 0.0:
            heading_delta = tau / 2.0
        heading = (
            left.heading_rad
            if fraction == 0.0
            else right.heading_rad
            if fraction == 1.0
            else left.heading_rad + fraction * heading_delta
        )
        regular_events.append(
            (
                distance,
                len(regular_events),
                PoseStateV2(
                    left.x_m + fraction * (right.x_m - left.x_m),
                    left.y_m + fraction * (right.y_m - left.y_m),
                    heading,
                ),
            )
        )
        distance += OBSERVATION_ROUTE_SAMPLE_STEP_M_V2

    ordered_events = [
        (event_distance, 0, sequence_index, state)
        for event_distance, sequence_index, state in turn_events
    ]
    ordered_events.extend(
        (event_distance, 1, sequence_index, state)
        for event_distance, sequence_index, state in regular_events
    )
    ordered_events.sort(key=lambda item: (item[0], item[1], item[2]))
    samples = [event[3] for event in ordered_events]
    endpoint = waypoints[-1]
    samples.append(PoseStateV2(endpoint.x_m, endpoint.y_m, endpoint_theta_rad))
    return tuple(samples)


def _hopper_samples(
    route: TypedRouteV2,
    endpoint_theta_rad: float,
) -> tuple[PoseStateV2, ...]:
    samples = [route.primitives[0].start_state]
    samples.extend(primitive.end_state for primitive in route.primitives)
    endpoint = samples[-1]
    samples[-1] = PoseStateV2(endpoint.x_m, endpoint.y_m, endpoint_theta_rad)
    return tuple(samples)


def _ray_cells(
    pose: PoseStateV2,
    angle_rad: float,
    geometry: FineGridGeometryV2,
) -> Iterator[Cell]:
    current = geometry.world_to_cell(WorldPoint(pose.x_m, pose.y_m))
    direction_x = cos(angle_rad)
    direction_y = sin(angle_rad)
    step_x = 1 if direction_x > 0.0 else -1 if direction_x < 0.0 else 0
    step_y = 1 if direction_y > 0.0 else -1 if direction_y < 0.0 else 0
    resolution = geometry.resolution_m
    if step_x:
        boundary_x = geometry.origin[0] + (current.x + (1 if step_x > 0 else 0)) * resolution
        t_max_x = (boundary_x - pose.x_m) / direction_x
        t_delta_x = resolution / abs(direction_x)
    else:
        t_max_x = inf
        t_delta_x = inf
    if step_y:
        boundary_y = geometry.origin[1] + (current.y + (1 if step_y > 0 else 0)) * resolution
        t_max_y = (boundary_y - pose.y_m) / direction_y
        t_delta_y = resolution / abs(direction_y)
    else:
        t_max_y = inf
        t_delta_y = inf

    while geometry.in_bounds(current):
        yield current
        next_distance = min(t_max_x, t_max_y)
        if next_distance > OBSERVATION_RANGE_M_V2:
            break
        if isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-12):
            current = Cell(current.x + step_x, current.y + step_y)
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            current = Cell(current.x + step_x, current.y)
            t_max_x += t_delta_x
        else:
            current = Cell(current.x, current.y + step_y)
            t_max_y += t_delta_y


def _visible_unknown_cells(
    samples: tuple[PoseStateV2, ...],
    snapshot: TerrainSnapshotV2,
) -> tuple[Cell, ...]:
    geometry = snapshot.geometry
    visible_unknown: set[Cell] = set()
    ray_count = int(round(OBSERVATION_FOV_DEG_V2 / OBSERVATION_RAY_STEP_DEG_V2)) + 1
    half_fov = OBSERVATION_FOV_DEG_V2 / 2.0
    for pose in samples:
        for ray_index in range(ray_count):
            offset_deg = -half_fov + ray_index * OBSERVATION_RAY_STEP_DEG_V2
            angle = pose.heading_rad + radians(offset_deg)
            for ray_cell_index, cell in enumerate(_ray_cells(pose, angle, geometry)):
                center = geometry.cell_center(cell)
                if hypot(center.x - pose.x_m, center.y - pose.y_m) > OBSERVATION_RANGE_M_V2:
                    continue
                row, column = cell.y, cell.x
                if not bool(snapshot.observed_mask[row, column]):
                    visible_unknown.add(cell)
                    break
                if ray_cell_index > 0 and (
                    bool(snapshot.hard_obstacle_mask[row, column])
                    or not bool(snapshot.traversable_mask[row, column])
                ):
                    break
    return tuple(sorted(visible_unknown, key=lambda cell: (cell.y, cell.x)))


def project_route_observation_v2(
    route: TypedRouteV2,
    observed_terrain: ObservedTerrainInputV2,
    *,
    endpoint_theta_rad: float,
) -> ObservationProjectionV2:
    if type(route) is not TypedRouteV2:
        raise TypeError("route must be exact TypedRouteV2")
    if type(observed_terrain) is not ObservedTerrainInputV2:
        raise TypeError("observed_terrain must be exact ObservedTerrainInputV2")
    endpoint_theta = PoseStateV2(0.0, 0.0, endpoint_theta_rad).heading_rad
    if route.platform_kind is PlatformKindV2.HOPPER:
        samples = _hopper_samples(route, endpoint_theta)
    else:
        samples = _arc_length_samples(_route_waypoints(route), endpoint_theta)
    visible_unknown = _visible_unknown_cells(samples, observed_terrain.terrain_snapshot)
    return ObservationProjectionV2(
        source=OBSERVATION_PROJECTION_SOURCE_V2,
        sample_states=samples,
        expected_new_observed_cells=float(len(visible_unknown)),
        expected_information_gain=0.0,
    )


__all__ = (
    "OBSERVATION_FOV_DEG_V2",
    "OBSERVATION_PROJECTION_SOURCE_V2",
    "OBSERVATION_RANGE_M_V2",
    "OBSERVATION_RAY_STEP_DEG_V2",
    "OBSERVATION_ROUTE_SAMPLE_STEP_M_V2",
    "OBSERVED_TERRAIN_CANONICALIZATION_V2",
    "OBSERVED_TERRAIN_SOURCE_ID_V2",
    "OBSERVED_TERRAIN_SOURCE_KIND_V2",
    "ObservedTerrainInputV2",
    "project_route_observation_v2",
)
