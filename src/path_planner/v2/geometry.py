from __future__ import annotations

from collections.abc import Callable
from math import ceil, cos, floor, hypot, isfinite, nextafter, sin, sqrt
from numbers import Real

from path_planner.core import Cell
from path_planner.search import MotionPrimitive, Pose2D, replay_motion_primitive
from path_planner.search.hybrid_astar import MAX_REPLAY_STEPS
from path_planner.v2.terrain import FineGridGeometryV2


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_real(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_real(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _motion_bound(
    primitive: MotionPrimitive,
    *,
    body_length_m: float,
    body_width_m: float,
    safety_margin_m: float,
) -> tuple[float, float, float, float]:
    if not isinstance(primitive, MotionPrimitive):
        raise TypeError("primitive must be MotionPrimitive")
    length = _positive_real(body_length_m, "body_length_m")
    width = _positive_real(body_width_m, "body_width_m")
    margin = _nonnegative_real(safety_margin_m, "safety_margin_m")
    speed = abs(_finite_real(primitive.v_mps, "primitive v_mps"))
    angular_speed = abs(
        _finite_real(primitive.omega_radps, "primitive omega_radps")
    )
    duration = _positive_real(primitive.duration_s, "primitive duration_s")
    half_length = length / 2.0 + margin
    half_width = width / 2.0 + margin
    radius = hypot(half_length, half_width)
    point_speed_bound = speed + radius * angular_speed
    if not isfinite(point_speed_bound):
        raise ValueError("wheel body point motion bound must be finite")
    return half_length, half_width, point_speed_bound, duration


def dense_wheel_replay_step_count(
    primitive: MotionPrimitive,
    *,
    body_length_m: float,
    body_width_m: float,
    safety_margin_m: float,
    resolution_m: float,
) -> int:
    """Return the deterministic dense replay count for conservative wheel sweep."""

    _, _, point_speed_bound, duration = _motion_bound(
        primitive,
        body_length_m=body_length_m,
        body_width_m=body_width_m,
        safety_margin_m=safety_margin_m,
    )
    resolution = _positive_real(resolution_m, "resolution_m")
    step_ratio = point_speed_bound * duration * 8.0 / resolution
    if not isfinite(step_ratio) or step_ratio > MAX_REPLAY_STEPS:
        raise ValueError(
            f"dense wheel replay steps must not exceed public replay cap {MAX_REPLAY_STEPS}"
        )
    return max(1, int(ceil(step_ratio)))


def conservative_wheel_pose_cells(
    pose: Pose2D,
    geometry: FineGridGeometryV2,
    *,
    body_length_m: float,
    body_width_m: float,
    safety_margin_m: float,
    expansion_m: float = 0.0,
) -> tuple[Cell, ...]:
    """Return sorted fine cells conservatively touched by one expanded wheel pose."""

    if not isinstance(pose, Pose2D):
        raise TypeError("pose must be Pose2D")
    if not isinstance(geometry, FineGridGeometryV2):
        raise TypeError("geometry must be FineGridGeometryV2")
    pose_x = _finite_real(pose.x_m, "pose x_m")
    pose_y = _finite_real(pose.y_m, "pose y_m")
    heading = _finite_real(pose.theta_rad, "pose theta_rad")
    length = _positive_real(body_length_m, "body_length_m")
    width = _positive_real(body_width_m, "body_width_m")
    margin = _nonnegative_real(safety_margin_m, "safety_margin_m")
    expansion = _nonnegative_real(expansion_m, "expansion_m")

    cell_square_radius = geometry.resolution_m / sqrt(2.0)
    half_length = nextafter(
        length / 2.0 + margin + expansion + cell_square_radius,
        float("inf"),
    )
    half_width = nextafter(
        width / 2.0 + margin + expansion + cell_square_radius,
        float("inf"),
    )
    cos_t = cos(heading)
    sin_t = sin(heading)
    extent_x = nextafter(
        abs(cos_t) * half_length + abs(sin_t) * half_width,
        float("inf"),
    )
    extent_y = nextafter(
        abs(sin_t) * half_length + abs(cos_t) * half_width,
        float("inf"),
    )
    resolution = geometry.resolution_m
    origin_x, origin_y = geometry.origin

    min_x = floor((pose_x - extent_x - origin_x) / resolution - 0.5)
    max_x = ceil((pose_x + extent_x - origin_x) / resolution - 0.5)
    min_y = floor((pose_y - extent_y - origin_y) / resolution - 0.5)
    max_y = ceil((pose_y + extent_y - origin_y) / resolution - 0.5)

    cells: set[Cell] = set()
    for y in range(min_y, max_y + 1):
        center_y = origin_y + (y + 0.5) * resolution
        for x in range(min_x, max_x + 1):
            center_x = origin_x + (x + 0.5) * resolution
            dx = center_x - pose_x
            dy = center_y - pose_y
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            if abs(local_x) <= half_length and abs(local_y) <= half_width:
                cells.add(Cell(x, y))
    return tuple(sorted(cells, key=lambda cell: (cell.y, cell.x)))


def conservative_wheel_sweep_cells(
    start: Pose2D,
    primitive: MotionPrimitive,
    geometry: FineGridGeometryV2,
    *,
    body_length_m: float,
    body_width_m: float,
    safety_margin_m: float,
    deadline_checker: Callable[[], bool] | None = None,
) -> tuple[Cell, ...]:
    """Replay densely and return every conservatively contacted fine cell."""

    if not isinstance(start, Pose2D):
        raise TypeError("start must be Pose2D")
    if not isinstance(geometry, FineGridGeometryV2):
        raise TypeError("geometry must be FineGridGeometryV2")
    if deadline_checker is not None and not callable(deadline_checker):
        raise TypeError("deadline_checker must be callable")

    steps = dense_wheel_replay_step_count(
        primitive,
        body_length_m=body_length_m,
        body_width_m=body_width_m,
        safety_margin_m=safety_margin_m,
        resolution_m=geometry.resolution_m,
    )
    _, _, point_speed_bound, duration = _motion_bound(
        primitive,
        body_length_m=body_length_m,
        body_width_m=body_width_m,
        safety_margin_m=safety_margin_m,
    )
    interval_dt = duration / float(steps)
    transition = replay_motion_primitive(
        start,
        primitive,
        interval_dt,
        deadline_checker=deadline_checker,
    )
    interval_expansion = point_speed_bound * interval_dt
    cells: set[Cell] = set()
    for pose in transition.samples[:-1]:
        if deadline_checker is not None:
            expired = deadline_checker()
            if type(expired) is not bool:
                raise TypeError("deadline_checker must return exact bool")
            if expired:
                raise TimeoutError("wheel sweep deadline expired")
        cells.update(
            conservative_wheel_pose_cells(
                pose,
                geometry,
                body_length_m=body_length_m,
                body_width_m=body_width_m,
                safety_margin_m=safety_margin_m,
                expansion_m=interval_expansion,
            )
        )
    cells.update(
        conservative_wheel_pose_cells(
            transition.end,
            geometry,
            body_length_m=body_length_m,
            body_width_m=body_width_m,
            safety_margin_m=safety_margin_m,
            expansion_m=0.0,
        )
    )
    return tuple(sorted(cells, key=lambda cell: (cell.y, cell.x)))
